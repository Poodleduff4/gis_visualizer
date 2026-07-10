use anyhow::bail;
use bitvec::{bitarr, bitvec, vec::BitVec, BitArr};
use flatgeobuf::{
    ColumnType, FallibleStreamingIterator, FeatureProperties, FgbReader, GeometryType, Header,
};
use geo_types::Geometry;
use geozero::{error::GeozeroError, ColumnValue, PropertyProcessor, ToGeo};
use std::{
    collections::HashMap,
    io::{BufReader, Read, Seek},
    sync::mpsc,
};

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;

// Arrow array/schema types — same crate on both targets (arrow = "53").
// On desktop datafusion re-exports these from the same crate version.
use arrow::array::{
    Array, BinaryArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use std::sync::{atomic::AtomicBool, Arc};

// Desktop: DataFusion SQL engine for GeoParquet queries.
#[cfg(not(target_arch = "wasm32"))]
use datafusion::prelude::*;

// Wasm: read parquet directly from bytes using the parquet crate.
#[cfg(target_arch = "wasm32")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

#[cfg(target_arch = "wasm32")]
use std::io::Cursor;

#[cfg(target_arch = "wasm32")]
pub type FgbReaderCache = std::rc::Rc<
    std::cell::RefCell<std::collections::HashMap<String, Vec<flatgeobuf::HttpFgbReader>>>,
>;

use crate::{
    filter::FilterLogic,
    gis_layer::{AttributeValue, BatchMessage, GisFeature, GisLayer, LayerEntry, LayerKind},
    point_cloud_layer::{AttributeColumn, PointCloudLayer},
};

#[derive(Clone)]
pub enum GisFilePath {
    LocalFile(String),
    HttpLocation(String),
    /// In-memory bytes from a local file pick on wasm (parquet only).
    Bytes(Arc<[u8]>, String),
}
impl GisFilePath {
    pub fn to_string(&self) -> String {
        match self {
            GisFilePath::LocalFile(p) => p.clone(),
            GisFilePath::HttpLocation(p) => p.clone(),
            GisFilePath::Bytes(_, name) => name.clone(),
        }
    }
}

#[derive(Clone)]
pub struct LayerDescriptor {
    pub name: String,
    pub num_features: u64,
    pub field_names: Vec<String>,
    pub geometry_type: GeometryType,
    pub location: GisFilePath,
}

struct PairCollector<'a> {
    selected: &'a std::collections::HashSet<String>,
    pairs: Vec<(String, AttributeValue)>,
}

impl PropertyProcessor for PairCollector<'_> {
    fn property(
        &mut self,
        _idx: usize,
        name: &str,
        value: &ColumnValue,
    ) -> std::result::Result<bool, GeozeroError> {
        if self.selected.contains(name) {
            let attr = match value {
                ColumnValue::Int(v) => Some(AttributeValue::Integer(*v as i64)),
                ColumnValue::Long(v) => Some(AttributeValue::Integer(*v)),
                ColumnValue::Float(v) => Some(AttributeValue::Float(*v as f64)),
                ColumnValue::Double(v) => Some(AttributeValue::Float(*v)),
                ColumnValue::String(v) => Some(AttributeValue::Text(v.to_string())),
                _ => None,
            };
            if let Some(a) = attr {
                self.pairs.push((name.to_string(), a));
            }
        }
        Ok(false)
    }
}

struct PropertyExtractor<'a> {
    cols: &'a mut Vec<(String, AttributeColumn)>,
    idx: Option<u32>,
}

impl PropertyProcessor for PropertyExtractor<'_> {
    fn property(
        &mut self,
        _i: usize,
        name: &str,
        value: &ColumnValue,
    ) -> std::result::Result<bool, GeozeroError> {
        if name == "idx" {
            self.idx = match value {
                ColumnValue::UInt(v) => Some(*v),
                ColumnValue::Int(v) => Some(*v as u32),
                ColumnValue::Long(v) => Some(*v as u32),
                ColumnValue::ULong(v) => Some(*v as u32),
                _ => None,
            };
        }
        if let Some((_, col)) = self.cols.iter_mut().find(|(n, _)| n == name) {
            match (col, value) {
                (AttributeColumn::Integer(v), ColumnValue::Int(i)) => v.push(*i as i64),
                (AttributeColumn::Integer(v), ColumnValue::Long(i)) => v.push(*i),
                (AttributeColumn::Float(v), ColumnValue::Float(f)) => v.push(*f as f64),
                (AttributeColumn::Float(v), ColumnValue::Double(f)) => v.push(*f),
                (AttributeColumn::Text(v), ColumnValue::String(s)) => v.push(s.to_string()),
                (col, _) => col.push_default(),
            }
        }
        Ok(false)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VectorFileType {
    FlatGeobuf,
    GeoParquet,
}

fn vector_file_type(path: &str) -> anyhow::Result<VectorFileType> {
    if path.ends_with(".parquet") {
        Ok(VectorFileType::GeoParquet)
    } else if path.ends_with(".fgb") {
        Ok(VectorFileType::FlatGeobuf)
    } else {
        bail!("Unsupported vector file extension: {path}");
    }
}

// ── GeoParquetReader shared types and helpers ─────────────────────────────

const XY_CANDIDATES: &[(&str, &str)] = &[
    ("x", "y"),
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("lng", "lat"),
    ("long", "lat"),
];

enum GeometrySource {
    XYColumns { x_col: String, y_col: String },
    WkbColumn,
}

fn decode_wkb_point(bytes: &[u8]) -> Option<[f64; 2]> {
    if bytes.len() < 21 {
        return None;
    }
    let le = bytes[0] == 1;
    let geom_type = if le {
        u32::from_le_bytes(bytes[1..5].try_into().ok()?)
    } else {
        u32::from_be_bytes(bytes[1..5].try_into().ok()?)
    };
    if geom_type != 1 {
        return None;
    }
    let x = if le {
        f64::from_le_bytes(bytes[5..13].try_into().ok()?)
    } else {
        f64::from_be_bytes(bytes[5..13].try_into().ok()?)
    };
    let y = if le {
        f64::from_le_bytes(bytes[13..21].try_into().ok()?)
    } else {
        f64::from_be_bytes(bytes[13..21].try_into().ok()?)
    };
    Some([x, y])
}

fn pq_extract_f64(arr: &dyn Array, i: usize) -> Option<f64> {
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        Some(a.value(i))
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        Some(a.value(i) as f64)
    } else {
        None
    }
}

fn pq_extract_i64(arr: &dyn Array, i: usize) -> Option<i64> {
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        Some(a.value(i))
    } else if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
        Some(a.value(i) as i64)
    } else if let Some(a) = arr.as_any().downcast_ref::<UInt32Array>() {
        Some(a.value(i) as i64)
    } else {
        None
    }
}

fn pq_attr_col(dt: &DataType, cap: usize) -> crate::point_cloud_layer::AttributeColumn {
    use crate::point_cloud_layer::AttributeColumn;
    match dt {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => AttributeColumn::Integer(Vec::with_capacity(cap)),
        DataType::Float32 | DataType::Float64 => AttributeColumn::Float(Vec::with_capacity(cap)),
        _ => AttributeColumn::Text(Vec::with_capacity(cap)),
    }
}

pub struct GeoParquetReader;

#[cfg(not(target_arch = "wasm32"))]
impl GeoParquetReader {
    async fn make_ctx(path: &str) -> datafusion::error::Result<SessionContext> {
        let config = SessionConfig::new().with_target_partitions(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        );
        let ctx = SessionContext::new_with_config(config);
        ctx.register_parquet("layer", path, ParquetReadOptions::default())
            .await?;
        Ok(ctx)
    }

    fn detect_geometry_source(schema: &datafusion::arrow::datatypes::Schema) -> GeometrySource {
        if schema.field_with_name("geometry").is_ok() {
            return GeometrySource::WkbColumn;
        }
        let names: Vec<String> = schema.fields().iter().map(|f| f.name().to_lowercase()).collect();
        let orig: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        for (xk, yk) in XY_CANDIDATES {
            if let (Some(xi), Some(yi)) = (
                names.iter().position(|n| n == xk),
                names.iter().position(|n| n == yk),
            ) {
                return GeometrySource::XYColumns {
                    x_col: orig[xi].to_string(),
                    y_col: orig[yi].to_string(),
                };
            }
        }
        // Substring scan for lon/lat-style column names
        let x_col = orig.iter().find(|n| {
            let l = n.to_lowercase();
            l.contains("longitude") || l.contains("_lon") || l.contains("lon_") || l == "lon"
                || l.contains("_lng") || l.contains("lng_")
        });
        let y_col = orig.iter().find(|n| {
            let l = n.to_lowercase();
            l.contains("latitude") || l.contains("_lat") || l.contains("lat_") || l == "lat"
        });
        if let (Some(x), Some(y)) = (x_col, y_col) {
            return GeometrySource::XYColumns {
                x_col: x.to_string(),
                y_col: y.to_string(),
            };
        }
        GeometrySource::XYColumns { x_col: "x".to_string(), y_col: "y".to_string() }
    }

    async fn load_descriptor_async(path: &str) -> anyhow::Result<LayerDescriptor> {
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = Self::detect_geometry_source(&schema);

        let count_batches = ctx.sql("SELECT COUNT(*) FROM layer").await?.collect().await?;
        let num_features = count_batches
            .first()
            .and_then(|b| {
                let col = b.column(0);
                if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                    Some(a.value(0) as u64)
                } else if let Some(a) = col.as_any().downcast_ref::<UInt64Array>() {
                    Some(a.value(0))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let geom_cols: std::collections::HashSet<String> = match &geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                [x_col.clone(), y_col.clone(), "idx".to_string()].into()
            }
            GeometrySource::WkbColumn => ["geometry".to_string(), "idx".to_string()].into(),
        };
        let field_names = schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| !geom_cols.contains(n.as_str()))
            .collect();

        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("layer")
            .to_string();

        Ok(LayerDescriptor {
            name,
            num_features,
            field_names,
            geometry_type: GeometryType(1), // Point
            location: GisFilePath::LocalFile(path.to_string()),
        })
    }

    pub fn load_descriptor(path: &str) -> anyhow::Result<LayerDescriptor> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(Self::load_descriptor_async(path))
    }

    fn build_select(
        geom_src: &GeometrySource,
        selected_fields: Option<&[String]>,
        schema: &datafusion::arrow::datatypes::Schema,
    ) -> String {
        // DataFusion normalises unquoted identifiers to lowercase, so any mixed-case
        // column name (e.g. "RateCodeID") must be double-quoted to preserve its case.
        let q = |s: &str| format!("\"{}\"", s);
        let mut cols: Vec<String> = Vec::new();
        if schema.field_with_name("idx").is_ok() {
            cols.push(q("idx"));
        }
        match geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                cols.push(q(x_col));
                cols.push(q(y_col));
            }
            GeometrySource::WkbColumn => cols.push(q("geometry")),
        }
        if let Some(fields) = selected_fields {
            for f in fields {
                if schema.field_with_name(f).is_ok() {
                    cols.push(q(f));
                }
            }
        }
        cols.join(", ")
    }

    fn extract_points(
        batch: &RecordBatch,
        dest_idx: usize,
        geom_src: &GeometrySource,
        bbox_filter: Option<[f64; 4]>,
        selected_fields: Option<&[String]>,
        id_base: u32,
    ) -> anyhow::Result<Option<crate::gis_layer::BatchMessage>> {
        use crate::gis_layer::BatchMessage;
        use crate::point_cloud_layer::AttributeColumn;

        let nrows = batch.num_rows();
        if nrows == 0 {
            return Ok(None);
        }

        // `id_base` keeps ids unique/stable across a multi-batch file when no
        // `idx` column is present — without it every batch would restart at
        // 0, so ids collide and filter/selection lookups by id silently
        // match the wrong rows past the first batch.
        let ids: Vec<u32> = match batch.column_by_name("idx") {
            Some(c) => {
                if let Some(a) = c.as_any().downcast_ref::<UInt32Array>() {
                    a.values().to_vec()
                } else if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
                    a.values().iter().map(|v| *v as u32).collect()
                } else if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                    a.values().iter().map(|v| *v as u32).collect()
                } else {
                    (id_base..id_base + nrows as u32).collect()
                }
            }
            None => (id_base..id_base + nrows as u32).collect(),
        };

        let coords: Vec<Option<[f64; 2]>> = match geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                let xs = batch.column_by_name(x_col);
                let ys = batch.column_by_name(y_col);
                (0..nrows)
                    .map(|i| {
                        let x = xs.and_then(|a| pq_extract_f64(a.as_ref(), i))?;
                        let y = ys.and_then(|a| pq_extract_f64(a.as_ref(), i))?;
                        Some([x, y])
                    })
                    .collect()
            }
            GeometrySource::WkbColumn => {
                let casted = batch
                    .column_by_name("geometry")
                    .and_then(|arr| datafusion::arrow::compute::cast(arr, &ArrowDataType::Binary).ok());
                (0..nrows)
                    .map(|i| {
                        let a = casted.as_ref()?.as_any().downcast_ref::<BinaryArray>()?;
                        decode_wkb_point(a.value(i))
                    })
                    .collect()
            }
        };

        let mut columns: Vec<(String, AttributeColumn)> = match selected_fields {
            Some(fields) => fields
                .iter()
                .filter_map(|f| {
                    let idx = batch.schema().index_of(f).ok()?;
                    let dt = batch.schema().field(idx).data_type().clone();
                    Some((f.clone(), pq_attr_col(&dt, nrows)))
                })
                .collect(),
            None => vec![],
        };

        let mut points: Vec<(u32, [f64; 2])> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let Some(xy) = coords[i] else { continue };
            if let Some(bb) = bbox_filter {
                if xy[0] < bb[0] || xy[0] > bb[2] || xy[1] < bb[1] || xy[1] > bb[3] {
                    continue;
                }
            }
            points.push((ids[i], xy));
            for (name, col) in &mut columns {
                let arr = batch.column_by_name(name);
                match col {
                    AttributeColumn::Integer(v) => {
                        v.push(arr.and_then(|a| pq_extract_i64(a.as_ref(), i)).unwrap_or(0));
                    }
                    AttributeColumn::Float(v) => {
                        v.push(arr.and_then(|a| pq_extract_f64(a.as_ref(), i)).unwrap_or(0.0));
                    }
                    AttributeColumn::Text(v) => {
                        let val = arr
                            .and_then(|a| {
                                a.as_any()
                                    .downcast_ref::<StringArray>()
                                    .map(|sa| sa.value(i).to_string())
                            })
                            .unwrap_or_default();
                        v.push(val);
                    }
                }
            }
        }

        if points.is_empty() {
            return Ok(None);
        }
        Ok(Some(BatchMessage::Points(dest_idx, points, columns)))
    }

    async fn load_point_layer_batched_async(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<crate::gis_layer::BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = Self::detect_geometry_source(&schema);
        let sel = Self::build_select(&geom_src, selected_fields.as_deref(), &schema);

        let batches = ctx
            .sql(&format!("SELECT {} FROM layer", sel))
            .await?
            .collect()
            .await?;

        let mut id_base: u32 = 0;
        for batch in &batches {
            let nrows = batch.num_rows();
            if let Some(msg) = Self::extract_points(batch, dest_idx, &geom_src, None, selected_fields.as_deref(), id_base)? {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
        }
        Ok(())
    }

    pub fn load_point_layer_batched(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<crate::gis_layer::BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(Self::load_point_layer_batched_async(path, dest_idx, tx, selected_fields))
    }

    /// Bbox-filtered stream. XY-column files push bbox into SQL; WKB files filter in Rust.
    pub async fn stream_bbox(
        path: &str,
        bbox: [f64; 4],
        dest_idx: usize,
        tx: mpsc::SyncSender<crate::gis_layer::BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = Self::detect_geometry_source(&schema);
        let sel = Self::build_select(&geom_src, selected_fields.as_deref(), &schema);

        let sql = match &geom_src {
            GeometrySource::XYColumns { x_col, y_col } => format!(
                "SELECT {sel} FROM layer WHERE \"{x_col}\" BETWEEN {xmin} AND {xmax} AND \"{y_col}\" BETWEEN {ymin} AND {ymax}",
                xmin = bbox[0], ymin = bbox[1], xmax = bbox[2], ymax = bbox[3],
            ),
            GeometrySource::WkbColumn => format!("SELECT {sel} FROM layer"),
        };

        let batches = ctx.sql(&sql).await?.collect().await?;
        let mut id_base: u32 = 0;
        for batch in &batches {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let nrows = batch.num_rows();
            let wkb_bbox = matches!(geom_src, GeometrySource::WkbColumn).then_some(bbox);
            if let Some(msg) = Self::extract_points(batch, dest_idx, &geom_src, wkb_bbox, selected_fields.as_deref(), id_base)? {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
        }
        Ok(())
    }
}

// ── GeoParquetReader — wasm impl (reads from in-memory bytes) ────────────

#[cfg(target_arch = "wasm32")]
impl GeoParquetReader {
    fn detect_geometry_source(schema: &arrow::datatypes::Schema) -> GeometrySource {
        if schema.field_with_name("geometry").is_ok() {
            return GeometrySource::WkbColumn;
        }
        let names: Vec<String> =
            schema.fields().iter().map(|f| f.name().to_lowercase()).collect();
        let orig: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        for (xk, yk) in XY_CANDIDATES {
            if let (Some(xi), Some(yi)) = (
                names.iter().position(|n| n == xk),
                names.iter().position(|n| n == yk),
            ) {
                return GeometrySource::XYColumns {
                    x_col: orig[xi].to_string(),
                    y_col: orig[yi].to_string(),
                };
            }
        }
        let x_col = orig.iter().find(|n| {
            let l = n.to_lowercase();
            l.contains("longitude") || l.contains("_lon") || l.contains("lon_") || l == "lon"
                || l.contains("_lng") || l.contains("lng_")
        });
        let y_col = orig.iter().find(|n| {
            let l = n.to_lowercase();
            l.contains("latitude") || l.contains("_lat") || l.contains("lat_") || l == "lat"
        });
        if let (Some(x), Some(y)) = (x_col, y_col) {
            return GeometrySource::XYColumns {
                x_col: x.to_string(),
                y_col: y.to_string(),
            };
        }
        GeometrySource::XYColumns { x_col: "x".to_string(), y_col: "y".to_string() }
    }

    fn extract_points_from_batch(
        batch: &RecordBatch,
        dest_idx: usize,
        geom_src: &GeometrySource,
        bbox_filter: Option<[f64; 4]>,
        selected_fields: Option<&[String]>,
        id_base: u32,
    ) -> anyhow::Result<Option<crate::gis_layer::BatchMessage>> {
        use crate::gis_layer::BatchMessage;
        use crate::point_cloud_layer::AttributeColumn;

        let nrows = batch.num_rows();
        if nrows == 0 {
            return Ok(None);
        }
        // See extract_points (native side) for why id_base matters across batches.
        let ids: Vec<u32> = match batch.column_by_name("idx") {
            Some(c) => {
                if let Some(a) = c.as_any().downcast_ref::<UInt32Array>() {
                    a.values().to_vec()
                } else if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
                    a.values().iter().map(|v| *v as u32).collect()
                } else if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                    a.values().iter().map(|v| *v as u32).collect()
                } else {
                    (id_base..id_base + nrows as u32).collect()
                }
            }
            None => (id_base..id_base + nrows as u32).collect(),
        };
        let coords: Vec<Option<[f64; 2]>> = match geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                let xs = batch.column_by_name(x_col);
                let ys = batch.column_by_name(y_col);
                (0..nrows)
                    .map(|i| {
                        let x = xs.and_then(|a| pq_extract_f64(a.as_ref(), i))?;
                        let y = ys.and_then(|a| pq_extract_f64(a.as_ref(), i))?;
                        Some([x, y])
                    })
                    .collect()
            }
            GeometrySource::WkbColumn => {
                let geom_col = batch.column_by_name("geometry");
                (0..nrows)
                    .map(|i| {
                        let arr = geom_col?;
                        if let Some(a) = arr.as_any().downcast_ref::<BinaryArray>() {
                            decode_wkb_point(a.value(i))
                        } else {
                            use arrow::array::LargeBinaryArray;
                            arr.as_any().downcast_ref::<LargeBinaryArray>().and_then(|a| decode_wkb_point(a.value(i)))
                        }
                    })
                    .collect()
            }
        };
        let mut columns: Vec<(String, AttributeColumn)> = match selected_fields {
            Some(fields) => fields
                .iter()
                .filter_map(|f| {
                    let idx = batch.schema().index_of(f).ok()?;
                    let dt = batch.schema().field(idx).data_type().clone();
                    Some((f.clone(), pq_attr_col(&dt, nrows)))
                })
                .collect(),
            None => vec![],
        };
        let mut points: Vec<(u32, [f64; 2])> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let Some(xy) = coords[i] else { continue };
            if let Some(bb) = bbox_filter {
                if xy[0] < bb[0] || xy[0] > bb[2] || xy[1] < bb[1] || xy[1] > bb[3] {
                    continue;
                }
            }
            points.push((ids[i], xy));
            for (name, col) in &mut columns {
                let arr = batch.column_by_name(name);
                match col {
                    AttributeColumn::Integer(v) => {
                        v.push(arr.and_then(|a| pq_extract_i64(a.as_ref(), i)).unwrap_or(0));
                    }
                    AttributeColumn::Float(v) => {
                        v.push(arr.and_then(|a| pq_extract_f64(a.as_ref(), i)).unwrap_or(0.0));
                    }
                    AttributeColumn::Text(v) => {
                        let val = arr
                            .and_then(|a| {
                                a.as_any()
                                    .downcast_ref::<StringArray>()
                                    .map(|sa| sa.value(i).to_string())
                            })
                            .unwrap_or_default();
                        v.push(val);
                    }
                }
            }
        }
        if points.is_empty() {
            return Ok(None);
        }
        Ok(Some(BatchMessage::Points(dest_idx, points, columns)))
    }

    pub fn load_descriptor_from_bytes(bytes: &[u8], name: &str) -> anyhow::Result<LayerDescriptor> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let bytes = ::bytes::Bytes::copy_from_slice(bytes);
        let raw: Arc<[u8]> = Arc::from(bytes.as_ref());
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let schema = builder.schema().as_ref().clone();
        let num_features = builder.metadata().file_metadata().num_rows() as u64;
        let geom_src = Self::detect_geometry_source(&schema);
        let geom_cols: std::collections::HashSet<String> = match &geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                [x_col.clone(), y_col.clone(), "idx".to_string()].into()
            }
            GeometrySource::WkbColumn => ["geometry".to_string(), "idx".to_string()].into(),
        };
        let field_names = schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| !geom_cols.contains(n.as_str()))
            .collect();
        let display_name = std::path::Path::new(name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name)
            .to_string();
        Ok(LayerDescriptor {
            name: display_name,
            num_features,
            field_names,
            geometry_type: flatgeobuf::GeometryType(1),
            location: GisFilePath::Bytes(raw, name.to_string()),
        })
    }

    pub fn load_point_layer_batched_from_bytes(
        bytes: Arc<[u8]>,
        dest_idx: usize,
        tx: mpsc::SyncSender<crate::gis_layer::BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let pq_bytes = ::bytes::Bytes::copy_from_slice(&bytes);
        let builder = ParquetRecordBatchReaderBuilder::try_new(pq_bytes)?;
        let schema = builder.schema().as_ref().clone();
        let geom_src = Self::detect_geometry_source(&schema);
        let reader = builder.build()?;
        let mut id_base: u32 = 0;
        for result in reader {
            let batch = result?;
            let nrows = batch.num_rows();
            if let Some(msg) = Self::extract_points_from_batch(
                &batch,
                dest_idx,
                &geom_src,
                None,
                selected_fields.as_deref(),
                id_base,
            )? {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
        }
        Ok(())
    }
}

pub struct GisReader {}

impl GisReader {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_layer_descriptor(path: &str) -> anyhow::Result<LayerDescriptor> {
        match vector_file_type(path)? {
            VectorFileType::GeoParquet => GeoParquetReader::load_descriptor(path),
            VectorFileType::FlatGeobuf => {
                let reader = FgbReader::open(BufReader::new(File::open(path)?))?;
                Self::make_layer_descriptor(reader.header(), GisFilePath::LocalFile(path.to_string()))
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_layer_descriptor(url: &str) -> anyhow::Result<LayerDescriptor> {
        use flatgeobuf::HttpFgbReader;
        use wasm_bindgen::JsValue;

        web_sys::console::log_1(&JsValue::from_str(&format!(
            "load_layer_descriptor: {}",
            url
        )));
        let reader = HttpFgbReader::open(url).await?;
        Self::make_layer_descriptor(reader.header(), GisFilePath::HttpLocation(url.to_string()))
    }

    fn make_layer_descriptor<'a>(
        header: Header<'a>,
        path: GisFilePath,
    ) -> anyhow::Result<LayerDescriptor> {
        Ok(LayerDescriptor {
            name: header.name().unwrap_or("N/A").to_string(),
            num_features: header.features_count(),
            field_names: header
                .columns()
                .map(|cols| cols.iter().map(|c| c.name().to_string()).collect())
                .unwrap_or_default(),
            geometry_type: header.geometry_type(),
            location: path,
        })
    }

    // ── load_layer_batched ────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_layer_batched(
        path: GisFilePath,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), anyhow::Error> {
        let GisFilePath::LocalFile(str_path) = path else {
            bail!("Wrong GisFilePath type!");
        };
        let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
        Self::load_layer_batched_impl(reader, dest_idx, tx, selected_fields)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_layer_batched_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), anyhow::Error> {
        let mut iter = reader.select_all()?;
        let selected_set: std::collections::HashSet<String> = selected_fields
            .map(|f| f.into_iter().collect())
            .unwrap_or_default();
        const BATCH_SIZE: usize = 10_000;
        let mut batch: Vec<GisFeature> = Vec::with_capacity(BATCH_SIZE);
        let mut count = 0usize;
        while let Some(feature) = iter.next()? {
            let geo = match feature.to_geo() {
                Ok(g) => g,
                Err(_) => continue,
            };
            let attributes: HashMap<String, AttributeValue> = if !selected_set.is_empty() {
                let mut collector = PairCollector {
                    selected: &selected_set,
                    pairs: Vec::new(),
                };
                feature.process_properties(&mut collector).ok();
                collector.pairs.into_iter().collect()
            } else {
                HashMap::new()
            };
            batch.push(GisFeature::new(count, geo, attributes));
            count += 1;
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Vector(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Vector(dest_idx, batch))?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_layer_batched(
        bytes: std::sync::Arc<[u8]>,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(Cursor::new(bytes)))?;
        let mut iter = reader.select_all()?;
        let selected_set: std::collections::HashSet<String> = selected_fields
            .map(|f| f.into_iter().collect())
            .unwrap_or_default();
        const BATCH_SIZE: usize = 30_000;
        let mut batch: Vec<GisFeature> = Vec::with_capacity(BATCH_SIZE);
        let mut count = 0usize;
        while let Some(feature) = iter.next()? {
            let geo = match feature.to_geo() {
                Ok(g) => g,
                Err(_) => continue,
            };
            let attributes: HashMap<String, AttributeValue> = if !selected_set.is_empty() {
                let mut collector = PairCollector {
                    selected: &selected_set,
                    pairs: Vec::new(),
                };
                feature.process_properties(&mut collector).ok();
                collector.pairs.into_iter().collect()
            } else {
                HashMap::new()
            };
            batch.push(GisFeature::new(count, geo, attributes));
            count += 1;
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Vector(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                ))
                .ok();
                // Yield to browser event loop so rAF can fire between batches.
                gloo_timers::future::sleep(std::time::Duration::ZERO).await;
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Vector(dest_idx, batch))?;
        }
        Ok(())
    }

    // ── load_point_layer_batched ──────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_point_layer_batched(
        path: GisFilePath,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), anyhow::Error> {
        let GisFilePath::LocalFile(str_path) = path else {
            bail!("Wrong GisFilePath type!");
        };
        match vector_file_type(&str_path)? {
            VectorFileType::GeoParquet => {
                GeoParquetReader::load_point_layer_batched(&str_path, dest_idx, tx, selected_fields)
            }
            VectorFileType::FlatGeobuf => {
                let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
                Self::load_point_layer_batched_impl(reader, dest_idx, tx, selected_fields)
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_point_layer_batched_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), anyhow::Error> {
        const BATCH_SIZE: usize = 10_000;
        let col_schema: Vec<(String, ColumnType)> = {
            let header = reader.header();
            header
                .columns()
                .map(|cols| {
                    cols.iter()
                        .filter(|c| {
                            selected_fields
                                .as_ref()
                                .map_or(false, |sel| sel.iter().any(|s| s.as_str() == c.name()))
                        })
                        .map(|c| (c.name().to_string(), c.type_()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let make_batch_cols = || -> Vec<(String, AttributeColumn)> {
            col_schema
                .iter()
                .map(|(name, col_type)| {
                    let col = match *col_type {
                        ColumnType::Byte
                        | ColumnType::UByte
                        | ColumnType::Short
                        | ColumnType::UShort
                        | ColumnType::Int
                        | ColumnType::UInt
                        | ColumnType::Long
                        | ColumnType::ULong => {
                            AttributeColumn::Integer(Vec::with_capacity(BATCH_SIZE))
                        }
                        ColumnType::Float | ColumnType::Double => {
                            AttributeColumn::Float(Vec::with_capacity(BATCH_SIZE))
                        }
                        _ => AttributeColumn::Text(Vec::with_capacity(BATCH_SIZE)),
                    };
                    (name.clone(), col)
                })
                .collect()
        };
        let mut iter = reader.select_all()?;
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        let mut id_counter = 0_u32;
        while let Some(feature) = iter.next()? {
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            let mut extractor = PropertyExtractor {
                cols: &mut batch_cols,
                idx: None,
            };
            feature.process_properties(&mut extractor).ok();
            let id = extractor.idx.unwrap_or(id_counter);
            id_counter += 1;
            batch.push((id, [x, y]));
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, make_batch_cols()),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        println!("Layer Empty, returning from feature streamer!");

        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_point_layer_batched(
        bytes: std::sync::Arc<[u8]>,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(Cursor::new(bytes)))?;
        const BATCH_SIZE: usize = 50_000;
        let col_schema: Vec<(String, ColumnType)> = {
            let header = reader.header();
            header
                .columns()
                .map(|cols| {
                    cols.iter()
                        .filter(|c| {
                            selected_fields
                                .as_ref()
                                .map_or(false, |sel| sel.iter().any(|s| s.as_str() == c.name()))
                        })
                        .map(|c| (c.name().to_string(), c.type_()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let make_batch_cols = || -> Vec<(String, AttributeColumn)> {
            col_schema
                .iter()
                .map(|(name, col_type)| {
                    let col = match *col_type {
                        ColumnType::Byte
                        | ColumnType::UByte
                        | ColumnType::Short
                        | ColumnType::UShort
                        | ColumnType::Int
                        | ColumnType::UInt
                        | ColumnType::Long
                        | ColumnType::ULong => {
                            AttributeColumn::Integer(Vec::with_capacity(BATCH_SIZE))
                        }
                        ColumnType::Float | ColumnType::Double => {
                            AttributeColumn::Float(Vec::with_capacity(BATCH_SIZE))
                        }
                        _ => AttributeColumn::Text(Vec::with_capacity(BATCH_SIZE)),
                    };
                    (name.clone(), col)
                })
                .collect()
        };
        let mut iter = reader.select_all()?;
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        let mut id_counter = 0_u32;
        while let Some(feature) = iter.next()? {
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            let mut extractor = PropertyExtractor {
                cols: &mut batch_cols,
                idx: None,
            };
            feature.process_properties(&mut extractor).ok();
            let id = extractor.idx.unwrap_or(id_counter);
            id_counter += 1;
            batch.push((id, [x, y]));
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, make_batch_cols()),
                ))
                .ok();
                // Yield to browser event loop so rAF can fire between batches.
                gloo_timers::future::sleep(std::time::Duration::ZERO).await;
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn stream_fgb_bbox(
        path: &GisFilePath,
        bbox: [f64; 4],
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel_stream: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), anyhow::Error> {
        use std::sync::atomic::Ordering;

        if cancel_stream.load(Ordering::Relaxed) {
            return Ok(());
        }
        let GisFilePath::LocalFile(str_path) = path else {
            bail!("Wrong GisFilePath type!");
        };
        let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
        const BATCH_SIZE: usize = 50_000;
        let col_schema: Vec<(String, ColumnType)> = {
            let header = reader.header();
            header
                .columns()
                .map(|cols| {
                    cols.iter()
                        .filter(|c| {
                            selected_fields
                                .as_ref()
                                .map_or(false, |sel| sel.iter().any(|s| s.as_str() == c.name()))
                        })
                        .map(|c| (c.name().to_string(), c.type_()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let make_batch_cols = || -> Vec<(String, AttributeColumn)> {
            col_schema
                .iter()
                .map(|(name, col_type)| {
                    let col = match *col_type {
                        ColumnType::Byte
                        | ColumnType::UByte
                        | ColumnType::Short
                        | ColumnType::UShort
                        | ColumnType::Int
                        | ColumnType::UInt
                        | ColumnType::Long
                        | ColumnType::ULong => {
                            AttributeColumn::Integer(Vec::with_capacity(BATCH_SIZE))
                        }
                        ColumnType::Float | ColumnType::Double => {
                            AttributeColumn::Float(Vec::with_capacity(BATCH_SIZE))
                        }
                        _ => AttributeColumn::Text(Vec::with_capacity(BATCH_SIZE)),
                    };
                    (name.clone(), col)
                })
                .collect()
        };
        let mut iter = reader.select_bbox(bbox[0], bbox[1], bbox[2], bbox[3])?;
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        let mut id_counter = 0_u32;
        while let Some(feature) = iter.next()? {
            if cancel_stream.load(Ordering::Relaxed) {
                return Ok(());
            }
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            let mut extractor = PropertyExtractor {
                cols: &mut batch_cols,
                idx: None,
            };
            feature.process_properties(&mut extractor).ok();
            let id = extractor.idx.unwrap_or(id_counter);
            id_counter += 1;
            batch.push((id, [x, y]));
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, make_batch_cols()),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn stream_fgb_bbox(
        path: &GisFilePath,
        bbox: [f64; 4],
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel_stream: Arc<AtomicBool>,
        reader_cache: FgbReaderCache,
    ) -> Result<(), anyhow::Error> {
        use std::sync::atomic::Ordering;

        use flatgeobuf::HttpFgbReader;
        use wasm_bindgen::JsValue;

        if cancel_stream.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Parquet bytes path — dispatch to GeoParquetReader, skip FGB logic.
        if let GisFilePath::Bytes(bytes, _) = path {
            return GeoParquetReader::load_point_layer_batched_from_bytes(
                bytes.clone(),
                dest_idx,
                tx,
                selected_fields,
            );
        }
        let GisFilePath::HttpLocation(url) = path else {
            bail!("Wrong GisFilePath type!");
        };
        // Extract before the match so RefMut is dropped before any await point.
        let cached_reader = reader_cache.borrow_mut().get_mut(url).and_then(|v| v.pop());
        let reader = match cached_reader {
            Some(r) => {
                web_sys::console::log_1(&JsValue::from_str("stream_fgb_bbox: using cached reader"));
                r
            }
            None => {
                web_sys::console::log_1(&JsValue::from_str(&format!(
                    "stream_fgb_bbox: opening reader {}",
                    url
                )));
                HttpFgbReader::open(url).await?
            }
        };
        let header = reader.header().clone();
        const BATCH_SIZE: usize = 50_000;
        let col_schema: Vec<(String, ColumnType)> = {
            header
                .columns()
                .map(|cols| {
                    cols.iter()
                        .filter(|c| {
                            selected_fields
                                .as_ref()
                                .map_or(false, |sel| sel.iter().any(|s| s.as_str() == c.name()))
                        })
                        .map(|c| (c.name().to_string(), c.type_()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let make_batch_cols = || -> Vec<(String, AttributeColumn)> {
            col_schema
                .iter()
                .map(|(name, col_type)| {
                    let col = match *col_type {
                        ColumnType::Byte
                        | ColumnType::UByte
                        | ColumnType::Short
                        | ColumnType::UShort
                        | ColumnType::Int
                        | ColumnType::UInt
                        | ColumnType::Long
                        | ColumnType::ULong => {
                            AttributeColumn::Integer(Vec::with_capacity(BATCH_SIZE))
                        }
                        ColumnType::Float | ColumnType::Double => {
                            AttributeColumn::Float(Vec::with_capacity(BATCH_SIZE))
                        }
                        _ => AttributeColumn::Text(Vec::with_capacity(BATCH_SIZE)),
                    };
                    (name.clone(), col)
                })
                .collect()
        };
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        let mut id_counter = 0_u32;
        let mut features = reader
            .select_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
            .await?;
        web_sys::console::log_1(&JsValue::from_str("After Feature BBOX query"));
        let mut last_yield_ms = js_sys::Date::now();
        while let Some(feature) = features.next().await? {
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            let mut extractor = PropertyExtractor {
                cols: &mut batch_cols,
                idx: None,
            };
            feature.process_properties(&mut extractor).ok();
            let id = extractor.idx.unwrap_or(id_counter);
            id_counter += 1;
            batch.push((id, [x, y]));
            // Flush and yield to browser every ~16ms regardless of batch size,
            // avoiding setTimeout throttling that caps throughput at 1 batch/sec.
            let now_ms = js_sys::Date::now();
            if now_ms - last_yield_ms >= 16.0 {
                if !batch.is_empty() {
                    tx.send(BatchMessage::Points(
                        dest_idx,
                        std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                        std::mem::replace(&mut batch_cols, make_batch_cols()),
                    ))
                    .ok();
                }
                gloo_timers::future::sleep(std::time::Duration::ZERO).await;
                last_yield_ms = js_sys::Date::now();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        // Pre-open next reader while caller renders current batch.
        // With Cache-Control headers on the server, this is served from
        // browser cache and is nearly instant after the first load.
        if !cancel_stream.load(std::sync::atomic::Ordering::Relaxed) {
            if let Ok(next_reader) = HttpFgbReader::open(url).await {
                reader_cache
                    .borrow_mut()
                    .entry(url.to_string())
                    .or_default()
                    .push(next_reader);
            }
        }
        Ok(())
    }

    // ── load_selected_without_features ───────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_selected_without_features(
        path: GisFilePath,
        _indices: &[usize],
        field_names: Option<Vec<String>>,
    ) -> Result<Vec<LayerEntry>, anyhow::Error> {
        let GisFilePath::LocalFile(str_path) = &path else {
            use anyhow::bail;
            bail!("Wrong GisFilePath type!");
        };
        let descriptor = match vector_file_type(str_path)? {
            VectorFileType::GeoParquet => GeoParquetReader::load_descriptor(str_path)?,
            VectorFileType::FlatGeobuf => {
                let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
                Self::make_layer_descriptor(reader.header(), path)?
            }
        };
        Ok(vec![Self::layer_entry_from_descriptor(
            descriptor,
            field_names,
        )])
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_selected_without_features(
        path: GisFilePath,
        descriptor: LayerDescriptor,
        field_names: Option<Vec<String>>,
    ) -> Result<Vec<LayerEntry>, anyhow::Error> {
        // use flatgeobuf::HttpFgbReader;

        // let GisFilePath::HttpLocation(url) = path.clone() else {
        //     use anyhow::bail;
        //     use std::fmt::Error;
        //     bail!("brogen");
        // };
        // let reader = HttpFgbReader::open(&url).await?;
        let field_names = descriptor.field_names.clone();
        Ok(vec![Self::layer_entry_from_descriptor(
            descriptor,
            Some(field_names),
        )])
    }

    fn layer_entry_from_descriptor(
        descriptor: LayerDescriptor,
        field_names: Option<Vec<String>>,
    ) -> LayerEntry {
        println!("{:?}", field_names);
        let layer_kind = match descriptor.geometry_type.0 {
            1 => LayerKind::Points(PointCloudLayer {
                points: std::sync::Arc::new(Vec::new()),
                attributes: Vec::new(),
                field_names: field_names.unwrap_or(Vec::new()),
                index: None,
                bbox: None,
                viewport_mask: bitvec![0;descriptor.num_features as usize],
                filter_mask: bitvec![1;descriptor.num_features as usize],
            }),
            _ => LayerKind::Vector(GisLayer {
                name: descriptor.name.clone(),
                file_path: descriptor.location.to_string(),
                features: Vec::new(),
                field_names: field_names.unwrap_or(Vec::new()),
                extra_field_names: Vec::new(),
                quadtree: None,
                hilbert: None,
                point_only: true,
                world_bbox: [0., 0., 0., 0.],
            }),
        };
        LayerEntry {
            data: layer_kind,
            visible: true,
            name: descriptor.name.clone(),
            color: [0, 0, 255],
            opacity: 255,
            descriptor: descriptor.clone(),
            filters: Vec::new(),
            filter_logic: FilterLogic::default(),
            roi_bboxes: Vec::new(),
            selections: Vec::new(),
            active_selection: None,
        }
    }

    // fn layer_entry_from_reader<R: Read + Seek>(
    //     reader: FgbReader<R>,
    // ) -> Result<LayerEntry, Box<dyn std::error::Error>> {
    //     let descriptor = Self::make_layer_descriptor(reader.header())?;
    //     let layer_kind = match descriptor.geometry_type {
    //         GeometryType::Point => LayerKind::Points(PointCloudLayer::default()),
    //         _ => LayerKind::Vector(GisLayer::default()),
    //     };
    //     Ok(LayerEntry {
    //         data: layer_kind,
    //         visible: true,
    //         name: descriptor.name,
    //         color: [0, 0, 255],
    //         opacity: 255,
    //     })
    // }
}
