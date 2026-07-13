use anyhow::anyhow;
use datafusion::arrow::array::{
    Array, BinaryArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};

// ── local types (mirrors BatchMessage / AttributeColumn shapes) ───────────

#[derive(Debug)]
pub enum AttributeCol {
    Integer(Vec<i64>),
    Float(Vec<f64>),
    Text(Vec<String>),
}

impl AttributeCol {
    fn with_capacity(dt: &DataType, cap: usize) -> Self {
        match dt {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => AttributeCol::Integer(Vec::with_capacity(cap)),
            DataType::Float32 | DataType::Float64 => AttributeCol::Float(Vec::with_capacity(cap)),
            _ => AttributeCol::Text(Vec::with_capacity(cap)),
        }
    }
}

/// Parallel to BatchMessage::Points content.
#[derive(Debug)]
pub struct PointBatch {
    pub dest_idx: usize,
    pub points: Vec<(u32, [f64; 2])>,
    pub columns: Vec<(String, AttributeCol)>,
}

/// Candidate column name pairs to probe for lon/lat, in priority order.
/// First pair whose both columns exist in the schema wins.
const XY_CANDIDATES: &[(&str, &str)] = &[
    ("x", "y"),
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("lng", "lat"),
    ("long", "lat"),
];

#[derive(Debug)]
pub enum GeometrySource {
    /// Two float columns carrying x/lon and y/lat, detected by name heuristic.
    XYColumns { x_col: String, y_col: String },
    /// Standard GeoParquet `geometry` WKB binary column.
    WkbColumn,
}

#[derive(Debug)]
pub struct LayerDescriptor {
    pub name: String,
    pub num_features: u64,
    pub field_names: Vec<String>,
    pub geometry_source: GeometrySource,
}

// ── WKB point decode (no extra dep: spec is fixed 21 bytes for Point) ─────

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
        return None; // not a Point
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

// ── helpers ───────────────────────────────────────────────────────────────

fn extract_f64_at(arr: &dyn Array, i: usize) -> Option<f64> {
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        Some(a.value(i))
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        Some(a.value(i) as f64)
    } else {
        None
    }
}

fn extract_i64_at(arr: &dyn Array, i: usize) -> Option<i64> {
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

// ── GeoParquetReader ──────────────────────────────────────────────────────

pub struct GeoParquetReader;

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
        // Check exact candidates first, then do a substring scan for columns whose
        // lowercased name contains "lon"/"lat" or "x"/"y" as whole words.
        let names: Vec<String> = schema.fields().iter().map(|f| f.name().to_lowercase()).collect();
        let orig: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Priority 1: known exact pairs
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
        // Priority 2: substring scan — find first col containing "lon"/"lng" and "lat"
        let x_col = orig.iter().find(|n| {
            let l = n.to_lowercase();
            l.contains("longitude") || l.contains("_lon") || l.contains("lon_") || l == "lon" || l.contains("_lng") || l.contains("lng_")
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
        // Fallback: no geometry detected — caller should handle this
        GeometrySource::XYColumns {
            x_col: "x".to_string(),
            y_col: "y".to_string(),
        }
    }

    /// Mirrors GisReader::load_layer_descriptor.
    pub async fn load_descriptor(path: &str) -> anyhow::Result<LayerDescriptor> {
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geometry_source = Self::detect_geometry_source(&schema);

        let count_batches = ctx
            .sql("SELECT COUNT(*) FROM layer")
            .await?
            .collect()
            .await?;
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

        let geom_cols: std::collections::HashSet<String> = match &geometry_source {
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
            geometry_source,
        })
    }

    /// Mirrors GisReader::read_file(ReadOp::Full) — streams all points via channel.
    pub async fn load_point_layer_batched(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<PointBatch>,
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

        for batch in &batches {
            if let Some(pb) = Self::extract_points(batch, dest_idx, &geom_src, None, selected_fields.as_deref())? {
                tx.send(pb).ok();
            }
        }
        Ok(())
    }

    /// Mirrors GisReader::read_file(ReadOp::Bbox) — bbox-filtered point stream via channel.
    ///
    /// XY-column files push the bbox predicate into SQL (fast, uses row-group stats
    /// if data is spatially sorted). WKB-column files do a full scan and filter in Rust.
    pub async fn stream_bbox(
        path: &str,
        bbox: [f64; 4],
        dest_idx: usize,
        tx: mpsc::SyncSender<PointBatch>,
        selected_fields: Option<Vec<String>>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
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
        for batch in &batches {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let wkb_bbox = matches!(geom_src, GeometrySource::WkbColumn).then_some(bbox);
            if let Some(pb) = Self::extract_points(batch, dest_idx, &geom_src, wkb_bbox, selected_fields.as_deref())? {
                tx.send(pb).ok();
            }
        }
        Ok(())
    }

    /// Returns idx values matching a SQL WHERE clause — replaces the separate parquet
    /// query used for attribute filtering in app.rs.
    pub async fn query_attribute_filter(path: &str, where_clause: &str) -> anyhow::Result<Vec<u32>> {
        let ctx = Self::make_ctx(path).await?;
        let sql = format!("SELECT idx FROM layer WHERE {where_clause}");
        let batches = ctx.sql(&sql).await?.collect().await?;
        let mut ids = Vec::new();
        for batch in &batches {
            let col = batch
                .column_by_name("idx")
                .ok_or_else(|| anyhow!("no idx column in parquet file"))?;
            if let Some(a) = col.as_any().downcast_ref::<UInt32Array>() {
                ids.extend_from_slice(a.values());
            } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
                ids.extend(a.values().iter().map(|v| *v as u32));
            } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                ids.extend(a.values().iter().map(|v| *v as u32));
            }
        }
        Ok(ids)
    }

    fn build_select(
        geom_src: &GeometrySource,
        selected_fields: Option<&[String]>,
        schema: &datafusion::arrow::datatypes::Schema,
    ) -> String {
        let mut cols = vec!["idx".to_string()];
        match geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                cols.push(x_col.clone());
                cols.push(y_col.clone());
            }
            GeometrySource::WkbColumn => cols.push("geometry".to_string()),
        }
        if let Some(fields) = selected_fields {
            for f in fields {
                if schema.field_with_name(f).is_ok() {
                    cols.push(f.clone());
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
    ) -> anyhow::Result<Option<PointBatch>> {
        let nrows = batch.num_rows();
        if nrows == 0 {
            return Ok(None);
        }

        // idx column → fallback to row position
        let ids: Vec<u32> = {
            let col = batch.column_by_name("idx");
            match col {
                Some(c) => {
                    if let Some(a) = c.as_any().downcast_ref::<UInt32Array>() {
                        a.values().to_vec()
                    } else if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
                        a.values().iter().map(|v| *v as u32).collect()
                    } else if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                        a.values().iter().map(|v| *v as u32).collect()
                    } else {
                        (0..nrows as u32).collect()
                    }
                }
                None => (0..nrows as u32).collect(),
            }
        };

        // geometry → per-row Option<[f64;2]>
        let coords: Vec<Option<[f64; 2]>> = match geom_src {
            GeometrySource::XYColumns { x_col, y_col } => {
                let xs = batch.column_by_name(x_col);
                let ys = batch.column_by_name(y_col);
                (0..nrows)
                    .map(|i| {
                        let x = xs.and_then(|a| extract_f64_at(a.as_ref(), i))?;
                        let y = ys.and_then(|a| extract_f64_at(a.as_ref(), i))?;
                        Some([x, y])
                    })
                    .collect()
            }
            GeometrySource::WkbColumn => {
                let geom_col = batch.column_by_name("geometry");
                (0..nrows)
                    .map(|i| {
                        let arr = geom_col?.as_any().downcast_ref::<BinaryArray>()?;
                        decode_wkb_point(arr.value(i))
                    })
                    .collect()
            }
        };

        // attribute columns — parallel to output points (not to input rows)
        let mut columns: Vec<(String, AttributeCol)> = match selected_fields {
            Some(fields) => fields
                .iter()
                .filter_map(|f| {
                    let idx = batch.schema().index_of(f).ok()?;
                    let dt = batch.schema().field(idx).data_type().clone();
                    Some((f.clone(), AttributeCol::with_capacity(&dt, nrows)))
                })
                .collect(),
            None => vec![],
        };

        let mut points = Vec::with_capacity(nrows);
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
                    AttributeCol::Integer(v) => {
                        v.push(arr.and_then(|a| extract_i64_at(a.as_ref(), i)).unwrap_or(0));
                    }
                    AttributeCol::Float(v) => {
                        let val = arr
                            .and_then(|a| extract_f64_at(a.as_ref(), i))
                            .unwrap_or(0.0);
                        v.push(val);
                    }
                    AttributeCol::Text(v) => {
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
        Ok(Some(PointBatch { dest_idx, points, columns }))
    }
}

// ── main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./assets/output.parquet".to_string());

    println!("=== GeoParquetReader test ===");
    println!("file: {path}");

    let desc = GeoParquetReader::load_descriptor(&path).await?;
    println!("name:     {}", desc.name);
    println!("features: {}", desc.num_features);
    println!("fields:   {:?}", desc.field_names);
    println!("geom:     {:?}", desc.geometry_source);

    // Load all points, count them
    let (tx, rx) = mpsc::sync_channel::<PointBatch>(64);
    let path_clone = path.clone();
    tokio::spawn(async move {
        if let Err(e) = GeoParquetReader::load_point_layer_batched(&path_clone, 0, tx, None).await
        {
            eprintln!("load error: {e}");
        }
    });
    let mut total = 0usize;
    while let Ok(batch) = rx.recv() {
        total += batch.points.len();
    }
    println!("loaded:   {total} points");

    // Bbox stream — hard-coded world bbox as smoke test
    let (tx2, rx2) = mpsc::sync_channel::<PointBatch>(64);
    let path_clone = path.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        let bbox = [-180.0, -90.0, 180.0, 90.0];
        if let Err(e) =
            GeoParquetReader::stream_bbox(&path_clone, bbox, 0, tx2, None, cancel_clone).await
        {
            eprintln!("stream error: {e}");
        }
    });
    let mut bbox_total = 0usize;
    while let Ok(batch) = rx2.recv() {
        bbox_total += batch.points.len();
    }
    println!("bbox:     {bbox_total} points (world bbox)");

    Ok(())
}
