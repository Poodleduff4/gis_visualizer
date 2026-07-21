use anyhow::bail;
use bitvec::{bitarr, bitvec, vec::BitVec, BitArr};
use flatgeobuf::{
    ColumnType, FallibleStreamingIterator, FeatureProperties, FgbReader, GeometryType, Header,
};
use geo_types::{Coord, Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
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
    Array, BinaryArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeBinaryArray,
    ListArray, StringArray, StructArray, UInt32Array, UInt64Array,
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
impl std::fmt::Display for GisFilePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GisFilePath::LocalFile(p) => write!(f, "{p}"),
            GisFilePath::HttpLocation(p) => write!(f, "{p}"),
            GisFilePath::Bytes(_, name) => write!(f, "{name}"),
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
    /// Human-readable CRS label (e.g. "EPSG:4326"), if the file declares one.
    pub crs: Option<String>,
    /// Numeric EPSG code parsed out of `crs`, when it's EPSG-identifiable.
    /// Drives the on-load "Convert to WGS84" option.
    pub crs_epsg: Option<u16>,
}

/// Extracts the numeric code from a `"EPSG:<code>"`-prefixed label (the
/// format every CRS-label helper in this file uses), tolerating a trailing
/// note like `" (default)"`.
fn parse_epsg_from_label(label: &str) -> Option<u16> {
    label.strip_prefix("EPSG:")?.split_whitespace().next()?.parse().ok()
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
pub(crate) enum VectorFileType {
    FlatGeobuf,
    GeoParquet,
    GeoJson,
}

pub(crate) fn vector_file_type(path: &str) -> anyhow::Result<VectorFileType> {
    if path.ends_with(".parquet") {
        Ok(VectorFileType::GeoParquet)
    } else if path.ends_with(".fgb") {
        Ok(VectorFileType::FlatGeobuf)
    } else if path.ends_with(".geojson") || path.ends_with(".json") {
        Ok(VectorFileType::GeoJson)
    } else {
        bail!("Unsupported vector file extension: {path}");
    }
}

/// How much of a file to read: everything, only what intersects a bbox, or
/// a contiguous slice of features (progressive/batch loading).
#[derive(Clone, Copy)]
pub enum ReadOp {
    Full,
    Bbox([f64; 4]),
    /// Skip `offset` features, then read up to `limit` of them.
    Range { offset: u64, limit: u64 },
}

// ── FlatGeobuf point-batch helpers — shared by every full-scan / bbox-scan
// site (native full-scan, native bbox-scan, wasm HTTP bbox-scan) ──────────

/// Column schema restricted to the caller's selected attribute fields.
fn fgb_column_schema(
    header: Header<'_>,
    selected_fields: Option<&[String]>,
) -> Vec<(String, ColumnType)> {
    header
        .columns()
        .map(|cols| {
            cols.iter()
                .filter(|c| {
                    selected_fields.is_some_and(|sel| sel.iter().any(|s| s.as_str() == c.name()))
                })
                .map(|c| (c.name().to_string(), c.type_()))
                .collect()
        })
        .unwrap_or_default()
}

fn make_batch_cols(col_schema: &[(String, ColumnType)], cap: usize) -> Vec<(String, AttributeColumn)> {
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
                | ColumnType::ULong => AttributeColumn::Integer(Vec::with_capacity(cap)),
                ColumnType::Float | ColumnType::Double => {
                    AttributeColumn::Float(Vec::with_capacity(cap))
                }
                _ => AttributeColumn::Text(Vec::with_capacity(cap)),
            };
            (name.clone(), col)
        })
        .collect()
}

/// Pulls the point coords + selected attributes out of one FGB feature.
/// Returns `None` for non-point geometries (caller skips the row).
fn extract_point_row<F>(
    feature: &F,
    cols: &mut Vec<(String, AttributeColumn)>,
    id_counter: &mut u32,
) -> Option<(u32, [f64; 2])>
where
    F: ToGeo + FeatureProperties,
{
    let [x, y] = match feature.to_geo() {
        Ok(Geometry::Point(p)) => [p.x(), p.y()],
        _ => return None,
    };
    let mut extractor = PropertyExtractor { cols, idx: None };
    feature.process_properties(&mut extractor).ok();
    let id = extractor.idx.unwrap_or(*id_counter);
    *id_counter += 1;
    Some((id, [x, y]))
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

/// Reads just the WKB header's byte order + geometry-type code (1=Point,
/// 2=LineString, 3=Polygon, …), stripping the Z/M/ZM thousands-place flag
/// some writers set (EWKB-style 1001/2001/…) so callers only see the base
/// 2D type. Used to sniff whether a GeoParquet `geometry` column actually
/// holds points before committing to the point-cloud loading path — the
/// column's Arrow type alone (Binary) can't tell point from line/polygon.
fn wkb_header_geom_type(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 5 {
        return None;
    }
    let le = bytes[0] == 1;
    let raw = if le {
        u32::from_le_bytes(bytes[1..5].try_into().ok()?)
    } else {
        u32::from_be_bytes(bytes[1..5].try_into().ok()?)
    };
    Some(raw % 1000)
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

/// Generic per-cell Arrow -> `AttributeValue` decode for the GeoParquet
/// vector path (unlike `pq_attr_col`, which builds typed point-cloud
/// columns up front, vector features hold a loose `HashMap` so the type
/// is decided per value instead).
fn pq_attr_value(arr: &dyn Array, i: usize) -> Option<AttributeValue> {
    if arr.is_null(i) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Some(AttributeValue::Text(a.value(i).to_string()));
    }
    if let Some(v) = pq_extract_i64(arr, i) {
        return Some(AttributeValue::Integer(v));
    }
    if let Some(v) = pq_extract_f64(arr, i) {
        return Some(AttributeValue::Float(v));
    }
    None
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

/// Shared by native (DataFusion) and wasm (parquet-bytes) GeoParquet readers —
/// both operate on the same `arrow::datatypes::Schema` type (arrow = "53" is
/// the same crate on both targets; DataFusion re-exports it rather than
/// wrapping it).
fn detect_geometry_source(schema: &arrow::datatypes::Schema) -> GeometrySource {
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

/// Parses the GeoParquet spec's "geo" key-value metadata entry for a
/// human-readable CRS label. A missing/null `crs` means the column defaults
/// to OGC:CRS84 (== EPSG:4326) per spec.
fn parquet_geo_crs_label(kv: Option<&Vec<parquet::format::KeyValue>>) -> Option<String> {
    let geo_json = kv?.iter().find(|k| k.key == "geo")?.value.as_ref()?;
    let geo: serde_json::Value = serde_json::from_str(geo_json).ok()?;
    let primary = geo
        .get("primary_column")
        .and_then(|v| v.as_str())
        .unwrap_or("geometry");
    let crs = geo.get("columns")?.get(primary)?.get("crs");
    match crs {
        None | Some(serde_json::Value::Null) => Some("EPSG:4326 (default)".to_string()),
        Some(crs) => {
            if let Some(code) = crs.get("id").and_then(|id| {
                let authority = id.get("authority")?.as_str()?;
                let code = id.get("code")?;
                Some(format!("{authority}:{code}"))
            }) {
                Some(code)
            } else if let Some(name) = crs.get("name").and_then(|v| v.as_str()) {
                Some(name.to_string())
            } else {
                Some("custom CRS".to_string())
            }
        }
    }
}

/// Parses the GeoParquet spec's "geo" key-value metadata entry for the
/// primary geometry column's `encoding`. GeoParquet supports two families:
/// `"WKB"` (a `Binary` column of well-known-binary blobs, the default when
/// this key is missing) and the "native GeoArrow" encodings — `"point"`,
/// `"linestring"`, `"polygon"`, `"multipoint"`, `"multilinestring"`,
/// `"multipolygon"` — where the column is itself a (possibly nested) Arrow
/// `List<Struct<x, y>>` and there's no WKB to decode at all. Lowercased so
/// callers can match without re-normalizing case.
fn parquet_geo_encoding(kv: Option<&Vec<parquet::format::KeyValue>>) -> Option<String> {
    let geo_json = kv?.iter().find(|k| k.key == "geo")?.value.as_ref()?;
    let geo: serde_json::Value = serde_json::from_str(geo_json).ok()?;
    let primary = geo
        .get("primary_column")
        .and_then(|v| v.as_str())
        .unwrap_or("geometry");
    geo.get("columns")?
        .get(primary)?
        .get("encoding")?
        .as_str()
        .map(|s| s.to_lowercase())
}

/// A single coordinate pair or an ordered list of nested `CoordNest`s,
/// mirroring the shape of a native-GeoArrow geometry column: a `linestring`
/// column is `List<Struct<x,y>>` (one level of nesting around points), a
/// `polygon` column is `List<List<Struct<x,y>>>` (rings around points), a
/// `multipolygon` column adds one more level still. Recursing through
/// `ListArray`/`StructArray` without hard-coding a depth lets one function
/// (`geoarrow_extract_row`) read any of the encodings — the encoding name
/// only matters once turning this into a `geo_types::Geometry`.
enum CoordNest {
    Point([f64; 2]),
    List(Vec<CoordNest>),
}

/// Recursively unpacks one row of a native-GeoArrow geometry column: peels
/// off `ListArray` levels until it reaches the `Struct<x: f64, y: f64>`
/// leaf. Returns `None` on a null slot or an unexpected (non list/struct)
/// column shape.
fn geoarrow_extract_row(array: &dyn Array, row: usize) -> Option<CoordNest> {
    if array.is_null(row) {
        return None;
    }
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        let value = list.value(row);
        let mut items = Vec::with_capacity(value.len());
        for i in 0..value.len() {
            items.push(geoarrow_extract_row(value.as_ref(), i)?);
        }
        return Some(CoordNest::List(items));
    }
    if let Some(st) = array.as_any().downcast_ref::<StructArray>() {
        let x = st.column_by_name("x")?.as_any().downcast_ref::<Float64Array>()?.value(row);
        let y = st.column_by_name("y")?.as_any().downcast_ref::<Float64Array>()?.value(row);
        return Some(CoordNest::Point([x, y]));
    }
    None
}

fn coordnest_flatten(nest: &CoordNest, out: &mut Vec<Coord<f64>>) {
    match nest {
        CoordNest::Point([x, y]) => out.push(Coord { x: *x, y: *y }),
        CoordNest::List(items) => {
            for item in items {
                coordnest_flatten(item, out);
            }
        }
    }
}

fn coordnest_to_linestring(nest: &CoordNest) -> LineString<f64> {
    let mut coords = Vec::new();
    coordnest_flatten(nest, &mut coords);
    LineString(coords)
}

fn coordnest_to_polygon(nest: &CoordNest) -> Option<Polygon<f64>> {
    let CoordNest::List(rings) = nest else { return None };
    let mut rings = rings.iter().map(coordnest_to_linestring);
    let exterior = rings.next()?;
    Some(Polygon::new(exterior, rings.collect()))
}

/// Converts one row's already-extracted `CoordNest` into the
/// `geo_types::Geometry` matching `encoding` (a lowercased GeoParquet
/// native-encoding name — see `parquet_geo_encoding`). `None` means the
/// nesting depth didn't match what `encoding` expects (a malformed file).
fn coordnest_to_geometry(nest: CoordNest, encoding: &str) -> Option<Geometry<f64>> {
    match encoding {
        "point" => match nest {
            CoordNest::Point([x, y]) => Some(Geometry::Point(Point::new(x, y))),
            _ => None,
        },
        "multipoint" => match nest {
            CoordNest::List(pts) => Some(Geometry::MultiPoint(MultiPoint(
                pts.into_iter()
                    .filter_map(|p| match p {
                        CoordNest::Point([x, y]) => Some(Point::new(x, y)),
                        _ => None,
                    })
                    .collect(),
            ))),
            _ => None,
        },
        "linestring" => match &nest {
            CoordNest::List(_) => Some(Geometry::LineString(coordnest_to_linestring(&nest))),
            _ => None,
        },
        "multilinestring" => match nest {
            CoordNest::List(lines) => Some(Geometry::MultiLineString(MultiLineString(
                lines.iter().map(coordnest_to_linestring).collect(),
            ))),
            _ => None,
        },
        "polygon" => coordnest_to_polygon(&nest).map(Geometry::Polygon),
        "multipolygon" => match nest {
            CoordNest::List(polys) => Some(Geometry::MultiPolygon(MultiPolygon(
                polys.iter().filter_map(coordnest_to_polygon).collect(),
            ))),
            _ => None,
        },
        _ => None,
    }
}

/// Shared by native and wasm GeoParquet readers. `id_base` keeps ids
/// unique/stable across a multi-batch file when no `idx` column is present —
/// without it every batch would restart at 0, so ids collide and
/// filter/selection lookups by id silently match the wrong rows past the
/// first batch.
fn extract_points(
    batch: &RecordBatch,
    dest_idx: usize,
    geom_src: &GeometrySource,
    bbox_filter: Option<[f64; 4]>,
    selected_fields: Option<&[String]>,
    id_base: u32,
) -> anyhow::Result<Option<BatchMessage>> {
    let nrows = batch.num_rows();
    if nrows == 0 {
        return Ok(None);
    }

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
                .and_then(|arr| arrow::compute::cast(arr, &ArrowDataType::Binary).ok());
            (0..nrows)
                .map(|i| {
                    let arr = casted.as_ref()?;
                    if let Some(a) = arr.as_any().downcast_ref::<BinaryArray>() {
                        decode_wkb_point(a.value(i))
                    } else {
                        arr.as_any()
                            .downcast_ref::<LargeBinaryArray>()
                            .and_then(|a| decode_wkb_point(a.value(i)))
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

/// Vector-geometry counterpart to `extract_points`: decodes each row's
/// `geometry` WKB into a full `geo_types::Geometry` (points, lines,
/// polygons — whatever the file holds) rather than reducing it to an
/// `[x, y]` pair, so files like line-based transit routes actually load
/// instead of every row silently failing `decode_wkb_point`'s
/// Point-only check. Only reachable for `GeometrySource::WkbColumn` — an
/// XY-columns file is definitionally points-only, so callers route those
/// through `extract_points` instead.
fn extract_vector_batch(
    batch: &RecordBatch,
    dest_idx: usize,
    id_base: u32,
    selected_fields: Option<&[String]>,
    encoding: &str,
) -> anyhow::Result<Option<BatchMessage>> {
    let nrows = batch.num_rows();
    if nrows == 0 {
        return Ok(None);
    }
    let Some(geom_col) = batch.column_by_name("geometry") else {
        return Ok(None);
    };

    // WKB: cast to Binary up front and decode each row's bytes via geozero.
    // Native GeoArrow (`encoding` is one of "point"/"linestring"/…): the
    // column is already a nested `List<Struct<x,y>>` Arrow array, so each
    // row is walked directly with `geoarrow_extract_row` instead — there's
    // no bytes to decode.
    let is_wkb = encoding == "wkb";
    let casted = is_wkb.then(|| arrow::compute::cast(geom_col, &ArrowDataType::Binary)).transpose()?;
    if is_wkb && casted.is_none() {
        return Ok(None);
    }

    let schema = batch.schema();
    let attr_fields: Vec<&str> = schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .filter(|n| *n != "geometry" && *n != "idx")
        .filter(|n| selected_fields.is_none_or(|f| f.iter().any(|s| s == n)))
        .collect();

    let mut features = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let geo = if is_wkb {
            let casted = casted.as_ref().expect("checked above");
            let wkb_bytes = if let Some(a) = casted.as_any().downcast_ref::<BinaryArray>() {
                if a.is_null(i) {
                    continue;
                }
                a.value(i)
            } else if let Some(a) = casted.as_any().downcast_ref::<LargeBinaryArray>() {
                if a.is_null(i) {
                    continue;
                }
                a.value(i)
            } else {
                continue;
            };
            let Ok(geo) = geozero::wkb::Wkb(wkb_bytes).to_geo() else {
                continue;
            };
            geo
        } else {
            let Some(nest) = geoarrow_extract_row(geom_col.as_ref(), i) else {
                continue;
            };
            let Some(geo) = coordnest_to_geometry(nest, encoding) else {
                continue;
            };
            geo
        };
        let mut attributes = HashMap::new();
        for name in &attr_fields {
            if let Some(col) = batch.column_by_name(name) {
                if let Some(v) = pq_attr_value(col.as_ref(), i) {
                    attributes.insert((*name).to_string(), v);
                }
            }
        }
        features.push(GisFeature::new(id_base as usize + i, geo, attributes));
    }
    if features.is_empty() {
        return Ok(None);
    }
    Ok(Some(BatchMessage::Vector(dest_idx, features)))
}

pub struct GeoParquetReader;

#[cfg(not(target_arch = "wasm32"))]
impl GeoParquetReader {
    async fn make_ctx(path: &str) -> datafusion::error::Result<SessionContext> {
        // Single partition: with >1, DataFusion's ParquetExec can split a file
        // across partitions by byte range (repartition_file_scans) rather than
        // by whole row group, which has silently dropped a contiguous run of
        // rows here (visible as a spatial "hole" — the file's point order is
        // spatially clustered). We already materialize the whole result via
        // `.collect()` before handing it to the channel, so there's no
        // wall-clock win from parallel decode worth the correctness risk.
        let config = SessionConfig::new().with_target_partitions(1);
        let ctx = SessionContext::new_with_config(config);
        ctx.register_parquet("layer", path, ParquetReadOptions::default())
            .await?;
        Ok(ctx)
    }

    /// Reads one non-null `geometry` value and decodes its WKB header to
    /// find the actual geometry type stored in the column. `detect_geometry_source`
    /// only tells us *how* geometry is encoded (WKB column vs. XY columns),
    /// not *what kind* of geometry it is — without this, every GeoParquet
    /// file was assumed to hold points, so lines/polygons loaded 0 features
    /// (every row silently failed the point-only WKB decode).
    async fn sniff_wkb_geometry_type(ctx: &SessionContext) -> Option<u32> {
        let batches = ctx
            .sql("SELECT geometry FROM layer WHERE geometry IS NOT NULL LIMIT 1")
            .await
            .ok()?
            .collect()
            .await
            .ok()?;
        let batch = batches.first()?;
        let casted = arrow::compute::cast(batch.column(0), &ArrowDataType::Binary).ok()?;
        let bytes = if let Some(a) = casted.as_any().downcast_ref::<BinaryArray>() {
            (!a.is_empty()).then(|| a.value(0))?
        } else if let Some(a) = casted.as_any().downcast_ref::<LargeBinaryArray>() {
            (!a.is_empty()).then(|| a.value(0))?
        } else {
            return None;
        };
        wkb_header_geom_type(bytes)
    }

    async fn load_descriptor_async(path: &str) -> anyhow::Result<LayerDescriptor> {
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = detect_geometry_source(&schema);
        let encoding = Self::file_geo_encoding(path);
        let geometry_type = match &geom_src {
            GeometrySource::XYColumns { .. } => GeometryType(1),
            GeometrySource::WkbColumn if encoding == "wkb" => {
                match Self::sniff_wkb_geometry_type(&ctx).await {
                    Some(1) => GeometryType(1),
                    _ => GeometryType(0),
                }
            }
            // Native GeoArrow encoding: the "geo" metadata already names
            // the geometry kind directly, no need to sniff row bytes.
            GeometrySource::WkbColumn => {
                if encoding == "point" {
                    GeometryType(1)
                } else {
                    GeometryType(0)
                }
            }
        };

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

        let crs = Self::file_crs(path);
        Ok(LayerDescriptor {
            name,
            num_features,
            field_names,
            geometry_type,
            location: GisFilePath::LocalFile(path.to_string()),
            crs_epsg: crs.as_deref().and_then(parse_epsg_from_label),
            crs,
        })
    }

    /// Reads the parquet file's own key-value metadata synchronously — a
    /// separate, cheap open outside the DataFusion session, since the async
    /// `SessionContext`/table API doesn't expose raw file metadata.
    fn file_crs(path: &str) -> Option<String> {
        use parquet::file::reader::FileReader;
        let file = std::fs::File::open(path).ok()?;
        let reader = parquet::file::reader::SerializedFileReader::new(file).ok()?;
        parquet_geo_crs_label(reader.metadata().file_metadata().key_value_metadata())
    }

    /// The primary geometry column's GeoParquet `encoding`, lowercased and
    /// defaulted to `"wkb"` when the file's "geo" metadata omits it (or has
    /// no "geo" key at all — an ordinary XY-columns file, say). Every loader
    /// below calls this once to decide whether `extract_points`/
    /// `extract_vector_batch` should cast `geometry` to `Binary` and decode
    /// WKB, or walk it as a native nested `List<Struct<x,y>>` instead.
    fn file_geo_encoding(path: &str) -> String {
        use parquet::file::reader::FileReader;
        (|| -> Option<String> {
            let file = std::fs::File::open(path).ok()?;
            let reader = parquet::file::reader::SerializedFileReader::new(file).ok()?;
            parquet_geo_encoding(reader.metadata().file_metadata().key_value_metadata())
        })()
        .unwrap_or_else(|| "wkb".to_string())
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

    async fn load_point_layer_batched_async(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = detect_geometry_source(&schema);
        let sel = Self::build_select(&geom_src, selected_fields.as_deref(), &schema);

        let batches = ctx
            .sql(&format!("SELECT {} FROM layer", sel))
            .await?
            .collect()
            .await?;

        let mut id_base: u32 = 0;
        for batch in &batches {
            let nrows = batch.num_rows();
            if let Some(msg) = extract_points(batch, dest_idx, &geom_src, None, selected_fields.as_deref(), id_base)? {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
        }
        Ok(())
    }

    pub fn load_point_layer_batched(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(Self::load_point_layer_batched_async(path, dest_idx, tx, selected_fields))
    }

    /// Full-scan loader for GeoParquet files whose `geometry` column holds
    /// non-point geometry (lines/polygons), used instead of
    /// `load_point_layer_batched` once `load_descriptor`'s WKB sniff finds
    /// something other than Point.
    async fn load_vector_layer_batched_async(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = detect_geometry_source(&schema);
        let encoding = Self::file_geo_encoding(path);
        let sel = Self::build_select(&geom_src, selected_fields.as_deref(), &schema);

        let batches = ctx
            .sql(&format!("SELECT {} FROM layer", sel))
            .await?
            .collect()
            .await?;

        let mut id_base: u32 = 0;
        for batch in &batches {
            let nrows = batch.num_rows();
            if let Some(msg) =
                extract_vector_batch(batch, dest_idx, id_base, selected_fields.as_deref(), &encoding)?
            {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
        }
        Ok(())
    }

    pub fn load_vector_layer_batched(
        path: &str,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(Self::load_vector_layer_batched_async(path, dest_idx, tx, selected_fields))
    }

    /// Batch/progressive-load entry point: seeks straight to `offset` and
    /// reads at most `limit` rows. Bypasses DataFusion — its SQL
    /// `LIMIT`/`OFFSET` isn't pushed down to row-group skipping, so it
    /// re-decodes every row from the start of the file on every call, which
    /// made later batches quadratically slower. Row groups are cheap to skip
    /// via metadata (no decode), so only the row groups actually covering
    /// `[offset, offset+limit)` get read — real seeking.
    pub fn load_point_layer_range(
        path: &str,
        offset: u64,
        limit: u64,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().as_ref().clone();
        let geom_src = detect_geometry_source(&schema);

        let row_group_counts: Vec<u64> =
            builder.metadata().row_groups().iter().map(|rg| rg.num_rows() as u64).collect();
        let mut cumulative = 0u64;
        let mut selected_rgs = Vec::new();
        let mut skip_in_first = 0u64;
        let mut started = false;
        for (i, &count) in row_group_counts.iter().enumerate() {
            let rg_start = cumulative;
            let rg_end = cumulative + count;
            if rg_end > offset && rg_start < offset + limit {
                if !started {
                    skip_in_first = offset - rg_start;
                    started = true;
                }
                selected_rgs.push(i);
            }
            cumulative = rg_end;
            if rg_start >= offset + limit {
                break;
            }
        }

        // Default reader batch_size is 1024 rows, independent of row-group
        // size — without raising it, a 100k-row batch still gets decoded
        // (and sent/extended into the layer) in ~100 small pieces. Size it
        // to the whole range so it collapses to as few `BatchMessage`s as
        // possible (merged across row-group boundaries by the reader).
        let reader = builder
            .with_row_groups(selected_rgs)
            .with_batch_size(limit.max(1) as usize)
            .build()?;
        let mut id_base: u32 = offset as u32;
        let mut to_skip = skip_in_first as usize;
        let mut remaining = limit as usize;
        for result in reader {
            if remaining == 0 {
                break;
            }
            let mut batch = result?;
            if to_skip > 0 {
                let n = batch.num_rows();
                if to_skip >= n {
                    to_skip -= n;
                    continue;
                }
                batch = batch.slice(to_skip, n - to_skip);
                to_skip = 0;
            }
            if batch.num_rows() > remaining {
                batch = batch.slice(0, remaining);
            }
            let nrows = batch.num_rows();
            if let Some(msg) =
                extract_points(&batch, dest_idx, &geom_src, None, selected_fields.as_deref(), id_base)?
            {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
            remaining -= nrows;
        }
        Ok(())
    }

    /// Vector-geometry counterpart to `load_point_layer_range`: same
    /// row-group seeking, but decodes full geometries via
    /// `extract_vector_batch` instead of point-only coordinates.
    pub fn load_vector_layer_range(
        path: &str,
        offset: u64,
        limit: u64,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let encoding = Self::file_geo_encoding(path);
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        let row_group_counts: Vec<u64> =
            builder.metadata().row_groups().iter().map(|rg| rg.num_rows() as u64).collect();
        let mut cumulative = 0u64;
        let mut selected_rgs = Vec::new();
        let mut skip_in_first = 0u64;
        let mut started = false;
        for (i, &count) in row_group_counts.iter().enumerate() {
            let rg_start = cumulative;
            let rg_end = cumulative + count;
            if rg_end > offset && rg_start < offset + limit {
                if !started {
                    skip_in_first = offset - rg_start;
                    started = true;
                }
                selected_rgs.push(i);
            }
            cumulative = rg_end;
            if rg_start >= offset + limit {
                break;
            }
        }

        let reader = builder
            .with_row_groups(selected_rgs)
            .with_batch_size(limit.max(1) as usize)
            .build()?;
        let mut id_base: u32 = offset as u32;
        let mut to_skip = skip_in_first as usize;
        let mut remaining = limit as usize;
        for result in reader {
            if remaining == 0 {
                break;
            }
            let mut batch = result?;
            if to_skip > 0 {
                let n = batch.num_rows();
                if to_skip >= n {
                    to_skip -= n;
                    continue;
                }
                batch = batch.slice(to_skip, n - to_skip);
                to_skip = 0;
            }
            if batch.num_rows() > remaining {
                batch = batch.slice(0, remaining);
            }
            let nrows = batch.num_rows();
            if let Some(msg) =
                extract_vector_batch(&batch, dest_idx, id_base, selected_fields.as_deref(), &encoding)?
            {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
            remaining -= nrows;
        }
        Ok(())
    }

    /// Bbox-filtered stream. XY-column files push bbox into SQL; WKB files filter in Rust.
    pub async fn stream_bbox(
        path: &str,
        bbox: [f64; 4],
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        let ctx = Self::make_ctx(path).await?;
        let schema = ctx.table("layer").await?.schema().as_arrow().clone();
        let geom_src = detect_geometry_source(&schema);
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
            if let Some(msg) = extract_points(batch, dest_idx, &geom_src, wkb_bbox, selected_fields.as_deref(), id_base)? {
                tx.send(msg).ok();
            }
            id_base += nrows as u32;
        }
        Ok(())
    }

    /// Sync wrapper around [`Self::stream_bbox`] for callers outside an async runtime.
    pub fn stream_bbox_sync(
        path: &str,
        bbox: [f64; 4],
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(Self::stream_bbox(path, bbox, dest_idx, tx, selected_fields, cancel))
    }
}

// ── GeoParquetReader — wasm impl (reads from in-memory bytes) ────────────

#[cfg(target_arch = "wasm32")]
impl GeoParquetReader {
    pub fn load_descriptor_from_bytes(bytes: &[u8], name: &str) -> anyhow::Result<LayerDescriptor> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let bytes = ::bytes::Bytes::copy_from_slice(bytes);
        let raw: Arc<[u8]> = Arc::from(bytes.as_ref());
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let schema = builder.schema().as_ref().clone();
        let num_features = builder.metadata().file_metadata().num_rows() as u64;
        let crs = parquet_geo_crs_label(builder.metadata().file_metadata().key_value_metadata());
        let geom_src = detect_geometry_source(&schema);
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
            crs_epsg: crs.as_deref().and_then(parse_epsg_from_label),
            crs,
        })
    }

    pub fn load_point_layer_batched_from_bytes(
        bytes: Arc<[u8]>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let pq_bytes = ::bytes::Bytes::copy_from_slice(&bytes);
        let builder = ParquetRecordBatchReaderBuilder::try_new(pq_bytes)?;
        let schema = builder.schema().as_ref().clone();
        let geom_src = detect_geometry_source(&schema);
        let reader = builder.build()?;
        let mut id_base: u32 = 0;
        for result in reader {
            let batch = result?;
            let nrows = batch.num_rows();
            if let Some(msg) = extract_points(
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

// ── GeoJsonReader — pure-Rust parsing (geojson crate), identical on every
// target, so unlike GeoParquetReader this needs no cfg-gated duplication. ──

fn geojson_parse_collection(bytes: &[u8]) -> anyhow::Result<geojson::FeatureCollection> {
    let text = std::str::from_utf8(bytes)?;
    let parsed: geojson::GeoJson = text.parse()?;
    Ok(match parsed {
        geojson::GeoJson::FeatureCollection(fc) => fc,
        geojson::GeoJson::Feature(f) => geojson::FeatureCollection {
            bbox: None,
            features: vec![f],
            foreign_members: None,
        },
        geojson::GeoJson::Geometry(g) => geojson::FeatureCollection {
            bbox: None,
            features: vec![geojson::Feature {
                bbox: None,
                geometry: Some(g),
                id: None,
                properties: None,
                foreign_members: None,
            }],
            foreign_members: None,
        },
    })
}

fn geojson_feature_geometry(feature: &geojson::Feature) -> Option<Geometry<f64>> {
    feature
        .geometry
        .as_ref()
        .and_then(|g| Geometry::<f64>::try_from(g).ok())
}

fn geojson_is_points_only(fc: &geojson::FeatureCollection) -> bool {
    !fc.features.is_empty()
        && fc
            .features
            .iter()
            .all(|f| matches!(geojson_feature_geometry(f), Some(Geometry::Point(_))))
}

fn geojson_field_names(fc: &geojson::FeatureCollection) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for f in &fc.features {
        if let Some(props) = &f.properties {
            for k in props.keys() {
                if seen.insert(k.clone()) {
                    out.push(k.clone());
                }
            }
        }
    }
    out
}

/// GeoJSON is spec-locked to WGS84 (RFC 7946 §4), but pre-RFC7946 files
/// sometimes still carry the deprecated top-level `crs` member — check for
/// it before falling back to the spec default.
fn geojson_crs_label(bytes: &[u8]) -> Option<String> {
    let raw: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let name = raw
        .get("crs")
        .filter(|c| !c.is_null())
        .and_then(|c| c.get("properties"))
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str());
    match name {
        // e.g. "urn:ogc:def:crs:EPSG::3857" or "urn:ogc:def:crs:OGC:1.3:CRS84"
        Some(urn) if urn.to_uppercase().contains("EPSG") => {
            urn.rsplit(':').next().map(|code| format!("EPSG:{code}"))
        }
        Some(name) => Some(name.to_string()),
        None => Some("EPSG:4326 (default, per GeoJSON spec)".to_string()),
    }
}

/// Properties are dynamically typed in GeoJSON, unlike FGB/Parquet's fixed
/// columnar schema — the first non-null value of each selected field across
/// features decides the `AttributeColumn` storage type.
fn geojson_make_batch_cols(
    fc: &geojson::FeatureCollection,
    selected: &[String],
    cap: usize,
) -> Vec<(String, AttributeColumn)> {
    selected
        .iter()
        .map(|field| {
            let col = fc
                .features
                .iter()
                .find_map(|f| {
                    f.properties
                        .as_ref()?
                        .get(field)
                        .filter(|v| !v.is_null())
                })
                .map(|v| match v {
                    serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => {
                        AttributeColumn::Integer(Vec::with_capacity(cap))
                    }
                    serde_json::Value::Number(_) => AttributeColumn::Float(Vec::with_capacity(cap)),
                    _ => AttributeColumn::Text(Vec::with_capacity(cap)),
                })
                .unwrap_or_else(|| AttributeColumn::Text(Vec::with_capacity(cap)));
            (field.clone(), col)
        })
        .collect()
}

fn geojson_push_attrs(
    cols: &mut [(String, AttributeColumn)],
    props: Option<&serde_json::Map<String, serde_json::Value>>,
) {
    for (name, col) in cols.iter_mut() {
        let val = props.and_then(|p| p.get(name));
        match (col, val) {
            (AttributeColumn::Integer(v), Some(serde_json::Value::Number(n))) if n.is_i64() => {
                v.push(n.as_i64().unwrap());
            }
            (AttributeColumn::Integer(v), Some(serde_json::Value::Number(n))) if n.is_u64() => {
                v.push(n.as_u64().unwrap() as i64);
            }
            (AttributeColumn::Float(v), Some(serde_json::Value::Number(n))) => {
                v.push(n.as_f64().unwrap_or(0.0));
            }
            (AttributeColumn::Text(v), Some(serde_json::Value::String(s))) => v.push(s.clone()),
            (AttributeColumn::Text(v), Some(serde_json::Value::Bool(b))) => v.push(b.to_string()),
            (col, _) => col.push_default(),
        }
    }
}

fn geojson_attrs_map(
    props: Option<&serde_json::Map<String, serde_json::Value>>,
    selected: &std::collections::HashSet<String>,
) -> HashMap<String, AttributeValue> {
    let mut out = HashMap::new();
    let Some(props) = props else { return out };
    for (k, v) in props {
        if !selected.is_empty() && !selected.contains(k) {
            continue;
        }
        let attr = match v {
            serde_json::Value::String(s) => Some(AttributeValue::Text(s.clone())),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(AttributeValue::Integer(i))
                } else {
                    n.as_f64().map(AttributeValue::Float)
                }
            }
            serde_json::Value::Bool(b) => Some(AttributeValue::Text(b.to_string())),
            _ => None,
        };
        if let Some(a) = attr {
            out.insert(k.clone(), a);
        }
    }
    out
}

pub struct GeoJsonReader;

impl GeoJsonReader {
    pub fn load_descriptor_from_bytes(
        bytes: &[u8],
        name: &str,
        location: GisFilePath,
    ) -> anyhow::Result<LayerDescriptor> {
        let fc = geojson_parse_collection(bytes)?;
        let geometry_type = if geojson_is_points_only(&fc) {
            GeometryType(1)
        } else {
            GeometryType(0)
        };
        let display_name = std::path::Path::new(name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name)
            .to_string();
        let crs = geojson_crs_label(bytes);
        Ok(LayerDescriptor {
            name: display_name,
            num_features: fc.features.len() as u64,
            field_names: geojson_field_names(&fc),
            geometry_type,
            location,
            crs_epsg: crs.as_deref().and_then(parse_epsg_from_label),
            crs,
        })
    }

    pub fn load_point_batches_from_bytes(
        bytes: &[u8],
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let fc = geojson_parse_collection(bytes)?;
        const BATCH_SIZE: usize = 10_000;
        let selected = selected_fields.unwrap_or_default();
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = geojson_make_batch_cols(&fc, &selected, BATCH_SIZE);
        let mut id_counter = 0_u32;
        for feature in &fc.features {
            let Some(Geometry::Point(p)) = geojson_feature_geometry(feature) else {
                continue;
            };
            geojson_push_attrs(&mut batch_cols, feature.properties.as_ref());
            batch.push((id_counter, [p.x(), p.y()]));
            id_counter += 1;
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, geojson_make_batch_cols(&fc, &selected, BATCH_SIZE)),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    pub fn load_vector_batches_from_bytes(
        bytes: &[u8],
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let fc = geojson_parse_collection(bytes)?;
        const BATCH_SIZE: usize = 10_000;
        let selected_set: std::collections::HashSet<String> = selected_fields
            .map(|f| f.into_iter().collect())
            .unwrap_or_default();
        let mut batch: Vec<GisFeature> = Vec::with_capacity(BATCH_SIZE);
        let mut count = 0usize;
        for feature in &fc.features {
            let Some(geo) = geojson_feature_geometry(feature) else {
                continue;
            };
            let attributes = geojson_attrs_map(feature.properties.as_ref(), &selected_set);
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

    /// Like [`Self::load_point_batches_from_bytes`], but restricted to the
    /// `[offset, offset+limit)` slice of features — GeoJSON has no native
    /// byte offsets, so this still re-parses the whole file each call.
    pub fn load_point_batches_range_from_bytes(
        bytes: &[u8],
        offset: u64,
        limit: u64,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let fc = geojson_parse_collection(bytes)?;
        const BATCH_SIZE: usize = 10_000;
        let selected = selected_fields.unwrap_or_default();
        let start = (offset as usize).min(fc.features.len());
        let end = ((offset + limit) as usize).min(fc.features.len());
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = geojson_make_batch_cols(&fc, &selected, BATCH_SIZE);
        let mut id_counter = offset as u32;
        for feature in &fc.features[start..end] {
            let Some(Geometry::Point(p)) = geojson_feature_geometry(feature) else {
                continue;
            };
            geojson_push_attrs(&mut batch_cols, feature.properties.as_ref());
            batch.push((id_counter, [p.x(), p.y()]));
            id_counter += 1;
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, geojson_make_batch_cols(&fc, &selected, BATCH_SIZE)),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    /// Like [`Self::load_vector_batches_from_bytes`], but restricted to the
    /// `[offset, offset+limit)` slice of features.
    pub fn load_vector_batches_range_from_bytes(
        bytes: &[u8],
        offset: u64,
        limit: u64,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let fc = geojson_parse_collection(bytes)?;
        const BATCH_SIZE: usize = 10_000;
        let selected_set: std::collections::HashSet<String> = selected_fields
            .map(|f| f.into_iter().collect())
            .unwrap_or_default();
        let start = (offset as usize).min(fc.features.len());
        let end = ((offset + limit) as usize).min(fc.features.len());
        let mut batch: Vec<GisFeature> = Vec::with_capacity(BATCH_SIZE);
        let mut count = offset as usize;
        for feature in &fc.features[start..end] {
            let Some(geo) = geojson_feature_geometry(feature) else {
                continue;
            };
            let attributes = geojson_attrs_map(feature.properties.as_ref(), &selected_set);
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
            VectorFileType::GeoJson => {
                let bytes = std::fs::read(path)?;
                GeoJsonReader::load_descriptor_from_bytes(
                    &bytes,
                    path,
                    GisFilePath::LocalFile(path.to_string()),
                )
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
        let crs = header.crs().and_then(|c| {
            if let Some(org) = c.org() {
                if c.code() != 0 {
                    return Some(format!("{org}:{}", c.code()));
                }
            }
            c.name()
                .map(|s| s.to_string())
                .or_else(|| c.wkt().map(|w| w.to_string()))
        });
        Ok(LayerDescriptor {
            name: header.name().unwrap_or("N/A").to_string(),
            num_features: header.features_count(),
            field_names: header
                .columns()
                .map(|cols| cols.iter().map(|c| c.name().to_string()).collect())
                .unwrap_or_default(),
            geometry_type: header.geometry_type(),
            location: path,
            crs_epsg: crs.as_deref().and_then(parse_epsg_from_label),
            crs,
        })
    }

    // ── read_file ────────────────────────────────────────────────────────────
    //
    // Single streaming entry point: dispatches on file extension (fgb/parquet)
    // and `op` (full scan vs bbox-filtered), and — for FlatGeobuf — on whether
    // the layer holds points or arbitrary vector geometry. Descriptor loading
    // (`load_layer_descriptor`) and the featureless-selection path
    // (`load_selected_without_features`) stay separate: they return different
    // types than the `BatchMessage`-over-a-channel streaming ops below.

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_file(
        path: GisFilePath,
        dest_idx: usize,
        is_points: bool,
        op: ReadOp,
        selected_fields: Option<Vec<String>>,
        tx: mpsc::SyncSender<BatchMessage>,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        let GisFilePath::LocalFile(str_path) = &path else {
            bail!("Wrong GisFilePath type!");
        };
        match (vector_file_type(str_path)?, op) {
            (VectorFileType::FlatGeobuf, ReadOp::Full) => {
                let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
                if is_points {
                    Self::load_point_layer_batched_impl(reader, dest_idx, tx, selected_fields)
                } else {
                    Self::load_layer_batched_impl(reader, dest_idx, tx, selected_fields)
                }
            }
            (VectorFileType::FlatGeobuf, ReadOp::Bbox(bbox)) => {
                if !is_points {
                    bail!("bbox streaming is only supported for point layers");
                }
                Self::stream_fgb_bbox_impl(str_path, bbox, dest_idx, tx, selected_fields, cancel)
            }
            (VectorFileType::FlatGeobuf, ReadOp::Range { offset, limit }) => {
                let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
                if is_points {
                    Self::load_point_layer_range_impl(
                        reader,
                        offset,
                        limit,
                        dest_idx,
                        tx,
                        selected_fields,
                    )
                } else {
                    Self::load_layer_range_impl(reader, offset, limit, dest_idx, tx, selected_fields)
                }
            }
            (VectorFileType::GeoParquet, ReadOp::Full) => {
                if is_points {
                    GeoParquetReader::load_point_layer_batched(str_path, dest_idx, tx, selected_fields)
                } else {
                    GeoParquetReader::load_vector_layer_batched(str_path, dest_idx, tx, selected_fields)
                }
            }
            (VectorFileType::GeoParquet, ReadOp::Bbox(bbox)) => {
                if !is_points {
                    bail!("bbox streaming is only supported for point layers");
                }
                GeoParquetReader::stream_bbox_sync(str_path, bbox, dest_idx, tx, selected_fields, cancel)
            }
            (VectorFileType::GeoParquet, ReadOp::Range { offset, limit }) => {
                if is_points {
                    GeoParquetReader::load_point_layer_range(
                        str_path,
                        offset,
                        limit,
                        dest_idx,
                        tx,
                        selected_fields,
                    )
                } else {
                    GeoParquetReader::load_vector_layer_range(
                        str_path,
                        offset,
                        limit,
                        dest_idx,
                        tx,
                        selected_fields,
                    )
                }
            }
            // GeoJSON has no spatial index and viewport filtering already happens in the
            // shaders, so Bbox behaves exactly like Full here — just load everything.
            (VectorFileType::GeoJson, ReadOp::Full | ReadOp::Bbox(_)) => {
                let bytes = std::fs::read(str_path)?;
                if is_points {
                    GeoJsonReader::load_point_batches_from_bytes(&bytes, dest_idx, tx, selected_fields)
                } else {
                    GeoJsonReader::load_vector_batches_from_bytes(&bytes, dest_idx, tx, selected_fields)
                }
            }
            (VectorFileType::GeoJson, ReadOp::Range { offset, limit }) => {
                let bytes = std::fs::read(str_path)?;
                if is_points {
                    GeoJsonReader::load_point_batches_range_from_bytes(
                        &bytes,
                        offset,
                        limit,
                        dest_idx,
                        tx,
                        selected_fields,
                    )
                } else {
                    GeoJsonReader::load_vector_batches_range_from_bytes(
                        &bytes,
                        offset,
                        limit,
                        dest_idx,
                        tx,
                        selected_fields,
                    )
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_layer_batched_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
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

    #[cfg(not(target_arch = "wasm32"))]
    fn load_point_layer_batched_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        const BATCH_SIZE: usize = 10_000;
        let col_schema = fgb_column_schema(reader.header(), selected_fields.as_deref());
        let mut iter = reader.select_all()?;
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols(&col_schema, BATCH_SIZE);
        let mut id_counter = 0_u32;
        while let Some(feature) = iter.next()? {
            if let Some(row) = extract_point_row(feature, &mut batch_cols, &mut id_counter) {
                batch.push(row);
            }
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, make_batch_cols(&col_schema, BATCH_SIZE)),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    /// Like [`Self::load_layer_batched_impl`], but skips `offset` features
    /// and stops after collecting `limit` of them.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_layer_range_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        offset: u64,
        limit: u64,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let mut iter = reader.select_all()?;
        let selected_set: std::collections::HashSet<String> = selected_fields
            .map(|f| f.into_iter().collect())
            .unwrap_or_default();
        const BATCH_SIZE: usize = 10_000;
        let mut batch: Vec<GisFeature> = Vec::with_capacity(BATCH_SIZE);
        let mut count = offset as usize;
        let mut skipped = 0u64;
        let mut taken = 0u64;
        while taken < limit {
            let Some(feature) = iter.next()? else { break };
            if skipped < offset {
                skipped += 1;
                continue;
            }
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
            taken += 1;
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

    /// Like [`Self::load_point_layer_batched_impl`], but skips `offset`
    /// features and stops after collecting `limit` of them.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_point_layer_range_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        offset: u64,
        limit: u64,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        const BATCH_SIZE: usize = 10_000;
        let col_schema = fgb_column_schema(reader.header(), selected_fields.as_deref());
        let mut iter = reader.select_all()?;
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols(&col_schema, BATCH_SIZE);
        let mut id_counter = offset as u32;
        let mut skipped = 0u64;
        let mut taken = 0u64;
        while taken < limit {
            let Some(feature) = iter.next()? else { break };
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if let Some(row) = extract_point_row(feature, &mut batch_cols, &mut id_counter) {
                batch.push(row);
                taken += 1;
            }
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, make_batch_cols(&col_schema, BATCH_SIZE)),
                ))
                .ok();
            }
        }
        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn stream_fgb_bbox_impl(
        str_path: &str,
        bbox: [f64; 4],
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel_stream: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;

        if cancel_stream.load(Ordering::Relaxed) {
            return Ok(());
        }
        let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
        const BATCH_SIZE: usize = 50_000;
        let col_schema = fgb_column_schema(reader.header(), selected_fields.as_deref());
        let mut iter = reader.select_bbox(bbox[0], bbox[1], bbox[2], bbox[3])?;
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols(&col_schema, BATCH_SIZE);
        let mut id_counter = 0_u32;
        while let Some(feature) = iter.next()? {
            if cancel_stream.load(Ordering::Relaxed) {
                return Ok(());
            }
            if let Some(row) = extract_point_row(feature, &mut batch_cols, &mut id_counter) {
                batch.push(row);
            }
            if batch.len() >= BATCH_SIZE {
                tx.send(BatchMessage::Points(
                    dest_idx,
                    std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                    std::mem::replace(&mut batch_cols, make_batch_cols(&col_schema, BATCH_SIZE)),
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
    pub async fn read_file(
        path: GisFilePath,
        dest_idx: usize,
        is_points: bool,
        op: ReadOp,
        selected_fields: Option<Vec<String>>,
        tx: mpsc::SyncSender<BatchMessage>,
        cancel: Arc<AtomicBool>,
        reader_cache: FgbReaderCache,
    ) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;

        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        match &path {
            // In-memory bytes (parquet or geojson): no bbox filtering exists for
            // these paths yet, so `op` is ignored and the whole file is loaded.
            GisFilePath::Bytes(bytes, name) => match (vector_file_type(name)?, is_points) {
                (VectorFileType::GeoParquet, true) => {
                    GeoParquetReader::load_point_layer_batched_from_bytes(
                        bytes.clone(),
                        dest_idx,
                        tx,
                        selected_fields,
                    )
                }
                (VectorFileType::GeoParquet, false) => Ok(()),
                (VectorFileType::GeoJson, true) => {
                    GeoJsonReader::load_point_batches_from_bytes(bytes, dest_idx, tx, selected_fields)
                }
                (VectorFileType::GeoJson, false) => {
                    GeoJsonReader::load_vector_batches_from_bytes(bytes, dest_idx, tx, selected_fields)
                }
                (VectorFileType::FlatGeobuf, _) => {
                    bail!("FlatGeobuf bytes not supported; use HttpLocation")
                }
            },
            GisFilePath::HttpLocation(url) => match op {
                ReadOp::Bbox(bbox) if is_points => {
                    Self::stream_fgb_bbox_http_impl(
                        url,
                        bbox,
                        dest_idx,
                        tx,
                        selected_fields,
                        cancel,
                        reader_cache,
                    )
                    .await
                }
                // Full-scan / non-point loading over HTTP isn't implemented —
                // the web app always streams by viewport bbox.
                _ => Ok(()),
            },
            GisFilePath::LocalFile(_) => bail!("Wrong GisFilePath type for wasm read_file"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    async fn stream_fgb_bbox_http_impl(
        url: &str,
        bbox: [f64; 4],
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
        cancel_stream: Arc<AtomicBool>,
        reader_cache: FgbReaderCache,
    ) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;

        use flatgeobuf::HttpFgbReader;
        use wasm_bindgen::JsValue;

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
        let col_schema = fgb_column_schema(header, selected_fields.as_deref());
        let mut batch: Vec<(u32, [f64; 2])> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols(&col_schema, BATCH_SIZE);
        let mut id_counter = 0_u32;
        let mut features = reader
            .select_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
            .await?;
        web_sys::console::log_1(&JsValue::from_str("After Feature BBOX query"));
        let mut last_yield_ms = js_sys::Date::now();
        while let Some(feature) = features.next().await? {
            if let Some(row) = extract_point_row(feature, &mut batch_cols, &mut id_counter) {
                batch.push(row);
            }
            // Flush and yield to browser every ~16ms regardless of batch size,
            // avoiding setTimeout throttling that caps throughput at 1 batch/sec.
            let now_ms = js_sys::Date::now();
            if now_ms - last_yield_ms >= 16.0 {
                if !batch.is_empty() {
                    tx.send(BatchMessage::Points(
                        dest_idx,
                        std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE)),
                        std::mem::replace(&mut batch_cols, make_batch_cols(&col_schema, BATCH_SIZE)),
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
        if !cancel_stream.load(Ordering::Relaxed) {
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
            VectorFileType::GeoJson => {
                let owned_path = str_path.to_string();
                let bytes = std::fs::read(&owned_path)?;
                GeoJsonReader::load_descriptor_from_bytes(&bytes, &owned_path, path)?
            }
        };
        Ok(vec![Self::layer_entry_from_descriptor(
            descriptor,
            field_names,
        )])
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_selected_without_features(
        _path: GisFilePath,
        descriptor: LayerDescriptor,
        field_names: Option<Vec<String>>,
    ) -> Result<Vec<LayerEntry>, anyhow::Error> {
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
        let layer_kind = match descriptor.geometry_type.0 {
            1 => LayerKind::Points(PointCloudLayer {
                points: std::sync::Arc::new(Vec::new()),
                attributes: Vec::new(),
                field_names: field_names.unwrap_or(Vec::new()),
                index: None,
                bbox: None,
                // Empty, matching the empty `points` above — pre-sizing
                // these to the full file's feature count here (as before)
                // desyncs them from `points` during progressive/batch
                // loading: `filter_mask` would already cover rows that
                // haven't streamed in yet, so its `count_ones()` (the
                // "filtered" count shown in the sidebar) stops tracking
                // newly-loaded points at all. Both masks now grow in
                // lockstep with `points` as batches merge in (see
                // `apply_batch_message`/`flush_batch_staging`).
                viewport_mask: bitvec![0; 0],
                filter_mask: bitvec![1; 0],
            }),
            _ => LayerKind::Vector(GisLayer {
                name: descriptor.name.clone(),
                file_path: descriptor.location.to_string(),
                features: Vec::new(),
                field_names: field_names.unwrap_or(Vec::new()),
                extra_field_names: Vec::new(),
                quadtree: None,
                point_only: true,
                world_bbox: [0., 0., 0., 0.],
                filter_mask: bitvec![1; 0],
            }),
        };
        LayerEntry {
            data: layer_kind,
            visible: true,
            show_points: true,
            name: descriptor.name.clone(),
            color: [0, 0, 255],
            color_by: None,
            opacity: 255,
            descriptor: descriptor.clone(),
            filters: Vec::new(),
            filter_logic: FilterLogic::default(),
            roi_bboxes: Vec::new(),
            selections: Vec::new(),
            active_selection: None,
            crs_transform: None,
            show_index: false,
            index_kind: crate::spatial_index::IndexKind::Quadtree,
            show_heatmap: false,
            heatmap_metric: crate::heatmap::HeatmapMetric::Density,
            heatmap_cache: None,
            heatmap_dirty: true,
            show_kde: false,
            kde_cache: None,
            saved_heatmaps: Vec::new(),
            active_saved_heatmap: None,
            show_gridbin: false,
            gridbin_cache: None,
            gridbin_metric: crate::heatmap::HeatmapMetric::Density,
            show_bivariate_grid: false,
            bivariate_grid_cache: None,
            saved_bivariate_grids: Vec::new(),
            active_saved_bivariate_grid: None,
            batch_load: None,
        }
    }
}

#[cfg(test)]
mod geojson_tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.5, 2.5]},
                "properties": {"name": "alpha", "count": 3, "score": 1.25}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                "properties": {"name": "beta", "count": 7, "score": 9.5}
            }
        ]
    }"#;

    const MIXED_SAMPLE: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {"name": "point-feature"}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]
                },
                "properties": {"name": "polygon-feature"}
            }
        ]
    }"#;

    #[test]
    fn descriptor_detects_points_and_fields() {
        let desc = GeoJsonReader::load_descriptor_from_bytes(
            SAMPLE.as_bytes(),
            "sample.geojson",
            GisFilePath::LocalFile("sample.geojson".to_string()),
        )
        .unwrap();
        assert_eq!(desc.num_features, 2);
        assert_eq!(desc.geometry_type.0, 1);
        let mut fields = desc.field_names.clone();
        fields.sort();
        assert_eq!(fields, vec!["count", "name", "score"]);
        assert_eq!(desc.crs.as_deref(), Some("EPSG:4326 (default, per GeoJSON spec)"));
    }

    #[test]
    fn descriptor_reads_legacy_crs_member() {
        const WITH_CRS: &str = r#"{
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {}}
            ]
        }"#;
        let desc = GeoJsonReader::load_descriptor_from_bytes(
            WITH_CRS.as_bytes(),
            "with_crs.geojson",
            GisFilePath::LocalFile("with_crs.geojson".to_string()),
        )
        .unwrap();
        assert_eq!(desc.crs.as_deref(), Some("EPSG:3857"));
    }

    #[test]
    fn descriptor_detects_mixed_geometry_as_vector() {
        let desc = GeoJsonReader::load_descriptor_from_bytes(
            MIXED_SAMPLE.as_bytes(),
            "mixed.geojson",
            GisFilePath::LocalFile("mixed.geojson".to_string()),
        )
        .unwrap();
        assert_eq!(desc.geometry_type.0, 0);
    }

    #[test]
    fn point_batches_carry_typed_attributes() {
        let (tx, rx) = mpsc::sync_channel(10);
        GeoJsonReader::load_point_batches_from_bytes(
            SAMPLE.as_bytes(),
            0,
            tx,
            Some(vec!["name".to_string(), "count".to_string(), "score".to_string()]),
        )
        .unwrap();
        let BatchMessage::Points(dest_idx, points, cols) = rx.recv().unwrap() else {
            panic!("expected Points batch");
        };
        assert_eq!(dest_idx, 0);
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].1, [1.5, 2.5]);
        let count_col = cols.iter().find(|(n, _)| n == "count").unwrap();
        assert!(matches!(count_col.1, AttributeColumn::Integer(_)));
        let score_col = cols.iter().find(|(n, _)| n == "score").unwrap();
        assert!(matches!(score_col.1, AttributeColumn::Float(_)));
        let name_col = cols.iter().find(|(n, _)| n == "name").unwrap();
        if let AttributeColumn::Text(v) = &name_col.1 {
            assert_eq!(v, &vec!["alpha".to_string(), "beta".to_string()]);
        } else {
            panic!("expected Text column");
        }
    }

    #[test]
    fn vector_batches_skip_points_and_keep_polygons() {
        let (tx, rx) = mpsc::sync_channel(10);
        GeoJsonReader::load_vector_batches_from_bytes(
            MIXED_SAMPLE.as_bytes(),
            0,
            tx,
            Some(vec!["name".to_string()]),
        )
        .unwrap();
        let BatchMessage::Vector(_, features) = rx.recv().unwrap() else {
            panic!("expected Vector batch");
        };
        assert_eq!(features.len(), 2);
        assert!(matches!(features[0].geometry, Geometry::Point(_)));
        assert!(matches!(features[1].geometry, Geometry::Polygon(_)));
    }

    #[test]
    fn geojson_range_covers_file_without_overlap_or_gaps() {
        let bytes = std::fs::read("assets/sample_points.geojson").unwrap();
        let total = geojson_parse_collection(&bytes).unwrap().features.len() as u64;
        assert_eq!(total, 5);

        let batch_size = 2u64;
        let mut seen_ids: Vec<u32> = Vec::new();
        let mut offset = 0u64;
        while offset < total {
            let limit = batch_size.min(total - offset);
            let (tx, rx) = mpsc::sync_channel(10);
            GeoJsonReader::load_point_batches_range_from_bytes(
                &bytes, offset, limit, 0, tx, None,
            )
            .unwrap();
            for msg in rx.try_iter() {
                let BatchMessage::Points(_, pts, _) = msg else { panic!("expected Points batch") };
                seen_ids.extend(pts.iter().map(|(id, _)| *id));
            }
            offset += limit;
        }
        seen_ids.sort();
        assert_eq!(seen_ids, (0..total as u32).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod range_load_native_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn collect_point_ids(path: &str, op: ReadOp) -> Vec<u32> {
        let (tx, rx) = mpsc::sync_channel(10_000);
        GisReader::read_file(
            GisFilePath::LocalFile(path.to_string()),
            0,
            true,
            op,
            None,
            tx,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let mut ids = Vec::new();
        for msg in rx.try_iter() {
            if let BatchMessage::Points(_, pts, _) = msg {
                ids.extend(pts.iter().map(|(id, _)| *id));
            }
        }
        ids
    }

    #[test]
    fn fgb_range_batches_cover_file_without_overlap() {
        let path = "assets/pickup_points_smol.fgb";
        let full = collect_point_ids(path, ReadOp::Full);
        let total = full.len() as u64;
        assert!(total > 0);

        let batch_size = (total / 7).max(1);
        let mut seen: Vec<u32> = Vec::new();
        let mut offset = 0u64;
        while offset < total {
            let limit = batch_size.min(total - offset);
            seen.extend(collect_point_ids(path, ReadOp::Range { offset, limit }));
            offset += limit;
        }
        seen.sort();
        let mut expected = full;
        expected.sort();
        assert_eq!(seen, expected);
    }

    #[test]
    fn parquet_range_batches_cover_file_without_overlap() {
        let path = "assets/pickup_points.parquet";
        let first_1000 = collect_point_ids(path, ReadOp::Range { offset: 0, limit: 1_000 });
        assert_eq!(first_1000.len(), 1_000);
        let next_1000 = collect_point_ids(path, ReadOp::Range { offset: 1_000, limit: 1_000 });
        assert_eq!(next_1000.len(), 1_000);
        let mut combined: Vec<u32> = first_1000.iter().chain(next_1000.iter()).copied().collect();
        combined.sort();
        combined.dedup();
        assert_eq!(combined.len(), 2_000, "ranges must not overlap");
    }
}

#[cfg(test)]
mod range_perf_probe {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    #[test]
    fn parquet_range_late_offset_is_fast() {
        let path = "assets/pickup_points.parquet";
        let (tx, rx) = mpsc::sync_channel(10_000);
        let start = Instant::now();
        GisReader::read_file(
            GisFilePath::LocalFile(path.to_string()),
            0,
            true,
            ReadOp::Range { offset: 10_000_000, limit: 100_000 },
            None,
            tx,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let elapsed = start.elapsed();
        let mut n = 0usize;
        for msg in rx.try_iter() {
            if let BatchMessage::Points(_, pts, _) = msg {
                n += pts.len();
            }
        }
        eprintln!("late-offset batch: {n} points in {elapsed:?}");
        assert!(elapsed.as_secs_f64() < 1.0, "late-offset batch took {elapsed:?}, expected sub-second seek");
    }
}

#[cfg(test)]
mod range_batch_count_probe {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    #[test]
    fn parquet_range_sends_few_batch_messages() {
        let path = "assets/pickup_points.parquet";
        let (tx, rx) = mpsc::sync_channel(10_000);
        let start = Instant::now();
        GisReader::read_file(
            GisFilePath::LocalFile(path.to_string()),
            0,
            true,
            ReadOp::Range { offset: 0, limit: 100_000 },
            None,
            tx,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let elapsed = start.elapsed();
        let mut msg_count = 0usize;
        let mut total = 0usize;
        for msg in rx.try_iter() {
            if let BatchMessage::Points(_, pts, _) = msg {
                msg_count += 1;
                total += pts.len();
            }
        }
        eprintln!("100k batch: {msg_count} messages, {total} points, {elapsed:?}");
        assert_eq!(total, 100_000);
        assert!(msg_count <= 3, "expected batch to collapse into a few messages, got {msg_count}");
    }
}
