//! Codec between the app's layer types and the Arrow IPC buffers that cross
//! the plugin subprocess boundary.
//!
//! `Vector` layers (`GisLayer`) encode as one `geometry` column of WKB bytes
//! plus one column per attribute field, typed off `AttributeValue`.
//!
//! `Points` layers (`PointCloudLayer`) encode as flat `id`/`x`/`y` columns
//! plus one column per attribute — no WKB, no per-point `GisFeature`
//! overhead, since that per-point cost is already the scaling bottleneck
//! for this layer kind (~430B/point; see the point-cloud-scale project
//! memory). This is read-only: a plugin's output always comes back through
//! `decode_vector_layer` regardless of what kind it read, since the result
//! of an analysis is naturally a new (usually smaller) vector layer, not
//! another multi-million-row point cloud.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bitvec::bitvec;
use geo_types::Geometry;
use arrow::array::{Array, BinaryArray, Float32Array, Float64Array, Int64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use geo::BoundingRect;
use geozero::{wkb::Wkb, CoordDimensions, ToGeo, ToWkb};

use crate::filter::FilterLogic;
use crate::gis_layer::{
    AttributeValue, GisFeature, GisLayer, LayerEntry, LayerKind, RasterBand, RasterData,
    RasterDisplayMode,
};
use crate::gis_reader::{GisFilePath, LayerDescriptor};
use crate::point_cloud_layer::{AttributeColumn, PointCloudLayer};

/// Scans features for the first value present under `field`, used to decide
/// that column's Arrow type. Defaults to text when no feature has the field
/// (an all-null column still needs a concrete type).
fn column_type(features: &[GisFeature], field: &str) -> AttributeValue {
    features
        .iter()
        .find_map(|f| f.attributes.get(field).cloned())
        .unwrap_or(AttributeValue::Text(String::new()))
}

pub fn encode_vector_layer(layer: &GisLayer) -> Result<Vec<u8>> {
    let geom_bytes: Vec<Vec<u8>> = layer
        .features
        .iter()
        .map(|f| {
            f.geometry
                .to_wkb(CoordDimensions::xy())
                .map_err(|e| anyhow!("WKB encode failed for feature {}: {e}", f.id))
        })
        .collect::<Result<_>>()?;
    let geom_refs: Vec<&[u8]> = geom_bytes.iter().map(Vec::as_slice).collect();

    let mut fields = vec![Field::new("geometry", DataType::Binary, false)];
    let mut arrays: Vec<Arc<dyn Array>> = vec![Arc::new(BinaryArray::from_iter_values(geom_refs))];

    for field_name in &layer.field_names {
        match column_type(&layer.features, field_name) {
            AttributeValue::Float(_) => {
                let vals: Float64Array = layer
                    .features
                    .iter()
                    .map(|f| match f.attributes.get(field_name) {
                        Some(AttributeValue::Float(v)) => Some(*v),
                        _ => None,
                    })
                    .collect();
                fields.push(Field::new(field_name, DataType::Float64, true));
                arrays.push(Arc::new(vals));
            }
            AttributeValue::Integer(_) => {
                let vals: Int64Array = layer
                    .features
                    .iter()
                    .map(|f| match f.attributes.get(field_name) {
                        Some(AttributeValue::Integer(v)) => Some(*v),
                        _ => None,
                    })
                    .collect();
                fields.push(Field::new(field_name, DataType::Int64, true));
                arrays.push(Arc::new(vals));
            }
            AttributeValue::Text(_) => {
                let vals: StringArray = layer
                    .features
                    .iter()
                    .map(|f| match f.attributes.get(field_name) {
                        Some(AttributeValue::Text(v)) => Some(v.as_str()),
                        _ => None,
                    })
                    .collect();
                fields.push(Field::new(field_name, DataType::Utf8, true));
                arrays.push(Arc::new(vals));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Encodes a `PointCloudLayer` as flat columns: `id` (the point's stored
/// id, matching `PointCloudLayer.points`'s `u32`), `x`, `y`, then one column
/// per attribute. `PointCloudLayer`'s attribute arrays are already dense —
/// every point has every field — so unlike `encode_vector_layer` there's no
/// null-handling to do here.
pub fn encode_points_layer(layer: &PointCloudLayer) -> Result<Vec<u8>> {
    let mut fields = vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ];
    let mut arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(UInt32Array::from_iter_values(
            layer.points.iter().map(|(id, _)| *id),
        )),
        Arc::new(Float64Array::from_iter_values(
            layer.points.iter().map(|(_, [x, _])| *x),
        )),
        Arc::new(Float64Array::from_iter_values(
            layer.points.iter().map(|(_, [_, y])| *y),
        )),
    ];

    for (name, col) in layer.field_names.iter().zip(layer.attributes.iter()) {
        match col {
            AttributeColumn::Float(v) => {
                fields.push(Field::new(name, DataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from_iter_values(v.iter().copied())));
            }
            AttributeColumn::Integer(v) => {
                fields.push(Field::new(name, DataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from_iter_values(v.iter().copied())));
            }
            AttributeColumn::Text(v) => {
                fields.push(Field::new(name, DataType::Utf8, false));
                arrays.push(Arc::new(StringArray::from_iter_values(v.iter())));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Encodes a `RasterData` as one `Float32` column per band (named after
/// `RasterBand.name`, or `band_N` if empty), row-major with row 0 = north —
/// same convention as the grid itself. There's no `x`/`y`/`geometry` column
/// like the other two encoders: a raster's shape is regular, so `width`,
/// `height`, `units`, and `extent` travel as schema metadata instead of being
/// repeated per row, and a reader reconstructs pixel coordinates from those
/// plus its row/column index.
pub fn encode_raster_layer(raster: &RasterData) -> Result<Vec<u8>> {
    let mut fields = Vec::with_capacity(raster.bands.len());
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(raster.bands.len());
    for (i, band) in raster.bands.iter().enumerate() {
        let col_name = if band.name.is_empty() {
            format!("band_{}", i + 1)
        } else {
            band.name.clone()
        };
        fields.push(Field::new(&col_name, DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from_iter_values(
            band.values.iter().copied(),
        )));
    }

    let metadata = HashMap::from([
        ("width".to_string(), raster.width.to_string()),
        ("height".to_string(), raster.height.to_string()),
        ("units".to_string(), raster.units.clone()),
        (
            "extent".to_string(),
            raster
                .extent
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);

    let schema = Arc::new(Schema::new(fields).with_metadata(metadata));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Decodes an Arrow IPC buffer (as produced by `encode_vector_layer`, or by
/// a plugin's own `AddLayer`/`UpdateLayer` reply) into a fresh `GisLayer`.
/// Only the first record batch is read — plugins are expected to send one
/// layer's worth of data per call, not a chunked stream.
pub fn decode_vector_layer(bytes: &[u8], name: String) -> Result<GisLayer> {
    let mut reader = StreamReader::try_new(bytes, None)?;
    let batch = reader
        .next()
        .context("Arrow IPC buffer contained no record batches")??;

    let geom_col = batch
        .column_by_name("geometry")
        .context("Arrow buffer missing a 'geometry' column")?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .context("'geometry' column isn't Binary-typed")?;

    let field_names: Vec<String> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|n| n != "geometry")
        .collect();

    let mut features = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let geometry = Wkb(geom_col.value(row))
            .to_geo()
            .map_err(|e| anyhow!("WKB decode failed at row {row}: {e}"))?;

        let mut attributes = HashMap::new();
        for field_name in &field_names {
            let col = batch.column_by_name(field_name).unwrap();
            let value = if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                (!a.is_null(row)).then(|| AttributeValue::Float(a.value(row)))
            } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                (!a.is_null(row)).then(|| AttributeValue::Integer(a.value(row)))
            } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                (!a.is_null(row)).then(|| AttributeValue::Text(a.value(row).to_string()))
            } else {
                None
            };
            if let Some(v) = value {
                attributes.insert(field_name.clone(), v);
            }
        }

        features.push(GisFeature::new(row, geometry, attributes));
    }

    let world_bbox = features
        .iter()
        .filter_map(|f| f.geometry.bounding_rect())
        .fold(None, |acc: Option<[f64; 4]>, r| {
            let b = [r.min().x, r.min().y, r.max().x, r.max().y];
            Some(match acc {
                Some(a) => [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ],
                None => b,
            })
        })
        .unwrap_or([0.0, 0.0, 0.0, 0.0]);

    let filter_mask = bitvec::bitvec![1; features.len()];
    Ok(GisLayer {
        name,
        file_path: String::new(),
        features,
        field_names,
        extra_field_names: Vec::new(),
        quadtree: None,
        point_only: false,
        filter_mask,
        world_bbox,
    })
}

/// Decodes an Arrow IPC buffer produced by `encode_raster_layer`'s schema
/// (one `Float32` column per band, `width`/`height`/`units`/`extent` as
/// schema metadata) into a fresh `RasterData`. `data_min`/`data_max` per
/// band are recomputed from the values rather than trusted from the
/// plugin, since nothing round-trips them; `display_min`/`display_max`
/// default to that range so the band renders through the full color ramp
/// immediately, same as a freshly-loaded GeoTIFF.
pub fn decode_raster_layer(bytes: &[u8]) -> Result<RasterData> {
    let mut reader = StreamReader::try_new(bytes, None)?;
    let meta = reader.schema().metadata().clone();
    let width: usize = meta
        .get("width")
        .context("raster Arrow buffer missing 'width' metadata")?
        .parse()
        .context("'width' metadata isn't a valid integer")?;
    let height: usize = meta
        .get("height")
        .context("raster Arrow buffer missing 'height' metadata")?
        .parse()
        .context("'height' metadata isn't a valid integer")?;
    let units = meta.get("units").cloned().unwrap_or_default();
    let extent_parts: Vec<f64> = meta
        .get("extent")
        .context("raster Arrow buffer missing 'extent' metadata")?
        .split(',')
        .map(|s| s.parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .context("'extent' metadata isn't 4 comma-separated numbers")?;
    let extent: [f64; 4] = extent_parts
        .try_into()
        .map_err(|_| anyhow!("'extent' metadata must have exactly 4 values"))?;

    let batch = reader
        .next()
        .context("Arrow IPC buffer contained no record batches")??;

    let mut bands = Vec::with_capacity(batch.num_columns());
    for field in batch.schema().fields() {
        let arr = batch
            .column_by_name(field.name())
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .with_context(|| format!("raster band '{}' isn't Float32-typed", field.name()))?;
        let values: Vec<f32> = arr.values().to_vec();
        if values.len() != width * height {
            return Err(anyhow!(
                "raster band '{}' has {} values, expected {}x{}={}",
                field.name(),
                values.len(),
                width,
                height,
                width * height
            ));
        }

        let (mut data_min, mut data_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &values {
            if !v.is_nan() {
                data_min = data_min.min(v as f64);
                data_max = data_max.max(v as f64);
            }
        }
        if !data_min.is_finite() || !data_max.is_finite() {
            data_min = 0.0;
            data_max = 0.0;
        }

        bands.push(RasterBand {
            name: field.name().clone(),
            values,
            data_min,
            data_max,
            display_min: data_min,
            display_max: data_max,
        });
    }

    Ok(RasterData {
        width,
        height,
        bands,
        units,
        display_mode: RasterDisplayMode::Single(0),
        extent,
    })
}

/// Dispatches an `AddLayer`/`UpdateLayer` Arrow buffer to `decode_vector_layer`
/// or `decode_raster_layer` based on schema shape — a raster buffer carries
/// `width`/`height` schema metadata (see `encode_raster_layer`) that a
/// vector buffer never has, so that's a cheap and unambiguous discriminator.
pub fn decode_layer_from_arrow(bytes: &[u8], name: String) -> Result<LayerKind> {
    let reader = StreamReader::try_new(bytes, None)?;
    if reader.schema().metadata().contains_key("width") {
        decode_raster_layer(bytes).map(LayerKind::Raster)
    } else {
        decode_vector_layer(bytes, name).map(LayerKind::Vector)
    }
}

/// If every feature in `layer` is a bare `Point`, converts it into a
/// `PointCloudLayer` instead of leaving it as a `GisLayer` — so a
/// plugin-added point layer rides the exact same GPU-instanced point-cloud
/// rendering path (`point_cloud.rs`/`gpu_collect.rs`) as a Points layer
/// loaded from a file, rather than the Vector path meant for polygons/lines
/// (which can only ever CPU-paint points, since that path's GPU mesh is
/// triangulated fill/outline geometry with no concept of a constant-pixel-
/// size point sprite). Returns `None` for empty or mixed-geometry layers,
/// which have no equivalent in the points-only `PointCloudLayer` shape.
pub fn vector_layer_to_points(layer: &GisLayer) -> Option<PointCloudLayer> {
    if layer.features.is_empty()
        || !layer
            .features
            .iter()
            .all(|f| matches!(f.geometry, Geometry::Point(_)))
    {
        return None;
    }

    let points: Arc<Vec<(u32, [f64; 2])>> = Arc::new(
        layer
            .features
            .iter()
            .map(|f| {
                let Geometry::Point(p) = &f.geometry else {
                    unreachable!("checked above")
                };
                (f.id as u32, [p.x(), p.y()])
            })
            .collect(),
    );

    let attributes: Vec<AttributeColumn> = layer
        .field_names
        .iter()
        .map(|name| match column_type(&layer.features, name) {
            AttributeValue::Float(_) => AttributeColumn::Float(
                layer
                    .features
                    .iter()
                    .map(|f| match f.attributes.get(name) {
                        Some(AttributeValue::Float(v)) => *v,
                        Some(AttributeValue::Integer(v)) => *v as f64,
                        _ => 0.0,
                    })
                    .collect(),
            ),
            AttributeValue::Integer(_) => AttributeColumn::Integer(
                layer
                    .features
                    .iter()
                    .map(|f| match f.attributes.get(name) {
                        Some(AttributeValue::Integer(v)) => *v,
                        _ => 0,
                    })
                    .collect(),
            ),
            AttributeValue::Text(_) => AttributeColumn::Text(
                layer
                    .features
                    .iter()
                    .map(|f| match f.attributes.get(name) {
                        Some(AttributeValue::Text(v)) => v.clone(),
                        _ => String::new(),
                    })
                    .collect(),
            ),
        })
        .collect();

    let n = layer.features.len();
    Some(PointCloudLayer {
        points,
        attributes,
        field_names: layer.field_names.clone(),
        index: None,
        bbox: None,
        viewport_mask: bitvec![0; n],
        filter_mask: bitvec![1; n],
    })
}

/// Wraps a plugin-decoded layer in the `LayerEntry` scaffolding the rest of
/// the app expects, mirroring `gis_reader::layer_entry_from_descriptor`'s
/// defaults for a freshly-loaded layer. `data` is usually `LayerKind::Vector`
/// straight from `decode_vector_layer`, or `LayerKind::Points` when
/// `vector_layer_to_points` found an all-point result.
pub fn layer_entry_for(name: String, data: LayerKind) -> LayerEntry {
    let (num_features, field_names, geometry_type) = match &data {
        LayerKind::Vector(gl) => (
            gl.features.len() as u64,
            gl.field_names.clone(),
            flatgeobuf::GeometryType(0),
        ),
        LayerKind::Points(pc) => (
            pc.points.len() as u64,
            pc.field_names.clone(),
            flatgeobuf::GeometryType(1),
        ),
        LayerKind::Raster(_) => (0, Vec::new(), flatgeobuf::GeometryType(0)),
    };
    let descriptor = LayerDescriptor {
        name: name.clone(),
        num_features,
        field_names,
        geometry_type,
        location: GisFilePath::LocalFile(String::new()),
        crs: None,
        crs_epsg: None,
    };
    LayerEntry {
        data,
        visible: true,
        show_points: true,
        name,
        color: [0, 128, 255],
        color_by: None,
        opacity: 255,
        descriptor,
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

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry, Point};

    fn sample_layer() -> GisLayer {
        let mut a = HashMap::new();
        a.insert("name".to_string(), AttributeValue::Text("alpha".into()));
        a.insert("count".to_string(), AttributeValue::Integer(3));
        a.insert("score".to_string(), AttributeValue::Float(1.5));

        // Deliberately sparse: this feature omits "score", exercising the
        // null-handling path on both encode and decode.
        let mut b = HashMap::new();
        b.insert("name".to_string(), AttributeValue::Text("beta".into()));
        b.insert("count".to_string(), AttributeValue::Integer(7));

        GisLayer {
            name: "test".into(),
            file_path: String::new(),
            features: vec![
                GisFeature::new(0, Geometry::Point(Point::new(1.0, 2.0)), a),
                GisFeature::new(1, Geometry::Point(Point::new(3.0, 4.0)), b),
            ],
            field_names: vec!["name".into(), "count".into(), "score".into()],
            extra_field_names: Vec::new(),
            quadtree: None,
            point_only: true,
            world_bbox: [1.0, 2.0, 3.0, 4.0],
            filter_mask: bitvec::bitvec![1; 2],
        }
    }

    #[test]
    fn encode_decode_round_trips_geometry_and_attributes() {
        let layer = sample_layer();
        let bytes = encode_vector_layer(&layer).expect("encode");
        let decoded = decode_vector_layer(&bytes, "test".into()).expect("decode");

        assert_eq!(decoded.features.len(), 2);

        match &decoded.features[0].geometry {
            Geometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            other => panic!("expected Point, got {other:?}"),
        }
        assert_eq!(
            decoded.features[0].attributes.get("name"),
            Some(&AttributeValue::Text("alpha".into()))
        );
        assert_eq!(
            decoded.features[0].attributes.get("count"),
            Some(&AttributeValue::Integer(3))
        );
        assert_eq!(
            decoded.features[0].attributes.get("score"),
            Some(&AttributeValue::Float(1.5))
        );

        // The sparse feature's missing field must stay missing, not become
        // a zero/empty default — plugins should see the same shape they'd
        // get by reading `GisFeature.attributes` directly.
        assert!(!decoded.features[1].attributes.contains_key("score"));
        assert_eq!(
            decoded.features[1].attributes.get("name"),
            Some(&AttributeValue::Text("beta".into()))
        );
    }

    #[test]
    fn encodes_raster_layer_as_band_columns_with_metadata() {
        use crate::gis_layer::{RasterBand, RasterDisplayMode};

        let raster = RasterData {
            width: 2,
            height: 2,
            bands: vec![RasterBand {
                name: "elevation".into(),
                values: vec![1.0, 2.0, 3.0, 4.0],
                data_min: 1.0,
                data_max: 4.0,
                display_min: 1.0,
                display_max: 4.0,
            }],
            units: "m".into(),
            display_mode: RasterDisplayMode::Single(0),
            extent: [-10.0, -20.0, 10.0, 20.0],
        };

        let bytes = encode_raster_layer(&raster).expect("encode");
        let mut reader = StreamReader::try_new(bytes.as_slice(), None).expect("reader");
        let batch = reader.next().expect("batch").expect("batch");

        let metadata = batch.schema().metadata().clone();
        assert_eq!(metadata.get("width").map(String::as_str), Some("2"));
        assert_eq!(metadata.get("height").map(String::as_str), Some("2"));
        assert_eq!(metadata.get("units").map(String::as_str), Some("m"));
        assert_eq!(
            metadata.get("extent").map(String::as_str),
            Some("-10,-20,10,20")
        );

        let elevation = batch
            .column_by_name("elevation")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(elevation.value(0), 1.0);
        assert_eq!(elevation.value(3), 4.0);
    }

    #[test]
    fn decode_raster_layer_round_trips_and_recomputes_data_range() {
        use crate::gis_layer::{RasterBand, RasterDisplayMode};

        let raster = RasterData {
            width: 2,
            height: 2,
            bands: vec![RasterBand {
                name: "elevation".into(),
                values: vec![1.0, 2.0, 3.0, 4.0],
                data_min: 999.0, // deliberately wrong — decode must recompute, not trust this
                data_max: 999.0,
                display_min: 1.0,
                display_max: 4.0,
            }],
            units: "m".into(),
            display_mode: RasterDisplayMode::Single(0),
            extent: [-10.0, -20.0, 10.0, 20.0],
        };

        let bytes = encode_raster_layer(&raster).expect("encode");
        let decoded = decode_raster_layer(&bytes).expect("decode");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.units, "m");
        assert_eq!(decoded.extent, [-10.0, -20.0, 10.0, 20.0]);
        assert_eq!(decoded.bands.len(), 1);
        assert_eq!(decoded.bands[0].values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(decoded.bands[0].data_min, 1.0);
        assert_eq!(decoded.bands[0].data_max, 4.0);
        assert_eq!(decoded.bands[0].display_min, 1.0);
        assert_eq!(decoded.bands[0].display_max, 4.0);
    }

    #[test]
    fn decode_layer_from_arrow_dispatches_on_schema_metadata() {
        use crate::gis_layer::{RasterBand, RasterDisplayMode};

        let vector_bytes = encode_vector_layer(&sample_layer()).expect("encode vector");
        match decode_layer_from_arrow(&vector_bytes, "test".into()).expect("decode vector") {
            LayerKind::Vector(gl) => assert_eq!(gl.features.len(), 2),
            _ => panic!("expected Vector"),
        }

        let raster = RasterData {
            width: 1,
            height: 1,
            bands: vec![RasterBand {
                name: "b".into(),
                values: vec![5.0],
                data_min: 5.0,
                data_max: 5.0,
                display_min: 5.0,
                display_max: 5.0,
            }],
            units: String::new(),
            display_mode: RasterDisplayMode::Single(0),
            extent: [0.0, 0.0, 1.0, 1.0],
        };
        let raster_bytes = encode_raster_layer(&raster).expect("encode raster");
        match decode_layer_from_arrow(&raster_bytes, "r".into()).expect("decode raster") {
            LayerKind::Raster(r) => assert_eq!(r.bands[0].values, vec![5.0]),
            _ => panic!("expected Raster"),
        }
    }

    #[test]
    fn encodes_points_layer_as_flat_columns() {
        let layer = PointCloudLayer {
            points: Arc::new(vec![(10, [1.0, 2.0]), (11, [3.0, 4.0])]),
            attributes: vec![
                AttributeColumn::Float(vec![1.5, 2.5]),
                AttributeColumn::Integer(vec![100, 200]),
            ],
            field_names: vec!["elevation".into(), "intensity".into()],
            index: None,
            bbox: None,
            viewport_mask: Default::default(),
            filter_mask: Default::default(),
        };

        let bytes = encode_points_layer(&layer).expect("encode");
        let mut reader = StreamReader::try_new(bytes.as_slice(), None).expect("reader");
        let batch = reader.next().expect("batch").expect("batch");

        assert_eq!(batch.num_rows(), 2);
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .unwrap();
        assert_eq!(ids.value(0), 10);
        assert_eq!(ids.value(1), 11);

        let elevation = batch
            .column_by_name("elevation")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(elevation.value(1), 2.5);

        let intensity = batch
            .column_by_name("intensity")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(intensity.value(0), 100);
    }
}
