use std::sync::Arc;

use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::gis_layer::{AttributeValue, GisFeature, GisLayer};
use crate::point_cloud_layer::{AttributeColumn, PointCloudLayer};

/// Builds the geometry + attribute + `idx` record batch shared by every
/// export path (native file, wasm in-memory download). `idx` is the point's
/// stored id, not its row position — reload/filter logic (gis_reader.rs,
/// ui_sidebar.rs) matches points by this id, so writing loop position
/// instead would silently break filters on re-loaded exports.
fn build_export_batch(
    pc: &PointCloudLayer,
    ids: &[usize],
) -> anyhow::Result<(Arc<Schema>, RecordBatch)> {
    // WKB point: byte order (1) + type (1=Point, 4 bytes LE) + x (f64 LE) + y (f64 LE)
    let geom_data: Vec<Vec<u8>> = ids
        .iter()
        .map(|&i| {
            let (_, [x, y]) = pc.points[i];
            let mut wkb = Vec::with_capacity(21);
            wkb.push(1u8); // little-endian
            wkb.extend_from_slice(&1u32.to_le_bytes()); // WKBPoint
            wkb.extend_from_slice(&x.to_le_bytes());
            wkb.extend_from_slice(&y.to_le_bytes());
            wkb
        })
        .collect();
    let geom_refs: Vec<&[u8]> = geom_data.iter().map(|v| v.as_slice()).collect();

    let idx_vals: UInt32Array = ids.iter().map(|&i| pc.points[i].0).collect();

    let mut fields: Vec<Field> = vec![
        Field::new("geometry", DataType::Binary, false),
        Field::new("idx", DataType::UInt32, false),
    ];
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(BinaryArray::from_iter_values(geom_refs)),
        Arc::new(idx_vals),
    ];

    for (name, col) in pc.field_names.iter().zip(pc.attributes.iter()) {
        match col {
            AttributeColumn::Float(v) => {
                let vals: Float64Array = ids.iter().map(|&i| v[i]).collect();
                fields.push(Field::new(name, DataType::Float64, true));
                arrays.push(Arc::new(vals));
            }
            AttributeColumn::Integer(v) => {
                let vals: Int64Array = ids.iter().map(|&i| v[i]).collect();
                fields.push(Field::new(name, DataType::Int64, true));
                arrays.push(Arc::new(vals));
            }
            AttributeColumn::Text(v) => {
                let vals = StringArray::from_iter_values(ids.iter().map(|&i| v[i].as_str()));
                fields.push(Field::new(name, DataType::Utf8, true));
                arrays.push(Arc::new(vals));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    Ok((schema, batch))
}

fn filtered_ids(pc: &PointCloudLayer) -> Vec<usize> {
    pc.points
        .iter()
        .enumerate()
        .filter(|(i, _)| pc.filter_mask[*i])
        .map(|(i, _)| i)
        .collect()
}

/// Serializes points (by row index) to parquet bytes in memory. Shared by
/// the native file writer and the wasm browser-download path.
pub fn export_points_by_ids_bytes(pc: &PointCloudLayer, ids: &[usize]) -> anyhow::Result<Vec<u8>> {
    let (schema, batch) = build_export_batch(pc, ids)?;
    let mut buf: Vec<u8> = Vec::new();
    let mut writer = parquet::arrow::ArrowWriter::try_new(&mut buf, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(buf)
}

pub fn export_filtered_points_bytes(pc: &PointCloudLayer) -> anyhow::Result<Vec<u8>> {
    export_points_by_ids_bytes(pc, &filtered_ids(pc))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn export_filtered_points(pc: &PointCloudLayer, path: &str) -> anyhow::Result<()> {
    export_points_by_ids(pc, &filtered_ids(pc), path)
}

/// Exports an explicit subset of point row indices (e.g. a saved
/// box-selection), bypassing `filter_mask` entirely.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_points_by_ids(pc: &PointCloudLayer, ids: &[usize], path: &str) -> anyhow::Result<()> {
    let bytes = export_points_by_ids_bytes(pc, ids)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn feature_to_geojson(f: &GisFeature) -> geojson::Feature {
    let mut properties = geojson::JsonObject::new();
    for (k, v) in &f.attributes {
        let val = match v {
            AttributeValue::Text(s) => serde_json::Value::String(s.clone()),
            AttributeValue::Integer(i) => serde_json::json!(*i),
            AttributeValue::Float(x) => serde_json::json!(*x),
        };
        properties.insert(k.clone(), val);
    }
    geojson::Feature {
        bbox: None,
        geometry: Some(geojson::Geometry::new(geojson::Value::from(&f.geometry))),
        id: None,
        properties: Some(properties),
        foreign_members: None,
    }
}

/// Serializes every feature in a vector layer to GeoJSON bytes — no filter
/// concept exists for vector layers today (unlike Points), so this is
/// always the whole layer.
pub fn export_vector_geojson_bytes(gl: &GisLayer) -> anyhow::Result<Vec<u8>> {
    let fc = geojson::FeatureCollection {
        bbox: None,
        features: gl.features.iter().map(feature_to_geojson).collect(),
        foreign_members: None,
    };
    Ok(geojson::GeoJson::from(fc).to_string().into_bytes())
}
