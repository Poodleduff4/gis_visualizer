#[cfg(not(target_arch = "wasm32"))]
pub use desktop::{export_filtered_points, export_points_by_ids};

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use std::fs::File;
    use std::sync::Arc;

    use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use crate::point_cloud_layer::{AttributeColumn, PointCloudLayer};

    pub fn export_filtered_points(pc: &PointCloudLayer, path: &str) -> anyhow::Result<()> {
        let filtered: Vec<usize> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(i, _)| i)
            .collect();
        export_points_by_ids(pc, &filtered, path)
    }

    /// Exports an explicit subset of point row indices (e.g. a saved
    /// box-selection), bypassing `filter_mask` entirely.
    pub fn export_points_by_ids(
        pc: &PointCloudLayer,
        ids: &[usize],
        path: &str,
    ) -> anyhow::Result<()> {
        let filtered: Vec<usize> = ids.to_vec();

        // WKB point: byte order (1) + type (1=Point, 4 bytes LE) + x (f64 LE) + y (f64 LE)
        let geom_data: Vec<Vec<u8>> = filtered
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

        let mut fields: Vec<Field> = vec![Field::new("geometry", DataType::Binary, false)];
        let mut arrays: Vec<Arc<dyn arrow::array::Array>> =
            vec![Arc::new(BinaryArray::from_iter_values(geom_refs))];

        for (name, col) in pc.field_names.iter().zip(pc.attributes.iter()) {
            match col {
                AttributeColumn::Float(v) => {
                    let vals: Float64Array = filtered.iter().map(|&i| v[i]).collect();
                    fields.push(Field::new(name, DataType::Float64, true));
                    arrays.push(Arc::new(vals));
                }
                AttributeColumn::Integer(v) => {
                    let vals: Int64Array = filtered.iter().map(|&i| v[i]).collect();
                    fields.push(Field::new(name, DataType::Int64, true));
                    arrays.push(Arc::new(vals));
                }
                AttributeColumn::Text(v) => {
                    let vals = StringArray::from_iter_values(filtered.iter().map(|&i| v[i].as_str()));
                    fields.push(Field::new(name, DataType::Utf8, true));
                    arrays.push(Arc::new(vals));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays)?;
        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }
}
