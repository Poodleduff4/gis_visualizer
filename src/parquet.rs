use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::Instant;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::{
    Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array,
};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::*;
use futures_channel::oneshot;

pub async fn query_parquet(path: &str, sql: String) -> datafusion::error::Result<Vec<RecordBatch>> {
    let config = SessionConfig::new().with_target_partitions(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("layer", path, ParquetReadOptions::default())
        .await?;
    let df = ctx.sql(sql.as_str()).await?;
    df.collect().await
}

/// Returns a parquet path guaranteed to have a physical `idx: UInt32` column
/// (0-based row position), so filter queries can `SELECT "idx"` directly
/// instead of deriving one with `ROW_NUMBER()` — which needs a single-
/// partition scan to stay correctly ordered and serializes the whole file on
/// every filter. Files this app itself exported already have `idx`
/// (`exporter.rs`) and are returned unchanged; anything else (a foreign/
/// hand-built GeoParquet) gets a one-time rewrite into a cache file next to
/// the temp dir, reused on subsequent filters as long as the source's mtime
/// and size haven't changed.
pub async fn ensure_idx_column(path: &str) -> anyhow::Result<PathBuf> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    mtime.map(|d| d.as_nanos()).hash(&mut hasher);
    let cache_path = std::env::temp_dir()
        .join("gis_editor_idx_cache")
        .join(format!("{:016x}.parquet", hasher.finish()));

    if cache_path.exists() {
        return Ok(cache_path);
    }

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("layer", path, ParquetReadOptions::default())
        .await?;
    let schema = ctx.table("layer").await?.schema().clone();
    if schema.field_with_unqualified_name("idx").is_ok() {
        return Ok(PathBuf::from(path));
    }

    std::fs::create_dir_all(cache_path.parent().unwrap())?;
    // Single partition here isn't a perf concern (this happens once and is
    // cached) but is what keeps `ROW_NUMBER()` aligned with the file's
    // physical row order — the same order the app's own sequential loader
    // (`ParquetRecordBatchReaderBuilder` in gis_reader.rs) assigns as each
    // point's `parquet_id` when it first reads the file.
    let df = ctx
        .sql("SELECT *, CAST(ROW_NUMBER() OVER () - 1 AS INT UNSIGNED) AS idx FROM layer")
        .await?;
    df.write_parquet(
        cache_path.to_str().expect("temp dir path is valid UTF-8"),
        DataFrameWriteOptions::new(),
        None,
    )
    .await?;
    Ok(cache_path)
}

pub fn extract_batch_as_u32(batch: &RecordBatch, col: &str) -> Option<Vec<u32>> {
    let arr = batch.column_by_name(col)?;
    if let Some(a) = arr.as_any().downcast_ref::<UInt32Array>() {
        return Some(a.values().to_vec());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(a.values().iter().map(|v| *v as u32).collect());
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(a.values().iter().map(|v| *v as u32).collect());
    }
    // `ROW_NUMBER() OVER (...)` (used by the filter query's generated row
    // index) is typed UInt64 by DataFusion.
    if let Some(a) = arr.as_any().downcast_ref::<UInt64Array>() {
        return Some(a.values().iter().map(|v| *v as u32).collect());
    }
    None
}
pub fn extract_u32(batch: &RecordBatch, col: &str, row: usize) -> Option<u32> {
    let arr = batch.column_by_name(col)?;
    if let Some(a) = arr.as_any().downcast_ref::<UInt32Array>() {
        return Some(a.value(row));
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(a.value(row) as u32);
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(a.value(row) as u32);
    }
    None
}

fn extract_f64(batch: &RecordBatch, col: &str, row: usize) -> Option<f64> {
    let arr = batch.column_by_name(col)?;
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return Some(a.value(row));
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        return Some(a.value(row) as f64);
    }
    None
}

// pub fn load_point_layer_batched(
//     path: &str,
//     dest_idx: usize,
//     tx: std::sync::mpsc::SyncSender<crate::gis_layer::BatchMessage>,
// ) -> anyhow::Result<()> {
//     let rt = tokio::runtime::Runtime::new()?;
//     let batches = rt.block_on(query_parquet(path, "SELECT idx, x, y FROM layer"))?;
//     let mut id_counter = 0_u32;
//     for batch in batches {
//         let points: Vec<(u32, [f64; 2])> = (0..batch.num_rows())
//             .filter_map(|i| {
//                 let x = extract_f64(&batch, "x", i)?;
//                 let y = extract_f64(&batch, "y", i)?;
//                 let id = extract_u32(&batch, "idx", i).unwrap_or_else(|| {
//                     let c = id_counter;
//                     id_counter += 1;
//                     c
//                 });
//                 Some((id, [x, y]))
//             })
//             .collect();
//         if !points.is_empty() {
//             tx.send(crate::gis_layer::BatchMessage::Points(
//                 dest_idx,
//                 points,
//                 vec![],
//             ))?;
//         }
//     }
//     Ok(())
// }

// fn main() {
//     let start_time = Instant::now();

//     let rt = tokio::runtime::Runtime::new().unwrap();
//     let batches = rt
//         .block_on(query_parquet(
//             "./assets/pickup_points.parquet",
//             "SELECT fare_amount FROM layer WHERE fare_amount > 10.0",
//         ))
//         .unwrap();

//     let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
//     println!(
//         "# features: {}\nquery time: {:.3}s",
//         row_count,
//         start_time.elapsed().as_secs_f32()
//     );
// }

#[cfg(test)]
mod idx_tests {
    use super::*;

    #[tokio::test]
    async fn adds_idx_column_to_foreign_parquet_and_caches_it() {
        use datafusion::arrow::array::Float64Array;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("trip_distance", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![1.0, 5.0, 2.0, 9.0, 0.5]))],
        )
        .unwrap();

        let dir = std::env::temp_dir().join("gis_editor_idx_test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("no_idx.parquet");
        {
            let file = std::fs::File::create(&src).unwrap();
            let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let out_path = ensure_idx_column(src.to_str().unwrap()).await.unwrap();
        assert_ne!(out_path, src, "should have rewritten to a cache path");

        let batches = query_parquet(
            out_path.to_str().unwrap(),
            "SELECT idx, trip_distance FROM layer WHERE trip_distance > 4.0 ORDER BY idx".into(),
        )
        .await
        .unwrap();
        let ids: Vec<u32> = batches.iter().flat_map(|b| extract_batch_as_u32(b, "idx").unwrap()).collect();
        assert_eq!(ids, vec![1, 3], "rows 1 and 3 (0-based) have trip_distance > 4.0");

        // Second call should hit the cache, not rewrite again.
        let out_path_2 = ensure_idx_column(src.to_str().unwrap()).await.unwrap();
        assert_eq!(out_path, out_path_2);
    }
}
