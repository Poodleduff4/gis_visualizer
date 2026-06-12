use std::time::Instant;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array};
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
