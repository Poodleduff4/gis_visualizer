use std::time::Instant;

use datafusion::arrow::array::RecordBatch;
use datafusion::prelude::*;

async fn query_parquet(
    path: &str,
    sql: &str,
    select_mode: ParquetSelectMode,
) -> datafusion::error::Result<Vec<RecordBatch>> {
    let config = SessionConfig::new().with_target_partitions(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("layer", path, ParquetReadOptions::default())
        .await?;
    let df = ctx.sql(sql).await?;
    let batches = df.collect().await?;
    println!("{:?}", batches[0].schema());
    // let output = match select_mode {
    // ParquetSelectMode::BitMask => ,
    // ParquetSelectMode::Columns(items) => todo!(),
    // }
    Ok(batches)
}

pub enum ParquetSelectMode {
    BitMask,
    Columns(Vec<String>),
}

fn main() {
    let start_time = Instant::now();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let batches = rt
        .block_on(query_parquet(
            "./assets/pickup_points.parquet",
            "SELECT fare_amount FROM layer WHERE fare_amount > 10.0",
            ParquetSelectMode::BitMask,
        ))
        .unwrap();

    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "# features: {}\nquery time: {:.3}s",
        row_count,
        start_time.elapsed().as_secs_f32()
    );
}
