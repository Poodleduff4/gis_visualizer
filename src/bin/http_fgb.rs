use flatgeobuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fetch_fgb("localhost:8000/pickup_points.fgb");
    Ok(())
}

async fn fetch_fgb(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let fgb = flatgeobuf::HttpFgbReader::open(url)
        .await?
        .select_all()
        .await?;
    while let Some(feature) = fgb.next().await? {
        let props = feature.properties()?;
        println!("{}", props["name"]);
        println!("{}", feature.to_wkt()?);
    }
    Ok(())
}
