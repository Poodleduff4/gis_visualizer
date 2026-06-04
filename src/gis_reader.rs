use anyhow::bail;
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

#[cfg(target_arch = "wasm32")]
use std::{
    io::Cursor,
    sync::{atomic::AtomicBool, Arc},
};

#[cfg(target_arch = "wasm32")]
pub type FgbReaderCache = std::rc::Rc<
    std::cell::RefCell<std::collections::HashMap<String, flatgeobuf::HttpFgbReader>>,
>;

use crate::{
    gis_layer::{AttributeValue, BatchMessage, GisFeature, GisLayer, LayerEntry, LayerKind},
    point_cloud_layer::{AttributeColumn, PointCloudLayer},
};

#[derive(Clone)]
pub enum GisFilePath {
    LocalFile(String),
    HttpLocation(String),
}
impl GisFilePath {
    pub fn to_string(&self) -> String {
        match self {
            GisFilePath::LocalFile(p) => p.clone(),
            GisFilePath::HttpLocation(p) => p.clone(),
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

struct ColumnPusher<'a> {
    cols: &'a mut Vec<(String, AttributeColumn)>,
}

impl PropertyProcessor for ColumnPusher<'_> {
    fn property(
        &mut self,
        _idx: usize,
        name: &str,
        value: &ColumnValue,
    ) -> std::result::Result<bool, GeozeroError> {
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

pub struct GisReader {}

impl GisReader {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_layer_descriptor(path: &str) -> anyhow::Result<LayerDescriptor> {
        let reader = FgbReader::open(BufReader::new(File::open(path)?))?;

        Self::make_layer_descriptor(reader.header(), GisFilePath::LocalFile(path.to_string()))
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
        let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
        Self::load_point_layer_batched_impl(reader, dest_idx, tx, selected_fields)
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
        let mut batch: Vec<[f64; 2]> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        while let Some(feature) = iter.next()? {
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            batch.push([x, y]);
            if !col_schema.is_empty() {
                let mut pusher = ColumnPusher {
                    cols: &mut batch_cols,
                };
                feature.process_properties(&mut pusher).ok();
            }
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
        let mut batch: Vec<[f64; 2]> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        while let Some(feature) = iter.next()? {
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            batch.push([x, y]);
            if !col_schema.is_empty() {
                let mut pusher = ColumnPusher {
                    cols: &mut batch_cols,
                };
                feature.process_properties(&mut pusher).ok();
            }
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
        let GisFilePath::HttpLocation(url) = path else {
            bail!("Wrong GisFilePath type!");
        };
        let reader = match reader_cache.borrow_mut().remove(url) {
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
        let mut batch: Vec<[f64; 2]> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_cols = make_batch_cols();
        let mut features = reader
            .select_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
            .await?;
        web_sys::console::log_1(&JsValue::from_str(&format!("After Feature BBOX query")));
        while let Some(feature) = features.next().await? {
            let [x, y] = match feature.to_geo() {
                Ok(Geometry::Point(p)) => [p.x(), p.y()],
                _ => continue,
            };
            batch.push([x, y]);
            if !col_schema.is_empty() {
                let mut pusher = ColumnPusher {
                    cols: &mut batch_cols,
                };
                feature.process_properties(&mut pusher).ok();
            }
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
        let reader = FgbReader::open(BufReader::new(File::open(str_path)?))?;
        let descriptor = Self::make_layer_descriptor(reader.header(), path)?;
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
                points: Vec::new(),
                attributes: Vec::new(),
                field_names: field_names.unwrap_or(Vec::new()),
                index: None,
                bbox: None,
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
