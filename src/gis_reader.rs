use flatgeobuf::{
    ColumnType, FallibleStreamingIterator, FeatureProperties, FgbReader, GeometryType,
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
use std::io::Cursor;

use crate::{
    gis_layer::{AttributeValue, BatchMessage, GisFeature, GisLayer, LayerEntry, LayerKind},
    point_cloud_layer::{AttributeColumn, PointCloudLayer},
};

pub struct LayerDescriptor {
    pub name: String,
    pub num_features: u64,
    pub field_names: Vec<String>,
    pub geometry_type: GeometryType,
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
        Self::make_layer_descriptor(reader)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_layer_descriptor(bytes: &[u8]) -> anyhow::Result<LayerDescriptor> {
        let reader = FgbReader::open(BufReader::new(Cursor::new(bytes)))?;
        Self::make_layer_descriptor(reader)
    }

    fn make_layer_descriptor<R: Read + Seek>(
        dataset: FgbReader<R>,
    ) -> anyhow::Result<LayerDescriptor> {
        let header = dataset.header();
        Ok(LayerDescriptor {
            name: header.name().unwrap_or("N/A").to_string(),
            num_features: header.features_count(),
            field_names: header
                .columns()
                .map(|cols| cols.iter().map(|c| c.name().to_string()).collect())
                .unwrap_or_default(),
            geometry_type: header.geometry_type(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_layer_batched(
        path: &str,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(File::open(path)?))?;
        Self::load_layer_batched_impl(reader, dest_idx, tx, selected_fields)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_layer_batched(
        bytes: Vec<u8>,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(Cursor::new(bytes)))?;
        Self::load_layer_batched_impl(reader, dest_idx, tx, selected_fields)
    }

    fn load_layer_batched_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    pub fn load_point_layer_batched(
        path: &str,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(File::open(path)?))?;
        Self::load_point_layer_batched_impl(reader, dest_idx, tx, selected_fields)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_point_layer_batched(
        bytes: Vec<u8>,
        _layer_idx: usize,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(Cursor::new(bytes)))?;
        Self::load_point_layer_batched_impl(reader, dest_idx, tx, selected_fields)
    }

    fn load_point_layer_batched_impl<R: Read + Seek>(
        reader: FgbReader<R>,
        dest_idx: usize,
        tx: mpsc::SyncSender<BatchMessage>,
        selected_fields: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
            }
        }

        if !batch.is_empty() {
            tx.send(BatchMessage::Points(dest_idx, batch, batch_cols))?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_selected_without_features(
        path: &str,
        _indices: &[usize],
    ) -> Result<Vec<LayerEntry>, Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(File::open(path)?))?;
        Ok(vec![Self::layer_entry_from_reader(reader)?])
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_selected_without_features(
        bytes: &[u8],
        _indices: &[usize],
    ) -> Result<Vec<LayerEntry>, Box<dyn std::error::Error>> {
        let reader = FgbReader::open(BufReader::new(Cursor::new(bytes)))?;
        Ok(vec![Self::layer_entry_from_reader(reader)?])
    }

    fn layer_entry_from_reader<R: Read + Seek>(
        reader: FgbReader<R>,
    ) -> Result<LayerEntry, Box<dyn std::error::Error>> {
        let descriptor = Self::make_layer_descriptor(reader)?;
        let layer_kind = match descriptor.geometry_type {
            GeometryType::Point => LayerKind::Points(PointCloudLayer::default()),
            _ => LayerKind::Vector(GisLayer::default()),
        };
        Ok(LayerEntry {
            data: layer_kind,
            visible: true,
            name: descriptor.name,
            color: [0, 0, 255],
            opacity: 255,
        })
    }
}
