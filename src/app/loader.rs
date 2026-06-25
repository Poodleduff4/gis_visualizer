use std::sync::mpsc;

use crate::gis_reader::{GeoParquetReader, GisFilePath, GisReader, LayerDescriptor};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use super::{GisEditorApp, LoadMode};

impl GisEditorApp {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn open_file(&mut self, path: GisFilePath) {
        match GisReader::load_layer_descriptor(&path.to_string()) {
            Ok(descriptor) => self.apply_layer(descriptor, path),
            Err(e) => self.status = format!("Error reading layers: {e}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn open_file(
        &mut self,
        file_url: GisFilePath,
        tx: mpsc::SyncSender<LayerDescriptor>,
    ) -> Result<(), anyhow::Error> {
        match file_url {
            GisFilePath::HttpLocation(url) => {
                spawn_local(async move {
                    match GisReader::load_layer_descriptor(&url).await {
                        Ok(descriptor) => {
                            web_sys::console::log_1(&JsValue::from_str(&format!("{}", url)));
                            tx.send(descriptor);
                        }
                        Err(_e) => {}
                    }
                });
            }
            GisFilePath::Bytes(bytes, name) => {
                match GeoParquetReader::load_descriptor_from_bytes(&bytes, &name) {
                    Ok(descriptor) => {
                        let _ = tx.send(descriptor);
                    }
                    Err(e) => {
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "parquet open error: {e}"
                        )));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn apply_layer(&mut self, descriptor: LayerDescriptor, file_path: GisFilePath) {
        let mut seen = std::collections::HashSet::new();
        let mut all_fields: Vec<String> = Vec::new();
        for f in &descriptor.field_names {
            if seen.insert(f.clone()) {
                all_fields.push(f.clone());
            }
        }
        all_fields.sort();
        self.pending_field_selection = all_fields.into_iter().map(|f| (f, true)).collect();
        self.pending_load_mode = LoadMode::GeometryOnly;
        self.pending_layers = vec![descriptor.clone()]
            .into_iter()
            .map(|d| (d, true))
            .collect();
        self.pending_file = Some(file_path);
        self.pending_file_descriptor = Some(descriptor.clone());
    }
}
