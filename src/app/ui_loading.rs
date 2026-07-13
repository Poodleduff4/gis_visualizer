use std::sync::mpsc::{self, TryRecvError};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use crate::gis_layer::{BatchMessage, LayerKind};
use crate::gis_reader::{GisReader, ReadOp};
#[cfg(target_arch = "wasm32")]
use crate::raster_reader::load_raster_bytes;
#[cfg(not(target_arch = "wasm32"))]
use crate::raster_reader::load_raster_sync;

use super::{GisEditorApp, LoadMode};

impl GisEditorApp {
    pub(super) fn poll_loading(&mut self, ui: &mut egui::Ui) {
        let pending_file = self.file_pick_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(file) = pending_file {
            self.file_pick_rx = None;
            #[cfg(not(target_arch = "wasm32"))]
            self.open_file(file);
            #[cfg(target_arch = "wasm32")]
            self.open_file(file, self.load_layer_descriptor_tx.clone());
        }

        // ── Raster descriptor pick → preview window ───────────────────────────
        if let Some(desc) = self
            .raster_descriptor_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok())
        {
            self.raster_descriptor_rx = None;
            self.pending_raster_descriptor = Some(desc);
        }
        if let Some(desc) = &self.pending_raster_descriptor {
            let mut do_load = false;
            let mut do_cancel = false;

            egui::Window::new("Raster Info")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.strong(&desc.name);
                    ui.separator();
                    egui::Grid::new("raster_info_grid")
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Variable:");
                            ui.label(&desc.variable);
                            ui.end_row();
                            if !desc.date.is_empty() {
                                ui.label("Date:");
                                ui.label(&desc.date);
                                ui.end_row();
                            }
                            ui.label("Dimensions:");
                            ui.label(format!("{} × {} px", desc.width, desc.height));
                            ui.end_row();
                            ui.label("Pixel count:");
                            ui.label(format!("{}", desc.width as u64 * desc.height as u64));
                            ui.end_row();
                            if !desc.units.is_empty() {
                                ui.label("Units:");
                                ui.label(&desc.units);
                                ui.end_row();
                            }
                            ui.label("Sample format:");
                            ui.label(if desc.is_f32 {
                                "32-bit float".to_string()
                            } else {
                                format!("{}-bit (unsupported)", desc.bits_per_sample)
                            });
                            ui.end_row();
                            ui.label("File size:");
                            ui.label(format_bytes(desc.file_size));
                            ui.end_row();
                        });
                    ui.separator();
                    if !desc.is_f32 {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 80, 80),
                            "Expected a 32-bit float TIFF — load may fail.",
                        );
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            do_load = true;
                        }
                        if ui.button("Cancel").clicked() {
                            do_cancel = true;
                        }
                    });
                });

            if do_load {
                let desc = self.pending_raster_descriptor.take().unwrap();
                let (tx, rx) = mpsc::channel::<Result<crate::gis_layer::LayerEntry, String>>();
                self.raster_load_rx = Some(rx);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let path = desc.path.unwrap();
                    std::thread::spawn(move || {
                        let result = load_raster_sync(&path).map_err(|e| e.to_string());
                        let _ = tx.send(result);
                    });
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let (bytes, name) = desc.bytes.unwrap();
                    let result = load_raster_bytes(bytes, &name).map_err(|e| e.to_string());
                    let _ = tx.send(result);
                }
            } else if do_cancel {
                self.pending_raster_descriptor = None;
            }
        }
        if let Some(result) = self
            .raster_load_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok())
        {
            self.raster_load_rx = None;
            match result {
                Ok(layer) => {
                    if let Some(extent) = layer.data.extent() {
                        if let Some(rect) = self.last_canvas_rect {
                            self.viewport.fit_to(extent, rect);
                            self.fitted = true;
                        }
                    }
                    self.layers.push(layer);
                    self.active_layer_idx = Some(self.layers.len() - 1);
                    self.points_dirty = true;
                    self.globe_points_dirty = true;
                    self.raster_dirty = true;
                    self.flat_raster_dirty = true;
                    self.map_render_ttl = 3;
                    self.status = "Raster loaded.".to_string();
                    #[cfg(not(target_arch = "wasm32"))]
                    if self.snapshot_restore.is_some() {
                        self.apply_snapshot_progress();
                    }
                }
                Err(e) => {
                    self.status = format!("Raster load failed: {e}");
                }
            }
        }

        // ── Snapshot file pick ────────────────────────────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self
            .snapshot_pick_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok())
        {
            self.snapshot_pick_rx = None;
            match std::fs::read_to_string(&path) {
                Ok(toml_str) => match toml::from_str::<crate::snapshot::AppSnapshot>(&toml_str) {
                    Ok(snap) => {
                        let mut queue: std::collections::VecDeque<_> =
                            snap.layers.into_iter().collect();
                        if let Some(first) = queue.pop_front() {
                            // Cancel any in-progress load so its messages don't land on the new layer.
                            self.load_rx = None;
                            self.streaming_features = true;
                            self.layers.clear();
                            self.fitted = true; // prevent auto-fit from overriding restored viewport
                            self.histogram = None;
                            self.bivariate = None;
                            self.lisa_results = None;
                            self.local_variance_results = None;
                            self.field_stats = None;
                            self.active_layer_idx = None;
                            self.selected_id = None;
                            self.show_histogram = false;
                            self.show_bivariate = false;
                            self.show_lisa = false;
                            self.show_local_variance = false;
                            self.snapshot_restore = Some(crate::snapshot::PendingSnapshotRestore {
                                queue,
                                pending_layer_settings: None,
                                viewport: snap.viewport,
                                display: snap.display,
                                analysis: snap.analysis,
                            });
                            self.open_snapshot_layer(first);
                        } else {
                            self.status = "Snapshot has no layers.".to_string();
                        }
                    }
                    Err(e) => self.status = format!("Snapshot parse error: {e}"),
                },
                Err(e) => self.status = format!("Snapshot read error: {e}"),
            }
        }

        // ── Layer selector (shown after file pick) ────────────────────────────
        let mut load_indices: Option<Vec<usize>> = None;
        let mut cancel_pending = false;

        // Auto-confirm layer dialog for snapshot loading.
        #[cfg(not(target_arch = "wasm32"))]
        if self.snapshot_restore.is_some() && self.pending_file.is_some() {
            self.pending_load_mode = super::LoadMode::WithAttributes;
            let selected_attrs = self
                .snapshot_restore
                .as_ref()
                .and_then(|r| r.pending_layer_settings.as_ref())
                .map(|s| s.selected_attributes.clone())
                .unwrap_or_default();
            for (name, sel) in &mut self.pending_field_selection {
                *sel = selected_attrs.is_empty() || selected_attrs.contains(name);
            }
            load_indices = Some(
                self.pending_layers
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i)
                    .collect(),
            );
        }

        if self.pending_file.is_some() && load_indices.is_none() {
            egui::Window::new("Select Layers")
                .collapsible(false)
                .resizable(true)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label("Choose which layers to load:");
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .id_salt("layer_scroll")
                        .max_height(200.0)
                        .show(ui, |ui| {
                            for (desc, selected, convert) in &mut self.pending_layers {
                                ui.checkbox(
                                    selected,
                                    format!("{} ({} features)", desc.name, desc.num_features),
                                );
                                if let Some(crs) = &desc.crs {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new(format!("CRS: {crs}")).weak());
                                        if desc.crs_epsg.is_some() {
                                            ui.checkbox(convert, "Convert to WGS84");
                                        }
                                    });
                                }
                            }
                        });

                    ui.separator();
                    ui.label("Attributes:");
                    ui.radio_value(
                        &mut self.pending_load_mode,
                        LoadMode::GeometryOnly,
                        "Geometry only",
                    );
                    ui.radio_value(
                        &mut self.pending_load_mode,
                        LoadMode::WithAttributes,
                        "With attributes",
                    );

                    if self.pending_load_mode == LoadMode::WithAttributes
                        && !self.pending_field_selection.is_empty()
                    {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Select attributes:");
                            if ui.small_button("All").clicked() {
                                for (_, s) in &mut self.pending_field_selection {
                                    *s = true;
                                }
                            }
                            if ui.small_button("None").clicked() {
                                for (_, s) in &mut self.pending_field_selection {
                                    *s = false;
                                }
                            }
                        });
                        egui::ScrollArea::vertical()
                            .id_salt("attr_scroll")
                            .max_height(200.0)
                            .show(ui, |ui| {
                                for (name, selected) in &mut self.pending_field_selection {
                                    ui.checkbox(selected, name.as_str());
                                }
                            });
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        let any_selected = self.pending_layers.iter().any(|(_, s, _)| *s);
                        if ui
                            .add_enabled(any_selected, egui::Button::new("Load"))
                            .clicked()
                        {
                            load_indices = Some(
                                self.pending_layers
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, (_, s, _))| *s)
                                    .map(|(i, _)| i)
                                    .collect(),
                            );
                            self.layer_picker_window_open = false;
                            self.streaming_features = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel_pending = true;
                        }
                    });
                });
        }
        if let Ok(new_layer_descriptor) = self.load_layer_descriptor_rx.try_recv() {
            let path = new_layer_descriptor.location.clone();
            self.apply_layer(new_layer_descriptor, path);
        } else {
        }
        if let Some(indices) = load_indices {
            let path = self.pending_file.take().unwrap();
            let attr_fields: Option<Vec<String>> = match self.pending_load_mode {
                LoadMode::GeometryOnly => None,
                LoadMode::WithAttributes => Some(
                    self.pending_field_selection
                        .iter()
                        .filter(|(_, sel)| *sel)
                        .map(|(name, _)| name.clone())
                        .collect(),
                ),
            };

            // Captured before `pending_layers` is cleared — one entry per selected
            // index, in the same order `load_selected_without_features` returns layers.
            let mut crs_notice: Option<String> = None;
            let crs_transforms: Vec<Option<crate::crs::CrsTransform>> = indices
                .iter()
                .map(|&i| {
                    let (desc, _selected, convert) = &self.pending_layers[i];
                    let Some(epsg) = convert.then(|| desc.crs_epsg).flatten() else {
                        return None;
                    };
                    match crate::crs::CrsTransform::from_epsg(epsg) {
                        Ok(t) => {
                            if t.approximate_datum {
                                crs_notice = Some(format!(
                                    "Note: EPSG:{epsg}'s datum needs a grid file this app doesn't \
                                     have — reprojected with an approximate (ellipsoid-only) shift, \
                                     expect up to ~100m offset."
                                ));
                            }
                            Some(t)
                        }
                        Err(e) => {
                            crs_notice = Some(format!(
                                "CRS conversion failed for EPSG:{epsg} ({e}) — layer loaded \
                                 un-reprojected; it will likely be off the map."
                            ));
                            None
                        }
                    }
                })
                .collect();

            self.pending_layers.clear();
            self.pending_field_selection.clear();

            #[cfg(not(target_arch = "wasm32"))]
            let mut layers = GisReader::load_selected_without_features(
                path.clone(),
                &indices,
                attr_fields.clone(),
            )
            .expect("Error loading featureless layers!");
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "Before load layers: {}",
                path.to_string()
            )));
            #[cfg(target_arch = "wasm32")]
            let mut layers = GisReader::load_selected_without_features(
                path.clone(),
                self.pending_file_descriptor.clone().unwrap(),
                attr_fields.clone(),
            )
            .expect("Error loading featureless layers!");
            let first_new = self.layers.len();
            let is_points: Vec<bool> = layers
                .iter()
                .map(|l| matches!(l.data, LayerKind::Points(_)))
                .collect();
            for (layer, t) in layers.iter_mut().zip(crs_transforms.iter()) {
                layer.crs_transform = t.clone();
            }
            self.layers.extend(layers.into_iter());
            self.active_layer_idx = Some(first_new);
            self.status = crs_notice.unwrap_or_else(|| format!("Loading {} layer(s)…", indices.len()));

            let rect_clone = self
                .viewport
                .viewport_bbox(self.last_canvas_rect.clone().unwrap());
            #[cfg(target_arch = "wasm32")]
            let (load_tx, load_rx) = mpsc::sync_channel::<BatchMessage>(100_000);
            #[cfg(not(target_arch = "wasm32"))]
            let (load_tx, load_rx) = mpsc::sync_channel::<BatchMessage>(10);
            self.load_rx = Some(load_rx);
            let cancel_clone = self.cancel_stream.clone();
            let path_clone = path.clone();
            #[cfg(target_arch = "wasm32")]
            let reader_cache_for_load = self.fgb_reader_cache.clone();
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::spawn(move || {
                for (pos, _file_idx) in indices.into_iter().enumerate() {
                    let dest = first_new + pos;
                    let result = GisReader::read_file(
                        path_clone.clone(),
                        dest,
                        is_points[pos],
                        ReadOp::Full,
                        attr_fields.clone(),
                        load_tx.clone(),
                        cancel_clone.clone(),
                    );
                    if let Err(e) = result {
                        eprintln!("[load thread] error: {e:#}");
                    }
                }
            });
            #[cfg(target_arch = "wasm32")]
            spawn_local(async move {
                for (pos, _file_idx) in indices.into_iter().enumerate() {
                    let dest = first_new + pos;
                    let result = GisReader::read_file(
                        path_clone.clone(),
                        dest,
                        is_points[pos],
                        ReadOp::Bbox(rect_clone),
                        attr_fields.clone(),
                        load_tx.clone(),
                        cancel_clone.clone(),
                        reader_cache_for_load.clone(),
                    )
                    .await;
                    if let Err(e) = result {
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "read_file error: {e}"
                        )));
                    }
                }
            });
        } else if cancel_pending {
            self.pending_file = None;
            self.pending_layers.clear();
            self.pending_field_selection.clear();
        }
        if let Some(load_rx) = &self.load_rx {
            for msg in load_rx.try_iter() {
                match msg {
                    BatchMessage::Points(layer_idx, mut pts, named_cols) => {
                        #[cfg(target_arch = "wasm32")]
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "BatchMessage::Points layer={layer_idx} count={}",
                            pts.len()
                        )));
                        let transform =
                            self.layers.get(layer_idx).and_then(|l| l.crs_transform.clone());
                        if let Some(t) = &transform {
                            for (_, xy) in pts.iter_mut() {
                                t.convert(xy);
                            }
                        }
                        if let Some(LayerKind::Points(pc)) =
                            &mut self.layers.get_mut(layer_idx).map(|l| &mut l.data)
                        {
                            std::sync::Arc::make_mut(&mut pc.points).extend(pts);
                            if pc.attributes.is_empty() && !named_cols.is_empty() {
                                pc.field_names =
                                    named_cols.iter().map(|(n, _)| n.clone()).collect();
                                for (_, col) in named_cols {
                                    pc.attributes.push(col);
                                }
                            } else {
                                for (dst, (_, src)) in pc.attributes.iter_mut().zip(named_cols) {
                                    dst.extend_from(src);
                                }
                            }
                        }
                        self.points_dirty = true;
                    }
                    BatchMessage::ViewportPoints(layer_idx, pts) => {
                        if let Some(LayerKind::Points(pc)) =
                            &mut self.layers.get_mut(layer_idx).map(|l| &mut l.data)
                        {
                            pc.viewport_mask.set_elements(0);
                            pts.iter()
                                .for_each(|idx| pc.viewport_mask.set(*idx as usize, true));
                        }
                    }
                    BatchMessage::Vector(layer_idx, mut features) => {
                        let transform =
                            self.layers.get(layer_idx).and_then(|l| l.crs_transform.clone());
                        if let Some(t) = &transform {
                            for f in &mut features {
                                f.reproject(t);
                            }
                        }
                        if let Some(LayerKind::Vector(gl)) =
                            &mut self.layers.get_mut(layer_idx).map(|l| &mut l.data)
                        {
                            gl.features.extend(features);
                        }
                        self.points_dirty = true;
                    }
                }
            }
            self.map_render_ttl = 10;
            // Keep polling so the bounded channel doesn't fill and block the stream future.
            // 16 ms cap is fast enough to drain without pinning the UI at full vsync rate.
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
            if let Err(TryRecvError::Disconnected) = load_rx.try_recv() {
                self.status = "Ready".to_string();
                self.load_rx = None;
                self.streaming_features = false;
                #[cfg(not(target_arch = "wasm32"))]
                self.apply_snapshot_progress();
            }
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}
