use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;

use bitvec::{bitvec, vec::BitVec};
use futures_channel::oneshot;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use egui::{CentralPanel, UiKind};

use crate::filter::{FilterLogic, FilterOperation, LayerAttributeFilter};
use crate::gis_layer::{
    ramp_rgba, AttributeValue, BatchMessage, LayerKind, LayerSelection, RasterDisplayMode,
};
#[cfg(target_arch = "wasm32")]
use crate::gis_reader::GeoParquetReader;
use crate::gis_reader::{GisFilePath, GisReader};
use crate::globe::{collect_globe_points, GlobeCallback, GlobePipeline};
use crate::gpu_collect::collect_gpu_points;
use crate::heatmap::{HeatmapLayer, HeatmapMetric};
use crate::histogram::{
    compute_bivariate, compute_field_stats, compute_histogram, extract_field_values, lisa_inner,
    local_variance_inner,
};
use crate::map_view::{
    draw_lisa_overlay, draw_local_variance_overlay, draw_selection_bboxes, render_raster_overlay,
    show_map, show_quadtree_heatmap, show_spatial_index_grid,
};
use crate::selection_stats::{
    compute_selection_bivariate, compute_selection_field_stats, compute_selection_histogram,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::parquet::{extract_batch_as_u32, query_parquet};
use crate::point_cloud::{PointCloudCallback, PointCloudPipeline};
#[cfg(target_arch = "wasm32")]
use crate::raster_reader::{load_raster_bytes, read_raster_descriptor_bytes};
#[cfg(not(target_arch = "wasm32"))]
use crate::raster_reader::{load_raster_sync, read_raster_descriptor_sync};
use crate::sidebar::{show_sidebar, SidebarAction};
use crate::spatial_index::{IndexKind, SpatialIndex};
use crate::uncertainty_quadtree::MeasurementType;

use super::{now_ms, ClickTarget, GisEditorApp, LoadMode, MapView, LAYER_PANEL_WIDTH};

fn bbox_contains(outer: &[f64; 4], inner: &[f64; 4]) -> bool {
    outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3]
}

fn union_bboxes(bboxes: &[[f64; 4]]) -> Option<[f64; 4]> {
    bboxes.iter().copied().reduce(|a, b| {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    })
}

impl eframe::App for GisEditorApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::Panel::top("menu_bar").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_basemap, "Basemap");
                    ui.checkbox(&mut self.show_index, "Spatial Index");
                    if self.show_index {
                        ui.indent("index_kind", |ui| {
                            ui.radio_value(&mut self.index_kind, IndexKind::Quadtree, "Quadtree");
                            ui.radio_value(
                                &mut self.index_kind,
                                IndexKind::Hilbert,
                                "Hilbert R-Tree",
                            );
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Quadtree Split Density:");
                        ui.add(
                            egui::Slider::new(&mut self.spatial_index_split_density, 100..=10000)
                                .step_by(5.0),
                        );
                    });
                    ui.vertical(|ui| {
                        ui.label("Uncertainty Quadtree Split Type:");
                        ui.horizontal(|ui| {
                            if ui.button("Variance").clicked() {
                                self.selected_split_measurement_type = MeasurementType::Variance;
                                self.heatmap_dirty = true;
                            }
                            if ui.button("Kernel-Density Entropy").clicked() {
                                self.selected_split_measurement_type =
                                    MeasurementType::KernalDensity;
                                self.heatmap_dirty = true;
                            }
                        })
                    });
                    ui.horizontal(|ui| {
                        ui.label("Uncertainty Quadtree Threshold:");
                        ui.add(
                            egui::Slider::new(&mut self.uncertainty_split_threshold, 0_f32..=2.)
                                .step_by(0.01),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Uncertainty Quadtree Max Depth:");
                        ui.add(
                            egui::Slider::new(&mut self.uncertainty_max_depth, 1..=20).step_by(1.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Heatmap Opacity:");
                        ui.add(egui::Slider::new(&mut self.hilbert_order, 1..=12).step_by(1.0));
                    });
                    if ui
                        .checkbox(&mut self.show_heatmap, "Quadtree Heatmap")
                        .changed()
                        && self.show_heatmap
                    {
                        self.heatmap_dirty = true;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Heatmap Opacity:");
                        ui.add(egui::Slider::new(&mut self.heatmap_opacity, 0..=255).step_by(1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Heatmap Metric:");
                        ui.radio_value(
                            &mut self.heatmap_metric,
                            HeatmapMetric::Density,
                            "Density",
                        );
                        ui.radio_value(
                            &mut self.heatmap_metric,
                            HeatmapMetric::Unpredictability,
                            "Unpredictability",
                        );
                    });
                    ui.horizontal(|ui| {
                        if let Some(idx) = self.active_layer_idx {
                            if !self.layers[idx].roi_bboxes.is_empty() {
                                ui.label(format!(
                                    "ROI regions: {}",
                                    self.layers[idx].roi_bboxes.len()
                                ));
                                if ui.button("Clear ROI").clicked() {
                                    self.layers[idx].roi_bboxes.clear();
                                    self.updated_filters = true;
                                    self.roi_rebuild_pending = true;
                                    self.heatmap_dirty = true;
                                }
                            }
                        }
                    });
                    if self.has_gpu {
                        ui.horizontal(|ui| {
                            ui.label("Point size:");
                            ui.add(
                                egui::Slider::new(&mut self.point_size, 1.0..=20.0).step_by(0.5),
                            );
                        });
                        if ui
                            .checkbox(&mut self.viewport_culling, "Viewport Culling")
                            .changed()
                        {
                            self.points_dirty = true;
                        }
                    }
                    ui.separator();
                    ui.label("Click target:");
                    ui.radio_value(&mut self.click_target, ClickTarget::Feature, "Feature");
                    ui.radio_value(&mut self.click_target, ClickTarget::GridCell, "Grid Cell");
                    ui.radio_value(
                        &mut self.click_target,
                        ClickTarget::HeatmapRoi,
                        "Heatmap ROI",
                    );
                    if self.has_gpu {
                        ui.separator();
                        ui.label("Map view:");
                        if ui.radio(self.map_view == MapView::Flat, "Flat").clicked() {
                            self.map_view = MapView::Flat;
                            self.map_render_ttl = 3;
                        }
                        if ui.radio(self.map_view == MapView::Globe, "Globe").clicked() {
                            self.map_view = MapView::Globe;
                            self.globe_points_dirty = true;
                            self.raster_dirty = true;
                            self.map_render_ttl = 3;
                        }
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("Open…").clicked() {
                        ui.close_kind(UiKind::Menu);
                        let (f_tx, f_rx) = mpsc::channel::<GisFilePath>();
                        self.file_pick_rx = Some(f_rx);

                        #[cfg(not(target_arch = "wasm32"))]
                        std::thread::spawn(move || {
                            if let Some(f) = pollster::block_on(
                                rfd::AsyncFileDialog::new()
                                    .add_filter("All Supported", &["fgb", "parquet"])
                                    .add_filter("FlatGeobuf", &["fgb"])
                                    .add_filter("GeoParquet", &["parquet"])
                                    .pick_file(),
                            ) {
                                let path =
                                    GisFilePath::LocalFile(f.path().to_string_lossy().into_owned());
                                let _ = f_tx.send(path);
                            }
                        });

                        #[cfg(target_arch = "wasm32")]
                        {
                            let mut base_url = self.fgb_file_url.clone();
                            spawn_local(async move {
                                if let Some(f) = rfd::AsyncFileDialog::new()
                                    .add_filter("All Supported", &["fgb", "parquet"])
                                    .add_filter("FlatGeobuf", &["fgb"])
                                    .add_filter("GeoParquet", &["parquet"])
                                    .pick_file()
                                    .await
                                {
                                    let name = f.file_name();
                                    let path = if name.ends_with(".parquet") {
                                        let raw = f.read().await;
                                        let arc: std::sync::Arc<[u8]> = raw.into();
                                        GisFilePath::Bytes(arc, name)
                                    } else {
                                        base_url.push_str(&name);
                                        GisFilePath::HttpLocation(base_url)
                                    };
                                    let _ = f_tx.send(path);
                                }
                            });
                        }
                    }
                    if ui.button("Open Raster (GeoTIFF)…").clicked() {
                        ui.close_kind(UiKind::Menu);
                        let (tx, rx) = mpsc::channel::<crate::raster_reader::RasterDescriptor>();
                        self.raster_descriptor_rx = Some(rx);

                        #[cfg(not(target_arch = "wasm32"))]
                        std::thread::spawn(move || {
                            if let Some(f) = pollster::block_on(
                                rfd::AsyncFileDialog::new()
                                    .add_filter("GeoTIFF", &["tif", "tiff"])
                                    .pick_file(),
                            ) {
                                if let Ok(desc) =
                                    read_raster_descriptor_sync(&f.path().to_path_buf())
                                {
                                    let _ = tx.send(desc);
                                }
                            }
                        });

                        #[cfg(target_arch = "wasm32")]
                        spawn_local(async move {
                            if let Some(f) = rfd::AsyncFileDialog::new()
                                .add_filter("GeoTIFF", &["tif", "tiff"])
                                .pick_file()
                                .await
                            {
                                let name = f.file_name();
                                let bytes = f.read().await;
                                if let Ok(desc) = read_raster_descriptor_bytes(bytes, &name) {
                                    let _ = tx.send(desc);
                                }
                            }
                        });
                    }
                    ui.separator();
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Save Snapshot…").clicked() {
                        ui.close_kind(UiKind::Menu);
                        let snap = self.capture_snapshot();
                        match toml::to_string_pretty(&snap) {
                            Ok(toml_str) => {
                                std::thread::spawn(move || {
                                    if let Some(f) = pollster::block_on(
                                        rfd::AsyncFileDialog::new()
                                            .set_file_name("snapshot.toml")
                                            .add_filter("Snapshot", &["toml"])
                                            .save_file(),
                                    ) {
                                        let _ = std::fs::write(f.path(), toml_str);
                                    }
                                });
                            }
                            Err(e) => self.status = format!("Snapshot error: {e}"),
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Load Snapshot…").clicked() {
                        ui.close_kind(UiKind::Menu);
                        let (tx, rx) = std::sync::mpsc::channel::<std::path::PathBuf>();
                        self.snapshot_pick_rx = Some(rx);
                        std::thread::spawn(move || {
                            if let Some(f) = pollster::block_on(
                                rfd::AsyncFileDialog::new()
                                    .add_filter("Snapshot", &["toml"])
                                    .pick_file(),
                            ) {
                                let _ = tx.send(f.path().to_path_buf());
                            }
                        });
                    }
                    if ui.button("Quit").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                if ui
                    .toggle_value(&mut self.select_mode, "🔲 Select mode")
                    .changed()
                    && self.select_mode
                {
                    self.select_drag_start = None;
                }
            });
        });

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
                            for (desc, selected) in &mut self.pending_layers {
                                ui.checkbox(
                                    selected,
                                    format!("{} ({} features)", desc.name, desc.num_features),
                                );
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
                        let any_selected = self.pending_layers.iter().any(|(_, s)| *s);
                        if ui
                            .add_enabled(any_selected, egui::Button::new("Load"))
                            .clicked()
                        {
                            load_indices = Some(
                                self.pending_layers
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, (_, s))| *s)
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

            self.pending_layers.clear();
            self.pending_field_selection.clear();

            #[cfg(not(target_arch = "wasm32"))]
            let layers = GisReader::load_selected_without_features(
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
            let layers = GisReader::load_selected_without_features(
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
            self.layers.extend(layers.into_iter());
            self.active_layer_idx = Some(first_new);
            self.status = format!("Loading {} layer(s)…", indices.len());

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
                for (pos, file_idx) in indices.into_iter().enumerate() {
                    let dest = first_new + pos;
                    let result = if is_points[pos] {
                        GisReader::load_point_layer_batched(
                            path_clone.clone(),
                            file_idx,
                            dest,
                            load_tx.clone(),
                            attr_fields.clone(),
                        )
                    } else {
                        GisReader::load_layer_batched(
                            path_clone.clone(),
                            file_idx,
                            dest,
                            load_tx.clone(),
                            attr_fields.clone(),
                        )
                    };
                    if let Err(e) = result {
                        eprintln!("[load thread] error: {e:#}");
                    }
                }
            });
            #[cfg(target_arch = "wasm32")]
            spawn_local(async move {
                web_sys::console::log_1(&JsValue::from_str("spawn_local: starting load"));
                for (pos, file_idx) in indices.into_iter().enumerate() {
                    let dest = first_new + pos;
                    let result: anyhow::Result<()> = match &path_clone {
                        crate::gis_reader::GisFilePath::Bytes(bytes, _) => {
                            web_sys::console::log_1(&JsValue::from_str(&format!(
                                "spawn_local: loading parquet bytes={} is_points={}",
                                bytes.len(),
                                is_points[pos]
                            )));
                            if is_points[pos] {
                                let r = crate::gis_reader::GeoParquetReader::load_point_layer_batched_from_bytes(
                                        bytes.clone(),
                                        dest,
                                        load_tx.clone(),
                                        attr_fields.clone(),
                                    );
                                web_sys::console::log_1(&JsValue::from_str(&format!(
                                    "spawn_local: load result ok={}",
                                    r.is_ok()
                                )));
                                if let Err(ref e) = r {
                                    web_sys::console::log_1(&JsValue::from_str(&format!(
                                        "spawn_local: load error: {e}"
                                    )));
                                }
                                r
                            } else {
                                Ok(())
                            }
                        }
                        _ => {
                            if is_points[pos] {
                                GisReader::stream_fgb_bbox(
                                    &path_clone,
                                    rect_clone,
                                    file_idx,
                                    dest,
                                    load_tx.clone(),
                                    attr_fields.clone(),
                                    cancel_clone.clone(),
                                    reader_cache_for_load.clone(),
                                )
                                .await
                            } else {
                                Ok(())
                            }
                        }
                    };
                    let _ = result;
                }
                web_sys::console::log_1(&JsValue::from_str("spawn_local: done"));
            });
        } else if cancel_pending {
            self.pending_file = None;
            self.pending_layers.clear();
            self.pending_field_selection.clear();
        }
        if let Some(load_rx) = &self.load_rx {
            for msg in load_rx.try_iter() {
                match msg {
                    BatchMessage::Points(layer_idx, pts, named_cols) => {
                        #[cfg(target_arch = "wasm32")]
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "BatchMessage::Points layer={layer_idx} count={}",
                            pts.len()
                        )));
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
                    BatchMessage::Vector(layer_idx, features) => {
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

        // ── Histogram window ─────────────────────────────────────────────────
        if self.show_histogram {
            let mut open = true;
            let mut hist_recompute = false;
            let mut hist_apply_filter: Option<(String, f64, f64)> = None;

            egui::Window::new("Histogram")
                .open(&mut open)
                .resizable(true)
                .default_size([480.0, 320.0])
                .show(ui.ctx(), |ui| {
                    if let Some(hist) = &mut self.histogram {
                        ui.horizontal(|ui| {
                            ui.label("Field:");
                            ui.label(egui::RichText::new(&hist.field).strong());
                            if ui
                                .checkbox(&mut hist.filtered_only, "Filtered only")
                                .changed()
                            {
                                hist_recompute = true;
                            }
                            if ui.button("Recompute").clicked() {
                                hist_recompute = true;
                            }
                        });

                        let counts = hist.counts.clone();
                        let bin_edges = hist.bin_edges.clone();
                        let n = counts.len();
                        let range_lo = hist.range_lo;
                        let range_hi = hist.range_hi;
                        egui_plot::Plot::new("histogram_plot")
                            .height(220.0)
                            .allow_drag(false)
                            .allow_scroll(false)
                            .show(ui, |plot_ui| {
                                let bars: Vec<egui_plot::Bar> = counts
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &c)| {
                                        let center = (bin_edges[i] + bin_edges[i + 1]) * 0.5;
                                        let width = bin_edges[i + 1] - bin_edges[i];
                                        egui_plot::Bar::new(center, c as f64).width(width * 0.95)
                                    })
                                    .collect();
                                plot_ui.bar_chart(egui_plot::BarChart::new("counts", bars));
                                plot_ui.vline(
                                    egui_plot::VLine::new("lo", range_lo)
                                        .color(egui::Color32::from_rgb(255, 100, 100)),
                                );
                                plot_ui.vline(
                                    egui_plot::VLine::new("hi", range_hi)
                                        .color(egui::Color32::from_rgb(100, 200, 100)),
                                );
                            });

                        ui.separator();
                        let speed = (hist.max - hist.min) / 200.0;
                        let lo_max = hist.range_hi;
                        let hi_min = hist.range_lo;
                        ui.horizontal(|ui| {
                            ui.label("Range:");
                            ui.add(
                                egui::DragValue::new(&mut hist.range_lo)
                                    .speed(speed)
                                    .range(hist.min..=lo_max),
                            );
                            ui.label("to");
                            ui.add(
                                egui::DragValue::new(&mut hist.range_hi)
                                    .speed(speed)
                                    .range(hi_min..=hist.max),
                            );
                            if ui.button("Apply as Range Filter").clicked() {
                                hist_apply_filter =
                                    Some((hist.field.clone(), hist.range_lo, hist.range_hi));
                            }
                        });
                        ui.label(format!(
                            "min: {:.4}  max: {:.4}  bins: {}",
                            hist.min, hist.max, n
                        ));
                    }
                });

            if !open {
                self.show_histogram = false;
            }
            if hist_recompute {
                if let Some(idx) = self.active_layer_idx {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        let (field, filtered_only) = self
                            .histogram
                            .as_ref()
                            .map(|h| (h.field.clone(), h.filtered_only))
                            .unwrap_or_default();
                        self.histogram = compute_histogram(pc, &field, 50, filtered_only);
                    }
                }
            }
            if let Some((field, lo, hi)) = hist_apply_filter {
                if let Some(idx) = self.active_layer_idx {
                    let entry = &mut self.layers[idx];
                    entry
                        .filters
                        .retain(|f| f.attribute.as_deref() != Some(field.as_str()));
                    entry.filters.push(LayerAttributeFilter {
                        attribute: Some(field.clone()),
                        operation: Some(FilterOperation::GreaterThan),
                        comparitor: AttributeValue::Float(lo),
                        comparitor_raw: lo.to_string(),
                    });
                    entry.filters.push(LayerAttributeFilter {
                        attribute: Some(field.clone()),
                        operation: Some(FilterOperation::LessThan),
                        comparitor: AttributeValue::Float(hi),
                        comparitor_raw: hi.to_string(),
                    });
                    self.updated_filters = true;
                }
            }
        }

        // ── Bivariate / Scatter window ────────────────────────────────────────
        if self.show_bivariate {
            let mut open = true;
            egui::Window::new("Scatter / Correlation")
                .open(&mut open)
                .resizable(true)
                .default_size([520.0, 400.0])
                .show(ui.ctx(), |ui| {
                    if let Some(bv) = &self.bivariate {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "X: {}   Y: {}",
                                    bv.x_field, bv.y_field
                                ))
                                .strong(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(format!("n = {}", bv.n));
                                },
                            );
                        });

                        let points = bv.scatter_points.clone();
                        egui_plot::Plot::new("bivariate_scatter")
                            .height(260.0)
                            .x_axis_label(&bv.x_field)
                            .y_axis_label(&bv.y_field)
                            .show(ui, |plot_ui| {
                                let pts: egui_plot::PlotPoints =
                                    points.into_iter().map(|[x, y]| [x, y]).collect();
                                plot_ui.points(
                                    egui_plot::Points::new("pts", pts).radius(2.0).color(
                                        egui::Color32::from_rgba_unmultiplied(80, 160, 220, 160),
                                    ),
                                );
                            });

                        ui.separator();
                        egui::Grid::new("bv_stats_grid")
                            .num_columns(2)
                            .striped(true)
                            .min_col_width(120.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Stat").strong());
                                ui.label(egui::RichText::new("Value").strong());
                                ui.end_row();

                                ui.label("Pearson r");
                                ui.label(format!("{:.6}", bv.pearson_r));
                                ui.end_row();

                                ui.label("r²");
                                ui.label(format!("{:.6}", bv.pearson_r * bv.pearson_r));
                                ui.end_row();

                                ui.label("Covariance");
                                ui.label(format!("{:.4}", bv.covariance));
                                ui.end_row();

                                ui.label(format!("Mean {}", bv.x_field));
                                ui.label(format!("{:.4}", bv.x_mean));
                                ui.end_row();

                                ui.label(format!("Std {}", bv.x_field));
                                ui.label(format!("{:.4}", bv.x_std));
                                ui.end_row();

                                ui.label(format!("Mean {}", bv.y_field));
                                ui.label(format!("{:.4}", bv.y_mean));
                                ui.end_row();

                                ui.label(format!("Std {}", bv.y_field));
                                ui.label(format!("{:.4}", bv.y_std));
                                ui.end_row();
                            });

                        let strength = match bv.pearson_r.abs() {
                            r if r >= 0.7 => "strong",
                            r if r >= 0.4 => "moderate",
                            r if r >= 0.2 => "weak",
                            _ => "negligible",
                        };
                        let direction = if bv.pearson_r >= 0.0 {
                            "positive"
                        } else {
                            "negative"
                        };
                        ui.label(format!("{} {} correlation", strength, direction));
                    }
                });
            if !open {
                self.show_bivariate = false;
            }
        }


        // ── Layer color picker window ─────────────────────────────────────────
        if let Some(layer_idx) = self.color_picker_layer {
            if layer_idx < self.layers.len() {
                let mut open = true;
                let name = self.layers[layer_idx].name.clone();
                let mut color = self.layers[layer_idx].color;
                let mut color_changed = false;
                egui::Window::new("Layer Color")
                    .open(&mut open)
                    .resizable(false)
                    .collapsible(false)
                    .default_size([220.0, 240.0])
                    .show(ui.ctx(), |ui| {
                        ui.label(&name);
                        ui.separator();
                        if egui::color_picker::color_edit_button_srgb(ui, &mut color).changed() {
                            color_changed = true;
                        }
                    });
                if color_changed {
                    self.layers[layer_idx].color = color;
                    self.points_dirty = true;
                    self.globe_points_dirty = true;
                    self.map_render_ttl = 3;
                }
                if !open {
                    self.color_picker_layer = None;
                }
            } else {
                self.color_picker_layer = None;
            }
        }

        // ── Status bar ────────────────────────────────────────────────────────
        egui::Panel::bottom("status_bar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                if self.local_variance_rx.is_some() || self.lisa_rx.is_some() {
                    ui.spinner();
                }
            });
        });

        // ── Layer panel (left) ────────────────────────────────────────────────
        egui::Panel::left("layer_panel")
            .exact_size(LAYER_PANEL_WIDTH)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.heading("Layers");
                ui.separator();
                if self.layers.is_empty() {
                    ui.label("No layers loaded.");
                } else {
                    let mut remove_idx: Option<usize> = None;
                    let mut rebuild_quadtree_idx: Option<usize> = None;
                    let mut rebuild_rtree_idx: Option<usize> = None;
                    let mut rebuild_uncertainty_quadtree_idx: Option<usize> = None;
                    let mut rebuild_hilbert_idx: Option<usize> = None;
                    let mut visibility_changed = false;
                    let mut set_active_selection: Option<(usize, usize)> = None;
                    let mut remove_selection: Option<(usize, usize)> = None;
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                        .show(ui, |ui| {
                        for (i, entry) in self.layers.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut entry.visible, "").changed() {
                                    visibility_changed = true;
                                }
                                let is_active = self.active_layer_idx == Some(i);
                                let label = if is_active {
                                    egui::RichText::new(&entry.name).strong()
                                } else {
                                    egui::RichText::new(&entry.name)
                                };
                                let label_resp = ui.selectable_label(is_active, label);
                                if label_resp.clicked() {
                                    if !is_active {
                                        self.active_layer_idx = Some(i);
                                        self.selected_id = None;
                                        self.points_dirty = true;
                                    }
                                    if let LayerKind::Points(pc) = &mut entry.data {
                                        if !pc.points.is_empty() {
                                            pc.ensure_bbox();
                                        }
                                    }
                                    self.fitted = false;
                                }
                                label_resp.context_menu(|ui| {
                                    if ui.button("Build Quadtree").clicked() {
                                        rebuild_quadtree_idx = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                    ui.separator();

                                    let field_names = entry.data.field_names().clone();
                                    if !field_names.is_empty() {
                                        let attr_label = format!(
                                            "Uncertainty attribute: {}",
                                            self.selected_uncertainty_attribute
                                                .as_deref()
                                                .unwrap_or("—")
                                        );
                                        ui.menu_button(attr_label, |ui| {
                                            for name in field_names.iter() {
                                                if ui
                                                    .selectable_value(
                                                        &mut self.selected_uncertainty_attribute,
                                                        Some(name.clone()),
                                                        name.as_str(),
                                                    )
                                                    .clicked()
                                                {
                                                    ui.close_kind(UiKind::Menu);
                                                    self.heatmap_dirty = true;
                                                }
                                            }
                                        });
                                    }
                                    if ui.button("Build R-Tree").clicked() {
                                        rebuild_rtree_idx = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                    if ui.button("Build Uncertainty Quadtree").clicked() {
                                        rebuild_uncertainty_quadtree_idx = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                    ui.separator();
                                    if ui.button("Build Hilbert").clicked() {
                                        rebuild_hilbert_idx = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                    ui.separator();
                                    if ui.button("Change Color…").clicked() {
                                        self.color_picker_layer = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                });
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("✕").clicked() {
                                            remove_idx = Some(i);
                                        }
                                    },
                                );
                            });
                            if !entry.selections.is_empty() {
                                egui::CollapsingHeader::new(format!(
                                    "Selections ({})",
                                    entry.selections.len()
                                ))
                                .id_salt(("selections_hdr", i))
                                .default_open(false)
                                .show(ui, |ui| {
                                    for (sidx, sel) in entry.selections.iter().enumerate() {
                                        ui.horizontal(|ui| {
                                            let is_active_sel =
                                                entry.active_selection == Some(sidx);
                                            if ui
                                                .selectable_label(
                                                    is_active_sel,
                                                    format!(
                                                        "{} ({} feat.)",
                                                        sel.name,
                                                        sel.ids.len()
                                                    ),
                                                )
                                                .clicked()
                                            {
                                                set_active_selection = Some((i, sidx));
                                            }
                                            if ui.small_button("✕").clicked() {
                                                remove_selection = Some((i, sidx));
                                            }
                                        });
                                    }
                                });
                            }
                        }
                    });
                    if let Some(idx) = remove_idx {
                        self.layers.remove(idx);
                        self.active_layer_idx = match self.active_layer_idx {
                            Some(a) if a == idx => {
                                if self.layers.is_empty() {
                                    None
                                } else {
                                    Some(0)
                                }
                            }
                            Some(a) if a > idx => Some(a - 1),
                            other => other,
                        };
                        self.selected_id = None;
                        self.points_dirty = true;
                        self.globe_points_dirty = true;
                        self.raster_dirty = true;
                        self.flat_raster_dirty = true;
                        self.map_render_ttl = 3;
                    }
                    if let Some(idx) = rebuild_rtree_idx {
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_rtree(),
                            LayerKind::Vector(_) | LayerKind::Raster(_) => {}
                        }
                        self.fitted = true;
                        self.viewport_load_pending = true;
                        self.viewport_stable_frames = 0;
                    }
                    if let Some(idx) = rebuild_quadtree_idx {
                        let capacity = self.spatial_index_split_density;
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_quadtree(capacity),
                            LayerKind::Vector(gl) => gl.rebuild_quadtree(capacity),
                            LayerKind::Raster(_) => {}
                        }
                        self.fitted = true;
                        self.viewport_load_pending = true;
                        self.viewport_stable_frames = 0;
                        self.heatmap_dirty = true;
                    }
                    if let Some(idx) = rebuild_uncertainty_quadtree_idx {
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => {
                                if let Some(attr) = &self.selected_uncertainty_attribute {
                                    pc.rebuild_uncertainty_quadtree(
                                        attr.clone(),
                                        self.uncertainty_split_threshold,
                                        self.selected_split_measurement_type.clone(),
                                        self.uncertainty_max_depth,
                                    );
                                }
                            }
                            LayerKind::Vector(_gl) => {}
                            LayerKind::Raster(_) => {}
                        }
                        self.heatmap_dirty = true;
                    }
                    if let Some(idx) = rebuild_hilbert_idx {
                        let order = self.hilbert_order;
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_hilbert_tree(order),
                            LayerKind::Vector(gl) => gl.rebuild_hilbert_tree(order),
                            LayerKind::Raster(_) => {}
                        }
                        self.fitted = true;
                        self.viewport_load_pending = true;
                        self.viewport_stable_frames = 0;
                    }
                    if visibility_changed {
                        self.points_dirty = true;
                        self.globe_points_dirty = true;
                        self.raster_dirty = true;
                        self.flat_raster_dirty = true;
                        self.map_render_ttl = 3;
                    }
                    if let Some((li, sidx)) = set_active_selection {
                        let entry = &mut self.layers[li];
                        entry.active_selection = if entry.active_selection == Some(sidx) {
                            None
                        } else {
                            Some(sidx)
                        };
                        self.selection_histogram = None;
                        self.selection_bivariate = None;
                        self.selection_field_stats = None;
                    }
                    if let Some((li, sidx)) = remove_selection {
                        self.layers[li].selections.remove(sidx);
                        let fixup = |sel: &mut Option<usize>| match *sel {
                            Some(s) if s == sidx => *sel = None,
                            Some(s) if s > sidx => *sel = Some(s - 1),
                            _ => {}
                        };
                        fixup(&mut self.layers[li].active_selection);
                    }
                }
            });

        // ── Sidebar (right) ───────────────────────────────────────────────────
        egui::Panel::right("sidebar")
            .min_size(260.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Recompute stats when field or filters change
                    let stats_stale = self.histogram_field != self.last_stats_field
                        || (self.updated_filters && !self.histogram_field.is_empty());
                    if stats_stale {
                        self.field_stats = self.active_layer_idx.and_then(|idx| {
                            if let LayerKind::Points(pc) = &self.layers[idx].data {
                                compute_field_stats(pc, &self.histogram_field, false)
                            } else {
                                None
                            }
                        });
                        self.last_stats_field = self.histogram_field.clone();
                    }

                    // ── Raster controls (band/range/legend) ────────────────────
                    if let Some(idx) = self.active_layer_idx {
                        if let LayerKind::Raster(raster) = &mut self.layers[idx].data {
                            ui.heading("Raster");
                            ui.label(format!("Variable: {}", raster.variable()));
                            ui.label(format!("Grid: {}×{}", raster.width, raster.height));

                            let mut changed = false;

                            if raster.bands.len() > 1 {
                                let is_rgb =
                                    matches!(raster.display_mode, RasterDisplayMode::Rgb { .. });
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(!is_rgb, "Single band").clicked()
                                        && is_rgb
                                    {
                                        raster.display_mode = RasterDisplayMode::Single(0);
                                        changed = true;
                                    }
                                    if ui.selectable_label(is_rgb, "RGB composite").clicked()
                                        && !is_rgb
                                    {
                                        let n = raster.bands.len();
                                        raster.display_mode = RasterDisplayMode::Rgb {
                                            r: 0,
                                            g: 1.min(n - 1),
                                            b: 2.min(n - 1),
                                        };
                                        changed = true;
                                    }
                                });
                            }

                            match &mut raster.display_mode {
                                RasterDisplayMode::Single(band_idx) => {
                                    if raster.bands.len() > 1 {
                                        let names: Vec<String> =
                                            raster.bands.iter().map(|b| b.name.clone()).collect();
                                        egui::ComboBox::from_label("Band")
                                            .selected_text(names[*band_idx].clone())
                                            .show_ui(ui, |ui| {
                                                for (i, name) in names.iter().enumerate() {
                                                    if ui
                                                        .selectable_value(band_idx, i, name)
                                                        .clicked()
                                                    {
                                                        changed = true;
                                                    }
                                                }
                                            });
                                    }
                                    if raster.bands.len() > 1 {
                                        ui.horizontal(|ui| {
                                            let label = if self.raster_playback_enabled {
                                                "⏸ Pause"
                                            } else {
                                                "▶ Play"
                                            };
                                            if ui.button(label).clicked() {
                                                self.raster_playback_enabled =
                                                    !self.raster_playback_enabled;
                                                self.raster_playback_last_tick_ms = now_ms();
                                            }
                                            ui.label("Interval (s):");
                                            ui.add(
                                                egui::DragValue::new(
                                                    &mut self.raster_playback_interval_secs,
                                                )
                                                .speed(0.1)
                                                .range(0.05..=30.0),
                                            );
                                        });
                                        if self.raster_playback_enabled {
                                            let now = now_ms();
                                            let elapsed_secs =
                                                (now - self.raster_playback_last_tick_ms) / 1000.0;
                                            if elapsed_secs
                                                >= self.raster_playback_interval_secs as f64
                                            {
                                                *band_idx = (*band_idx + 1) % raster.bands.len();
                                                changed = true;
                                                self.raster_playback_last_tick_ms = now;
                                            }
                                            ui.ctx().request_repaint_after(
                                                std::time::Duration::from_millis(100),
                                            );
                                        }
                                    }
                                    let band = &raster.bands[*band_idx];
                                    let (data_min, data_max) = (band.data_min, band.data_max);
                                    let mut display_min = band.display_min;
                                    let mut display_max = band.display_max;
                                    let mut range_changed = false;
                                    ui.label(format!(
                                        "Data range: {:.2} .. {:.2}",
                                        data_min, data_max
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.label("Min:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut display_min)
                                                    .speed((data_max - data_min) / 200.0),
                                            )
                                            .changed()
                                        {
                                            range_changed = true;
                                        }
                                        ui.label("Max:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut display_max)
                                                    .speed((data_max - data_min) / 200.0),
                                            )
                                            .changed()
                                        {
                                            range_changed = true;
                                        }
                                        if ui.small_button("Reset range").clicked() {
                                            display_min = data_min;
                                            display_max = data_max;
                                            range_changed = true;
                                        }
                                    });
                                    // Color range is shared across all bands of this layer
                                    // (not per-band) so playback doesn't jump contrast frame
                                    // to frame.
                                    if range_changed {
                                        for b in raster.bands.iter_mut() {
                                            b.display_min = display_min;
                                            b.display_max = display_max;
                                        }
                                        changed = true;
                                    }
                                    let band = &mut raster.bands[*band_idx];

                                    // Gradient legend
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(ui.available_width(), 16.0),
                                        egui::Sense::hover(),
                                    );
                                    let steps = 32;
                                    let w = rect.width() / steps as f32;
                                    for i in 0..steps {
                                        let t = i as f64 / (steps - 1) as f64;
                                        let [r, g, b, a] = ramp_rgba(t);
                                        let seg = egui::Rect::from_min_size(
                                            egui::pos2(rect.left() + i as f32 * w, rect.top()),
                                            egui::vec2(w + 1.0, rect.height()),
                                        );
                                        ui.painter().rect_filled(
                                            seg,
                                            0.0,
                                            egui::Color32::from_rgba_unmultiplied(r, g, b, a),
                                        );
                                    }
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:.1}", band.display_min))
                                                .small(),
                                        );
                                        let units = if raster.units.is_empty() {
                                            String::new()
                                        } else {
                                            format!(" ({})", raster.units)
                                        };
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{:.1}{units}",
                                                        band.display_max
                                                    ))
                                                    .small(),
                                                );
                                            },
                                        );
                                    });
                                }
                                RasterDisplayMode::Rgb { r, g, b } => {
                                    let names: Vec<String> =
                                        raster.bands.iter().map(|bd| bd.name.clone()).collect();
                                    for (label, idx) in [("Red", r), ("Green", g), ("Blue", b)] {
                                        egui::ComboBox::from_label(label)
                                            .selected_text(names[*idx].clone())
                                            .show_ui(ui, |ui| {
                                                for (i, name) in names.iter().enumerate() {
                                                    if ui.selectable_value(idx, i, name).clicked() {
                                                        changed = true;
                                                    }
                                                }
                                            });
                                    }
                                }
                            }

                            if changed {
                                self.raster_dirty = true;
                                self.flat_raster_dirty = true;
                                self.map_render_ttl = 3;
                            }
                            ui.separator();
                        }
                    }

                    let selection_ctx = self.active_layer_idx.and_then(|idx| {
                        self.layers
                            .get(idx)
                            .and_then(|e| e.active_selection.map(|sidx| (idx, sidx)))
                    });

                    if let Some((li, sidx)) = selection_ctx {
                        self.show_selection_sidebar(ui, li, sidx);
                        return;
                    }

                    let action = show_sidebar(
                        ui,
                        &mut self.layers,
                        self.active_layer_idx,
                        self.selected_id,
                        &mut self.add_form,
                        &mut self.save_path,
                        self.selected_index_cell_data.as_ref(),
                        &mut self.adding_filter,
                        &mut self.updated_filters,
                        &mut self.histogram_field,
                        &mut self.bivariate_y_field,
                        self.field_stats.as_ref(),
                        &mut self.spatial_field,
                        &mut self.spatial_radius,
                    );

                    match action {
                        SidebarAction::AddAttribute {
                            feature_id: _,
                            name: _,
                            value: _,
                        } => {}
                        SidebarAction::SaveAs(_path) => {}
                        SidebarAction::OpenHistogram(field) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    self.histogram = compute_histogram(pc, &field, 50, true);
                                    self.show_histogram = self.histogram.is_some();
                                }
                            }
                        }
                        SidebarAction::OpenBivariate(x_field, y_field) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    self.bivariate =
                                        compute_bivariate(pc, &x_field, &y_field, true, 5000);
                                    self.show_bivariate = self.bivariate.is_some();
                                }
                            }
                        }
                        SidebarAction::ExportFiltered =>
                        {
                            #[cfg(not(target_arch = "wasm32"))]
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    let points: Vec<(u32, [f64; 2])> =
                                        pc.points.iter().cloned().collect();
                                    let field_names = pc.field_names.clone();
                                    let filter_mask = pc.filter_mask.clone();
                                    let attrs: Vec<_> = pc
                                        .attributes
                                        .iter()
                                        .map(|col| {
                                            use crate::point_cloud_layer::AttributeColumn;
                                            match col {
                                                AttributeColumn::Float(v) => {
                                                    AttributeColumn::Float(v.clone())
                                                }
                                                AttributeColumn::Integer(v) => {
                                                    AttributeColumn::Integer(v.clone())
                                                }
                                                AttributeColumn::Text(v) => {
                                                    AttributeColumn::Text(v.clone())
                                                }
                                            }
                                        })
                                        .collect();
                                    let name = self.layers[idx].name.clone();
                                    std::thread::spawn(move || {
                                        if let Some(path) = pollster::block_on(
                                            rfd::AsyncFileDialog::new()
                                                .add_filter("GeoParquet", &["parquet"])
                                                .set_file_name(format!("{}_export.parquet", name))
                                                .save_file(),
                                        ) {
                                            use crate::point_cloud_layer::PointCloudLayer;
                                            let pc_export = PointCloudLayer {
                                                points: std::sync::Arc::new(points),
                                                attributes: attrs,
                                                field_names,
                                                filter_mask,
                                                index: None,
                                                bbox: None,
                                                viewport_mask: bitvec::bitvec![0; 0],
                                            };
                                            let _ = crate::exporter::export_filtered_points(
                                                &pc_export,
                                                path.path().to_string_lossy().as_ref(),
                                            );
                                        }
                                    });
                                }
                            }
                        }
                        SidebarAction::ComputeLocalVariance(field, radius) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    if let Some(values) = extract_field_values(pc, &field) {
                                        let points = pc.points.clone();
                                        let filter_mask = pc.filter_mask.clone();
                                        let index = pc.index.clone();
                                        let (tx, rx) = oneshot::channel();
                                        self.local_variance_rx = Some(rx);
                                        self.show_local_variance = false;
                                        self.show_lisa = false;
                                        self.status = format!(
                                            "Computing local variance ({} pts)…",
                                            pc.filter_mask.count_ones()
                                        );
                                        ui.ctx().request_repaint();
                                        std::thread::spawn(move || {
                                            let result = local_variance_inner(
                                                &points,
                                                &filter_mask,
                                                &values,
                                                radius,
                                                index.as_deref(),
                                            );
                                            tx.send(result).ok();
                                        });
                                    }
                                }
                            }
                        }
                        SidebarAction::ComputeLisa(field, radius) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    if let Some(values) = extract_field_values(pc, &field) {
                                        let points = pc.points.clone();
                                        let filter_mask = pc.filter_mask.clone();
                                        let index = pc.index.clone();
                                        let (tx, rx) = oneshot::channel();
                                        self.lisa_rx = Some(rx);
                                        self.show_lisa = false;
                                        self.show_local_variance = false;
                                        self.status = format!(
                                            "Computing LISA ({} pts)…",
                                            pc.filter_mask.count_ones()
                                        );
                                        ui.ctx().request_repaint();
                                        std::thread::spawn(move || {
                                            let result = lisa_inner(
                                                &points,
                                                &filter_mask,
                                                &values,
                                                radius,
                                                index.as_deref(),
                                            );
                                            tx.send(result).ok();
                                        });
                                    }
                                }
                            }
                        }
                        SidebarAction::None => {}
                    }
                });
            });

        if self.updated_filters {
            let layer = &mut self.layers[self.active_layer_idx.unwrap()];
            let idx = self.active_layer_idx.unwrap();
            match layer.filters.len() {
                0 => {
                    use crate::point_cloud_layer::PointCloudLayer;
                    layer.data.reset_filter_mask();
                    if !layer.roi_bboxes.is_empty() {
                        let roi_bboxes = layer.roi_bboxes.clone();
                        if let LayerKind::Points(pc) = &mut layer.data {
                            for (pos, (_, p)) in pc.points.iter().enumerate() {
                                if !PointCloudLayer::point_in_any_roi(*p, &roi_bboxes) {
                                    pc.filter_mask.set(pos, false);
                                }
                            }
                        }
                    }
                    self.points_dirty = true;
                    self.updated_filters = false;
                    self.roi_rebuild_pending = true;
                }
                _ => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let join_op = match layer.filter_logic {
                            FilterLogic::And => " AND ",
                            FilterLogic::Or => " OR ",
                        };
                        let where_clause = layer
                            .filters
                            .iter()
                            .map(|f| {
                                let attr = f.attribute.as_deref().unwrap_or("");
                                let op = f.operation.clone().unwrap().to_string();
                                let val = match &f.comparitor {
                                    AttributeValue::Text(s) => {
                                        format!("'{}'", s.replace('\'', "''"))
                                    }
                                    AttributeValue::Integer(n) => n.to_string(),
                                    AttributeValue::Float(v) => v.to_string(),
                                };
                                format!("\"{}\" {} {}", attr, op, val)
                            })
                            .collect::<Vec<String>>()
                            .join(join_op);
                        let query = format!("SELECT \"idx\" FROM layer WHERE {}", where_clause);
                        let file_path = layer.descriptor.location.to_string();
                        let (tx, rx) = oneshot::channel::<(usize, Vec<u32>)>();
                        self.filtered_idx_rx = Some(rx);
                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Runtime::new().unwrap();
                            rt.block_on(async {
                                let matching_ids = match query_parquet(&file_path, query).await {
                                    Ok(batch_vec) => batch_vec
                                        .iter()
                                        .filter_map(|b| extract_batch_as_u32(b, "idx"))
                                        .flatten()
                                        .collect::<Vec<u32>>(),
                                    Err(e) => {
                                        eprintln!("[filter] {e:#}");
                                        Vec::new()
                                    }
                                };
                                let _ = tx.send((idx, matching_ids));
                            });
                        });
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        use crate::point_cloud_layer::AttributeColumn;
                        let use_and = layer.filter_logic == FilterLogic::And;
                        let matching_ids: Vec<u32> = if let LayerKind::Points(pc) = &layer.data {
                            let filters = &layer.filters;
                            let field_names = &pc.field_names;
                            let attributes = &pc.attributes;
                            pc.points
                                .iter()
                                .enumerate()
                                .filter_map(|(pos, (parquet_id, _))| {
                                    let eval = |f: &LayerAttributeFilter| {
                                        let Some(attr) = f.attribute.as_deref() else {
                                            return false;
                                        };
                                        let Some(col_pos) =
                                            field_names.iter().position(|n| n == attr)
                                        else {
                                            return false;
                                        };
                                        let Some(col) = attributes.get(col_pos) else {
                                            return false;
                                        };
                                        let raw = &f.comparitor_raw;
                                        match (&f.operation, col) {
                                            (
                                                Some(FilterOperation::GreaterThan),
                                                AttributeColumn::Float(v),
                                            ) => raw
                                                .parse::<f64>()
                                                .map(|t| v[pos] > t)
                                                .unwrap_or(false),
                                            (
                                                Some(FilterOperation::LessThan),
                                                AttributeColumn::Float(v),
                                            ) => raw
                                                .parse::<f64>()
                                                .map(|t| v[pos] < t)
                                                .unwrap_or(false),
                                            (
                                                Some(FilterOperation::Equal),
                                                AttributeColumn::Float(v),
                                            ) => raw
                                                .parse::<f64>()
                                                .map(|t| (v[pos] - t).abs() < 1e-9)
                                                .unwrap_or(false),
                                            (
                                                Some(FilterOperation::GreaterThan),
                                                AttributeColumn::Integer(v),
                                            ) => raw
                                                .parse::<i64>()
                                                .map(|t| v[pos] > t)
                                                .unwrap_or(false),
                                            (
                                                Some(FilterOperation::LessThan),
                                                AttributeColumn::Integer(v),
                                            ) => raw
                                                .parse::<i64>()
                                                .map(|t| v[pos] < t)
                                                .unwrap_or(false),
                                            (
                                                Some(FilterOperation::Equal),
                                                AttributeColumn::Integer(v),
                                            ) => raw
                                                .parse::<i64>()
                                                .map(|t| v[pos] == t)
                                                .unwrap_or(false),
                                            (
                                                Some(FilterOperation::Equal),
                                                AttributeColumn::Text(v),
                                            ) => v[pos] == *raw,
                                            _ => false,
                                        }
                                    };
                                    let passes = if use_and {
                                        filters.iter().all(|f| eval(f))
                                    } else {
                                        filters.iter().any(|f| eval(f))
                                    };
                                    if passes {
                                        Some(*parquet_id)
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };
                        let (tx, rx) = oneshot::channel::<(usize, Vec<u32>)>();
                        self.filtered_idx_rx = Some(rx);
                        let _ = tx.send((idx, matching_ids));
                    }
                    self.updated_filters = false;
                }
            };
        }
        if let Some(rx) = &mut self.filtered_idx_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, idx_vec))) => {
                    use crate::point_cloud_layer::PointCloudLayer;
                    println!("{}", idx_vec.len());
                    if let Some(l) = self.layers.get_mut(layer_idx) {
                        let roi_bboxes = l.roi_bboxes.clone();
                        match &mut l.data {
                            LayerKind::Points(point_cloud_layer) => {
                                let matching: std::collections::HashSet<u32> =
                                    idx_vec.into_iter().collect();
                                let mut mask: BitVec = bitvec![0;point_cloud_layer.points.len()];
                                for (pos, (parquet_id, p)) in
                                    point_cloud_layer.points.iter().enumerate()
                                {
                                    if matching.contains(parquet_id)
                                        && PointCloudLayer::point_in_any_roi(*p, &roi_bboxes)
                                    {
                                        mask.set(pos, true);
                                    }
                                }
                                point_cloud_layer.filter_mask &= mask;
                                self.points_dirty = true;
                                self.roi_rebuild_pending = true;
                                ui.request_repaint();
                            }
                            LayerKind::Vector(_) | LayerKind::Raster(_) => {}
                        }
                    }
                }
                Ok(None) => {
                    println!("Not Ready Yet")
                }
                Err(_e) => self.filtered_idx_rx = None,
            }
        }

        // ── Progressive drill-down: rebuild finer index scoped to ROI ─────────
        if self.roi_rebuild_pending {
            self.roi_rebuild_pending = false;
            if let Some(idx) = self.active_layer_idx {
                let roi_bboxes = self.layers[idx].roi_bboxes.clone();
                let was_uncertainty = matches!(
                    self.layers[idx].data.index(IndexKind::Quadtree),
                    Some(SpatialIndex::UncertaintyQuadtree(_))
                );
                if let LayerKind::Points(pc) = &mut self.layers[idx].data {
                    pc.ensure_bbox();
                    let bbox = union_bboxes(&roi_bboxes).or(pc.bbox);
                    if let Some(bbox) = bbox {
                        if was_uncertainty {
                            if let Some(attr) = &self.selected_uncertainty_attribute {
                                pc.rebuild_uncertainty_quadtree_bounded(
                                    attr.clone(),
                                    self.uncertainty_split_threshold,
                                    self.selected_split_measurement_type.clone(),
                                    self.uncertainty_max_depth,
                                    bbox,
                                );
                            }
                        } else {
                            pc.rebuild_quadtree_bounded(
                                self.spatial_index_split_density,
                                bbox,
                            );
                        }
                    }
                }
            }
            self.points_dirty = true;
            self.heatmap_dirty = true;
        }

        // ── Poll spatial analysis background results ──────────────────────────
        if let Some(rx) = &mut self.local_variance_rx {
            match rx.try_recv() {
                Ok(Some(result)) => {
                    self.local_variance_results = Some(result);
                    self.show_local_variance = true;
                    self.local_variance_rx = None;
                    self.status = "Local variance done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.local_variance_rx = None;
                    self.status = "Local variance failed.".to_string();
                }
            }
        }
        if let Some(rx) = &mut self.lisa_rx {
            match rx.try_recv() {
                Ok(Some(result)) => {
                    self.lisa_results = result;
                    self.show_lisa = self.lisa_results.is_some();
                    self.lisa_rx = None;
                    self.status = "LISA done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.lisa_rx = None;
                    self.status = "LISA failed.".to_string();
                }
            }
        }

        // ── Rebuild quadtree when split-density slider changes ────────────────
        if self.spatial_index_split_density != self.last_split_density {
            let capacity = self.spatial_index_split_density;
            for entry in &mut self.layers {
                match &mut entry.data {
                    LayerKind::Points(point_layer) => {
                        point_layer.rebuild_quadtree(capacity);
                    }
                    LayerKind::Vector(vector_layer) => {
                        vector_layer.rebuild_quadtree(capacity);
                    }
                    LayerKind::Raster(_) => {}
                }
            }
            self.last_split_density = capacity;
            self.points_dirty = true;
            self.viewport_load_pending = true;
            self.viewport_stable_frames = 0;
        }
        if self.hilbert_order != self.last_hilbert_order {
            let order = self.hilbert_order;
            for entry in &mut self.layers {
                match &mut entry.data {
                    LayerKind::Points(point_layer) => {
                        point_layer.rebuild_hilbert_tree(order);
                    }
                    LayerKind::Vector(vector_layer) => {
                        vector_layer.rebuild_hilbert_tree(order);
                    }
                    LayerKind::Raster(_) => {}
                }
            }
            self.last_hilbert_order = order;
            self.points_dirty = true;
            self.viewport_load_pending = true;
            self.viewport_stable_frames = 0;
        }

        // ── Re-upload GPU points when data or style changes ───────────────────
        if self.has_gpu {
            let layer_changed = self.layers.len() != self.last_layer_count;
            let selection_changed = self.selected_id != self.last_selected_id;
            let size_changed = self.point_size != self.last_point_size;
            let viewport_changed = self.viewport.center != self.last_viewport_center
                || self.last_viewport_ppu != self.viewport.pixels_per_unit;
            if viewport_changed {
                self.viewport_stable_frames = 0;
                self.viewport_load_pending = true;
                self.last_viewport_center = self.viewport.center;
                self.last_viewport_ppu = self.viewport.pixels_per_unit;
                self.map_render_ttl = 2;
            } else if self.viewport_load_pending {
                self.viewport_stable_frames += 1;
            }
            if self.viewport_load_pending {
                let cursor_in_map = self
                    .last_canvas_rect
                    .and_then(|rect| ui.ctx().pointer_latest_pos().map(|p| rect.contains(p)))
                    .unwrap_or(false);
                if cursor_in_map {
                    ui.ctx().request_repaint();
                }
            }

            if (self.points_dirty || layer_changed || selection_changed || size_changed)
                && !self.streaming_features
            {
                if let Some(wrs) = frame.wgpu_render_state() {
                    let device = &wrs.device;
                    let queue = &wrs.queue;
                    let mut renderer = wrs.renderer.write();
                    if let Some(pipeline) =
                        renderer.callback_resources.get_mut::<PointCloudPipeline>()
                    {
                        collect_gpu_points(
                            &self.layers,
                            self.active_layer_idx,
                            self.selected_id,
                            if self.viewport_culling {
                                self.last_canvas_rect
                                    .map(|rect| self.viewport.viewport_bbox(rect))
                            } else {
                                None
                            },
                            self.point_size,
                            &mut self.gpu_points_buf,
                        );
                        pipeline.upload_points(device, queue, &self.gpu_points_buf);
                    }
                }
                self.map_render_ttl = 2;
                self.points_dirty = false;
                self.last_selected_id = self.selected_id;
                self.last_point_size = self.point_size;
                self.last_layer_count = self.layers.len();
                self.last_viewport_center = self.viewport.center;
                self.last_viewport_ppu = self.viewport.pixels_per_unit;
            }

            #[cfg(not(target_arch = "wasm32"))]
            if self.viewport_load_pending
                && self.viewport_stable_frames >= 3
                && !self.streaming_features
            {
                self.viewport_load_pending = false;
                self.cancel_stream
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.cancel_stream = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let (tx, rx) = mpsc::sync_channel(40);
                self.load_rx = Some(rx);
                let full_bbox = self
                    .viewport
                    .viewport_bbox(self.last_canvas_rect.clone().unwrap());
                for (actual_idx, layer) in self.layers.iter_mut().enumerate() {
                    if !layer.visible {
                        continue;
                    }
                    if let LayerKind::Points(pc) = &mut layer.data {
                        let pts_clone = Arc::clone(&pc.points);
                        let idx_clone = pc.index.clone();
                        let cancel_clone = self.cancel_stream.clone();
                        let tx_clone = tx.clone();
                        std::thread::spawn(move || {
                            crate::point_cloud_layer::query_and_stream_viewport(
                                actual_idx,
                                pts_clone,
                                idx_clone,
                                full_bbox,
                                tx_clone,
                                cancel_clone,
                            );
                        });
                    }
                }
            }

            #[cfg(target_arch = "wasm32")]
            if self.viewport_load_pending
                && self.viewport_stable_frames >= 3
                && !self.streaming_features
            {
                self.viewport_load_pending = false;
                let (tx, rx) = mpsc::sync_channel(40);
                self.load_rx = Some(rx);
                let full_bbox = self
                    .viewport
                    .viewport_bbox(self.last_canvas_rect.clone().unwrap());
                for (actual_idx, layer) in self.layers.iter_mut().enumerate() {
                    if !layer.visible {
                        continue;
                    }
                    if let LayerKind::Points(pc) = &mut layer.data {
                        let pts_clone = Arc::clone(&pc.points);
                        let idx_clone = pc.index.clone();
                        let tx_clone = tx.clone();
                        let cancel_clone = self.cancel_stream.clone();
                        spawn_local(async move {
                            crate::point_cloud_layer::query_and_stream_viewport(
                                actual_idx,
                                pts_clone,
                                idx_clone,
                                full_bbox,
                                tx_clone,
                                cancel_clone,
                            );
                        });
                    }
                }
            }
        }

        // ── Map (central panel) ───────────────────────────────────────────────
        CentralPanel::default().show_inside(ui, |ui| {
            if self.map_view == MapView::Globe {
                self.show_globe(ui, frame);
                return;
            }
            let active_layer = self.active_layer_idx.and_then(|i| self.layers.get(i));
            if !self.fitted {
                if let Some(entry) = active_layer {
                    if let Some(extent) = entry.data.extent() {
                        self.viewport
                            .fit_to(extent, ui.available_rect_before_wrap());
                        self.fitted = true;
                    }
                }
            }

            let bm = if self.show_basemap {
                Some(&self.basemap_cache)
            } else {
                None
            };

            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
            self.last_canvas_rect = Some(response.rect);

            let render_points = !self.has_gpu;
            let mut roi_toggle: Option<[f64; 4]> = None;
            let mut pending_selection: Option<[f64; 4]> = None;
            show_map(
                ui,
                &response,
                &painter,
                &self.layers,
                active_layer,
                &mut self.viewport,
                &mut self.selected_id,
                bm,
                render_points,
                &self.click_target,
                &mut self.selected_index_cell_data,
                &mut roi_toggle,
                self.select_mode,
                &mut self.select_drag_start,
                &mut pending_selection,
            );
            if let Some(bbox) = pending_selection {
                if let Some(idx) = self.active_layer_idx {
                    let ids = self.layers[idx].data.ids_in_bbox_with_fallback(bbox);
                    let entry = &mut self.layers[idx];
                    let name = format!("Selection {}", entry.selections.len() + 1);
                    entry.selections.push(LayerSelection {
                        name,
                        bbox,
                        ids,
                    });
                    entry.active_selection = Some(entry.selections.len() - 1);
                }
            }
            if let Some(idx) = self.active_layer_idx {
                if let Some(entry) = self.layers.get(idx) {
                    draw_selection_bboxes(
                        &painter,
                        &entry.selections,
                        entry.active_selection,
                        &self.viewport,
                        response.rect,
                    );
                }
            }
            if let Some(bbox) = roi_toggle {
                if let Some(idx) = self.active_layer_idx {
                    let roi = &mut self.layers[idx].roi_bboxes;
                    if let Some(pos) = roi.iter().position(|b| *b == bbox) {
                        // Exact same cell clicked again -> toggle off.
                        roi.remove(pos);
                    } else if let Some(pos) = roi
                        .iter()
                        .position(|b| bbox_contains(b, &bbox) || bbox_contains(&bbox, b))
                    {
                        // Nested with an existing selection (drilling into or
                        // out of it) -> narrow/replace instead of adding a
                        // second bbox, otherwise the union never shrinks.
                        roi[pos] = bbox;
                    } else {
                        roi.push(bbox);
                    }
                    self.updated_filters = true;
                    self.roi_rebuild_pending = true;
                }
            }
            let active_layer = self.active_layer_idx.and_then(|i| self.layers.get(i));

            // GPU point cloud: always blit the cached offscreen texture (cheap).
            // The offscreen re-render only happens when map_render_ttl > 0 (viewport/data changed).
            if self.has_gpu {
                let rect = response.rect;
                let [wx_min, wy_min, wx_max, wy_max] = self.viewport.viewport_bbox(rect);
                let world_size = [(wx_max - wx_min) as f32, (wy_max - wy_min) as f32];
                let render_dirty = self.map_render_ttl > 0;
                if self.map_render_ttl > 0 {
                    self.map_render_ttl -= 1;
                }
                painter.add(egui::Shape::Callback(
                    egui_wgpu::Callback::new_paint_callback(
                        rect,
                        PointCloudCallback {
                            world_min: [wx_min as f32, wy_min as f32],
                            world_size,
                            screen_min: [rect.left(), rect.top()],
                            screen_size: [rect.width(), rect.height()],
                            render_dirty,
                        },
                    ),
                ));
            }

            let visible_raster = self.layers.iter().find_map(|l| {
                if !l.visible {
                    return None;
                }
                if let LayerKind::Raster(r) = &l.data {
                    Some(r)
                } else {
                    None
                }
            });
            if let Some(raster) = visible_raster {
                render_raster_overlay(
                    ui,
                    &painter,
                    raster,
                    &self.viewport,
                    response.rect,
                    &mut self.raster_texture,
                    self.flat_raster_dirty,
                );
                self.flat_raster_dirty = false;
            }

            if self.selected_id != self.last_selected_id {
                self.points_dirty = true;
            }

            if self.show_heatmap {
                if self.active_layer_idx != self.last_heatmap_layer_idx {
                    self.last_heatmap_layer_idx = self.active_layer_idx;
                    self.heatmap_dirty = true;
                }
                if self.heatmap_dirty {
                    use crate::point_cloud_layer::AttributeColumn;
                    self.heatmap_cache = active_layer.and_then(|e| {
                        let LayerKind::Points(pc) = &e.data else {
                            return None;
                        };
                        let index = pc.index.as_deref()?;
                        let attr = self.selected_uncertainty_attribute.as_ref()?;
                        let field_idx = pc.field_names.iter().position(|n| n == attr)?;
                        let values: Vec<f64> = match &pc.attributes[field_idx] {
                            AttributeColumn::Float(v) => v.clone(),
                            AttributeColumn::Integer(v) => v.iter().map(|x| *x as f64).collect(),
                            AttributeColumn::Text(_) => return None,
                        };
                        Some(HeatmapLayer::build(
                            index,
                            &values,
                            self.selected_split_measurement_type.clone(),
                        ))
                    });
                    self.heatmap_dirty = false;
                }
                if let Some(heatmap) = &self.heatmap_cache {
                    let roi_bboxes = active_layer.map(|e| e.roi_bboxes.as_slice()).unwrap_or(&[]);
                    show_quadtree_heatmap(
                        &painter,
                        heatmap,
                        self.heatmap_metric,
                        roi_bboxes,
                        &self.viewport,
                        response.rect,
                        self.heatmap_opacity,
                    );

                    // ── Legend: gradient bar + range + meaning ──────────────────
                    let (title, max_val, unit) = match self.heatmap_metric {
                        HeatmapMetric::Density => {
                            ("Density (points/cell)".to_string(), heatmap.max_density, "")
                        }
                        HeatmapMetric::Unpredictability => {
                            let label = match &heatmap.measurement_type {
                                MeasurementType::Variance => "Unpredictability (variance)",
                                MeasurementType::KernalDensity => "Unpredictability (entropy)",
                            };
                            (label.to_string(), heatmap.max_unpredictability, "")
                        }
                    };
                    let r = response.rect;
                    let bar_w = 200.0_f32;
                    let bar_h = 14.0_f32;
                    let x = r.min.x + 10.0;
                    let y = r.max.y - 46.0;
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x - 4.0, y - 18.0),
                            egui::vec2(bar_w + 8.0, bar_h + 40.0),
                        ),
                        4.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
                    );
                    painter.text(
                        egui::pos2(x, y - 16.0),
                        egui::Align2::LEFT_TOP,
                        &title,
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    let steps = 40;
                    for i in 0..steps {
                        let t0 = i as f32 / steps as f32;
                        let t1 = (i + 1) as f32 / steps as f32;
                        let color = crate::map_view::heat_color(t0, 255);
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(x + t0 * bar_w, y),
                                egui::pos2(x + t1 * bar_w, y + bar_h),
                            ),
                            0.0,
                            color,
                        );
                    }
                    painter.text(
                        egui::pos2(x, y + bar_h + 2.0),
                        egui::Align2::LEFT_TOP,
                        format!("0{}", unit),
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    painter.text(
                        egui::pos2(x + bar_w, y + bar_h + 2.0),
                        egui::Align2::RIGHT_TOP,
                        format!("{:.3}{}", max_val, unit),
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                }
            }

            if self.show_index {
                let index = active_layer
                    .map(|e| e.data.index(self.index_kind))
                    .flatten();
                show_spatial_index_grid(&painter, index, &mut self.viewport, response.rect);
            }

            if self.show_local_variance {
                if let (Some(variances), Some(idx)) =
                    (&self.local_variance_results, self.active_layer_idx)
                {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        draw_local_variance_overlay(
                            &painter,
                            &pc.points,
                            &pc.filter_mask,
                            variances,
                            &self.viewport,
                            response.rect,
                            200,
                        );
                        // legend
                        let r = response.rect;
                        let x = r.min.x + 10.0;
                        let mut y = r.max.y - 80.0;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(x - 4.0, y - 4.0),
                                egui::vec2(140.0, 72.0),
                            ),
                            4.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
                        );
                        painter.text(
                            egui::pos2(x, y),
                            egui::Align2::LEFT_TOP,
                            "Local Variance",
                            egui::FontId::proportional(11.0),
                            egui::Color32::WHITE,
                        );
                        y += 16.0;
                        for (label, color) in [
                            ("Low", egui::Color32::from_rgb(0, 0, 255)),
                            ("Medium", egui::Color32::from_rgb(0, 200, 0)),
                            ("High", egui::Color32::from_rgb(255, 0, 0)),
                        ] {
                            painter.circle_filled(egui::pos2(x + 6.0, y + 6.0), 5.0, color);
                            painter.text(
                                egui::pos2(x + 16.0, y),
                                egui::Align2::LEFT_TOP,
                                label,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                            y += 16.0;
                        }
                    }
                }
            }

            if self.show_lisa {
                if let (Some(lisa), Some(idx)) = (&self.lisa_results, self.active_layer_idx) {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        draw_lisa_overlay(
                            &painter,
                            &pc.points,
                            &pc.filter_mask,
                            lisa,
                            &self.viewport,
                            response.rect,
                            200,
                        );
                        // legend
                        let r = response.rect;
                        let x = r.min.x + 10.0;
                        let mut y = r.max.y - 96.0;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(x - 4.0, y - 4.0),
                                egui::vec2(170.0, 88.0),
                            ),
                            4.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
                        );
                        painter.text(
                            egui::pos2(x, y),
                            egui::Align2::LEFT_TOP,
                            "LISA Clusters",
                            egui::FontId::proportional(11.0),
                            egui::Color32::WHITE,
                        );
                        y += 16.0;
                        for (label, color) in [
                            ("HH — high cluster", egui::Color32::from_rgb(220, 30, 30)),
                            ("LL — low cluster", egui::Color32::from_rgb(30, 80, 220)),
                            ("HL — high outlier", egui::Color32::from_rgb(240, 140, 20)),
                            ("LH — low outlier", egui::Color32::from_rgb(20, 200, 220)),
                        ] {
                            painter.circle_filled(egui::pos2(x + 6.0, y + 6.0), 5.0, color);
                            painter.text(
                                egui::pos2(x + 16.0, y),
                                egui::Align2::LEFT_TOP,
                                label,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                            y += 16.0;
                        }
                    }
                }
            }
        });
    }
}

impl GisEditorApp {
    /// Right-sidebar view shown instead of `show_sidebar` while a saved
    /// box-selection is active: Distribution / Spatial Analysis / Export,
    /// scoped to just the selection's ids rather than the whole layer.
    fn show_selection_sidebar(&mut self, ui: &mut egui::Ui, li: usize, sidx: usize) {
        let layer_name = self.layers[li].name.clone();
        let (sel_name, sel_ids) = {
            let sel = &self.layers[li].selections[sidx];
            (sel.name.clone(), sel.ids.clone())
        };

        ui.heading("GIS Viewer");
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("{layer_name} › {sel_name}")).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("✕ Deselect").clicked() {
                    self.layers[li].active_selection = None;
                }
            });
        });
        ui.label(format!("{} features selected", sel_ids.len()));
        ui.separator();

        let numeric_fields = self.layers[li].data.numeric_field_names();

        // ── Distribution ─────────────────────────────────────────────────
        ui.label(egui::RichText::new("Distribution").strong());
        if numeric_fields.is_empty() {
            ui.label("No numeric fields.");
        } else {
            if self.selection_field_a.is_empty() {
                self.selection_field_a = numeric_fields[0].clone();
            }
            ui.label("X field:");
            egui::ComboBox::from_id_salt("sel_dist_field_a")
                .selected_text(&self.selection_field_a)
                .show_ui(ui, |ui| {
                    for f in &numeric_fields {
                        ui.selectable_value(&mut self.selection_field_a, f.clone(), f);
                    }
                });

            {
                let entry = &self.layers[li];
                let sel = &entry.selections[sidx];
                let hist = compute_selection_histogram(&entry.data, sel, &self.selection_field_a, 30);
                let stats =
                    compute_selection_field_stats(&entry.data, sel, &self.selection_field_a);
                self.selection_histogram = hist;
                self.selection_field_stats = stats;
            }
            if let Some(hist) = &self.selection_histogram {
                let counts = hist.counts.clone();
                let bin_edges = hist.bin_edges.clone();
                egui_plot::Plot::new("sel_hist_plot")
                    .height(160.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .show(ui, |plot_ui| {
                        let bars: Vec<egui_plot::Bar> = counts
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| {
                                let center = (bin_edges[i] + bin_edges[i + 1]) * 0.5;
                                let width = bin_edges[i + 1] - bin_edges[i];
                                egui_plot::Bar::new(center, c as f64).width(width * 0.95)
                            })
                            .collect();
                        plot_ui.bar_chart(egui_plot::BarChart::new("counts", bars));
                    });
            } else {
                ui.label("No numeric data for this field.");
            }
            if let Some(stats) = &self.selection_field_stats {
                egui::Grid::new("sel_stats_grid")
                    .num_columns(2)
                    .striped(true)
                    .min_col_width(60.0)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Stat").strong());
                        ui.label(egui::RichText::new("Value").strong());
                        ui.end_row();
                        ui.label("Count");
                        ui.label(stats.count.to_string());
                        ui.end_row();
                        ui.label("Min");
                        ui.label(format!("{:.4}", stats.min));
                        ui.end_row();
                        ui.label("Max");
                        ui.label(format!("{:.4}", stats.max));
                        ui.end_row();
                        ui.label("Mean");
                        ui.label(format!("{:.4}", stats.mean));
                        ui.end_row();
                        ui.label("Std Dev");
                        ui.label(format!("{:.4}", stats.std_dev));
                        ui.end_row();
                        ui.label("P25 / P50 / P75");
                        ui.label(format!("{:.4} / {:.4} / {:.4}", stats.p25, stats.p50, stats.p75));
                        ui.end_row();
                    });
            }

            ui.add_space(4.0);
            ui.label("Y field (scatter):");
            egui::ComboBox::from_id_salt("sel_dist_field_b")
                .selected_text(if self.selection_field_b.is_empty() {
                    "<select field>"
                } else {
                    self.selection_field_b.as_str()
                })
                .show_ui(ui, |ui| {
                    for f in &numeric_fields {
                        ui.selectable_value(&mut self.selection_field_b, f.clone(), f);
                    }
                });
            if !self.selection_field_b.is_empty()
                && self.selection_field_b != self.selection_field_a
            {
                let entry = &self.layers[li];
                let sel = &entry.selections[sidx];
                self.selection_bivariate = compute_selection_bivariate(
                    &entry.data,
                    sel,
                    &self.selection_field_a,
                    &self.selection_field_b,
                    2000,
                );
                if let Some(bv) = &self.selection_bivariate {
                    ui.label(format!("Pearson r = {:.4}  (n = {})", bv.pearson_r, bv.n));
                    let points = bv.scatter_points.clone();
                    egui_plot::Plot::new("sel_scatter_plot")
                        .height(160.0)
                        .x_axis_label(&bv.x_field)
                        .y_axis_label(&bv.y_field)
                        .show(ui, |plot_ui| {
                            let pts: egui_plot::PlotPoints =
                                points.into_iter().map(|[x, y]| [x, y]).collect();
                            plot_ui.points(
                                egui_plot::Points::new("pts", pts).radius(2.0).color(
                                    egui::Color32::from_rgba_unmultiplied(80, 160, 220, 160),
                                ),
                            );
                        });
                } else {
                    ui.label("No numeric data for these fields.");
                }
            } else {
                self.selection_bivariate = None;
            }
        }

        ui.separator();

        // ── Spatial Analysis (Points layers only — needs a spatial index) ──
        ui.label(egui::RichText::new("Spatial Analysis").strong());
        if let LayerKind::Points(pc) = &self.layers[li].data {
            if numeric_fields.is_empty() {
                ui.label("No numeric fields.");
            } else {
                if self.spatial_field.is_empty() {
                    self.spatial_field = numeric_fields[0].clone();
                }
                ui.label("Field:");
                egui::ComboBox::from_id_salt("sel_spatial_field")
                    .selected_text(&self.spatial_field)
                    .show_ui(ui, |ui| {
                        for f in &numeric_fields {
                            ui.selectable_value(&mut self.spatial_field, f.clone(), f);
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(
                        egui::DragValue::new(&mut self.spatial_radius)
                            .speed(0.0001)
                            .range(1e-9..=1e6)
                            .max_decimals(6),
                    );
                });

                let mut mask = bitvec![0; pc.points.len()];
                for &id in &sel_ids {
                    if id < pc.filter_mask.len() && pc.filter_mask[id] {
                        mask.set(id, true);
                    }
                }

                ui.horizontal(|ui| {
                    if ui.button("Local Variance").clicked() {
                        if let Some(values) = extract_field_values(pc, &self.spatial_field) {
                            let points = pc.points.clone();
                            let index = pc.index.clone();
                            let radius = self.spatial_radius;
                            let thread_mask = mask.clone();
                            let (tx, rx) = oneshot::channel();
                            self.local_variance_rx = Some(rx);
                            self.show_local_variance = false;
                            self.show_lisa = false;
                            self.status =
                                format!("Computing local variance ({} pts)…", mask.count_ones());
                            ui.ctx().request_repaint();
                            std::thread::spawn(move || {
                                let result = local_variance_inner(
                                    &points,
                                    &thread_mask,
                                    &values,
                                    radius,
                                    index.as_deref(),
                                );
                                tx.send(result).ok();
                            });
                        }
                    }
                    if ui.button("LISA").clicked() {
                        if let Some(values) = extract_field_values(pc, &self.spatial_field) {
                            let points = pc.points.clone();
                            let index = pc.index.clone();
                            let radius = self.spatial_radius;
                            let thread_mask = mask.clone();
                            let (tx, rx) = oneshot::channel();
                            self.lisa_rx = Some(rx);
                            self.show_lisa = false;
                            self.show_local_variance = false;
                            self.status = format!("Computing LISA ({} pts)…", mask.count_ones());
                            ui.ctx().request_repaint();
                            std::thread::spawn(move || {
                                let result =
                                    lisa_inner(&points, &thread_mask, &values, radius, index.as_deref());
                                tx.send(result).ok();
                            });
                        }
                    }
                });
            }
        } else {
            ui.label("Only available for point-cloud layers.");
        }

        ui.separator();

        // ── Export ───────────────────────────────────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.label(egui::RichText::new("Export").strong());
            if let LayerKind::Points(pc) = &self.layers[li].data {
                let label = format!("Export selection ({} pts)", sel_ids.len());
                if ui.button(label).clicked() {
                    let points: Vec<(u32, [f64; 2])> = pc.points.iter().cloned().collect();
                    let field_names = pc.field_names.clone();
                    let filter_mask = pc.filter_mask.clone();
                    let attrs: Vec<_> = pc
                        .attributes
                        .iter()
                        .map(|col| {
                            use crate::point_cloud_layer::AttributeColumn;
                            match col {
                                AttributeColumn::Float(v) => AttributeColumn::Float(v.clone()),
                                AttributeColumn::Integer(v) => AttributeColumn::Integer(v.clone()),
                                AttributeColumn::Text(v) => AttributeColumn::Text(v.clone()),
                            }
                        })
                        .collect();
                    let ids = sel_ids.clone();
                    let name = format!("{}_{}", layer_name, sel_name);
                    std::thread::spawn(move || {
                        if let Some(path) = pollster::block_on(
                            rfd::AsyncFileDialog::new()
                                .add_filter("GeoParquet", &["parquet"])
                                .set_file_name(format!("{}_export.parquet", name))
                                .save_file(),
                        ) {
                            use crate::point_cloud_layer::PointCloudLayer;
                            let pc_export = PointCloudLayer {
                                points: std::sync::Arc::new(points),
                                attributes: attrs,
                                field_names,
                                filter_mask,
                                index: None,
                                bbox: None,
                                viewport_mask: bitvec::bitvec![0; 0],
                            };
                            let _ = crate::exporter::export_points_by_ids(
                                &pc_export,
                                &ids,
                                path.path().to_string_lossy().as_ref(),
                            );
                        }
                    });
                }
            } else {
                ui.label("Only available for point-cloud layers.");
            }
        }
    }

    fn show_globe(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        self.last_canvas_rect = Some(rect);
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.globe_camera.orbit(delta.x, delta.y);
            self.map_render_ttl = 3;
        }
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            self.globe_camera.zoom(scroll);
            self.map_render_ttl = 3;
        }

        if !self.has_gpu {
            return;
        }

        let render_dirty = self.map_render_ttl > 0;
        if self.map_render_ttl > 0 {
            self.map_render_ttl -= 1;
        }

        if self.globe_points_dirty {
            if let Some(wrs) = frame.wgpu_render_state() {
                let device = &wrs.device;
                let queue = &wrs.queue;
                let mut renderer = wrs.renderer.write();
                if let Some(pipeline) = renderer.callback_resources.get_mut::<GlobePipeline>() {
                    collect_globe_points(&self.layers, self.point_size, &mut self.globe_points_buf);
                    pipeline.upload_points(device, queue, &self.globe_points_buf);
                }
            }
            self.globe_points_dirty = false;
        }

        if self.raster_dirty {
            if let Some(wrs) = frame.wgpu_render_state() {
                let device = &wrs.device;
                let queue = &wrs.queue;
                let mut renderer = wrs.renderer.write();
                if let Some(pipeline) = renderer.callback_resources.get_mut::<GlobePipeline>() {
                    let raster = self.layers.iter().find_map(|l| {
                        if !l.visible {
                            return None;
                        }
                        if let LayerKind::Raster(r) = &l.data {
                            Some(r)
                        } else {
                            None
                        }
                    });
                    pipeline.update_raster(device, queue, raster);
                }
            }
            self.raster_dirty = false;
        }

        let painter = ui.painter_at(rect);
        painter.add(egui::Shape::Callback(
            egui_wgpu::Callback::new_paint_callback(
                rect,
                GlobeCallback {
                    camera: self.globe_camera.clone(),
                    screen_size: [rect.width(), rect.height()],
                    render_dirty,
                },
            ),
        ));

        if render_dirty {
            ui.ctx().request_repaint();
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
