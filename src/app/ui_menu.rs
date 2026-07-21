use std::sync::mpsc;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use egui::UiKind;

use crate::gis_reader::GisFilePath;
#[cfg(target_arch = "wasm32")]
use crate::raster_reader::read_raster_descriptor_bytes;
#[cfg(not(target_arch = "wasm32"))]
use crate::raster_reader::read_raster_descriptor_sync;
use crate::uncertainty_quadtree::MeasurementType;

use super::{ClickTarget, GisEditorApp, MapView, SelectShape};

impl GisEditorApp {
    pub(super) fn show_menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("menu_bar").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_basemap, "Basemap");
                    ui.horizontal(|ui| {
                        ui.label("Quadtree Split Density:");
                        ui.add(
                            egui::Slider::new(&mut self.spatial_index_split_density, 100..=50000)
                                .step_by(5.0),
                        );
                    });
                    ui.vertical(|ui| {
                        ui.label("Uncertainty Quadtree Split Type:");
                        ui.horizontal(|ui| {
                            if ui.button("Variance").clicked() {
                                self.selected_split_measurement_type = MeasurementType::Variance;
                            }
                            if ui.button("Kernel-Density Entropy").clicked() {
                                self.selected_split_measurement_type =
                                    MeasurementType::KernalDensity;
                            }
                        })
                    });
                    ui.horizontal(|ui| {
                        ui.label("Uncertainty Quadtree Threshold:");
                        ui.add(
                            egui::Slider::new(&mut self.uncertainty_split_threshold, 0_f32..=5.)
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
                        ui.add(egui::Slider::new(&mut self.heatmap_opacity, 0..=255).step_by(1.0));
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
                                    self.layers[idx].heatmap_dirty = true;
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
                    ui.horizontal(|ui| {
                        ui.label("Vector line width:");
                        ui.add(
                            egui::Slider::new(&mut self.vector_line_width, 0.5..=10.0).step_by(0.5),
                        );
                    });
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
                ui.menu_button("Analysis", |ui| {
                    if ui.button("Kernel Density Estimation…").clicked() {
                        self.kde_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }
                    if ui.button("Bivariate Grid Analysis…").clicked() {
                        self.bivariate_grid_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }
                    if ui.button("Grid Binning (Hexbin)…").clicked() {
                        self.gridbin_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }
                });
                ui.menu_button("Sampling", |ui| {
                    if ui.button("Sample Layer…").clicked() {
                        self.sampling_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }
                });
                #[cfg(not(target_arch = "wasm32"))]
                ui.menu_button("Plugins", |ui| {
                    if ui.button("Manage Plugins…").clicked() {
                        self.available_plugins = crate::plugin::discover_plugins(&self.plugins_dir);
                        self.plugin_window_open = true;
                        ui.close_kind(UiKind::Menu);
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
                                    .add_filter("All Supported", &["fgb", "parquet", "geojson"])
                                    .add_filter("FlatGeobuf", &["fgb"])
                                    .add_filter("GeoParquet", &["parquet"])
                                    .add_filter("GeoJSON", &["geojson"])
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
                                    .add_filter("All Supported", &["fgb", "parquet", "geojson"])
                                    .add_filter("FlatGeobuf", &["fgb"])
                                    .add_filter("GeoParquet", &["parquet"])
                                    .add_filter("GeoJSON", &["geojson"])
                                    .pick_file()
                                    .await
                                {
                                    let name = f.file_name();
                                    // Parquet and GeoJSON are read fully into memory (no
                                    // streaming format exists for either); FlatGeobuf streams
                                    // over HTTP instead.
                                    let path = if name.ends_with(".parquet")
                                        || name.ends_with(".geojson")
                                    {
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
                    if ui.button("Export…").clicked() {
                        ui.close_kind(UiKind::Menu);
                        self.export_window_open = true;
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
                ui.menu_button("Select", |ui| {
                    if ui
                        .radio_value(&mut self.select_shape, SelectShape::Rectangle, "🔲 Rectangle")
                        .changed()
                    {
                        self.select_drag_start = None;
                        self.select_polygon.clear();
                    }
                    if ui
                        .radio_value(&mut self.select_shape, SelectShape::Polygon, "🔺 Polygon")
                        .on_hover_text(
                            "Click to add vertices, double-click or right-click to close",
                        )
                        .changed()
                    {
                        self.select_drag_start = None;
                        self.select_polygon.clear();
                    }
                    ui.separator();
                    if ui.toggle_value(&mut self.select_mode, "Select mode").changed() {
                        self.select_drag_start = None;
                        self.select_polygon.clear();
                    }
                });
            });
        });
    }
}
