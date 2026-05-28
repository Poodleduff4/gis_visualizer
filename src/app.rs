use egui::{CentralPanel, Color32, UiKind};
use std::sync::mpsc::{self, SyncSender, TryRecvError};
use wgpu::naga::proc::vector_size_str;

use crate::basemap::BasemapCache;
use crate::gis_layer::{BatchMessage, GisLayer, LayerDescriptor, LayerEntry, LayerKind};
use crate::heatmap::HeatmapLayer;
use crate::map_view::{show_map, show_quadtree_heatmap, show_spatial_index_grid, Viewport};
use crate::point_cloud::{GpuPoint, PointCloudCallback, PointCloudPipeline};
use crate::quadtree::Quadtree;
use crate::sidebar::{show_sidebar, AddAttributeForm, SidebarAction};
use crate::spatial_index::IndexKind;

const LAYER_PANEL_WIDTH: f32 = 180.0;

const FILL_NORMAL: Color32 = Color32::from_rgb(100, 149, 237);
const FILL_SELECTED: Color32 = Color32::from_rgb(255, 165, 0);

pub struct GisEditorApp {
    layers: Vec<LayerEntry>,
    active_layer_idx: Option<usize>,
    layer_picker_window_open: bool,
    viewport: Viewport,
    selected_id: Option<usize>,
    add_form: AddAttributeForm,
    save_path: String,
    status: String,
    fitted: bool,
    show_basemap: bool,
    basemap_cache: BasemapCache,
    show_index: bool,
    index_kind: IndexKind,
    show_heatmap: bool,

    // Layer selector state (populated after file pick, before load)
    pending_file: Option<String>,
    pending_layers: Vec<(LayerDescriptor, bool)>,
    load_rx: Option<mpsc::Receiver<BatchMessage>>,

    // GPU point cloud state
    has_gpu: bool,
    pub point_size: f32,
    points_dirty: bool,
    last_selected_id: Option<usize>,
    last_point_size: f32,
    last_layer_count: usize,
    heatmap_opacity: u8,
    gpu_points_buf: Vec<GpuPoint>,
    spatial_index_split_density: usize,
    last_split_density: usize,
    hilbert_order: u32,
    last_hilbert_order: u32,
}

impl GisEditorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let has_gpu = if let Some(wrs) = cc.wgpu_render_state.as_ref() {
            let pipeline = PointCloudPipeline::new(&wrs.device, wrs.target_format);
            wrs.renderer.write().callback_resources.insert(pipeline);
            true
        } else {
            false
        };

        GisEditorApp {
            layers: Vec::new(),
            active_layer_idx: None,
            layer_picker_window_open: false,
            pending_file: None,
            pending_layers: Vec::new(),
            viewport: Viewport::default(),
            selected_id: None,
            add_form: AddAttributeForm::default(),
            save_path: String::new(),
            status: "Ready.".to_string(),
            fitted: false,
            show_basemap: true,
            basemap_cache: BasemapCache::default(),
            show_index: false,
            index_kind: IndexKind::Quadtree,
            show_heatmap: false,
            has_gpu,
            point_size: 5.0,
            points_dirty: false,
            last_selected_id: None,
            last_point_size: 5.0,
            last_layer_count: 0,
            heatmap_opacity: 255,
            gpu_points_buf: Vec::new(),
            spatial_index_split_density: 10000,
            last_split_density: 10000,
            hilbert_order: 6,
            last_hilbert_order: 6,
            load_rx: None,
        }
    }

    fn open_file(&mut self, path: &str) {
        match GisLayer::get_layers(path) {
            Ok(descriptors) if descriptors.is_empty() => {
                self.status = "No layers found in file.".to_string();
            }
            Ok(descriptors) => {
                self.pending_layers = descriptors.into_iter().map(|d| (d, true)).collect();
                self.pending_file = Some(path.to_string());
            }
            Err(e) => {
                self.status = format!("Error reading layers: {e}");
            }
        }
    }

    // UNUSED
    // fn load_pending(&mut self, selected_layer_indices: Vec<usize>) {
    //     let Some(path) = self.pending_file.take() else {
    //         return;
    //     };
    //     self.pending_layers.clear();
    //     match GisLayer::load_selected(&path, &selected_layer_indices) {
    //         Ok(layers) if layers.is_empty() => {
    //             self.status = "No layers loaded.".to_string();
    //         }
    //         Ok(layers) => {
    //             let total: usize = layers.iter().map(|l| l.features.len()).sum();
    //             self.status = format!("Loaded {} layer(s), {} total features", layers.len(), total);
    //             let first_new = self.layers.len();
    //             self.layers.extend(layers.into_iter().map(|l| {
    //                 let name = l.name.clone();
    //                 LayerEntry {
    //                     layer: l,
    //                     visible: true,
    //                     name,
    //                     color: [100, 149, 237],
    //                     opacity: 255,
    //                 }
    //             }));
    //             self.active_layer_idx = Some(first_new);
    //             self.selected_id = None;
    //             self.fitted = false;
    //             self.points_dirty = true;
    //         }
    //         Err(e) => {
    //             self.status = format!("Error: {e}");
    //         }
    //     }
    // }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn pack_color(c: Color32) -> u32 {
    let [r, g, b, a] = c.to_array();
    r as u32 | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

fn collect_gpu_points(
    layers: &[LayerEntry],
    active_idx: Option<usize>,
    selected_id: Option<usize>,
    point_size: f32,
    out: &mut Vec<GpuPoint>,
) {
    out.clear();
    for (i, entry) in layers.iter().enumerate() {
        if !entry.visible
            || match &entry.data {
                LayerKind::Points(point_cloud_layer) => false,
                LayerKind::Vector(gis_layer) => true,
            }
        {
            continue;
        }
        let is_active = active_idx == Some(i);
        let point_cloud_layer = match &entry.data {
            LayerKind::Points(pc) => pc,
            LayerKind::Vector(gis_layer) => panic!("Unexpected layer kind in collect_gpu_points!"),
        };
        for (idx, point) in point_cloud_layer.points.iter().enumerate() {
            let fill = if is_active && selected_id == Some(idx) {
                FILL_SELECTED
            } else {
                FILL_NORMAL
            };
            let packed = pack_color(fill);
            out.push(GpuPoint {
                position: [point[0] as f32, point[1] as f32],
                color: packed,
                size: point_size,
            });
        }
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

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
                        ui.label("Heatmap Opacity:");
                        ui.add(
                            egui::Slider::new(&mut self.spatial_index_split_density, 100..=10000)
                                .step_by(5.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Heatmap Opacity:");
                        ui.add(egui::Slider::new(&mut self.hilbert_order, 1..=12).step_by(1.0));
                    });
                    ui.checkbox(&mut self.show_heatmap, "Quadtree Heatmap");
                    ui.horizontal(|ui| {
                        ui.label("Heatmap Opacity:");
                        ui.add(egui::Slider::new(&mut self.heatmap_opacity, 0..=255).step_by(1.0));
                    });
                    if self.has_gpu {
                        ui.horizontal(|ui| {
                            ui.label("Point size:");
                            ui.add(
                                egui::Slider::new(&mut self.point_size, 1.0..=20.0).step_by(0.5),
                            );
                        });
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("Open…").clicked() {
                        ui.close_kind(UiKind::Menu);
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("GIS files", &["shp", "gpkg", "geojson", "json", "kml"])
                            .add_filter("Shapefile", &["shp"])
                            .add_filter("GeoPackage", &["gpkg"])
                            .add_filter("GeoJSON", &["geojson", "json"])
                            .pick_file()
                        {
                            let path_str = path.to_string_lossy().to_string();
                            self.save_path = path_str.clone();
                            self.open_file(&path_str);
                        }
                    }
                    if ui.button("Quit").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        // ── Layer selector (shown after file pick) ────────────────────────────
        let mut load_indices: Option<Vec<usize>> = None;
        let mut cancel_pending = false;
        if self.pending_file.is_some() {
            egui::Window::new("Select Layers")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label("Choose which layers to load:");
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            for (desc, selected) in &mut self.pending_layers {
                                ui.checkbox(
                                    selected,
                                    format!("{} ({} features)", desc.name, desc.num_features),
                                );
                            }
                        });
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
                        }
                        if ui.button("Cancel").clicked() {
                            cancel_pending = true;
                        }
                    });
                });
        }
        if let Some(indices) = load_indices {
            let (tx, rx) = mpsc::sync_channel::<BatchMessage>(4);
            let path = self.pending_file.take().unwrap();
            self.pending_layers.clear();
            let layers = GisLayer::load_selected_without_features(&path, &indices)
                .expect("Error loading featureless layers!");
            let first_new = self.layers.len();
            // Record which dest indices are point layers before consuming layers
            let is_points: Vec<bool> = layers
                .iter()
                .map(|l| matches!(l.data, LayerKind::Points(_)))
                .collect();
            self.layers.extend(layers.into_iter());
            self.active_layer_idx = Some(first_new);
            self.status = format!("Loading {} layer(s)…", indices.len());
            std::thread::spawn(move || {
                for (pos, file_idx) in indices.into_iter().enumerate() {
                    let dest = first_new + pos;
                    let result = if is_points[pos] {
                        GisLayer::load_point_layer_batched(
                            path.as_str(),
                            file_idx,
                            dest,
                            tx.clone(),
                        )
                    } else {
                        GisLayer::load_layer_batched(path.as_str(), file_idx, dest, tx.clone())
                    };
                    result.expect("Batch layer read failed");
                }
            });
            self.load_rx = Some(rx);
        } else if cancel_pending {
            self.pending_file = None;
            self.pending_layers.clear();
        }

        if let Some(rx) = &self.load_rx {
            for msg in rx.try_iter() {
                match msg {
                    BatchMessage::Points(layer_idx, pts) => {
                        if let LayerKind::Points(pc) = &mut self.layers[layer_idx].data {
                            pc.points.extend(pts);
                            // pc.attributes — commented out until columnar layout is implemented
                        }
                        self.points_dirty = true;
                    }
                    BatchMessage::Vector(layer_idx, features) => {
                        if let LayerKind::Vector(gl) = &mut self.layers[layer_idx].data {
                            gl.features.extend(features);
                        }
                        self.points_dirty = true;
                    }
                }
            }
            if let Err(TryRecvError::Disconnected) = rx.try_recv() {
                self.load_rx = None;
                self.status = "Ready".to_string();
            }
        }

        // ── Status bar ────────────────────────────────────────────────────────
        egui::Panel::bottom("status_bar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        // ── Layer panel (left) ────────────────────────────────────────────────
        egui::Panel::left("layer_panel")
            .exact_size(LAYER_PANEL_WIDTH)
            .show_inside(ui, |ui| {
                ui.heading("Layers");
                ui.separator();
                if self.layers.is_empty() {
                    ui.label("No layers loaded.");
                } else {
                    let mut remove_idx: Option<usize> = None;
                    let mut rebuild_quadtree_idx: Option<usize> = None;
                    let mut rebuild_hilbert_idx: Option<usize> = None;
                    let mut visibility_changed = false;
                    egui::ScrollArea::vertical().show(ui, |ui| {
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
                                        self.fitted = false;
                                        self.points_dirty = true;
                                    }
                                }
                                label_resp.context_menu(|ui| {
                                    if ui.button("Build Quadtree").clicked() {
                                        rebuild_quadtree_idx = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                    if ui.button("Build Hilbert").clicked() {
                                        rebuild_hilbert_idx = Some(i);
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
                    }
                    if let Some(idx) = rebuild_quadtree_idx {
                        let capacity = self.spatial_index_split_density;
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_quadtree(capacity),
                            LayerKind::Vector(gl) => gl.rebuild_quadtree(capacity),
                        }
                    }
                    if let Some(idx) = rebuild_hilbert_idx {
                        let order = self.hilbert_order;
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_hilbert_tree(order),
                            LayerKind::Vector(gl) => gl.rebuild_hilbert_tree(order),
                        }
                    }
                    if visibility_changed {
                        self.points_dirty = true;
                    }
                }
            });

        // ── Sidebar (right) ───────────────────────────────────────────────────
        egui::Panel::right("sidebar")
            .min_size(260.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let action = show_sidebar(
                        ui,
                        &self.layers,
                        self.active_layer_idx,
                        self.selected_id,
                        &mut self.add_form,
                        &mut self.save_path,
                    );

                    match action {
                        SidebarAction::AddAttribute {
                            feature_id,
                            name,
                            value,
                        } => {
                            // COMMENTED OUT FOR NOW BECAUSE THIS BRANCH IS MORE ABOUT VIEWING
                            // if let Some(entry) =
                            //     self.active_layer_idx.and_then(|i| self.layers.get_mut(i))
                            // {
                            //     if !entry.layer.extra_field_names.contains(&name) {
                            //         entry.layer.extra_field_names.push(name.clone());
                            //     }
                            //     entry.layer.features[feature_id]
                            //         .attributes
                            //         .insert(name.clone(), value);
                            //     self.status =
                            //         format!("Added attribute '{name}' to feature #{feature_id}");
                            // }
                        }
                        SidebarAction::SaveAs(path) => {
                            // COMMENTED OUT FOR NOW BECAUSE THIS BRANCH IS MORE ABOUT VIEWING
                            // if let Some(entry) =
                            //     self.active_layer_idx.and_then(|i| self.layers.get(i))
                            // {
                            //     match entry.layer.save(&path) {
                            //         Ok(()) => self.status = format!("Saved to {path}"),
                            //         Err(e) => self.status = format!("Save failed: {e}"),
                            //     }
                            // }
                        }
                        SidebarAction::None => {}
                    }
                });
            });

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
                }
            }
            self.last_split_density = capacity;
            self.points_dirty = true;
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
                }
            }
            self.last_hilbert_order = order;
            self.points_dirty = true;
        }

        let active_layer = self.active_layer_idx.and_then(|i| self.layers.get(i));

        // ── Re-upload GPU points when data or style changes ───────────────────
        if self.has_gpu {
            let layer_changed = self.layers.len() != self.last_layer_count;
            let selection_changed = self.selected_id != self.last_selected_id;
            let size_changed = self.point_size != self.last_point_size;

            if self.points_dirty || layer_changed || selection_changed || size_changed {
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
                            self.point_size,
                            &mut self.gpu_points_buf,
                        );
                        pipeline.upload_points(device, queue, &self.gpu_points_buf);
                    }
                }
                self.points_dirty = false;
                self.last_selected_id = self.selected_id;
                self.last_point_size = self.point_size;
                self.last_layer_count = self.layers.len();
            }
        }

        // ── Map (central panel) ───────────────────────────────────────────────
        CentralPanel::default().show_inside(ui, |ui| {
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

            // CPU geometry: background, basemap, polygons, lines.
            // Points are skipped here when the GPU path is active.
            let render_points = !self.has_gpu;
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
            );

            // GPU point cloud — added AFTER show_map so it composites on top
            // of the background and basemap, but still below the index overlay.
            if self.has_gpu {
                let rect = response.rect;
                let [wx_min, wy_min, wx_max, wy_max] = self.viewport.viewport_bbox(rect);
                let world_size = [(wx_max - wx_min) as f32, (wy_max - wy_min) as f32];
                painter.add(egui::Shape::Callback(
                    egui_wgpu::Callback::new_paint_callback(
                        rect,
                        PointCloudCallback {
                            world_min: [wx_min as f32, wy_min as f32],
                            world_size,
                            screen_min: [rect.left(), rect.top()],
                            screen_size: [rect.width(), rect.height()],
                        },
                    ),
                ));
            }

            // Detect selection change driven by click inside show_map so we
            // re-upload colors on the next frame.
            if self.selected_id != self.last_selected_id {
                self.points_dirty = true;
            }

            if self.show_heatmap {
                let heatmap = active_layer
                    .map(|e| {
                        e.data
                            .index(IndexKind::Quadtree)
                            .map(|index| HeatmapLayer::build_from_spatial_index(index))
                    })
                    .unwrap_or(Some(HeatmapLayer { cells: vec![] }))
                    .unwrap_or(HeatmapLayer { cells: vec![] });
                show_quadtree_heatmap(
                    &painter,
                    &heatmap,
                    &self.viewport,
                    response.rect,
                    self.heatmap_opacity,
                );
            }

            if self.show_index {
                let index = active_layer
                    .map(|e| e.data.index(self.index_kind))
                    .flatten();
                show_spatial_index_grid(&painter, index, &mut self.viewport, response.rect);
            }
        });
    }
}
