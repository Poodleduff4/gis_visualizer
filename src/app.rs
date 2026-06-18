use bitvec::vec::BitVec;
use bitvec::{bitarr, bitvec, BitArr};
use egui::{CentralPanel, Color32, UiKind};
use futures_channel::oneshot;
use rfd::FileHandle;
use rstar::primitives::GeomWithData;
use std::cell::Cell;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use std::time::Duration;
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[cfg(not(target_arch = "wasm32"))]
type PendingFile = String;
#[cfg(target_arch = "wasm32")]
type PendingFile = (Vec<u8>, String);

#[derive(Default, PartialEq)]
enum LoadMode {
    #[default]
    GeometryOnly,
    WithAttributes,
}
use wgpu::naga::proc::vector_size_str;

use crate::basemap::BasemapCache;
use crate::gis_layer::{AttributeValue, BatchMessage, GisLayer, LayerEntry, LayerKind};
#[cfg(target_arch = "wasm32")]
use crate::gis_reader::FgbReaderCache;
use crate::gis_reader::{GeoParquetReader, GisFilePath, GisReader, LayerDescriptor};
use crate::heatmap::HeatmapLayer;
use crate::map_view::{show_map, show_quadtree_heatmap, show_spatial_index_grid, Viewport};
use crate::parquet::{extract_batch_as_u32, extract_u32, query_parquet};
use crate::point_cloud::{GpuPoint, PointCloudCallback, PointCloudPipeline};
use crate::quadtree::Quadtree;
use crate::sidebar::{show_sidebar, AddAttributeForm, SidebarAction};
use crate::spatial_index::{IndexKind, SpatialIndex};
use crate::uncertainty_quadtree::{MeasurementType, UncertaintyMeasure, UncertaintyMeasurement};

const LAYER_PANEL_WIDTH: f32 = 180.0;

const FILL_NORMAL: Color32 = Color32::from_rgb(100, 149, 237);
const FILL_SELECTED: Color32 = Color32::from_rgb(255, 165, 0);
#[derive(PartialEq)]
pub enum ClickTarget {
    Feature,
    GridCell,
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() // returns f64 milliseconds
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        * 1000.0
}
#[derive(PartialEq, Clone)]
pub enum FilterOperation {
    LessThan,
    GreaterThan,
    Equal,
}
impl fmt::Display for FilterOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterOperation::Equal => write!(f, "="),
            FilterOperation::GreaterThan => write!(f, ">"),
            FilterOperation::LessThan => write!(f, "<"),
        }
    }
}

pub struct LayerAttributeFilter {
    pub attribute: Option<String>,
    pub operation: Option<FilterOperation>,
    pub comparitor: AttributeValue,
    pub comparitor_raw: String,
}

pub struct GisEditorApp {
    layers: Vec<LayerEntry>,
    active_layer_idx: Option<usize>,
    layer_picker_window_open: bool,
    viewport: Viewport,
    selected_id: Option<usize>,
    file_pick_rx: Option<mpsc::Receiver<GisFilePath>>,
    #[cfg(target_arch = "wasm32")]
    pending_bytes: Option<Vec<u8>>,
    #[cfg(target_arch = "wasm32")]
    file_handle_slot: std::rc::Rc<std::cell::RefCell<Option<web_sys::File>>>,
    add_form: AddAttributeForm,
    save_path: String,
    status: String,
    fitted: bool,
    show_basemap: bool,
    basemap_cache: BasemapCache,
    show_index: bool,
    index_kind: IndexKind,
    show_heatmap: bool,
    click_target: ClickTarget,

    // Layer selector state (populated after file pick, before load)
    pending_file: Option<GisFilePath>,
    pending_file_descriptor: Option<LayerDescriptor>,
    pending_layers: Vec<(LayerDescriptor, bool)>,
    pending_load_mode: LoadMode,
    pending_field_selection: Vec<(String, bool)>,
    load_rx: Option<mpsc::Receiver<BatchMessage>>,
    // load_tx: Option<mpsc::SyncSender<BatchMessage>>,
    load_layer_descriptor_rx: mpsc::Receiver<LayerDescriptor>,
    load_layer_descriptor_tx: mpsc::SyncSender<LayerDescriptor>,
    viewport_stable_frames: u32,
    viewport_load_pending: bool,

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
    last_viewport_center: [f64; 2],
    last_viewport_ppu: f64,
    last_canvas_rect: Option<egui::Rect>,
    selected_uncertainty_attribute: Option<String>,
    selected_index_cell_data: Option<UncertaintyMeasure>,
    uncertainty_split_threshold: f32,
    viewport_culling: bool,
    selected_split_measurement_type: MeasurementType,
    fgb_file_url: String,
    cancel_stream: Arc<AtomicBool>,
    streaming_features: bool,
    #[cfg(target_arch = "wasm32")]
    fgb_reader_cache: FgbReaderCache,
    current_filters: Vec<LayerAttributeFilter>,
    adding_filter: Option<LayerAttributeFilter>,
    updated_filters: bool,
    filtered_idx_rx: Option<oneshot::Receiver<(usize, Vec<u32>)>>,
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
        // let (tx, rx) = mpsc::sync_channel::<BatchMessage>(10);
        let (ld_tx, ld_rx) = mpsc::sync_channel::<LayerDescriptor>(1);

        GisEditorApp {
            layers: Vec::new(),
            active_layer_idx: None,
            layer_picker_window_open: false,
            pending_file: None,
            pending_layers: Vec::new(),
            pending_load_mode: LoadMode::default(),
            pending_field_selection: Vec::new(),
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
            click_target: ClickTarget::GridCell,
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
            // load_tx: tx,
            load_layer_descriptor_rx: ld_rx,
            load_layer_descriptor_tx: ld_tx,
            last_viewport_center: Default::default(),
            last_viewport_ppu: Default::default(),
            last_canvas_rect: None,
            selected_uncertainty_attribute: None,
            selected_index_cell_data: None,
            uncertainty_split_threshold: 0.5,
            viewport_culling: false,
            selected_split_measurement_type: MeasurementType::Variance,
            file_pick_rx: None,
            #[cfg(target_arch = "wasm32")]
            pending_bytes: None,
            #[cfg(target_arch = "wasm32")]
            file_handle_slot: std::rc::Rc::new(std::cell::RefCell::new(None)),
            fgb_file_url: "http://localhost:8001/".to_string(),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            streaming_features: false,
            pending_file_descriptor: None,
            viewport_stable_frames: 0,
            viewport_load_pending: false,
            #[cfg(target_arch = "wasm32")]
            fgb_reader_cache: std::rc::Rc::new(std::cell::RefCell::new(
                std::collections::HashMap::new(),
            )),
            current_filters: Vec::new(),
            adding_filter: None,
            updated_filters: false,
            filtered_idx_rx: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_file(&mut self, path: GisFilePath) {
        match GisReader::load_layer_descriptor(&path.to_string()) {
            Ok(descriptor) => self.apply_layer(descriptor, path),
            Err(e) => self.status = format!("Error reading layers: {e}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn open_file(
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

    fn apply_layer(&mut self, descriptor: LayerDescriptor, file_path: GisFilePath) {
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
    _viewport_bbox: Option<[f64; 4]>,
    point_size: f32,
    out: &mut Vec<GpuPoint>,
) {
    out.clear();
    for (i, entry) in layers.iter().enumerate() {
        if !entry.visible
            || match &entry.data {
                LayerKind::Points(_) => false,
                LayerKind::Vector(_) => true,
            }
        {
            continue;
        }
        let is_active = active_idx == Some(i);
        let point_cloud_layer = match &entry.data {
            LayerKind::Points(pc) => pc,
            LayerKind::Vector(_) => panic!("Unexpected layer kind in collect_gpu_points!"),
        };
        let visible_points = point_cloud_layer
            .points
            .iter()
            .enumerate()
            .filter(|(i, (_pi, _pv))| point_cloud_layer.filter_mask[*i])
            .map(|(_, (_pi, pv))| *pv)
            .collect::<Vec<[f64; 2]>>();
        // let render_pts: &[[f64; 2]] = if !point_cloud_layer.viewport_mask.is_empty() {
        //     &point_cloud_layer.viewport_mask
        // } else {
        //     point_cloud_layer
        //         .points
        //         .iter()
        //         .map(|(i, p)| p)
        //         .collect::<&Vec<&[f64; 2]>>()
        // };
        for (idx, &point) in visible_points.iter().enumerate() {
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
                                self.selected_split_measurement_type = MeasurementType::Variance
                            }
                            if ui.button("Kernel-Density Entropy").clicked() {
                                self.selected_split_measurement_type =
                                    MeasurementType::KernalDensity
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
                                    .add_filter("FlatGeobuf", &["fgb"])
                                    .add_filter("GeoParquet", &["parquet"])
                                    .add_filter("All Supported", &["fgb", "parquet"])
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
                    if ui.button("Quit").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
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

        // ── Layer selector (shown after file pick) ────────────────────────────
        let mut load_indices: Option<Vec<usize>> = None;
        let mut cancel_pending = false;
        if self.pending_file.is_some() {
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
            // let header_bytes = self.pending_bytes.take().unwrap_or_default();
            #[cfg(target_arch = "wasm32")]
            // let wasm_file = self.file_handle_slot.borrow_mut().take();
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
            let fgb_file_clone = self.fgb_file_url.clone();

            let rect_clone = self
                .viewport
                .viewport_bbox(self.last_canvas_rect.clone().unwrap());
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
                    let _ = result; // SendError just means receiver was dropped (new load started)
                }
            });
            #[cfg(target_arch = "wasm32")]
            spawn_local(async move {
                // let file = wasm_file.expect("no file handle for batch load");
                // let ab = wasm_bindgen_futures::JsFuture::from(file.array_buffer())
                //     .await
                //     .expect("failed to read file bytes");
                // let bytes: std::sync::Arc<[u8]> =
                //     std::sync::Arc::from(js_sys::Uint8Array::new(&ab).to_vec());
                for (pos, file_idx) in indices.into_iter().enumerate() {
                    let dest = first_new + pos;
                    let result = if is_points[pos] {
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
                        // GisReader::load_layer_batched(
                        //     bytes.clone(),
                        //     file_idx,
                        //     dest,
                        //     load_tx.clone(),
                        //     attr_fields.clone(),
                        // )
                        // .await
                        Ok(())
                    };
                    let _ = result;
                }
            });
            // self.load_rx = Some(rx);
        } else if cancel_pending {
            self.pending_file = None;
            self.pending_layers.clear();
            self.pending_field_selection.clear();
        }
        if let Some(load_rx) = &self.load_rx {
            for msg in load_rx.try_iter() {
                match msg {
                    BatchMessage::Points(layer_idx, pts, named_cols) => {
                        if let Some(LayerKind::Points(pc)) =
                            &mut self.layers.get_mut(layer_idx).map(|l| &mut l.data)
                        {
                            std::sync::Arc::make_mut(&mut pc.points).extend(pts);
                            if pc.attributes.is_empty() && !named_cols.is_empty() {
                                for (name, col) in named_cols {
                                    // pc.field_names.push(name);
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
                        // viewport_mask no longer drives GPU rendering; no points_dirty needed
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
            // Keep egui polling so the channel is drained every frame.
            // Without this, egui sleeps when there's no input and the
            // bounded channel fills up, blocking the stream future.
            ui.ctx().request_repaint();
            if let Err(TryRecvError::Disconnected) = load_rx.try_recv() {
                self.status = "Ready".to_string();
                self.load_rx = None;
                self.streaming_features = false;
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
                    let mut rebuild_rtree_idx: Option<usize> = None;
                    let mut rebuild_uncertainty_quadtree_idx: Option<usize> = None;
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
                    if let Some(idx) = rebuild_rtree_idx {
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_rtree(),
                            LayerKind::Vector(_) => {}
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
                        }
                        self.fitted = true;
                        self.viewport_load_pending = true;
                        self.viewport_stable_frames = 0;
                    }
                    if let Some(idx) = rebuild_uncertainty_quadtree_idx {
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => {
                                if let Some(attr) = &self.selected_uncertainty_attribute {
                                    pc.rebuild_uncertainty_quadtree(
                                        attr.clone(),
                                        self.uncertainty_split_threshold,
                                        self.selected_split_measurement_type.clone(),
                                    );
                                }
                            }
                            LayerKind::Vector(gl) => {}
                        }
                    }
                    if let Some(idx) = rebuild_hilbert_idx {
                        let order = self.hilbert_order;
                        match &mut self.layers[idx].data {
                            LayerKind::Points(pc) => pc.rebuild_hilbert_tree(order),
                            LayerKind::Vector(gl) => gl.rebuild_hilbert_tree(order),
                        }
                        self.fitted = true;
                        self.viewport_load_pending = true;
                        self.viewport_stable_frames = 0;
                    }
                    if visibility_changed {
                        self.points_dirty = true;
                    }
                }
            });

        // ── Sidebar (right) ───────────────────────────────────────────────────
        let mut layer_filters = self.active_layer_idx.map(|i| &mut self.layers[i].filters);
        egui::Panel::right("sidebar")
            .min_size(260.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let action = show_sidebar(
                        ui,
                        &mut self.layers,
                        self.active_layer_idx,
                        self.selected_id,
                        &mut self.add_form,
                        &mut self.save_path,
                        self.selected_index_cell_data.as_ref(),
                        // &mut layer_filters,
                        &mut self.adding_filter,
                        &mut self.updated_filters,
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

        if self.updated_filters {
            let layer = &mut self.layers[self.active_layer_idx.unwrap()];
            let idx = self.active_layer_idx.unwrap().clone();
            match layer.filters.len() {
                0 => {
                    layer.data.reset_filter_mask();
                    self.points_dirty = true;
                    self.updated_filters = false;
                    // ui.request_repaint();
                }
                _ => {
                    let query = format!(
                        "SELECT idx,{} from layer WHERE {}",
                        layer.data.field_names().join(","),
                        layer
                            .filters
                            .iter()
                            .map(|f| {
                                format!(
                                    "{} {} {}",
                                    f.attribute.clone().unwrap(),
                                    f.operation.clone().unwrap().to_string(),
                                    f.comparitor_raw
                                )
                            })
                            .collect::<Vec<String>>()
                            .join(" and ")
                    );
                    println!("{}", query);
                    let (tx, rx) = oneshot::channel::<(usize, Vec<u32>)>();
                    self.filtered_idx_rx = Some(rx);
                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async {
                            let matching_ids =
                                match query_parquet("./assets/output.parquet", query).await {
                                    Ok(batch_vec) => {
                                        println!("Ok Query!");
                                        batch_vec
                                            .iter()
                                            .filter_map(|b| extract_batch_as_u32(b, "idx"))
                                            .flatten()
                                            .collect::<Vec<u32>>()
                                    }
                                    Err(e) => {
                                        println!("{:?}", e);
                                        Vec::new()
                                    }
                                };
                            println!("Thread DOne!");
                            let _ = tx.send((idx, matching_ids));
                        });
                    });
                    self.updated_filters = false;
                }
            };
        }
        if let Some(rx) = &mut self.filtered_idx_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, idx_vec))) => {
                    println!("{}", idx_vec.len());
                    if let Some(l) = self.layers.get_mut(layer_idx) {
                        match &mut l.data {
                            LayerKind::Points(point_cloud_layer) => {
                                // idx_vec contains parquet idx column values, NOT enumerate positions.
                                // Build a HashSet of matching parquet ids, then scan pc.points to find
                                // their enumerate positions for filter_mask.
                                let matching: std::collections::HashSet<u32> =
                                    idx_vec.into_iter().collect();
                                let mut mask: BitVec = bitvec![0;point_cloud_layer.points.len()];
                                for (pos, (parquet_id, _)) in
                                    point_cloud_layer.points.iter().enumerate()
                                {
                                    if matching.contains(parquet_id) {
                                        mask.set(pos, true);
                                    }
                                }
                                point_cloud_layer.filter_mask &= mask;
                                self.points_dirty = true;
                                ui.request_repaint();
                            }
                            LayerKind::Vector(_) => {}
                        }
                    }
                }
                Ok(None) => {
                    println!("Not Ready Yet")
                }
                Err(e) => self.filtered_idx_rx = None,
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
            } else if self.viewport_load_pending {
                self.viewport_stable_frames += 1;
            }
            if self.viewport_load_pending {
                ui.ctx().request_repaint();
            }

            let viewport_reload_ready =
                self.viewport_load_pending && self.viewport_stable_frames >= 3;
            if (self.points_dirty
                || layer_changed
                || selection_changed
                || size_changed
                || viewport_reload_ready)
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
                // println!("Reloading From Index!");
                self.viewport_load_pending = false;
                // Cancel previous quad threads before starting new ones.
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
                        pc.viewport_points.clear();
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

        let active_layer = self.active_layer_idx.and_then(|i| self.layers.get(i));
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
            self.last_canvas_rect = Some(response.rect);

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
                &self.click_target,
                &mut self.selected_index_cell_data,
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
