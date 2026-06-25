mod loader;
mod ui;

use std::sync::mpsc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use futures_channel::oneshot;
use egui::Rect;

use crate::basemap::BasemapCache;
use crate::filter::LayerAttributeFilter;
use crate::gis_layer::{BatchMessage, LayerEntry};
#[cfg(target_arch = "wasm32")]
use crate::gis_reader::FgbReaderCache;
use crate::gis_reader::{GisFilePath, LayerDescriptor};
use crate::histogram::{FieldStats, HistogramState};
use crate::map_view::Viewport;
use crate::point_cloud::{GpuPoint, PointCloudPipeline};
use crate::sidebar::AddAttributeForm;
use crate::spatial_index::IndexKind;
use crate::uncertainty_quadtree::{MeasurementType, UncertaintyMeasure};

pub const LAYER_PANEL_WIDTH: f32 = 180.0;

#[derive(Default, PartialEq)]
pub(super) enum LoadMode {
    #[default]
    GeometryOnly,
    WithAttributes,
}

#[derive(PartialEq)]
pub enum ClickTarget {
    Feature,
    GridCell,
}

#[cfg(target_arch = "wasm32")]
pub(super) fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        * 1000.0
}

pub struct GisEditorApp {
    pub(super) layers: Vec<LayerEntry>,
    pub(super) active_layer_idx: Option<usize>,
    pub(super) layer_picker_window_open: bool,
    pub(super) viewport: Viewport,
    pub(super) selected_id: Option<usize>,
    pub(super) file_pick_rx: Option<mpsc::Receiver<GisFilePath>>,
    #[cfg(target_arch = "wasm32")]
    pub(super) pending_bytes: Option<Vec<u8>>,
    #[cfg(target_arch = "wasm32")]
    pub(super) file_handle_slot: std::rc::Rc<std::cell::RefCell<Option<web_sys::File>>>,
    pub(super) add_form: AddAttributeForm,
    pub(super) save_path: String,
    pub(super) status: String,
    pub(super) fitted: bool,
    pub(super) show_basemap: bool,
    pub(super) basemap_cache: BasemapCache,
    pub(super) show_index: bool,
    pub(super) index_kind: IndexKind,
    pub(super) show_heatmap: bool,
    pub(super) click_target: ClickTarget,

    pub(super) pending_file: Option<GisFilePath>,
    pub(super) pending_file_descriptor: Option<LayerDescriptor>,
    pub(super) pending_layers: Vec<(LayerDescriptor, bool)>,
    pub(super) pending_load_mode: LoadMode,
    pub(super) pending_field_selection: Vec<(String, bool)>,
    pub(super) load_rx: Option<mpsc::Receiver<BatchMessage>>,
    pub(super) load_layer_descriptor_rx: mpsc::Receiver<LayerDescriptor>,
    pub(super) load_layer_descriptor_tx: mpsc::SyncSender<LayerDescriptor>,
    pub(super) viewport_stable_frames: u32,
    pub(super) viewport_load_pending: bool,

    pub(super) has_gpu: bool,
    pub point_size: f32,
    pub(super) points_dirty: bool,
    pub(super) last_selected_id: Option<usize>,
    pub(super) last_point_size: f32,
    pub(super) last_layer_count: usize,
    pub(super) heatmap_opacity: u8,
    pub(super) gpu_points_buf: Vec<GpuPoint>,
    pub(super) spatial_index_split_density: usize,
    pub(super) last_split_density: usize,
    pub(super) hilbert_order: u32,
    pub(super) last_hilbert_order: u32,
    pub(super) last_viewport_center: [f64; 2],
    pub(super) last_viewport_ppu: f64,
    pub(super) last_canvas_rect: Option<Rect>,
    pub(super) selected_uncertainty_attribute: Option<String>,
    pub(super) selected_index_cell_data: Option<UncertaintyMeasure>,
    pub(super) uncertainty_split_threshold: f32,
    pub(super) viewport_culling: bool,
    pub(super) selected_split_measurement_type: MeasurementType,
    pub(super) fgb_file_url: String,
    pub(super) cancel_stream: Arc<AtomicBool>,
    pub(super) streaming_features: bool,
    #[cfg(target_arch = "wasm32")]
    pub(super) fgb_reader_cache: FgbReaderCache,
    pub(super) current_filters: Vec<LayerAttributeFilter>,
    pub(super) adding_filter: Option<LayerAttributeFilter>,
    pub(super) updated_filters: bool,
    pub(super) filtered_idx_rx: Option<oneshot::Receiver<(usize, Vec<u32>)>>,
    pub(super) histogram: Option<HistogramState>,
    pub(super) show_histogram: bool,
    pub(super) histogram_field: String,
    pub(super) field_stats: Option<FieldStats>,
    pub(super) last_stats_field: String,
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
            histogram: None,
            show_histogram: false,
            histogram_field: String::new(),
            field_stats: None,
            last_stats_field: String::new(),
        }
    }
}
