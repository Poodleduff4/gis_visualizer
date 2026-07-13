mod loader;
mod plot_style;
mod ui;
mod ui_layer_panel;
mod ui_loading;
mod ui_map;
mod ui_menu;
#[cfg(not(target_arch = "wasm32"))]
mod plugin_bridge;
#[cfg(not(target_arch = "wasm32"))]
mod ui_plugins;
mod ui_sidebar;
mod ui_windows;

use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::Arc;

use egui::Rect;
use futures_channel::oneshot;

use crate::basemap::BasemapCache;
use crate::filter::LayerAttributeFilter;
use crate::gis_layer::{BatchMessage, LayerEntry, LayerKind};
#[cfg(target_arch = "wasm32")]
use crate::gis_reader::FgbReaderCache;
use crate::gis_reader::{GisFilePath, LayerDescriptor};
use crate::globe::{GlobeCamera, GlobePipeline, GlobePoint};
use crate::histogram::{BivariateStats, FieldStats, HistogramState, LisaPoint};
use crate::map_view::Viewport;
use crate::selection_stats::{
    SelectionBivariate, SelectionFieldStats, SelectionHistogram,
};
use crate::point_cloud::{GpuPoint, PointCloudPipeline};
use crate::vector_gpu::{GpuVertex, VectorPipeline};
use crate::raster_reader::RasterDescriptor;
use crate::sidebar::AddAttributeForm;
#[cfg(not(target_arch = "wasm32"))]
use crate::snapshot::{
    filter_logic_to_str, filter_snapshot_to_filter, filter_to_snapshot, str_to_filter_logic,
    AnalysisSnapshot, AppSnapshot, DisplaySnapshot, LayerSnapshot, PendingSnapshotRestore,
    ViewportSnapshot,
};
use crate::uncertainty_quadtree::{MeasurementType, UncertaintyMeasure};

pub const LAYER_PANEL_WIDTH: f32 = 180.0;

#[derive(Clone, PartialEq, Default)]
pub enum MapView {
    #[default]
    Flat,
    Globe,
}

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
    HeatmapRoi,
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
    pub(super) roi_rebuild_pending: bool,
    pub(super) click_target: ClickTarget,
    pub(super) select_mode: bool,
    pub(super) select_drag_start: Option<egui::Pos2>,

    pub(super) pending_file: Option<GisFilePath>,
    pub(super) pending_file_descriptor: Option<LayerDescriptor>,
    /// (descriptor, selected, convert-to-WGS84) per pending layer awaiting load.
    pub(super) pending_layers: Vec<(LayerDescriptor, bool, bool)>,
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
    pub(super) gpu_vector_fill_verts_buf: Vec<GpuVertex>,
    pub(super) gpu_vector_fill_indices_buf: Vec<u32>,
    pub(super) gpu_vector_line_verts_buf: Vec<GpuVertex>,
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
    pub(super) uncertainty_max_depth: usize,
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
    pub(super) bivariate: Option<BivariateStats>,
    pub(super) show_bivariate: bool,
    pub(super) bivariate_y_field: String,
    pub(super) selection_field_a: String,
    pub(super) selection_field_b: String,
    pub(super) selection_bivariate: Option<SelectionBivariate>,
    pub(super) selection_histogram: Option<SelectionHistogram>,
    pub(super) selection_field_stats: Option<SelectionFieldStats>,
    /// Counts down from N after any map-relevant change; GPU callback runs while > 0 or cursor is in map.
    pub(super) map_render_ttl: u32,
    pub(super) color_picker_layer: Option<usize>,

    pub(super) spatial_field: String,
    pub(super) spatial_radius: f64,
    pub(super) local_variance_results: Option<Vec<Option<f64>>>,
    pub(super) show_local_variance: bool,
    pub(super) local_variance_rx: Option<oneshot::Receiver<Vec<Option<f64>>>>,
    pub(super) lisa_results: Option<Vec<Option<LisaPoint>>>,
    pub(super) show_lisa: bool,
    pub(super) lisa_rx: Option<oneshot::Receiver<Option<Vec<Option<LisaPoint>>>>>,

    // ── Kernel density estimation (grid-based heatmap, à la QGIS) ─────────
    pub(super) kde_window_open: bool,
    pub(super) kde_cell_size: f64,
    pub(super) kde_radius: f64,
    pub(super) kde_kernel: crate::kde::KdeKernel,
    pub(super) kde_weight_field: Option<String>,
    pub(super) kde_running: bool,
    pub(super) kde_rx: Option<
        oneshot::Receiver<(usize, crate::heatmap::HeatmapLayer, crate::heatmap::SavedHeatmap)>,
    >,

    pub(super) export_window_open: bool,

    // ── Globe view + raster ──────────────────────────────────────────────
    pub(super) map_view: MapView,
    pub(super) globe_camera: GlobeCamera,
    pub(super) globe_points_buf: Vec<GlobePoint>,
    pub(super) globe_points_dirty: bool,
    /// Consumed by the globe's GPU raster texture upload.
    pub(super) raster_dirty: bool,
    /// Consumed independently by the flat map's CPU raster texture cache —
    /// separate from `raster_dirty` so switching views doesn't miss a rebake
    /// that the other view already consumed this frame.
    pub(super) flat_raster_dirty: bool,
    pub(super) raster_texture: Option<egui::TextureHandle>,
    pub(super) raster_descriptor_rx: Option<mpsc::Receiver<RasterDescriptor>>,
    pub(super) pending_raster_descriptor: Option<RasterDescriptor>,
    pub(super) raster_load_rx: Option<mpsc::Receiver<Result<LayerEntry, String>>>,

    /// Auto-cycles the active raster layer's `Single` band on a timer, like
    /// repeatedly picking the next entry in the band dropdown.
    pub(super) raster_playback_enabled: bool,
    pub(super) raster_playback_interval_secs: f32,
    pub(super) raster_playback_last_tick_ms: f64,

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) snapshot_restore: Option<PendingSnapshotRestore>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) snapshot_pick_rx: Option<std::sync::mpsc::Receiver<std::path::PathBuf>>,

    // ── Plugins (subprocess + msgpack protocol; native-only) ──────────────
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) plugin_window_open: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) plugins_dir: std::path::PathBuf,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) available_plugins: Vec<crate::plugin::PluginManifest>,
    /// Name of the plugin currently running, if any — drives the spinner
    /// and disables concurrent runs (one plugin process at a time for now).
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) plugin_running: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) plugin_events_rx: Option<std::sync::mpsc::Receiver<crate::plugin::PluginEvent>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) plugin_log: Vec<(crate::plugin::LogLevel, String)>,
    /// Current values for each plugin's `[[params]]`, keyed by plugin name
    /// then param name. Populated (from each param's `default`) the first
    /// time a plugin with params is shown in the list.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) plugin_param_values:
        std::collections::HashMap<String, std::collections::HashMap<String, ui_plugins::ParamEditValue>>,
}

impl GisEditorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let has_gpu = if let Some(wrs) = cc.wgpu_render_state.as_ref() {
            let pipeline = PointCloudPipeline::new(&wrs.device, wrs.target_format);
            let vector_pipeline = VectorPipeline::new(&wrs.device, wrs.target_format);
            let globe_pipeline = GlobePipeline::new(&wrs.device, &wrs.queue, wrs.target_format);
            let mut renderer = wrs.renderer.write();
            renderer.callback_resources.insert(pipeline);
            renderer.callback_resources.insert(vector_pipeline);
            renderer.callback_resources.insert(globe_pipeline);
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
            roi_rebuild_pending: false,
            click_target: ClickTarget::GridCell,
            select_mode: false,
            select_drag_start: None,
            has_gpu,
            point_size: 5.0,
            points_dirty: false,
            last_selected_id: None,
            last_point_size: 5.0,
            last_layer_count: 0,
            heatmap_opacity: 255,
            gpu_points_buf: Vec::new(),
            gpu_vector_fill_verts_buf: Vec::new(),
            gpu_vector_fill_indices_buf: Vec::new(),
            gpu_vector_line_verts_buf: Vec::new(),
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
            uncertainty_split_threshold: 1.0,
            uncertainty_max_depth: 12,
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
            bivariate: None,
            show_bivariate: false,
            bivariate_y_field: String::new(),
            selection_field_a: String::new(),
            selection_field_b: String::new(),
            selection_bivariate: None,
            selection_histogram: None,
            selection_field_stats: None,
            map_render_ttl: 0,
            color_picker_layer: None,
            spatial_field: String::new(),
            spatial_radius: 0.01,
            local_variance_results: None,
            show_local_variance: false,
            local_variance_rx: None,
            lisa_results: None,
            show_lisa: false,
            lisa_rx: None,
            kde_window_open: false,
            kde_cell_size: 0.01,
            kde_radius: 0.05,
            kde_kernel: crate::kde::KdeKernel::Quartic,
            kde_weight_field: None,
            kde_running: false,
            kde_rx: None,
            export_window_open: false,
            map_view: MapView::default(),
            globe_camera: GlobeCamera::default(),
            globe_points_buf: Vec::new(),
            globe_points_dirty: false,
            raster_dirty: false,
            flat_raster_dirty: false,
            raster_texture: None,
            raster_descriptor_rx: None,
            pending_raster_descriptor: None,
            raster_load_rx: None,
            raster_playback_enabled: false,
            raster_playback_interval_secs: 1.0,
            raster_playback_last_tick_ms: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            snapshot_restore: None,
            #[cfg(not(target_arch = "wasm32"))]
            snapshot_pick_rx: None,
            #[cfg(not(target_arch = "wasm32"))]
            plugin_window_open: false,
            #[cfg(not(target_arch = "wasm32"))]
            plugins_dir: std::path::PathBuf::from("plugins"),
            #[cfg(not(target_arch = "wasm32"))]
            available_plugins: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            plugin_running: None,
            #[cfg(not(target_arch = "wasm32"))]
            plugin_events_rx: None,
            #[cfg(not(target_arch = "wasm32"))]
            plugin_log: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            plugin_param_values: std::collections::HashMap::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GisEditorApp {
    pub(super) fn capture_snapshot(&self) -> AppSnapshot {
        let layers: Vec<LayerSnapshot> = self
            .layers
            .iter()
            .map(|le| {
                let (quadtree_capacity, hilbert_order, built_rtree, uncertainty) = match &le.data {
                    LayerKind::Vector(gl) => {
                        let quadtree_capacity =
                            gl.quadtree.as_ref().and_then(|si| si.get_capacity());
                        let hilbert_order = match &gl.hilbert {
                            Some(crate::spatial_index::SpatialIndex::HilbertCurve(ht)) => {
                                Some(ht.get_order())
                            }
                            _ => None,
                        };
                        (quadtree_capacity, hilbert_order, false, None)
                    }
                    LayerKind::Points(pc) => match pc.index.as_deref() {
                        Some(crate::spatial_index::SpatialIndex::Quadtree(qt)) => {
                            (qt.get_capacity(), None, false, None)
                        }
                        Some(crate::spatial_index::SpatialIndex::HilbertCurve(ht)) => {
                            (None, Some(ht.get_order()), false, None)
                        }
                        Some(crate::spatial_index::SpatialIndex::RTree(_)) => {
                            (None, None, true, None)
                        }
                        Some(crate::spatial_index::SpatialIndex::UncertaintyQuadtree(uq)) => (
                            None,
                            None,
                            false,
                            Some(crate::snapshot::UncertaintySnapshot {
                                attribute: uq.attribute().to_string(),
                                threshold: uq.threshold(),
                                measurement_type: crate::snapshot::measurement_type_to_str(
                                    &uq.measurement_type(),
                                ),
                                max_depth: uq.max_depth(),
                            }),
                        ),
                        None => (None, None, false, None),
                    },
                    LayerKind::Raster(_) => (None, None, false, None),
                };
                LayerSnapshot {
                    file_path: le.descriptor.location.to_string(),
                    is_raster: matches!(le.data, LayerKind::Raster(_)),
                    selected_attributes: le.data.field_names(),
                    name: le.name.clone(),
                    visible: le.visible,
                    color: le.color,
                    opacity: le.opacity,
                    filter_logic: filter_logic_to_str(le.filter_logic),
                    filters: le.filters.iter().filter_map(filter_to_snapshot).collect(),
                    quadtree_capacity,
                    hilbert_order,
                    built_rtree,
                    uncertainty,
                    selections: le
                        .selections
                        .iter()
                        .map(|s| crate::snapshot::SelectionSnapshot {
                            name: s.name.clone(),
                            bbox: s.bbox,
                        })
                        .collect(),
                    active_selection: le.active_selection,
                    show_index: le.show_index,
                    index_kind: crate::snapshot::index_kind_to_str(&le.index_kind),
                    show_heatmap: le.show_heatmap,
                    heatmap_metric: crate::snapshot::heatmap_metric_to_str(&le.heatmap_metric),
                    show_points: le.show_points,
                }
            })
            .collect();

        AppSnapshot {
            viewport: ViewportSnapshot {
                center: self.viewport.center,
                pixels_per_unit: self.viewport.pixels_per_unit,
            },
            display: DisplaySnapshot {
                show_basemap: self.show_basemap,
                point_size: self.point_size,
                heatmap_opacity: self.heatmap_opacity,
            },
            analysis: AnalysisSnapshot {
                active_layer_idx: self.active_layer_idx,
                histogram_field: self.histogram_field.clone(),
                show_histogram: self.show_histogram,
                bivariate_y_field: self.bivariate_y_field.clone(),
                show_bivariate: self.show_bivariate,
                spatial_field: self.spatial_field.clone(),
                spatial_radius: self.spatial_radius,
                show_lisa: self.show_lisa,
                show_local_variance: self.show_local_variance,
            },
            layers,
        }
    }

    pub(super) fn apply_snapshot_progress(&mut self) {
        if self.snapshot_restore.is_none() {
            return;
        }

        // Apply settings from the layer we just finished loading.
        if let Some(layer_snap) = self
            .snapshot_restore
            .as_mut()
            .unwrap()
            .pending_layer_settings
            .take()
        {
            let has_filters = !layer_snap.filters.is_empty();
            if let Some(layer) = self.layers.last_mut() {
                layer.visible = layer_snap.visible;
                layer.color = layer_snap.color;
                layer.opacity = layer_snap.opacity;
                layer.filter_logic = str_to_filter_logic(&layer_snap.filter_logic);
                layer.filters = layer_snap
                    .filters
                    .iter()
                    .map(filter_snapshot_to_filter)
                    .collect();
                match &mut layer.data {
                    LayerKind::Vector(gl) => {
                        if let Some(cap) = layer_snap.quadtree_capacity {
                            gl.rebuild_quadtree(cap);
                        }
                        if let Some(order) = layer_snap.hilbert_order {
                            gl.rebuild_hilbert_tree(order);
                        }
                    }
                    LayerKind::Points(pc) => {
                        if let Some(cap) = layer_snap.quadtree_capacity {
                            pc.rebuild_quadtree(cap);
                        } else if let Some(order) = layer_snap.hilbert_order {
                            pc.rebuild_hilbert_tree(order);
                        } else if layer_snap.built_rtree {
                            pc.rebuild_rtree();
                        } else if let Some(u) = &layer_snap.uncertainty {
                            pc.rebuild_uncertainty_quadtree(
                                u.attribute.clone(),
                                u.threshold,
                                crate::snapshot::str_to_measurement_type(&u.measurement_type),
                                u.max_depth,
                            );
                        }
                    }
                    LayerKind::Raster(_) => {}
                }
                for s in &layer_snap.selections {
                    let ids = layer.data.ids_in_bbox_with_fallback(s.bbox);
                    layer.selections.push(crate::gis_layer::LayerSelection {
                        name: s.name.clone(),
                        bbox: s.bbox,
                        ids,
                    });
                }
                layer.active_selection = layer_snap.active_selection;
                layer.show_index = layer_snap.show_index;
                layer.index_kind = crate::snapshot::str_to_index_kind(&layer_snap.index_kind);
                layer.show_heatmap = layer_snap.show_heatmap;
                layer.heatmap_metric =
                    crate::snapshot::str_to_heatmap_metric(&layer_snap.heatmap_metric);
                layer.heatmap_dirty = layer_snap.show_heatmap;
                layer.show_points = layer_snap.show_points;
            }
            if has_filters {
                self.updated_filters = true;
            }
        }

        let queue_empty = self.snapshot_restore.as_ref().unwrap().queue.is_empty();

        if queue_empty {
            let r = self.snapshot_restore.take().unwrap();
            self.viewport.center = r.viewport.center;
            self.viewport.pixels_per_unit = r.viewport.pixels_per_unit;
            self.show_basemap = r.display.show_basemap;
            self.point_size = r.display.point_size;
            self.heatmap_opacity = r.display.heatmap_opacity;
            self.histogram_field = r.analysis.histogram_field;
            self.show_histogram = r.analysis.show_histogram;
            self.bivariate_y_field = r.analysis.bivariate_y_field;
            self.show_bivariate = r.analysis.show_bivariate;
            self.spatial_field = r.analysis.spatial_field;
            self.spatial_radius = r.analysis.spatial_radius;
            self.show_lisa = r.analysis.show_lisa;
            self.show_local_variance = r.analysis.show_local_variance;
            if let Some(idx) = r.analysis.active_layer_idx {
                if idx < self.layers.len() {
                    self.active_layer_idx = Some(idx);
                }
            }
            self.points_dirty = true;
            self.fitted = true; // viewport already restored above — block auto-fit
            self.map_render_ttl = 10;
            self.status = "Snapshot loaded.".to_string();
        } else {
            let next = self
                .snapshot_restore
                .as_mut()
                .unwrap()
                .queue
                .pop_front()
                .unwrap();
            self.open_snapshot_layer(next);
        }
    }

    pub(super) fn open_snapshot_layer(&mut self, next: LayerSnapshot) {
        let path_str = next.file_path.clone();
        if next.is_raster {
            self.snapshot_restore
                .as_mut()
                .unwrap()
                .pending_layer_settings = Some(next);
            let (tx, rx) = mpsc::channel::<Result<LayerEntry, String>>();
            self.raster_load_rx = Some(rx);
            std::thread::spawn(move || {
                let result = crate::raster_reader::load_raster_sync(std::path::Path::new(&path_str))
                    .map_err(|e| e.to_string());
                let _ = tx.send(result);
            });
        } else {
            self.snapshot_restore
                .as_mut()
                .unwrap()
                .pending_layer_settings = Some(next);
            self.open_file(GisFilePath::LocalFile(path_str));
        }
    }
}
