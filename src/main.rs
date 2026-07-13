mod app;
mod basemap;
mod crs;
mod snapshot;
mod exporter;
mod filter;
mod gpu_collect;
mod histogram;
mod gis_layer;
mod gis_reader;
mod heatmap;
mod hilbert_curve;
mod hilbert_r_tree;
mod map_view;
#[cfg(not(target_arch = "wasm32"))]
mod parquet;
mod point_cloud;
mod point_cloud_layer;
mod quadtree;
mod raster_reader;
mod globe;
mod sidebar;
mod spatial_index;
mod rtree_index;
mod selection_stats;
mod stats_core;
mod uncertainty_quadtree;
mod vector_gpu;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("GIS Editor"),
        ..Default::default()
    };
    eframe::run_native(
        "GIS Editor",
        options,
        Box::new(|cc| Ok(Box::new(app::GisEditorApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
async fn start() {
    use std::sync::Arc;

    use egui_wgpu::WgpuSetupCreateNew;
    use wasm_bindgen::JsCast;
    use web_sys::HtmlCanvasElement;
    use wgpu::InstanceFlags;

    use crate::app::GisEditorApp;
    use console_error_panic_hook;
    console_error_panic_hook::set_once();

    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    let options = eframe::WebOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
                    flags: wgpu::InstanceFlags::default(),
                    memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                    backend_options: wgpu::BackendOptions::default(),
                    display: None,
                },
                display_handle: None,
                power_preference: wgpu::PowerPreference::HighPerformance,
                native_adapter_selector: None,
                device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(), // use what the adapter actually supports
                    ..Default::default()
                }),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::WebRunner::new()
        .start(
            canvas, // id of canvas element in index.html
            options,
            Box::new(|cc| Ok(Box::new(GisEditorApp::new(cc)))),
        )
        .await
        .unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
