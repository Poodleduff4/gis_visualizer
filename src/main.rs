mod app;
mod basemap;
mod gis_layer;
mod gis_reader;
mod heatmap;
mod hilbert_curve;
mod hilbert_r_tree;
mod map_view;
mod point_cloud;
mod point_cloud_layer;
mod quadtree;
mod sidebar;
mod spatial_index;
mod uncertainty_quadtree;

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
    use wasm_bindgen::JsCast;
    use web_sys::HtmlCanvasElement;

    use crate::app::GisEditorApp;

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

fn main() {}
