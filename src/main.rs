mod app;
mod basemap;
mod gis_layer;
mod heatmap;
mod hilbert_curve;
mod hilbert_r_tree;
mod map_view;
mod point_cloud;
mod quadtree;
mod sidebar;
mod spatial_index;

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
