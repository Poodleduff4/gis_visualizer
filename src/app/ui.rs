use super::GisEditorApp;

impl eframe::App for GisEditorApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.show_menu_bar(ui);
        self.poll_loading(ui);
        self.show_windows(ui);
        self.show_status_bar(ui);
        self.show_layer_panel(ui);
        self.show_sidebar_panel(ui);
        self.apply_filters(ui);
        self.roi_progressive_rebuild();
        self.poll_spatial_analysis(ui);
        self.rebuild_indices_on_slider_change();
        self.upload_gpu_points(ui, frame);
        self.show_map_panel(ui, frame);
    }
}
