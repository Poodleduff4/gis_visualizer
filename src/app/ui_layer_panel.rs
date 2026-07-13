use egui::UiKind;

use crate::gis_layer::LayerKind;
use crate::spatial_index::IndexKind;

use super::{GisEditorApp, LAYER_PANEL_WIDTH};

impl GisEditorApp {
    pub(super) fn show_status_bar(&mut self, ui: &mut egui::Ui) {
        // ── Status bar ────────────────────────────────────────────────────────
        egui::Panel::bottom("status_bar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                if self.local_variance_rx.is_some() || self.lisa_rx.is_some() {
                    ui.spinner();
                }
            });
        });
    }

    pub(super) fn show_layer_panel(&mut self, ui: &mut egui::Ui) {
        // ── Layer panel (left) ────────────────────────────────────────────────
        egui::Panel::left("layer_panel")
            .default_size(LAYER_PANEL_WIDTH)
            .min_size(140.0)
            .max_size(420.0)
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
                                if matches!(entry.data, LayerKind::Points(_)) {
                                    if ui
                                        .checkbox(&mut entry.show_points, "")
                                        .on_hover_text("Show points")
                                        .changed()
                                    {
                                        visibility_changed = true;
                                    }
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
                                                    entry.heatmap_dirty = true;
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

                                    let has_quadtree =
                                        entry.data.index(IndexKind::Quadtree).is_some();
                                    let has_hilbert =
                                        entry.data.index(IndexKind::Hilbert).is_some();
                                    if has_quadtree || has_hilbert {
                                        ui.checkbox(&mut entry.show_index, "Show Spatial Index");
                                        if entry.show_index {
                                            ui.indent("layer_index_kind", |ui| {
                                                if has_quadtree {
                                                    ui.radio_value(
                                                        &mut entry.index_kind,
                                                        IndexKind::Quadtree,
                                                        "Quadtree",
                                                    );
                                                }
                                                if has_hilbert {
                                                    ui.radio_value(
                                                        &mut entry.index_kind,
                                                        IndexKind::Hilbert,
                                                        "Hilbert R-Tree",
                                                    );
                                                }
                                            });
                                        }
                                        ui.separator();
                                    }

                                    let has_point_index = matches!(
                                        &entry.data,
                                        LayerKind::Points(pc) if pc.index.is_some()
                                    );
                                    if has_point_index {
                                        if ui
                                            .checkbox(&mut entry.show_heatmap, "Show Heatmap")
                                            .changed()
                                            && entry.show_heatmap
                                        {
                                            entry.heatmap_dirty = true;
                                        }
                                        if entry.show_heatmap {
                                            ui.indent("layer_heatmap_metric", |ui| {
                                                ui.radio_value(
                                                    &mut entry.heatmap_metric,
                                                    crate::heatmap::HeatmapMetric::Density,
                                                    "Density",
                                                );
                                                ui.radio_value(
                                                    &mut entry.heatmap_metric,
                                                    crate::heatmap::HeatmapMetric::Unpredictability,
                                                    "Unpredictability",
                                                );
                                                ui.radio_value(
                                                    &mut entry.heatmap_metric,
                                                    crate::heatmap::HeatmapMetric::AttributeMean,
                                                    "Attribute Average",
                                                );
                                            });
                                        }
                                        ui.separator();
                                    }

                                    if entry.kde_cache.is_some() {
                                        ui.checkbox(&mut entry.show_kde, "Show KDE Heatmap");
                                        ui.separator();
                                    }

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
                        self.layers[idx].heatmap_dirty = true;
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
                        self.layers[idx].heatmap_dirty = true;
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
    }
}
