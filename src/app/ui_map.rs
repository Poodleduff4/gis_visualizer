use egui::CentralPanel;

use crate::gis_layer::{LayerKind, LayerSelection};
use crate::globe::{collect_globe_points, GlobeCallback, GlobePipeline};
use crate::heatmap::{HeatmapLayer, HeatmapMetric};
use crate::map_view::{
    draw_lisa_overlay, draw_local_variance_overlay, draw_selection_bboxes, render_raster_overlay,
    show_map, show_quadtree_heatmap, show_spatial_index_grid,
};
use crate::point_cloud::PointCloudCallback;
use crate::uncertainty_quadtree::MeasurementType;

use super::{GisEditorApp, MapView};

fn bbox_contains(outer: &[f64; 4], inner: &[f64; 4]) -> bool {
    outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3]
}

impl GisEditorApp {
    pub(super) fn show_map_panel(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // ── Map (central panel) ───────────────────────────────────────────────
        CentralPanel::default().show_inside(ui, |ui| {
            if self.map_view == MapView::Globe {
                self.show_globe(ui, frame);
                return;
            }
            let active_layer = self.active_layer_idx.and_then(|i| self.layers.get(i));
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

            let render_points = !self.has_gpu;
            let mut roi_toggle: Option<[f64; 4]> = None;
            let mut pending_selection: Option<[f64; 4]> = None;
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
                &mut roi_toggle,
                self.select_mode,
                &mut self.select_drag_start,
                &mut pending_selection,
            );
            if let Some(bbox) = pending_selection {
                if let Some(idx) = self.active_layer_idx {
                    let ids = self.layers[idx].data.ids_in_bbox_with_fallback(bbox);
                    let entry = &mut self.layers[idx];
                    let name = format!("Selection {}", entry.selections.len() + 1);
                    entry.selections.push(LayerSelection {
                        name,
                        bbox,
                        ids,
                    });
                    entry.active_selection = Some(entry.selections.len() - 1);
                }
            }
            if let Some(idx) = self.active_layer_idx {
                if let Some(entry) = self.layers.get(idx) {
                    draw_selection_bboxes(
                        &painter,
                        &entry.selections,
                        entry.active_selection,
                        &self.viewport,
                        response.rect,
                    );
                }
            }
            if let Some(bbox) = roi_toggle {
                if let Some(idx) = self.active_layer_idx {
                    let roi = &mut self.layers[idx].roi_bboxes;
                    if let Some(pos) = roi.iter().position(|b| *b == bbox) {
                        // Exact same cell clicked again -> toggle off.
                        roi.remove(pos);
                    } else if let Some(pos) = roi
                        .iter()
                        .position(|b| bbox_contains(b, &bbox) || bbox_contains(&bbox, b))
                    {
                        // Nested with an existing selection (drilling into or
                        // out of it) -> narrow/replace instead of adding a
                        // second bbox, otherwise the union never shrinks.
                        roi[pos] = bbox;
                    } else {
                        roi.push(bbox);
                    }
                    self.updated_filters = true;
                    self.roi_rebuild_pending = true;
                }
            }
            let active_layer = self.active_layer_idx.and_then(|i| self.layers.get(i));

            // GPU point cloud: always blit the cached offscreen texture (cheap).
            // The offscreen re-render only happens when map_render_ttl > 0 (viewport/data changed).
            if self.has_gpu {
                let rect = response.rect;
                let [wx_min, wy_min, wx_max, wy_max] = self.viewport.viewport_bbox(rect);
                let world_size = [(wx_max - wx_min) as f32, (wy_max - wy_min) as f32];
                let render_dirty = self.map_render_ttl > 0;
                if self.map_render_ttl > 0 {
                    self.map_render_ttl -= 1;
                }
                painter.add(egui::Shape::Callback(
                    egui_wgpu::Callback::new_paint_callback(
                        rect,
                        PointCloudCallback {
                            world_min: [wx_min as f32, wy_min as f32],
                            world_size,
                            screen_min: [rect.left(), rect.top()],
                            screen_size: [rect.width(), rect.height()],
                            render_dirty,
                        },
                    ),
                ));
            }

            let visible_raster = self.layers.iter().find_map(|l| {
                if !l.visible {
                    return None;
                }
                if let LayerKind::Raster(r) = &l.data {
                    Some(r)
                } else {
                    None
                }
            });
            if let Some(raster) = visible_raster {
                render_raster_overlay(
                    ui,
                    &painter,
                    raster,
                    &self.viewport,
                    response.rect,
                    &mut self.raster_texture,
                    self.flat_raster_dirty,
                );
                self.flat_raster_dirty = false;
            }

            if self.selected_id != self.last_selected_id {
                self.points_dirty = true;
            }

            if self.show_heatmap {
                if self.active_layer_idx != self.last_heatmap_layer_idx {
                    self.last_heatmap_layer_idx = self.active_layer_idx;
                    self.heatmap_dirty = true;
                }
                if self.heatmap_dirty {
                    use crate::point_cloud_layer::AttributeColumn;
                    self.heatmap_cache = active_layer.and_then(|e| {
                        let LayerKind::Points(pc) = &e.data else {
                            return None;
                        };
                        let index = pc.index.as_deref()?;
                        let attr = self.selected_uncertainty_attribute.as_ref()?;
                        let field_idx = pc.field_names.iter().position(|n| n == attr)?;
                        let values: Vec<f64> = match &pc.attributes[field_idx] {
                            AttributeColumn::Float(v) => v.clone(),
                            AttributeColumn::Integer(v) => v.iter().map(|x| *x as f64).collect(),
                            AttributeColumn::Text(_) => return None,
                        };
                        Some(HeatmapLayer::build(
                            index,
                            &values,
                            self.selected_split_measurement_type.clone(),
                        ))
                    });
                    self.heatmap_dirty = false;
                }
                if let Some(heatmap) = &self.heatmap_cache {
                    let roi_bboxes = active_layer.map(|e| e.roi_bboxes.as_slice()).unwrap_or(&[]);
                    show_quadtree_heatmap(
                        &painter,
                        heatmap,
                        self.heatmap_metric,
                        roi_bboxes,
                        &self.viewport,
                        response.rect,
                        self.heatmap_opacity,
                    );

                    // ── Legend: gradient bar + range + meaning ──────────────────
                    let (title, max_val, unit) = match self.heatmap_metric {
                        HeatmapMetric::Density => {
                            ("Density (points/cell)".to_string(), heatmap.max_density, "")
                        }
                        HeatmapMetric::Unpredictability => {
                            let label = match &heatmap.measurement_type {
                                MeasurementType::Variance => "Unpredictability (variance)",
                                MeasurementType::KernalDensity => "Unpredictability (entropy)",
                            };
                            (label.to_string(), heatmap.max_unpredictability, "")
                        }
                    };
                    let r = response.rect;
                    let bar_w = 200.0_f32;
                    let bar_h = 14.0_f32;
                    let x = r.min.x + 10.0;
                    let y = r.max.y - 46.0;
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x - 4.0, y - 18.0),
                            egui::vec2(bar_w + 8.0, bar_h + 40.0),
                        ),
                        4.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
                    );
                    painter.text(
                        egui::pos2(x, y - 16.0),
                        egui::Align2::LEFT_TOP,
                        &title,
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    let steps = 40;
                    for i in 0..steps {
                        let t0 = i as f32 / steps as f32;
                        let t1 = (i + 1) as f32 / steps as f32;
                        let color = crate::map_view::heat_color(t0, 255);
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(x + t0 * bar_w, y),
                                egui::pos2(x + t1 * bar_w, y + bar_h),
                            ),
                            0.0,
                            color,
                        );
                    }
                    painter.text(
                        egui::pos2(x, y + bar_h + 2.0),
                        egui::Align2::LEFT_TOP,
                        format!("0{}", unit),
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    painter.text(
                        egui::pos2(x + bar_w, y + bar_h + 2.0),
                        egui::Align2::RIGHT_TOP,
                        format!("{:.3}{}", max_val, unit),
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                }
            }

            if self.show_index {
                let index = active_layer
                    .map(|e| e.data.index(self.index_kind))
                    .flatten();
                show_spatial_index_grid(&painter, index, &mut self.viewport, response.rect);
            }

            if self.show_local_variance {
                if let (Some(variances), Some(idx)) =
                    (&self.local_variance_results, self.active_layer_idx)
                {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        draw_local_variance_overlay(
                            &painter,
                            &pc.points,
                            &pc.filter_mask,
                            variances,
                            &self.viewport,
                            response.rect,
                            200,
                        );
                        // legend
                        let r = response.rect;
                        let x = r.min.x + 10.0;
                        let mut y = r.max.y - 80.0;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(x - 4.0, y - 4.0),
                                egui::vec2(140.0, 72.0),
                            ),
                            4.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
                        );
                        painter.text(
                            egui::pos2(x, y),
                            egui::Align2::LEFT_TOP,
                            "Local Variance",
                            egui::FontId::proportional(11.0),
                            egui::Color32::WHITE,
                        );
                        y += 16.0;
                        for (label, color) in [
                            ("Low", egui::Color32::from_rgb(0, 0, 255)),
                            ("Medium", egui::Color32::from_rgb(0, 200, 0)),
                            ("High", egui::Color32::from_rgb(255, 0, 0)),
                        ] {
                            painter.circle_filled(egui::pos2(x + 6.0, y + 6.0), 5.0, color);
                            painter.text(
                                egui::pos2(x + 16.0, y),
                                egui::Align2::LEFT_TOP,
                                label,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                            y += 16.0;
                        }
                    }
                }
            }

            if self.show_lisa {
                if let (Some(lisa), Some(idx)) = (&self.lisa_results, self.active_layer_idx) {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        draw_lisa_overlay(
                            &painter,
                            &pc.points,
                            &pc.filter_mask,
                            lisa,
                            &self.viewport,
                            response.rect,
                            200,
                        );
                        // legend
                        let r = response.rect;
                        let x = r.min.x + 10.0;
                        let mut y = r.max.y - 96.0;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(x - 4.0, y - 4.0),
                                egui::vec2(170.0, 88.0),
                            ),
                            4.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
                        );
                        painter.text(
                            egui::pos2(x, y),
                            egui::Align2::LEFT_TOP,
                            "LISA Clusters",
                            egui::FontId::proportional(11.0),
                            egui::Color32::WHITE,
                        );
                        y += 16.0;
                        for (label, color) in [
                            ("HH — high cluster", egui::Color32::from_rgb(220, 30, 30)),
                            ("LL — low cluster", egui::Color32::from_rgb(30, 80, 220)),
                            ("HL — high outlier", egui::Color32::from_rgb(240, 140, 20)),
                            ("LH — low outlier", egui::Color32::from_rgb(20, 200, 220)),
                        ] {
                            painter.circle_filled(egui::pos2(x + 6.0, y + 6.0), 5.0, color);
                            painter.text(
                                egui::pos2(x + 16.0, y),
                                egui::Align2::LEFT_TOP,
                                label,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                            y += 16.0;
                        }
                    }
                }
            }
        });
    }

    fn show_globe(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        self.last_canvas_rect = Some(rect);
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.globe_camera.orbit(delta.x, delta.y);
            self.map_render_ttl = 3;
        }
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            self.globe_camera.zoom(scroll);
            self.map_render_ttl = 3;
        }

        if !self.has_gpu {
            return;
        }

        let render_dirty = self.map_render_ttl > 0;
        if self.map_render_ttl > 0 {
            self.map_render_ttl -= 1;
        }

        if self.globe_points_dirty {
            if let Some(wrs) = frame.wgpu_render_state() {
                let device = &wrs.device;
                let queue = &wrs.queue;
                let mut renderer = wrs.renderer.write();
                if let Some(pipeline) = renderer.callback_resources.get_mut::<GlobePipeline>() {
                    collect_globe_points(&self.layers, self.point_size, &mut self.globe_points_buf);
                    pipeline.upload_points(device, queue, &self.globe_points_buf);
                }
            }
            self.globe_points_dirty = false;
        }

        if self.raster_dirty {
            if let Some(wrs) = frame.wgpu_render_state() {
                let device = &wrs.device;
                let queue = &wrs.queue;
                let mut renderer = wrs.renderer.write();
                if let Some(pipeline) = renderer.callback_resources.get_mut::<GlobePipeline>() {
                    let raster = self.layers.iter().find_map(|l| {
                        if !l.visible {
                            return None;
                        }
                        if let LayerKind::Raster(r) = &l.data {
                            Some(r)
                        } else {
                            None
                        }
                    });
                    pipeline.update_raster(device, queue, raster);
                }
            }
            self.raster_dirty = false;
        }

        let painter = ui.painter_at(rect);
        painter.add(egui::Shape::Callback(
            egui_wgpu::Callback::new_paint_callback(
                rect,
                GlobeCallback {
                    camera: self.globe_camera.clone(),
                    screen_size: [rect.width(), rect.height()],
                    render_dirty,
                },
            ),
        ));

        if render_dirty {
            ui.ctx().request_repaint();
        }
    }
}
