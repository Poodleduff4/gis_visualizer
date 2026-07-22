use crate::filter::{FilterOperation, LayerAttributeFilter};
use crate::gis_layer::{AttributeValue, LayerKind};
use crate::histogram::compute_histogram;

use super::plot_style;
use super::GisEditorApp;

impl GisEditorApp {
    pub(super) fn show_windows(&mut self, ui: &mut egui::Ui) {
        // ── Histogram window ─────────────────────────────────────────────────
        if self.show_histogram {
            let mut open = true;
            let mut hist_recompute = false;
            let mut hist_apply_filter: Option<(String, f64, f64)> = None;
            let mut hist_select_on_map: Option<(String, f64, f64)> = None;

            egui::Window::new("Histogram")
                .open(&mut open)
                .resizable(true)
                .default_size([480.0, 320.0])
                .show(ui.ctx(), |ui| {
                    if let Some(hist) = &mut self.histogram {
                        ui.horizontal(|ui| {
                            ui.label("Field:");
                            ui.label(egui::RichText::new(&hist.field).strong());
                            if ui
                                .checkbox(&mut hist.filtered_only, "Filtered only")
                                .changed()
                            {
                                hist_recompute = true;
                            }
                            if ui.button("Recompute").clicked() {
                                hist_recompute = true;
                            }
                        });

                        let counts = hist.counts.clone();
                        let bin_edges = hist.bin_edges.clone();
                        let n = counts.len();
                        let range_lo = hist.range_lo;
                        let range_hi = hist.range_hi;
                        let mean = counts
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| ((bin_edges[i] + bin_edges[i + 1]) * 0.5) * c as f64)
                            .sum::<f64>()
                            / counts.iter().copied().sum::<u32>().max(1) as f64;
                        plot_style::card(ui, |ui| {
                            let counts_max = counts.iter().copied().max().unwrap_or(0);
                            plot_style::style(
                                egui_plot::Plot::new("histogram_plot")
                                    .height(220.0)
                                    .allow_drag(false)
                                    .allow_scroll(false)
                                    .include_y(0.0),
                            )
                            .legend(egui_plot::Legend::default())
                            .show(ui, |plot_ui| {
                                let bars: Vec<egui_plot::Bar> = counts
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &c)| {
                                        let center = (bin_edges[i] + bin_edges[i + 1]) * 0.5;
                                        let width = bin_edges[i + 1] - bin_edges[i];
                                        egui_plot::Bar::new(center, c as f64)
                                            .width(width * 0.95)
                                            .fill(plot_style::bar_color(counts_max, c))
                                            .stroke(egui::Stroke::NONE)
                                    })
                                    .collect();
                                plot_ui.bar_chart(egui_plot::BarChart::new("counts", bars));
                                plot_ui.vline(
                                    egui_plot::VLine::new("Mean", mean)
                                        .color(plot_style::MEAN)
                                        .style(egui_plot::LineStyle::dashed_loose())
                                        .width(1.5),
                                );
                                plot_ui.vline(
                                    egui_plot::VLine::new("Range lo", range_lo)
                                        .color(plot_style::BAD)
                                        .width(1.5),
                                );
                                plot_ui.vline(
                                    egui_plot::VLine::new("Range hi", range_hi)
                                        .color(plot_style::GOOD)
                                        .width(1.5),
                                );
                            });
                        });

                        ui.separator();
                        let speed = (hist.max - hist.min) / 200.0;
                        let lo_max = hist.range_hi;
                        let hi_min = hist.range_lo;
                        ui.horizontal(|ui| {
                            ui.label("Range:");
                            ui.add(
                                egui::DragValue::new(&mut hist.range_lo)
                                    .speed(speed)
                                    .range(hist.min..=lo_max),
                            );
                            ui.label("to");
                            ui.add(
                                egui::DragValue::new(&mut hist.range_hi)
                                    .speed(speed)
                                    .range(hi_min..=hist.max),
                            );
                            if ui.button("Apply as Range Filter").clicked() {
                                hist_apply_filter =
                                    Some((hist.field.clone(), hist.range_lo, hist.range_hi));
                            }
                            if ui
                                .button("🔗 Select on Map")
                                .on_hover_text(
                                    "Brush this range: highlights matching points on the map \
                                     and drives Selection Stats, without hiding the rest.",
                                )
                                .clicked()
                            {
                                hist_select_on_map =
                                    Some((hist.field.clone(), hist.range_lo, hist.range_hi));
                            }
                        });
                        ui.label(format!(
                            "min: {:.4}  max: {:.4}  bins: {}",
                            hist.min, hist.max, n
                        ));
                    }
                });

            if !open {
                self.show_histogram = false;
            }
            if hist_recompute {
                if let Some(idx) = self.active_layer_idx {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        let (field, filtered_only) = self
                            .histogram
                            .as_ref()
                            .map(|h| (h.field.clone(), h.filtered_only))
                            .unwrap_or_default();
                        self.histogram = compute_histogram(pc, &field, 50, filtered_only);
                    }
                }
            }
            if let Some((field, lo, hi)) = hist_apply_filter {
                if let Some(idx) = self.active_layer_idx {
                    let entry = &mut self.layers[idx];
                    entry
                        .filters
                        .retain(|f| f.attribute.as_deref() != Some(field.as_str()));
                    entry.filters.push(LayerAttributeFilter {
                        attribute: Some(field.clone()),
                        operation: Some(FilterOperation::GreaterThan),
                        comparitor: AttributeValue::Float(lo),
                        comparitor_raw: lo.to_string(),
                    });
                    entry.filters.push(LayerAttributeFilter {
                        attribute: Some(field.clone()),
                        operation: Some(FilterOperation::LessThan),
                        comparitor: AttributeValue::Float(hi),
                        comparitor_raw: hi.to_string(),
                    });
                    self.updated_filters = true;
                }
            }
            if let Some((field, lo, hi)) = hist_select_on_map {
                if let Some(idx) = self.active_layer_idx {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        if let Some(col_idx) = pc.field_names.iter().position(|n| n == &field) {
                            let ids: Vec<usize> = pc
                                .points
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| pc.filter_mask[*i])
                                .filter_map(|(i, _)| {
                                    let v = match &pc.attributes[col_idx] {
                                        crate::point_cloud_layer::AttributeColumn::Float(v) => {
                                            v[i]
                                        }
                                        crate::point_cloud_layer::AttributeColumn::Integer(v) => {
                                            v[i] as f64
                                        }
                                        crate::point_cloud_layer::AttributeColumn::Text(_) => {
                                            return None
                                        }
                                    };
                                    (v >= lo && v <= hi).then_some(i)
                                })
                                .collect();
                            let bbox = pc.bbox.unwrap_or([0.0, 0.0, 0.0, 0.0]);
                            let entry = &mut self.layers[idx];
                            let name = format!("Histogram: {field} ∈ [{lo:.3}, {hi:.3}]");
                            entry.selections.push(crate::gis_layer::LayerSelection {
                                name,
                                bbox,
                                ids,
                            });
                            entry.active_selection = Some(entry.selections.len() - 1);
                            self.points_dirty = true;
                        }
                    }
                }
            }
        }

        // ── Bivariate / Scatter window ────────────────────────────────────────
        if self.show_bivariate {
            let mut open = true;
            let mut lasso_clear_clicked = false;
            let mut lasso_select_on_map_clicked = false;
            egui::Window::new("Scatter / Correlation")
                .open(&mut open)
                .resizable(true)
                .default_size([520.0, 400.0])
                .show(ui.ctx(), |ui| {
                    // Cloned so the plot closure below can freely mutate
                    // `self.bivariate_lasso*` without fighting the borrow on
                    // `self.bivariate` (same field-disjointness problem the
                    // map's polygon select tool dodges by taking explicit
                    // `&mut` params instead of closing over `self`).
                    let bv = self.bivariate.clone();
                    if let Some(bv) = bv {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "X: {}   Y: {}",
                                    bv.x_field, bv.y_field
                                ))
                                .strong(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(format!("n = {}", bv.n));
                                },
                            );
                        });
                        ui.horizontal(|ui| {
                            if ui
                                .toggle_value(&mut self.bivariate_lasso_active, "✏️ Lasso Select")
                                .on_hover_text(
                                    "Click to add vertices, double-click or right-click to \
                                     close the ring",
                                )
                                .changed()
                                && !self.bivariate_lasso_active
                            {
                                self.bivariate_lasso.clear();
                            }
                            if ui
                                .add_enabled(
                                    !self.bivariate_lasso_ready.is_empty(),
                                    egui::Button::new("🔗 Select on Map"),
                                )
                                .on_hover_text(
                                    "Highlights points inside the closed lasso ring on the map \
                                     and drives Selection Stats, without hiding the rest.",
                                )
                                .clicked()
                            {
                                lasso_select_on_map_clicked = true;
                            }
                            if ui
                                .add_enabled(
                                    !self.bivariate_lasso.is_empty()
                                        || !self.bivariate_lasso_ready.is_empty(),
                                    egui::Button::new("✖ Clear Lasso"),
                                )
                                .clicked()
                            {
                                lasso_clear_clicked = true;
                            }
                        });

                        let points = bv.scatter_points.clone();
                        let trend = plot_style::linear_fit(bv.x_mean, bv.y_mean, bv.covariance, bv.x_std);
                        let plot_resp = plot_style::card(ui, |ui| {
                            plot_style::style(
                                egui_plot::Plot::new("bivariate_scatter")
                                    .height(260.0)
                                    .x_axis_label(&bv.x_field)
                                    .y_axis_label(&bv.y_field),
                            )
                            .legend(egui_plot::Legend::default())
                            .show(ui, |plot_ui| {
                                let x_min = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
                                let x_max = points.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
                                let pts: egui_plot::PlotPoints =
                                    points.into_iter().map(|[x, y]| [x, y]).collect();
                                plot_ui.points(
                                    egui_plot::Points::new("Data", pts)
                                        .radius(2.5)
                                        .filled(true)
                                        .shape(egui_plot::MarkerShape::Circle)
                                        .color(plot_style::ACCENT_FILL),
                                );
                                if let Some((slope, intercept)) = trend {
                                    let line_pts: egui_plot::PlotPoints = vec![
                                        [x_min, slope * x_min + intercept],
                                        [x_max, slope * x_max + intercept],
                                    ]
                                    .into();
                                    plot_ui.line(
                                        egui_plot::Line::new(
                                            format!("Trend (r = {:.3})", bv.pearson_r),
                                            line_pts,
                                        )
                                        .color(plot_style::TREND)
                                        .width(2.0),
                                    );
                                }
                            })
                        });

                        if self.bivariate_lasso_active {
                            if plot_resp.response.clicked() {
                                if let Some(pos) = plot_resp.response.interact_pointer_pos() {
                                    let v = plot_resp.transform.value_from_position(pos);
                                    self.bivariate_lasso.push([v.x, v.y]);
                                }
                            }
                            if (plot_resp.response.double_clicked()
                                || plot_resp.response.secondary_clicked())
                                && self.bivariate_lasso.len() >= 3
                            {
                                self.bivariate_lasso_ready =
                                    std::mem::take(&mut self.bivariate_lasso);
                            }
                            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                self.bivariate_lasso.clear();
                            }
                        }

                        // Draw the in-progress ring (yellow, rubber-banded to
                        // the cursor) and the closed ring awaiting commit
                        // (green) directly over the plot in screen space.
                        let painter = ui.painter();
                        if !self.bivariate_lasso.is_empty() {
                            let mut screen_pts: Vec<egui::Pos2> = self
                                .bivariate_lasso
                                .iter()
                                .map(|p| {
                                    plot_resp
                                        .transform
                                        .position_from_point(&egui_plot::PlotPoint::new(p[0], p[1]))
                                })
                                .collect();
                            let vertex_count = screen_pts.len();
                            if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
                                if plot_resp.response.rect.contains(hover) {
                                    screen_pts.push(hover);
                                }
                            }
                            for w in screen_pts.windows(2) {
                                painter.line_segment(
                                    [w[0], w[1]],
                                    egui::Stroke::new(1.5, egui::Color32::YELLOW),
                                );
                            }
                            for p in &screen_pts[..vertex_count] {
                                painter.circle_filled(*p, 3.0, egui::Color32::YELLOW);
                            }
                            if vertex_count >= 3 {
                                painter.line_segment(
                                    [screen_pts[vertex_count - 1], screen_pts[0]],
                                    egui::Stroke::new(
                                        1.0,
                                        egui::Color32::from_rgba_unmultiplied(255, 255, 0, 120),
                                    ),
                                );
                            }
                        }
                        if !self.bivariate_lasso_ready.is_empty() {
                            let screen_pts: Vec<egui::Pos2> = self
                                .bivariate_lasso_ready
                                .iter()
                                .map(|p| {
                                    plot_resp
                                        .transform
                                        .position_from_point(&egui_plot::PlotPoint::new(p[0], p[1]))
                                })
                                .collect();
                            for w in screen_pts.windows(2) {
                                painter.line_segment(
                                    [w[0], w[1]],
                                    egui::Stroke::new(2.0, plot_style::GOOD),
                                );
                            }
                            if let (Some(&first), Some(&last)) =
                                (screen_pts.first(), screen_pts.last())
                            {
                                painter.line_segment([last, first], egui::Stroke::new(2.0, plot_style::GOOD));
                            }
                            for p in &screen_pts {
                                painter.circle_filled(*p, 3.0, plot_style::GOOD);
                            }
                        }

                        ui.separator();
                        egui::Grid::new("bv_stats_grid")
                            .num_columns(2)
                            .striped(true)
                            .min_col_width(120.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Stat").strong());
                                ui.label(egui::RichText::new("Value").strong());
                                ui.end_row();

                                ui.label("Pearson r");
                                ui.label(format!("{:.6}", bv.pearson_r));
                                ui.end_row();

                                ui.label("r²");
                                ui.label(format!("{:.6}", bv.pearson_r * bv.pearson_r));
                                ui.end_row();

                                ui.label("Covariance");
                                ui.label(format!("{:.4}", bv.covariance));
                                ui.end_row();

                                ui.label(format!("Mean {}", bv.x_field));
                                ui.label(format!("{:.4}", bv.x_mean));
                                ui.end_row();

                                ui.label(format!("Std {}", bv.x_field));
                                ui.label(format!("{:.4}", bv.x_std));
                                ui.end_row();

                                ui.label(format!("Mean {}", bv.y_field));
                                ui.label(format!("{:.4}", bv.y_mean));
                                ui.end_row();

                                ui.label(format!("Std {}", bv.y_field));
                                ui.label(format!("{:.4}", bv.y_std));
                                ui.end_row();
                            });

                        let strength = match bv.pearson_r.abs() {
                            r if r >= 0.7 => "strong",
                            r if r >= 0.4 => "moderate",
                            r if r >= 0.2 => "weak",
                            _ => "negligible",
                        };
                        let direction = if bv.pearson_r >= 0.0 {
                            "positive"
                        } else {
                            "negative"
                        };
                        ui.label(format!("{} {} correlation", strength, direction));
                    }
                });
            if !open {
                self.show_bivariate = false;
            }
            if lasso_clear_clicked {
                self.bivariate_lasso.clear();
                self.bivariate_lasso_ready.clear();
            }
            if lasso_select_on_map_clicked {
                if let (Some(bv), Some(idx)) = (&self.bivariate, self.active_layer_idx) {
                    if let LayerKind::Points(pc) = &self.layers[idx].data {
                        let x_idx = pc.field_names.iter().position(|n| n == &bv.x_field);
                        let y_idx = pc.field_names.iter().position(|n| n == &bv.y_field);
                        if let (Some(xi), Some(yi)) = (x_idx, y_idx) {
                            use geo::Contains;
                            let ring: geo_types::LineString<f64> = self
                                .bivariate_lasso_ready
                                .iter()
                                .map(|p| (p[0], p[1]))
                                .collect();
                            let poly = geo_types::Polygon::new(ring, vec![]);
                            let col_value = |col: &crate::point_cloud_layer::AttributeColumn, i: usize| -> Option<f64> {
                                match col {
                                    crate::point_cloud_layer::AttributeColumn::Float(v) => Some(v[i]),
                                    crate::point_cloud_layer::AttributeColumn::Integer(v) => Some(v[i] as f64),
                                    crate::point_cloud_layer::AttributeColumn::Text(_) => None,
                                }
                            };
                            let ids: Vec<usize> = pc
                                .points
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| pc.filter_mask[*i])
                                .filter_map(|(i, _)| {
                                    let x = col_value(&pc.attributes[xi], i)?;
                                    let y = col_value(&pc.attributes[yi], i)?;
                                    poly.contains(&geo_types::Point::new(x, y)).then_some(i)
                                })
                                .collect();
                            let bbox = pc.bbox.unwrap_or([0.0, 0.0, 0.0, 0.0]);
                            let name = format!("Scatter lasso: {} x {}", bv.x_field, bv.y_field);
                            let entry = &mut self.layers[idx];
                            entry.selections.push(crate::gis_layer::LayerSelection {
                                name,
                                bbox,
                                ids,
                            });
                            entry.active_selection = Some(entry.selections.len() - 1);
                            self.points_dirty = true;
                        }
                    }
                }
                self.bivariate_lasso_ready.clear();
            }
        }


        // ── Layer color picker window ─────────────────────────────────────────
        if let Some(layer_idx) = self.color_picker_layer {
            if layer_idx < self.layers.len() {
                let mut open = true;
                let name = self.layers[layer_idx].name.clone();
                let mut color = self.layers[layer_idx].color;
                let mut opacity = self.layers[layer_idx].opacity;
                let mut color_by = self.layers[layer_idx].color_by.clone();
                let vector_fields = match &self.layers[layer_idx].data {
                    crate::gis_layer::LayerKind::Vector(gl) => Some(gl.field_names.clone()),
                    _ => None,
                };
                let mut changed = false;
                let mut color_by_changed = false;
                egui::Window::new("Layer Color")
                    .open(&mut open)
                    .resizable(false)
                    .collapsible(false)
                    .default_size([220.0, 240.0])
                    .show(ui.ctx(), |ui| {
                        ui.label(&name);
                        ui.separator();
                        if egui::color_picker::color_edit_button_srgb(ui, &mut color).changed() {
                            changed = true;
                        }
                        ui.separator();
                        ui.label("Opacity:");
                        if ui.add(egui::Slider::new(&mut opacity, 0..=255)).changed() {
                            changed = true;
                        }
                        if let Some(fields) = &vector_fields {
                            ui.separator();
                            ui.label("Color by attribute:");
                            egui::ComboBox::from_id_salt("color_by_attr")
                                .selected_text(color_by.as_deref().unwrap_or("(fixed color)"))
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(color_by.is_none(), "(fixed color)")
                                        .clicked()
                                    {
                                        color_by = None;
                                        color_by_changed = true;
                                    }
                                    for field in fields {
                                        if ui
                                            .selectable_label(
                                                color_by.as_deref() == Some(field.as_str()),
                                                field,
                                            )
                                            .clicked()
                                        {
                                            color_by = Some(field.clone());
                                            color_by_changed = true;
                                        }
                                    }
                                });
                        }
                    });
                if changed {
                    self.layers[layer_idx].color = color;
                    self.layers[layer_idx].opacity = opacity;
                    self.points_dirty = true;
                    self.globe_points_dirty = true;
                    self.map_render_ttl = 3;
                }
                if color_by_changed {
                    self.layers[layer_idx].color_by = color_by;
                    self.points_dirty = true;
                    self.map_render_ttl = 3;
                }
                if !open {
                    self.color_picker_layer = None;
                }
            } else {
                self.color_picker_layer = None;
            }
        }

        // ── Layer rename window ─────────────────────────────────────────────────
        if let Some(layer_idx) = self.rename_layer_idx {
            if layer_idx < self.layers.len() {
                let mut open = true;
                let mut confirmed = false;
                let mut cancelled = false;
                egui::Window::new("Rename Layer")
                    .open(&mut open)
                    .resizable(false)
                    .collapsible(false)
                    .default_size([260.0, 80.0])
                    .show(ui.ctx(), |ui| {
                        let resp = ui.text_edit_singleline(&mut self.rename_layer_buffer);
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            confirmed = true;
                        }
                        resp.request_focus();
                        ui.horizontal(|ui| {
                            if ui.button("OK").clicked() {
                                confirmed = true;
                            }
                            if ui.button("Cancel").clicked() {
                                cancelled = true;
                            }
                        });
                    });
                if cancelled {
                    open = false;
                }
                if confirmed {
                    let trimmed = self.rename_layer_buffer.trim();
                    if !trimmed.is_empty() {
                        self.layers[layer_idx].name = trimmed.to_string();
                    }
                    self.rename_layer_idx = None;
                }
                if !open {
                    self.rename_layer_idx = None;
                }
            } else {
                self.rename_layer_idx = None;
            }
        }

        // ── Batch Load Manager window ────────────────────────────────────────────
        if let Some(layer_idx) = self.batch_load_window_idx {
            if let Some(state) = self
                .layers
                .get(layer_idx)
                .and_then(|l| l.batch_load.as_ref())
            {
                let total_batches = state.total_batches;
                let loaded_count = state.loaded.len() as u64;
                let batch_size = state.batch_size;
                let total_features = self.layers[layer_idx].descriptor.num_features;
                let loaded_features = (loaded_count * batch_size).min(total_features);
                let fully_loaded = loaded_count >= total_batches;
                let mut open = true;
                let mut load_next = false;
                let mut load_range = false;
                egui::Window::new("Batch Load Manager")
                    .open(&mut open)
                    .resizable(false)
                    .collapsible(false)
                    .default_size([300.0, 160.0])
                    .show(ui.ctx(), |ui| {
                        ui.label(format!("Layer: {}", self.layers[layer_idx].name));
                        ui.label(format!(
                            "Loaded {loaded_count} / {total_batches} batches ({loaded_features} / {total_features} features)"
                        ));
                        ui.separator();
                        ui.add_enabled_ui(!fully_loaded, |ui| {
                            if ui.button("Load next batch").clicked() {
                                load_next = true;
                            }
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("From batch:");
                                ui.add(
                                    egui::DragValue::new(&mut self.batch_range_from)
                                        .range(0..=total_batches.saturating_sub(1)),
                                );
                                ui.label("to:");
                                ui.add(
                                    egui::DragValue::new(&mut self.batch_range_to)
                                        .range(0..=total_batches.saturating_sub(1)),
                                );
                            });
                            if ui.button("Load range").clicked() {
                                load_range = true;
                            }
                        });
                        if fully_loaded {
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 180, 100),
                                "All batches loaded.",
                            );
                        }
                    });

                if load_next {
                    if let Some(idx) = (0..total_batches).find(|b| !state.loaded.contains(b)) {
                        let offset = idx * batch_size;
                        let limit = batch_size.min(total_features - offset);
                        self.pending_batch_jobs.push_back((layer_idx, idx, offset, limit));
                        self.pending_batch_group_remaining += 1;
                    }
                }
                if load_range {
                    let (from, to) = (
                        self.batch_range_from.min(self.batch_range_to),
                        self.batch_range_from.max(self.batch_range_to),
                    );
                    for idx in from..=to.min(total_batches.saturating_sub(1)) {
                        if !state.loaded.contains(&idx) {
                            let offset = idx * batch_size;
                            let limit = batch_size.min(total_features - offset);
                            self.pending_batch_jobs.push_back((layer_idx, idx, offset, limit));
                            self.pending_batch_group_remaining += 1;
                        }
                    }
                }
                if !open {
                    self.batch_load_window_idx = None;
                }
            } else {
                self.batch_load_window_idx = None;
            }
        }

        // ── Kernel Density Estimation window ───────────────────────────────────
        if self.kde_window_open {
            let mut open = true;
            let mut run_clicked = false;
            let mut entropy_clicked = false;
            egui::Window::new("Kernel Density Estimation")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .default_size([320.0, 260.0])
                .show(ui.ctx(), |ui| {
                    let Some(idx) = self.active_layer_idx else {
                        ui.label("Select a Points layer first.");
                        return;
                    };
                    let Some(entry) = self.layers.get(idx) else {
                        ui.label("Select a Points layer first.");
                        return;
                    };
                    if !matches!(entry.data, LayerKind::Points(_)) {
                        ui.label("Active layer must be a Points layer.");
                        return;
                    }
                    ui.label(format!("Layer: {}", entry.name));
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Cell size:");
                        ui.add(
                            egui::DragValue::new(&mut self.kde_cell_size)
                                .speed(0.001)
                                .range(1e-9..=f64::MAX),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Search radius (bandwidth):");
                        ui.add(
                            egui::DragValue::new(&mut self.kde_radius)
                                .speed(0.001)
                                .range(1e-9..=f64::MAX),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Kernel:");
                        egui::ComboBox::from_id_salt("kde_kernel_combo")
                            .selected_text(self.kde_kernel.label())
                            .show_ui(ui, |ui| {
                                for k in crate::kde::KdeKernel::ALL {
                                    ui.selectable_value(&mut self.kde_kernel, k, k.label());
                                }
                            });
                    });
                    let numeric_fields = entry.data.numeric_field_names();
                    ui.horizontal(|ui| {
                        ui.label("Weight field:");
                        egui::ComboBox::from_id_salt("kde_weight_field_combo")
                            .selected_text(self.kde_weight_field.as_deref().unwrap_or("None (count)"))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.kde_weight_field, None, "None (count)");
                                for f in &numeric_fields {
                                    ui.selectable_value(
                                        &mut self.kde_weight_field,
                                        Some(f.clone()),
                                        f,
                                    );
                                }
                            });
                    });
                    ui.checkbox(&mut self.kde_normalize, "Normalize to 0-1 (max cell = 1.0)");
                    ui.horizontal(|ui| {
                        ui.label("Entropy window (cells):");
                        ui.add(
                            egui::DragValue::new(&mut self.kde_entropy_window)
                                .speed(1)
                                .range(1..=50),
                        );
                    });
                    ui.separator();
                    if self.kde_running {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Computing…");
                        });
                    } else {
                        ui.horizontal(|ui| {
                            if ui.button("Run").clicked() {
                                run_clicked = true;
                            }
                            if ui
                                .button("Compute Entropy")
                                .on_hover_text(
                                    "Local Shannon entropy of the KDE density surface — \
                                     low = sharp/concentrated hotspot, high = flat/diffuse",
                                )
                                .clicked()
                            {
                                entropy_clicked = true;
                            }
                        });
                    }
                });
            if run_clicked {
                self.start_kde_compute();
            }
            if entropy_clicked {
                self.start_kde_entropy_compute();
            }
            self.kde_window_open = open;
        }

        // ── Bivariate Grid Analysis window ─────────────────────────────────────
        if self.bivariate_grid_window_open {
            let mut open = true;
            let mut run_clicked = false;
            egui::Window::new("Bivariate Grid Analysis")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .default_size([320.0, 260.0])
                .show(ui.ctx(), |ui| {
                    let Some(idx) = self.active_layer_idx else {
                        ui.label("Select a Points layer first.");
                        return;
                    };
                    let Some(entry) = self.layers.get(idx) else {
                        ui.label("Select a Points layer first.");
                        return;
                    };
                    if !matches!(entry.data, LayerKind::Points(_)) {
                        ui.label("Active layer must be a Points layer.");
                        return;
                    }
                    ui.label(format!("Layer: {}", entry.name));
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Cell size:");
                        ui.add(
                            egui::DragValue::new(&mut self.bivariate_grid_cell_size)
                                .speed(0.001)
                                .range(1e-9..=f64::MAX),
                        );
                    });
                    let numeric_fields = entry.data.numeric_field_names();
                    ui.horizontal(|ui| {
                        ui.label("Attribute A:");
                        egui::ComboBox::from_id_salt("bivariate_field_a_combo")
                            .selected_text(
                                self.bivariate_grid_field_a.as_deref().unwrap_or("Select…"),
                            )
                            .show_ui(ui, |ui| {
                                for f in &numeric_fields {
                                    ui.selectable_value(
                                        &mut self.bivariate_grid_field_a,
                                        Some(f.clone()),
                                        f,
                                    );
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Attribute B:");
                        egui::ComboBox::from_id_salt("bivariate_field_b_combo")
                            .selected_text(
                                self.bivariate_grid_field_b.as_deref().unwrap_or("Select…"),
                            )
                            .show_ui(ui, |ui| {
                                for f in &numeric_fields {
                                    ui.selectable_value(
                                        &mut self.bivariate_grid_field_b,
                                        Some(f.clone()),
                                        f,
                                    );
                                }
                            });
                    });
                    ui.separator();
                    let ready = self.bivariate_grid_field_a.is_some()
                        && self.bivariate_grid_field_b.is_some();
                    if self.bivariate_grid_running {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Computing…");
                        });
                    } else if ready {
                        if ui.button("Run").clicked() {
                            run_clicked = true;
                        }
                    } else {
                        ui.label("Pick both attributes first.");
                    }
                });
            if run_clicked {
                self.start_bivariate_grid_compute();
            }
            self.bivariate_grid_window_open = open;
        }

        // ── Grid Binning (uniform hexbin/gridbin) window ────────────────────────
        if self.gridbin_window_open {
            let mut open = true;
            let mut run_clicked = false;
            egui::Window::new("Grid Binning")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .default_size([320.0, 260.0])
                .show(ui.ctx(), |ui| {
                    let Some(idx) = self.active_layer_idx else {
                        ui.label("Select a Points layer first.");
                        return;
                    };
                    let Some(entry) = self.layers.get(idx) else {
                        ui.label("Select a Points layer first.");
                        return;
                    };
                    if !matches!(entry.data, LayerKind::Points(_)) {
                        ui.label("Active layer must be a Points layer.");
                        return;
                    }
                    ui.label(format!("Layer: {}", entry.name));
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Cell size:");
                        ui.add(
                            egui::DragValue::new(&mut self.gridbin_cell_size)
                                .speed(0.001)
                                .range(1e-9..=f64::MAX),
                        );
                    });
                    let numeric_fields = entry.data.numeric_field_names();
                    ui.horizontal(|ui| {
                        ui.label("Color by:");
                        egui::ComboBox::from_id_salt("gridbin_field_combo")
                            .selected_text(
                                self.gridbin_field.as_deref().unwrap_or("Density (count)"),
                            )
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.gridbin_field,
                                    None,
                                    "Density (count)",
                                );
                                for f in &numeric_fields {
                                    ui.selectable_value(&mut self.gridbin_field, Some(f.clone()), f);
                                }
                            });
                    });
                    ui.separator();
                    if self.gridbin_running {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Computing…");
                        });
                    } else if ui.button("Run").clicked() {
                        run_clicked = true;
                    }
                });
            if run_clicked {
                self.start_gridbin_compute();
            }
            self.gridbin_window_open = open;
        }

        // ── Sampling window ──────────────────────────────────────────────────
        if self.sampling_window_open {
            let mut open = true;
            let mut run_clicked = false;
            egui::Window::new("Sample Layer")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .default_size([320.0, 260.0])
                .show(ui.ctx(), |ui| {
                    let Some(idx) = self.active_layer_idx else {
                        ui.label("Select a layer first.");
                        return;
                    };
                    let Some(entry) = self.layers.get(idx) else {
                        ui.label("Select a layer first.");
                        return;
                    };
                    let field_names: Vec<String> = match &entry.data {
                        LayerKind::Points(pc) => pc.field_names.clone(),
                        LayerKind::Vector(gl) => gl.field_names.clone(),
                        LayerKind::Raster(_) => {
                            ui.label("Active layer must be a Points or Vector layer.");
                            return;
                        }
                    };
                    ui.label(format!("Layer: {}", entry.name));
                    ui.separator();

                    ui.label("Method:");
                    for method in crate::sampling::SamplingMethod::ALL {
                        if ui
                            .radio(self.sampling_method == method, method.label())
                            .clicked()
                        {
                            self.sampling_method = method;
                        }
                    }
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Sample fraction:");
                        let mut pct = self.sampling_fraction * 100.0;
                        if ui
                            .add(egui::Slider::new(&mut pct, 1.0..=100.0).suffix("%"))
                            .changed()
                        {
                            self.sampling_fraction = pct / 100.0;
                        }
                    });

                    if self.sampling_method == crate::sampling::SamplingMethod::Stratified {
                        ui.separator();
                        ui.label("Stratify by attribute:");
                        egui::ComboBox::from_id_salt("sampling_stratify_field")
                            .selected_text(
                                self.sampling_stratify_field.as_deref().unwrap_or("<select field>"),
                            )
                            .show_ui(ui, |ui| {
                                for f in &field_names {
                                    ui.selectable_value(
                                        &mut self.sampling_stratify_field,
                                        Some(f.clone()),
                                        f,
                                    );
                                }
                            });
                    }

                    ui.separator();
                    let ready = self.sampling_method != crate::sampling::SamplingMethod::Stratified
                        || self.sampling_stratify_field.is_some();
                    ui.add_enabled_ui(ready, |ui| {
                        if ui.button("Run").clicked() {
                            run_clicked = true;
                        }
                    });
                    if !ready {
                        ui.label("Pick a field to stratify by first.");
                    }
                });
            if run_clicked {
                self.run_sampling();
            }
            self.sampling_window_open = open;
        }

        // ── Export window ───────────────────────────────────────────────────────
        if self.export_window_open {
            let mut open = true;
            egui::Window::new("Export Layer")
                .open(&mut open)
                .resizable(false)
                .collapsible(false)
                .default_size([320.0, 140.0])
                .show(ui.ctx(), |ui| {
                    let Some(idx) = self.active_layer_idx else {
                        ui.label("Select a layer first.");
                        return;
                    };
                    let Some(entry) = self.layers.get(idx) else {
                        ui.label("Select a layer first.");
                        return;
                    };
                    ui.label(format!("Layer: {}", entry.name));
                    ui.separator();
                    match &entry.data {
                        LayerKind::Points(_) => {
                            ui.label(format!(
                                "{} of {} points pass current filters.",
                                entry.data.filtered_count(),
                                entry.data.feature_count()
                            ));
                            if ui.button("Export as GeoParquet…").clicked() {
                                self.export_active_layer();
                            }
                        }
                        LayerKind::Vector(gl) => {
                            ui.label(format!("{} features.", gl.features.len()));
                            if ui.button("Export as GeoJSON…").clicked() {
                                self.export_active_layer();
                            }
                        }
                        LayerKind::Raster(r) => {
                            ui.label(format!(
                                "{} × {} px, band: {}",
                                r.width,
                                r.height,
                                r.variable()
                            ));
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui.button("Export as GeoTIFF…").clicked() {
                                self.export_active_layer();
                            }
                            #[cfg(target_arch = "wasm32")]
                            ui.label("GeoTIFF export isn't available in the browser build.");
                        }
                    }
                });
            self.export_window_open = open;
        }
    }

    /// Dispatches to the export routine matching the active layer's kind —
    /// GeoParquet (Points), GeoJSON (Vector), GeoTIFF (Raster) — via a native
    /// save dialog (desktop) or browser download (wasm).
    fn export_active_layer(&mut self) {
        let Some(idx) = self.active_layer_idx else { return };
        let Some(entry) = self.layers.get(idx) else { return };
        let name = entry.name.clone();

        match &entry.data {
            LayerKind::Points(pc) => {
                let pc_export = super::ui_sidebar::clone_pc_for_export(pc);
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::spawn(move || {
                    if let Some(path) = pollster::block_on(
                        rfd::AsyncFileDialog::new()
                            .add_filter("GeoParquet", &["parquet"])
                            .set_file_name(format!("{name}_export.parquet"))
                            .save_file(),
                    ) {
                        let _ = crate::exporter::export_filtered_points(
                            &pc_export,
                            path.path().to_string_lossy().as_ref(),
                        );
                    }
                });
                #[cfg(target_arch = "wasm32")]
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(bytes) = crate::exporter::export_filtered_points_bytes(&pc_export) {
                        if let Some(file) = rfd::AsyncFileDialog::new()
                            .add_filter("GeoParquet", &["parquet"])
                            .set_file_name(format!("{name}_export.parquet"))
                            .save_file()
                            .await
                        {
                            let _ = file.write(&bytes).await;
                        }
                    }
                });
            }
            LayerKind::Vector(gl) => {
                let bytes = crate::exporter::export_vector_geojson_bytes(gl);
                let Ok(bytes) = bytes else { return };
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::spawn(move || {
                    if let Some(path) = pollster::block_on(
                        rfd::AsyncFileDialog::new()
                            .add_filter("GeoJSON", &["geojson", "json"])
                            .set_file_name(format!("{name}_export.geojson"))
                            .save_file(),
                    ) {
                        let _ = std::fs::write(path.path(), bytes);
                    }
                });
                #[cfg(target_arch = "wasm32")]
                wasm_bindgen_futures::spawn_local(async move {
                    if let Some(file) = rfd::AsyncFileDialog::new()
                        .add_filter("GeoJSON", &["geojson", "json"])
                        .set_file_name(format!("{name}_export.geojson"))
                        .save_file()
                        .await
                    {
                        let _ = file.write(&bytes).await;
                    }
                });
            }
            #[cfg(not(target_arch = "wasm32"))]
            LayerKind::Raster(r) => {
                let width = r.width;
                let height = r.height;
                let extent = r.extent;
                let band = match &r.display_mode {
                    crate::gis_layer::RasterDisplayMode::Single(i) => &r.bands[*i],
                    crate::gis_layer::RasterDisplayMode::Rgb { r: bi, .. } => &r.bands[*bi],
                };
                // Clip to the display range currently set on the band's
                // min/max sliders, not the full raw data range — so the
                // exported file reflects what's actually on screen.
                let (lo, hi) = (band.display_min as f32, band.display_max as f32);
                let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                let values: Vec<f32> = band.values.iter().map(|v| v.clamp(lo, hi)).collect();
                let (tx, rx) = futures_channel::oneshot::channel();
                self.raster_export_rx = Some(rx);
                std::thread::spawn(move || {
                    if let Some(path) = pollster::block_on(
                        rfd::AsyncFileDialog::new()
                            .add_filter("GeoTIFF", &["tif", "tiff"])
                            .set_file_name(format!("{name}_export.tif"))
                            .save_file(),
                    ) {
                        let path_buf = path.path().to_path_buf();
                        if let Err(e) = crate::raster_reader::write_geotiff(
                            &path_buf,
                            width,
                            height,
                            &values,
                            extent,
                        ) {
                            eprintln!("[raster export] error: {e:#}");
                        } else {
                            tx.send((idx, path_buf.to_string_lossy().into_owned())).ok();
                        }
                    }
                });
            }
            #[cfg(target_arch = "wasm32")]
            LayerKind::Raster(_) => {}
        }
    }

    /// Spawns a background thread that builds a KDE grid over the active
    /// Points layer's (filtered) points, then sends the result back through
    /// `kde_rx` for `poll_spatial_analysis` to install as that layer's
    /// `kde_cache`.
    fn start_kde_compute(&mut self) {
        let Some(idx) = self.active_layer_idx else {
            return;
        };
        let Some(entry) = self.layers.get_mut(idx) else {
            return;
        };
        let LayerKind::Points(pc) = &mut entry.data else {
            return;
        };
        pc.ensure_bbox();
        let Some(bbox) = pc.bbox else {
            return;
        };

        let weights = self
            .kde_weight_field
            .as_ref()
            .and_then(|f| crate::histogram::extract_field_values(pc, f));
        let points: Vec<[f64; 2]> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(_, (_, p))| *p)
            .collect();
        let weights: Option<Vec<f64>> = weights.map(|w| {
            pc.points
                .iter()
                .enumerate()
                .filter(|(i, _)| pc.filter_mask[*i])
                .map(|(i, _)| w[i])
                .collect()
        });

        let params = crate::kde::KdeParams {
            cell_size: self.kde_cell_size,
            radius: self.kde_radius,
            kernel: self.kde_kernel,
            normalize: self.kde_normalize,
        };
        let attribute_name = self
            .kde_weight_field
            .clone()
            .unwrap_or_else(|| "KDE Density".to_string());

        let (tx, rx) = futures_channel::oneshot::channel();
        self.kde_rx = Some(rx);
        self.kde_running = true;
        self.status = format!("Computing KDE ({} pts)…", points.len());

        let compute = move || {
            let cells = crate::kde::build_kde_grid(&points, weights.as_deref(), bbox, &params);
            let saved = crate::heatmap::SavedHeatmap::new(
                format!(
                    "KDE {} r={:.3} — {}",
                    params.kernel.short_label(),
                    params.radius,
                    attribute_name
                ),
                crate::heatmap::HeatmapKind::Kde,
                cells.clone(),
                attribute_name.clone(),
            );
            let heatmap = crate::heatmap::HeatmapLayer::from_kde_cells(cells, attribute_name);
            (idx, heatmap, saved)
        };
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            tx.send(compute()).ok();
        });
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            tx.send(compute()).ok();
        });
    }

    /// Spawns a background thread that builds a spatial KDE-entropy grid
    /// (local Shannon entropy of the KDE density surface, per
    /// `kde::build_kde_entropy_grid`) over the active Points layer's
    /// (filtered) points, then sends the result back through `kde_rx` — same
    /// channel/cache as `start_kde_compute`, since both produce a single
    /// scalar-per-cell grid rendered the same way.
    fn start_kde_entropy_compute(&mut self) {
        let Some(idx) = self.active_layer_idx else {
            return;
        };
        let Some(entry) = self.layers.get_mut(idx) else {
            return;
        };
        let LayerKind::Points(pc) = &mut entry.data else {
            return;
        };
        pc.ensure_bbox();
        let Some(bbox) = pc.bbox else {
            return;
        };

        let weights = self
            .kde_weight_field
            .as_ref()
            .and_then(|f| crate::histogram::extract_field_values(pc, f));
        let points: Vec<[f64; 2]> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(_, (_, p))| *p)
            .collect();
        let weights: Option<Vec<f64>> = weights.map(|w| {
            pc.points
                .iter()
                .enumerate()
                .filter(|(i, _)| pc.filter_mask[*i])
                .map(|(i, _)| w[i])
                .collect()
        });

        let params = crate::kde::KdeParams {
            cell_size: self.kde_cell_size,
            radius: self.kde_radius,
            kernel: self.kde_kernel,
            normalize: self.kde_normalize,
        };
        let window = self.kde_entropy_window;
        let attribute_name = format!(
            "KDE Entropy ({})",
            self.kde_weight_field.clone().unwrap_or_else(|| "count".to_string())
        );

        let (tx, rx) = futures_channel::oneshot::channel();
        self.kde_rx = Some(rx);
        self.kde_running = true;
        self.status = format!("Computing KDE entropy ({} pts)…", points.len());

        let compute = move || {
            let cells =
                crate::kde::build_kde_entropy_grid(&points, weights.as_deref(), bbox, &params, window);
            let saved = crate::heatmap::SavedHeatmap::new(
                format!(
                    "KDE Entropy {} r={:.3} w={} — {}",
                    params.kernel.short_label(),
                    params.radius,
                    window,
                    attribute_name
                ),
                crate::heatmap::HeatmapKind::KdeEntropy,
                cells.clone(),
                attribute_name.clone(),
            );
            let heatmap = crate::heatmap::HeatmapLayer::from_kde_cells(cells, attribute_name);
            (idx, heatmap, saved)
        };
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            tx.send(compute()).ok();
        });
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            tx.send(compute()).ok();
        });
    }

    /// Spawns a background thread that bins the active Points layer's
    /// (filtered) points into a bivariate grid over two chosen attributes,
    /// then sends the result back through `bivariate_grid_rx` for
    /// `poll_spatial_analysis` to install as that layer's `bivariate_grid_cache`.
    fn start_bivariate_grid_compute(&mut self) {
        let Some(idx) = self.active_layer_idx else {
            return;
        };
        let Some(entry) = self.layers.get_mut(idx) else {
            return;
        };
        let LayerKind::Points(pc) = &mut entry.data else {
            return;
        };
        let (Some(field_a), Some(field_b)) = (
            self.bivariate_grid_field_a.clone(),
            self.bivariate_grid_field_b.clone(),
        ) else {
            return;
        };
        pc.ensure_bbox();
        let Some(bbox) = pc.bbox else {
            return;
        };
        let Some(values_a) = crate::histogram::extract_field_values(pc, &field_a) else {
            return;
        };
        let Some(values_b) = crate::histogram::extract_field_values(pc, &field_b) else {
            return;
        };

        let points: Vec<[f64; 2]> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(_, (_, p))| *p)
            .collect();
        let values_a: Vec<f64> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(i, _)| values_a[i])
            .collect();
        let values_b: Vec<f64> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(i, _)| values_b[i])
            .collect();

        let cell_size = self.bivariate_grid_cell_size;

        let (tx, rx) = futures_channel::oneshot::channel();
        self.bivariate_grid_rx = Some(rx);
        self.bivariate_grid_running = true;
        self.status = format!("Computing bivariate grid ({} pts)…", points.len());

        let compute = move || {
            let layer = crate::bivariate::BivariateGridLayer::build(
                &points,
                &values_a,
                &values_b,
                bbox,
                cell_size,
                field_a,
                field_b,
            )?;
            let saved = crate::bivariate::SavedBivariateGrid::from_layer(
                format!("Bivariate {} x {}", layer.attr_a, layer.attr_b),
                cell_size,
                &layer,
            );
            Some((idx, layer, saved))
        };
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            if let Some(result) = compute() {
                tx.send(result).ok();
            }
        });
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(result) = compute() {
                tx.send(result).ok();
            }
        });
    }

    /// Spawns a background thread that bins the active Points layer's
    /// (filtered) points into a uniform grid, then sends the result back
    /// through `gridbin_rx` for `poll_spatial_analysis` to install as that
    /// layer's `gridbin_cache`.
    /// Samples a sub-population of the active layer's currently-visible
    /// (filter-mask-respecting) features per `self.sampling_method`, and
    /// adds the result as a new layer — same "subset by ids" path
    /// `CreateLayerFromSelection` uses, just fed by
    /// `crate::sampling::sample_ids` instead of a box-selection.
    fn run_sampling(&mut self) {
        let Some(idx) = self.active_layer_idx else { return };
        let Some(entry) = self.layers.get(idx) else { return };

        let (ids, group_of): (Vec<usize>, Option<Box<dyn Fn(usize) -> String>>) = match &entry.data
        {
            LayerKind::Points(pc) => {
                let ids: Vec<usize> =
                    (0..pc.points.len()).filter(|&i| pc.filter_mask[i]).collect();
                let group_of = self.sampling_stratify_field.clone().and_then(|field| {
                    let col_idx = pc.field_names.iter().position(|n| *n == field)?;
                    let col = pc.attributes.get(col_idx)?.clone();
                    let f: Box<dyn Fn(usize) -> String> =
                        Box::new(move |id: usize| col.get_display(id));
                    Some(f)
                });
                (ids, group_of)
            }
            LayerKind::Vector(gl) => {
                let ids: Vec<usize> =
                    (0..gl.features.len()).filter(|&i| gl.filter_mask[i]).collect();
                let group_of = self.sampling_stratify_field.clone().map(|field| {
                    let groups: std::collections::HashMap<usize, String> = gl
                        .features
                        .iter()
                        .map(|feat| {
                            let key = feat
                                .attributes
                                .get(&field)
                                .map(|v| v.as_display_string())
                                .unwrap_or_default();
                            (feat.id, key)
                        })
                        .collect();
                    let f: Box<dyn Fn(usize) -> String> =
                        Box::new(move |id: usize| groups.get(&id).cloned().unwrap_or_default());
                    f
                });
                (ids, group_of)
            }
            LayerKind::Raster(_) => return,
        };

        let sampled_ids = crate::sampling::sample_ids(
            &ids,
            self.sampling_method,
            self.sampling_fraction,
            group_of.as_deref(),
        );
        let new_name = format!("{}_sampled", entry.name);
        if let Some(new_entry) = entry.subset_by_ids(&sampled_ids, new_name) {
            self.layers.push(new_entry);
            self.active_layer_idx = Some(self.layers.len() - 1);
        }
    }

    fn start_gridbin_compute(&mut self) {
        let Some(idx) = self.active_layer_idx else {
            return;
        };
        let Some(entry) = self.layers.get_mut(idx) else {
            return;
        };
        let LayerKind::Points(pc) = &mut entry.data else {
            return;
        };
        pc.ensure_bbox();
        let Some(bbox) = pc.bbox else {
            return;
        };

        let field = self.gridbin_field.clone();
        let values = field
            .as_ref()
            .and_then(|f| crate::histogram::extract_field_values(pc, f));
        let points: Vec<[f64; 2]> = pc
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| pc.filter_mask[*i])
            .map(|(_, (_, p))| *p)
            .collect();
        let values: Option<Vec<f64>> = values.map(|v| {
            pc.points
                .iter()
                .enumerate()
                .filter(|(i, _)| pc.filter_mask[*i])
                .map(|(i, _)| v[i])
                .collect()
        });

        let cell_size = self.gridbin_cell_size;
        let attribute_name = field.clone().unwrap_or_else(|| "Density".to_string());

        let (tx, rx) = futures_channel::oneshot::channel();
        self.gridbin_rx = Some(rx);
        self.gridbin_running = true;
        self.status = format!("Computing grid bins ({} pts)…", points.len());

        let compute = move || {
            let cells = crate::gridbin::build_gridbin(&points, values.as_deref(), bbox, cell_size);
            let raw_cells: Vec<([f64; 4], f32)> = cells
                .iter()
                .map(|c| (c.bbox, c.mean.unwrap_or(c.count as f32)))
                .collect();
            let saved = crate::heatmap::SavedHeatmap::new(
                format!("Gridbin {:.4} — {}", cell_size, attribute_name),
                crate::heatmap::HeatmapKind::GridBin,
                raw_cells,
                attribute_name.clone(),
            );
            let heatmap = crate::heatmap::HeatmapLayer::from_grid_cells(cells, attribute_name);
            (idx, heatmap, saved)
        };
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            tx.send(compute()).ok();
        });
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            tx.send(compute()).ok();
        });
    }
}
