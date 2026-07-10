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
        }

        // ── Bivariate / Scatter window ────────────────────────────────────────
        if self.show_bivariate {
            let mut open = true;
            egui::Window::new("Scatter / Correlation")
                .open(&mut open)
                .resizable(true)
                .default_size([520.0, 400.0])
                .show(ui.ctx(), |ui| {
                    if let Some(bv) = &self.bivariate {
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

                        let points = bv.scatter_points.clone();
                        let trend = plot_style::linear_fit(bv.x_mean, bv.y_mean, bv.covariance, bv.x_std);
                        plot_style::card(ui, |ui| {
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
                            });
                        });

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
        }


        // ── Layer color picker window ─────────────────────────────────────────
        if let Some(layer_idx) = self.color_picker_layer {
            if layer_idx < self.layers.len() {
                let mut open = true;
                let name = self.layers[layer_idx].name.clone();
                let mut color = self.layers[layer_idx].color;
                let mut color_changed = false;
                egui::Window::new("Layer Color")
                    .open(&mut open)
                    .resizable(false)
                    .collapsible(false)
                    .default_size([220.0, 240.0])
                    .show(ui.ctx(), |ui| {
                        ui.label(&name);
                        ui.separator();
                        if egui::color_picker::color_edit_button_srgb(ui, &mut color).changed() {
                            color_changed = true;
                        }
                    });
                if color_changed {
                    self.layers[layer_idx].color = color;
                    self.points_dirty = true;
                    self.globe_points_dirty = true;
                    self.map_render_ttl = 3;
                }
                if !open {
                    self.color_picker_layer = None;
                }
            } else {
                self.color_picker_layer = None;
            }
        }
    }
}
