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
                let mut opacity = self.layers[layer_idx].opacity;
                let mut changed = false;
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
                    });
                if changed {
                    self.layers[layer_idx].color = color;
                    self.layers[layer_idx].opacity = opacity;
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

        // ── Kernel Density Estimation window ───────────────────────────────────
        if self.kde_window_open {
            let mut open = true;
            let mut run_clicked = false;
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
                    ui.separator();
                    if self.kde_running {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Computing…");
                        });
                    } else if ui.button("Run").clicked() {
                        run_clicked = true;
                    }
                });
            if run_clicked {
                self.start_kde_compute();
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
        };
        let attribute_name = self
            .kde_weight_field
            .clone()
            .unwrap_or_else(|| "KDE Density".to_string());

        let (tx, rx) = futures_channel::oneshot::channel();
        self.kde_rx = Some(rx);
        self.kde_running = true;
        self.status = format!("Computing KDE ({} pts)…", points.len());

        std::thread::spawn(move || {
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
            tx.send((idx, heatmap, saved)).ok();
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

        std::thread::spawn(move || {
            let Some(layer) = crate::bivariate::BivariateGridLayer::build(
                &points,
                &values_a,
                &values_b,
                bbox,
                cell_size,
                field_a,
                field_b,
            ) else {
                return;
            };
            let saved = crate::bivariate::SavedBivariateGrid::from_layer(
                format!("Bivariate {} x {}", layer.attr_a, layer.attr_b),
                cell_size,
                &layer,
            );
            tx.send((idx, layer, saved)).ok();
        });
    }
}
