use std::sync::mpsc;
use std::sync::Arc;

use bitvec::{bitvec, vec::BitVec};
use futures_channel::oneshot;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

use crate::filter::{FilterLogic, FilterOperation, LayerAttributeFilter};
use crate::gis_layer::{ramp_rgba, AttributeValue, LayerKind, RasterDisplayMode};
use crate::gpu_collect::collect_gpu_points;
use crate::histogram::{
    compute_bivariate, compute_field_stats, compute_histogram, extract_field_values, lisa_inner,
    local_variance_inner,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::parquet::{extract_batch_as_u32, query_parquet};
use crate::point_cloud::PointCloudPipeline;
use crate::selection_stats::{
    compute_selection_bivariate, compute_selection_field_stats, compute_selection_histogram,
};
use crate::sidebar::{show_sidebar, SidebarAction};
use crate::spatial_index::{IndexKind, SpatialIndex};

use super::{now_ms, GisEditorApp};

pub(super) fn clone_pc_for_export(
    pc: &crate::point_cloud_layer::PointCloudLayer,
) -> crate::point_cloud_layer::PointCloudLayer {
    crate::point_cloud_layer::PointCloudLayer {
        points: std::sync::Arc::new(pc.points.iter().cloned().collect()),
        attributes: pc.attributes.clone(),
        field_names: pc.field_names.clone(),
        filter_mask: pc.filter_mask.clone(),
        index: None,
        bbox: None,
        viewport_mask: bitvec![0; 0],
    }
}

/// Evaluates `filters` against `pc`'s in-memory attribute columns, returning
/// the (parquet-file-relative) ids of points that pass. Used as the wasm
/// path always (no local DataFusion), and as the native path's fallback for
/// any Points layer with no real backing file to query — e.g. one a plugin
/// added via `AddLayer`/`UpdateLayer`, whose `descriptor.location` is empty.
fn evaluate_filters_in_memory(
    pc: &crate::point_cloud_layer::PointCloudLayer,
    filters: &[LayerAttributeFilter],
    use_and: bool,
) -> Vec<u32> {
    use crate::point_cloud_layer::AttributeColumn;
    let field_names = &pc.field_names;
    let attributes = &pc.attributes;

    // Resolve each filter's column and comparison value once, up front —
    // `comparitor` is already the typed value (no per-point string parse
    // needed), and the field-name lookup no longer has to repeat per point.
    // `None` means the filter can never pass (missing attr/column).
    let resolved: Vec<Option<(&AttributeColumn, &LayerAttributeFilter)>> = filters
        .iter()
        .map(|f| {
            let attr = f.attribute.as_deref()?;
            let col_pos = field_names.iter().position(|n| n == attr)?;
            let col = attributes.get(col_pos)?;
            Some((col, f))
        })
        .collect();

    let eval_one = |pos: usize, resolved: &Option<(&AttributeColumn, &LayerAttributeFilter)>| {
        let Some((col, f)) = resolved else {
            return false;
        };
        match (&f.operation, col, &f.comparitor) {
            (Some(FilterOperation::GreaterThan), AttributeColumn::Float(v), AttributeValue::Float(t)) => v[pos] > *t,
            (Some(FilterOperation::LessThan), AttributeColumn::Float(v), AttributeValue::Float(t)) => v[pos] < *t,
            (Some(FilterOperation::Equal), AttributeColumn::Float(v), AttributeValue::Float(t)) => (v[pos] - t).abs() < 1e-9,
            (Some(FilterOperation::GreaterThan), AttributeColumn::Integer(v), AttributeValue::Integer(t)) => v[pos] > *t,
            (Some(FilterOperation::LessThan), AttributeColumn::Integer(v), AttributeValue::Integer(t)) => v[pos] < *t,
            (Some(FilterOperation::Equal), AttributeColumn::Integer(v), AttributeValue::Integer(t)) => v[pos] == *t,
            (Some(FilterOperation::Equal), AttributeColumn::Text(v), AttributeValue::Text(t)) => v[pos] == *t,
            _ => false,
        }
    };

    pc.points
        .iter()
        .enumerate()
        .filter_map(|(pos, (parquet_id, _))| {
            let passes = if use_and {
                resolved.iter().all(|r| eval_one(pos, r))
            } else {
                resolved.iter().any(|r| eval_one(pos, r))
            };
            passes.then_some(*parquet_id)
        })
        .collect()
}

/// Evaluates `filters` against `gl`'s in-memory feature attributes, returning
/// the ids (== index in `gl.features`) of features that pass. Vector features
/// live fully in memory already (unlike points, which may stream from
/// parquet), so this is the only evaluation path — no backing-file query.
fn evaluate_filters_in_memory_vector(
    gl: &crate::gis_layer::GisLayer,
    filters: &[LayerAttributeFilter],
    use_and: bool,
) -> Vec<u32> {
    let eval = |feature: &crate::gis_layer::GisFeature, f: &LayerAttributeFilter| {
        let Some(attr) = f.attribute.as_deref() else {
            return false;
        };
        let Some(value) = feature.attributes.get(attr) else {
            return false;
        };
        match (&f.operation, value, &f.comparitor) {
            (Some(FilterOperation::GreaterThan), AttributeValue::Float(v), AttributeValue::Float(t)) => *v > *t,
            (Some(FilterOperation::LessThan), AttributeValue::Float(v), AttributeValue::Float(t)) => *v < *t,
            (Some(FilterOperation::Equal), AttributeValue::Float(v), AttributeValue::Float(t)) => (*v - t).abs() < 1e-9,
            (Some(FilterOperation::GreaterThan), AttributeValue::Integer(v), AttributeValue::Integer(t)) => *v > *t,
            (Some(FilterOperation::LessThan), AttributeValue::Integer(v), AttributeValue::Integer(t)) => *v < *t,
            (Some(FilterOperation::Equal), AttributeValue::Integer(v), AttributeValue::Integer(t)) => *v == *t,
            (Some(FilterOperation::Equal), AttributeValue::Text(v), AttributeValue::Text(t)) => v == t,
            _ => false,
        }
    };
    gl.features
        .iter()
        .filter_map(|feature| {
            let passes = if use_and {
                filters.iter().all(|f| eval(feature, f))
            } else {
                filters.iter().any(|f| eval(feature, f))
            };
            passes.then_some(feature.id as u32)
        })
        .collect()
}

fn union_bboxes(bboxes: &[[f64; 4]]) -> Option<[f64; 4]> {
    bboxes.iter().copied().reduce(|a, b| {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    })
}

impl GisEditorApp {
    pub(super) fn show_sidebar_panel(&mut self, ui: &mut egui::Ui) {
        // ── Sidebar (right) ───────────────────────────────────────────────────
        egui::Panel::right("sidebar")
            .min_size(260.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Recompute stats when field, active layer, or the
                    // layer's filter_mask change. Layer index catches
                    // switching onto a just-created sampled/selection layer
                    // even when the chosen field name didn't change;
                    // `stats_dirty` catches filter_mask changing under the
                    // *same* layer/field (set wherever filter_mask actually
                    // gets mutated — including once an async filter query's
                    // result lands, not just when it's dispatched).
                    let stats_stale = self.histogram_field != self.last_stats_field
                        || self.active_layer_idx != self.last_stats_layer
                        || self.stats_dirty;
                    if stats_stale {
                        self.field_stats = self.active_layer_idx.and_then(|idx| {
                            if let LayerKind::Points(pc) = &self.layers[idx].data {
                                // `true`: respect the layer's current
                                // filter_mask, so applied filters/samples
                                // are reflected instead of always summarizing
                                // the full unfiltered layer.
                                compute_field_stats(pc, &self.histogram_field, true)
                            } else {
                                None
                            }
                        });
                        self.last_stats_field = self.histogram_field.clone();
                        self.last_stats_layer = self.active_layer_idx;
                        self.stats_dirty = false;
                    }

                    // ── Raster controls (band/range/legend) ────────────────────
                    if let Some(idx) = self.active_layer_idx {
                        if let LayerKind::Raster(raster) = &mut self.layers[idx].data {
                            ui.heading("Raster");
                            ui.label(format!("Variable: {}", raster.variable()));
                            ui.label(format!("Grid: {}×{}", raster.width, raster.height));

                            let mut changed = false;

                            if raster.bands.len() > 1 {
                                let is_rgb =
                                    matches!(raster.display_mode, RasterDisplayMode::Rgb { .. });
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(!is_rgb, "Single band").clicked()
                                        && is_rgb
                                    {
                                        raster.display_mode = RasterDisplayMode::Single(0);
                                        changed = true;
                                    }
                                    if ui.selectable_label(is_rgb, "RGB composite").clicked()
                                        && !is_rgb
                                    {
                                        let n = raster.bands.len();
                                        raster.display_mode = RasterDisplayMode::Rgb {
                                            r: 0,
                                            g: 1.min(n - 1),
                                            b: 2.min(n - 1),
                                        };
                                        changed = true;
                                    }
                                });
                            }

                            match &mut raster.display_mode {
                                RasterDisplayMode::Single(band_idx) => {
                                    if raster.bands.len() > 1 {
                                        let names: Vec<String> =
                                            raster.bands.iter().map(|b| b.name.clone()).collect();
                                        egui::ComboBox::from_label("Band")
                                            .selected_text(names[*band_idx].clone())
                                            .show_ui(ui, |ui| {
                                                for (i, name) in names.iter().enumerate() {
                                                    if ui
                                                        .selectable_value(band_idx, i, name)
                                                        .clicked()
                                                    {
                                                        changed = true;
                                                    }
                                                }
                                            });
                                    }
                                    if raster.bands.len() > 1 {
                                        ui.horizontal(|ui| {
                                            let label = if self.raster_playback_enabled {
                                                "⏸ Pause"
                                            } else {
                                                "▶ Play"
                                            };
                                            if ui.button(label).clicked() {
                                                self.raster_playback_enabled =
                                                    !self.raster_playback_enabled;
                                                self.raster_playback_last_tick_ms = now_ms();
                                            }
                                            ui.label("Interval (s):");
                                            ui.add(
                                                egui::DragValue::new(
                                                    &mut self.raster_playback_interval_secs,
                                                )
                                                .speed(0.1)
                                                .range(0.05..=30.0),
                                            );
                                        });
                                        if self.raster_playback_enabled {
                                            let now = now_ms();
                                            let elapsed_secs =
                                                (now - self.raster_playback_last_tick_ms) / 1000.0;
                                            if elapsed_secs
                                                >= self.raster_playback_interval_secs as f64
                                            {
                                                *band_idx = (*band_idx + 1) % raster.bands.len();
                                                changed = true;
                                                self.raster_playback_last_tick_ms = now;
                                            }
                                            ui.ctx().request_repaint_after(
                                                std::time::Duration::from_millis(100),
                                            );
                                        }
                                    }
                                    let band = &raster.bands[*band_idx];
                                    let (data_min, data_max) = (band.data_min, band.data_max);
                                    let mut display_min = band.display_min;
                                    let mut display_max = band.display_max;
                                    let mut range_changed = false;
                                    ui.label(format!(
                                        "Data range: {:.2} .. {:.2}",
                                        data_min, data_max
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.label("Min:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut display_min)
                                                    .speed((data_max - data_min) / 200.0),
                                            )
                                            .changed()
                                        {
                                            range_changed = true;
                                        }
                                        ui.label("Max:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut display_max)
                                                    .speed((data_max - data_min) / 200.0),
                                            )
                                            .changed()
                                        {
                                            range_changed = true;
                                        }
                                        if ui.small_button("Reset range").clicked() {
                                            display_min = data_min;
                                            display_max = data_max;
                                            range_changed = true;
                                        }
                                    });
                                    // Color range is shared across all bands of this layer
                                    // (not per-band) so playback doesn't jump contrast frame
                                    // to frame.
                                    if range_changed {
                                        for b in raster.bands.iter_mut() {
                                            b.display_min = display_min;
                                            b.display_max = display_max;
                                        }
                                        changed = true;
                                    }
                                    let band = &mut raster.bands[*band_idx];

                                    // Gradient legend
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(ui.available_width(), 16.0),
                                        egui::Sense::hover(),
                                    );
                                    let steps = 32;
                                    let w = rect.width() / steps as f32;
                                    for i in 0..steps {
                                        let t = i as f64 / (steps - 1) as f64;
                                        let [r, g, b, a] = ramp_rgba(t);
                                        let seg = egui::Rect::from_min_size(
                                            egui::pos2(rect.left() + i as f32 * w, rect.top()),
                                            egui::vec2(w + 1.0, rect.height()),
                                        );
                                        ui.painter().rect_filled(
                                            seg,
                                            0.0,
                                            egui::Color32::from_rgba_unmultiplied(r, g, b, a),
                                        );
                                    }
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:.1}", band.display_min))
                                                .small(),
                                        );
                                        let units = if raster.units.is_empty() {
                                            String::new()
                                        } else {
                                            format!(" ({})", raster.units)
                                        };
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{:.1}{units}",
                                                        band.display_max
                                                    ))
                                                    .small(),
                                                );
                                            },
                                        );
                                    });
                                }
                                RasterDisplayMode::Rgb { r, g, b } => {
                                    let names: Vec<String> =
                                        raster.bands.iter().map(|bd| bd.name.clone()).collect();
                                    for (label, idx) in [("Red", r), ("Green", g), ("Blue", b)] {
                                        egui::ComboBox::from_label(label)
                                            .selected_text(names[*idx].clone())
                                            .show_ui(ui, |ui| {
                                                for (i, name) in names.iter().enumerate() {
                                                    if ui.selectable_value(idx, i, name).clicked() {
                                                        changed = true;
                                                    }
                                                }
                                            });
                                    }
                                }
                            }

                            if changed {
                                self.raster_dirty = true;
                                self.flat_raster_dirty = true;
                                self.map_render_ttl = 3;
                            }
                            ui.separator();
                        }
                    }

                    let selection_ctx = self.active_layer_idx.and_then(|idx| {
                        self.layers
                            .get(idx)
                            .and_then(|e| e.active_selection.map(|sidx| (idx, sidx)))
                    });

                    let action = if let Some((li, sidx)) = selection_ctx {
                        self.show_selection_sidebar(ui, li, sidx)
                    } else {
                        show_sidebar(
                            ui,
                            &mut self.layers,
                            self.active_layer_idx,
                            self.selected_id,
                            &mut self.add_form,
                            &mut self.save_path,
                            self.selected_index_cell_data.as_ref(),
                            &mut self.adding_filter,
                            &mut self.updated_filters,
                            &mut self.histogram_field,
                            &mut self.bivariate_y_field,
                            self.field_stats.as_ref(),
                            &mut self.spatial_field,
                            &mut self.spatial_radius,
                        )
                    };

                    match action {
                        SidebarAction::AddAttribute {
                            feature_id: _,
                            name: _,
                            value: _,
                        } => {}
                        SidebarAction::SaveAs(_path) => {}
                        SidebarAction::OpenHistogram(field) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    self.histogram = compute_histogram(pc, &field, 50, true);
                                    self.show_histogram = self.histogram.is_some();
                                }
                            }
                        }
                        SidebarAction::OpenBivariate(x_field, y_field) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    self.bivariate =
                                        compute_bivariate(pc, &x_field, &y_field, true, 5000);
                                    self.show_bivariate = self.bivariate.is_some();
                                }
                            }
                        }
                        SidebarAction::ExportFiltered =>
                        {
                            #[cfg(not(target_arch = "wasm32"))]
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    let pc_export = clone_pc_for_export(pc);
                                    let name = self.layers[idx].name.clone();
                                    std::thread::spawn(move || {
                                        if let Some(path) = pollster::block_on(
                                            rfd::AsyncFileDialog::new()
                                                .add_filter("GeoParquet", &["parquet"])
                                                .set_file_name(format!("{}_export.parquet", name))
                                                .save_file(),
                                        ) {
                                            let _ = crate::exporter::export_filtered_points(
                                                &pc_export,
                                                path.path().to_string_lossy().as_ref(),
                                            );
                                        }
                                    });
                                }
                            }
                            #[cfg(target_arch = "wasm32")]
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    let pc_export = clone_pc_for_export(pc);
                                    let name = self.layers[idx].name.clone();
                                    spawn_local(async move {
                                        if let Ok(bytes) =
                                            crate::exporter::export_filtered_points_bytes(&pc_export)
                                        {
                                            if let Some(file) = rfd::AsyncFileDialog::new()
                                                .add_filter("GeoParquet", &["parquet"])
                                                .set_file_name(format!("{}_export.parquet", name))
                                                .save_file()
                                                .await
                                            {
                                                let _ = file.write(&bytes).await;
                                            }
                                        }
                                    });
                                }
                            }
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        SidebarAction::ExportSelection => {
                            if let Some((li, sidx)) = selection_ctx {
                                if let LayerKind::Points(pc) = &self.layers[li].data {
                                    let pc_export = clone_pc_for_export(pc);
                                    let sel = &self.layers[li].selections[sidx];
                                    let ids = sel.ids.clone();
                                    let name = format!("{}_{}", self.layers[li].name, sel.name);
                                    std::thread::spawn(move || {
                                        if let Some(path) = pollster::block_on(
                                            rfd::AsyncFileDialog::new()
                                                .add_filter("GeoParquet", &["parquet"])
                                                .set_file_name(format!("{}_export.parquet", name))
                                                .save_file(),
                                        ) {
                                            let _ = crate::exporter::export_points_by_ids(
                                                &pc_export,
                                                &ids,
                                                path.path().to_string_lossy().as_ref(),
                                            );
                                        }
                                    });
                                }
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        SidebarAction::ExportSelection => {
                            if let Some((li, sidx)) = selection_ctx {
                                if let LayerKind::Points(pc) = &self.layers[li].data {
                                    let pc_export = clone_pc_for_export(pc);
                                    let sel = &self.layers[li].selections[sidx];
                                    let ids = sel.ids.clone();
                                    let name = format!("{}_{}", self.layers[li].name, sel.name);
                                    spawn_local(async move {
                                        if let Ok(bytes) =
                                            crate::exporter::export_points_by_ids_bytes(&pc_export, &ids)
                                        {
                                            if let Some(file) = rfd::AsyncFileDialog::new()
                                                .add_filter("GeoParquet", &["parquet"])
                                                .set_file_name(format!("{}_export.parquet", name))
                                                .save_file()
                                                .await
                                            {
                                                let _ = file.write(&bytes).await;
                                            }
                                        }
                                    });
                                }
                            }
                        }
                        SidebarAction::CreateLayerFromSelection => {
                            if let Some((li, sidx)) = selection_ctx {
                                let sel_name = self.layers[li].selections[sidx].name.clone();
                                let ids = self.layers[li].selections[sidx].ids.clone();
                                let new_name = format!("{}_{}", self.layers[li].name, sel_name);
                                let new_entry = self.layers[li].subset_by_ids(&ids, new_name);

                                if let Some(entry) = new_entry {
                                    self.layers.push(entry);
                                    self.active_layer_idx = Some(self.layers.len() - 1);

                                    // Promoted to its own layer — drop the
                                    // source selection so it doesn't linger
                                    // duplicated (and highlighted) on top of
                                    // the new layer.
                                    let entry = &mut self.layers[li];
                                    entry.selections.remove(sidx);
                                    let fixup = |sel: &mut Option<usize>| match *sel {
                                        Some(s) if s == sidx => *sel = None,
                                        Some(s) if s > sidx => *sel = Some(s - 1),
                                        _ => {}
                                    };
                                    fixup(&mut entry.active_selection);
                                    self.points_dirty = true;
                                }
                            }
                        }
                        SidebarAction::CreateLayerFromFiltered => {
                            if let Some(li) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[li].data {
                                    let ids: Vec<usize> = pc
                                        .filter_mask
                                        .iter()
                                        .enumerate()
                                        .filter(|(_, b)| **b)
                                        .map(|(i, _)| i)
                                        .collect();
                                    let n = ids.len();
                                    let new_name = format!("{}_filtered", self.layers[li].name);
                                    let color = self.layers[li].color;
                                    let opacity = self.layers[li].opacity;
                                    let mut descriptor = self.layers[li].descriptor.clone();
                                    descriptor.name = new_name.clone();
                                    descriptor.num_features = n as u64;

                                    let new_points: Vec<(u32, [f64; 2])> =
                                        ids.iter().map(|&id| pc.points[id]).collect();
                                    let new_attrs: Vec<crate::point_cloud_layer::AttributeColumn> = pc
                                        .attributes
                                        .iter()
                                        .map(|col| match col {
                                            crate::point_cloud_layer::AttributeColumn::Text(v) => {
                                                crate::point_cloud_layer::AttributeColumn::Text(
                                                    ids.iter().map(|&id| v[id].clone()).collect(),
                                                )
                                            }
                                            crate::point_cloud_layer::AttributeColumn::Integer(v) => {
                                                crate::point_cloud_layer::AttributeColumn::Integer(
                                                    ids.iter().map(|&id| v[id]).collect(),
                                                )
                                            }
                                            crate::point_cloud_layer::AttributeColumn::Float(v) => {
                                                crate::point_cloud_layer::AttributeColumn::Float(
                                                    ids.iter().map(|&id| v[id]).collect(),
                                                )
                                            }
                                        })
                                        .collect();
                                    let mut new_pc = crate::point_cloud_layer::PointCloudLayer {
                                        points: std::sync::Arc::new(new_points),
                                        attributes: new_attrs,
                                        field_names: pc.field_names.clone(),
                                        index: None,
                                        bbox: None,
                                        viewport_mask: bitvec![0; n],
                                        filter_mask: bitvec![1; n],
                                    };
                                    new_pc.ensure_bbox();

                                    self.layers.push(crate::gis_layer::LayerEntry {
                                        data: LayerKind::Points(new_pc),
                                        visible: true,
                                        show_points: true,
                                        name: new_name,
                                        color,
                                        color_by: None,
                                        opacity,
                                        descriptor,
                                        filters: Vec::new(),
                                        filter_logic: FilterLogic::default(),
                                        roi_bboxes: Vec::new(),
                                        selections: Vec::new(),
                                        active_selection: None,
                                        crs_transform: None,
                                        show_index: false,
                                        index_kind: crate::spatial_index::IndexKind::Quadtree,
                                        show_heatmap: false,
                                        heatmap_metric: crate::heatmap::HeatmapMetric::Density,
                                        heatmap_cache: None,
                                        heatmap_dirty: true,
                                        show_kde: false,
                                        kde_cache: None,
                                        saved_heatmaps: Vec::new(),
                                        active_saved_heatmap: None,
                                        show_gridbin: false,
                                        gridbin_cache: None,
                                        gridbin_metric: crate::heatmap::HeatmapMetric::Density,
                                        show_bivariate_grid: false,
                                        bivariate_grid_cache: None,
                                        saved_bivariate_grids: Vec::new(),
                                        active_saved_bivariate_grid: None,
                                        batch_load: None,
                                    });
                                    self.active_layer_idx = Some(self.layers.len() - 1);
                                }
                            }
                        }
                        SidebarAction::ComputeLocalVarianceSelection(field, radius) => {
                            if let Some((li, sidx)) = selection_ctx {
                                if let LayerKind::Points(pc) = &self.layers[li].data {
                                    if let Some(values) = extract_field_values(pc, &field) {
                                        let sel_ids = self.layers[li].selections[sidx].ids.clone();
                                        let mut mask = bitvec![0; pc.points.len()];
                                        for &id in &sel_ids {
                                            if id < pc.filter_mask.len() && pc.filter_mask[id] {
                                                mask.set(id, true);
                                            }
                                        }
                                        let points = pc.points.clone();
                                        let index = pc.index.clone();
                                        let (tx, rx) = oneshot::channel();
                                        self.local_variance_rx = Some(rx);
                                        self.show_local_variance = false;
                                        self.show_lisa = false;
                                        self.status = format!(
                                            "Computing local variance ({} pts)…",
                                            mask.count_ones()
                                        );
                                        ui.ctx().request_repaint();
                                        std::thread::spawn(move || {
                                            let result = local_variance_inner(
                                                &points,
                                                &mask,
                                                &values,
                                                radius,
                                                index.as_deref(),
                                            );
                                            tx.send(result).ok();
                                        });
                                    }
                                }
                            }
                        }
                        SidebarAction::ComputeLisaSelection(field, radius) => {
                            if let Some((li, sidx)) = selection_ctx {
                                if let LayerKind::Points(pc) = &self.layers[li].data {
                                    if let Some(values) = extract_field_values(pc, &field) {
                                        let sel_ids = self.layers[li].selections[sidx].ids.clone();
                                        let mut mask = bitvec![0; pc.points.len()];
                                        for &id in &sel_ids {
                                            if id < pc.filter_mask.len() && pc.filter_mask[id] {
                                                mask.set(id, true);
                                            }
                                        }
                                        let points = pc.points.clone();
                                        let index = pc.index.clone();
                                        let (tx, rx) = oneshot::channel();
                                        self.lisa_rx = Some(rx);
                                        self.show_lisa = false;
                                        self.show_local_variance = false;
                                        self.status =
                                            format!("Computing LISA ({} pts)…", mask.count_ones());
                                        ui.ctx().request_repaint();
                                        std::thread::spawn(move || {
                                            let result = lisa_inner(
                                                &points,
                                                &mask,
                                                &values,
                                                radius,
                                                index.as_deref(),
                                            );
                                            tx.send(result).ok();
                                        });
                                    }
                                }
                            }
                        }
                        SidebarAction::ComputeLocalVariance(field, radius) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    if let Some(values) = extract_field_values(pc, &field) {
                                        let points = pc.points.clone();
                                        let filter_mask = pc.filter_mask.clone();
                                        let index = pc.index.clone();
                                        let (tx, rx) = oneshot::channel();
                                        self.local_variance_rx = Some(rx);
                                        self.show_local_variance = false;
                                        self.show_lisa = false;
                                        self.status = format!(
                                            "Computing local variance ({} pts)…",
                                            pc.filter_mask.count_ones()
                                        );
                                        ui.ctx().request_repaint();
                                        std::thread::spawn(move || {
                                            let result = local_variance_inner(
                                                &points,
                                                &filter_mask,
                                                &values,
                                                radius,
                                                index.as_deref(),
                                            );
                                            tx.send(result).ok();
                                        });
                                    }
                                }
                            }
                        }
                        SidebarAction::ComputeLisa(field, radius) => {
                            if let Some(idx) = self.active_layer_idx {
                                if let LayerKind::Points(pc) = &self.layers[idx].data {
                                    if let Some(values) = extract_field_values(pc, &field) {
                                        let points = pc.points.clone();
                                        let filter_mask = pc.filter_mask.clone();
                                        let index = pc.index.clone();
                                        let (tx, rx) = oneshot::channel();
                                        self.lisa_rx = Some(rx);
                                        self.show_lisa = false;
                                        self.show_local_variance = false;
                                        self.status = format!(
                                            "Computing LISA ({} pts)…",
                                            pc.filter_mask.count_ones()
                                        );
                                        ui.ctx().request_repaint();
                                        std::thread::spawn(move || {
                                            let result = lisa_inner(
                                                &points,
                                                &filter_mask,
                                                &values,
                                                radius,
                                                index.as_deref(),
                                            );
                                            tx.send(result).ok();
                                        });
                                    }
                                }
                            }
                        }
                        SidebarAction::None => {}
                    }
                });
            });
    }

    pub(super) fn apply_filters(&mut self, ui: &mut egui::Ui) {
        if self.updated_filters {
            let layer = &mut self.layers[self.active_layer_idx.unwrap()];
            let idx = self.active_layer_idx.unwrap();
            match layer.filters.len() {
                0 => {
                    use crate::point_cloud_layer::PointCloudLayer;
                    layer.data.reset_filter_mask();
                    if !layer.roi_bboxes.is_empty() {
                        let roi_bboxes = layer.roi_bboxes.clone();
                        if let LayerKind::Points(pc) = &mut layer.data {
                            for (pos, (_, p)) in pc.points.iter().enumerate() {
                                if !PointCloudLayer::point_in_any_roi(*p, &roi_bboxes) {
                                    pc.filter_mask.set(pos, false);
                                }
                            }
                        }
                    }
                    self.points_dirty = true;
                    self.updated_filters = false;
                    self.roi_rebuild_pending = true;
                    self.stats_dirty = true;
                    self.refresh_graphs_for_layer(idx);
                }
                _ => {
                    let use_and = layer.filter_logic == FilterLogic::And;
                    // No real backing file to query (e.g. a plugin-added
                    // Points layer via AddLayer/UpdateLayer — its
                    // `descriptor.location` is always empty) — the
                    // DataFusion query below would just run against a
                    // nonexistent file and return zero rows, filtering out
                    // every point. Evaluate in memory instead, same as wasm
                    // (which has no local DataFusion at all) always does.
                    let file_path = layer.descriptor.location.to_string();
                    // Vector (FlatGeobuf) layers are never DataFusion/parquet
                    // backed, regardless of `location` — always evaluate
                    // those in memory.
                    let no_backing_file =
                        file_path.is_empty() || matches!(layer.data, LayerKind::Vector(_));

                    #[cfg(not(target_arch = "wasm32"))]
                    if !no_backing_file {
                        let join_op = match layer.filter_logic {
                            FilterLogic::And => " AND ",
                            FilterLogic::Or => " OR ",
                        };
                        let where_clause = layer
                            .filters
                            .iter()
                            .map(|f| {
                                let attr = f.attribute.as_deref().unwrap_or("");
                                let op = f.operation.clone().unwrap().to_string();
                                let val = match &f.comparitor {
                                    AttributeValue::Text(s) => {
                                        format!("'{}'", s.replace('\'', "''"))
                                    }
                                    AttributeValue::Integer(n) => n.to_string(),
                                    AttributeValue::Float(v) => v.to_string(),
                                };
                                format!("\"{}\" {} {}", attr, op, val)
                            })
                            .collect::<Vec<String>>()
                            .join(join_op);
                        let query = format!("SELECT \"idx\" FROM layer WHERE {}", where_clause);
                        let (tx, rx) = oneshot::channel::<(usize, Vec<u32>)>();
                        self.filtered_idx_rx = Some(rx);
                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Runtime::new().unwrap();
                            rt.block_on(async {
                                // A literal "idx" column only exists in
                                // parquet files this app itself exported
                                // (`exporter.rs`'s `idx: UInt32` field) —
                                // `ensure_idx_column` transparently rewrites
                                // (and caches) a copy with one appended for
                                // any foreign/hand-built GeoParquet lacking
                                // it, so this query can stay a plain,
                                // fully-parallel scan either way.
                                let query_path = match crate::parquet::ensure_idx_column(&file_path).await {
                                    Ok(p) => p,
                                    Err(e) => {
                                        eprintln!("[filter] {e:#}");
                                        let _ = tx.send((idx, Vec::new()));
                                        return;
                                    }
                                };
                                let matching_ids = match query_parquet(
                                    query_path.to_str().unwrap_or(&file_path),
                                    query,
                                )
                                .await
                                {
                                    Ok(batch_vec) => batch_vec
                                        .iter()
                                        .filter_map(|b| extract_batch_as_u32(b, "idx"))
                                        .flatten()
                                        .collect::<Vec<u32>>(),
                                    Err(e) => {
                                        eprintln!("[filter] {e:#}");
                                        Vec::new()
                                    }
                                };
                                let _ = tx.send((idx, matching_ids));
                            });
                        });
                    }
                    #[cfg(target_arch = "wasm32")]
                    let no_backing_file = true;

                    if no_backing_file {
                        let matching_ids = match &layer.data {
                            LayerKind::Points(pc) => {
                                evaluate_filters_in_memory(pc, &layer.filters, use_and)
                            }
                            LayerKind::Vector(gl) => {
                                evaluate_filters_in_memory_vector(gl, &layer.filters, use_and)
                            }
                            LayerKind::Raster(_) => Vec::new(),
                        };
                        let (tx, rx) = oneshot::channel::<(usize, Vec<u32>)>();
                        self.filtered_idx_rx = Some(rx);
                        let _ = tx.send((idx, matching_ids));
                    }
                    self.updated_filters = false;
                }
            };
        }
        if let Some(rx) = &mut self.filtered_idx_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, idx_vec))) => {
                    use crate::point_cloud_layer::PointCloudLayer;
                    println!("{}", idx_vec.len());
                    if let Some(l) = self.layers.get_mut(layer_idx) {
                        let roi_bboxes = l.roi_bboxes.clone();
                        // `idx_vec` can carry up to one entry per point (11M+
                        // on large datasets); a HashSet<u32> here means
                        // hashing every id on insert and again on every
                        // membership check below. Since ids are small,
                        // non-negative integers, a BitVec indexed directly by
                        // id does the same job in O(1) with no hashing.
                        let matching_mask = |ids: &[u32]| -> BitVec {
                            let max_id = ids.iter().copied().max().unwrap_or(0) as usize;
                            let mut m: BitVec = bitvec![0; max_id + 1];
                            for &id in ids {
                                m.set(id as usize, true);
                            }
                            m
                        };
                        match &mut l.data {
                            LayerKind::Points(point_cloud_layer) => {
                                let matching = matching_mask(&idx_vec);
                                let mut mask: BitVec = bitvec![0;point_cloud_layer.points.len()];
                                for (pos, (parquet_id, p)) in
                                    point_cloud_layer.points.iter().enumerate()
                                {
                                    if matching
                                        .get(*parquet_id as usize)
                                        .is_some_and(|b| *b)
                                        && PointCloudLayer::point_in_any_roi(*p, &roi_bboxes)
                                    {
                                        mask.set(pos, true);
                                    }
                                }
                                point_cloud_layer.filter_mask &= mask;
                                self.points_dirty = true;
                                self.roi_rebuild_pending = true;
                                self.stats_dirty = true;
                                ui.request_repaint();
                            }
                            LayerKind::Vector(gl) => {
                                let matching = matching_mask(&idx_vec);
                                let mut mask: BitVec = bitvec![0; gl.features.len()];
                                for feature in &gl.features {
                                    if matching.get(feature.id).is_some_and(|b| *b) {
                                        mask.set(feature.id, true);
                                    }
                                }
                                gl.filter_mask = mask;
                                self.stats_dirty = true;
                                ui.request_repaint();
                            }
                            LayerKind::Raster(_) => {}
                        }
                        self.refresh_graphs_for_layer(layer_idx);
                    }
                }
                Ok(None) => {
                    println!("Not Ready Yet")
                }
                Err(_e) => self.filtered_idx_rx = None,
            }
        }
    }

    pub(super) fn roi_progressive_rebuild(&mut self) {
        // ── Progressive drill-down: rebuild finer index scoped to ROI ─────────
        if self.roi_rebuild_pending {
            self.roi_rebuild_pending = false;
            if let Some(idx) = self.active_layer_idx {
                let roi_bboxes = self.layers[idx].roi_bboxes.clone();
                let was_uncertainty = matches!(
                    self.layers[idx].data.index(IndexKind::Quadtree),
                    Some(SpatialIndex::UncertaintyQuadtree(_))
                );
                if let LayerKind::Points(pc) = &mut self.layers[idx].data {
                    pc.ensure_bbox();
                    let bbox = union_bboxes(&roi_bboxes).or(pc.bbox);
                    if let Some(bbox) = bbox {
                        if was_uncertainty {
                            if let Some(attr) = &self.selected_uncertainty_attribute {
                                pc.rebuild_uncertainty_quadtree_bounded(
                                    attr.clone(),
                                    self.uncertainty_split_threshold,
                                    self.selected_split_measurement_type.clone(),
                                    self.uncertainty_max_depth,
                                    bbox,
                                );
                            }
                        } else {
                            pc.rebuild_quadtree_bounded(
                                self.spatial_index_split_density,
                                bbox,
                            );
                        }
                    }
                }
                self.layers[idx].heatmap_dirty = true;
            }
            self.points_dirty = true;
        }
    }

    pub(super) fn poll_spatial_analysis(&mut self, ui: &mut egui::Ui) {
        // ── Poll spatial analysis background results ──────────────────────────
        if let Some(rx) = &mut self.local_variance_rx {
            match rx.try_recv() {
                Ok(Some(result)) => {
                    self.local_variance_results = Some(result);
                    self.show_local_variance = true;
                    self.local_variance_rx = None;
                    self.status = "Local variance done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.local_variance_rx = None;
                    self.status = "Local variance failed.".to_string();
                }
            }
        }
        if let Some(rx) = &mut self.lisa_rx {
            match rx.try_recv() {
                Ok(Some(result)) => {
                    self.lisa_results = result;
                    self.show_lisa = self.lisa_results.is_some();
                    self.lisa_rx = None;
                    self.status = "LISA done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.lisa_rx = None;
                    self.status = "LISA failed.".to_string();
                }
            }
        }
        if let Some(rx) = &mut self.kde_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, heatmap, saved))) => {
                    self.kde_rx = None;
                    self.kde_running = false;
                    if let Some(entry) = self.layers.get_mut(layer_idx) {
                        entry.kde_cache = Some(heatmap);
                        entry.show_kde = true;
                        entry.saved_heatmaps.push(saved);
                        entry.active_saved_heatmap = Some(entry.saved_heatmaps.len() - 1);
                    }
                    self.status = "KDE done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.kde_rx = None;
                    self.kde_running = false;
                    self.status = "KDE failed.".to_string();
                }
            }
        }
        if let Some(rx) = &mut self.bivariate_grid_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, layer, saved))) => {
                    self.bivariate_grid_rx = None;
                    self.bivariate_grid_running = false;
                    if let Some(entry) = self.layers.get_mut(layer_idx) {
                        entry.bivariate_grid_cache = Some(layer);
                        entry.show_bivariate_grid = true;
                        entry.saved_bivariate_grids.push(saved);
                        entry.active_saved_bivariate_grid =
                            Some(entry.saved_bivariate_grids.len() - 1);
                    }
                    self.status = "Bivariate grid done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.bivariate_grid_rx = None;
                    self.bivariate_grid_running = false;
                    self.status = "Bivariate grid failed.".to_string();
                }
            }
        }
        if let Some(rx) = &mut self.gridbin_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, heatmap, saved))) => {
                    self.gridbin_rx = None;
                    self.gridbin_running = false;
                    if let Some(entry) = self.layers.get_mut(layer_idx) {
                        entry.gridbin_cache = Some(heatmap);
                        entry.gridbin_metric = if saved.units == "Density" {
                            crate::heatmap::HeatmapMetric::Density
                        } else {
                            crate::heatmap::HeatmapMetric::AttributeMean
                        };
                        entry.show_gridbin = true;
                        entry.saved_heatmaps.push(saved);
                        entry.active_saved_heatmap = Some(entry.saved_heatmaps.len() - 1);
                    }
                    self.status = "Grid binning done.".to_string();
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.gridbin_rx = None;
                    self.gridbin_running = false;
                    self.status = "Grid binning failed.".to_string();
                }
            }
        }
        if let Some(rx) = &mut self.raster_export_rx {
            match rx.try_recv() {
                Ok(Some((layer_idx, path_str))) => {
                    self.raster_export_rx = None;
                    if let Some(entry) = self.layers.get_mut(layer_idx) {
                        entry.descriptor.location = crate::gis_reader::GisFilePath::LocalFile(path_str);
                    }
                }
                Ok(None) => {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(_) => {
                    self.raster_export_rx = None;
                }
            }
        }
    }

    /// Turns snapshot-restored selections into real `LayerSelection`s once
    /// their layer's filters have actually resolved (`updated_filters`
    /// cleared and no `filtered_idx_rx` query in flight) — computing ids any
    /// earlier would bake in a stale, unfiltered result. See
    /// `open_snapshot_layer`/the layer-settings-restore block in `mod.rs`,
    /// which defers into `pending_selections` instead of building ids inline.
    pub(super) fn resolve_pending_snapshot_selections(&mut self) {
        let Some(restore) = &self.snapshot_restore else { return };
        if restore.pending_selections.is_empty() {
            return;
        }
        if self.updated_filters || self.filtered_idx_rx.is_some() {
            return;
        }
        let pending = std::mem::take(
            &mut self.snapshot_restore.as_mut().unwrap().pending_selections,
        );
        for (layer_idx, selections, active) in pending {
            if let Some(layer) = self.layers.get_mut(layer_idx) {
                for s in &selections {
                    let ids = layer.data.ids_in_bbox_with_fallback(s.bbox);
                    layer.selections.push(crate::gis_layer::LayerSelection {
                        name: s.name.clone(),
                        bbox: s.bbox,
                        ids,
                    });
                }
                layer.active_selection = active;
            }
        }
    }

    pub(super) fn rebuild_indices_on_slider_change(&mut self) {
        // ── Rebuild quadtree when split-density slider changes ────────────────
        if self.spatial_index_split_density != self.last_split_density {
            let capacity = self.spatial_index_split_density;
            for entry in &mut self.layers {
                match &mut entry.data {
                    LayerKind::Points(point_layer) => {
                        point_layer.rebuild_quadtree(capacity);
                    }
                    LayerKind::Vector(vector_layer) => {
                        vector_layer.rebuild_quadtree(capacity);
                    }
                    LayerKind::Raster(_) => {}
                }
            }
            self.last_split_density = capacity;
            self.points_dirty = true;
            self.viewport_load_pending = true;
            self.viewport_stable_frames = 0;
        }
    }

    pub(super) fn upload_gpu_points(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // ── Re-upload GPU points when data or style changes ───────────────────
        if self.has_gpu {
            let layer_changed = self.layers.len() != self.last_layer_count;
            let selection_changed = self.selected_id != self.last_selected_id;
            let size_changed = self.point_size != self.last_point_size;
            let line_width_changed = self.vector_line_width != self.last_vector_line_width;
            let viewport_changed = self.viewport.center != self.last_viewport_center
                || self.last_viewport_ppu != self.viewport.pixels_per_unit;
            if viewport_changed {
                self.viewport_stable_frames = 0;
                self.viewport_load_pending = true;
                self.last_viewport_center = self.viewport.center;
                self.last_viewport_ppu = self.viewport.pixels_per_unit;
                self.map_render_ttl = 2;
            } else if self.viewport_load_pending {
                self.viewport_stable_frames += 1;
            }
            if self.viewport_load_pending {
                let cursor_in_map = self
                    .last_canvas_rect
                    .and_then(|rect| ui.ctx().pointer_latest_pos().map(|p| rect.contains(p)))
                    .unwrap_or(false);
                if cursor_in_map {
                    ui.ctx().request_repaint();
                }
            }

            if (self.points_dirty
                || layer_changed
                || selection_changed
                || size_changed
                || line_width_changed)
                && !self.streaming_features
            {
                if let Some(wrs) = frame.wgpu_render_state() {
                    let device = &wrs.device;
                    let queue = &wrs.queue;
                    let mut renderer = wrs.renderer.write();
                    if let Some(pipeline) =
                        renderer.callback_resources.get_mut::<PointCloudPipeline>()
                    {
                        collect_gpu_points(
                            &self.layers,
                            self.active_layer_idx,
                            self.selected_id,
                            if self.viewport_culling {
                                self.last_canvas_rect
                                    .map(|rect| self.viewport.viewport_bbox(rect))
                            } else {
                                None
                            },
                            self.point_size,
                            &mut self.gpu_points_buf,
                        );
                        pipeline.upload_points(device, queue, &self.gpu_points_buf);
                    }
                    if let Some(pipeline) =
                        renderer.callback_resources.get_mut::<crate::vector_gpu::VectorPipeline>()
                    {
                        crate::gpu_collect::collect_gpu_vector_mesh(
                            &self.layers,
                            self.active_layer_idx,
                            self.selected_id,
                            &mut self.gpu_vector_fill_verts_buf,
                            &mut self.gpu_vector_fill_indices_buf,
                            &mut self.gpu_vector_line_verts_buf,
                        );
                        pipeline.upload(
                            device,
                            queue,
                            &self.gpu_vector_fill_verts_buf,
                            &self.gpu_vector_fill_indices_buf,
                            &self.gpu_vector_line_verts_buf,
                        );
                    }
                }
                self.map_render_ttl = 2;
                self.points_dirty = false;
                self.last_selected_id = self.selected_id;
                self.last_point_size = self.point_size;
                self.last_vector_line_width = self.vector_line_width;
                self.last_layer_count = self.layers.len();
                self.last_viewport_center = self.viewport.center;
                self.last_viewport_ppu = self.viewport.pixels_per_unit;
            }

            // Index-based viewport streaming (query_and_stream_viewport) is
            // disabled: the shader already does viewport culling on the GPU
            // path, and the CPU-side spatial index can go stale relative to
            // `pc.points`/`viewport_mask` (returns ids past their length),
            // which crashed here. Just clear the pending flag so state stays
            // consistent; no query/thread-spawn work happens.
            if self.viewport_load_pending && self.viewport_stable_frames >= 3 {
                self.viewport_load_pending = false;
            }
        }
    }

    /// Right-sidebar view shown instead of `show_sidebar` while a saved
    /// box-selection is active: Distribution / Spatial Analysis / Export,
    /// scoped to just the selection's ids rather than the whole layer.
    ///
    /// Returns a `SidebarAction` (selection-scoped variants) so that
    /// `show_sidebar_panel` dispatches both this and the regular
    /// `sidebar::show_sidebar` through the same match statement, instead of
    /// each spawning its own worker threads inline.
    fn show_selection_sidebar(&mut self, ui: &mut egui::Ui, li: usize, sidx: usize) -> SidebarAction {
        let mut action = SidebarAction::None;
        let layer_name = self.layers[li].name.clone();
        let (sel_name, sel_ids) = {
            let sel = &self.layers[li].selections[sidx];
            (sel.name.clone(), sel.ids.clone())
        };

        ui.heading("GIS Viewer");
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("{layer_name} › {sel_name}")).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("X Deselect").clicked() {
                    self.layers[li].active_selection = None;
                    self.points_dirty = true;
                }
            });
        });
        ui.label(format!("{} features selected", sel_ids.len()));
        ui.separator();

        let numeric_fields = self.layers[li].data.numeric_field_names();

        // ── Distribution ─────────────────────────────────────────────────
        ui.label(egui::RichText::new("Distribution").strong());
        if numeric_fields.is_empty() {
            ui.label("No numeric fields.");
        } else {
            if self.selection_field_a.is_empty() {
                self.selection_field_a = numeric_fields[0].clone();
            }
            ui.label("X field:");
            egui::ComboBox::from_id_salt("sel_dist_field_a")
                .selected_text(&self.selection_field_a)
                .show_ui(ui, |ui| {
                    for f in &numeric_fields {
                        ui.selectable_value(&mut self.selection_field_a, f.clone(), f);
                    }
                });

            {
                let key = (li, sidx, self.selection_field_a.clone());
                if self.last_selection_stats_key.as_ref() != Some(&key) {
                    let entry = &self.layers[li];
                    let sel = &entry.selections[sidx];
                    let hist =
                        compute_selection_histogram(&entry.data, sel, &self.selection_field_a, 30);
                    let stats =
                        compute_selection_field_stats(&entry.data, sel, &self.selection_field_a);
                    self.selection_histogram = hist;
                    self.selection_field_stats = stats;
                    self.last_selection_stats_key = Some(key);
                }
            }
            if let Some(hist) = &self.selection_histogram {
                let counts = hist.counts.clone();
                let bin_edges = hist.bin_edges.clone();
                let mean = counts
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| ((bin_edges[i] + bin_edges[i + 1]) * 0.5) * c as f64)
                    .sum::<f64>()
                    / counts.iter().copied().sum::<u32>().max(1) as f64;
                super::plot_style::card(ui, |ui| {
                    let counts_max = counts.iter().copied().max().unwrap_or(0);
                    super::plot_style::style(
                        egui_plot::Plot::new("sel_hist_plot")
                            .height(160.0)
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
                                    .fill(super::plot_style::bar_color(counts_max, c))
                                    .stroke(egui::Stroke::NONE)
                            })
                            .collect();
                        plot_ui.bar_chart(egui_plot::BarChart::new("counts", bars));
                        plot_ui.vline(
                            egui_plot::VLine::new("Mean", mean)
                                .color(super::plot_style::MEAN)
                                .style(egui_plot::LineStyle::dashed_loose())
                                .width(1.5),
                        );
                    });
                });
            } else {
                ui.label("No numeric data for this field.");
            }
            if let Some(stats) = &self.selection_field_stats {
                egui::Grid::new("sel_stats_grid")
                    .num_columns(2)
                    .striped(true)
                    .min_col_width(60.0)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Stat").strong());
                        ui.label(egui::RichText::new("Value").strong());
                        ui.end_row();
                        ui.label("Count");
                        ui.label(stats.count.to_string());
                        ui.end_row();
                        ui.label("Min");
                        ui.label(format!("{:.4}", stats.min));
                        ui.end_row();
                        ui.label("Max");
                        ui.label(format!("{:.4}", stats.max));
                        ui.end_row();
                        ui.label("Mean");
                        ui.label(format!("{:.4}", stats.mean));
                        ui.end_row();
                        ui.label("Std Dev");
                        ui.label(format!("{:.4}", stats.std_dev));
                        ui.end_row();
                        ui.label("P25 / P50 / P75");
                        ui.label(format!("{:.4} / {:.4} / {:.4}", stats.p25, stats.p50, stats.p75));
                        ui.end_row();
                    });
            }

            ui.add_space(4.0);
            ui.label("Y field (scatter):");
            egui::ComboBox::from_id_salt("sel_dist_field_b")
                .selected_text(if self.selection_field_b.is_empty() {
                    "<select field>"
                } else {
                    self.selection_field_b.as_str()
                })
                .show_ui(ui, |ui| {
                    for f in &numeric_fields {
                        ui.selectable_value(&mut self.selection_field_b, f.clone(), f);
                    }
                });
            if !self.selection_field_b.is_empty()
                && self.selection_field_b != self.selection_field_a
            {
                let key = (li, sidx, self.selection_field_a.clone(), self.selection_field_b.clone());
                if self.last_selection_bivariate_key.as_ref() != Some(&key) {
                    let entry = &self.layers[li];
                    let sel = &entry.selections[sidx];
                    self.selection_bivariate = compute_selection_bivariate(
                        &entry.data,
                        sel,
                        &self.selection_field_a,
                        &self.selection_field_b,
                        2000,
                    );
                    self.last_selection_bivariate_key = Some(key);
                }
                if let Some(bv) = &self.selection_bivariate {
                    ui.label(format!("Pearson r = {:.4}  (n = {})", bv.pearson_r, bv.n));
                    let points = bv.scatter_points.clone();
                    let trend =
                        super::plot_style::linear_fit(bv.x_mean, bv.y_mean, bv.covariance, bv.x_std);
                    super::plot_style::card(ui, |ui| {
                        super::plot_style::style(
                            egui_plot::Plot::new("sel_scatter_plot")
                                .height(160.0)
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
                                    .color(super::plot_style::ACCENT_FILL),
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
                                    .color(super::plot_style::TREND)
                                    .width(2.0),
                                );
                            }
                        });
                    });
                } else {
                    ui.label("No numeric data for these fields.");
                }
            } else {
                self.selection_bivariate = None;
            }
        }

        ui.separator();

        // ── Spatial Analysis (Points layers only — needs a spatial index) ──
        ui.label(egui::RichText::new("Spatial Analysis").strong());
        if let LayerKind::Points(_) = &self.layers[li].data {
            if numeric_fields.is_empty() {
                ui.label("No numeric fields.");
            } else {
                if self.spatial_field.is_empty() {
                    self.spatial_field = numeric_fields[0].clone();
                }
                ui.label("Field:");
                egui::ComboBox::from_id_salt("sel_spatial_field")
                    .selected_text(&self.spatial_field)
                    .show_ui(ui, |ui| {
                        for f in &numeric_fields {
                            ui.selectable_value(&mut self.spatial_field, f.clone(), f);
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(
                        egui::DragValue::new(&mut self.spatial_radius)
                            .speed(0.0001)
                            .range(1e-9..=1e6)
                            .max_decimals(6),
                    );
                });

                ui.horizontal(|ui| {
                    if ui.button("Local Variance").clicked() {
                        action = SidebarAction::ComputeLocalVarianceSelection(
                            self.spatial_field.clone(),
                            self.spatial_radius,
                        );
                    }
                    if ui.button("LISA").clicked() {
                        action = SidebarAction::ComputeLisaSelection(
                            self.spatial_field.clone(),
                            self.spatial_radius,
                        );
                    }
                });
            }
        } else {
            ui.label("Only available for point-cloud layers.");
        }

        ui.separator();

        // ── Export ───────────────────────────────────────────────────────
        {
            ui.label(egui::RichText::new("Export").strong());
            if let LayerKind::Points(_) = &self.layers[li].data {
                let label = format!("Export selection ({} pts)", sel_ids.len());
                if ui.button(label).clicked() {
                    action = SidebarAction::ExportSelection;
                }
            } else {
                ui.label("Only available for point-cloud layers.");
            }
        }
        if ui.button("Create new Layer from Selection").clicked() {
            action = SidebarAction::CreateLayerFromSelection;
        }

        action
    }
}
