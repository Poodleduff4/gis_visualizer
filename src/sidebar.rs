use egui::{ComboBox, RichText, ScrollArea, Ui};

use crate::{
    filter::{FilterLogic, FilterOperation, LayerAttributeFilter},
    gis_layer::{AttributeType, AttributeValue, LayerEntry},
    histogram::FieldStats,
    uncertainty_quadtree::UncertaintyMeasure,
};

// ── Add-attribute form state ──────────────────────────────────────────────────

#[derive(Default)]
pub struct AddAttributeForm {
    pub name: String,
    pub attr_type: AttributeType,
    pub value_raw: String,
    pub error: Option<String>,
}

// ── Actions emitted by the sidebar ───────────────────────────────────────────

pub enum SidebarAction {
    None,
    AddAttribute {
        feature_id: usize,
        name: String,
        value: AttributeValue,
    },
    SaveAs(String),
    OpenHistogram(String),
    OpenBivariate(String, String),
    ExportFiltered,
    ComputeLocalVariance(String, f64),
    ComputeLisa(String, f64),
    /// Selection-scoped counterparts, emitted by `show_selection_sidebar`
    /// instead of it spawning worker threads itself — keeps a single
    /// dispatch point (in `ui_sidebar.rs`'s match on `SidebarAction`) for
    /// both the whole-layer and selection-scoped analysis actions.
    ExportSelection,
    ComputeLocalVarianceSelection(String, f64),
    ComputeLisaSelection(String, f64),
}

// ── Main sidebar widget ───────────────────────────────────────────────────────

pub fn show_sidebar(
    ui: &mut Ui,
    layer_entries: &mut [LayerEntry],
    active_layer_idx: Option<usize>,
    selected_id: Option<usize>,
    form: &mut AddAttributeForm,
    _save_path: &mut String,
    selected_index_cell_data: Option<&UncertaintyMeasure>,
    adding_filter: &mut Option<LayerAttributeFilter>,
    updated_filters: &mut bool,
    histogram_field: &mut String,
    bivariate_y_field: &mut String,
    field_stats: Option<&FieldStats>,
    spatial_field: &mut String,
    spatial_radius: &mut f64,
) -> SidebarAction {
    let mut action = SidebarAction::None;

    ui.heading("GIS Viewer");
    ui.separator();

    let Some(active_idx) = active_layer_idx else {
        ui.label("No layer selected.");
        ui.label("Use File -> Open to load a GIS file.");
        return action;
    };

    let layer = &mut layer_entries[active_idx];

    // ── Layer info ────────────────────────────────────────────────────────────
    ui.label(RichText::new(&layer.name).strong());
    let total = layer.data.feature_count();
    let visible = layer.data.filtered_count();
    if visible < total {
        ui.label(format!("{} / {} features (filtered)", visible, total));
    } else {
        ui.label(format!("{} features", total));
    }
    ui.separator();

    // ── Filters ───────────────────────────────────────────────────────────────
    ui.label(RichText::new("Filters").strong());

    if !layer.filters.is_empty() {
        // AND / OR toggle
        ui.horizontal(|ui| {
            ui.label("Logic:");
            ui.selectable_value(&mut layer.filter_logic, FilterLogic::And, "AND");
            ui.selectable_value(&mut layer.filter_logic, FilterLogic::Or, "OR");
        });

        let mut to_remove: Option<usize> = None;
        for (idx, filter) in layer.filters.iter().enumerate() {
            let attr = filter.attribute.as_deref().unwrap_or("?");
            let op = filter
                .operation
                .as_ref()
                .map(|o| o.to_string())
                .unwrap_or_default();
            let val = &filter.comparitor_raw;
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{attr} {op} {val}")).monospace());
                if ui.small_button("✕").clicked() {
                    to_remove = Some(idx);
                }
            });
        }
        if let Some(idx) = to_remove {
            layer.filters.remove(idx);
            *updated_filters = true;
        }
    }

    let field_names = layer.data.field_names();
    let mut do_confirm = false;
    let mut do_cancel = false;

    if let Some(filter) = adding_filter.as_mut() {
        ui.horizontal(|ui| {
            ComboBox::from_id_salt("filter_attr")
                .selected_text(filter.attribute.as_deref().unwrap_or("<field>"))
                .show_ui(ui, |ui| {
                    for name in field_names.iter() {
                        ui.selectable_value(
                            &mut filter.attribute,
                            Some(name.clone()),
                            name.as_str(),
                        );
                    }
                });
            ComboBox::from_id_salt("filter_op")
                .selected_text(
                    filter
                        .operation
                        .as_ref()
                        .map(|o| o.to_string())
                        .unwrap_or_else(|| "<op>".into()),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut filter.operation,
                        Some(FilterOperation::LessThan),
                        "<",
                    );
                    ui.selectable_value(
                        &mut filter.operation,
                        Some(FilterOperation::GreaterThan),
                        ">",
                    );
                    ui.selectable_value(
                        &mut filter.operation,
                        Some(FilterOperation::Equal),
                        "=",
                    );
                });
            ui.text_edit_singleline(&mut filter.comparitor_raw);
        });
        let ready = filter.attribute.is_some()
            && filter.operation.is_some()
            && !filter.comparitor_raw.is_empty();
        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("Add")).clicked() {
                do_confirm = true;
            }
            if ui.button("Cancel").clicked() {
                do_cancel = true;
            }
        });
    }

    if do_confirm {
        if let Some(filter) = adding_filter.take() {
            let attr_type = filter
                .attribute
                .as_deref()
                .and_then(|name| layer.data.column_type_for(name))
                .unwrap_or(AttributeType::Text);
            let comparitor = attr_type
                .parse_value(&filter.comparitor_raw)
                .unwrap_or(AttributeValue::Text(filter.comparitor_raw.clone()));
            layer.filters.push(LayerAttributeFilter {
                attribute: filter.attribute,
                operation: filter.operation,
                comparitor,
                comparitor_raw: filter.comparitor_raw,
            });
            *updated_filters = true;
        }
    } else if do_cancel {
        *adding_filter = None;
    }

    if adding_filter.is_none() {
        ui.horizontal(|ui| {
            if ui.button("+ Filter").clicked() {
                adding_filter.replace(LayerAttributeFilter {
                    attribute: None,
                    operation: None,
                    comparitor: AttributeValue::Text(String::new()),
                    comparitor_raw: String::new(),
                });
            }
            if !layer.filters.is_empty() && ui.button("Clear All").clicked() {
                layer.filters.clear();
                *updated_filters = true;
            }
        });
    }

    ui.separator();

    // ── Distribution / Stats ──────────────────────────────────────────────────
    ui.label(RichText::new("Distribution").strong());
    let numeric_fields = layer.data.numeric_field_names();
    if numeric_fields.is_empty() {
        ui.label("No numeric fields.");
    } else {
        ui.label("X field:");
        ComboBox::from_id_salt("histogram_field")
            .selected_text(if histogram_field.is_empty() {
                "<select field>"
            } else {
                histogram_field.as_str()
            })
            .show_ui(ui, |ui| {
                for name in &numeric_fields {
                    ui.selectable_value(histogram_field, name.clone(), name.as_str());
                }
            });

        if !histogram_field.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("Histogram").clicked() {
                    action = SidebarAction::OpenHistogram(histogram_field.clone());
                }
            });

            ui.label("Y field (scatter):");
            ComboBox::from_id_salt("bivariate_y_field")
                .selected_text(if bivariate_y_field.is_empty() {
                    "<select field>"
                } else {
                    bivariate_y_field.as_str()
                })
                .show_ui(ui, |ui| {
                    for name in &numeric_fields {
                        ui.selectable_value(bivariate_y_field, name.clone(), name.as_str());
                    }
                });

            if !bivariate_y_field.is_empty() && bivariate_y_field != histogram_field.as_str() {
                if ui.button("Scatter / Correlation").clicked() {
                    action = SidebarAction::OpenBivariate(
                        histogram_field.clone(),
                        bivariate_y_field.clone(),
                    );
                }
            }

            if let Some(stats) = field_stats {
                egui::Grid::new("stats_grid")
                    .num_columns(2)
                    .striped(true)
                    .min_col_width(60.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new("Stat").strong());
                        ui.label(RichText::new("Value").strong());
                        ui.end_row();

                        if stats.filtered_count < stats.count {
                            ui.label("Count");
                            ui.label(format!("{} / {}", stats.filtered_count, stats.count));
                            ui.end_row();
                        } else {
                            ui.label("Count");
                            ui.label(stats.count.to_string());
                            ui.end_row();
                        }
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
                        ui.label("P25");
                        ui.label(format!("{:.4}", stats.p25));
                        ui.end_row();
                        ui.label("P50");
                        ui.label(format!("{:.4}", stats.p50));
                        ui.end_row();
                        ui.label("P75");
                        ui.label(format!("{:.4}", stats.p75));
                        ui.end_row();
                    });
            }
        }
    }

    ui.separator();

    // ── Spatial Analysis ─────────────────────────────────────────────────────
    ui.label(RichText::new("Spatial Analysis").strong());
    let numeric_fields = layer.data.numeric_field_names();
    if !numeric_fields.is_empty() {
        ui.label("Field:");
        ComboBox::from_id_salt("spatial_field_combo")
            .selected_text(if spatial_field.is_empty() {
                "<select field>"
            } else {
                spatial_field.as_str()
            })
            .show_ui(ui, |ui| {
                for name in &numeric_fields {
                    ui.selectable_value(spatial_field, name.clone(), name.as_str());
                }
            });

        ui.horizontal(|ui| {
            ui.label("Radius:");
            ui.add(
                egui::DragValue::new(spatial_radius)
                    .speed(0.0001)
                    .range(1e-9..=1e6)
                    .max_decimals(6),
            );
        });

        if !spatial_field.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("Local Variance").clicked() {
                    action = SidebarAction::ComputeLocalVariance(
                        spatial_field.clone(),
                        *spatial_radius,
                    );
                }
                if ui.button("LISA").clicked() {
                    action = SidebarAction::ComputeLisa(
                        spatial_field.clone(),
                        *spatial_radius,
                    );
                }
            });
        }
    } else {
        ui.label("No numeric fields.");
    }

    ui.separator();

    // ── Export ────────────────────────────────────────────────────────────────
    #[cfg(not(target_arch = "wasm32"))]
    {
        ui.label(RichText::new("Export").strong());
        let filtered_count = layer.data.filtered_count();
        let label = if filtered_count < layer.data.feature_count() {
            format!("Export filtered ({} pts)", filtered_count)
        } else {
            format!("Export all ({} pts)", filtered_count)
        };
        if ui.button(label).clicked() {
            action = SidebarAction::ExportFiltered;
        }
        ui.separator();
    }

    if let Some(cell_data) = selected_index_cell_data {
        match cell_data {
            UncertaintyMeasure::Variance {
                variance,
                std_dev,
                mean,
            } => {
                ui.label(RichText::new("Selected index cell").strong());
                ui.label(format!("Standard Deviation: {}", std_dev));
                ui.label(format!("Variance: {}", variance));
                ui.label(format!("Mean: {}", mean));
                ui.separator();
            }
            UncertaintyMeasure::KernalDensity { entropy } => {
                ui.label(format!("Kernel-Density Entropy: {}", entropy));
                ui.separator();
            }
        }
    }

    // ── Selected feature attributes ───────────────────────────────────────────
    let Some(sel_id) = selected_id else {
        ui.label("Click a feature to inspect it.");
        return action;
    };

    let feature = layer.data.feature(sel_id);
    let point_attrs = layer.data.point_attrs_display(sel_id);
    ui.label(RichText::new(format!("Feature #{sel_id}")).strong());
    ui.add_space(4.0);

    let all_names: Vec<String> = layer
        .data
        .field_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
        egui::Grid::new("attr_grid")
            .num_columns(2)
            .striped(true)
            .min_col_width(80.0)
            .show(ui, |ui| {
                ui.label(RichText::new("Field").strong());
                ui.label(RichText::new("Value").strong());
                ui.end_row();

                if let Some(ref vals) = point_attrs {
                    for (name, val) in all_names.iter().zip(vals.iter()) {
                        ui.label(name.as_str());
                        ui.label(val.as_str());
                        ui.end_row();
                    }
                } else if let Some(feat) = feature {
                    for name in &all_names {
                        ui.label(name.as_str());
                        let val = feat.attributes.get(name.as_str());
                        ui.label(val.map(|v| v.as_display_string()).unwrap_or_default());
                        ui.end_row();
                    }
                }
            });
    });

    ui.separator();

    // ── Add attribute form ────────────────────────────────────────────────────
    ui.label(RichText::new("Add Attribute").strong());
    ui.add_space(2.0);

    ui.horizontal(|ui| {
        ui.label("Name:");
        ui.text_edit_singleline(&mut form.name);
    });

    ui.horizontal(|ui| {
        ui.label("Type:");
        ComboBox::from_id_salt("attr_type")
            .selected_text(form.attr_type.label())
            .show_ui(ui, |ui| {
                for t in AttributeType::ALL {
                    ui.selectable_value(&mut form.attr_type, t.clone(), t.label());
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label("Value:");
        ui.text_edit_singleline(&mut form.value_raw);
    });

    if let Some(err) = &form.error {
        ui.label(RichText::new(err).color(egui::Color32::RED).small());
    }

    if ui.button("Add").clicked() {
        let name = form.name.trim().to_string();
        if name.is_empty() {
            form.error = Some("Name cannot be empty.".to_string());
        } else {
            match form.attr_type.parse_value(&form.value_raw) {
                Ok(val) => {
                    form.error = None;
                    action = SidebarAction::AddAttribute {
                        feature_id: sel_id,
                        name,
                        value: val,
                    };
                    form.name.clear();
                    form.value_raw.clear();
                }
                Err(e) => form.error = Some(e.to_string()),
            }
        }
    }

    action
}
