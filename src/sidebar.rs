use egui::{ComboBox, RichText, ScrollArea, Ui};

use crate::{
    app::LayerAttributeFilter,
    gis_layer::{AttributeType, AttributeValue, LayerEntry},
    uncertainty_quadtree::{UncertaintyMeasure, UncertaintyMeasurement},
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
}

// ── Main sidebar widget ───────────────────────────────────────────────────────

pub fn show_sidebar(
    ui: &mut Ui,
    layer_entries: &mut [LayerEntry],
    active_layer_idx: Option<usize>,
    selected_id: Option<usize>,
    form: &mut AddAttributeForm,
    save_path: &mut String,
    selected_index_cell_data: Option<&UncertaintyMeasure>,
    // current_filters: &mut Option<&mut Vec<LayerAttributeFilter>>,
    adding_filter: &mut Option<LayerAttributeFilter>,
    updated_filters: &mut bool,
    histogram_field: &mut String,
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
    ui.label(format!("{} features", layer.data.feature_count()));
    ui.separator();
    ui.label(RichText::new("Layer Filters").strong());
    let field_names = &layer.data.field_names();
    let mut to_remove: Option<usize> = None;

    for (idx, filter) in layer.filters.iter().enumerate() {
        ui.label(&format!("Filter on: {}", filter.attribute.clone().unwrap()));
        ui.label(&format!(
            "By: {}",
            filter.operation.clone().unwrap().to_string()
        ));
        ui.label(&format!("Compare to: {}", filter.comparitor_raw));
        if ui.button("Delete Filter").clicked() {
            println!("Delete Filter");
            to_remove = Some(idx);
        }
    }

    if let Some(idx) = to_remove {
        layer.filters.remove(idx);
        *updated_filters = true;
    }
    if let Some(filter) = adding_filter {
        ui.menu_button("Attribute", |ui| {
            for name in field_names.iter() {
                ui.selectable_value(&mut filter.attribute, Some(name.clone()), name.as_str());
            }
        });
        ui.menu_button("Operation", |ui| {
            ui.selectable_value(
                &mut filter.operation,
                Some(crate::app::FilterOperation::LessThan),
                "Less Than",
            );
            ui.selectable_value(
                &mut filter.operation,
                Some(crate::app::FilterOperation::GreaterThan),
                "Greater Than",
            );
            ui.selectable_value(
                &mut filter.operation,
                Some(crate::app::FilterOperation::Equal),
                "Equal",
            );
        });
        ui.horizontal(|ui| {
            ui.label("Compare to:");
            ui.text_edit_singleline(&mut filter.comparitor_raw);
        });
        if ui.button("Create Filter").clicked() {
            let attr_type = filter
                .attribute
                .as_deref()
                .and_then(|name| layer.data.column_type_for(name))
                .unwrap_or(AttributeType::Text);
            let comparitor = attr_type
                .parse_value(&filter.comparitor_raw)
                .unwrap_or(AttributeValue::Text(filter.comparitor_raw.clone()));
            layer.filters.push(LayerAttributeFilter {
                attribute: filter.attribute.clone(),
                operation: filter.operation.clone(),
                comparitor,
                comparitor_raw: filter.comparitor_raw.clone(),
            });
            adding_filter.take();
            // layer.filters.push(adding_filter.take().unwrap());
            *updated_filters = true;
            // current_filters.push(adding_filter.take().unwrap());
        }
    } else {
        if ui.button("Add Filter").clicked() {
            adding_filter.replace(LayerAttributeFilter {
                attribute: None,
                operation: None,
                comparitor: AttributeValue::Text(String::new()),
                comparitor_raw: String::new(),
            });
        }
    }
    ui.separator();
    ui.label(RichText::new("Distribution").strong());
    let numeric_fields = layer.data.numeric_field_names();
    if numeric_fields.is_empty() {
        ui.label("No numeric fields.");
    } else {
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
        if ui.button("Show Histogram").clicked() && !histogram_field.is_empty() {
            action = SidebarAction::OpenHistogram(histogram_field.clone());
        }
    }
    ui.separator();
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

    // Attribute table
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

    ui.separator();

    // ── Save ──────────────────────────────────────────────────────────────────
    // ui.label(RichText::new("Save").strong());
    // ui.add_space(2.0);
    // ui.label("Output path:");
    // ui.text_edit_singleline(save_path);

    // ui.horizontal(|ui| {
    //     if ui.button("Browse…").clicked() {
    //         if let Some(path) = rfd::FileDialog::new()
    //             .add_filter("Shapefile", &["shp"])
    //             .add_filter("GeoPackage", &["gpkg"])
    //             .add_filter("GeoJSON", &["geojson"])
    //             .save_file()
    //         {
    //             *save_path = path.to_string_lossy().to_string();
    //         }
    //     }

    //     if ui.button("Save").clicked() {
    //         let path = save_path.trim().to_string();
    //         if !path.is_empty() {
    //             action = SidebarAction::SaveAs(path);
    //         }
    //     }
    // });

    action
}
