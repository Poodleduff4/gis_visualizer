use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::filter::{FilterLogic, FilterOperation, LayerAttributeFilter};
use crate::gis_layer::AttributeValue;

#[derive(Serialize, Deserialize, Clone)]
pub struct AppSnapshot {
    pub viewport: ViewportSnapshot,
    pub display: DisplaySnapshot,
    pub analysis: AnalysisSnapshot,
    pub layers: Vec<LayerSnapshot>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ViewportSnapshot {
    pub center: [f64; 2],
    pub pixels_per_unit: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DisplaySnapshot {
    pub show_basemap: bool,
    pub show_heatmap: bool,
    pub show_index: bool,
    pub point_size: f32,
    pub heatmap_opacity: u8,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AnalysisSnapshot {
    pub active_layer_idx: Option<usize>,
    pub histogram_field: String,
    pub show_histogram: bool,
    pub bivariate_y_field: String,
    pub show_bivariate: bool,
    pub spatial_field: String,
    pub spatial_radius: f64,
    pub show_lisa: bool,
    pub show_local_variance: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerSnapshot {
    pub file_path: String,
    #[serde(default)]
    pub is_raster: bool,
    #[serde(default)]
    pub selected_attributes: Vec<String>,
    pub name: String,
    pub visible: bool,
    pub color: [u8; 3],
    pub opacity: u8,
    pub filter_logic: String,
    pub filters: Vec<FilterSnapshot>,
    #[serde(default)]
    pub quadtree_capacity: Option<usize>,
    #[serde(default)]
    pub hilbert_order: Option<u32>,
    #[serde(default)]
    pub built_rtree: bool,
    #[serde(default)]
    pub uncertainty: Option<UncertaintySnapshot>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct UncertaintySnapshot {
    pub attribute: String,
    pub threshold: f32,
    pub measurement_type: String,
    pub max_depth: usize,
}

pub fn measurement_type_to_str(mt: &crate::uncertainty_quadtree::MeasurementType) -> String {
    match mt {
        crate::uncertainty_quadtree::MeasurementType::Variance => "Variance".to_string(),
        crate::uncertainty_quadtree::MeasurementType::KernalDensity => "KernalDensity".to_string(),
    }
}

pub fn str_to_measurement_type(s: &str) -> crate::uncertainty_quadtree::MeasurementType {
    match s {
        "KernalDensity" => crate::uncertainty_quadtree::MeasurementType::KernalDensity,
        _ => crate::uncertainty_quadtree::MeasurementType::Variance,
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FilterSnapshot {
    pub attribute: String,
    pub operation: String,
    pub comparitor_raw: String,
}

pub struct PendingSnapshotRestore {
    /// Layers not yet started loading.
    pub queue: VecDeque<LayerSnapshot>,
    /// Settings for the layer currently being loaded.
    pub pending_layer_settings: Option<LayerSnapshot>,
    pub viewport: ViewportSnapshot,
    pub display: DisplaySnapshot,
    pub analysis: AnalysisSnapshot,
}

pub fn parse_comparitor_best(raw: &str) -> AttributeValue {
    if let Ok(i) = raw.parse::<i64>() {
        return AttributeValue::Integer(i);
    }
    if let Ok(f) = raw.parse::<f64>() {
        return AttributeValue::Float(f);
    }
    AttributeValue::Text(raw.to_string())
}

pub fn filter_snapshot_to_filter(f: &FilterSnapshot) -> LayerAttributeFilter {
    LayerAttributeFilter {
        attribute: Some(f.attribute.clone()),
        operation: Some(match f.operation.as_str() {
            ">" => FilterOperation::GreaterThan,
            "<" => FilterOperation::LessThan,
            _ => FilterOperation::Equal,
        }),
        comparitor: parse_comparitor_best(&f.comparitor_raw),
        comparitor_raw: f.comparitor_raw.clone(),
    }
}

pub fn filter_to_snapshot(f: &LayerAttributeFilter) -> Option<FilterSnapshot> {
    Some(FilterSnapshot {
        attribute: f.attribute.clone()?,
        operation: f.operation.as_ref()?.to_string(),
        comparitor_raw: f.comparitor_raw.clone(),
    })
}

pub fn filter_logic_to_str(logic: FilterLogic) -> String {
    match logic {
        FilterLogic::And => "And".to_string(),
        FilterLogic::Or => "Or".to_string(),
    }
}

pub fn str_to_filter_logic(s: &str) -> FilterLogic {
    match s {
        "Or" => FilterLogic::Or,
        _ => FilterLogic::And,
    }
}
