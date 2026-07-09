use crate::gis_layer::{AttributeValue, LayerKind, LayerSelection};
use crate::point_cloud_layer::AttributeColumn;
use crate::stats_core;

/// Numeric values of `field` for the ids captured by `sel`, or `None` if the
/// field doesn't exist / isn't numeric. Dispatches on layer kind since Vector
/// features store attributes per-feature (HashMap) while Points store them
/// column-major (AttributeColumn) — same numeric-only rule as
/// `histogram.rs::col_values`.
pub fn field_values_for_selection(
    layer: &LayerKind,
    sel: &LayerSelection,
    field: &str,
) -> Option<Vec<f64>> {
    match layer {
        LayerKind::Vector(gl) => {
            let values: Vec<f64> = sel
                .ids
                .iter()
                .filter_map(|&id| gl.features.get(id))
                .filter_map(|f| match f.attributes.get(field) {
                    Some(AttributeValue::Float(v)) => Some(*v),
                    Some(AttributeValue::Integer(v)) => Some(*v as f64),
                    _ => None,
                })
                .collect();
            if values.is_empty() {
                None
            } else {
                Some(values)
            }
        }
        LayerKind::Points(pc) => {
            let col_idx = pc.field_names.iter().position(|n| n == field)?;
            let col = pc.attributes.get(col_idx)?;
            if matches!(col, AttributeColumn::Text(_)) {
                return None;
            }
            let values: Vec<f64> = sel
                .ids
                .iter()
                .filter_map(|&i| match col {
                    AttributeColumn::Float(v) => v.get(i).copied(),
                    AttributeColumn::Integer(v) => v.get(i).map(|x| *x as f64),
                    AttributeColumn::Text(_) => None,
                })
                .collect();
            if values.is_empty() {
                None
            } else {
                Some(values)
            }
        }
        LayerKind::Raster(_) => None,
    }
}

pub struct SelectionBivariate {
    pub x_field: String,
    pub y_field: String,
    pub n: usize,
    pub pearson_r: f64,
    pub covariance: f64,
    pub x_mean: f64,
    pub y_mean: f64,
    pub x_std: f64,
    pub y_std: f64,
    pub scatter_points: Vec<[f64; 2]>,
}

pub fn compute_selection_bivariate(
    layer: &LayerKind,
    sel: &LayerSelection,
    x_field: &str,
    y_field: &str,
    max_plot_points: usize,
) -> Option<SelectionBivariate> {
    let xs = field_values_for_selection(layer, sel, x_field)?;
    let ys = field_values_for_selection(layer, sel, y_field)?;
    let b = stats_core::bivariate(&xs, &ys, max_plot_points)?;
    Some(SelectionBivariate {
        x_field: x_field.to_string(),
        y_field: y_field.to_string(),
        n: b.n,
        pearson_r: b.pearson_r,
        covariance: b.covariance,
        x_mean: b.x_mean,
        y_mean: b.y_mean,
        x_std: b.x_std,
        y_std: b.y_std,
        scatter_points: b.scatter_points,
    })
}

pub struct SelectionHistogram {
    pub field: String,
    pub counts: Vec<u32>,
    pub bin_edges: Vec<f64>,
    pub min: f64,
    pub max: f64,
}

pub fn compute_selection_histogram(
    layer: &LayerKind,
    sel: &LayerSelection,
    field: &str,
    bin_count: usize,
) -> Option<SelectionHistogram> {
    let values = field_values_for_selection(layer, sel, field)?;
    let h = stats_core::histogram(&values, bin_count)?;
    Some(SelectionHistogram {
        field: field.to_string(),
        counts: h.counts,
        bin_edges: h.bin_edges,
        min: h.min,
        max: h.max,
    })
}

pub struct SelectionFieldStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
}

pub fn compute_selection_field_stats(
    layer: &LayerKind,
    sel: &LayerSelection,
    field: &str,
) -> Option<SelectionFieldStats> {
    let values = field_values_for_selection(layer, sel, field)?;
    let s = stats_core::basic_stats(&values)?;
    Some(SelectionFieldStats {
        count: s.count,
        min: s.min,
        max: s.max,
        mean: s.mean,
        std_dev: s.std_dev,
        p25: s.p25,
        p50: s.p50,
        p75: s.p75,
    })
}
