use crate::gis_layer::{AttributeValue, LayerKind, LayerSelection};
use crate::point_cloud_layer::AttributeColumn;

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
    if xs.len() != ys.len() || xs.is_empty() {
        return None;
    }
    let n = xs.len();
    let x_mean = xs.iter().sum::<f64>() / n as f64;
    let y_mean = ys.iter().sum::<f64>() / n as f64;
    let cov = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>()
        / n as f64;
    let x_var = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>() / n as f64;
    let y_var = ys.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n as f64;
    let x_std = x_var.sqrt();
    let y_std = y_var.sqrt();
    let pearson_r = if x_std > 1e-12 && y_std > 1e-12 {
        cov / (x_std * y_std)
    } else {
        0.0
    };

    let scatter_points: Vec<[f64; 2]> = if n <= max_plot_points {
        xs.iter().zip(ys.iter()).map(|(&x, &y)| [x, y]).collect()
    } else {
        let step = n / max_plot_points;
        xs.iter()
            .zip(ys.iter())
            .step_by(step)
            .map(|(&x, &y)| [x, y])
            .collect()
    };

    Some(SelectionBivariate {
        x_field: x_field.to_string(),
        y_field: y_field.to_string(),
        n,
        pearson_r,
        covariance: cov,
        x_mean,
        y_mean,
        x_std,
        y_std,
        scatter_points,
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
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return None;
    }
    let mut counts = vec![0u32; bin_count];
    for v in &values {
        let idx = ((v - min) / (max - min) * bin_count as f64) as usize;
        counts[idx.min(bin_count - 1)] += 1;
    }
    let bin_width = (max - min) / bin_count as f64;
    let bin_edges: Vec<f64> = (0..=bin_count).map(|i| min + i as f64 * bin_width).collect();
    Some(SelectionHistogram {
        field: field.to_string(),
        counts,
        bin_edges,
        min,
        max,
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
    let n = values.len();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |p: f64| -> f64 {
        let idx = ((n - 1) as f64 * p) as usize;
        sorted[idx]
    };
    Some(SelectionFieldStats {
        count: n,
        min,
        max,
        mean,
        std_dev,
        p25: percentile(0.25),
        p50: percentile(0.50),
        p75: percentile(0.75),
    })
}
