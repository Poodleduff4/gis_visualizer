use crate::point_cloud_layer::{AttributeColumn, PointCloudLayer};

pub struct BivariateStats {
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

fn col_values(pc: &PointCloudLayer, field: &str, filtered_only: bool) -> Option<Vec<f64>> {
    let col_idx = pc.field_names.iter().position(|n| n == field)?;
    let col = pc.attributes.get(col_idx)?;
    if matches!(col, AttributeColumn::Text(_)) {
        return None;
    }
    let values: Vec<f64> = pc
        .points
        .iter()
        .enumerate()
        .filter(|(i, _)| !filtered_only || pc.filter_mask[*i])
        .map(|(i, _)| match col {
            AttributeColumn::Float(v) => v[i],
            AttributeColumn::Integer(v) => v[i] as f64,
            AttributeColumn::Text(_) => 0.0,
        })
        .collect();
    Some(values)
}

pub fn compute_bivariate(
    pc: &PointCloudLayer,
    x_field: &str,
    y_field: &str,
    filtered_only: bool,
    max_plot_points: usize,
) -> Option<BivariateStats> {
    let xs = col_values(pc, x_field, filtered_only)?;
    let ys = col_values(pc, y_field, filtered_only)?;
    if xs.len() != ys.len() || xs.is_empty() {
        return None;
    }
    let n = xs.len();
    let x_mean = xs.iter().sum::<f64>() / n as f64;
    let y_mean = ys.iter().sum::<f64>() / n as f64;
    let cov = xs.iter().zip(ys.iter()).map(|(x, y)| (x - x_mean) * (y - y_mean)).sum::<f64>() / n as f64;
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
        xs.iter().zip(ys.iter()).step_by(step).map(|(&x, &y)| [x, y]).collect()
    };

    Some(BivariateStats {
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

pub struct FieldStats {
    pub count: usize,
    pub filtered_count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
}

pub fn compute_field_stats(pc: &PointCloudLayer, field: &str, filtered_only: bool) -> Option<FieldStats> {
    let all_count = pc.points.len();
    let values = col_values(pc, field, filtered_only)?;
    if values.is_empty() {
        return None;
    }
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
    Some(FieldStats {
        count: all_count,
        filtered_count: n,
        min,
        max,
        mean,
        std_dev,
        p25: percentile(0.25),
        p50: percentile(0.50),
        p75: percentile(0.75),
    })
}

pub struct HistogramState {
    pub field: String,
    pub counts: Vec<u32>,
    pub bin_edges: Vec<f64>,
    pub min: f64,
    pub max: f64,
    pub range_lo: f64,
    pub range_hi: f64,
    pub filtered_only: bool,
}

pub fn compute_histogram(
    pc: &PointCloudLayer,
    field: &str,
    bin_count: usize,
    filtered_only: bool,
) -> Option<HistogramState> {
    let values = col_values(pc, field, filtered_only)?;
    if values.is_empty() {
        return None;
    }
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
    Some(HistogramState {
        field: field.to_string(),
        counts,
        bin_edges,
        min,
        max,
        range_lo: min,
        range_hi: max,
        filtered_only,
    })
}
