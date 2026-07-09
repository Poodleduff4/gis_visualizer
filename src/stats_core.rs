//! Shared numeric stat math used by both `histogram.rs` (whole-layer stats)
//! and `selection_stats.rs` (stats scoped to a `LayerSelection`). Those two
//! callers differ only in how they extract a `Vec<f64>` from a layer; the
//! math itself was previously duplicated verbatim in both files.

pub struct BasicStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
}

pub fn basic_stats(values: &[f64]) -> Option<BasicStats> {
    if values.is_empty() {
        return None;
    }
    let n = values.len();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |p: f64| -> f64 {
        let idx = ((n - 1) as f64 * p) as usize;
        sorted[idx]
    };
    Some(BasicStats {
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

pub struct Bivariate {
    pub n: usize,
    pub pearson_r: f64,
    pub covariance: f64,
    pub x_mean: f64,
    pub y_mean: f64,
    pub x_std: f64,
    pub y_std: f64,
    pub scatter_points: Vec<[f64; 2]>,
}

pub fn bivariate(xs: &[f64], ys: &[f64], max_plot_points: usize) -> Option<Bivariate> {
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

    Some(Bivariate {
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

pub struct Histogram {
    pub counts: Vec<u32>,
    pub bin_edges: Vec<f64>,
    pub min: f64,
    pub max: f64,
}

pub fn histogram(values: &[f64], bin_count: usize) -> Option<Histogram> {
    if values.is_empty() {
        return None;
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return None;
    }
    let mut counts = vec![0u32; bin_count];
    for v in values {
        let idx = ((v - min) / (max - min) * bin_count as f64) as usize;
        counts[idx.min(bin_count - 1)] += 1;
    }
    let bin_width = (max - min) / bin_count as f64;
    let bin_edges: Vec<f64> = (0..=bin_count).map(|i| min + i as f64 * bin_width).collect();
    Some(Histogram { counts, bin_edges, min, max })
}
