use bitvec::vec::BitVec;
use rstar::{primitives::GeomWithData, RTree, AABB};

use crate::point_cloud_layer::{AttributeColumn, PointCloudLayer};
use crate::spatial_index::SpatialIndex;
use crate::stats_core;

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
    let b = stats_core::bivariate(&xs, &ys, max_plot_points)?;
    Some(BivariateStats {
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
    let s = stats_core::basic_stats(&values)?;
    Some(FieldStats {
        count: all_count,
        filtered_count: s.count,
        min: s.min,
        max: s.max,
        mean: s.mean,
        std_dev: s.std_dev,
        p25: s.p25,
        p50: s.p50,
        p75: s.p75,
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
    let h = stats_core::histogram(&values, bin_count)?;
    Some(HistogramState {
        field: field.to_string(),
        counts: h.counts,
        bin_edges: h.bin_edges,
        min: h.min,
        max: h.max,
        range_lo: h.min,
        range_hi: h.max,
        filtered_only,
    })
}

// ── Spatial analysis ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum LisaCluster {
    HighHigh,
    LowLow,
    HighLow,
    LowHigh,
}

pub struct LisaPoint {
    pub local_i: f64,
    pub z_score: f64,
    pub cluster: LisaCluster,
}

/// Extract numeric field values for all points (unfiltered). Returns None for Text columns.
pub fn extract_field_values(pc: &PointCloudLayer, field: &str) -> Option<Vec<f64>> {
    col_values(pc, field, false)
}

// Build a fallback RTree from the filtered subset of points when no reusable index exists.
fn build_temp_rtree(
    points: &[(u32, [f64; 2])],
    filter_mask: &BitVec,
) -> RTree<GeomWithData<[f64; 2], usize>> {
    RTree::bulk_load(
        points
            .iter()
            .enumerate()
            .filter(|(i, _)| filter_mask[*i])
            .map(|(i, (_, p))| GeomWithData::new(*p, i))
            .collect(),
    )
}

/// Welford's online mean+variance for a stream of neighbor values looked up via index query.
/// Returns (variance, n) or None if fewer than 2 neighbors.
#[inline]
fn welford_variance_from_index(
    pos: [f64; 2],
    radius: f64,
    values: &[f64],
    filter_mask: &BitVec,
    index: &SpatialIndex,
) -> Option<f64> {
    let bbox = [pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius];
    let mut n = 0usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for j in index.search(&bbox) {
        if !filter_mask[j] {
            continue;
        }
        let x = values[j];
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        m2 += delta * (x - mean);
    }
    if n < 2 { None } else { Some(m2 / n as f64) }
}

#[inline]
fn welford_variance_from_rtree(
    pos: [f64; 2],
    radius: f64,
    values: &[f64],
    filter_mask: &BitVec,
    tree: &RTree<GeomWithData<[f64; 2], usize>>,
) -> Option<f64> {
    let mut n = 0usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for e in tree.locate_in_envelope(&AABB::from_corners(
        [pos[0] - radius, pos[1] - radius],
        [pos[0] + radius, pos[1] + radius],
    )) {
        let j = e.data;
        if !filter_mask[j] {
            continue;
        }
        let x = values[j];
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        m2 += delta * (x - mean);
    }
    if n < 2 { None } else { Some(m2 / n as f64) }
}

/// Per-point variance of attribute values within `radius` (in data units).
///
/// Reuses `index` when available (avoids rebuilding spatial structure).
/// Uses Welford's algorithm — no per-point Vec allocation.
/// Designed to run in a background thread: takes owned/Arc-cloneable inputs.
pub fn local_variance_inner(
    points: &[(u32, [f64; 2])],
    filter_mask: &BitVec,
    values: &[f64],
    radius: f64,
    index: Option<&SpatialIndex>,
) -> Vec<Option<f64>> {
    match index {
        Some(idx) => points
            .iter()
            .enumerate()
            .map(|(i, (_, p))| {
                if !filter_mask[i] { return None; }
                welford_variance_from_index(*p, radius, values, filter_mask, idx)
            })
            .collect(),
        None => {
            let tree = build_temp_rtree(points, filter_mask);
            points
                .iter()
                .enumerate()
                .map(|(i, (_, p))| {
                    if !filter_mask[i] { return None; }
                    welford_variance_from_rtree(*p, radius, values, filter_mask, &tree)
                })
                .collect()
        }
    }
}

/// Per-point Local Moran's I (LISA).
///
/// For each point i:
///   z_i = (v_i - global_mean) / global_std
///   lag_i = mean(z_j) for all neighbors j ≠ i within `radius`
///   local_I_i = z_i × lag_i
///
/// Cluster types:
///   HH (red)    — high surrounded by high  → spatial cluster
///   LL (blue)   — low surrounded by low    → spatial cluster
///   HL (orange) — high surrounded by low   → spatial outlier / low explainability
///   LH (cyan)   — low surrounded by high   → spatial outlier / low explainability
///
/// Reuses `index` when available. Designed to run in a background thread.
pub fn lisa_inner(
    points: &[(u32, [f64; 2])],
    filter_mask: &BitVec,
    values: &[f64],
    radius: f64,
    index: Option<&SpatialIndex>,
) -> Option<Vec<Option<LisaPoint>>> {
    // Global mean + std (one pass, Welford)
    let mut n = 0usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, _) in points.iter().enumerate() {
        if !filter_mask[i] { continue; }
        let x = values[i];
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        m2 += delta * (x - mean);
    }
    if n < 2 { return None; }
    let std = (m2 / n as f64).sqrt();
    if std < 1e-12 { return None; }

    let z: Vec<f64> = values.iter().map(|v| (v - mean) / std).collect();

    let compute = |tree_query: &dyn Fn([f64; 2]) -> Vec<usize>| -> Vec<Option<LisaPoint>> {
        points
            .iter()
            .enumerate()
            .map(|(i, (_, p))| {
                if !filter_mask[i] { return None; }
                let neighbors = tree_query(*p);
                let k = neighbors.len() as f64;
                if k == 0.0 { return None; }
                let lag = neighbors.iter().map(|&j| z[j]).sum::<f64>() / k;
                let local_i = z[i] * lag;
                let cluster = match (z[i] > 0.0, local_i > 0.0) {
                    (true, true)   => LisaCluster::HighHigh,
                    (false, true)  => LisaCluster::LowLow,
                    (true, false)  => LisaCluster::HighLow,
                    (false, false) => LisaCluster::LowHigh,
                };
                Some(LisaPoint { local_i, z_score: z[i], cluster })
            })
            .collect()
    };

    Some(match index {
        Some(idx) => compute(&|p: [f64; 2]| {
            let bbox = [p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius];
            idx.search(&bbox)
                .into_iter()
                .filter(|&j| filter_mask[j])
                .collect()
        }),
        None => {
            let tree = build_temp_rtree(points, filter_mask);
            compute(&|p: [f64; 2]| {
                tree.locate_in_envelope(&AABB::from_corners(
                    [p[0] - radius, p[1] - radius],
                    [p[0] + radius, p[1] + radius],
                ))
                .filter_map(|e| if filter_mask[e.data] { Some(e.data) } else { None })
                .collect()
            })
        }
    })
}

// Convenience wrappers used when calling synchronously (e.g. small datasets).
pub fn compute_local_variance(
    pc: &PointCloudLayer,
    field: &str,
    radius: f64,
) -> Option<Vec<Option<f64>>> {
    let values = extract_field_values(pc, field)?;
    Some(local_variance_inner(
        &pc.points,
        &pc.filter_mask,
        &values,
        radius,
        pc.index.as_deref(),
    ))
}

pub fn compute_lisa(
    pc: &PointCloudLayer,
    field: &str,
    radius: f64,
) -> Option<Vec<Option<LisaPoint>>> {
    let values = extract_field_values(pc, field)?;
    lisa_inner(&pc.points, &pc.filter_mask, &values, radius, pc.index.as_deref())
}
