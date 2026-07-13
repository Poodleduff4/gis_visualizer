use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;

use crate::spatial_index::SpatialIndex;
use crate::uncertainty_quadtree::{kernal_density, variance, MeasurementType, UncertaintyMeasure};

#[derive(Clone, Copy, PartialEq)]
pub enum HeatmapMetric {
    Density,
    Unpredictability,
    /// Average value of the selected attribute within each cell (e.g. mean
    /// fare_cost) — higher average maps to more red via `heat_color`.
    AttributeMean,
}

pub struct ScoredCell {
    pub bbox: [f64; 4],
    pub density: f32,
    pub unpredictability: f32,
    /// Normalized (0..1) average of the selected attribute within the cell.
    pub attribute_mean: f32,
}

pub struct HeatmapLayer {
    pub cells: Vec<ScoredCell>,
    /// Raw (pre-normalization) max point count across cells — legend range for `Density`.
    pub max_density: f32,
    /// Raw (pre-normalization) max variance/entropy across cells — legend range for `Unpredictability`.
    pub max_unpredictability: f32,
    /// Raw (pre-normalization) min/max cell average — legend range for `AttributeMean`.
    pub min_attribute_value: f32,
    pub max_attribute_value: f32,
    /// Name of the attribute averaged for `AttributeMean` (legend title).
    pub attribute_name: String,
    pub measurement_type: MeasurementType,
}

impl HeatmapLayer {
    /// Builds a heatmap from `index`'s leaf cells, scoring each cell by
    /// point density, attribute unpredictability (variance or entropy, per
    /// `measurement_type`), and the attribute's average value, all at once.
    /// `values` is the attribute column for the analyzed attribute, indexed
    /// by point id; `attribute_name` is only used for the legend label.
    pub fn build(
        index: &SpatialIndex,
        values: &[f64],
        measurement_type: MeasurementType,
        attribute_name: String,
    ) -> Self {
        let leaf_cells = index.heatmap_cells();
        // Seeded (not rand::rng()'s thread-local entropy) so rebuilding the
        // same index twice — e.g. re-running "Build Quadtree" on unchanged
        // data — samples the same points and produces the same colors.
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);

        let mut raw: Vec<([f64; 4], f32, f32, f32)> = Vec::with_capacity(leaf_cells.len());
        for cell in &leaf_cells {
            let density = cell.point_ids.len() as f32;

            // Exact, not sampled: it's cheap (one pass already touching every
            // id in the cell) and "average fare cost" should be the real
            // average, not a noisy estimate that shifts between rebuilds.
            let attribute_mean = {
                let vals: Vec<f64> = cell
                    .point_ids
                    .iter()
                    .filter_map(|&id| values.get(id).copied())
                    .collect();
                if vals.is_empty() {
                    0.0
                } else {
                    (vals.iter().sum::<f64>() / vals.len() as f64) as f32
                }
            };

            let sample_size = cell.point_ids.len().min(200);
            let sample: Vec<f64> = cell
                .point_ids
                .iter()
                .choose_multiple(&mut rng, sample_size)
                .into_iter()
                .filter_map(|&id| values.get(id).copied())
                .collect();

            // Reuse the node's already-computed uncertainty when available
            // (UncertaintyQuadtree) so a clicked cell's value always matches
            // what the heatmap shows, instead of resampling independently.
            let unpredictability = if let Some(u) = &cell.uncertainty {
                match u {
                    UncertaintyMeasure::Variance { variance, .. } => *variance as f32,
                    UncertaintyMeasure::KernalDensity { entropy } => *entropy as f32,
                }
            } else if sample.is_empty() {
                0.0
            } else {
                match measurement_type {
                    MeasurementType::Variance => match variance(&sample) {
                        UncertaintyMeasure::Variance { variance, .. } => variance as f32,
                        _ => 0.0,
                    },
                    MeasurementType::KernalDensity => match kernal_density(sample.clone()) {
                        UncertaintyMeasure::KernalDensity { entropy } => entropy as f32,
                        _ => 0.0,
                    },
                }
            };

            raw.push((cell.bbox, density, unpredictability, attribute_mean));
        }

        let max_density = raw.iter().map(|(_, d, _, _)| *d).fold(0.0_f32, f32::max);
        let max_unpredictability = raw.iter().map(|(_, _, u, _)| *u).fold(0.0_f32, f32::max);
        let min_attribute_value = raw
            .iter()
            .map(|(_, _, _, m)| *m)
            .fold(f32::MAX, f32::min);
        let max_attribute_value = raw
            .iter()
            .map(|(_, _, _, m)| *m)
            .fold(f32::MIN, f32::max);
        let attribute_range = max_attribute_value - min_attribute_value;

        let cells = raw
            .into_iter()
            .map(|(bbox, density, unpredictability, attribute_mean)| ScoredCell {
                bbox,
                density: if max_density > 0.0 {
                    density / max_density
                } else {
                    0.0
                },
                unpredictability: if max_unpredictability > 0.0 {
                    unpredictability / max_unpredictability
                } else {
                    0.0
                },
                attribute_mean: if attribute_range > 0.0 {
                    (attribute_mean - min_attribute_value) / attribute_range
                } else {
                    0.0
                },
            })
            .collect();

        Self {
            cells,
            max_density,
            max_unpredictability,
            min_attribute_value: if min_attribute_value.is_finite() {
                min_attribute_value
            } else {
                0.0
            },
            max_attribute_value: if max_attribute_value.is_finite() {
                max_attribute_value
            } else {
                0.0
            },
            attribute_name,
            measurement_type,
        }
    }
}
