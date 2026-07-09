use rand::seq::IteratorRandom;

use crate::spatial_index::SpatialIndex;
use crate::uncertainty_quadtree::{kernal_density, variance, MeasurementType, UncertaintyMeasure};

#[derive(Clone, Copy, PartialEq)]
pub enum HeatmapMetric {
    Density,
    Unpredictability,
}

pub struct ScoredCell {
    pub bbox: [f64; 4],
    pub density: f32,
    pub unpredictability: f32,
}

pub struct HeatmapLayer {
    pub cells: Vec<ScoredCell>,
    /// Raw (pre-normalization) max point count across cells — legend range for `Density`.
    pub max_density: f32,
    /// Raw (pre-normalization) max variance/entropy across cells — legend range for `Unpredictability`.
    pub max_unpredictability: f32,
    pub measurement_type: MeasurementType,
}

impl HeatmapLayer {
    /// Builds a heatmap from `index`'s leaf cells, scoring each cell by both
    /// point density and attribute unpredictability (variance or entropy,
    /// per `measurement_type`) at the same time. `values` is the attribute
    /// column for the analyzed attribute, indexed by point id.
    pub fn build(index: &SpatialIndex, values: &[f64], measurement_type: MeasurementType) -> Self {
        let leaf_cells = index.heatmap_cells();
        let mut rng = rand::rng();

        let mut raw: Vec<([f64; 4], f32, f32)> = Vec::with_capacity(leaf_cells.len());
        for cell in &leaf_cells {
            let density = cell.point_ids.len() as f32;

            // Reuse the node's already-computed uncertainty when available
            // (UncertaintyQuadtree) so a clicked cell's value always matches
            // what the heatmap shows, instead of resampling independently.
            let unpredictability = if let Some(u) = &cell.uncertainty {
                match u {
                    UncertaintyMeasure::Variance { variance, .. } => *variance as f32,
                    UncertaintyMeasure::KernalDensity { entropy } => *entropy as f32,
                }
            } else {
                let sample_size = cell.point_ids.len().min(200);
                let sample: Vec<f64> = cell
                    .point_ids
                    .iter()
                    .choose_multiple(&mut rng, sample_size)
                    .into_iter()
                    .filter_map(|&id| values.get(id).copied())
                    .collect();

                if sample.is_empty() {
                    0.0
                } else {
                    match measurement_type {
                        MeasurementType::Variance => match variance(&sample) {
                            UncertaintyMeasure::Variance { variance, .. } => variance as f32,
                            _ => 0.0,
                        },
                        MeasurementType::KernalDensity => match kernal_density(sample) {
                            UncertaintyMeasure::KernalDensity { entropy } => entropy as f32,
                            _ => 0.0,
                        },
                    }
                }
            };

            raw.push((cell.bbox, density, unpredictability));
        }

        let max_density = raw.iter().map(|(_, d, _)| *d).fold(0.0_f32, f32::max);
        let max_unpredictability = raw.iter().map(|(_, _, u)| *u).fold(0.0_f32, f32::max);

        let cells = raw
            .into_iter()
            .map(|(bbox, density, unpredictability)| ScoredCell {
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
            })
            .collect();

        Self {
            cells,
            max_density,
            max_unpredictability,
            measurement_type,
        }
    }
}
