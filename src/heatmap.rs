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

    /// Wraps a KDE grid (from `kde::build_kde_grid`) as a `HeatmapLayer` so it
    /// can reuse `show_quadtree_heatmap`'s rendering — `cells` are already
    /// scored by density only (no per-cell variance/entropy/attribute-mean).
    pub fn from_kde_cells(cells: Vec<([f64; 4], f32)>, attribute_name: String) -> Self {
        let max_density = cells.iter().map(|(_, v)| *v).fold(0.0_f32, f32::max);
        let cells = cells
            .into_iter()
            .map(|(bbox, v)| ScoredCell {
                bbox,
                density: if max_density > 0.0 { v / max_density } else { 0.0 },
                unpredictability: 0.0,
                attribute_mean: 0.0,
            })
            .collect();

        Self {
            cells,
            max_density,
            max_unpredictability: 0.0,
            min_attribute_value: 0.0,
            max_attribute_value: 0.0,
            attribute_name,
            measurement_type: MeasurementType::Variance,
        }
    }

    /// Wraps a uniform-grid binning (`gridbin::build_gridbin`) as a
    /// `HeatmapLayer`, supporting both `Density` (points/area, real density
    /// since every cell is the same size — unlike an adaptive quadtree's
    /// leaves) and `AttributeMean` when the caller supplied per-point values.
    /// No `Unpredictability`: that needs per-cell samples, not just a sum.
    pub fn from_grid_cells(cells: Vec<crate::gridbin::GridBinCell>, attribute_name: String) -> Self {
        let max_density = cells
            .iter()
            .map(|c| c.count as f32)
            .fold(0.0_f32, f32::max);
        let min_attribute_value = cells
            .iter()
            .filter_map(|c| c.mean)
            .fold(f32::MAX, f32::min);
        let max_attribute_value = cells
            .iter()
            .filter_map(|c| c.mean)
            .fold(f32::MIN, f32::max);
        let attribute_range = max_attribute_value - min_attribute_value;

        let cells = cells
            .into_iter()
            .map(|c| ScoredCell {
                bbox: c.bbox,
                density: if max_density > 0.0 {
                    c.count as f32 / max_density
                } else {
                    0.0
                },
                unpredictability: 0.0,
                attribute_mean: match c.mean {
                    Some(m) if attribute_range > 0.0 => (m - min_attribute_value) / attribute_range,
                    _ => 0.0,
                },
            })
            .collect();

        Self {
            cells,
            max_density,
            max_unpredictability: 0.0,
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
            measurement_type: MeasurementType::Variance,
        }
    }

    /// Recovers raw (un-normalized) per-cell values for `metric` — inverts the
    /// normalization `build`/`from_kde_cells` applies, so a saved snapshot
    /// carries physical-unit magnitudes rather than 0..1 scores.
    pub fn raw_cells(&self, metric: HeatmapMetric) -> Vec<([f64; 4], f32)> {
        let attr_range = self.max_attribute_value - self.min_attribute_value;
        self.cells
            .iter()
            .map(|c| {
                let value = match metric {
                    HeatmapMetric::Density => c.density * self.max_density,
                    HeatmapMetric::Unpredictability => {
                        c.unpredictability * self.max_unpredictability
                    }
                    HeatmapMetric::AttributeMean => {
                        c.attribute_mean * attr_range + self.min_attribute_value
                    }
                };
                (c.bbox, value)
            })
            .collect()
    }

    /// Legend/units label for `metric`, for naming a saved snapshot.
    pub fn metric_label(&self, metric: HeatmapMetric) -> String {
        match metric {
            HeatmapMetric::Density => "density (points/cell)".to_string(),
            HeatmapMetric::Unpredictability => match self.measurement_type {
                MeasurementType::Variance => "unpredictability (variance)".to_string(),
                MeasurementType::KernalDensity => "unpredictability (entropy)".to_string(),
            },
            HeatmapMetric::AttributeMean => format!("{} (avg)", self.attribute_name),
        }
    }
}

/// Kind of analysis a `SavedHeatmap` snapshot came from — just for the layer
/// panel's icon/label, doesn't affect export/render logic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeatmapKind {
    Quadtree,
    Kde,
    KdeEntropy,
    GridBin,
}

/// A named, persisted snapshot of a built heatmap/KDE grid, stored under the
/// layer it was built for (mirrors `LayerSelection`). Cells are raw
/// (physical-unit) values, not the 0..1 scores `HeatmapLayer` renders with,
/// so it can be rasterized to an exact-magnitude GeoTIFF or raster layer.
pub struct SavedHeatmap {
    pub name: String,
    pub kind: HeatmapKind,
    /// Enclosing bbox of every cell — the default export/promote extent.
    pub bbox: [f64; 4],
    pub cells: Vec<([f64; 4], f32)>,
    pub units: String,
}

impl SavedHeatmap {
    pub fn new(name: String, kind: HeatmapKind, cells: Vec<([f64; 4], f32)>, units: String) -> Self {
        let bbox = cells.iter().fold(
            [f64::MAX, f64::MAX, f64::MIN, f64::MIN],
            |acc, (b, _)| {
                [
                    acc[0].min(b[0]),
                    acc[1].min(b[1]),
                    acc[2].max(b[2]),
                    acc[3].max(b[3]),
                ]
            },
        );
        Self { name, kind, bbox, cells, units }
    }
}

/// Cap so a too-fine `cell_size` can't allocate unbounded memory; doubles the
/// cell size until the grid fits instead of failing outright.
const RASTERIZE_MAX_CELLS: usize = 16_000_000;

/// Rasterizes arbitrary (possibly irregular, e.g. quadtree-leaf) `cells` onto
/// a uniform grid covering `bbox`. Each source cell writes to every output
/// pixel its bbox overlaps, so cost is linear in output pixels (source cells
/// tile space without overlap) rather than pixels × cells — same technique as
/// `kde::build_kde_grid`'s per-point cell range. Returns
/// `(width, height, actual_cell_size, values)`; `values` is row-major, row 0
/// = north edge, `NAN` where no source cell covered that pixel.
pub fn rasterize_cells(
    cells: &[([f64; 4], f32)],
    bbox: [f64; 4],
    cell_size: f64,
) -> (usize, usize, f64, Vec<f32>) {
    let [xmin, ymin, xmax, ymax] = bbox;
    let mut cell_size = cell_size.max(1e-12);
    let mut width = (((xmax - xmin) / cell_size).ceil() as usize).max(1);
    let mut height = (((ymax - ymin) / cell_size).ceil() as usize).max(1);
    while width.saturating_mul(height) > RASTERIZE_MAX_CELLS {
        cell_size *= 2.0;
        width = (((xmax - xmin) / cell_size).ceil() as usize).max(1);
        height = (((ymax - ymin) / cell_size).ceil() as usize).max(1);
    }

    let mut grid = vec![f32::NAN; width * height];
    for (cbbox, value) in cells {
        let col0 = (((cbbox[0] - xmin) / cell_size).floor().max(0.0)) as usize;
        let col1 = ((((cbbox[2] - xmin) / cell_size).ceil()) as usize).min(width);
        let row0 = (((ymax - cbbox[3]) / cell_size).floor().max(0.0)) as usize;
        let row1 = ((((ymax - cbbox[1]) / cell_size).ceil()) as usize).min(height);
        for row in row0..row1 {
            for col in col0..col1 {
                grid[row * width + col] = *value;
            }
        }
    }
    (width, height, cell_size, grid)
}
