use crate::spatial_index::{HeatmapCell, SpatialIndex};

pub struct HeatmapLayer {
    pub cells: Vec<HeatmapCell>,
}

impl HeatmapLayer {
    pub fn build_from_spatial_index(index: &dyn SpatialIndex) -> Self {
        HeatmapLayer {
            cells: index.heatmap_cells(),
        }
    }
}
