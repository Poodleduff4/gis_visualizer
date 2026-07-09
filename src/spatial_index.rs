use crate::{
    hilbert_r_tree::HilbertRTree, quadtree::Quadtree, rtree_index::SpatialTree,
    uncertainty_quadtree::{UncertaintyMeasure, UncertaintyQuadtree},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Quadtree,
    Hilbert,
}

pub enum SpatialIndex {
    Quadtree(Quadtree),
    HilbertCurve(HilbertRTree),
    UncertaintyQuadtree(UncertaintyQuadtree),
    RTree(SpatialTree),
}

impl SpatialIndex {
    pub fn get_capacity(&self) -> Option<usize> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.get_capacity(),
            SpatialIndex::HilbertCurve(ht) => ht.get_capacity(),
            SpatialIndex::UncertaintyQuadtree(_) => None,
            SpatialIndex::RTree(rt) => Some(rt.size()),
        }
    }

    pub fn insert(&mut self, id: usize, rect: [f64; 4]) {
        match self {
            SpatialIndex::Quadtree(qt) => qt.insert(id, rect),
            SpatialIndex::HilbertCurve(ht) => ht.insert(id, rect),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.insert(id, rect, 0.0),
            SpatialIndex::RTree(_) => {} // built via bulk_load, not incremental insert
        }
    }

    pub fn search(&self, rect: &[f64; 4]) -> Vec<usize> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.search(rect),
            SpatialIndex::HilbertCurve(ht) => ht.search(rect),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.search(rect),
            SpatialIndex::RTree(rt) => crate::rtree_index::query_bbox(rt, *rect).collect(),
        }
    }

    pub fn delete(&mut self, id: usize) {
        match self {
            SpatialIndex::Quadtree(qt) => qt.delete(id),
            SpatialIndex::HilbertCurve(ht) => ht.delete(id),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.delete(id),
            SpatialIndex::RTree(_) => {}
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SpatialIndex::Quadtree(qt) => qt.len(),
            SpatialIndex::HilbertCurve(ht) => ht.len(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.len(),
            SpatialIndex::RTree(rt) => rt.size(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            SpatialIndex::Quadtree(qt) => qt.is_empty(),
            SpatialIndex::HilbertCurve(ht) => ht.is_empty(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.is_empty(),
            SpatialIndex::RTree(rt) => rt.size() == 0,
        }
    }

    pub fn clear(&mut self) {
        match self {
            SpatialIndex::Quadtree(qt) => qt.clear(),
            SpatialIndex::HilbertCurve(ht) => ht.clear(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.clear(),
            SpatialIndex::RTree(_) => {}
        }
    }

    pub fn shapes(&self) -> Vec<LineSegment> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.shapes(),
            SpatialIndex::HilbertCurve(ht) => ht.shapes(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.shapes(),
            SpatialIndex::RTree(_) => vec![],
        }
    }

    // pub fn points_in_bbox(&self, points: &[[f64; 2]], bbox: [f64; 4]) -> Vec<[f64; 2]> {
    //     self.search(&bbox).into_iter().map(|i| points[i]).collect()
    // }

    pub fn points_idx_in_bbox(&self, bbox: [f64; 4]) -> Vec<u32> {
        self.search(&bbox).into_iter().map(|i| i as u32).collect()
    }

    pub fn heatmap_cells(&self) -> Vec<HeatmapCell> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.heatmap_cells(),
            SpatialIndex::HilbertCurve(_) => vec![],
            SpatialIndex::UncertaintyQuadtree(uq) => uq.heatmap_cells(),
            SpatialIndex::RTree(_) => vec![],
        }
    }

    pub fn leaf_bbox_at(&self, pos: [f64; 2]) -> Option<[f64; 4]> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.leaf_bbox_at(pos),
            SpatialIndex::HilbertCurve(_) => None,
            SpatialIndex::UncertaintyQuadtree(uq) => uq.leaf_bbox_at(pos),
            SpatialIndex::RTree(_) => None,
        }
    }
}

pub struct HeatmapCell {
    pub bbox: [f64; 4],
    pub depth: usize,
    pub point_ids: Vec<usize>,
    /// Node's already-computed uncertainty (UncertaintyQuadtree only). When
    /// present, reuse it instead of resampling so a clicked cell's value
    /// (`ClickTarget::GridCell`) always matches what the heatmap shows.
    pub uncertainty: Option<UncertaintyMeasure>,
}

pub struct LineSegment {
    pub start: [f64; 2],
    pub end: [f64; 2],
}

impl LineSegment {
    pub fn new(start: [f64; 2], end: [f64; 2]) -> Self {
        LineSegment { start, end }
    }
}
