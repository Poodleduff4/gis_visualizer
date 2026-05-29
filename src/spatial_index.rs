use crate::{
    hilbert_r_tree::HilbertRTree, quadtree::Quadtree, uncertainty_quadtree::UncertaintyQuadtree,
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
}

impl SpatialIndex {
    pub fn get_capacity(&self) -> Option<usize> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.get_capacity(),
            SpatialIndex::HilbertCurve(ht) => ht.get_capacity(),
            SpatialIndex::UncertaintyQuadtree(uncertainty_quadtree) => None,
        }
    }

    pub fn insert(&mut self, id: usize, rect: [f64; 4]) {
        match self {
            SpatialIndex::Quadtree(qt) => qt.insert(id, rect),
            SpatialIndex::HilbertCurve(ht) => ht.insert(id, rect),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.insert(id, rect, 0.0),
        }
    }

    pub fn search(&self, rect: &[f64; 4]) -> Vec<usize> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.search(rect),
            SpatialIndex::HilbertCurve(ht) => ht.search(rect),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.search(rect),
        }
    }

    pub fn delete(&mut self, id: usize) {
        match self {
            SpatialIndex::Quadtree(qt) => qt.delete(id),
            SpatialIndex::HilbertCurve(ht) => ht.delete(id),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.delete(id),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SpatialIndex::Quadtree(qt) => qt.len(),
            SpatialIndex::HilbertCurve(ht) => ht.len(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            SpatialIndex::Quadtree(qt) => qt.is_empty(),
            SpatialIndex::HilbertCurve(ht) => ht.is_empty(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.is_empty(),
        }
    }

    pub fn clear(&mut self) {
        match self {
            SpatialIndex::Quadtree(qt) => qt.clear(),
            SpatialIndex::HilbertCurve(ht) => ht.clear(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.clear(),
        }
    }

    pub fn shapes(&self) -> Vec<LineSegment> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.shapes(),
            SpatialIndex::HilbertCurve(ht) => ht.shapes(),
            SpatialIndex::UncertaintyQuadtree(uq) => uq.shapes(),
        }
    }

    pub fn heatmap_cells(&self) -> Vec<HeatmapCell> {
        match self {
            SpatialIndex::Quadtree(qt) => qt.heatmap_cells(),
            SpatialIndex::HilbertCurve(_) => vec![],
            SpatialIndex::UncertaintyQuadtree(uq) => uq.heatmap_cells(),
        }
    }
}

pub struct HeatmapCell {
    pub bbox: [f64; 4],
    pub depth: usize,
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
