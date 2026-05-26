#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Quadtree,
    Hilbert,
}

pub trait SpatialIndex {
    fn get_capacity(&self) -> Option<usize>;
    fn insert(&mut self, id: usize, rect: [f64; 4]);
    fn search(&self, rect: &[f64; 4]) -> Vec<usize>;
    fn delete(&mut self, id: usize);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&mut self);
    fn shapes(&self) -> Vec<LineSegment>;
    fn heatmap_cells(&self) -> Vec<HeatmapCell> {
        vec![]
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
