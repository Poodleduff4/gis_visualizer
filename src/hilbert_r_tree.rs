use crate::{
    hilbert_curve::HilbertCurve,
    spatial_index::{LineSegment, SpatialIndex},
};

struct HilbertEntry {
    lhv: u64,
    id: usize,
    centroid: [f64; 2],
    rect: [f64; 4],
}

pub struct HilbertRTree {
    entries: Vec<HilbertEntry>,
    data_rect: [f64; 4],
    order: u32,
}

impl HilbertRTree {
    pub fn new(data_rect: [f64; 4], order: u32) -> Self {
        HilbertRTree {
            entries: Vec::new(),
            data_rect,
            order,
        }
    }

    fn jump_threshold_sq(&self) -> f64 {
        let w = self.data_rect[2] - self.data_rect[0];
        let h = self.data_rect[3] - self.data_rect[1];
        let t = (w * w + h * h).sqrt() * 0.05;
        t * t
    }
}

impl SpatialIndex for HilbertRTree {
    fn insert(&mut self, id: usize, rect: [f64; 4]) {
        let lhv = match HilbertCurve::hilbert_index(self.order, &rect, &self.data_rect) {
            Some(h) => h,
            None => return,
        };
        let centroid = [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0];
        let pos = self.entries.partition_point(|e| e.lhv <= lhv);
        self.entries.insert(
            pos,
            HilbertEntry {
                lhv,
                id,
                centroid,
                rect,
            },
        );
    }

    fn search(&self, rect: &[f64; 4]) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| intersects(&e.rect, rect))
            .map(|e| e.id)
            .collect()
    }

    fn delete(&mut self, id: usize) {
        self.entries.retain(|e| e.id != id);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Hilbert path through feature centroids, skipping long inter-quadrant jumps.
    fn shapes(&self) -> Vec<LineSegment> {
        let threshold_sq = self.jump_threshold_sq();
        self.entries
            .windows(2)
            .filter_map(|w| {
                let [ax, ay] = w[0].centroid;
                let [bx, by] = w[1].centroid;
                let dx = bx - ax;
                let dy = by - ay;
                if dx * dx + dy * dy <= threshold_sq {
                    Some(LineSegment::new([ax, ay], [bx, by]))
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_capacity(&self) -> Option<usize> {
        None
    }
}

fn intersects(r1: &[f64; 4], r2: &[f64; 4]) -> bool {
    r1[0] <= r2[2] && r1[2] >= r2[0] && r1[1] <= r2[3] && r1[3] >= r2[1]
}
