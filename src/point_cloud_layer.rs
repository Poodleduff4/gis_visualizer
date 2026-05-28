use crate::{hilbert_r_tree::HilbertRTree, quadtree::Quadtree, spatial_index::SpatialIndex};

#[derive(Default)]
pub struct PointCloudLayer {
    pub points: Vec<[f64; 2]>,
    // pub attributes: Vec<HashMap<String, AttributeValue>>,  // TODO: replace with columnar Vec<Vec<AttributeValue>>
    pub field_names: Vec<String>,
    pub index: Option<Box<dyn SpatialIndex>>,
    pub bbox: Option<[f64; 4]>,
}
impl PointCloudLayer {
    fn ensure_bbox(&mut self) {
        if self.bbox.is_some() || self.points.is_empty() {
            return;
        }
        let mut xmin = f64::MAX;
        let mut ymin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymax = f64::MIN;
        for p in &self.points {
            xmin = xmin.min(p[0]);
            ymin = ymin.min(p[1]);
            xmax = xmax.max(p[0]);
            ymax = ymax.max(p[1]);
        }
        self.bbox = Some([xmin, ymin, xmax, ymax]);
    }

    pub fn rebuild_quadtree(&mut self, capacity: usize) {
        self.ensure_bbox();
        if let Some(bbox) = self.bbox {
            let mut qt: Box<dyn SpatialIndex> = Box::new(Quadtree::new(bbox, capacity));
            for (i, p) in self.points.iter().enumerate() {
                qt.insert(i, [p[0], p[1], p[0], p[1]]);
            }
            self.index = Some(qt);
        }
    }

    pub fn rebuild_hilbert_tree(&mut self, order: u32) {
        self.ensure_bbox();
        if let Some(bbox) = self.bbox {
            let mut ht: Box<dyn SpatialIndex> = Box::new(HilbertRTree::new(bbox, order));
            for (i, p) in self.points.iter().enumerate() {
                ht.insert(i, [p[0], p[1], p[0], p[1]]);
            }
            self.index = Some(ht);
        }
    }

    pub fn hit_test(&self, x: f64, y: f64, tolerance: f64) -> Option<usize> {
        if let Some(index) = &self.index {
            let results =
                index.search(&[x - tolerance, y - tolerance, x + tolerance, y + tolerance]);
            if !results.is_empty() {
                let mut b_dist = f64::MAX;
                let mut b_idx: Option<usize> = None;
                for idx in results.iter() {
                    let p = self.points[*idx];
                    let dist = ((x - p[0]).powf(2.) + (y - p[1]).powf(2.)).sqrt();
                    if dist < b_dist {
                        b_idx = Some(*idx);
                        b_dist = dist;
                    }
                }
                return b_idx;
            } else {
                return None;
            }
        } else {
            let mut b_dist = f64::MAX;
            let mut b_idx: Option<usize> = None;
            for (idx, p) in self.points.iter().enumerate() {
                let dist = ((x - p[0]).powf(2.) + (y - p[1]).powf(2.)).sqrt();
                if dist < b_dist {
                    b_idx = Some(idx);
                    b_dist = dist;
                }
            }
            return b_idx;
        }
    }
}
