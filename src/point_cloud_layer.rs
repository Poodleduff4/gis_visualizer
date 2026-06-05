use crate::{
    hilbert_r_tree::HilbertRTree,
    quadtree::Quadtree,
    spatial_index::SpatialIndex,
    uncertainty_quadtree::{MeasurementType, UncertaintyQuadtree},
};

pub enum AttributeColumn {
    Text(Vec<String>),
    Integer(Vec<i64>),
    Float(Vec<f64>),
}

impl AttributeColumn {
    pub fn get_display(&self, idx: usize) -> String {
        match self {
            AttributeColumn::Text(v) => v[idx].clone(),
            AttributeColumn::Integer(v) => v[idx].to_string(),
            AttributeColumn::Float(v) => format!("{:.6}", v[idx]),
        }
    }

    pub fn push_default(&mut self) {
        match self {
            AttributeColumn::Text(v) => v.push(String::new()),
            AttributeColumn::Integer(v) => v.push(0),
            AttributeColumn::Float(v) => v.push(0.0),
        }
    }

    pub fn extend_from(&mut self, other: AttributeColumn) {
        match (self, other) {
            (AttributeColumn::Float(a), AttributeColumn::Float(b)) => a.extend(b),
            (AttributeColumn::Integer(a), AttributeColumn::Integer(b)) => a.extend(b),
            (AttributeColumn::Text(a), AttributeColumn::Text(b)) => a.extend(b),
            _ => {}
        }
    }
}

#[derive(Default)]
pub struct PointCloudLayer {
    pub points: Vec<[f64; 2]>,
    pub attributes: Vec<AttributeColumn>,
    pub field_names: Vec<String>,
    pub index: Option<SpatialIndex>,
    pub bbox: Option<[f64; 4]>,
}
impl PointCloudLayer {
    fn ensure_bbox(&mut self) {
        if self.points.is_empty() || self.bbox.is_none() {
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
    }

    pub fn rebuild_quadtree(&mut self, capacity: usize) {
        self.ensure_bbox();
        if let Some(bbox) = self.bbox {
            let mut qt = SpatialIndex::Quadtree(Quadtree::new(bbox, capacity));
            for (i, p) in self.points.iter().enumerate() {
                qt.insert(i, [p[0], p[1], p[0], p[1]]);
            }
            self.index = Some(qt);
        }
    }

    pub fn rebuild_hilbert_tree(&mut self, order: u32) {
        self.ensure_bbox();
        if let Some(bbox) = self.bbox {
            let mut ht = SpatialIndex::HilbertCurve(HilbertRTree::new(bbox, order));
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

    pub fn numeric_field_names(&self) -> Vec<String> {
        self.field_names
            .iter()
            .zip(self.attributes.iter())
            .filter(|(_, col)| !matches!(col, AttributeColumn::Text(_)))
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn rebuild_uncertainty_quadtree(
        &mut self,
        attribute: String,
        threshold: f32,
        measurement_type: MeasurementType,
    ) {
        self.ensure_bbox();
        let Some(bbox) = self.bbox else { return };
        let field_idx = self.field_names.iter().position(|n| n == &attribute);
        let mut uq = UncertaintyQuadtree::new(bbox, attribute.clone(), threshold, measurement_type);
        uq.insert_batch(self.points.iter().enumerate().map(|(i, p)| {
            let value = field_idx
                .and_then(|fi| self.attributes.get(fi))
                .map(|col| match col {
                    AttributeColumn::Float(v) => v.get(i).copied().unwrap_or(0.0),
                    AttributeColumn::Integer(v) => v.get(i).copied().unwrap_or(0) as f64,
                    AttributeColumn::Text(_) => 0.0,
                })
                .unwrap_or_else(|| {
                    eprintln!("attribute '{}' not found", attribute);
                    0.0
                });
            (i, [p[0], p[1], p[0], p[1]], value)
        }));
        self.index = Some(SpatialIndex::UncertaintyQuadtree(uq));
    }
}
