use bitvec::{array::BitArray, vec::BitVec};

use crate::{
    gis_layer::BatchMessage,
    hilbert_r_tree::HilbertRTree,
    quadtree::Quadtree,
    spatial_index::SpatialIndex,
    uncertainty_quadtree::{MeasurementType, UncertaintyQuadtree},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
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
    pub points: Arc<Vec<(u32, [f64; 2])>>,
    pub attributes: Vec<AttributeColumn>,
    pub field_names: Vec<String>,
    pub index: Option<Arc<SpatialIndex>>,
    pub bbox: Option<[f64; 4]>,
    pub viewport_mask: BitVec,
    pub filter_mask: BitVec,
}
impl PointCloudLayer {
    fn ensure_bbox(&mut self) {
        if self.points.is_empty() || self.bbox.is_none() {
            let mut xmin = f64::MAX;
            let mut ymin = f64::MAX;
            let mut xmax = f64::MIN;
            let mut ymax = f64::MIN;
            for (idx, p) in self.points.iter() {
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
            for (pos, (_, p)) in self.points.iter().enumerate() {
                if self.filter_mask[pos] {
                    qt.insert(pos, [p[0], p[1], p[0], p[1]]);
                }
            }
            self.index = Some(Arc::new(qt));
        }
    }

    pub fn rebuild_rtree(&mut self) {
        self.index = Some(Arc::new(SpatialIndex::RTree(crate::rtree_index::build(
            &self.points,
        ))));
    }

    pub fn has_rtree(&self) -> bool {
        matches!(self.index.as_deref(), Some(SpatialIndex::RTree(_)))
    }

    pub fn rebuild_hilbert_tree(&mut self, order: u32) {
        self.ensure_bbox();
        if let Some(bbox) = self.bbox {
            let mut ht = SpatialIndex::HilbertCurve(HilbertRTree::new(bbox, order));
            for (pos, (_, p)) in self.points.iter().enumerate() {
                ht.insert(pos, [p[0], p[1], p[0], p[1]]);
            }
            self.index = Some(Arc::new(ht));
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
                    let p = &self.points[*idx].1;
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
            for (idx, p) in self.points.iter() {
                let dist = ((x - p[0]).powf(2.) + (y - p[1]).powf(2.)).sqrt();
                if dist < b_dist {
                    b_idx = Some(*idx as usize);
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
        uq.insert_batch(self.points.iter().map(|(i, p)| {
            let value = field_idx
                .and_then(|fi| self.attributes.get(fi))
                .map(|col| match col {
                    AttributeColumn::Float(v) => v.get(*i as usize).copied().unwrap_or(0.0),
                    AttributeColumn::Integer(v) => v.get(*i as usize).copied().unwrap_or(0) as f64,
                    AttributeColumn::Text(_) => 0.0,
                })
                .unwrap_or_else(|| {
                    eprintln!("attribute '{}' not found", attribute);
                    0.0
                });
            (*i as usize, [p[0], p[1], p[0], p[1]], value)
        }));
        self.index = Some(Arc::new(SpatialIndex::UncertaintyQuadtree(uq)));
    }

    // Returns matching points + attribute columns for a bbox query against the spatial index.
    // Caller should invoke this BEFORE clear_layer() since it reads self.points.
    // pub fn extract_bbox(&self, bbox: [f64; 4]) -> (Vec<[f64; 2]>, Vec<(String, AttributeColumn)>) {
    //     let indices = match &self.index {
    //         Some(idx) => idx.search(&bbox),
    //         None => self
    //             .points
    //             .iter()
    //             .enumerate()
    //             .filter(|(_, p)| {
    //                 p[0] >= bbox[0] && p[0] <= bbox[2] && p[1] >= bbox[1] && p[1] <= bbox[3]
    //             })
    //             .map(|(i, _)| i)
    //             .collect(),
    //     };
    //     let pts = indices.iter().map(|&i| self.points[i]).collect();
    //     let cols = self
    //         .field_names
    //         .iter()
    //         .zip(self.attributes.iter())
    //         .map(|(name, col)| {
    //             let sub = match col {
    //                 AttributeColumn::Float(v) => {
    //                     AttributeColumn::Float(indices.iter().map(|&i| v[i]).collect())
    //                 }
    //                 AttributeColumn::Integer(v) => {
    //                     AttributeColumn::Integer(indices.iter().map(|&i| v[i]).collect())
    //                 }
    //                 AttributeColumn::Text(v) => {
    //                     AttributeColumn::Text(indices.iter().map(|&i| v[i].clone()).collect())
    //                 }
    //             };
    //             (name.clone(), sub)
    //         })
    //         .collect();
    //     (pts, cols)
    // }
}

pub fn stream_index_bbox(
    dest_idx: usize,
    pts: Vec<(u32, [f64; 2])>,
    cols: Vec<(String, AttributeColumn)>,
    tx: mpsc::SyncSender<BatchMessage>,
    cancel: Arc<AtomicBool>,
) {
    const BATCH: usize = 50_000;
    let n = pts.len();
    let mut start = 0;
    while start < n {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let end = (start + BATCH).min(n);
        let batch_pts = pts[start..end].to_vec();
        let batch_cols = cols
            .iter()
            .map(|(name, col)| {
                let sub = match col {
                    AttributeColumn::Float(v) => AttributeColumn::Float(v[start..end].to_vec()),
                    AttributeColumn::Integer(v) => AttributeColumn::Integer(v[start..end].to_vec()),
                    AttributeColumn::Text(v) => AttributeColumn::Text(v[start..end].to_vec()),
                };
                (name.clone(), sub)
            })
            .collect();
        tx.send(BatchMessage::Points(dest_idx, batch_pts, batch_cols))
            .ok();
        start = end;
    }
}

pub fn query_and_stream_viewport(
    dest_idx: usize,
    points: Arc<Vec<(u32, [f64; 2])>>,
    index: Option<Arc<SpatialIndex>>,
    bbox: [f64; 4],
    tx: mpsc::SyncSender<BatchMessage>,
    cancel: Arc<AtomicBool>,
) {
    let total_pts = points.len();
    let index_size = match &index {
        Some(idx) => idx.len(),
        None => 0,
    };
    println!("viewport_query: total_pts={total_pts} index_size={index_size} bbox={bbox:?}");

    let t0 = std::time::Instant::now();
    let viewport_pts: Vec<u32> = match &index {
        Some(idx) => idx.points_idx_in_bbox(bbox),
        None => Vec::new(),
    };
    let filter_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let pt_count = viewport_pts.len();
    let max_pos = viewport_pts.iter().copied().max();
    println!("viewport_query result: {pt_count} pts, max_pos={max_pos:?}, cancelled={}", cancel.load(Ordering::Relaxed));
    if !viewport_pts.is_empty() && !cancel.load(Ordering::Relaxed) {
        tx.send(BatchMessage::ViewportPoints(dest_idx, viewport_pts)).ok();
    }
    let send_time = t1.elapsed();

    println!(
        "filter: {:.2?} ({} pts) | send: {:.2?}",
        filter_time,
        pt_count,
        send_time,
    );
}
