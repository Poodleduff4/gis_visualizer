use rand::{rng, rngs::ThreadRng, seq::IteratorRandom};

use std::collections::HashMap;

use crate::{
    gis_layer::AttributeValue,
    spatial_index::{HeatmapCell, LineSegment},
};

#[derive(Debug, Clone)]
pub struct Entry {
    pub id: usize,
    pub coord: [f64; 2],
    pub measurement_value: f64,
}

impl Entry {
    pub fn new(id: usize, coord: [f64; 2], measurement_value: f64) -> Self {
        Entry {
            id,
            coord,
            measurement_value,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UncertaintyMeasurement {
    pub std_dev: f64,
    pub variance: f64,
    pub mean: f64,
}

const MAX_DEPTH: usize = 24;

pub struct UncertaintyQuadtree {
    bbox: [f64; 4],
    entries: Vec<Entry>,
    children: Vec<Box<UncertaintyQuadtree>>,
    divided: bool,
    attribute: String,
    pub uncertainty: Option<UncertaintyMeasurement>,
    uncertainty_threshold: f32,
    depth: usize,
}

impl UncertaintyQuadtree {
    pub fn insert(&mut self, id: usize, rect: [f64; 4], measurement_value: f64) {
        let cx = (rect[0] + rect[2]) / 2.0;
        let cy = (rect[1] + rect[3]) / 2.0;
        self.insert_point(Entry::new(id, [cx, cy], measurement_value));
    }

    pub fn search(&self, rect: &[f64; 4]) -> Vec<usize> {
        self.range_query(rect).into_iter().map(|e| e.id).collect()
    }

    pub fn delete(&mut self, id: usize) {
        self.entries.retain(|e| e.id != id);
        for child in &mut self.children {
            child.delete(id);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len() + self.children.iter().map(|c| c.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.children.iter().all(|c| c.is_empty())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.children.clear();
        self.divided = false;
    }

    pub fn shapes(&self) -> Vec<LineSegment> {
        let mut segments = Vec::new();
        if !self.divided {
            segments.push(LineSegment::new(
                [self.bbox[0], self.bbox[1]],
                [self.bbox[0], self.bbox[3]],
            ));
            segments.push(LineSegment::new(
                [self.bbox[0], self.bbox[1]],
                [self.bbox[2], self.bbox[1]],
            ));
            segments.push(LineSegment::new(
                [self.bbox[0], self.bbox[3]],
                [self.bbox[2], self.bbox[3]],
            ));
            segments.push(LineSegment::new(
                [self.bbox[2], self.bbox[1]],
                [self.bbox[2], self.bbox[3]],
            ));
        } else {
            for child in &self.children {
                segments.extend(child.shapes());
            }
        }
        segments
    }

    pub fn heatmap_cells(&self) -> Vec<HeatmapCell> {
        self.collect_cells_inner(0)
    }

    pub fn pos_to_node(&self, pos: [f64; 2]) -> Option<&UncertaintyQuadtree> {
        if self.intersects(&[pos[0], pos[1], pos[0], pos[1]]) && !self.divided {
            return Some(&self);
        }

        for child in &self.children {
            if let Some(ch) = child.pos_to_node(pos) {
                return Some(ch);
            }
        }
        return None;
    }
}

impl UncertaintyQuadtree {
    pub fn new(bbox: [f64; 4], attribute: String, threshold: f32) -> Self {
        Self::new_at_depth(bbox, attribute, 0, threshold)
    }

    fn new_at_depth(bbox: [f64; 4], attribute: String, depth: usize, threshold: f32) -> Self {
        UncertaintyQuadtree {
            bbox,
            entries: Vec::new(),
            children: Vec::new(),
            divided: false,
            attribute,
            uncertainty: None,
            uncertainty_threshold: threshold,
            depth,
        }
    }

    fn contains(&self, entry: &Entry) -> bool {
        self.bbox[0] <= entry.coord[0]
            && entry.coord[0] <= self.bbox[2]
            && self.bbox[1] <= entry.coord[1]
            && entry.coord[1] <= self.bbox[3]
    }

    fn intersects(&self, other: &[f64; 4]) -> bool {
        self.bbox[0] <= other[2]
            && self.bbox[2] >= other[0]
            && self.bbox[1] <= other[3]
            && self.bbox[3] >= other[1]
    }

    pub fn should_split(&self) -> bool {
        if let Some(split) = self
            .uncertainty
            .as_ref()
            .map(|u| u.variance > self.uncertainty_threshold as f64)
        {
            return true;
        }
        return false;
    }

    fn insert_point(&mut self, entry: Entry) -> bool {
        if !self.contains(&entry) {
            return false;
        }

        if !self.divided {
            self.entries.push(entry);
            self.calculate_uncertainty();
            let should_split = self.depth < MAX_DEPTH
                && self.uncertainty.as_ref().map_or(false, |u| {
                    if u.mean.abs() < f64::EPSILON {
                        return false;
                    }
                    let cv = u.std_dev / u.mean.abs();
                    cv > self.uncertainty_threshold as f64
                });
            if should_split {
                self.subdivide();
            }
            return true;
        }

        for child in &mut self.children {
            if child.insert_point(entry.clone()) {
                return true;
            }
        }

        false
    }

    pub fn range_query(&self, query: &[f64; 4]) -> Vec<Entry> {
        let mut results = Vec::new();

        if !self.intersects(query) {
            return results;
        }

        for point in &self.entries {
            if query[0] <= point.coord[0]
                && point.coord[0] <= query[2]
                && query[1] <= point.coord[1]
                && point.coord[1] <= query[3]
            {
                results.push(point.clone());
            }
        }

        for child in &self.children {
            results.extend(child.range_query(query));
        }

        results
    }

    pub fn calculate_uncertainty(&mut self) {
        let n = (self.entries.len().max(1)) as f64;
        let mut rng = rand::rng();
        let sample = self
            .entries
            .iter()
            .sample(&mut rng, (n / 10.).min(5.) as usize);

        let attrs: Vec<f64> = sample
            .iter()
            .map(|e| e.measurement_value)
            .collect::<Vec<f64>>();

        let mean = attrs.iter().sum::<f64>() / n;

        let dists: Vec<f64> = sample
            .iter()
            .enumerate()
            .map(|(i, _)| attrs[i] - mean)
            .collect::<Vec<f64>>();

        let mean_d = dists.iter().sum::<f64>() / n;
        let variance = dists.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / n;
        let min_d = dists.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_d = dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let uncertainty = UncertaintyMeasurement {
            std_dev: variance.sqrt(),
            variance,
            mean,
        };
        self.uncertainty = Some(uncertainty);
    }

    fn subdivide(&mut self) {
        let mid_x = (self.bbox[0] + self.bbox[2]) / 2.0;
        let mid_y = (self.bbox[1] + self.bbox[3]) / 2.0;
        let next_depth = self.depth + 1;

        self.children
            .push(Box::new(UncertaintyQuadtree::new_at_depth(
                [self.bbox[0], self.bbox[1], mid_x, mid_y],
                self.attribute.clone(),
                next_depth,
                self.uncertainty_threshold,
            )));
        self.children
            .push(Box::new(UncertaintyQuadtree::new_at_depth(
                [mid_x, self.bbox[1], self.bbox[2], mid_y],
                self.attribute.clone(),
                next_depth,
                self.uncertainty_threshold,
            )));
        self.children
            .push(Box::new(UncertaintyQuadtree::new_at_depth(
                [self.bbox[0], mid_y, mid_x, self.bbox[3]],
                self.attribute.clone(),
                next_depth,
                self.uncertainty_threshold,
            )));
        self.children
            .push(Box::new(UncertaintyQuadtree::new_at_depth(
                [mid_x, mid_y, self.bbox[2], self.bbox[3]],
                self.attribute.clone(),
                next_depth,
                self.uncertainty_threshold,
            )));

        self.divided = true;
        self.redistribute();
    }

    fn collect_cells_inner(&self, depth: usize) -> Vec<HeatmapCell> {
        if !self.divided {
            vec![HeatmapCell {
                bbox: self.bbox,
                depth,
            }]
        } else {
            self.children
                .iter()
                .flat_map(|c| c.collect_cells_inner(depth + 1))
                .collect()
        }
    }

    fn redistribute(&mut self) {
        let entries = std::mem::take(&mut self.entries);
        for entry in entries {
            for child in &mut self.children {
                if child.insert_point(entry.clone()) {
                    break;
                }
            }
        }
    }
}
