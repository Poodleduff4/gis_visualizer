use crate::spatial_index::{HeatmapCell, LineSegment};

#[derive(Debug, Clone)]
pub struct Entry {
    pub id: usize,
    pub coord: [f64; 2],
}

impl Entry {
    pub fn new(id: usize, coord: [f64; 2]) -> Self {
        Entry { id, coord }
    }
}

pub struct Quadtree {
    bbox: [f64; 4],
    capacity: usize,
    entries: Vec<Entry>,
    children: Vec<Box<Quadtree>>,
    divided: bool,
}

impl Quadtree {
    pub fn insert(&mut self, id: usize, rect: [f64; 4]) {
        let cx = (rect[0] + rect[2]) / 2.0;
        let cy = (rect[1] + rect[3]) / 2.0;
        self.insert_point(Entry::new(id, [cx, cy]));
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

    pub fn get_capacity(&self) -> Option<usize> {
        Some(self.capacity)
    }

    pub fn pos_to_node(&self, pos: [f64; 2]) -> Option<&Quadtree> {
        if self.intersects(&[pos[0], pos[1], pos[0], pos[1]]) && !self.divided {
            return Some(self);
        }

        for child in &self.children {
            if let Some(ch) = child.pos_to_node(pos) {
                return Some(ch);
            }
        }
        None
    }

    pub fn leaf_bbox_at(&self, pos: [f64; 2]) -> Option<[f64; 4]> {
        self.pos_to_node(pos).map(|n| n.bbox)
    }
}

impl Quadtree {
    pub fn new(bbox: [f64; 4], capacity: usize) -> Self {
        Quadtree {
            bbox,
            capacity,
            entries: Vec::new(),
            children: Vec::new(),
            divided: false,
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

    fn insert_point(&mut self, entry: Entry) -> bool {
        if !self.contains(&entry) {
            return false;
        }

        if self.entries.len() < self.capacity && !self.divided {
            self.entries.push(entry);
            return true;
        }

        if !self.divided {
            self.subdivide();
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

    fn subdivide(&mut self) {
        let mid_x = (self.bbox[0] + self.bbox[2]) / 2.0;
        let mid_y = (self.bbox[1] + self.bbox[3]) / 2.0;
        let cap = self.capacity;

        self.children.push(Box::new(Quadtree::new(
            [self.bbox[0], self.bbox[1], mid_x, mid_y],
            cap,
        )));
        self.children.push(Box::new(Quadtree::new(
            [mid_x, self.bbox[1], self.bbox[2], mid_y],
            cap,
        )));
        self.children.push(Box::new(Quadtree::new(
            [self.bbox[0], mid_y, mid_x, self.bbox[3]],
            cap,
        )));
        self.children.push(Box::new(Quadtree::new(
            [mid_x, mid_y, self.bbox[2], self.bbox[3]],
            cap,
        )));

        self.divided = true;
        self.redistribute();
    }

    fn collect_cells_inner(&self, depth: usize) -> Vec<HeatmapCell> {
        if !self.divided {
            vec![HeatmapCell {
                bbox: self.bbox,
                depth,
                point_ids: self.entries.iter().map(|e| e.id).collect(),
                uncertainty: None,
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
