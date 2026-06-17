use rstar::{primitives::GeomWithData, RTree, AABB};

pub type SpatialTree = RTree<GeomWithData<[f64; 2], usize>>;

pub fn build(coords: &[(u32, [f64; 2])]) -> SpatialTree {
    let entries = coords
        .iter()
        .enumerate()
        .map(|(pos, (_, pt))| GeomWithData::new(*pt, pos))
        .collect();
    RTree::bulk_load(entries)
}

pub fn query_bbox(tree: &SpatialTree, bbox: [f64; 4]) -> impl Iterator<Item = usize> + '_ {
    tree.locate_in_envelope(&AABB::from_corners([bbox[0], bbox[1]], [bbox[2], bbox[3]]))
        .map(|e| e.data)
}
