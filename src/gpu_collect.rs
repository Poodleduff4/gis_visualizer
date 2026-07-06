use egui::Color32;

use crate::gis_layer::{LayerEntry, LayerKind};
use crate::point_cloud::GpuPoint;

pub const FILL_SELECTED: Color32 = Color32::from_rgb(255, 165, 0);

pub fn pack_color(c: Color32) -> u32 {
    let [r, g, b, a] = c.to_array();
    r as u32 | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

pub fn collect_gpu_points(
    layers: &[LayerEntry],
    active_idx: Option<usize>,
    selected_id: Option<usize>,
    _viewport_bbox: Option<[f64; 4]>,
    point_size: f32,
    out: &mut Vec<GpuPoint>,
) {
    out.clear();
    println!("collecting...");
    for (i, entry) in layers.iter().enumerate() {
        if !entry.visible
            || match &entry.data {
                LayerKind::Points(_) => false,
                LayerKind::Vector(_) | LayerKind::Raster(_) => true,
            }
        {
            continue;
        }
        let is_active = active_idx == Some(i);
        let point_cloud_layer = match &entry.data {
            LayerKind::Points(pc) => pc,
            LayerKind::Vector(_) | LayerKind::Raster(_) => {
                panic!("Unexpected layer kind in collect_gpu_points!")
            }
        };
        let visible_points = point_cloud_layer
            .points
            .iter()
            .enumerate()
            .filter(|(i, (_pi, _pv))| point_cloud_layer.filter_mask[*i])
            .map(|(_, (_pi, pv))| *pv)
            .collect::<Vec<[f64; 2]>>();
        let layer_color = Color32::from_rgb(entry.color[0], entry.color[1], entry.color[2]);
        for (idx, &point) in visible_points.iter().enumerate() {
            let fill = if is_active && selected_id == Some(idx) {
                FILL_SELECTED
            } else {
                layer_color
            };
            let packed = pack_color(fill);
            out.push(GpuPoint {
                position: [point[0] as f32, point[1] as f32],
                color: packed,
                size: point_size,
            });
        }
    }
}
