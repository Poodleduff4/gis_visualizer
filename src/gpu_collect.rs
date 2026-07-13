use egui::Color32;

use crate::gis_layer::{LayerEntry, LayerKind};
use crate::point_cloud::GpuPoint;
use crate::vector_gpu::GpuVertex;

// Selection highlight always renders fully opaque, regardless of the layer's
// own opacity setting, so a selected feature stays clearly visible.
pub const FILL_SELECTED: Color32 = Color32::from_rgb(255, 165, 0);
const STROKE_SELECTED: Color32 = Color32::from_rgb(200, 80, 0);

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
            || !entry.show_points
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
        let layer_color = Color32::from_rgba_unmultiplied(
            entry.color[0],
            entry.color[1],
            entry.color[2],
            entry.opacity,
        );
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

/// Flattens every visible vector layer's cached `TessellatedGeom` into GPU
/// upload buffers, once, instead of the CPU painter re-submitting a mesh per
/// feature every frame. Fill triangles keep their index buffer (offset per
/// feature); each closed outline ring is expanded into consecutive
/// `LineList` vertex pairs since re-expanding per frame is exactly the cost
/// this replaces.
pub fn collect_gpu_vector_mesh(
    layers: &[LayerEntry],
    active_idx: Option<usize>,
    selected_id: Option<usize>,
    fill_verts_out: &mut Vec<GpuVertex>,
    fill_indices_out: &mut Vec<u32>,
    line_verts_out: &mut Vec<GpuVertex>,
) {
    fill_verts_out.clear();
    fill_indices_out.clear();
    line_verts_out.clear();

    let fill_selected = pack_color(FILL_SELECTED);
    let stroke_selected = pack_color(STROKE_SELECTED);

    for (i, entry) in layers.iter().enumerate() {
        if !entry.visible {
            continue;
        }
        let LayerKind::Vector(layer) = &entry.data else {
            continue;
        };
        let is_active = active_idx == Some(i);
        // Fill respects the layer's opacity; outline stays fully opaque so
        // the edge stays crisp even on a very transparent fill.
        let fill_normal = pack_color(Color32::from_rgba_unmultiplied(
            entry.color[0],
            entry.color[1],
            entry.color[2],
            entry.opacity,
        ));
        let stroke_normal =
            pack_color(Color32::from_rgb(entry.color[0], entry.color[1], entry.color[2]));

        for (id, feature) in layer.features.iter().enumerate() {
            let tess = &feature.tessellated;
            if tess.fill_idx.is_empty() && tess.outlines.is_empty() {
                continue;
            }
            let is_selected = is_active && selected_id == Some(id);
            let fill_color = if is_selected { fill_selected } else { fill_normal };
            let stroke_color = if is_selected { stroke_selected } else { stroke_normal };

            if !tess.fill_idx.is_empty() {
                let base = fill_verts_out.len() as u32;
                fill_verts_out.extend(tess.fill_verts.iter().map(|v| GpuVertex {
                    position: [v[0] as f32, v[1] as f32],
                    color: fill_color,
                }));
                // Index count must be a multiple of 3 -- same safety trim the
                // CPU path (render_tessellated) already applies.
                let trim = (tess.fill_idx.len() / 3) * 3;
                fill_indices_out.extend(tess.fill_idx[..trim].iter().map(|&ix| base + ix as u32));
            }

            for ring in &tess.outlines {
                let n = ring.len();
                if n < 2 {
                    continue;
                }
                for w in 0..n {
                    let a = ring[w];
                    let b = ring[(w + 1) % n];
                    line_verts_out.push(GpuVertex {
                        position: [a[0] as f32, a[1] as f32],
                        color: stroke_color,
                    });
                    line_verts_out.push(GpuVertex {
                        position: [b[0] as f32, b[1] as f32],
                        color: stroke_color,
                    });
                }
            }
        }
    }
}
