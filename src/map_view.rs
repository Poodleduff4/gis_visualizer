use crate::app::ClickTarget;
use crate::basemap::BasemapCache;
use crate::gis_layer::{LayerEntry, LayerKind, TessellatedGeom};
use crate::heatmap::HeatmapLayer;
use crate::spatial_index::{IndexKind, LineSegment, SpatialIndex};
use crate::uncertainty_quadtree::UncertaintyMeasurement;
use anyhow::bail;
use egui::epaint::Mesh;
use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Ui, Vec2};

// ── Viewport ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Viewport {
    /// World-space coordinate at the centre of the view.
    pub center: [f64; 2],
    /// Screen pixels per world unit.
    pub pixels_per_unit: f64,
}

impl Default for Viewport {
    fn default() -> Self {
        Viewport {
            center: [0.0, 0.0],
            pixels_per_unit: 1.0,
        }
    }
}

impl Viewport {
    /// Fit the viewport to the given world extent [xmin, ymin, xmax, ymax].
    pub fn fit_to(&mut self, extent: [f64; 4], screen_rect: Rect) {
        let w = extent[2] - extent[0];
        let h = extent[3] - extent[1];
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        self.center = [extent[0] + w * 0.5, extent[1] + h * 0.5];
        let scale_x = screen_rect.width() as f64 / w;
        let scale_y = screen_rect.height() as f64 / h;
        self.pixels_per_unit = scale_x.min(scale_y) * 0.9;
    }

    pub fn world_to_screen(&self, wx: f64, wy: f64, rect: Rect) -> Pos2 {
        let cx = rect.center().x as f64;
        let cy = rect.center().y as f64;
        let sx = cx + (wx - self.center[0]) * self.pixels_per_unit;
        // Y is flipped: world north-up → screen top-down
        let sy = cy - (wy - self.center[1]) * self.pixels_per_unit;
        Pos2::new(sx as f32, sy as f32)
    }

    pub fn screen_to_world(&self, sx: f32, sy: f32, rect: Rect) -> [f64; 2] {
        let cx = rect.center().x as f64;
        let cy = rect.center().y as f64;
        let wx = self.center[0] + (sx as f64 - cx) / self.pixels_per_unit;
        let wy = self.center[1] - (sy as f64 - cy) / self.pixels_per_unit;
        [wx, wy]
    }

    pub fn viewport_bbox(&self, rect: Rect) -> [f64; 4] {
        let tl = self.screen_to_world(rect.left(), rect.top(), rect);
        let br = self.screen_to_world(rect.right(), rect.bottom(), rect);
        [
            tl[0].min(br[0]),
            tl[1].min(br[1]),
            tl[0].max(br[0]),
            tl[1].max(br[1]),
        ]
    }
}

// ── Colours ───────────────────────────────────────────────────────────────────

const FILL_NORMAL: Color32 = Color32::from_rgb(100, 149, 237);
const FILL_SELECTED: Color32 = Color32::from_rgb(255, 165, 0);
const STROKE_NORMAL: Color32 = Color32::from_rgb(30, 60, 120);
const STROKE_SELECTED: Color32 = Color32::from_rgb(200, 80, 0);
const POINT_RADIUS: f32 = 5.0;

/// Renders the map and handles pan/zoom/click.
pub fn show_map(
    ui: &mut Ui,
    response: &egui::Response,
    painter: &Painter,
    layers: &[LayerEntry],
    active_entry: Option<&LayerEntry>,
    viewport: &mut Viewport,
    selected_id: &mut Option<usize>,
    basemap: Option<&BasemapCache>,
    render_points: bool,
    click_target: &ClickTarget,
    selected_index_cell_data: &mut Option<UncertaintyMeasurement>,
) {
    let ctx = ui.ctx().clone();
    let rect = response.rect;

    // Background
    painter.rect_filled(rect, 0.0, Color32::from_rgb(30, 30, 30));
    // Basemap tiles
    if let Some(bm) = basemap {
        bm.render(&painter, viewport, rect, &ctx);
    }

    // Pan via primary drag
    if response.dragged_by(egui::PointerButton::Primary) {
        let delta: Vec2 = response.drag_delta();
        viewport.center[0] -= delta.x as f64 / viewport.pixels_per_unit;
        viewport.center[1] += delta.y as f64 / viewport.pixels_per_unit;
    }

    // Zoom via scroll wheel, centred on cursor
    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
    if scroll != 0.0 {
        let zoom_factor = (scroll as f64 * 0.002).exp();
        if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
            if rect.contains(hover_pos) {
                let [wx, wy] = viewport.screen_to_world(hover_pos.x, hover_pos.y, rect);
                viewport.pixels_per_unit *= zoom_factor;
                // Keep the world point under the cursor fixed
                let cx = rect.center().x as f64;
                let cy = rect.center().y as f64;
                viewport.center[0] = wx - (hover_pos.x as f64 - cx) / viewport.pixels_per_unit;
                viewport.center[1] = wy + (hover_pos.y as f64 - cy) / viewport.pixels_per_unit;
            }
        } else {
            viewport.pixels_per_unit *= zoom_factor;
        }
    }

    // Click to select — only tests against the active layer
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(entry) = active_entry {
                let [wx, wy] = viewport.screen_to_world(pos.x, pos.y, rect);
                match click_target {
                    ClickTarget::Feature => {
                        let tolerance = 8.0 / viewport.pixels_per_unit;
                        *selected_id = entry.data.hit_test(wx, wy, tolerance);
                    }
                    ClickTarget::GridCell => {
                        if let Some(data) = entry.data.index(IndexKind::Quadtree) {
                            let selected_cell = match data {
                                SpatialIndex::UncertaintyQuadtree(uncertainty_quadtree) => {
                                    uncertainty_quadtree
                                        .pos_to_node([wx, wy])
                                        .map(|qt| qt.uncertainty.as_ref())
                                }
                                _ => None,
                            }
                            .flatten();
                            if let Some(c) = selected_cell {
                                *selected_index_cell_data = Some(c.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    // Render all visible layers bottom-to-top
    let [xmin, ymin, xmax, ymax] = viewport.viewport_bbox(rect);
    let active_layer_name = active_entry.map(|e| e.name.as_str());
    for entry in layers.iter().filter(|e| e.visible) {
        // When GPU handles points, skip layers that have no polygons or lines.
        if !render_points
            && match &entry.data {
                LayerKind::Points(point_cloud_layer) => true,
                LayerKind::Vector(gis_layer) => false,
            }
        {
            continue;
        }
        let LayerKind::Vector(layer) = &entry.data else {
            continue;
        };
        let is_active = active_layer_name == Some(entry.name.as_str());
        let visible_ids = layer.features_in_bbox(xmin, ymin, xmax, ymax);
        for id in visible_ids {
            let feature = &layer.features[id];
            let tess = &feature.tessellated;
            if !render_points && tess.fill_idx.is_empty() && tess.outlines.is_empty() {
                continue;
            }
            let is_selected = is_active && *selected_id == Some(id);
            render_tessellated(&painter, tess, viewport, rect, is_selected, render_points);
        }
    }

    // Status line at bottom of map
    if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
        if rect.contains(hover_pos) {
            let [wx, wy] = viewport.screen_to_world(hover_pos.x, hover_pos.y, rect);
            let label = format!(
                "x: {wx:.4}  y: {wy:.4}  zoom: {:.4}",
                viewport.pixels_per_unit
            );
            painter.text(
                Pos2::new(rect.left() + 6.0, rect.bottom() - 18.0),
                egui::Align2::LEFT_CENTER,
                label,
                egui::FontId::monospace(11.0),
                Color32::WHITE,
            );
        }
    }
}

pub fn show_spatial_index_grid(
    painter: &Painter,
    index: Option<&SpatialIndex>,
    viewport: &mut Viewport,
    rect: Rect,
) {
    let Some(index) = index else {
        return;
    };
    let stroke = Stroke::new(2.0, Color32::from_rgb(0, 0, 255));
    for LineSegment { start, end } in index.shapes().iter() {
        let p1 = viewport.world_to_screen(start[0], start[1], rect);
        let p2 = viewport.world_to_screen(end[0], end[1], rect);
        painter.line_segment([p1, p2], stroke);
    }
}

pub fn show_quadtree_heatmap(
    painter: &Painter,
    heatmap: &HeatmapLayer,
    viewport: &Viewport,
    rect: Rect,
    opacity: u8,
) {
    let max_depth = heatmap
        .cells
        .iter()
        .map(|c| c.depth)
        .max()
        .unwrap_or(1)
        .max(1);
    for cell in &heatmap.cells {
        let t = (cell.depth as f32 / max_depth as f32).powf(1.5);
        let color = heat_color(t, opacity);
        let p1 = viewport.world_to_screen(cell.bbox[0], cell.bbox[1], rect);
        let p2 = viewport.world_to_screen(cell.bbox[2], cell.bbox[3], rect);
        painter.rect_filled(Rect::from_two_pos(p1, p2), 0.0, color);
    }
}

fn heat_color(t: f32, alpha: u8) -> Color32 {
    let (r, g, b) = if t < 0.33 {
        let s = t / 0.33;
        (0, (s * 255.0) as u8, 255)
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (0, 255, (255.0 * (1.0 - s)) as u8)
    } else {
        let s = (t - 0.66) / 0.34;
        ((s * 255.0) as u8, (255.0 * (1.0 - s)) as u8, 0)
    };
    Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

fn render_tessellated(
    painter: &Painter,
    tess: &TessellatedGeom,
    viewport: &Viewport,
    rect: Rect,
    selected: bool,
    render_points: bool,
) {
    let fill = if selected { FILL_SELECTED } else { FILL_NORMAL };
    let stroke_color = if selected {
        STROKE_SELECTED
    } else {
        STROKE_NORMAL
    };
    let stroke = Stroke::new(1.0, stroke_color);

    // Filled polygon mesh
    if !tess.fill_idx.is_empty() {
        let mut mesh = Mesh::default();
        for v in &tess.fill_verts {
            let pos = viewport.world_to_screen(v[0], v[1], rect);
            mesh.colored_vertex(pos, fill);
        }
        for &idx in &tess.fill_idx {
            mesh.indices.push(idx as u32);
        }
        // Mesh must have index count divisible by 3
        let trim = (mesh.indices.len() / 3) * 3;
        mesh.indices.truncate(trim);
        if !mesh.indices.is_empty() {
            painter.add(Shape::mesh(mesh));
        }
    }

    for ring in &tess.outlines {
        if ring.len() < 2 {
            continue;
        }
        let pts: Vec<Pos2> = ring
            .iter()
            .map(|v| viewport.world_to_screen(v[0], v[1], rect))
            .collect();
        painter.add(Shape::closed_line(pts, stroke));
    }

    if render_points {
        for &[wx, wy] in &tess.points {
            let pos = viewport.world_to_screen(wx, wy, rect);
            let half = POINT_RADIUS;
            let r = Rect::from_center_size(pos, Vec2::splat(half * 2.0));
            painter.rect(r, 0.0, fill, stroke, egui::StrokeKind::Outside);
        }
    }
}
