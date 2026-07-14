use bitvec::vec::BitVec;

use crate::app::ClickTarget;
use crate::basemap::BasemapCache;
use crate::gis_layer::{
    bake_raster_rgba, LayerEntry, LayerKind, LayerSelection, RasterData, TessellatedGeom,
};
use crate::heatmap::{HeatmapLayer, HeatmapMetric};
use crate::histogram::{LisaCluster, LisaPoint};
use crate::spatial_index::{IndexKind, LineSegment, SpatialIndex};
use crate::uncertainty_quadtree::UncertaintyMeasure;
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

const FILL_SELECTED: Color32 = Color32::from_rgb(255, 165, 0);
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
    selected_index_cell_data: &mut Option<UncertaintyMeasure>,
    roi_toggle: &mut Option<[f64; 4]>,
    select_mode: bool,
    select_drag_start: &mut Option<Pos2>,
    selection_result: &mut Option<[f64; 4]>,
) {
    let ctx = ui.ctx().clone();
    let rect = response.rect;

    // Background
    painter.rect_filled(rect, 0.0, Color32::from_rgb(30, 30, 30));
    // Basemap tiles
    if let Some(bm) = basemap {
        bm.render(&painter, viewport, rect, &ctx);
    }

    // Pan via primary drag (disabled while box-selecting)
    if !select_mode && response.dragged_by(egui::PointerButton::Primary) {
        let delta: Vec2 = response.drag_delta();
        viewport.center[0] -= delta.x as f64 / viewport.pixels_per_unit;
        viewport.center[1] += delta.y as f64 / viewport.pixels_per_unit;
    }

    // Box-select: rubber-band drag while select_mode is on.
    if select_mode {
        if response.drag_started() {
            *select_drag_start = response.interact_pointer_pos();
        }
        if response.dragged() {
            if let (Some(start), Some(cur)) = (*select_drag_start, response.interact_pointer_pos())
            {
                let band = Rect::from_two_pos(start, cur);
                painter.rect_filled(band, 0.0, Color32::from_rgba_unmultiplied(255, 255, 0, 40));
                painter.rect_stroke(
                    band,
                    0.0,
                    Stroke::new(1.5, Color32::YELLOW),
                    egui::StrokeKind::Middle,
                );
            }
        }
        if response.drag_stopped() {
            if let (Some(start), Some(cur)) = (select_drag_start.take(), response.interact_pointer_pos())
            {
                let [wx0, wy0] = viewport.screen_to_world(start.x, start.y, rect);
                let [wx1, wy1] = viewport.screen_to_world(cur.x, cur.y, rect);
                *selection_result = Some([
                    wx0.min(wx1),
                    wy0.min(wy1),
                    wx0.max(wx1),
                    wy0.max(wy1),
                ]);
            }
        }
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

    // Click to select — only tests against the active layer (disabled while box-selecting)
    if !select_mode && response.clicked() {
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
                    ClickTarget::HeatmapRoi => {
                        if let Some(data) = entry.data.index(IndexKind::Quadtree) {
                            if let Some(bbox) = data.leaf_bbox_at([wx, wy]) {
                                *roi_toggle = Some(bbox);
                            }
                        }
                    }
                }
            }
        }
    }

    // Render all visible layers bottom-to-top.
    // `render_points` is really "no GPU available, use the CPU fallback path" —
    // when it's false, both points and vector layers are GPU-rendered instead
    // (see VectorCallback in ui_map.rs), so this whole CPU loop is skipped.
    if render_points {
        let [xmin, ymin, xmax, ymax] = viewport.viewport_bbox(rect);
        let active_layer_name = active_entry.map(|e| e.name.as_str());
        for entry in layers.iter().filter(|e| e.visible) {
            let LayerKind::Vector(layer) = &entry.data else {
                continue;
            };
            let is_active = active_layer_name == Some(entry.name.as_str());
            // Fill respects the layer's opacity; outline stays fully opaque
            // so the edge stays crisp even on a very transparent fill.
            let fill = Color32::from_rgba_unmultiplied(
                entry.color[0],
                entry.color[1],
                entry.color[2],
                entry.opacity,
            );
            let stroke = Color32::from_rgb(entry.color[0], entry.color[1], entry.color[2]);
            let visible_ids = layer.features_in_bbox(xmin, ymin, xmax, ymax);
            for id in visible_ids {
                let feature = &layer.features[id];
                let tess = &feature.tessellated;
                let is_selected = is_active && *selected_id == Some(id);
                render_tessellated(&painter, tess, viewport, rect, is_selected, fill, stroke);
            }
        }
    }

    // Vector-layer point/multipoint features always CPU-paint here — see
    // `render_vector_points`'s doc comment for why they can't just ride the
    // GPU fill/outline mesh path like polygons/lines do.
    render_vector_points(&painter, layers, active_entry, viewport, rect, *selected_id);

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

const MAX_INDEX_LINES: usize = 100_000;

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
    for LineSegment { start, end } in index.shapes().iter().take(MAX_INDEX_LINES) {
        let p1 = viewport.world_to_screen(start[0], start[1], rect);
        let p2 = viewport.world_to_screen(end[0], end[1], rect);
        painter.line_segment([p1, p2], stroke);
    }
}

/// Draws saved box-selection bboxes for a layer: magenta stroke for saved
/// ones, a heavier `STROKE_SELECTED`-style stroke for the active selection.
pub fn draw_selection_bboxes(
    painter: &Painter,
    selections: &[LayerSelection],
    active: Option<usize>,
    viewport: &Viewport,
    rect: Rect,
) {
    const SELECTION_STROKE: Color32 = Color32::from_rgb(230, 0, 200);
    for (i, sel) in selections.iter().enumerate() {
        let is_active = active == Some(i);
        let stroke = if is_active {
            Stroke::new(2.5, STROKE_SELECTED)
        } else {
            Stroke::new(1.5, SELECTION_STROKE)
        };
        let p1 = viewport.world_to_screen(sel.bbox[0], sel.bbox[1], rect);
        let p2 = viewport.world_to_screen(sel.bbox[2], sel.bbox[3], rect);
        painter.rect_stroke(
            Rect::from_two_pos(p1, p2),
            0.0,
            stroke,
            egui::StrokeKind::Middle,
        );
    }
}

const MAX_HEATMAP_CELLS: usize = 50_000;

pub fn show_quadtree_heatmap(
    painter: &Painter,
    heatmap: &HeatmapLayer,
    metric: HeatmapMetric,
    roi_bboxes: &[[f64; 4]],
    viewport: &Viewport,
    rect: Rect,
    opacity: u8,
) {
    let vp = viewport.viewport_bbox(rect);
    let visible: Vec<_> = heatmap
        .cells
        .iter()
        .filter(|c| {
            c.bbox[0] <= vp[2] && c.bbox[2] >= vp[0] && c.bbox[1] <= vp[3] && c.bbox[3] >= vp[1]
        })
        .take(MAX_HEATMAP_CELLS)
        .collect();
    for cell in visible {
        let t = match metric {
            HeatmapMetric::Density => cell.density,
            HeatmapMetric::Unpredictability => cell.unpredictability,
            HeatmapMetric::AttributeMean => cell.attribute_mean,
        };
        let color = heat_color(t, opacity);
        let p1 = viewport.world_to_screen(cell.bbox[0], cell.bbox[1], rect);
        let p2 = viewport.world_to_screen(cell.bbox[2], cell.bbox[3], rect);
        painter.rect_filled(Rect::from_two_pos(p1, p2), 0.0, color);
    }
    let roi_stroke = Stroke::new(2.0, Color32::from_rgb(255, 255, 0));
    for bbox in roi_bboxes {
        if bbox[0] > vp[2] || bbox[2] < vp[0] || bbox[1] > vp[3] || bbox[3] < vp[1] {
            continue;
        }
        let p1 = viewport.world_to_screen(bbox[0], bbox[1], rect);
        let p2 = viewport.world_to_screen(bbox[2], bbox[3], rect);
        painter.rect_stroke(
            Rect::from_two_pos(p1, p2),
            0.0,
            roi_stroke,
            egui::StrokeKind::Middle,
        );
    }
}

/// Draw per-point local variance as a blue→red gradient overlay.
/// `variances` is indexed by point position (same length as `points`).
pub fn draw_local_variance_overlay(
    painter: &Painter,
    points: &[(u32, [f64; 2])],
    filter_mask: &BitVec,
    variances: &[Option<f64>],
    viewport: &Viewport,
    rect: Rect,
    opacity: u8,
) {
    let vp = viewport.viewport_bbox(rect);
    let max_var = variances
        .iter()
        .filter_map(|v| *v)
        .fold(0.0_f64, f64::max);
    if max_var < 1e-12 {
        return;
    }
    for (i, (_, p)) in points.iter().enumerate() {
        let Some(var) = variances.get(i).and_then(|v| *v) else { continue };
        if !filter_mask[i] { continue; }
        if p[0] < vp[0] || p[0] > vp[2] || p[1] < vp[1] || p[1] > vp[3] { continue; }
        let t = ((var / max_var).sqrt() as f32).clamp(0.0, 1.0);
        let color = variance_color(t, opacity);
        let pos = viewport.world_to_screen(p[0], p[1], rect);
        painter.circle_filled(pos, 5.0, color);
    }
}

fn variance_color(t: f32, alpha: u8) -> Color32 {
    let (r, g, b) = if t < 0.5 {
        let s = t / 0.5;
        (0_u8, (s * 200.0) as u8, (255.0 * (1.0 - s)) as u8)
    } else {
        let s = (t - 0.5) / 0.5;
        ((s * 255.0) as u8, (200.0 * (1.0 - s)) as u8, 0_u8)
    };
    Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

/// Draw LISA cluster classification as colored points.
///
/// HH = red, LL = blue, HL = orange, LH = cyan.
pub fn draw_lisa_overlay(
    painter: &Painter,
    points: &[(u32, [f64; 2])],
    filter_mask: &BitVec,
    lisa: &[Option<LisaPoint>],
    viewport: &Viewport,
    rect: Rect,
    opacity: u8,
) {
    let vp = viewport.viewport_bbox(rect);
    for (i, (_, p)) in points.iter().enumerate() {
        let Some(result) = lisa.get(i).and_then(|r| r.as_ref()) else { continue };
        if !filter_mask[i] { continue; }
        if p[0] < vp[0] || p[0] > vp[2] || p[1] < vp[1] || p[1] > vp[3] { continue; }
        let color = match result.cluster {
            LisaCluster::HighHigh => Color32::from_rgba_unmultiplied(220, 30, 30, opacity),
            LisaCluster::LowLow => Color32::from_rgba_unmultiplied(30, 80, 220, opacity),
            LisaCluster::HighLow => Color32::from_rgba_unmultiplied(240, 140, 20, opacity),
            LisaCluster::LowHigh => Color32::from_rgba_unmultiplied(20, 200, 220, opacity),
        };
        let pos = viewport.world_to_screen(p[0], p[1], rect);
        painter.circle_filled(pos, 5.0, color);
    }
}

pub fn heat_color(t: f32, alpha: u8) -> Color32 {
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
    layer_fill: Color32,
    layer_stroke: Color32,
) {
    let fill = if selected { FILL_SELECTED } else { layer_fill };
    let stroke_color = if selected { STROKE_SELECTED } else { layer_stroke };
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
}

/// Draws every visible Vector layer's Point/MultiPoint features as small,
/// screen-space-constant-size squares. Unlike fill/outline geometry (which
/// GPU mode renders via `collect_gpu_vector_mesh`'s triangulated mesh),
/// points need a size that doesn't scale with zoom, which the world-space
/// GPU vertex buffer can't express — so, like the LISA/local-variance
/// overlays, they're always CPU-painted here regardless of GPU/CPU mode
/// rather than being gated behind `render_points` (which really means "no
/// GPU available").
pub fn render_vector_points(
    painter: &Painter,
    layers: &[LayerEntry],
    active_entry: Option<&LayerEntry>,
    viewport: &Viewport,
    rect: Rect,
    selected_id: Option<usize>,
) {
    let [xmin, ymin, xmax, ymax] = viewport.viewport_bbox(rect);
    let active_layer_name = active_entry.map(|e| e.name.as_str());
    for entry in layers.iter().filter(|e| e.visible) {
        let LayerKind::Vector(layer) = &entry.data else {
            continue;
        };
        let is_active = active_layer_name == Some(entry.name.as_str());
        let layer_fill = Color32::from_rgba_unmultiplied(
            entry.color[0],
            entry.color[1],
            entry.color[2],
            entry.opacity,
        );
        let layer_stroke_color = Color32::from_rgb(entry.color[0], entry.color[1], entry.color[2]);

        for id in layer.features_in_bbox(xmin, ymin, xmax, ymax) {
            let feature = &layer.features[id];
            if feature.tessellated.points.is_empty() {
                continue;
            }
            let is_selected = is_active && selected_id == Some(id);
            let fill = if is_selected { FILL_SELECTED } else { layer_fill };
            let stroke = Stroke::new(
                1.0,
                if is_selected { STROKE_SELECTED } else { layer_stroke_color },
            );
            for &[wx, wy] in &feature.tessellated.points {
                let pos = viewport.world_to_screen(wx, wy, rect);
                let r = Rect::from_center_size(pos, Vec2::splat(POINT_RADIUS * 2.0));
                painter.rect(r, 0.0, fill, stroke, egui::StrokeKind::Outside);
            }
        }
    }
}

/// Draw a GeoTIFF raster as a single textured rect spanning its full-globe
/// bbox [-180,-90,180,90]. `texture_cache` is re-baked only when `dirty` (or
/// empty) — baking + uploading a fresh texture every frame isn't free.
pub fn render_raster_overlay(
    ui: &Ui,
    painter: &Painter,
    raster: &RasterData,
    viewport: &Viewport,
    rect: Rect,
    texture_cache: &mut Option<egui::TextureHandle>,
    dirty: bool,
    opacity: u8,
) {
    if dirty || texture_cache.is_none() {
        let rgba = bake_raster_rgba(raster);
        let image = egui::ColorImage::from_rgba_unmultiplied([raster.width, raster.height], &rgba);
        *texture_cache = Some(ui.ctx().load_texture(
            "raster_overlay",
            image,
            // NEAREST, not LINEAR: matches QGIS's default resampling — crisp
            // cell edges at native/zoomed-in scale instead of blurring
            // neighboring pixels together.
            egui::TextureOptions::NEAREST,
        ));
    }
    let Some(texture) = texture_cache else { return };

    let [xmin, ymin, xmax, ymax] = raster.extent;
    let top_left = viewport.world_to_screen(xmin, ymax, rect);
    let bottom_right = viewport.world_to_screen(xmax, ymin, rect);
    let screen_rect = Rect::from_two_pos(top_left, bottom_right);
    painter.image(
        texture.id(),
        screen_rect,
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
        Color32::from_white_alpha(opacity),
    );
}
