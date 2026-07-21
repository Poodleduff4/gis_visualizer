use std::collections::HashMap;
use std::fmt::Error;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use std::sync::mpsc;

use crate::filter::{FilterLogic, LayerAttributeFilter};
use crate::gis_reader::LayerDescriptor;
use crate::point_cloud_layer::{AttributeColumn, PointCloudLayer};
use crate::quadtree::Quadtree;
use crate::spatial_index::{IndexKind, SpatialIndex};
use anyhow::{anyhow, Result};
use flatgeobuf::{FgbReader, GeometryType};
use geo::{BoundingRect, MapCoordsInPlace};
use geo_types::{
    Coord, Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
};

#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Text(String),
    Integer(i64),
    Float(f64),
}

impl AttributeValue {
    pub fn type_label(&self) -> &'static str {
        match self {
            AttributeValue::Text(_) => "Text",
            AttributeValue::Integer(_) => "Integer",
            AttributeValue::Float(_) => "Float",
        }
    }

    pub fn as_display_string(&self) -> String {
        match self {
            AttributeValue::Text(s) => s.clone(),
            AttributeValue::Integer(i) => i.to_string(),
            AttributeValue::Float(f) => format!("{f:.6}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum AttributeType {
    #[default]
    Text,
    Integer,
    Float,
}

impl AttributeType {
    pub const ALL: &'static [AttributeType] = &[
        AttributeType::Text,
        AttributeType::Integer,
        AttributeType::Float,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            AttributeType::Text => "Text",
            AttributeType::Integer => "Integer",
            AttributeType::Float => "Float",
        }
    }

    pub fn parse_value(&self, raw: &str) -> Result<AttributeValue> {
        match self {
            AttributeType::Text => Ok(AttributeValue::Text(raw.to_string())),
            AttributeType::Integer => raw
                .parse::<i64>()
                .map(AttributeValue::Integer)
                .map_err(|e| anyhow!("Not an integer: {e}")),
            AttributeType::Float => raw
                .parse::<f64>()
                .map(AttributeValue::Float)
                .map_err(|e| anyhow!("Not a float: {e}")),
        }
    }

    pub fn default_attr_value(&self) -> AttributeValue {
        match self {
            AttributeType::Text => AttributeValue::Text(String::new()),
            AttributeType::Integer => AttributeValue::Integer(0),
            AttributeType::Float => AttributeValue::Float(0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TessellatedGeom {
    pub fill_verts: Vec<[f64; 2]>,
    pub fill_idx: Vec<usize>,
    pub outlines: Vec<Vec<[f64; 2]>>,
    pub points: Vec<[f64; 2]>,
}

impl TessellatedGeom {
    fn empty() -> Self {
        TessellatedGeom {
            fill_verts: vec![],
            fill_idx: vec![],
            outlines: vec![],
            points: vec![],
        }
    }

    /// Reprojects every stored world-coordinate array in place.
    fn reproject(&mut self, t: &crate::crs::CrsTransform) {
        for v in &mut self.fill_verts {
            t.convert(v);
        }
        for outline in &mut self.outlines {
            for v in outline {
                t.convert(v);
            }
        }
        for v in &mut self.points {
            t.convert(v);
        }
    }

    fn from_polygon(poly: &Polygon<f64>) -> Self {
        let mut tess = TessellatedGeom::empty();
        tessellate_polygon(poly, &mut tess);
        tess
    }

    fn from_multipolygon(mp: &MultiPolygon<f64>) -> Self {
        let mut tess = TessellatedGeom::empty();
        for poly in mp.0.iter() {
            tessellate_polygon(poly, &mut tess);
        }
        tess
    }

    fn from_linestring(ls: &LineString<f64>) -> Self {
        TessellatedGeom {
            fill_verts: vec![],
            fill_idx: vec![],
            outlines: vec![coords_to_arr(ls.coords())],
            points: vec![],
        }
    }

    fn from_multilinestring(mls: &MultiLineString<f64>) -> Self {
        TessellatedGeom {
            fill_verts: vec![],
            fill_idx: vec![],
            outlines: mls.0.iter().map(|ls| coords_to_arr(ls.coords())).collect(),
            points: vec![],
        }
    }

    fn from_point(p: &Point<f64>) -> Self {
        TessellatedGeom {
            fill_verts: vec![],
            fill_idx: vec![],
            outlines: vec![],
            points: vec![[p.x(), p.y()]],
        }
    }

    fn from_multipoint(mp: &MultiPoint<f64>) -> Self {
        TessellatedGeom {
            fill_verts: vec![],
            fill_idx: vec![],
            outlines: vec![],
            points: mp.0.iter().map(|p| [p.x(), p.y()]).collect(),
        }
    }
}

fn coords_to_arr<'a>(coords: impl Iterator<Item = &'a Coord<f64>>) -> Vec<[f64; 2]> {
    coords.map(|c| [c.x, c.y]).collect()
}

fn tessellate_polygon(poly: &Polygon<f64>, tess: &mut TessellatedGeom) {
    let vertex_offset = tess.fill_verts.len();

    let exterior: Vec<[f64; 2]> = coords_to_arr(poly.exterior().coords());
    let exterior = if exterior.len() > 1 && exterior.first() == exterior.last() {
        &exterior[..exterior.len() - 1]
    } else {
        &exterior[..]
    };

    let mut flat: Vec<f64> = Vec::with_capacity(exterior.len() * 2);
    let mut hole_indices: Vec<usize> = Vec::new();

    for v in exterior {
        flat.push(v[0]);
        flat.push(v[1]);
    }

    for hole in poly.interiors() {
        let hole_verts: Vec<[f64; 2]> = coords_to_arr(hole.coords());
        let hole_verts = if hole_verts.len() > 1 && hole_verts.first() == hole_verts.last() {
            &hole_verts[..hole_verts.len() - 1]
        } else {
            &hole_verts[..]
        };
        hole_indices.push(flat.len() / 2);
        for v in hole_verts {
            flat.push(v[0]);
            flat.push(v[1]);
        }
    }

    // Store vertices
    for i in (0..flat.len()).step_by(2) {
        tess.fill_verts.push([flat[i], flat[i + 1]]);
    }

    // Tessellate
    if let Ok(indices) = earcutr::earcut(&flat, &hole_indices, 2) {
        for idx in indices {
            tess.fill_idx.push(vertex_offset + idx);
        }
    }

    tess.outlines.push(coords_to_arr(poly.exterior().coords()));
    for hole in poly.interiors() {
        tess.outlines.push(coords_to_arr(hole.coords()));
    }
}

pub struct GisFeature {
    pub id: usize,
    pub geometry: Geometry<f64>,
    pub tessellated: TessellatedGeom,
    pub attributes: HashMap<String, AttributeValue>,
}

impl GisFeature {
    pub fn new(
        id: usize,
        geometry: Geometry<f64>,
        attributes: HashMap<String, AttributeValue>,
    ) -> Self {
        // `tessellate` already handles Point/MultiPoint (via `from_point`/
        // `from_multipoint`, populating `TessellatedGeom.points`) — no
        // special-casing needed here.
        let tessellated = tessellate(&geometry);
        GisFeature {
            id,
            geometry,
            tessellated,
            attributes,
        }
    }

    pub fn bbox(&self) -> [f64; 4] {
        bounding_box(&self.geometry)
    }

    /// Reprojects geometry and its precomputed tessellation in place —
    /// coordinate-wise, topology-preserving, so no re-tessellation needed.
    pub fn reproject(&mut self, t: &crate::crs::CrsTransform) {
        self.geometry.map_coords_in_place(|c| {
            let mut xy = [c.x, c.y];
            t.convert(&mut xy);
            Coord { x: xy[0], y: xy[1] }
        });
        self.tessellated.reproject(t);
    }
}

fn tessellate(geom: &Geometry<f64>) -> TessellatedGeom {
    match geom {
        Geometry::Polygon(p) => TessellatedGeom::from_polygon(p),
        Geometry::MultiPolygon(mp) => TessellatedGeom::from_multipolygon(mp),
        Geometry::LineString(ls) => TessellatedGeom::from_linestring(ls),
        Geometry::MultiLineString(mls) => TessellatedGeom::from_multilinestring(mls),
        Geometry::Point(p) => TessellatedGeom::from_point(p),
        Geometry::MultiPoint(mp) => TessellatedGeom::from_multipoint(mp),
        _ => TessellatedGeom::empty(),
    }
}

fn bounding_box(geom: &Geometry<f64>) -> [f64; 4] {
    if let Some(r) = geom.bounding_rect() {
        [r.min().x, r.min().y, r.max().x, r.max().y]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    }
}

pub enum BatchMessage {
    Points(usize, Vec<(u32, [f64; 2])>, Vec<(String, AttributeColumn)>),
    ViewportPoints(usize, Vec<u32>),
    Vector(usize, Vec<GisFeature>),
}

// ── Raster (GeoTIFF) data model ────────────────────────────────────────────────

/// One channel of a (possibly multi-band) raster — same width×height grid as
/// its parent `RasterData`, row-major, `f32::NAN` marks cells with no data.
#[derive(Debug, Clone)]
pub struct RasterBand {
    pub name: String,
    pub values: Vec<f32>,
    pub data_min: f64,
    pub data_max: f64,
    /// Colormap range for single-band display — defaults to (data_min, data_max).
    pub display_min: f64,
    pub display_max: f64,
}

/// How a multi-band raster's bands are combined into the overlay texture.
#[derive(Debug, Clone)]
pub enum RasterDisplayMode {
    /// Single band through the blue→red ramp, using that band's display range.
    Single(usize),
    /// Three bands sampled straight into RGB channels (true/false color composite).
    Rgb { r: usize, g: usize, b: usize },
}

/// A dense grid, row 0 = north edge, col 0 = west edge, row-major. Loaded
/// GeoTIFFs use the canonical full -180..180 / -90..90 canvas; rasters
/// synthesized in-app (e.g. a saved heatmap promoted to a layer) carry
/// whatever local `extent` they were built over instead.
#[derive(Debug, Clone)]
pub struct RasterData {
    pub width: usize,
    pub height: usize,
    pub bands: Vec<RasterBand>,
    /// Unit label parsed from the source file, e.g. "K" — empty if unknown.
    pub units: String,
    pub display_mode: RasterDisplayMode,
    /// World bbox [xmin, ymin, xmax, ymax] this grid spans. The flat-map
    /// overlay renderer maps the grid onto this rect; the globe renderer
    /// still assumes the full -180..180/-90..90 canvas regardless of this
    /// field (arbitrary-extent rasters won't display correctly in Globe view).
    pub extent: [f64; 4],
}

impl RasterData {
    pub fn variable(&self) -> &str {
        match &self.display_mode {
            RasterDisplayMode::Single(i) => self.bands[*i].name.as_str(),
            RasterDisplayMode::Rgb { .. } => "RGB composite",
        }
    }
}

/// Blue→red heatmap colormap over normalized `t` in [0, 1].
pub fn ramp_rgba(t: f64) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 255.0) as u8;
    let g = (4.0 * t * (1.0 - t) * 200.0).min(255.0) as u8;
    let b = ((1.0 - t) * 255.0) as u8;
    [r, g, b, 255]
}

/// Fixed distinguishable palette for categorical (color-by-attribute)
/// styling — cycles if there are more distinct values than colors.
const CATEGORICAL_PALETTE: [[u8; 3]; 10] = [
    [0x1f, 0x77, 0xb4],
    [0xff, 0x7f, 0x0e],
    [0x2c, 0xa0, 0x2c],
    [0xd6, 0x27, 0x28],
    [0x94, 0x67, 0xbd],
    [0x8c, 0x56, 0x4b],
    [0xe3, 0x77, 0xc2],
    [0x7f, 0x7f, 0x7f],
    [0xbc, 0xbd, 0x22],
    [0x17, 0xbe, 0xcf],
];

pub fn categorical_color(index: usize) -> [u8; 3] {
    CATEGORICAL_PALETTE[index % CATEGORICAL_PALETTE.len()]
}

fn norm_channel(v: f32, lo: f64, hi: f64) -> Option<u8> {
    if v.is_nan() {
        return None;
    }
    let t = if hi > lo { (v as f64 - lo) / (hi - lo) } else { 0.0 };
    Some((t.clamp(0.0, 1.0) * 255.0) as u8)
}

/// Bake a raster's active display mode into an RGBA8 byte buffer (row-major,
/// same width×height as `data`). Shared by the flat CPU texture overlay and
/// the globe's GPU texture upload.
pub fn bake_raster_rgba(data: &RasterData) -> Vec<u8> {
    let px = data.width * data.height;
    let mut out = Vec::with_capacity(px * 4);

    match &data.display_mode {
        RasterDisplayMode::Single(i) => {
            let band = &data.bands[*i];
            let (lo, hi) = (band.display_min, band.display_max);
            for &v in &band.values {
                if v.is_nan() {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    let t = if hi > lo { (v as f64 - lo) / (hi - lo) } else { 0.0 };
                    out.extend_from_slice(&ramp_rgba(t));
                }
            }
        }
        RasterDisplayMode::Rgb { r, g, b } => {
            let (rb, gb, bb) = (&data.bands[*r], &data.bands[*g], &data.bands[*b]);
            for i in 0..px {
                let rv = norm_channel(rb.values[i], rb.display_min, rb.display_max);
                let gv = norm_channel(gb.values[i], gb.display_min, gb.display_max);
                let bv = norm_channel(bb.values[i], bb.display_min, bb.display_max);
                if rv.is_none() && gv.is_none() && bv.is_none() {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    out.extend_from_slice(&[rv.unwrap_or(0), gv.unwrap_or(0), bv.unwrap_or(0), 255]);
                }
            }
        }
    }
    out
}

pub enum LayerKind {
    Points(PointCloudLayer),
    Vector(GisLayer),
    Raster(RasterData),
}
impl LayerKind {
    pub fn reset_filter_mask(&mut self) {
        match self {
            LayerKind::Points(point_cloud_layer) => point_cloud_layer.filter_mask.fill(true),
            LayerKind::Vector(gis_layer) => gis_layer.filter_mask.fill(true),
            LayerKind::Raster(_) => {}
        }
    }
    pub fn clear_layer(&mut self) {
        match self {
            LayerKind::Points(point_cloud_layer) => {
                std::sync::Arc::make_mut(&mut point_cloud_layer.points).clear();
                point_cloud_layer.attributes.clear();
                point_cloud_layer.bbox = None;
            }
            LayerKind::Vector(gis_layer) => {
                gis_layer.features.clear();
            }
            LayerKind::Raster(_) => {}
        }
    }

    /// Ids of features/points within `bbox`, using the quadtree index when
    /// built, else a linear scan (mirrors `PointCloudLayer::hit_test`'s
    /// index-or-scan fallback). Used by box-select and snapshot restore.
    pub fn ids_in_bbox_with_fallback(&self, bbox: [f64; 4]) -> Vec<usize> {
        let ids = if let Some(idx) = self.index(IndexKind::Quadtree) {
            idx.search(&bbox)
        } else {
            let [xmin, ymin, xmax, ymax] = bbox;
            match self {
                LayerKind::Vector(gl) => gl
                    .features
                    .iter()
                    .filter(|f| {
                        let b = f.bbox();
                        b[0] <= xmax && b[2] >= xmin && b[1] <= ymax && b[3] >= ymin
                    })
                    .map(|f| f.id)
                    .collect(),
                LayerKind::Points(pc) => pc
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, p))| {
                        p[0] >= xmin && p[0] <= xmax && p[1] >= ymin && p[1] <= ymax
                    })
                    .map(|(i, _)| i)
                    .collect(),
                LayerKind::Raster(_) => Vec::new(),
            }
        };
        // The quadtree path above doesn't re-check the filter mask (it's
        // built excluding filtered-out points already, when up to date), but
        // a caller building a selection right after a filter change — before
        // the index has been rebuilt — would otherwise get stale ids back.
        // Filtering here makes the result correct regardless of index freshness.
        if let LayerKind::Points(pc) = self {
            ids.into_iter().filter(|&i| pc.filter_mask[i]).collect()
        } else {
            ids
        }
    }

    /// Ids of features/points inside the polygon ring `polygon` (world-space
    /// vertices, implicitly closed). Prunes candidates via the polygon's
    /// bbox (reusing `ids_in_bbox_with_fallback`, so it gets the same
    /// index-or-scan fallback and filter-mask handling), then tests each
    /// candidate exactly: point-in-polygon for a Points layer, geometry
    /// intersection for a Vector layer. Empty for fewer than 3 vertices.
    pub fn ids_in_polygon(&self, polygon: &[[f64; 2]]) -> Vec<usize> {
        use geo::{Contains, Intersects};

        if polygon.len() < 3 {
            return Vec::new();
        }
        let bbox = polygon.iter().fold(
            [f64::MAX, f64::MAX, f64::MIN, f64::MIN],
            |acc, p| {
                [
                    acc[0].min(p[0]),
                    acc[1].min(p[1]),
                    acc[2].max(p[0]),
                    acc[3].max(p[1]),
                ]
            },
        );
        let candidates = self.ids_in_bbox_with_fallback(bbox);
        let ring: geo_types::LineString<f64> =
            polygon.iter().map(|p| (p[0], p[1])).collect();
        let poly = geo_types::Polygon::new(ring, vec![]);

        match self {
            LayerKind::Vector(gl) => candidates
                .into_iter()
                .filter(|&id| {
                    gl.features
                        .get(id)
                        .map(|f| f.geometry.intersects(&poly))
                        .unwrap_or(false)
                })
                .collect(),
            LayerKind::Points(pc) => candidates
                .into_iter()
                .filter(|&i| {
                    pc.points
                        .get(i)
                        .map(|(_, p)| poly.contains(&geo_types::Point::new(p[0], p[1])))
                        .unwrap_or(false)
                })
                .collect(),
            LayerKind::Raster(_) => Vec::new(),
        }
    }

    pub fn feature_count(&self) -> usize {
        match self {
            LayerKind::Vector(gis_layer) => gis_layer.features.len(),
            LayerKind::Points(point_cloud_layer) => point_cloud_layer.points.len(),
            LayerKind::Raster(r) => r.width * r.height,
        }
    }
    pub fn filtered_count(&self) -> usize {
        match self {
            LayerKind::Points(pc) => pc.filter_mask.count_ones(),
            LayerKind::Vector(gl) => gl.features.len(),
            LayerKind::Raster(r) => r.width * r.height,
        }
    }
    pub fn feature(&self, idx: usize) -> Option<&GisFeature> {
        match self {
            LayerKind::Vector(gis_layer) => Some(&gis_layer.features[idx]),
            LayerKind::Points(point_cloud_layer) => None,
            LayerKind::Raster(_) => None,
        }
    }
    pub fn field_names(&self) -> Vec<String> {
        match self {
            LayerKind::Points(point_cloud_layer) => point_cloud_layer.field_names.clone(),
            LayerKind::Vector(gis_layer) => gis_layer.field_names.clone(),
            LayerKind::Raster(_) => Vec::new(),
        }
    }
    pub fn numeric_field_names(&self) -> Vec<String> {
        match self {
            LayerKind::Points(pc) => pc.numeric_field_names(),
            LayerKind::Vector(gl) => gl.field_names.clone(),
            LayerKind::Raster(_) => Vec::new(),
        }
    }
    pub fn column_type_for(&self, name: &str) -> Option<AttributeType> {
        match self {
            LayerKind::Points(pc) => pc
                .field_names
                .iter()
                .zip(pc.attributes.iter())
                .find(|(n, _)| n.as_str() == name)
                .map(|(_, col)| match col {
                    AttributeColumn::Float(_) => AttributeType::Float,
                    AttributeColumn::Integer(_) => AttributeType::Integer,
                    AttributeColumn::Text(_) => AttributeType::Text,
                }),
            LayerKind::Vector(gl) => gl
                .features
                .first()
                .and_then(|f| f.attributes.get(name))
                .map(|v| match v {
                    AttributeValue::Float(_) => AttributeType::Float,
                    AttributeValue::Integer(_) => AttributeType::Integer,
                    AttributeValue::Text(_) => AttributeType::Text,
                }),
            LayerKind::Raster(_) => None,
        }
    }
    pub fn index(&self, kind: IndexKind) -> Option<&SpatialIndex> {
        match self {
            LayerKind::Points(point_cloud_layer) => point_cloud_layer.index.as_deref(),
            LayerKind::Vector(gis_layer) => gis_layer.index(kind),
            LayerKind::Raster(_) => None,
        }
    }
    pub fn extent(&self) -> Option<[f64; 4]> {
        match self {
            LayerKind::Points(point_cloud_layer) => point_cloud_layer.bbox,
            LayerKind::Vector(gis_layer) => Some(gis_layer.world_bbox),
            LayerKind::Raster(r) => Some(r.extent),
        }
    }
    pub fn hit_test(&self, x: f64, y: f64, tolerance: f64) -> Option<usize> {
        match self {
            LayerKind::Points(point_cloud_layer) => point_cloud_layer.hit_test(x, y, tolerance),
            LayerKind::Vector(gis_layer) => gis_layer.hit_test(x, y, tolerance),
            LayerKind::Raster(_) => None,
        }
    }
    pub fn point_attrs_display(&self, idx: usize) -> Option<Vec<String>> {
        match self {
            LayerKind::Points(pc) if !pc.attributes.is_empty() => Some(
                pc.attributes
                    .iter()
                    .map(|col| col.get_display(idx))
                    .collect(),
            ),
            _ => None,
        }
    }
}
// LayerEntry
pub struct LayerEntry {
    pub data: LayerKind,
    pub visible: bool,
    /// Whether raw points/features render for this layer. Independent of
    /// `visible`, which also gates the layer's index/heatmap overlays — this
    /// lets a heatmap show without the point cloud cluttering it.
    pub show_points: bool,
    pub name: String,
    pub color: [u8; 3],
    /// When set, vector features are colored by the distinct values of this
    /// attribute (see `categorical_color`) instead of the uniform `color`
    /// above. Ignored for Points/Raster layers.
    pub color_by: Option<String>,
    pub opacity: u8,
    pub descriptor: LayerDescriptor,
    pub filters: Vec<LayerAttributeFilter>,
    pub filter_logic: FilterLogic,
    /// Heatmap-cell regions of interest selected by the user to progressively
    /// narrow the analysis area. Empty means no spatial restriction. Multiple
    /// entries are unioned (OR) with each other, then ANDed with `filters`.
    pub roi_bboxes: Vec<[f64; 4]>,
    /// Box-selections saved by the user, listed in the sidebar under this layer.
    pub selections: Vec<LayerSelection>,
    /// Index into `selections` currently driving the Selection Stats window.
    pub active_selection: Option<usize>,
    /// Set when the user opted to reproject this layer to WGS84 on load;
    /// applied to incoming `BatchMessage` batches as they stream in.
    pub crs_transform: Option<crate::crs::CrsTransform>,
    pub show_index: bool,
    pub index_kind: IndexKind,
    pub show_heatmap: bool,
    pub heatmap_metric: crate::heatmap::HeatmapMetric,
    pub heatmap_cache: Option<crate::heatmap::HeatmapLayer>,
    pub heatmap_dirty: bool,
    /// Whether the grid-based kernel density estimate overlay is shown.
    /// Independent of `show_heatmap`/`heatmap_cache` (quadtree-based).
    pub show_kde: bool,
    pub kde_cache: Option<crate::heatmap::HeatmapLayer>,
    /// Named heatmap/KDE snapshots saved under this layer, for later export
    /// to GeoTIFF or promotion to a raster layer — mirrors `selections`.
    pub saved_heatmaps: Vec<crate::heatmap::SavedHeatmap>,
    /// Index into `saved_heatmaps` currently loaded into `kde_cache` for
    /// display, if any — drives the selected-row highlight in the layer panel.
    pub active_saved_heatmap: Option<usize>,
    /// Whether the uniform-grid binning overlay is shown. Independent of
    /// `show_heatmap` (adaptive-quadtree-based, near-uniform leaf counts) and
    /// `show_kde` — a fixed cell size, so raw count-per-cell is a real
    /// density signal.
    pub show_gridbin: bool,
    pub gridbin_cache: Option<crate::heatmap::HeatmapLayer>,
    /// Which of `gridbin_cache`'s metrics colors the overlay — only
    /// `Density`/`AttributeMean` are meaningful (no per-cell samples for
    /// `Unpredictability`).
    pub gridbin_metric: crate::heatmap::HeatmapMetric,
    /// Whether the bivariate grid overlay is shown.
    pub show_bivariate_grid: bool,
    pub bivariate_grid_cache: Option<crate::bivariate::BivariateGridLayer>,
    /// Named bivariate grid snapshots saved under this layer — mirrors `saved_heatmaps`.
    pub saved_bivariate_grids: Vec<crate::bivariate::SavedBivariateGrid>,
    /// Index into `saved_bivariate_grids` currently loaded into `bivariate_grid_cache`.
    pub active_saved_bivariate_grid: Option<usize>,
    /// Set when this layer was loaded with "Batch load" on — tracks which
    /// batches have been pulled in so far, so the batch manager window can
    /// offer "load next" / "load range" for the rest of the file.
    pub batch_load: Option<BatchLoadState>,
}

/// Progressive-load bookkeeping for a layer loaded via `ReadOp::Range`.
/// `batch_size`/`selected_fields` are captured from the initial load so
/// later batches use the same schema; `loaded` records which 0-based batch
/// indices have been pulled in (out-of-order range loads leave gaps).
pub struct BatchLoadState {
    pub batch_size: u64,
    pub is_points: bool,
    pub selected_fields: Option<Vec<String>>,
    pub total_batches: u64,
    pub loaded: std::collections::HashSet<u64>,
}

impl LayerEntry {
    /// Builds a new layer containing only `ids` (positions into this
    /// layer's points/features) — the shared subsetting logic behind
    /// "create layer from selection" and "create layer from sample".
    /// `None` for a Raster source (rasters aren't feature-indexed).
    pub fn subset_by_ids(&self, ids: &[usize], new_name: String) -> Option<LayerEntry> {
        let mut descriptor = self.descriptor.clone();
        descriptor.name = new_name.clone();
        descriptor.num_features = ids.len() as u64;

        let data = match &self.data {
            LayerKind::Points(pc) => {
                let n = ids.len();
                let new_points: Vec<(u32, [f64; 2])> =
                    ids.iter().map(|&id| pc.points[id]).collect();
                let new_attrs: Vec<AttributeColumn> = pc
                    .attributes
                    .iter()
                    .map(|col| match col {
                        AttributeColumn::Text(v) => {
                            AttributeColumn::Text(ids.iter().map(|&id| v[id].clone()).collect())
                        }
                        AttributeColumn::Integer(v) => {
                            AttributeColumn::Integer(ids.iter().map(|&id| v[id]).collect())
                        }
                        AttributeColumn::Float(v) => {
                            AttributeColumn::Float(ids.iter().map(|&id| v[id]).collect())
                        }
                    })
                    .collect();
                let mut new_pc = PointCloudLayer {
                    points: std::sync::Arc::new(new_points),
                    attributes: new_attrs,
                    field_names: pc.field_names.clone(),
                    index: None,
                    bbox: None,
                    viewport_mask: bitvec::bitvec![0; n],
                    filter_mask: bitvec::bitvec![1; n],
                };
                new_pc.ensure_bbox();
                LayerKind::Points(new_pc)
            }
            LayerKind::Vector(gl) => {
                let new_features: Vec<GisFeature> = ids
                    .iter()
                    .enumerate()
                    .filter_map(|(new_id, &old_id)| {
                        gl.features.get(old_id).map(|f| {
                            GisFeature::new(new_id, f.geometry.clone(), f.attributes.clone())
                        })
                    })
                    .collect();
                let world_bbox = new_features.iter().map(|f| f.bbox()).fold(
                    [f64::MAX, f64::MAX, f64::MIN, f64::MIN],
                    |a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])],
                );
                LayerKind::Vector(GisLayer {
                    name: new_name.clone(),
                    file_path: String::new(),
                    filter_mask: bitvec::bitvec![1; new_features.len()],
                    features: new_features,
                    field_names: gl.field_names.clone(),
                    extra_field_names: gl.extra_field_names.clone(),
                    quadtree: None,
                    point_only: gl.point_only,
                    world_bbox,
                })
            }
            LayerKind::Raster(_) => return None,
        };

        Some(LayerEntry {
            data,
            visible: true,
            show_points: true,
            name: new_name,
            color: self.color,
            color_by: None,
            opacity: self.opacity,
            descriptor,
            filters: Vec::new(),
            filter_logic: FilterLogic::default(),
            roi_bboxes: Vec::new(),
            selections: Vec::new(),
            active_selection: None,
            // Features are already-transformed copies of the source layer's
            // data, not a fresh streamed load.
            crs_transform: None,
            show_index: false,
            index_kind: IndexKind::Quadtree,
            show_heatmap: false,
            heatmap_metric: crate::heatmap::HeatmapMetric::Density,
            heatmap_cache: None,
            heatmap_dirty: true,
            show_kde: false,
            kde_cache: None,
            saved_heatmaps: Vec::new(),
            active_saved_heatmap: None,
            show_gridbin: false,
            gridbin_cache: None,
            gridbin_metric: crate::heatmap::HeatmapMetric::Density,
            show_bivariate_grid: false,
            bivariate_grid_cache: None,
            saved_bivariate_grids: Vec::new(),
            active_saved_bivariate_grid: None,
            batch_load: None,
        })
    }
}

/// A saved box-selection: a bbox plus the ids of features/points it captured.
/// For `LayerKind::Vector`, `ids` are `GisFeature.id` values (== their index in
/// `GisLayer.features`); for `LayerKind::Points`, `ids` are row indices into
/// `PointCloudLayer.points`/`attributes`.
pub struct LayerSelection {
    pub name: String,
    pub bbox: [f64; 4],
    pub ids: Vec<usize>,
}

#[derive(Default)]
pub struct GisLayer {
    pub name: String,
    pub file_path: String,
    pub features: Vec<GisFeature>,
    pub field_names: Vec<String>,
    pub extra_field_names: Vec<String>,
    pub quadtree: Option<SpatialIndex>,
    pub point_only: bool,
    pub world_bbox: [f64; 4],
    /// One bit per `features` entry; `false` means filtered out (mirrors
    /// `PointCloudLayer::filter_mask`). Starts all-`true`.
    pub filter_mask: bitvec::vec::BitVec,
}

impl GisLayer {
    pub fn index(&self, kind: IndexKind) -> Option<&SpatialIndex> {
        match kind {
            IndexKind::Quadtree => self.quadtree.as_ref(),
        }
    }

    fn ensure_world_bbox(&mut self) {
        if self.features.is_empty() {
            return;
        }
        let mut xmin = f64::MAX;
        let mut ymin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymax = f64::MIN;
        for f in &self.features {
            let bb = f.bbox();
            xmin = xmin.min(bb[0]);
            ymin = ymin.min(bb[1]);
            xmax = xmax.max(bb[2]);
            ymax = ymax.max(bb[3]);
        }
        self.world_bbox = [xmin, ymin, xmax, ymax];
    }

    pub fn rebuild_quadtree(&mut self, capacity: usize) {
        self.ensure_world_bbox();
        let mut qt = SpatialIndex::Quadtree(Quadtree::new(self.world_bbox, capacity));
        for f in &self.features {
            qt.insert(f.id, f.bbox());
        }
        self.quadtree = Some(qt);
    }
}

impl GisLayer {
    // pub fn load_all(path: &str) -> Result<Vec<Self>> {
    //     let dataset = Dataset::open(Path::new(path))?;
    //     let count = dataset.layer_count();
    //     let mut layers = Vec::new();
    //     for i in 0..count {
    //         match Self::load_layer(&dataset, i, path) {
    //             Ok(layer) => layers.push(layer),
    //             Err(_) => {}
    //         }
    //     }
    //     Ok(layers)
    // }

    // pub fn load_selected(path: &str, indices: &[usize]) -> Result<Vec<Self>> {
    //     let dataset = Dataset::open(Path::new(path))?;
    //     let mut layers = Vec::new();
    //     for &i in indices {
    //         match Self::load_layer(&dataset, i, path) {
    //             Ok(layer) => layers.push(layer),
    //             Err(_) => {}
    //         }
    //     }
    //     Ok(layers)
    // }

    // fn load_layer(dataset: &Dataset, layer_idx: usize, path: &str) -> Result<Self> {
    //     let mut layer = dataset.layer(layer_idx)?;
    //     let name = layer.name();

    //     let field_names: Vec<String> = layer.defn().fields().map(|f| f.name()).collect();

    //     let mut features: Vec<GisFeature> = Vec::new();

    //     for feature in layer.features() {
    //         let mut attributes: HashMap<String, AttributeValue> = HashMap::new();

    //         for (i, fname) in field_names.iter().enumerate() {
    //             if let Ok(Some(val)) = feature.field(i) {
    //                 let attr = match val {
    //                     FieldValue::IntegerValue(v) => AttributeValue::Integer(v as i64),
    //                     FieldValue::Integer64Value(v) => AttributeValue::Integer(v),
    //                     FieldValue::RealValue(v) => AttributeValue::Float(v),
    //                     FieldValue::StringValue(v) => AttributeValue::Text(v),
    //                     _ => continue,
    //                 };
    //                 attributes.insert(fname.clone(), attr);
    //             }
    //         }

    //         if let Some(geom_ref) = feature.geometry() {
    //             if let Some(geo_geom) = gdal_geom_to_geo(geom_ref) {
    //                 let id = features.len();
    //                 features.push(GisFeature::new(id, geo_geom, attributes));
    //             }
    //         }
    //     }

    //     let world_bbox = if features.is_empty() {
    //         [-180.0, -90.0, 180.0, 90.0]
    //     } else {
    //         let mut xmin = f64::MAX;
    //         let mut ymin = f64::MAX;
    //         let mut xmax = f64::MIN;
    //         let mut ymax = f64::MIN;
    //         for f in &features {
    //             let bb = f.bbox();
    //             xmin = xmin.min(bb[0]);
    //             ymin = ymin.min(bb[1]);
    //             xmax = xmax.max(bb[2]);
    //             ymax = ymax.max(bb[3]);
    //         }
    //         let pad_x = (xmax - xmin).abs() * 0.01;
    //         let pad_y = (ymax - ymin).abs() * 0.01;
    //         [xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y]
    //     };

    //     let mut quadtree: Box<dyn SpatialIndex> = Box::new(Quadtree::new(world_bbox, 100));
    //     let mut hilbert: Box<dyn SpatialIndex> = Box::new(HilbertRTree::new(world_bbox, 4));
    //     let mut point_only = true;
    //     for f in &features {
    //         quadtree.insert(f.id, f.bbox());
    //         hilbert.insert(f.id, f.bbox());
    //         if !f.tessellated.fill_idx.is_empty() || !f.tessellated.outlines.is_empty() {
    //             point_only = false;
    //         }
    //     }

    //     Ok(GisLayer {
    //         name,
    //         file_path: path.to_string(),
    //         features,
    //         field_names,
    //         extra_field_names: Vec::new(),
    //         quadtree,
    //         hilbert,
    //         point_only,
    //         world_bbox,
    //     })
    // }

    /// Returns IDs of features whose center points fall within the viewport.
    /// Falls back to a linear bbox scan when no quadtree has been built yet —
    /// without this, a freshly loaded vector layer renders nothing until the
    /// user manually rebuilds the index (mirrors `ids_in_bbox_with_fallback`).
    pub fn features_in_bbox(&self, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Vec<usize> {
        if let Some(idx) = self.quadtree.as_ref() {
            return idx.search(&[xmin, ymin, xmax, ymax]);
        }
        self.features
            .iter()
            .filter(|f| {
                let b = f.bbox();
                b[0] <= xmax && b[2] >= xmin && b[1] <= ymax && b[3] >= ymin
            })
            .map(|f| f.id)
            .collect()
    }

    /// Returns the ID of the first feature containing (or near) the world point.
    pub fn hit_test(&self, x: f64, y: f64, tolerance: f64) -> Option<usize> {
        use geo::{Contains, Distance, Euclidean};

        let candidates: Vec<usize> = self
            .index(IndexKind::Quadtree)
            .map(|index| {
                index.search(&[x - tolerance, y - tolerance, x + tolerance, y + tolerance])
            })
            .unwrap_or(Vec::new());

        let pt = Point::new(x, y);

        for id in candidates {
            let f = &self.features[id];
            let hit = match &f.geometry {
                Geometry::Polygon(p) => p.contains(&pt),
                Geometry::MultiPolygon(mp) => mp.contains(&pt),
                Geometry::Point(p) => Euclidean.distance(*p, pt) < tolerance,
                Geometry::MultiPoint(mp) => {
                    mp.0.iter().any(|p| Euclidean.distance(*p, pt) < tolerance)
                }
                Geometry::LineString(ls) => Euclidean.distance(ls, &pt) < tolerance,
                Geometry::MultiLineString(mls) => mls
                    .0
                    .iter()
                    .any(|ls| Euclidean.distance(ls, &pt) < tolerance),
                _ => false,
            };
            if hit {
                return Some(id);
            }
        }
        None
    }

    /// Overall bounding box of all features: [xmin, ymin, xmax, ymax].
    pub fn extent(&self) -> Option<[f64; 4]> {
        if self.features.is_empty() {
            return None;
        }
        let mut xmin = f64::MAX;
        let mut ymin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymax = f64::MIN;
        for f in &self.features {
            let bb = f.bbox();
            xmin = xmin.min(bb[0]);
            ymin = ymin.min(bb[1]);
            xmax = xmax.max(bb[2]);
            ymax = ymax.max(bb[3]);
        }
        Some([xmin, ymin, xmax, ymax])
    }

    pub fn all_field_names(&self) -> Vec<&str> {
        self.field_names
            .iter()
            .chain(self.extra_field_names.iter())
            .map(|s| s.as_str())
            .collect()
    }

    // pub fn save(&self, path: &str) -> Result<()> {
    //     let ext = Path::new(path)
    //         .extension()
    //         .and_then(|e| e.to_str())
    //         .unwrap_or("shp")
    //         .to_lowercase();

    //     let driver_name = match ext.as_str() {
    //         "shp" => "ESRI Shapefile",
    //         "gpkg" => "GPKG",
    //         "geojson" | "json" => "GeoJSON",
    //         "kml" => "KML",
    //         _ => "ESRI Shapefile",
    //     };

    //     let geom_type = self
    //         .features
    //         .first()
    //         .map(|f| infer_ogr_type(&f.geometry))
    //         .unwrap_or(OGRwkbGeometryType::wkbUnknown);

    //     let driver = gdal::DriverManager::get_driver_by_name(driver_name)?;
    //     let mut out_ds = driver.create_vector_only(path)?;
    //     let layer = out_ds.create_layer(gdal::vector::LayerOptions {
    //         name: &self.name,
    //         ty: geom_type,
    //         ..Default::default()
    //     })?;

    //     // Collect all attribute names and their types
    //     let all_names = self.all_field_names();
    //     let mut field_types: Vec<(&str, OGRFieldType::Type)> = Vec::new();

    //     for name in &all_names {
    //         let ogr_type = self
    //             .features
    //             .iter()
    //             .find_map(|f| f.attributes.get(*name))
    //             .map(|v| match v {
    //                 AttributeValue::Integer(_) => OGRFieldType::OFTInteger64,
    //                 AttributeValue::Float(_) => OGRFieldType::OFTReal,
    //                 AttributeValue::Text(_) => OGRFieldType::OFTString,
    //             })
    //             .unwrap_or(OGRFieldType::OFTString);
    //         field_types.push((name, ogr_type));
    //     }

    //     layer.create_defn_fields(&field_types)?;

    //     let defn = layer.defn();

    //     for feature in &self.features {
    //         let gdal_geom = geo_to_gdal_geom(&feature.geometry)?;
    //         let mut out_feature = gdal::vector::Feature::new(&defn)?;
    //         out_feature.set_geometry(gdal_geom)?;

    //         for (i, name) in all_names.iter().enumerate() {
    //             if let Some(val) = feature.attributes.get(*name) {
    //                 match val {
    //                     AttributeValue::Text(s) => out_feature.set_field_string(i, s)?,
    //                     AttributeValue::Integer(v) => out_feature.set_field_integer64(i, *v)?,
    //                     AttributeValue::Float(v) => out_feature.set_field_double(i, *v)?,
    //                 }
    //             }
    //         }

    //         out_feature.create(&layer)?;
    //     }

    //     Ok(())
    // }
}

// fn gdal_geom_to_geo(geom: &gdal::vector::Geometry) -> Option<geo_types::Geometry> {
//     use OGRwkbGeometryType as T;

//     let gt = geom.geometry_type();
//     if gt == T::wkbPoint || gt == T::wkbPoint25D || gt == T::wkbPointM || gt == T::wkbPointZM {
//         let (x, y, _) = geom.get_point(0);
//         return Some(Geometry::Point(Point::new(x, y)));
//     }
//     if gt == T::wkbLineString
//         || gt == T::wkbLineString25D
//         || gt == T::wkbLineStringM
//         || gt == T::wkbLineStringZM
//     {
//         return Some(Geometry::LineString(ring_to_linestring(geom)));
//     }
//     if gt == T::wkbPolygon
//         || gt == T::wkbPolygon25D
//         || gt == T::wkbPolygonM
//         || gt == T::wkbPolygonZM
//     {
//         return Some(Geometry::Polygon(gdal_poly_to_geo(geom)));
//     }
//     if gt == T::wkbMultiPoint
//         || gt == T::wkbMultiPoint25D
//         || gt == T::wkbMultiPointM
//         || gt == T::wkbMultiPointZM
//     {
//         let pts: Vec<Point<f64>> = (0..geom.geometry_count())
//             .map(|i| {
//                 let sub = geom.get_geometry(i);
//                 let (x, y, _) = sub.get_point(0);
//                 Point::new(x, y)
//             })
//             .collect();
//         return Some(Geometry::MultiPoint(MultiPoint(pts)));
//     }
//     if gt == T::wkbMultiLineString
//         || gt == T::wkbMultiLineString25D
//         || gt == T::wkbMultiLineStringM
//         || gt == T::wkbMultiLineStringZM
//     {
//         let lines: Vec<LineString<f64>> = (0..geom.geometry_count())
//             .map(|i| ring_to_linestring(&geom.get_geometry(i)))
//             .collect();
//         return Some(Geometry::MultiLineString(MultiLineString(lines)));
//     }
//     if gt == T::wkbMultiPolygon
//         || gt == T::wkbMultiPolygon25D
//         || gt == T::wkbMultiPolygonM
//         || gt == T::wkbMultiPolygonZM
//     {
//         let polys: Vec<Polygon<f64>> = (0..geom.geometry_count())
//             .map(|i| gdal_poly_to_geo(&geom.get_geometry(i)))
//             .collect();
//         return Some(Geometry::MultiPolygon(MultiPolygon(polys)));
//     }
//     None
// }

// fn ring_to_linestring(geom: &gdal::vector::Geometry) -> LineString<f64> {
//     let coords: Vec<Coord<f64>> = (0..geom.point_count() as i32)
//         .map(|i| {
//             let (x, y, _) = geom.get_point(i);
//             Coord { x, y }
//         })
//         .collect();
//     LineString(coords)
// }

// fn gdal_poly_to_geo(geom: &gdal::vector::Geometry) -> Polygon<f64> {
//     let ring_count = geom.geometry_count();
//     if ring_count == 0 {
//         return Polygon::new(LineString(vec![]), vec![]);
//     }
//     let exterior = ring_to_linestring(&geom.get_geometry(0));
//     let interiors: Vec<LineString<f64>> = (1..ring_count)
//         .map(|i| ring_to_linestring(&geom.get_geometry(i)))
//         .collect();
//     Polygon::new(exterior, interiors)
// }

// fn geo_to_gdal_geom(geom: &Geometry<f64>) -> Result<gdal::vector::Geometry> {
//     use gdal::vector::Geometry as GGeom;

//     match geom {
//         Geometry::Point(p) => {
//             let mut g = GGeom::empty(OGRwkbGeometryType::wkbPoint)?;
//             g.add_point_2d((p.x(), p.y()));
//             Ok(g)
//         }
//         Geometry::LineString(ls) => {
//             let mut g = GGeom::empty(OGRwkbGeometryType::wkbLineString)?;
//             for c in ls.coords() {
//                 g.add_point_2d((c.x, c.y));
//             }
//             Ok(g)
//         }
//         Geometry::Polygon(poly) => {
//             let mut g = GGeom::empty(OGRwkbGeometryType::wkbPolygon)?;
//             let mut ext = GGeom::empty(OGRwkbGeometryType::wkbLinearRing)?;
//             for c in poly.exterior().coords() {
//                 ext.add_point_2d((c.x, c.y));
//             }
//             g.add_geometry(ext)?;
//             for hole in poly.interiors() {
//                 let mut ring = GGeom::empty(OGRwkbGeometryType::wkbLinearRing)?;
//                 for c in hole.coords() {
//                     ring.add_point_2d((c.x, c.y));
//                 }
//                 g.add_geometry(ring)?;
//             }
//             Ok(g)
//         }
//         Geometry::MultiPolygon(mp) => {
//             let mut g = GGeom::empty(OGRwkbGeometryType::wkbMultiPolygon)?;
//             for poly in &mp.0 {
//                 let sub = geo_to_gdal_geom(&Geometry::Polygon(poly.clone()))?;
//                 g.add_geometry(sub)?;
//             }
//             Ok(g)
//         }
//         Geometry::MultiPoint(mp) => {
//             let mut g = GGeom::empty(OGRwkbGeometryType::wkbMultiPoint)?;
//             for pt in &mp.0 {
//                 let sub = geo_to_gdal_geom(&Geometry::Point(*pt))?;
//                 g.add_geometry(sub)?;
//             }
//             Ok(g)
//         }
//         Geometry::MultiLineString(mls) => {
//             let mut g = GGeom::empty(OGRwkbGeometryType::wkbMultiLineString)?;
//             for ls in &mls.0 {
//                 let sub = geo_to_gdal_geom(&Geometry::LineString(ls.clone()))?;
//                 g.add_geometry(sub)?;
//             }
//             Ok(g)
//         }
//         _ => Err(anyhow!("Unsupported geometry type for export")),
//     }
// }

// fn infer_ogr_type(geom: &Geometry<f64>) -> OGRwkbGeometryType::Type {
//     match geom {
//         Geometry::Point(_) => OGRwkbGeometryType::wkbPoint,
//         Geometry::MultiPoint(_) => OGRwkbGeometryType::wkbMultiPoint,
//         Geometry::LineString(_) => OGRwkbGeometryType::wkbLineString,
//         Geometry::MultiLineString(_) => OGRwkbGeometryType::wkbMultiLineString,
//         Geometry::Polygon(_) => OGRwkbGeometryType::wkbPolygon,
//         Geometry::MultiPolygon(_) => OGRwkbGeometryType::wkbMultiPolygon,
//         _ => OGRwkbGeometryType::wkbUnknown,
//     }
// }
