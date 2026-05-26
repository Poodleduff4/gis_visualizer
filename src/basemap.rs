use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};

use egui::{Color32, ColorImage, Context, Painter, Pos2, Rect, TextureHandle, TextureOptions};

use crate::map_view::Viewport;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct TileId {
    z: u32,
    x: u32,
    y: u32,
}

enum TileState {
    Loading,
    Loaded(TextureHandle),
    Failed,
}

pub struct BasemapCache {
    tiles: Arc<Mutex<HashMap<TileId, TileState>>>,
}

impl Default for BasemapCache {
    fn default() -> Self {
        Self { tiles: Arc::new(Mutex::new(HashMap::new())) }
    }
}

impl BasemapCache {
    fn get_or_fetch(&self, id: TileId, ctx: &Context) -> Option<TextureHandle> {
        {
            let map = self.tiles.lock().unwrap();
            match map.get(&id) {
                Some(TileState::Loaded(h)) => return Some(h.clone()),
                Some(_) => return None,
                None => {}
            }
        }
        self.tiles.lock().unwrap().insert(id, TileState::Loading);
        let tiles = Arc::clone(&self.tiles);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let url = format!(
                "https://tile.openstreetmap.org/{}/{}/{}.png",
                id.z, id.x, id.y
            );
            let state = fetch_tile(&url, id, &ctx);
            tiles.lock().unwrap().insert(id, state);
            // Wake up once to display the newly arrived tile.
            ctx.request_repaint();
        });
        None
    }

    pub fn render(&self, painter: &Painter, viewport: &Viewport, rect: Rect, ctx: &Context) {
        let z = zoom_level(viewport);
        let n = 1u32 << z;
        let nf = n as f64;

        let [xmin, ymin, xmax, ymax] = viewport.viewport_bbox(rect);

        let lon_min = xmin.clamp(-180.0, 180.0);
        let lon_max = xmax.clamp(-180.0, 180.0);
        let lat_min = ymin.clamp(-85.051_129, 85.051_129);
        let lat_max = ymax.clamp(-85.051_129, 85.051_129);

        let tx0 = lon_to_tile(lon_min, nf);
        let tx1 = lon_to_tile(lon_max, nf).min(n - 1);
        // north = smaller tile y, south = larger tile y
        let ty0 = lat_to_tile(lat_max, nf);
        let ty1 = lat_to_tile(lat_min, nf).min(n - 1);

        // Don't request more than ~200 tiles per frame
        let cols = tx1.saturating_sub(tx0) + 1;
        let rows = ty1.saturating_sub(ty0) + 1;
        if cols * rows > 200 {
            return;
        }

        for tx in tx0..=tx1 {
            for ty in ty0..=ty1 {
                let tile_id = TileId { z, x: tx, y: ty };

                let w_lon0 = tile_to_lon(tx, nf);
                let w_lon1 = tile_to_lon(tx + 1, nf);
                let w_lat_n = tile_to_lat(ty, nf);       // north edge (larger lat)
                let w_lat_s = tile_to_lat(ty + 1, nf);   // south edge (smaller lat)

                // north edge → smaller screen y (top); south edge → larger screen y (bottom)
                let s_tl = viewport.world_to_screen(w_lon0, w_lat_n, rect);
                let s_br = viewport.world_to_screen(w_lon1, w_lat_s, rect);
                let tile_rect = Rect::from_two_pos(s_tl, s_br);

                if let Some(handle) = self.get_or_fetch(tile_id, ctx) {
                    painter.image(
                        handle.id(),
                        tile_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                } else {
                    painter.rect_filled(tile_rect, 0.0, Color32::from_rgb(210, 210, 210));
                }
            }
        }
    }
}

fn fetch_tile(url: &str, id: TileId, ctx: &Context) -> TileState {
    let result = ureq::get(url)
        .set("User-Agent", "gis_editor/0.1")
        .call();
    match result {
        Ok(resp) => {
            let mut bytes = Vec::new();
            if resp.into_reader().read_to_end(&mut bytes).is_err() {
                return TileState::Failed;
            }
            match image::load_from_memory(&bytes) {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    let color_image = ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        rgba.as_raw(),
                    );
                    let handle = ctx.load_texture(
                        format!("{}/{}/{}", id.z, id.x, id.y),
                        color_image,
                        TextureOptions::LINEAR,
                    );
                    TileState::Loaded(handle)
                }
                Err(_) => TileState::Failed,
            }
        }
        Err(_) => TileState::Failed,
    }
}

fn zoom_level(viewport: &Viewport) -> u32 {
    // pixels_per_unit = pixels per degree; one tile = 256px = (360/2^z) degrees
    let z = (viewport.pixels_per_unit * 360.0 / 256.0).log2().round() as i32;
    z.clamp(0, 19) as u32
}

fn lon_to_tile(lon: f64, n: f64) -> u32 {
    ((lon + 180.0) / 360.0 * n).floor().clamp(0.0, n - 1.0) as u32
}

fn lat_to_tile(lat: f64, n: f64) -> u32 {
    let lat_r = lat.to_radians();
    let y = (1.0 - (lat_r.tan() + 1.0 / lat_r.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
    y.floor().clamp(0.0, n - 1.0) as u32
}

fn tile_to_lon(x: u32, n: f64) -> f64 {
    x as f64 / n * 360.0 - 180.0
}

fn tile_to_lat(y: u32, n: f64) -> f64 {
    (std::f64::consts::PI * (1.0 - 2.0 * y as f64 / n))
        .sinh()
        .atan()
        .to_degrees()
}
