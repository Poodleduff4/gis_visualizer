/// Grid-based bivariate choropleth: bins points onto a regular grid of
/// `cell_size`-sided cells, averages two attributes per cell, then classifies
/// each cell into a 3x3 grid (tertiles of each attribute) for a bivariate
/// color map — same visual idiom as QGIS/ArcGIS bivariate renderers.

/// Cap grid cell count so a too-small `cell_size` can't allocate unbounded memory.
const MAX_CELLS: usize = 4_000_000;

pub struct BivariateCell {
    pub bbox: [f64; 4],
    /// Raw (pre-classification) per-cell mean of attribute A/B.
    pub mean_a: f32,
    pub mean_b: f32,
    /// Tertile class, 0 (low) ..= 2 (high), for each attribute.
    pub class_a: u8,
    pub class_b: u8,
}

pub struct BivariateGridLayer {
    pub cells: Vec<BivariateCell>,
    pub attr_a: String,
    pub attr_b: String,
    /// Tertile cut points (33rd/66th percentile of non-empty cell means).
    pub breaks_a: [f64; 2],
    pub breaks_b: [f64; 2],
}

impl BivariateGridLayer {
    /// Builds a bivariate grid over `bbox`. `points`, `values_a`, `values_b`
    /// are parallel arrays (same length, one entry per point).
    pub fn build(
        points: &[[f64; 2]],
        values_a: &[f64],
        values_b: &[f64],
        bbox: [f64; 4],
        cell_size: f64,
        attr_a: String,
        attr_b: String,
    ) -> Option<Self> {
        let cell_size = cell_size.max(1e-12);
        let [xmin, ymin, xmax, ymax] = bbox;
        let width = (((xmax - xmin) / cell_size).ceil() as usize).max(1);
        let height = (((ymax - ymin) / cell_size).ceil() as usize).max(1);
        if width.saturating_mul(height) > MAX_CELLS {
            return None;
        }

        let mut sum_a = vec![0.0f64; width * height];
        let mut sum_b = vec![0.0f64; width * height];
        let mut count = vec![0u32; width * height];

        // row 0 = north (largest y), matching this app's raster/heatmap convention.
        for (i, p) in points.iter().enumerate() {
            let (Some(&a), Some(&b)) = (values_a.get(i), values_b.get(i)) else {
                continue;
            };
            let col = (((p[0] - xmin) / cell_size).floor() as isize).clamp(0, width as isize - 1) as usize;
            let row = (((ymax - p[1]) / cell_size).floor() as isize).clamp(0, height as isize - 1) as usize;
            let idx = row * width + col;
            sum_a[idx] += a;
            sum_b[idx] += b;
            count[idx] += 1;
        }

        let mut means_a = Vec::new();
        let mut means_b = Vec::new();
        let mut raw: Vec<(usize, usize, f32, f32)> = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                if count[idx] == 0 {
                    continue;
                }
                let mean_a = (sum_a[idx] / count[idx] as f64) as f32;
                let mean_b = (sum_b[idx] / count[idx] as f64) as f32;
                means_a.push(mean_a as f64);
                means_b.push(mean_b as f64);
                raw.push((row, col, mean_a, mean_b));
            }
        }
        if raw.is_empty() {
            return None;
        }

        let breaks_a = tertile_breaks(&mut means_a);
        let breaks_b = tertile_breaks(&mut means_b);

        let cells = raw
            .into_iter()
            .map(|(row, col, mean_a, mean_b)| {
                let y1 = ymax - row as f64 * cell_size;
                let y0 = y1 - cell_size;
                let x0 = xmin + col as f64 * cell_size;
                let x1 = x0 + cell_size;
                BivariateCell {
                    bbox: [x0, y0, x1, y1],
                    mean_a,
                    mean_b,
                    class_a: classify(mean_a as f64, breaks_a),
                    class_b: classify(mean_b as f64, breaks_b),
                }
            })
            .collect();

        Some(Self {
            cells,
            attr_a,
            attr_b,
            breaks_a,
            breaks_b,
        })
    }
}

/// 33rd/66th percentile cut points of `values` (sorted in place).
fn tertile_breaks(values: &mut [f64]) -> [f64; 2] {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    let lo = values[(n / 3).min(n - 1)];
    let hi = values[(2 * n / 3).min(n - 1)];
    [lo, hi]
}

fn classify(value: f64, breaks: [f64; 2]) -> u8 {
    if value <= breaks[0] {
        0
    } else if value <= breaks[1] {
        1
    } else {
        2
    }
}

/// A named, persisted snapshot of a built bivariate grid, stored under the
/// layer it was built for — mirrors `heatmap::SavedHeatmap`. Cells are raw
/// (physical-unit) per-attribute means, not classes, so breaks can be
/// recomputed or the values re-rendered if reclassified later.
pub struct SavedBivariateGrid {
    pub name: String,
    pub bbox: [f64; 4],
    pub cell_size: f64,
    pub attr_a: String,
    pub attr_b: String,
    pub cells: Vec<([f64; 4], f32, f32)>,
    pub breaks_a: [f64; 2],
    pub breaks_b: [f64; 2],
}

impl SavedBivariateGrid {
    pub fn from_layer(name: String, cell_size: f64, layer: &BivariateGridLayer) -> Self {
        let bbox = layer.cells.iter().fold(
            [f64::MAX, f64::MAX, f64::MIN, f64::MIN],
            |acc, c| {
                [
                    acc[0].min(c.bbox[0]),
                    acc[1].min(c.bbox[1]),
                    acc[2].max(c.bbox[2]),
                    acc[3].max(c.bbox[3]),
                ]
            },
        );
        let cells = layer
            .cells
            .iter()
            .map(|c| (c.bbox, c.mean_a, c.mean_b))
            .collect();
        Self {
            name,
            bbox,
            cell_size,
            attr_a: layer.attr_a.clone(),
            attr_b: layer.attr_b.clone(),
            cells,
            breaks_a: layer.breaks_a,
            breaks_b: layer.breaks_b,
        }
    }

    /// Rasterizes the saved grid into three uniform bands for GeoTIFF export:
    /// raw mean A, raw mean B, and a combined class index (`class_a*3 +
    /// class_b`, 0..=8) so a GIS tool can either inspect the raw values or
    /// symbolize the class band directly. Returns `(width, height,
    /// cell_size, band_a, band_b, band_class)`.
    pub fn rasterize(&self) -> (usize, usize, f64, Vec<f32>, Vec<f32>, Vec<f32>) {
        let cells_a: Vec<([f64; 4], f32)> =
            self.cells.iter().map(|(b, a, _)| (*b, *a)).collect();
        let cells_b: Vec<([f64; 4], f32)> =
            self.cells.iter().map(|(b, _, bb)| (*b, *bb)).collect();
        let (width, height, cell_size, band_a) =
            crate::heatmap::rasterize_cells(&cells_a, self.bbox, self.cell_size);
        let (_, _, _, band_b) = crate::heatmap::rasterize_cells(&cells_b, self.bbox, self.cell_size);
        let band_class: Vec<f32> = band_a
            .iter()
            .zip(band_b.iter())
            .map(|(&a, &b)| {
                if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else {
                    let ca = classify(a as f64, self.breaks_a);
                    let cb = classify(b as f64, self.breaks_b);
                    (ca * 3 + cb) as f32
                }
            })
            .collect();
        (width, height, cell_size, band_a, band_b, band_class)
    }

    /// Rebuilds a renderable `BivariateGridLayer` from the saved raw means,
    /// reusing the breaks computed when the snapshot was built.
    pub fn to_layer(&self) -> BivariateGridLayer {
        let cells = self
            .cells
            .iter()
            .map(|(bbox, mean_a, mean_b)| BivariateCell {
                bbox: *bbox,
                mean_a: *mean_a,
                mean_b: *mean_b,
                class_a: classify(*mean_a as f64, self.breaks_a),
                class_b: classify(*mean_b as f64, self.breaks_b),
            })
            .collect();
        BivariateGridLayer {
            cells,
            attr_a: self.attr_a.clone(),
            attr_b: self.attr_b.clone(),
            breaks_a: self.breaks_a,
            breaks_b: self.breaks_b,
        }
    }
}

/// Joshua Stevens' 3x3 bivariate palette (blue/teal x pink/purple corners).
/// `class_a` picks the column (0=low..2=high), `class_b` the row.
pub fn bivariate_color(class_a: u8, class_b: u8, alpha: u8) -> egui::Color32 {
    const PALETTE: [[(u8, u8, u8); 3]; 3] = [
        // class_b = 0 (low)
        [(0xe8, 0xe8, 0xe8), (0xac, 0xe4, 0xe4), (0x5a, 0xc8, 0xc8)],
        // class_b = 1 (mid)
        [(0xdf, 0xb0, 0xd6), (0xa5, 0xad, 0xd3), (0x56, 0x98, 0xb9)],
        // class_b = 2 (high)
        [(0xbe, 0x64, 0xac), (0x8c, 0x62, 0xaa), (0x3b, 0x49, 0x94)],
    ];
    let (r, g, b) = PALETTE[class_b.min(2) as usize][class_a.min(2) as usize];
    egui::Color32::from_rgba_unmultiplied(r, g, b, alpha)
}
