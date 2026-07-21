/// Cap so a too-small `cell_size` can't allocate unbounded memory; doubles
/// the cell size until the grid fits instead of failing outright (mirrors
/// `kde::MAX_CELLS`).
const MAX_CELLS: usize = 4_000_000;

/// One occupied cell of a uniform grid binning: point count, plus (when an
/// attribute was supplied) that attribute's mean over the cell's points.
pub struct GridBinCell {
    pub bbox: [f64; 4],
    pub count: usize,
    pub mean: Option<f32>,
}

/// Bins `points` into a uniform grid over `bbox`, unlike the quadtree's
/// adaptive (capacity-driven) split: every cell is the same size, so raw
/// count-per-cell is a real density signal instead of the near-constant
/// value an adaptive quadtree's leaves produce. `values`, if given, must be
/// the same length as `points` (attribute value per point) and is averaged
/// per cell. Only occupied cells are returned.
pub fn build_gridbin(
    points: &[[f64; 2]],
    values: Option<&[f64]>,
    bbox: [f64; 4],
    cell_size: f64,
) -> Vec<GridBinCell> {
    let [xmin, ymin, xmax, ymax] = bbox;
    let mut cell_size = cell_size.max(1e-9);
    let mut cols = (((xmax - xmin) / cell_size).ceil() as usize).max(1);
    let mut rows = (((ymax - ymin) / cell_size).ceil() as usize).max(1);
    while cols.saturating_mul(rows) > MAX_CELLS {
        cell_size *= 2.0;
        cols = (((xmax - xmin) / cell_size).ceil() as usize).max(1);
        rows = (((ymax - ymin) / cell_size).ceil() as usize).max(1);
    }

    let mut counts = vec![0usize; cols * rows];
    let mut sums = vec![0f64; cols * rows];
    for (i, p) in points.iter().enumerate() {
        if p[0] < xmin || p[0] > xmax || p[1] < ymin || p[1] > ymax {
            continue;
        }
        let col = (((p[0] - xmin) / cell_size) as usize).min(cols - 1);
        let row = (((ymax - p[1]) / cell_size) as usize).min(rows - 1);
        let idx = row * cols + col;
        counts[idx] += 1;
        if let Some(vals) = values {
            sums[idx] += vals[i];
        }
    }

    let mut cells = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if counts[idx] == 0 {
                continue;
            }
            let cx0 = xmin + col as f64 * cell_size;
            let cy1 = ymax - row as f64 * cell_size;
            let cbbox = [cx0, cy1 - cell_size, cx0 + cell_size, cy1];
            let mean = values.map(|_| (sums[idx] / counts[idx] as f64) as f32);
            cells.push(GridBinCell {
                bbox: cbbox,
                count: counts[idx],
                mean,
            });
        }
    }
    cells
}
