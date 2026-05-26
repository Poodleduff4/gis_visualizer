pub struct HilbertCurve;

impl HilbertCurve {
    pub fn hilbert_index(order: u32, coords: &[f64], data_rect: &[f64; 4]) -> Option<u64> {
        let grid_size = 1u64 << order;
        match coords.len() {
            2 => {
                let ix = Self::to_grid(coords[0], data_rect[0], data_rect[2], grid_size);
                let iy = Self::to_grid(coords[1], data_rect[1], data_rect[3], grid_size);
                Some(Self::xy_to_idx(order, ix, iy))
            }
            4 => {
                let cx = (coords[0] + coords[2]) / 2.0;
                let cy = (coords[1] + coords[3]) / 2.0;
                let ix = Self::to_grid(cx, data_rect[0], data_rect[2], grid_size);
                let iy = Self::to_grid(cy, data_rect[1], data_rect[3], grid_size);
                Some(Self::xy_to_idx(order, ix, iy))
            }
            _ => None,
        }
    }

    pub fn to_grid(val: f64, val_min: f64, val_max: f64, grid_size: u64) -> u64 {
        if val_max == val_min {
            return 0;
        }
        let idx = ((val - val_min) / (val_max - val_min) * (grid_size - 1) as f64) as i64;
        idx.max(0).min((grid_size - 1) as i64) as u64
    }

    pub fn from_grid(grid_val: u64, val_min: f64, val_max: f64, grid_size: u64) -> f64 {
        if grid_size <= 1 {
            return (val_min + val_max) / 2.0;
        }
        val_min + (grid_val as f64 / (grid_size - 1) as f64) * (val_max - val_min)
    }

    pub fn xy_to_idx(order: u32, mut x: u64, mut y: u64) -> u64 {
        if order == 0 {
            return 0;
        }
        let grid_size = 1u64 << order;
        let mut s = grid_size >> 1;
        let mut d = 0u64;
        while s > 0 {
            let rx = if (x & s) > 0 { 1u64 } else { 0 };
            let ry = if (y & s) > 0 { 1u64 } else { 0 };
            d += s * s * ((3 * rx) ^ ry);
            let (nx, ny) = Self::rotate(grid_size, x, y, rx, ry);
            x = nx;
            y = ny;
            s >>= 1;
        }
        d
    }

    /// Returns the grid cell (ix, iy) whose centroid corresponds to `idx`.
    pub fn idx_to_xy(order: u32, mut idx: u64) -> (u64, u64) {
        if order == 0 {
            return (0, 0);
        }
        let mut x = 0u64;
        let mut y = 0u64;
        let mut s = 1u64;
        while s < (1u64 << order) {
            let rx = 1 & (idx / 2);
            let ry = 1 & (idx ^ rx);
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }
            x += s * rx;
            y += s * ry;
            idx /= 4;
            s *= 2;
        }
        (x, y)
    }

    /// Returns the world-space centroid of the grid cell at `idx`.
    pub fn idx_to_point(order: u32, idx: u64, data_rect: &[f64; 4]) -> [f64; 2] {
        let grid_size = 1u64 << order;
        let (ix, iy) = Self::idx_to_xy(order, idx);
        [
            Self::from_grid(ix, data_rect[0], data_rect[2], grid_size),
            Self::from_grid(iy, data_rect[1], data_rect[3], grid_size),
        ]
    }

    // n must be the total grid size (not the loop step), so n-1-x never underflows.
    pub fn rotate(n: u64, mut x: u64, mut y: u64, rx: u64, ry: u64) -> (u64, u64) {
        if ry == 0 {
            if rx == 1 {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        (x, y)
    }
}
