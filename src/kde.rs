/// Grid-based kernel density estimation, modeled on QGIS's Heatmap
/// (Kernel Density Estimation) processing tool: points are dropped onto a
/// regular grid of `cell_size`-sided cells, and each point spreads its mass
/// across every cell within `radius` according to a radially symmetric 2D
/// kernel, normalized so each kernel integrates to 1 over its disk.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KdeKernel {
    Uniform,
    Triangular,
    Epanechnikov,
    /// Quartic / biweight — QGIS's default kernel.
    Quartic,
    Triweight,
}

impl KdeKernel {
    pub const ALL: [KdeKernel; 5] = [
        KdeKernel::Uniform,
        KdeKernel::Triangular,
        KdeKernel::Epanechnikov,
        KdeKernel::Quartic,
        KdeKernel::Triweight,
    ];

    pub fn label(self) -> &'static str {
        match self {
            KdeKernel::Uniform => "Uniform",
            KdeKernel::Triangular => "Triangular",
            KdeKernel::Epanechnikov => "Epanechnikov",
            KdeKernel::Quartic => "Quartic (default)",
            KdeKernel::Triweight => "Triweight",
        }
    }

    /// Kernel value at `dist` from the point, for a kernel of bandwidth `radius`.
    /// Zero outside the radius; each shape is normalized to integrate to 1
    /// over the disk of that radius (standard 2D kernel-density formulas).
    fn eval(self, dist: f64, radius: f64) -> f64 {
        if radius <= 0.0 || dist >= radius {
            return 0.0;
        }
        let u = dist / radius;
        let r2 = radius * radius;
        let pi = std::f64::consts::PI;
        match self {
            KdeKernel::Uniform => 1.0 / (pi * r2),
            KdeKernel::Triangular => 3.0 / (pi * r2) * (1.0 - u),
            KdeKernel::Epanechnikov => 2.0 / (pi * r2) * (1.0 - u * u),
            KdeKernel::Quartic => 3.0 / (pi * r2) * (1.0 - u * u).powi(2),
            KdeKernel::Triweight => 4.0 / (pi * r2) * (1.0 - u * u).powi(3),
        }
    }
}

pub struct KdeParams {
    pub cell_size: f64,
    pub radius: f64,
    pub kernel: KdeKernel,
}

/// Cap grid cell count so a too-small `cell_size` can't allocate unbounded memory.
const MAX_CELLS: usize = 4_000_000;

/// Builds a KDE grid over `bbox` (padded by `radius` on each side, as QGIS
/// does, so points just outside the extent still contribute). `weights`,
/// when given, must be the same length as `points`; a point's mass is
/// `weight * kernel(dist)` instead of just `kernel(dist)`.
///
/// Returns `(bbox, density)` pairs for every non-zero cell. Empty if the
/// requested grid would exceed `MAX_CELLS`.
pub fn build_kde_grid(
    points: &[[f64; 2]],
    weights: Option<&[f64]>,
    bbox: [f64; 4],
    params: &KdeParams,
) -> Vec<([f64; 4], f32)> {
    let radius = params.radius.max(1e-12);
    let cell_size = params.cell_size.max(1e-12);

    let xmin = bbox[0] - radius;
    let ymin = bbox[1] - radius;
    let xmax = bbox[2] + radius;
    let ymax = bbox[3] + radius;

    let width = (((xmax - xmin) / cell_size).ceil() as usize).max(1);
    let height = (((ymax - ymin) / cell_size).ceil() as usize).max(1);
    if width.saturating_mul(height) > MAX_CELLS {
        return Vec::new();
    }

    let mut grid = vec![0.0f64; width * height];

    // row 0 = north (largest y), matching this app's raster/heatmap convention.
    let col_of_x = |x: f64| ((x - xmin) / cell_size).floor();
    let row_of_y = |y: f64| ((ymax - y) / cell_size).floor();

    for (i, p) in points.iter().enumerate() {
        let w = weights.and_then(|ws| ws.get(i)).copied().unwrap_or(1.0);
        if w == 0.0 {
            continue;
        }
        let col_min = (col_of_x(p[0] - radius) - 1.0).max(0.0) as usize;
        let col_max = ((col_of_x(p[0] + radius) + 1.0).max(0.0) as usize).min(width - 1);
        let row_min = (row_of_y(p[1] + radius) - 1.0).max(0.0) as usize;
        let row_max = ((row_of_y(p[1] - radius) + 1.0).max(0.0) as usize).min(height - 1);

        for row in row_min..=row_max {
            let y_c = ymax - (row as f64 + 0.5) * cell_size;
            for col in col_min..=col_max {
                let x_c = xmin + (col as f64 + 0.5) * cell_size;
                let dist = ((x_c - p[0]).powi(2) + (y_c - p[1]).powi(2)).sqrt();
                let k = params.kernel.eval(dist, radius);
                if k > 0.0 {
                    grid[row * width + col] += w * k;
                }
            }
        }
    }

    let mut cells = Vec::new();
    for row in 0..height {
        let y1 = ymax - row as f64 * cell_size;
        let y0 = y1 - cell_size;
        for col in 0..width {
            let value = grid[row * width + col];
            if value > 0.0 {
                let x0 = xmin + col as f64 * cell_size;
                let x1 = x0 + cell_size;
                cells.push(([x0, y0, x1, y1], value as f32));
            }
        }
    }
    cells
}
