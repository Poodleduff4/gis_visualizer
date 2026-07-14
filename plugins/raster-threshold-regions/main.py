"""Edge-detects a chosen raster layer's band (Sobel gradient magnitude),
thresholds the raw values, and vectorizes each connected group of
above-threshold pixels into a polygon — added as a new vector layer with
per-region stats (mean value, mean/max edge strength, pixel count, area).

`layer`/`band`/`threshold`/`min_region_pixels` come from this plugin's
`plugin.toml` `[[params]]` (set in the Plugins window before running).
"""

from collections import deque

import geopandas as gpd
import numpy as np
import shapely
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_REGION_PIXELS = 4


def sobel_edges(values: np.ndarray) -> np.ndarray:
    """Gradient magnitude via a 3x3 Sobel kernel, edge-padded so the output
    keeps the input's shape. Pure numpy (no scipy/skimage dependency) —
    convolution done with shifted slices instead of an explicit loop.
    """
    padded = np.pad(values, 1, mode="edge")
    gx = (
        padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[0:-2, 0:-2] - 2 * padded[1:-1, 0:-2] - padded[2:, 0:-2]
    )
    gy = (
        padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[0:-2, 0:-2] - 2 * padded[0:-2, 1:-1] - padded[0:-2, 2:]
    )
    return np.hypot(gx, gy)


def label_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected connected-component labeling of a boolean mask, via BFS.
    Every pixel is visited once overall regardless of region count/shape, so
    this is O(width*height) — plain Python rather than scipy.ndimage.label
    to keep the plugin dependency-free; fine at the sizes a threshold pass
    over one band is likely to produce, but a very large raster (many
    millions of pixels) would be slow.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[list[tuple[int, int]]] = []

    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or visited[start_row, start_col]:
                continue
            region = []
            queue = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            while queue:
                row, col = queue.popleft()
                region.append((row, col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r, c = row + dr, col + dc
                    if (
                        0 <= r < height
                        and 0 <= c < width
                        and mask[r, c]
                        and not visited[r, c]
                    ):
                        visited[r, c] = True
                        queue.append((r, c))
            regions.append(region)
    return regions


def run(host: Host, params: dict):
    layer_id = int(params["layer"])
    threshold = float(params.get("threshold", DEFAULT_THRESHOLD))
    min_region_pixels = int(params.get("min_region_pixels", DEFAULT_MIN_REGION_PIXELS))
    band_name = params.get("band") or ""

    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"

    host.progress(0.0, f"reading {name}")
    raster = host.get_layer(layer_id).to_raster()
    bands = raster["bands"]
    if band_name and band_name in bands:
        values = bands[band_name]
    else:
        band_name, values = next(iter(bands.items()))

    xmin, ymin, xmax, ymax = raster["extent"]
    width, height = raster["width"], raster["height"]
    cell_w = (xmax - xmin) / width
    cell_h = (ymax - ymin) / height

    host.progress(0.2, f"running edge detection on band '{band_name}'")
    edges = sobel_edges(values)

    host.progress(0.4, f"thresholding at {threshold}")
    mask = values > threshold

    host.progress(0.6, "labeling connected regions")
    regions = [r for r in label_regions(mask) if len(r) >= min_region_pixels]
    if not regions:
        host.log(
            f"no regions >= {min_region_pixels} px found above threshold {threshold}",
            level="Warn",
        )
        return

    host.progress(0.8, f"vectorizing {len(regions)} region(s)")

    def pixel_box(row: int, col: int):
        x0 = xmin + col * cell_w
        y1 = ymax - row * cell_h
        return shapely.box(x0, y1 - cell_h, x0 + cell_w, y1)

    geometries = []
    records = []
    for region in regions:
        boxes = [pixel_box(row, col) for row, col in region]
        geometries.append(shapely.union_all(boxes))
        region_values = np.array([values[row, col] for row, col in region])
        region_edges = np.array([edges[row, col] for row, col in region])
        records.append(
            {
                "pixel_count": len(region),
                "area": len(region) * cell_w * cell_h,
                "value_mean": float(region_values.mean()),
                "value_max": float(region_values.max()),
                "edge_mean": float(region_edges.mean()),
                "edge_max": float(region_edges.max()),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry=geometries)
    host.add_layer(f"{name} (regions > {threshold})", gdf)
    host.log(f"added {len(regions)} region(s) from '{name}'")


if __name__ == "__main__":
    run_plugin(run)
