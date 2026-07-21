"""Grids `source_layer` (points or vector) into square cells and, for each
cell, computes the aggregate distance from the source features that fall in
it to the nearest feature in `target_layer` (points or vector too — a line
layer like transit routes works, since distance-to-geometry doesn't care
whether the target is a point, line, or polygon). Output is a single-band
raster layer (`host.add_raster_layer`) — the host renders it through its
built-in blue->red colormap immediately, same as a loaded GeoTIFF, so higher
average distance shows up hotter with no extra styling step.

Nearest-neighbor search uses a `shapely.STRtree` over `target_layer`'s
geometries, queried in one vectorized `query_nearest` call rather than a
per-feature loop — O(n log m) instead of O(n*m). For a giant source layer
(millions of taxi points) that's still the dominant cost; `max_search_distance`
lets it skip target geometries far outside a source point's neighborhood,
which matters most when `target_layer` is a dense, complex line network.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host

STATS = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
}


def run(host: Host, params: dict):
    source_id = int(params["source_layer"])
    target_id = int(params["target_layer"])
    cell_size = float(params["cell_size"])
    stat_name = params.get("stat") or "mean"
    stat_fn = STATS[stat_name]
    max_search_distance = float(params.get("max_search_distance") or 0.0) or None

    if cell_size <= 0:
        host.log("cell_size must be > 0", level="Error")
        return

    layers = {l.id: l for l in host.list_layers()}
    source_name = layers[source_id].name if source_id in layers else f"layer {source_id}"
    target_name = layers[target_id].name if target_id in layers else f"layer {target_id}"

    host.progress(0.0, f"reading {source_name}")
    source: gpd.GeoDataFrame = host.get_layer(source_id).to_geodataframe()

    host.progress(0.2, f"reading {target_name}")
    target: gpd.GeoDataFrame = host.get_layer(target_id).to_geodataframe()

    if source.crs is not None and target.crs is not None and source.crs != target.crs:
        target = target.to_crs(source.crs)

    host.progress(0.4, f"finding nearest {target_name} feature for each {source_name} feature")
    tree = shapely.STRtree(target.geometry.values)

    # points (or geometries) with no target within max_search_distance are
    # dropped by query_nearest, so index by input_idx rather than assuming
    # alignment with `source`.
    idx_pair, dist = tree.query_nearest(
        source.geometry.values,
        max_distance=max_search_distance,
        return_distance=True,
        all_matches=False,
    )
    input_idx = idx_pair[0]
    if len(input_idx) == 0:
        host.log("no source features found within max_search_distance of any target feature", level="Warn")
        return
    if len(input_idx) < len(source):
        host.log(
            f"{len(source) - len(input_idx)} of {len(source)} source feature(s) had no "
            f"target within max_search_distance and were skipped",
            level="Warn",
        )

    points = source.geometry.values[input_idx]
    centroids = shapely.centroid(points)
    xs = shapely.get_x(centroids)
    ys = shapely.get_y(centroids)

    host.progress(0.7, "binning into grid cells")
    minx, miny, maxx, maxy = xs.min(), ys.min(), xs.max(), ys.max()
    nx = max(1, int(np.ceil((maxx - minx) / cell_size)) or 1)
    ny = max(1, int(np.ceil((maxy - miny) / cell_size)) or 1)

    if nx * ny > 4096 * 4096:
        host.log(
            f"grid would be {nx}x{ny} cells at this cell_size — increase cell_size",
            level="Error",
        )
        return

    col = np.clip(((xs - minx) / cell_size).astype(int), 0, nx - 1)
    row = np.clip(((maxy - ys) / cell_size).astype(int), 0, ny - 1)

    df = pd.DataFrame({"row": row, "col": col, "distance": dist})
    grouped = df.groupby(["row", "col"])["distance"].agg(stat_fn)

    host.progress(0.9, f"building {nx}x{ny} raster")
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    rows, cols = zip(*grouped.index)
    grid[np.array(rows), np.array(cols)] = grouped.values

    extent = (minx, maxy - ny * cell_size, minx + nx * cell_size, maxy)
    host.add_raster_layer(
        f"{source_name} distance to {target_name} ({stat_name})",
        {stat_name: grid},
        extent,
    )
    finite = grid[np.isfinite(grid)]
    host.log(
        f"added {nx}x{ny} raster from {len(input_idx)} feature(s), {stat_name} distance range "
        f"{finite.min():.1f}-{finite.max():.1f}"
    )


if __name__ == "__main__":
    run_plugin(run)
