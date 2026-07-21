"""Bins a layer's feature locations into a 2D grid and renders a 3D bar
chart (bar height = point count per cell) via matplotlib's mplot3d.
Interactive — `plt.show()` opens the plugin subprocess's own matplotlib
window (rotate/zoom/save from its toolbar), same pattern as the `plot`
plugin. The host protocol has no in-app image display path, so this rides
the subprocess's own display rather than round-tripping a PNG through the
host.

`bin_points` is kept pure/side-effect-free (no host, no matplotlib) so it's
unit-testable on its own.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host


def bin_points(xs: np.ndarray, ys: np.ndarray, bins: int):
    """2D histogram of (xs, ys) into `bins` x `bins` cells. Returns
    (counts, xedges, yedges), exactly like `numpy.histogram2d` — kept as a
    thin, pure wrapper so it's testable without matplotlib/geopandas.
    """
    if len(xs) == 0:
        raise ValueError("no points to bin")
    return np.histogram2d(xs, ys, bins=bins)


def plot_binned_3d(counts: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, title: str):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = (xedges[1] - xedges[0]) * 0.9 if len(xedges) > 1 else 1.0
    dy = (yedges[1] - yedges[0]) * 0.9 if len(yedges) > 1 else 1.0
    dz = counts.ravel()

    norm = dz / dz.max() if dz.max() > 0 else dz
    colors = plt.cm.viridis(norm)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("count")
    ax.set_title(title)

    plt.show()


def run(host: Host, params: dict):
    layer_id = int(params["layer"])
    bins = int(params.get("bins") or 20)
    title = params.get("title") or ""

    if bins < 1:
        host.log("bins must be >= 1", level="Error")
        return

    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"
    title = title or f"{name} — binned 3D density"

    host.progress(0.0, f"reading {name}")
    gdf: gpd.GeoDataFrame = host.get_layer(layer_id).to_geodataframe()
    if gdf.empty:
        host.log(f"{name} has no features", level="Warn")
        return

    host.progress(0.3, "binning points")
    centroids = gdf.geometry.centroid
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()
    counts, xedges, yedges = bin_points(xs, ys, bins)

    host.progress(0.6, "rendering 3D plot")
    plot_binned_3d(counts, xedges, yedges, title)

    host.log(f"closed 3D binned plot for {name}")


if __name__ == "__main__":
    run_plugin(run)
