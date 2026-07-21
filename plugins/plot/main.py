"""Computes the convex hull of the first Points layer's coordinates and adds
it as a new polygon Vector layer. Demonstrates a plugin reading a `Points`
layer specifically: `Host.get_layer` transparently hands back a GeoDataFrame
built from the point cloud's flat id/x/y columns (see
`gis_editor_sdk.geo.arrow_to_geodataframe`), so the read side looks
identical to reading a Vector layer even though the wire schema differs.
"""

import geopandas as gpd
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host
import matplotlib.pyplot as plt


def run(host: Host, params: dict):
    layer_id = int(params["layer"])
    x_col = params.get("x_col", "")
    y_col = params.get("y_col", "")

    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"

    host.progress(0.0, f"reading {name}")
    gdf: gpd.GeoDataFrame = host.get_layer(layer_id).to_geodataframe()

    host.progress(0.6, "computing plot")
    plt.scatter(gdf[x_col], gdf[y_col])
    plt.show()

    host.log(f"completed plot")


if __name__ == "__main__":
    run_plugin(run)
