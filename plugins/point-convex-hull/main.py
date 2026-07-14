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


def run(host: Host, params: dict):
    layer_id = int(params["layer"])
    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"

    host.progress(0.0, f"reading {name}")
    gdf: gpd.GeoDataFrame = host.get_layer(layer_id).to_geodataframe()

    host.progress(0.6, "computing convex hull")
    hull = gdf.union_all().convex_hull

    result = gpd.GeoDataFrame({"source_layer": [name]}, geometry=[hull], crs=gdf.crs)

    host.progress(0.9, "adding hull layer")
    host.add_layer(f"{name} (convex hull)", result)
    host.log(f"added convex hull of {len(gdf)} points from {name}")


if __name__ == "__main__":
    run_plugin(run)
