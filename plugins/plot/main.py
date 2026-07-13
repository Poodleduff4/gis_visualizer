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


def run(host: Host):
    layers = host.list_layers()
    point_layers = [l for l in layers if l.kind == "points"]
    if not point_layers:
        host.log("no point layers loaded", level="Warn")
        return

    target = point_layers[0]
    host.progress(0.0, f"reading {target.name} ({target.feature_count} points)")
    gdf: gpd.GeoDataFrame = host.get_layer(target.id).to_geodataframe()

    host.progress(0.6, "computing convex hull")
    hull = gdf.union_all().convex_hull

    result = gpd.GeoDataFrame(
        {"source_layer": [target.name]}, geometry=[hull], crs=gdf.crs
    )

    host.progress(0.9, "adding hull layer")
    host.add_layer(f"{target.name} (convex hull)", result)
    host.log(f"added convex hull of {target.feature_count} points from {target.name}")


if __name__ == "__main__":
    run_plugin(run)
