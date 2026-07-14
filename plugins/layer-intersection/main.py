"""Keeps only the points from a Points layer that fall within a polygon
layer's geometry, and adds the result as a new point layer. `points_layer`/
`polygon_layer` come from this plugin's `plugin.toml` `[[params]]` — pick
them in the Plugins window before running.
"""

import geopandas as gpd
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host


def run(host: Host, params: dict):
    points_id = int(params["points_layer"])
    polygon_id = int(params["polygon_layer"])

    layers = {l.id: l for l in host.list_layers()}
    points_name = layers[points_id].name if points_id in layers else f"layer {points_id}"
    polygon_name = layers[polygon_id].name if polygon_id in layers else f"layer {polygon_id}"

    host.progress(0.0, f"reading {points_name}")
    points: gpd.GeoDataFrame = host.get_layer(points_id).to_geodataframe()

    host.progress(0.3, f"reading {polygon_name}")
    polygons: gpd.GeoDataFrame = host.get_layer(polygon_id).to_geodataframe()

    host.progress(0.7, "intersecting")
    within_polygons = points.geometry.within(polygons.union_all())
    result = points[within_polygons].copy()

    host.progress(0.9, "adding result layer")
    host.add_layer(f"{points_name} in {polygon_name}", result)
    host.log(f"kept {len(result)} of {len(points)} points from {points_name}")


if __name__ == "__main__":
    run_plugin(run)
