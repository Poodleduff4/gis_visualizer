"""Buffers the active layer's geometry by a fixed distance (in the layer's
own CRS units) and adds the result as a new layer. `distance` comes from
this plugin's `plugin.toml` `[[params]]` — set it in the Plugins window
before running.
"""

from gis_editor_sdk import run_plugin
import geopandas as gpd
from gis_editor_sdk.host import Host

DEFAULT_BUFFER_DISTANCE = 100.0


def run(host: Host, params: dict):
    layers = host.list_layers()
    vector_layers = [l for l in layers if l.kind == "vector"]
    if not vector_layers:
        host.log("no vector layers loaded", level="Warn")
        return

    distance = float(params.get("distance", DEFAULT_BUFFER_DISTANCE))

    target = vector_layers[0]
    host.progress(0.0, f"reading {target.name}")
    gdf: gpd.GeoDataFrame = host.get_layer(target.id).to_geodataframe()

    host.progress(0.5, f"buffering {len(gdf)} features by {distance}")
    buffered = gdf.copy()
    buffered["geometry"] = buffered.geometry.buffer(distance)

    host.progress(0.9, "adding result layer")
    host.add_layer(f"{target.name} (buffered)", buffered)
    host.log(f"added buffered copy of {target.name}")


if __name__ == "__main__":
    run_plugin(run)
