"""Buffers a chosen layer's geometry by a fixed distance (in the layer's own
CRS units) and adds the result as a new layer. `layer`/`distance` come from
this plugin's `plugin.toml` `[[params]]` — set them in the Plugins window
before running.
"""

from gis_editor_sdk import run_plugin
import geopandas as gpd
from gis_editor_sdk.host import Host

DEFAULT_BUFFER_DISTANCE = 100.0


def run(host: Host, params: dict):
    layer_id = int(params["layer"])
    distance = float(params.get("distance", DEFAULT_BUFFER_DISTANCE))

    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"

    host.progress(0.0, f"reading {name}")
    gdf: gpd.GeoDataFrame = host.get_layer(layer_id).to_geodataframe()

    host.progress(0.5, f"buffering {len(gdf)} features by {distance}")
    buffered = gdf.copy()
    buffered["geometry"] = buffered.geometry.buffer(distance)

    host.progress(0.9, "adding result layer")
    host.add_layer(f"{name} (buffered)", buffered)
    host.log(f"added buffered copy of {name}")


if __name__ == "__main__":
    run_plugin(run)
