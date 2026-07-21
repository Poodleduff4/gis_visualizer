"""Extracts origin and/or destination points from an OD points layer as a
plain Points layer — lets the core app's existing KDE/heatmap tooling (Analysis
> Kernel Density Estimation) run directly on trip endpoints instead of
drawing every trip as a line, which gets cluttered fast at scale.

`endpoint_set` controls what's output:
  - "origin"      — just the layer's own geometry (one point per trip)
  - "destination" — just the `dest_lat_col`/`dest_lon_col` pair
  - "both"        — both, concatenated, tagged with a `role` column
                    ("origin"/"destination") so they can be colored apart
                    (Layer Color > Color by attribute) or KDE'd separately
                    by filtering on `role` first.

Output is otherwise geometry-only — no mode/activity/weight assumptions,
since this needs to work on any OD dataset regardless of what attributes it
happens to carry.
"""

import geopandas as gpd
import pandas as pd
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host


def run(host: Host, params: dict):
    layer_id = int(params["od_layer"])
    dest_lat_col = params.get("dest_lat_col") or "d_lat"
    dest_lon_col = params.get("dest_lon_col") or "d_lon"
    endpoint_set = (params.get("endpoint_set") or "both").strip().lower()

    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"

    host.progress(0.0, f"reading {name}")
    gdf: gpd.GeoDataFrame = host.get_layer(layer_id).to_geodataframe()

    need_dest = endpoint_set in ("destination", "both")
    if need_dest:
        missing = [c for c in (dest_lat_col, dest_lon_col) if c not in gdf.columns]
        if missing:
            host.log(f"missing destination columns {missing} on {name}", level="Error")
            return

    host.progress(0.4, "building endpoint layer")
    frames = []
    if endpoint_set in ("origin", "both"):
        origins = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
        if endpoint_set == "both":
            origins["role"] = "origin"
        frames.append(origins)
    if endpoint_set in ("destination", "both"):
        dests = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(gdf[dest_lon_col], gdf[dest_lat_col]),
            crs=gdf.crs,
        )
        if endpoint_set == "both":
            dests["role"] = "destination"
        frames.append(dests)

    if not frames:
        host.log(f"endpoint_set must be origin/destination/both, got {endpoint_set!r}", level="Error")
        return

    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=gdf.crs)
    result = result[result.geometry.notna()]

    host.progress(0.9, "adding result layer")
    host.add_layer(f"{name} ({endpoint_set} endpoints)", result)
    host.log(f"built {len(result)} endpoint points from {len(gdf)} trips in {name}")


if __name__ == "__main__":
    run_plugin(run)
