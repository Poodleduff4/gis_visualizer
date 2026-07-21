"""Turns an OD (origin-destination) points layer into flow lines: one
LineString per trip, origin taken from the layer's own geometry, destination
from a `dest_lat_col`/`dest_lon_col` attribute pair. Output is geometry only
— no assumptions about mode/activity/weight-style attributes, since those
aren't present in every OD dataset; downstream styling (categorical color,
filtering) works off whatever attributes the source layer actually has.

With `bin_size` > 0, origin and destination coordinates are snapped to a
grid of that size (in the layer's own coordinate units) and duplicate
(origin bin, destination bin) pairs are collapsed into one flow line —
keeps the output renderable when the source has hundreds of thousands of
raw trips.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from gis_editor_sdk import run_plugin
from gis_editor_sdk.host import Host


def snap(series: pd.Series, bin_size: float) -> pd.Series:
    if bin_size <= 0:
        return series
    return (series / bin_size).round() * bin_size


def run(host: Host, params: dict):
    layer_id = int(params["od_layer"])
    dest_lat_col = params.get("dest_lat_col") or "d_lat"
    dest_lon_col = params.get("dest_lon_col") or "d_lon"
    bin_size = float(params.get("bin_size", 0.0))

    target = next((l for l in host.list_layers() if l.id == layer_id), None)
    name = target.name if target else f"layer {layer_id}"

    host.progress(0.0, f"reading {name}")
    gdf: gpd.GeoDataFrame = host.get_layer(layer_id).to_geodataframe()

    missing = [c for c in (dest_lat_col, dest_lon_col) if c not in gdf.columns]
    if missing:
        host.log(f"missing destination columns {missing} on {name}", level="Error")
        return

    host.progress(0.2, "reading origin/destination coordinates")
    trips = pd.DataFrame({
        "o_lon": gdf.geometry.x,
        "o_lat": gdf.geometry.y,
        "d_lon": gdf[dest_lon_col],
        "d_lat": gdf[dest_lat_col],
    })
    trips = trips.dropna(subset=["o_lon", "o_lat", "d_lon", "d_lat"])

    if bin_size > 0:
        host.progress(0.5, f"binning {len(trips)} trips (bin size {bin_size})")
        for col in ("o_lon", "o_lat", "d_lon", "d_lat"):
            trips[col] = snap(trips[col], bin_size)
        flows = trips.drop_duplicates(subset=["o_lon", "o_lat", "d_lon", "d_lat"])
    else:
        flows = trips

    host.progress(0.8, f"building {len(flows)} flow lines")
    geometry = [
        LineString([(row.o_lon, row.o_lat), (row.d_lon, row.d_lat)])
        for row in flows.itertuples()
    ]
    result = gpd.GeoDataFrame(geometry=geometry, crs=gdf.crs)

    host.progress(0.95, "adding result layer")
    host.add_layer(f"{name} (OD flows)", result)
    host.log(f"built {len(result)} flow lines from {len(trips)} trips in {name}")


if __name__ == "__main__":
    run_plugin(run)
