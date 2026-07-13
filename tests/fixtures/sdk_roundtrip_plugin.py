"""Used by plugin::sdk_test — exercises Host.get_layer/add_layer for real
against a Rust process playing a minimal stand-in host."""
from gis_editor_sdk import run_plugin


def run(host):
    layer = host.get_layer(0)
    gdf = layer.to_geodataframe()
    host.log(f"got {len(gdf)} features")
    host.add_layer("copy", gdf)


if __name__ == "__main__":
    run_plugin(run)
