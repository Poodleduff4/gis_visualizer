"""SDK for writing gis_editor plugins in Python.

A plugin talks to the running app over stdio using a small framed msgpack
protocol (see the Rust side: `src/plugin/protocol.rs`); this package hides
that entirely behind `Host` and `run_plugin`::

    from gis_editor_sdk import run_plugin

    def run(host):
        gdf = host.get_layer(0).to_geodataframe()
        buffered = gdf.copy()
        buffered["geometry"] = buffered.buffer(100)
        host.add_layer("buffered", buffered)

    if __name__ == "__main__":
        run_plugin(run)
"""

from .host import Host, HostError, Layer, LayerSummary
from .runtime import run_plugin

__all__ = ["run_plugin", "Host", "Layer", "LayerSummary", "HostError"]
