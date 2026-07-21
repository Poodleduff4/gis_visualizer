import inspect
import sys

from . import protocol
from .host import Host

__all__ = ["run_plugin"]


def run_plugin(handler) -> None:
    """Standard plugin entrypoint. Call this from ``if __name__ ==
    "__main__":`` with your plugin's ``run`` function::

        from gis_editor_sdk import run_plugin

        def run(host, params):
            distance = params.get("distance", 100.0)
            gdf = host.get_layer(0).to_geodataframe()
            gdf["geometry"] = gdf.buffer(distance)
            host.add_layer("result", gdf)

        if __name__ == "__main__":
            run_plugin(run)

    `params` is a plain `dict` built from the values the user entered for
    this plugin's `plugin.toml` `[[params]]` (empty if it declares none).
    A `run(host)` with no second parameter is also accepted, for plugins
    that don't need any input.

    Reads the host's `Init`/`Run` requests, calls `handler`, and reports
    `Done`/`Error` back regardless of how `handler` exits.
    """
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    msg = protocol.read_frame(stdin)
    name, inner = protocol.variant_name(msg) if msg is not None else (None, None)
    params = {}
    if name == "Init":
        params = (inner or {}).get("plugin_args") or {}
        msg = protocol.read_frame(stdin)
        name, _ = protocol.variant_name(msg) if msg is not None else (None, None)
    if name != "Run":
        raise RuntimeError(f"expected a Run request, got {msg!r}")

    host = Host(stdin, stdout)
    accepts_params = len(inspect.signature(handler).parameters) >= 2
    try:
        if accepts_params:
            handler(host, params)
        else:
            handler(host)
    except Exception as exc:
        protocol.write_frame(
            stdout, {"Done": {"result": {"Failed": {"reason": str(exc)}}}}
        )
        raise
    else:
        protocol.write_frame(stdout, {"Done": {"result": "Ok"}})
