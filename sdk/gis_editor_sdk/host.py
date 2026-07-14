from . import protocol
from .geo import arrow_to_geodataframe, arrow_to_raster, geodataframe_to_arrow
from .protocol import HostError

__all__ = ["Host", "Layer", "LayerSummary", "HostError"]


class LayerSummary:
    def __init__(self, id: int, name: str, kind: str, feature_count: int, crs=None):
        self.id = id
        self.name = name
        self.kind = kind
        self.feature_count = feature_count
        self.crs = crs

    def __repr__(self) -> str:
        return (
            f"LayerSummary(id={self.id}, name={self.name!r}, kind={self.kind!r}, "
            f"feature_count={self.feature_count}, crs={self.crs!r})"
        )


class Layer:
    """A layer's data as pulled across the plugin boundary. Only converted
    to a GeoDataFrame on demand, since a plugin might just check
    `feature_count` from `list_layers` and skip pulling the geometry."""

    def __init__(self, arrow_ipc: bytes):
        self._arrow_ipc = arrow_ipc

    def to_geodataframe(self):
        """For vector/points layers — errors if called on a raster layer's
        `Layer` (check `LayerSummary.kind` from `list_layers` first, or use
        `to_raster()` instead)."""
        return arrow_to_geodataframe(self._arrow_ipc)

    def to_raster(self) -> dict:
        """For raster layers: `{"width", "height", "units", "extent",
        "bands": {name: np.ndarray of shape (height, width)}}`."""
        return arrow_to_raster(self._arrow_ipc)


class Host:
    """The plugin's view of the running app. Every method here is a
    blocking round-trip to the host process — there's exactly one plugin
    process talking to the host at a time, so there's nothing to gain from
    async here.
    """

    def __init__(self, stdin, stdout):
        self._stdin = stdin
        self._stdout = stdout

    def _call(self, call):
        protocol.write_frame(self._stdout, call)
        raw = protocol.read_frame(self._stdin)
        if raw is None:
            raise HostError("host closed the connection while awaiting a reply")
        name, inner = protocol.variant_name(raw)
        if name != "Reply":
            raise HostError(f"expected a Reply while awaiting an RPC result, got {name!r}")
        return inner

    def list_layers(self) -> list[LayerSummary]:
        reply = self._call("ListLayers")
        layers = protocol.unwrap(reply, "Layers")
        return [
            LayerSummary(l["id"], l["name"], l["kind"], l["feature_count"], l.get("crs"))
            for l in layers
        ]

    def get_layer(self, layer_id: int, want: str = "Both") -> Layer:
        reply = self._call({"GetLayer": {"layer_id": layer_id, "want": want}})
        data = protocol.unwrap(reply, "LayerData")
        return Layer(data["arrow_ipc"])

    def add_layer(self, name: str, gdf) -> None:
        arrow_ipc = geodataframe_to_arrow(gdf)
        reply = self._call({"AddLayer": {"name": name, "arrow_ipc": arrow_ipc}})
        protocol.unwrap(reply, "Ack")

    def update_layer(self, layer_id: int, gdf) -> None:
        arrow_ipc = geodataframe_to_arrow(gdf)
        reply = self._call({"UpdateLayer": {"layer_id": layer_id, "arrow_ipc": arrow_ipc}})
        protocol.unwrap(reply, "Ack")

    def log(self, msg: str, level: str = "Info") -> None:
        """Fire-and-forget — appears in the host's Plugins window log."""
        protocol.write_frame(self._stdout, {"Log": {"level": level, "msg": msg}})

    def progress(self, pct: float, msg: str = "") -> None:
        """`pct` in [0, 1]; drives the host's progress display."""
        protocol.write_frame(self._stdout, {"Progress": {"pct": pct, "msg": msg}})
