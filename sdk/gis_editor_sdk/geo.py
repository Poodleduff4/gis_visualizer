"""Conversion between the Arrow IPC byte buffers the host sends/expects and
a `geopandas.GeoDataFrame`. Two schemas come in, matching `plugin::bridge`:

- Vector layers: a `geometry` column of WKB bytes plus one nullable column
  per attribute field, typed `Float64` / `Int64` / `Utf8`.
- Points layers: flat `id` / `x` / `y` columns (no WKB — see
  `bridge::encode_points_layer`) plus one dense attribute column per field.

Either way, `arrow_to_geodataframe` always hands back a GeoDataFrame with
real Point/etc geometry, so plugin code doesn't need to care which kind of
layer it read. Writing back (`geodataframe_to_arrow`) always uses the
Vector/WKB schema — a plugin's output is naturally a new vector layer
regardless of what kind it read from, not another multi-million-row point
cloud.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely


def arrow_to_geodataframe(arrow_ipc: bytes) -> gpd.GeoDataFrame:
    reader = pa.ipc.open_stream(arrow_ipc)
    batch = reader.read_next_batch()
    table = pa.Table.from_batches([batch])
    is_points = "geometry" not in table.column_names

    # Geometry is pulled out before the pandas conversion, not popped after:
    # `to_pandas()`'s default numpy backend can't represent a nullable int
    # column (pandas has no NaN-capable int64), so any attribute column with
    # a null silently upcasts to float64 — an Integer(42) attribute would
    # come back as 42.0. `types_mapper=pd.ArrowDtype` keeps Arrow's own
    # nullable int/float/string types instead of numpy's lossy ones.
    if is_points:
        x = table.column("x").to_pylist()
        y = table.column("y").to_pylist()
        geometry = gpd.GeoSeries(gpd.points_from_xy(x, y))
        attrs_table = table.drop(["x", "y"])
    else:
        geometry = gpd.GeoSeries.from_wkb(table.column("geometry").to_pylist())
        attrs_table = table.drop(["geometry"])

    df = attrs_table.to_pandas(types_mapper=pd.ArrowDtype)
    return gpd.GeoDataFrame(df, geometry=geometry)


def arrow_to_raster(arrow_ipc: bytes) -> dict:
    """Decodes a raster layer (`bridge::encode_raster_layer`'s schema): one
    `Float32` column per band, `width`/`height`/`units`/`extent` carried as
    schema metadata rather than repeated per row. Returns a plain dict —
    `{"width", "height", "units", "extent", "bands": {name: np.ndarray}}` —
    with each band reshaped to `(height, width)`, row 0 = north, matching the
    host's grid convention.
    """
    reader = pa.ipc.open_stream(arrow_ipc)
    batch = reader.read_next_batch()
    schema = batch.schema

    meta = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items()}
    width = int(meta["width"])
    height = int(meta["height"])
    extent = tuple(float(v) for v in meta["extent"].split(","))

    bands = {
        name: batch.column(name).to_numpy(zero_copy_only=False).reshape(height, width)
        for name in schema.names
    }
    return {
        "width": width,
        "height": height,
        "units": meta.get("units", ""),
        "extent": extent,
        "bands": bands,
    }


def raster_to_arrow(
    bands: dict, extent: tuple[float, float, float, float], units: str = ""
) -> bytes:
    """Encodes a dict of 2D numpy arrays (`{name: array of shape (height,
    width)}`, row 0 = north, matching the host's grid convention — see
    `arrow_to_raster`) into the Arrow IPC schema `bridge::decode_raster_layer`
    expects: one `Float32` column per band (row-major flattened), with
    `width`/`height`/`units`/`extent` carried as schema metadata rather than
    repeated per row.
    """
    names = list(bands.keys())
    shapes = {bands[n].shape for n in names}
    if len(shapes) != 1:
        raise ValueError(f"all bands must share one (height, width) shape, got {shapes}")
    (height, width) = next(iter(shapes))

    fields = [pa.field(name, pa.float32(), nullable=False) for name in names]
    arrays = [
        pa.array(np.asarray(bands[name], dtype=np.float32).reshape(-1), type=pa.float32())
        for name in names
    ]

    metadata = {
        "width": str(width),
        "height": str(height),
        "units": units,
        "extent": ",".join(str(float(v)) for v in extent),
    }
    schema = pa.schema(fields, metadata=metadata)
    batch = pa.record_batch(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def geodataframe_to_arrow(gdf: gpd.GeoDataFrame) -> bytes:
    geometry_col = gdf.geometry.name
    geom_array = pa.array(shapely.to_wkb(gdf.geometry.values), type=pa.binary())

    fields = [pa.field("geometry", pa.binary(), nullable=False)]
    arrays = [geom_array]
    for col in gdf.columns:
        if col == geometry_col:
            continue
        series = gdf[col]
        if pd.api.types.is_float_dtype(series):
            arrow_type = pa.float64()
        elif pd.api.types.is_integer_dtype(series):
            arrow_type = pa.int64()
        else:
            arrow_type = pa.utf8()
            series = series.astype(object).where(series.notna(), None)
        fields.append(pa.field(col, arrow_type, nullable=True))
        arrays.append(pa.array(series, type=arrow_type))

    schema = pa.schema(fields)
    batch = pa.record_batch(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()
