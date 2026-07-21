"""Wire format shared with the Rust host's ``plugin::protocol`` module.

Framing: a little-endian ``u32`` byte length followed by that many bytes of
msgpack — identical in both directions. Rust enums use serde's default
external tagging, which this module's helpers mirror exactly:

- a unit variant (``HostRequest::Run``) becomes the bare string ``"Run"``
- a newtype variant (``HostReply::Err(String)``) becomes ``{"Err": "..."}``
  with the inner value unwrapped, not re-tagged
- a struct variant (``PluginCall::Log { level, msg }``) becomes
  ``{"Log": {"level": "Info", "msg": "..."}}``

Plugin authors should not need this module directly — see ``host.py``.
"""
import struct

import msgpack


def write_frame(stream, obj) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def read_frame(stream):
    """Returns ``None`` on clean EOF (the host closed the pipe)."""
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    payload = stream.read(length)
    return msgpack.unpackb(payload, raw=False)


def variant_name(value):
    """Splits a tagged enum value into ``(variant_name, inner_value)``.
    ``inner_value`` is ``None`` for unit variants.
    """
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict) and len(value) == 1:
        ((name, inner),) = value.items()
        return name, inner
    raise ValueError(f"not a tagged enum value: {value!r}")


class HostError(RuntimeError):
    """Raised when the host replies with an error to an RPC call, or the
    wire protocol is violated (unexpected variant, closed connection)."""


def unwrap(value, expected: str):
    """Asserts `value` is tagged `expected` and returns its inner payload.
    An `Err` variant always raises `HostError`, regardless of `expected`.
    """
    name, inner = variant_name(value)
    if name == "Err":
        raise HostError(inner)
    if name != expected:
        raise HostError(f"expected {expected!r}, got {name!r}: {inner!r}")
    return inner
