#!/usr/bin/env python3
"""Sample gis_editor plugin, speaking the host's msgpack protocol by hand
until the gis_editor_sdk package exists. Confirms the Plugins window
actually round-trips a real process, not just a UI placeholder.
"""
import struct
import sys
import time

import msgpack


def read_frame(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    return msgpack.unpackb(stream.read(length), raw=False)


def write_frame(stream, obj):
    payload = msgpack.packb(obj, use_bin_type=True)
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    msg = read_frame(stdin)
    if msg == "Init" or (isinstance(msg, dict) and "Init" in msg):
        # This plugin declares no params, but the host always sends Init
        # before Run, so skip it rather than erroring.
        msg = read_frame(stdin)
    if msg != "Run":
        write_frame(stdout, {"Error": {"msg": f"expected Run, got {msg!r}"}})
        return

    for pct in (0.0, 0.5, 1.0):
        write_frame(stdout, {"Progress": {"pct": pct, "msg": "working"}})
        time.sleep(0.3)

    write_frame(stdout, {"Log": {"level": "Info", "msg": "hello from hello-world"}})
    write_frame(stdout, {"Done": {"result": "Ok"}})


if __name__ == "__main__":
    main()
