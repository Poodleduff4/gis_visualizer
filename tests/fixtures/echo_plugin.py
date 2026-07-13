#!/usr/bin/env python3
"""Minimal plugin used by the Rust round-trip test in plugin::process.
Speaks the same length-prefixed msgpack framing as protocol.rs by hand
(no SDK yet) so the test exercises the real subprocess/pipe path end to end.
"""
import struct
import sys

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
    if msg != "Run":
        write_frame(stdout, {"Error": {"msg": f"expected Run, got {msg!r}"}})
        return

    write_frame(stdout, {"Log": {"level": "Info", "msg": "hello from plugin"}})
    write_frame(stdout, {"Done": {"result": "Ok"}})


if __name__ == "__main__":
    main()
