//! Wire protocol between the Rust host and a plugin subprocess.
//!
//! Framing is the same in both directions: a little-endian `u32` byte
//! length followed by that many bytes of msgpack (`rmp-serde`). Layer data
//! itself travels as an Arrow IPC byte buffer nested inside these messages
//! (see `plugin::bridge`, added once the read path lands) rather than being
//! modeled field-by-field here.

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

/// Messages the host sends to the plugin process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostRequest {
    /// Sent once at startup with plugin-specific arguments (from the menu
    /// invocation, a dialog, etc). Plugins that need no input may ignore it.
    Init { plugin_args: serde_json::Value },
    /// Invoke the plugin's registered entrypoint.
    Run,
    /// Ask the plugin to exit cleanly; the host follows up with a kill if it
    /// doesn't within a grace period.
    Shutdown,
    /// Reply to a `PluginCall` the plugin made.
    Reply(HostReply),
}

/// Messages the plugin sends back to the host. Most of these are RPC calls
/// awaiting a `HostReply`; `Log`/`Progress`/`Done`/`Error` are fire-and-forget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCall {
    ListLayers,
    GetLayer { layer_id: u32, want: LayerWant },
    AddLayer {
        name: String,
        #[serde(with = "serde_bytes")]
        arrow_ipc: Vec<u8>,
    },
    UpdateLayer {
        layer_id: u32,
        #[serde(with = "serde_bytes")]
        arrow_ipc: Vec<u8>,
    },
    Log { level: LogLevel, msg: String },
    Progress { pct: f32, msg: String },
    Done { result: PluginResult },
    Error { msg: String },
}

/// What part of a layer a plugin wants pulled across the boundary — kept
/// separate since attribute-only requests skip WKB geometry encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerWant {
    Geometry,
    Attributes,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginResult {
    Ok,
    Failed { reason: String },
}

/// Host replies to a `PluginCall` RPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostReply {
    Layers(Vec<LayerSummary>),
    LayerData {
        #[serde(with = "serde_bytes")]
        arrow_ipc: Vec<u8>,
    },
    Ack,
    Err(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub id: u32,
    pub name: String,
    pub kind: String,
    pub feature_count: usize,
    pub crs: Option<String>,
}

/// Write one length-prefixed msgpack frame: `[u32 LE len][bytes]`.
pub fn write_frame<W: Write, T: Serialize>(w: &mut W, msg: &T) -> io::Result<()> {
    let bytes = rmp_serde::to_vec_named(msg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()
}

/// Read one length-prefixed msgpack frame. Returns `Ok(None)` on clean EOF
/// (peer closed the pipe between frames, e.g. process exit).
pub fn read_frame<R: Read, T: for<'de> Deserialize<'de>>(r: &mut R) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    let msg = rmp_serde::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frame_round_trips() {
        let msg = PluginCall::Log { level: LogLevel::Info, msg: "hello".into() };
        let mut buf = Vec::new();
        write_frame(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(buf);
        let decoded: PluginCall = read_frame(&mut cursor).unwrap().unwrap();
        match decoded {
            PluginCall::Log { level, msg } => {
                assert_eq!(level, LogLevel::Info);
                assert_eq!(msg, "hello");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn empty_stream_reads_as_none() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        let decoded: Option<PluginCall> = read_frame(&mut cursor).unwrap();
        assert!(decoded.is_none());
    }
}
