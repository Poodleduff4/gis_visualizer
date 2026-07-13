//! Plugin host: runs plugins as isolated subprocesses speaking a framed
//! msgpack protocol over stdio. See `protocol` for the message schema,
//! `process` for the child-process transport, `manifest` for plugin
//! discovery. Native-only — plugins are never available on wasm32.

pub mod bridge;
mod manifest;
mod process;
mod protocol;
mod runner;
#[cfg(test)]
mod sdk_test;

pub use manifest::{discover_plugins, ParamKind, PluginManifest, PluginParam};
pub use process::{PluginProcess, PluginProcessError};
pub use protocol::{HostReply, HostRequest, LayerSummary, LayerWant, LogLevel, PluginCall, PluginResult};
pub use runner::{run_plugin, PluginEvent};
