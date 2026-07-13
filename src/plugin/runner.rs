//! Drives a `PluginProcess` on a background thread and streams its calls
//! back as `PluginEvent`s over a channel, so the UI thread never blocks on
//! plugin I/O — it just polls the receiver each frame.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use super::manifest::PluginManifest;
use super::process::PluginProcess;
use super::protocol::{HostReply, HostRequest, LogLevel, PluginCall, PluginResult};

#[derive(Debug, Clone)]
pub enum PluginEvent {
    Log { level: LogLevel, msg: String },
    Progress { pct: f32, msg: String },
    /// The plugin asked for layer data (`ListLayers`/`GetLayer`/`AddLayer`/
    /// `UpdateLayer`). Servicing it touches `GisEditorApp`, which only the UI
    /// thread may access, so the runner thread blocks on `respond_to` while
    /// the UI thread computes and sends back the `HostReply`.
    LayerRequest {
        call: PluginCall,
        respond_to: mpsc::Sender<HostReply>,
    },
    Finished(Result<(), String>),
}

/// Spawns `manifest`'s entrypoint, sends `plugin_args` (the values the user
/// entered for `manifest.params`) via `Init`, then `Run`, and streams
/// Log/Progress/Done/Error calls back as `PluginEvent`s.
pub fn run_plugin(
    manifest: PluginManifest,
    plugin_args: serde_json::Value,
) -> mpsc::Receiver<PluginEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut proc = match PluginProcess::spawn(&manifest) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(PluginEvent::Finished(Err(e.to_string())));
                return;
            }
        };
        if let Err(e) = proc.send(&HostRequest::Init { plugin_args }) {
            let _ = tx.send(PluginEvent::Finished(Err(e.to_string())));
            return;
        }
        if let Err(e) = proc.send(&HostRequest::Run) {
            let _ = tx.send(PluginEvent::Finished(Err(e.to_string())));
            return;
        }
        loop {
            match proc.recv_call() {
                Ok(Some(PluginCall::Log { level, msg })) => {
                    let _ = tx.send(PluginEvent::Log { level, msg });
                }
                Ok(Some(PluginCall::Progress { pct, msg })) => {
                    let _ = tx.send(PluginEvent::Progress { pct, msg });
                }
                Ok(Some(PluginCall::Done { result })) => {
                    let outcome = match result {
                        PluginResult::Ok => Ok(()),
                        PluginResult::Failed { reason } => Err(reason),
                    };
                    let _ = tx.send(PluginEvent::Finished(outcome));
                    break;
                }
                Ok(Some(PluginCall::Error { msg })) => {
                    let _ = tx.send(PluginEvent::Finished(Err(msg)));
                    break;
                }
                Ok(Some(call)) => {
                    // ListLayers/GetLayer/AddLayer/UpdateLayer all need
                    // GisEditorApp, which only the UI thread may touch.
                    let (reply_tx, reply_rx) = mpsc::channel();
                    if tx
                        .send(PluginEvent::LayerRequest { call, respond_to: reply_tx })
                        .is_err()
                    {
                        break; // UI side dropped the receiver; nothing left to serve
                    }
                    let reply = reply_rx
                        .recv()
                        .unwrap_or_else(|_| HostReply::Err("host unavailable".into()));
                    if proc.send(&HostRequest::Reply(reply)).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = tx.send(PluginEvent::Finished(Err(
                        "plugin exited without sending Done".into(),
                    )));
                    break;
                }
                Err(e) => {
                    let _ = tx.send(PluginEvent::Finished(Err(e.to_string())));
                    break;
                }
            }
        }
        let _ = proc.shutdown(Duration::from_millis(500));
    });
    rx
}
