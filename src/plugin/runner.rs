//! Drives a `PluginProcess` on a background thread and streams its calls
//! back as `PluginEvent`s over a channel, so the UI thread never blocks on
//! plugin I/O — it just polls the receiver each frame.

use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::manifest::PluginManifest;
use super::process::{PluginKillHandle, PluginProcess};
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

/// A handle the UI thread keeps for a running plugin: the event stream plus
/// a way to kill the subprocess outright when the user gives up waiting on
/// it (the "Terminate" button) — there's no cooperative-cancellation
/// protocol message, since a hung plugin is exactly the case that can't be
/// trusted to respond to one.
pub struct PluginHandle {
    pub events: mpsc::Receiver<PluginEvent>,
    kill: Arc<Mutex<Option<PluginKillHandle>>>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl PluginHandle {
    /// Kills the plugin subprocess immediately. Safe to call even before the
    /// subprocess has finished spawning (the kill is applied as soon as the
    /// handle becomes available) or after the plugin already finished (a
    /// no-op, since there's nothing left to kill).
    pub fn terminate(&self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(h) = self.kill.lock().unwrap().as_ref() {
            h.kill();
        }
    }
}

/// Spawns `manifest`'s entrypoint, sends `plugin_args` (the values the user
/// entered for `manifest.params`) via `Init`, then `Run`, and streams
/// Log/Progress/Done/Error calls back as `PluginEvent`s.
pub fn run_plugin(manifest: PluginManifest, plugin_args: serde_json::Value) -> PluginHandle {
    let (tx, rx) = mpsc::channel();
    let kill: Arc<Mutex<Option<PluginKillHandle>>> = Arc::new(Mutex::new(None));
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let kill_thread = kill.clone();
    let cancelled_thread = cancelled.clone();
    thread::spawn(move || {
        let mut proc = match PluginProcess::spawn(&manifest) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(PluginEvent::Finished(Err(e.to_string())));
                return;
            }
        };
        *kill_thread.lock().unwrap() = Some(proc.kill_handle());
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
                    let msg = if cancelled_thread.load(std::sync::atomic::Ordering::SeqCst) {
                        "terminated by user".into()
                    } else {
                        "plugin exited without sending Done".into()
                    };
                    let _ = tx.send(PluginEvent::Finished(Err(msg)));
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
    PluginHandle { events: rx, kill, cancelled }
}
