//! Owns a running plugin child process and the framed stdio transport to it.
//! Read/write here are blocking `std::io` — callers are expected to drive a
//! `PluginProcess` from a dedicated thread, never from the UI thread.

use std::io::{BufReader, BufWriter};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

use super::manifest::PluginManifest;
use super::protocol::{self, HostRequest, PluginCall};

#[derive(Debug, thiserror::Error)]
pub enum PluginProcessError {
    #[error("failed to spawn plugin process: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("plugin process transport error: {0}")]
    Io(#[from] std::io::Error),
    #[error("plugin process exited before sending a Done/Error message")]
    UnexpectedExit,
}

pub struct PluginProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl PluginProcess {
    /// Spawn the plugin's entrypoint with stdio piped for the framed
    /// protocol. Fails immediately if the interpreter or script can't launch.
    pub fn spawn(manifest: &PluginManifest) -> Result<Self, PluginProcessError> {
        let mut child = Command::new(&manifest.python)
            .arg(manifest.entrypoint_path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(PluginProcessError::Spawn)?;

        let stdin = BufWriter::new(child.stdin.take().expect("stdin was piped"));
        let stdout = BufReader::new(child.stdout.take().expect("stdout was piped"));

        Ok(Self { child, stdin, stdout })
    }

    pub fn send(&mut self, msg: &HostRequest) -> Result<(), PluginProcessError> {
        protocol::write_frame(&mut self.stdin, msg)?;
        Ok(())
    }

    /// Blocks for the plugin's next call. `Ok(None)` means the plugin closed
    /// its stdout (process exiting or already dead).
    pub fn recv_call(&mut self) -> Result<Option<PluginCall>, PluginProcessError> {
        Ok(protocol::read_frame(&mut self.stdout)?)
    }

    /// Ask the plugin to shut down, wait briefly, then kill it if it hasn't
    /// exited on its own — a hung plugin must never be able to block the app.
    pub fn shutdown(mut self, grace: Duration) -> Result<(), PluginProcessError> {
        let _ = self.send(&HostRequest::Shutdown);
        let deadline = std::time::Instant::now() + grace;
        while std::time::Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::protocol::{LogLevel, PluginResult};
    use std::path::PathBuf;

    fn echo_plugin_manifest() -> PluginManifest {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        PluginManifest {
            name: "echo".into(),
            entrypoint: "echo_plugin.py".into(),
            python: "python3".into(),
            capabilities: Vec::new(),
            params: Vec::new(),
            dir,
        }
    }

    #[test]
    fn round_trips_through_a_real_subprocess() {
        let manifest = echo_plugin_manifest();
        let mut proc = PluginProcess::spawn(&manifest).expect("spawn echo plugin");

        proc.send(&HostRequest::Run).expect("send Run");

        let log = proc.recv_call().expect("recv log").expect("log present");
        match log {
            PluginCall::Log { level, msg } => {
                assert_eq!(level, LogLevel::Info);
                assert_eq!(msg, "hello from plugin");
            }
            other => panic!("expected Log, got {other:?}"),
        }

        let done = proc.recv_call().expect("recv done").expect("done present");
        match done {
            PluginCall::Done { result: PluginResult::Ok } => {}
            other => panic!("expected Done(Ok), got {other:?}"),
        }

        proc.shutdown(Duration::from_millis(500)).expect("shutdown");
    }
}
