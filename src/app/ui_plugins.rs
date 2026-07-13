use std::collections::HashMap;

use crate::plugin::{self, LogLevel, ParamKind, PluginEvent, PluginManifest, PluginParam};

use super::GisEditorApp;

/// Editing buffer for one `PluginParam`'s current value in the Plugins
/// window. Numeric kinds keep their text as typed (not parsed until Run) so
/// an in-progress edit like `"-"` or `""` doesn't get clobbered every frame.
#[derive(Debug, Clone)]
pub(super) enum ParamEditValue {
    Text(String),
    Integer(String),
    Float(String),
    Bool(bool),
}

impl ParamEditValue {
    fn from_default(kind: ParamKind, default: Option<&toml::Value>) -> Self {
        match kind {
            ParamKind::Text => Self::Text(
                default
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ),
            ParamKind::Integer => Self::Integer(
                default
                    .and_then(toml::Value::as_integer)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            ),
            ParamKind::Float => Self::Float(
                default
                    .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            ),
            ParamKind::Bool => Self::Bool(default.and_then(toml::Value::as_bool).unwrap_or(false)),
        }
    }

    fn to_json(&self, label: &str) -> Result<serde_json::Value, String> {
        match self {
            Self::Text(s) => Ok(serde_json::Value::String(s.clone())),
            Self::Integer(s) => s
                .trim()
                .parse::<i64>()
                .map(serde_json::Value::from)
                .map_err(|_| format!("{label}: {s:?} isn't a whole number")),
            Self::Float(s) => s
                .trim()
                .parse::<f64>()
                .map(serde_json::Value::from)
                .map_err(|_| format!("{label}: {s:?} isn't a number")),
            Self::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        match self {
            Self::Text(s) => {
                ui.text_edit_singleline(s);
            }
            Self::Integer(s) | Self::Float(s) => {
                ui.add(egui::TextEdit::singleline(s).desired_width(80.0));
            }
            Self::Bool(b) => {
                ui.checkbox(b, "");
            }
        }
    }
}

fn default_param_values(manifest: &PluginManifest) -> HashMap<String, ParamEditValue> {
    manifest
        .params
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                ParamEditValue::from_default(p.kind, p.default.as_ref()),
            )
        })
        .collect()
}

fn build_param_args(
    params: &[PluginParam],
    values: &HashMap<String, ParamEditValue>,
) -> Result<serde_json::Value, String> {
    let mut map = serde_json::Map::new();
    for p in params {
        let value = values
            .get(&p.name)
            .ok_or_else(|| format!("missing value for {}", p.display_label()))?;
        map.insert(p.name.clone(), value.to_json(p.display_label())?);
    }
    Ok(serde_json::Value::Object(map))
}

impl GisEditorApp {
    /// Drains any events from the currently running plugin's channel. Called
    /// once per frame regardless of whether the window is open, so log
    /// output isn't lost while the user has the window closed.
    pub(super) fn poll_plugin_events(&mut self) {
        // Taken out of `self` for the loop's duration: `LayerRequest` needs
        // `&mut self` to answer the plugin, which a live borrow of
        // `self.plugin_events_rx` would forbid. Put back at the end unless
        // the run finished or the channel disconnected.
        let Some(rx) = self.plugin_events_rx.take() else {
            return;
        };
        let mut keep = true;
        loop {
            match rx.try_recv() {
                Ok(PluginEvent::Log { level, msg }) => self.plugin_log.push((level, msg)),
                Ok(PluginEvent::Progress { pct, msg }) => {
                    self.plugin_log
                        .push((LogLevel::Info, format!("[{:.0}%] {msg}", pct * 100.0)));
                }
                Ok(PluginEvent::LayerRequest { call, respond_to }) => {
                    let reply = self.handle_plugin_call(call);
                    let _ = respond_to.send(reply);
                }
                Ok(PluginEvent::Finished(result)) => {
                    let name = self.plugin_running.take().unwrap_or_default();
                    match result {
                        Ok(()) => self
                            .plugin_log
                            .push((LogLevel::Info, format!("{name} finished."))),
                        Err(reason) => self
                            .plugin_log
                            .push((LogLevel::Error, format!("{name} failed: {reason}"))),
                    }
                    keep = false;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    keep = false;
                    break;
                }
            }
        }
        if keep {
            self.plugin_events_rx = Some(rx);
        }
    }

    pub(super) fn show_plugins_window(&mut self, ui: &mut egui::Ui) {
        if !self.plugin_window_open {
            return;
        }
        let mut open = true;
        egui::Window::new("Plugins")
            .open(&mut open)
            .resizable(true)
            .default_size([460.0, 420.0])
            .min_size([320.0, 240.0])
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Directory: {}", self.plugins_dir.display()));
                    if ui.button("Refresh").clicked() {
                        self.available_plugins = plugin::discover_plugins(&self.plugins_dir);
                    }
                });
                ui.separator();

                if self.available_plugins.is_empty() {
                    ui.label("No plugins found. Each plugin is a subdirectory with a plugin.toml.");
                } else {
                    egui::ScrollArea::vertical()
                        .id_salt("plugin_list_scroll")
                        .max_height(220.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            for i in 0..self.available_plugins.len() {
                                let manifest = self.available_plugins[i].clone();
                                let name = manifest.name.clone();
                                let running = self.plugin_running.as_deref() == Some(name.as_str());
                                let busy = self.plugin_running.is_some();

                                ui.horizontal(|ui| {
                                    ui.label(&name);
                                    if running {
                                        ui.spinner();
                                    } else if ui.add_enabled(!busy, egui::Button::new("Run")).clicked() {
                                        let args_result = if manifest.params.is_empty() {
                                            Ok(serde_json::Value::Object(Default::default()))
                                        } else {
                                            let values = self
                                                .plugin_param_values
                                                .entry(name.clone())
                                                .or_insert_with(|| default_param_values(&manifest));
                                            build_param_args(&manifest.params, values)
                                        };
                                        match args_result {
                                            Ok(args) => {
                                                self.plugin_running = Some(name.clone());
                                                self.plugin_log
                                                    .push((LogLevel::Info, format!("Running {name}…")));
                                                self.plugin_events_rx =
                                                    Some(plugin::run_plugin(manifest.clone(), args));
                                            }
                                            Err(e) => {
                                                self.plugin_log.push((LogLevel::Error, format!("{name}: {e}")));
                                            }
                                        }
                                    }
                                });

                                if !manifest.params.is_empty() {
                                    let values = self
                                        .plugin_param_values
                                        .entry(name.clone())
                                        .or_insert_with(|| default_param_values(&manifest));
                                    egui::Frame::new().inner_margin(egui::Margin { left: 16, ..Default::default() }).show(ui, |ui| {
                                        for p in &manifest.params {
                                            if let Some(v) = values.get_mut(&p.name) {
                                                ui.horizontal(|ui| {
                                                    ui.label(p.display_label());
                                                    v.ui(ui);
                                                });
                                            }
                                        }
                                    });
                                }
                                ui.separator();
                            }
                        });
                }

                ui.separator();
                ui.label("Log:");
                // Fills whatever height the window has left, rather than a
                // fixed pixel height — otherwise resizing the window taller
                // doesn't grow the log area, which is the "weird scrolling"
                // (the area stops responding to available space) users hit.
                let log_height = ui.available_height();
                egui::ScrollArea::vertical()
                    .id_salt("plugin_log_scroll")
                    .max_height(log_height)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for (level, msg) in &self.plugin_log {
                            let color = match level {
                                LogLevel::Error => egui::Color32::LIGHT_RED,
                                LogLevel::Warn => egui::Color32::YELLOW,
                                LogLevel::Info => ui.visuals().text_color(),
                                LogLevel::Debug => ui.visuals().weak_text_color(),
                            };
                            // An explicit `Label` (rather than `colored_label`)
                            // so we can force wrapping: inside a horizontally
                            // auto-shrinking ScrollArea, unwrapped text just
                            // gets clipped at the window edge instead of
                            // wrapping or scrolling — the "cut off" log lines.
                            ui.add(
                                egui::Label::new(egui::RichText::new(msg).color(color))
                                    .wrap(),
                            );
                        }
                    });
            });
        self.plugin_window_open = open;
    }
}
