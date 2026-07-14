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
    /// Selected layer index, if any — `None` until the user picks one (or
    /// `ui` auto-selects the sole matching layer).
    Layer(Option<usize>),
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
            // No sane default across sessions — a saved layer index would
            // point at whatever's loaded next time, not necessarily the
            // same layer. Left unselected; `ui` auto-picks when there's
            // exactly one candidate.
            ParamKind::Layer => Self::Layer(None),
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
            Self::Layer(sel) => sel
                .map(|idx| serde_json::Value::from(idx as u64))
                .ok_or_else(|| format!("{label}: select a layer")),
        }
    }

    /// `layer_options` is `(layer index, display name)` pairs already
    /// filtered to this param's `layer_kind` — empty/ignored for every kind
    /// but `Layer`. `id_salt` keys the dropdown's egui id so two `Layer`
    /// params in the same plugin don't collide.
    fn ui(&mut self, ui: &mut egui::Ui, id_salt: &str, layer_options: &[(usize, String)]) {
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
            Self::Layer(sel) => {
                if sel.is_none() {
                    if let Some((only_idx, _)) = layer_options.first() {
                        *sel = Some(*only_idx);
                    }
                }
                let selected_text = sel
                    .and_then(|idx| layer_options.iter().find(|(i, _)| *i == idx))
                    .map(|(_, name)| name.clone())
                    .unwrap_or_else(|| "Select a layer…".to_string());
                egui::ComboBox::from_id_salt(("plugin_layer_param", id_salt.to_string()))
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        for (idx, name) in layer_options {
                            ui.selectable_value(sel, Some(*idx), name);
                        }
                    });
                if layer_options.is_empty() {
                    ui.label(
                        egui::RichText::new("no matching layers loaded")
                            .color(ui.visuals().error_fg_color),
                    );
                }
            }
        }
    }
}

/// Whether `data` matches a `PluginParam.layer_kind` string (`"points"`,
/// `"vector"`, `"raster"`).
fn layer_kind_matches(data: &crate::gis_layer::LayerKind, kind: &str) -> bool {
    use crate::gis_layer::LayerKind;
    matches!(
        (data, kind),
        (LayerKind::Points(_), "points")
            | (LayerKind::Vector(_), "vector")
            | (LayerKind::Raster(_), "raster")
    )
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
            .default_size([420.0, 420.0])
            .min_size([300.0, 240.0])
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
                        .max_height(160.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            for i in 0..self.available_plugins.len() {
                                let name = self.available_plugins[i].name.clone();
                                let running = self.plugin_running.as_deref() == Some(name.as_str());

                                ui.horizontal(|ui| {
                                    if running {
                                        ui.spinner();
                                        ui.label(&name);
                                    } else if ui.selectable_label(false, &name).clicked() {
                                        self.plugin_config_open = Some(name.clone());
                                    }
                                });
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

    /// The "Run plugin" dialog opened by clicking a plugin in the list:
    /// its params (if any) plus Run/Cancel. Separate from the list window
    /// so it can be shown, moved and closed independently.
    pub(super) fn show_plugin_config_window(&mut self, ui: &mut egui::Ui) {
        let Some(name) = self.plugin_config_open.clone() else {
            return;
        };
        let Some(manifest) = self
            .available_plugins
            .iter()
            .find(|p| p.name == name)
            .cloned()
        else {
            self.plugin_config_open = None;
            return;
        };

        let mut open = true;
        let mut run_clicked = false;
        let mut cancel_clicked = false;
        egui::Window::new(format!("Run {name}"))
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .show(ui.ctx(), |ui| {
                if manifest.params.is_empty() {
                    ui.label("This plugin takes no parameters.");
                } else {
                    let values = self
                        .plugin_param_values
                        .entry(name.clone())
                        .or_insert_with(|| default_param_values(&manifest));
                    let layers = &self.layers;
                    egui::Grid::new("plugin_param_grid")
                        .num_columns(2)
                        .spacing([12.0, 6.0])
                        .show(ui, |ui| {
                            for p in &manifest.params {
                                if let Some(v) = values.get_mut(&p.name) {
                                    ui.label(p.display_label());
                                    if p.kind == ParamKind::Layer {
                                        let options: Vec<(usize, String)> = layers
                                            .iter()
                                            .enumerate()
                                            .filter(|(_, l)| {
                                                p.layer_kind
                                                    .as_deref()
                                                    .is_none_or(|k| layer_kind_matches(&l.data, k))
                                            })
                                            .map(|(i, l)| (i, l.name.clone()))
                                            .collect();
                                        v.ui(ui, &p.name, &options);
                                    } else {
                                        v.ui(ui, &p.name, &[]);
                                    }
                                    ui.end_row();
                                }
                            }
                        });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Run").clicked() {
                        run_clicked = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                });
            });

        if run_clicked {
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
                    self.plugin_events_rx = Some(plugin::run_plugin(manifest, args));
                    self.plugin_config_open = None;
                }
                Err(e) => {
                    self.plugin_log.push((LogLevel::Error, format!("{name}: {e}")));
                }
            }
        } else if cancel_clicked || !open {
            self.plugin_config_open = None;
        }
    }
}
