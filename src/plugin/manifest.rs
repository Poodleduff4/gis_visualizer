//! Plugin discovery: each plugin is a directory containing `plugin.toml`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub entrypoint: String,
    #[serde(default = "default_python")]
    pub python: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// User-editable inputs shown in the Plugins window before Run, sent to
    /// the plugin as `HostRequest::Init { plugin_args }`. Order here is the
    /// order they're rendered in.
    #[serde(default)]
    pub params: Vec<PluginParam>,
    /// Directory the manifest was loaded from; not part of the toml itself.
    #[serde(skip)]
    pub dir: PathBuf,
}

fn default_python() -> String {
    "python3".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamKind {
    Text,
    Integer,
    Float,
    Bool,
    /// A dropdown of currently-loaded layers; the plugin receives the
    /// selected layer's index (the same id `list_layers`/`get_layer` use)
    /// as a plain integer under `plugin_args`.
    Layer,
    /// A dropdown of `attribute_of`'s currently-selected layer's field
    /// names, instead of a free-text column name — the plugin still
    /// receives a plain string under `plugin_args`, same as `Text`.
    Attribute,
    /// A dropdown over `options` (fixed list, set in the manifest) — the
    /// plugin receives the selected string under `plugin_args`, same as
    /// `Text`.
    Choice,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PluginParam {
    /// Key under `plugin_args` the plugin reads this value from.
    pub name: String,
    /// Shown in the UI in place of `name`, if set.
    #[serde(default)]
    pub label: Option<String>,
    pub kind: ParamKind,
    #[serde(default)]
    pub default: Option<toml::Value>,
    /// Only meaningful when `kind = "layer"`: restricts the dropdown to
    /// layers of this kind (`"points"`, `"vector"`, or `"raster"`). Absent
    /// shows every loaded layer regardless of kind.
    #[serde(default)]
    pub layer_kind: Option<String>,
    /// Only meaningful (and required) when `kind = "attribute"`: the `name`
    /// of this same plugin's `Layer`-kind param whose selected layer's
    /// fields populate the dropdown.
    #[serde(default)]
    pub attribute_of: Option<String>,
    /// Only meaningful (and required) when `kind = "choice"`: the fixed
    /// list of options shown in the dropdown.
    #[serde(default)]
    pub options: Vec<String>,
}

impl PluginParam {
    pub fn display_label(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.name)
    }
}

impl PluginManifest {
    pub fn entrypoint_path(&self) -> PathBuf {
        self.dir.join(&self.entrypoint)
    }

    fn load(manifest_path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(manifest_path)?;
        let mut manifest: PluginManifest = toml::from_str(&text)?;
        manifest.dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        Ok(manifest)
    }
}

/// Scan immediate subdirectories of `plugins_dir` for a `plugin.toml` each.
/// Plugins with an unreadable or invalid manifest are skipped rather than
/// failing the whole scan — one broken plugin shouldn't hide the rest.
pub fn discover_plugins(plugins_dir: &Path) -> Vec<PluginManifest> {
    let Ok(entries) = fs::read_dir(plugins_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .filter_map(|e| PluginManifest::load(&e.path().join("plugin.toml")).ok())
        .collect()
}
