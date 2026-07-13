//! Answers the `PluginCall`s a running plugin makes that need real layer
//! data — this is the only place plugin code touches `GisEditorApp`, kept
//! separate from `plugin::bridge`'s pure Arrow codec so that module stays
//! app-agnostic.

use crate::gis_layer::LayerKind;
use crate::plugin::{self, HostReply, LayerSummary, PluginCall};

use super::GisEditorApp;

impl GisEditorApp {
    pub(super) fn handle_plugin_call(&mut self, call: PluginCall) -> HostReply {
        match call {
            PluginCall::ListLayers => HostReply::Layers(
                self.layers
                    .iter()
                    .enumerate()
                    .map(|(id, entry)| LayerSummary {
                        id: id as u32,
                        name: entry.name.clone(),
                        kind: match &entry.data {
                            LayerKind::Vector(_) => "vector",
                            LayerKind::Points(_) => "points",
                            LayerKind::Raster(_) => "raster",
                        }
                        .to_string(),
                        feature_count: entry.data.feature_count(),
                        crs: None,
                    })
                    .collect(),
            ),

            PluginCall::GetLayer { layer_id, .. } => {
                match self.layers.get(layer_id as usize).map(|e| &e.data) {
                    Some(LayerKind::Vector(gl)) => match plugin::bridge::encode_vector_layer(gl) {
                        Ok(arrow_ipc) => HostReply::LayerData { arrow_ipc },
                        Err(e) => HostReply::Err(e.to_string()),
                    },
                    Some(LayerKind::Points(pc)) => match plugin::bridge::encode_points_layer(pc) {
                        Ok(arrow_ipc) => HostReply::LayerData { arrow_ipc },
                        Err(e) => HostReply::Err(e.to_string()),
                    },
                    Some(LayerKind::Raster(_)) => {
                        HostReply::Err("Raster layers aren't exposed to plugins yet".into())
                    }
                    None => HostReply::Err(format!("no layer with id {layer_id}")),
                }
            }

            PluginCall::AddLayer { name, arrow_ipc } => {
                match plugin::bridge::decode_vector_layer(&arrow_ipc, name.clone()) {
                    Ok(gl) => {
                        self.layers.push(plugin::bridge::layer_entry_for(name, gl));
                        self.points_dirty = true;
                        self.globe_points_dirty = true;
                        self.map_render_ttl = 3;
                        HostReply::Ack
                    }
                    Err(e) => HostReply::Err(e.to_string()),
                }
            }

            PluginCall::UpdateLayer { layer_id, arrow_ipc } => {
                let name = self
                    .layers
                    .get(layer_id as usize)
                    .map(|e| e.name.clone())
                    .unwrap_or_else(|| format!("layer-{layer_id}"));
                match plugin::bridge::decode_vector_layer(&arrow_ipc, name) {
                    Ok(gl) => match self.layers.get_mut(layer_id as usize) {
                        Some(entry) => {
                            entry.data = LayerKind::Vector(gl);
                            self.points_dirty = true;
                            self.globe_points_dirty = true;
                            self.map_render_ttl = 3;
                            HostReply::Ack
                        }
                        None => HostReply::Err(format!("no layer with id {layer_id}")),
                    },
                    Err(e) => HostReply::Err(e.to_string()),
                }
            }

            // Log/Progress/Done/Error never reach here — the runner thread
            // handles them directly as PluginEvents, not LayerRequests.
            other => HostReply::Err(format!("unexpected call on the layer-request path: {other:?}")),
        }
    }
}
