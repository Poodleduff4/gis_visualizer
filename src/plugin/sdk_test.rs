//! Wire-format check between the Rust host code and the Python SDK
//! (`sdk/gis_editor_sdk`): a real `python3` subprocess using
//! `Host.get_layer`/`Host.add_layer` talks to a minimal stand-in "host"
//! loop here that answers using `bridge::encode_vector_layer` /
//! `decode_vector_layer` directly — no `GisEditorApp` needed, since that
//! requires a live `eframe::CreationContext` a plain unit test can't build.
#![cfg(test)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use geo_types::{Geometry, Point};

use super::bridge::{decode_vector_layer, encode_points_layer, encode_vector_layer};
use super::manifest::PluginManifest;
use super::process::PluginProcess;
use super::protocol::{HostReply, HostRequest, LayerSummary, PluginCall};
use crate::gis_layer::{AttributeValue, GisFeature, GisLayer};
use crate::point_cloud_layer::PointCloudLayer;

fn sample_layer() -> GisLayer {
    let mut attrs = HashMap::new();
    attrs.insert("label".to_string(), AttributeValue::Text("original".into()));
    attrs.insert("count".to_string(), AttributeValue::Integer(42));
    GisLayer {
        name: "sample".into(),
        file_path: String::new(),
        features: vec![
            GisFeature::new(0, Geometry::Point(Point::new(10.0, 20.0)), attrs),
            GisFeature::new(1, Geometry::Point(Point::new(30.0, 40.0)), HashMap::new()),
        ],
        field_names: vec!["label".into(), "count".into()],
        extra_field_names: Vec::new(),
        quadtree: None,
        point_only: true,
        world_bbox: [10.0, 20.0, 30.0, 40.0],
    }
}

#[test]
fn sdk_round_trips_get_and_add_layer_through_a_real_python_process() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // SAFETY: this test doesn't spawn threads that also touch the
    // environment, and Command inherits the process env at spawn time.
    unsafe {
        std::env::set_var("PYTHONPATH", manifest_dir.join("sdk"));
    }

    let manifest = PluginManifest {
        name: "sdk-roundtrip".into(),
        entrypoint: "sdk_roundtrip_plugin.py".into(),
        python: "python3".into(),
        capabilities: Vec::new(),
        params: Vec::new(),
        dir: manifest_dir.join("tests/fixtures"),
    };

    let mut proc = PluginProcess::spawn(&manifest).expect("spawn sdk plugin");
    proc.send(&HostRequest::Run).expect("send Run");

    let layer = sample_layer();
    let mut added_layer: Option<GisLayer> = None;

    loop {
        match proc.recv_call().expect("recv_call") {
            Some(PluginCall::GetLayer { layer_id, .. }) => {
                assert_eq!(layer_id, 0);
                let arrow_ipc = encode_vector_layer(&layer).expect("encode");
                proc.send(&HostRequest::Reply(HostReply::LayerData { arrow_ipc }))
                    .expect("reply GetLayer");
            }
            Some(PluginCall::AddLayer { name, arrow_ipc }) => {
                assert_eq!(name, "copy");
                added_layer = Some(decode_vector_layer(&arrow_ipc, name).expect("decode"));
                proc.send(&HostRequest::Reply(HostReply::Ack))
                    .expect("reply AddLayer");
            }
            Some(PluginCall::Log { msg, .. }) => {
                println!("plugin log: {msg}");
            }
            Some(PluginCall::Done { .. }) | None => break,
            Some(PluginCall::Error { msg }) => panic!("plugin reported error: {msg}"),
            Some(other) => panic!("unexpected call: {other:?}"),
        }
    }

    let decoded = added_layer.expect("plugin never called add_layer");
    assert_eq!(decoded.features.len(), 2);
    assert_eq!(
        decoded.features[0].attributes.get("label"),
        Some(&AttributeValue::Text("original".into()))
    );
    assert_eq!(
        decoded.features[0].attributes.get("count"),
        Some(&AttributeValue::Integer(42))
    );
    assert!(!decoded.features[1].attributes.contains_key("label"));

    proc.shutdown(Duration::from_millis(500)).ok();
}

/// Runs the actual shipped `plugins/buffer-tool` plugin with a user-chosen
/// `distance` sent via `Init { plugin_args }` — the same path the Plugins
/// window's param widgets feed into `plugin::run_plugin`. Proves a
/// non-default param value actually reaches the plugin, not just that the
/// wire format round-trips: a point buffered by 3.0 must have a much
/// smaller bbox than the same point buffered by 30.0.
#[test]
fn buffer_tool_plugin_receives_its_distance_param_via_init() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    unsafe {
        std::env::set_var("PYTHONPATH", manifest_dir.join("sdk"));
    }

    let manifest = PluginManifest {
        name: "buffer-tool".into(),
        entrypoint: "main.py".into(),
        python: "python3".into(),
        capabilities: Vec::new(),
        params: Vec::new(),
        dir: manifest_dir.join("plugins/buffer-tool"),
    };

    let source = GisLayer {
        name: "points-of-interest".into(),
        file_path: String::new(),
        features: vec![GisFeature::new(
            0,
            Geometry::Point(Point::new(0.0, 0.0)),
            HashMap::new(),
        )],
        field_names: Vec::new(),
        extra_field_names: Vec::new(),
        quadtree: None,
        point_only: true,
        world_bbox: [0.0, 0.0, 0.0, 0.0],
    };

    let mut proc = PluginProcess::spawn(&manifest).expect("spawn buffer-tool plugin");
    let plugin_args = serde_json::json!({ "distance": 3.0 });
    proc.send(&HostRequest::Init { plugin_args }).expect("send Init");
    proc.send(&HostRequest::Run).expect("send Run");

    let mut added_layer: Option<GisLayer> = None;

    loop {
        match proc.recv_call().expect("recv_call") {
            Some(PluginCall::ListLayers) => {
                let summary = LayerSummary {
                    id: 0,
                    name: source.name.clone(),
                    kind: "vector".into(),
                    feature_count: source.features.len(),
                    crs: None,
                };
                proc.send(&HostRequest::Reply(HostReply::Layers(vec![summary])))
                    .expect("reply ListLayers");
            }
            Some(PluginCall::GetLayer { layer_id, .. }) => {
                assert_eq!(layer_id, 0);
                let arrow_ipc = encode_vector_layer(&source).expect("encode");
                proc.send(&HostRequest::Reply(HostReply::LayerData { arrow_ipc }))
                    .expect("reply GetLayer");
            }
            Some(PluginCall::AddLayer { name, arrow_ipc }) => {
                added_layer = Some(decode_vector_layer(&arrow_ipc, name).expect("decode"));
                proc.send(&HostRequest::Reply(HostReply::Ack))
                    .expect("reply AddLayer");
            }
            Some(PluginCall::Log { msg, .. }) => println!("plugin log: {msg}"),
            Some(PluginCall::Progress { pct, msg }) => println!("plugin progress: {pct} {msg}"),
            Some(PluginCall::Done { .. }) | None => break,
            Some(PluginCall::Error { msg }) => panic!("plugin reported error: {msg}"),
            Some(other) => panic!("unexpected call: {other:?}"),
        }
    }

    let decoded = added_layer.expect("plugin never called add_layer");
    let bbox = decoded.features[0].bbox();
    // A circle of radius 3 centered on the origin: bbox half-width should
    // land close to 3.0, nowhere near what a default (100.0) or some other
    // stray value would produce.
    let half_width = (bbox[2] - bbox[0]) / 2.0;
    assert!(
        (half_width - 3.0).abs() < 0.1,
        "expected a ~3.0 buffer radius, got bbox {bbox:?} (half-width {half_width})"
    );

    proc.shutdown(Duration::from_millis(500)).ok();
}

fn sample_points_layer() -> PointCloudLayer {
    // A square with one interior point: the convex hull must be exactly
    // the square, dropping the interior point.
    PointCloudLayer {
        points: Arc::new(vec![
            (0, [0.0, 0.0]),
            (1, [0.0, 10.0]),
            (2, [10.0, 10.0]),
            (3, [10.0, 0.0]),
            (4, [5.0, 5.0]),
        ]),
        attributes: Vec::new(),
        field_names: Vec::new(),
        index: None,
        bbox: None,
        viewport_mask: Default::default(),
        filter_mask: Default::default(),
    }
}

/// Runs the actual shipped `plugins/point-convex-hull` plugin (not a test
/// fixture) against a stand-in host, proving a Points layer round-trips
/// through `Host.get_layer` (flat id/x/y columns, no WKB — see
/// `bridge::encode_points_layer`) and that its polygon result decodes back
/// correctly through the ordinary Vector/WKB path.
#[test]
fn point_convex_hull_plugin_reads_a_points_layer_and_writes_a_polygon() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    unsafe {
        std::env::set_var("PYTHONPATH", manifest_dir.join("sdk"));
    }

    let manifest = PluginManifest {
        name: "point-convex-hull".into(),
        entrypoint: "main.py".into(),
        python: "python3".into(),
        capabilities: Vec::new(),
        params: Vec::new(),
        dir: manifest_dir.join("plugins/point-convex-hull"),
    };

    let mut proc = PluginProcess::spawn(&manifest).expect("spawn point-convex-hull plugin");
    proc.send(&HostRequest::Run).expect("send Run");

    let points = sample_points_layer();
    let mut added_layer: Option<GisLayer> = None;

    loop {
        match proc.recv_call().expect("recv_call") {
            Some(PluginCall::ListLayers) => {
                let summary = LayerSummary {
                    id: 0,
                    name: "elevation".into(),
                    kind: "points".into(),
                    feature_count: points.points.len(),
                    crs: None,
                };
                proc.send(&HostRequest::Reply(HostReply::Layers(vec![summary])))
                    .expect("reply ListLayers");
            }
            Some(PluginCall::GetLayer { layer_id, .. }) => {
                assert_eq!(layer_id, 0);
                let arrow_ipc = encode_points_layer(&points).expect("encode");
                proc.send(&HostRequest::Reply(HostReply::LayerData { arrow_ipc }))
                    .expect("reply GetLayer");
            }
            Some(PluginCall::AddLayer { name, arrow_ipc }) => {
                assert_eq!(name, "elevation (convex hull)");
                added_layer = Some(decode_vector_layer(&arrow_ipc, name).expect("decode"));
                proc.send(&HostRequest::Reply(HostReply::Ack))
                    .expect("reply AddLayer");
            }
            Some(PluginCall::Log { msg, .. }) => println!("plugin log: {msg}"),
            Some(PluginCall::Progress { pct, msg }) => println!("plugin progress: {pct} {msg}"),
            Some(PluginCall::Done { .. }) | None => break,
            Some(PluginCall::Error { msg }) => panic!("plugin reported error: {msg}"),
            Some(other) => panic!("unexpected call: {other:?}"),
        }
    }

    let decoded = added_layer.expect("plugin never called add_layer");
    assert_eq!(decoded.features.len(), 1);
    assert_eq!(
        decoded.features[0].attributes.get("source_layer"),
        Some(&AttributeValue::Text("elevation".into()))
    );
    match &decoded.features[0].geometry {
        geo_types::Geometry::Polygon(p) => {
            let bbox = decoded.features[0].bbox();
            assert_eq!(bbox, [0.0, 0.0, 10.0, 10.0]);
            // The interior point must not have pulled a vertex inward.
            assert!(p.exterior().coords().all(|c| {
                (c.x == 0.0 || c.x == 10.0) && (c.y == 0.0 || c.y == 10.0)
            }));
        }
        other => panic!("expected Polygon, got {other:?}"),
    }

    proc.shutdown(Duration::from_millis(500)).ok();
}
