//! Shared visual styling for egui_plot charts so histograms/scatter plots
//! look consistent and less like the raw egui_plot default.

use egui::{Color32, CornerRadius, Frame, Stroke};

pub const ACCENT: Color32 = Color32::from_rgb(90, 169, 230);
pub const ACCENT_FILL: Color32 = Color32::from_rgba_premultiplied(90, 169, 230, 130);
pub const GOOD: Color32 = Color32::from_rgb(110, 200, 140);
pub const BAD: Color32 = Color32::from_rgb(230, 110, 110);
pub const TREND: Color32 = Color32::from_rgb(240, 175, 60);
pub const MEAN: Color32 = Color32::from_rgb(190, 190, 200);

/// Card-like frame to host a plot: rounded corners, subtle border, slightly
/// recessed background so it reads as a distinct panel instead of a bare box.
pub fn card<R>(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let visuals = ui.visuals().clone();
    Frame::default()
        .fill(visuals.extreme_bg_color)
        .stroke(Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(8.0)
        .show(ui, add_contents)
        .inner
}

/// Apply consistent chrome to a Plot: no double background box (the card
/// frame already provides one), faded grid, matching cursor color, and
/// axis/hover labels formatted with thousands separators instead of raw
/// floats.
pub fn style(plot: egui_plot::Plot<'_>) -> egui_plot::Plot<'_> {
    plot.show_background(false)
        .show_axes([true, true])
        .grid_fade(0.35)
        .cursor_color(ACCENT)
        .x_axis_formatter(|mark, _range| format_number(mark.value))
        .y_axis_formatter(|mark, _range| format_number(mark.value))
        .label_formatter(|name, value| {
            let coords = format!("x: {}\ny: {}", format_number(value.x), format_number(value.y));
            if name.is_empty() {
                coords
            } else {
                format!("{name}\n{coords}")
            }
        })
}

/// Format a value for axis ticks / hover labels: thousands separators for
/// large magnitudes, fewer decimals as magnitude grows, so numbers read like
/// a finished report instead of raw `f64` debug output.
pub fn format_number(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if a >= 1000.0 {
        group_thousands(v.round() as i64)
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

fn group_thousands(v: i64) -> String {
    let neg = v < 0;
    let digits = v.unsigned_abs().to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    let grouped: String = grouped.chars().rev().collect();
    if neg {
        format!("-{grouped}")
    } else {
        grouped
    }
}

/// Ordinary-least-squares slope/intercept from bivariate summary stats
/// (all three `Bivariate`/`BivariateStats`/`SelectionBivariate` structs carry
/// the same `x_mean`/`y_mean`/`covariance`/`x_std` fields). `None` when x has
/// ~zero variance (fit is undefined / a vertical line).
pub fn linear_fit(x_mean: f64, y_mean: f64, covariance: f64, x_std: f64) -> Option<(f64, f64)> {
    if x_std < 1e-9 {
        return None;
    }
    let slope = covariance / (x_std * x_std);
    let intercept = y_mean - slope * x_mean;
    Some((slope, intercept))
}

pub fn bar_color(counts_max: u32, count: u32) -> Color32 {
    // Slightly brighten taller bars for a subtle gradient feel.
    let t = if counts_max == 0 {
        0.0
    } else {
        count as f32 / counts_max as f32
    };
    let base = ACCENT;
    Color32::from_rgb(
        (base.r() as f32 + (255.0 - base.r() as f32) * t * 0.15) as u8,
        (base.g() as f32 + (255.0 - base.g() as f32) * t * 0.15) as u8,
        base.b(),
    )
}
