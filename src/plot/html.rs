//! Interactive figures with plotly.js (HTML and JSON).
//!
//! Only the `plotly_embed_js` feature of the `plotly` crate is used. Nothing here starts
//! a browser or a driver, and nothing uses the network when the file is written.

use std::path::Path;

use plotly::{
    Configuration, Layout, Plot, Scatter,
    color::{Rgb, Rgba},
    common::{DashType, Fill, HoverInfo, Line, Mode, Title},
    configuration::DisplayModeBar,
    layout::{Axis, HoverMode},
};

use super::types::{
    LineOptions, PALETTE, PlotError, Prepared, Series, layout, prepare, threshold_color,
};

/// The version of plotly.js that the `plotly` crate embeds. The CDN link uses it, too.
const PLOTLY_JS_VERSION: &str = "3.0.1";

/// Where an HTML page gets the plotly.js library from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsSource {
    /// Put the library (about 4.7 MB) into the file. The page works offline.
    Embedded,
    /// Load the library from the plotly CDN when the page is opened. The file is small,
    /// but the viewer needs network access. The library itself never uses the network.
    Cdn,
}

/// The plotly configuration used for all figures: the mode bar appears on hover, no
/// logo, no link to plotly, responsive size, and zoom with the mouse wheel.
///
/// Math typesetting is off, because the page does not include MathJax.
pub fn configuration() -> Configuration {
    Configuration::new()
        .display_mode_bar(DisplayModeBar::Hover)
        .show_link(false)
        .display_logo(false)
        .editable(false)
        .responsive(true)
        .scroll_zoom(true)
        .typeset_math(false)
}

fn round(v: f64, decimals: Option<u32>) -> f64 {
    match decimals {
        // Adding 0.0 turns -0.0 into 0.0. NaN stays NaN.
        Some(d) => {
            let m = 10f64.powi(d as i32);
            (v * m).round() / m + 0.0
        }
        None => v,
    }
}

fn rounded(v: &[f64], decimals: Option<u32>) -> Vec<f64> {
    v.iter().map(|&v| round(v, decimals)).collect()
}

/// Adds the traces of one prepared series. A series with an envelope becomes two traces:
/// the maximum line, then the minimum line with a fill to the maximum line.
fn add_series(plot: &mut Plot, i: usize, p: &Prepared, opts: &LineOptions) {
    let (r, g, b) = PALETTE[i % PALETTE.len()];
    let x = rounded(&p.x, opts.decimals);
    let lo = rounded(&p.lo, opts.decimals);
    match &p.hi {
        None => {
            let line = Line::new()
                .width(opts.line_width)
                .color(Rgb::new(r, g, b))
                .simplify(false);
            plot.add_trace(
                Scatter::new(x, lo)
                    .mode(Mode::Lines)
                    .name(&p.name)
                    .line(line),
            );
        }
        Some(hi) => {
            let hi = rounded(hi, opts.decimals);
            let thin = || {
                Line::new()
                    .width(1.0)
                    .color(Rgb::new(r, g, b))
                    .simplify(false)
            };
            plot.add_trace(
                Scatter::new(x.clone(), hi)
                    .mode(Mode::Lines)
                    .name(&p.name)
                    .legend_group(&p.name)
                    .line(thin())
                    .hover_template(format!("%{{y:.3f}}<extra>{} max</extra>", p.name)),
            );
            plot.add_trace(
                Scatter::new(x, lo)
                    .mode(Mode::Lines)
                    .name(format!("{} min", p.name))
                    .legend_group(&p.name)
                    .show_legend(false)
                    .line(thin())
                    .fill(Fill::ToNextY)
                    .fill_color(Rgba::new(r, g, b, 0.35))
                    .hover_template(format!("%{{y:.3f}}<extra>{} min</extra>", p.name)),
            );
        }
    }
}

/// Builds an interactive line figure.
///
/// One trace (or one band, see below) is drawn per series. Each threshold in
/// `opts.thresholds` becomes a horizontal line at `+value`, and at `-value` if the plot
/// shows negative values. Threshold lines are in the legend and can be switched off.
/// Hover shows the values at the same x for all series.
///
/// A series without x values that has more than `opts.buckets` samples is reduced with
/// [`envelope`](super::envelope): the figure then has two traces for the series, the
/// maximum line and the minimum line with a filled band between them. The envelope is
/// fixed when the figure is built. If you zoom into a few hundred samples of a trace
/// with 10^6 samples, you still see the buckets, not single samples. For that, plot a
/// slice of the trace.
///
/// # Errors
///
/// Returns [`PlotError::Input`] if there is no series, a series is empty, x and y have
/// different lengths, or `opts.buckets` is 0.
pub fn line_figure(series: &[Series<'_>], opts: &LineOptions) -> Result<Plot, PlotError> {
    let prepared = prepare(series, opts)?;
    let lay = layout(&prepared, opts);
    let mut plot = Plot::new();
    plot.set_configuration(configuration());

    for (i, p) in prepared.iter().enumerate() {
        add_series(&mut plot, i, p, opts);
    }

    // Threshold lines as two-point traces. They span the full x range at any zoom level
    // that starts from the default view, and they appear in the legend.
    let (x0, x1) = lay.x_range;
    for (i, t) in opts.thresholds.iter().enumerate() {
        let (r, g, b) = threshold_color(i);
        let dash = match i {
            0 => DashType::Dot,
            1 => DashType::Dash,
            _ => DashType::DashDot,
        };
        let group = format!("threshold-{i}");
        let signs: &[f64] = if lay.negative_lines {
            &[1.0, -1.0]
        } else {
            &[1.0]
        };
        for (k, sign) in signs.iter().enumerate() {
            let y = sign * t.value;
            let trace = Scatter::new(vec![x0, x1], vec![y, y])
                .mode(Mode::Lines)
                .name(&t.label)
                .legend_group(&group)
                .show_legend(k == 0)
                .hover_info(HoverInfo::Skip)
                .line(
                    Line::new()
                        .width(1.0)
                        .dash(dash.clone())
                        .color(Rgb::new(r, g, b)),
                );
            plot.add_trace(trace);
        }
    }

    let mut y_axis = Axis::new().title(Title::with_text(&opts.y_label));
    y_axis = match lay.y_range {
        Some((lo, hi)) => y_axis.range(vec![lo, hi]).auto_range(false),
        None => y_axis.auto_range(true),
    };
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::with_text(&opts.x_label)))
            .y_axis(y_axis)
            .hover_mode(HoverMode::X)
            .show_legend(true),
    );
    Ok(plot)
}

/// Returns the `<script>` element with the plotly.js library from the `plotly` crate.
///
/// `Plot::offline_js_sources()` returns plotly.js (4.7 MB) *and* MathJax (2.1 MB).
/// None of these plots use math, so only the first script element is used.
fn embedded_plotly_script() -> String {
    let both = Plot::offline_js_sources();
    const END: &str = "</script>";
    // If the library text changes and has no end tag, use all of it. This cannot panic.
    let end = both.find(END).map_or(both.len(), |i| i + END.len());
    both[..end].to_string()
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Renders figures to a standalone HTML page. All figures share one copy of plotly.js.
///
/// Unlike `Plot::to_html`, this does not embed MathJax. The stock writer produces files
/// of at least 6.8 MB, even for a plot with no data. This page is about 4.7 MB plus the
/// data with [`JsSource::Embedded`], and a few hundred kilobytes with [`JsSource::Cdn`].
pub fn html_page(title: &str, plots: &[&Plot], js: JsSource) -> String {
    let script = match js {
        JsSource::Embedded => embedded_plotly_script(),
        JsSource::Cdn => format!(
            r#"<script src="https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"></script>"#
        ),
    };
    let mut page = String::with_capacity(script.len() + 4096);
    page.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\" />\n");
    page.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n");
    page.push_str(&format!("<title>{}</title>\n", escape_html(title)));
    page.push_str(
        "<style>body{margin:0}.figure{height:min(75vh,760px);min-height:360px;margin:8px}</style>\n",
    );
    page.push_str(&script);
    page.push_str("\n</head>\n<body>\n");
    for (i, plot) in plots.iter().enumerate() {
        page.push_str("<div class=\"figure\">\n");
        page.push_str(&plot.to_inline_html(Some(&format!("figure-{i}"))));
        page.push_str("\n</div>\n");
    }
    page.push_str("</body>\n</html>\n");
    page
}

/// Writes [`html_page`] to `path`.
pub fn write_html(
    path: impl AsRef<Path>,
    title: &str,
    plots: &[&Plot],
    js: JsSource,
) -> Result<(), PlotError> {
    std::fs::write(path, html_page(title, plots, js))?;
    Ok(())
}

/// Writes the JSON of a figure (`data`, `layout`, and `config`) to `path`. The JSON can
/// be passed to `Plotly.newPlot` in any page.
pub fn write_json(path: impl AsRef<Path>, plot: &Plot) -> Result<(), PlotError> {
    std::fs::write(path, plot.to_json())?;
    Ok(())
}
