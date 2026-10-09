//! Plots of side-channel leakage evaluation results (t-values, χ² values).
//!
//! The module has four parts.
//!
//! 1. [`envelope`] reduces a long trace (up to about 10^6 samples) to a few thousand
//!    min/max pairs. A plot of the envelope keeps every spike.
//! 2. [`line_figure`], [`html_page`], [`write_html`], and [`write_json`] build
//!    interactive [plotly](https://plotly.com/javascript/) figures. The HTML file
//!    contains the plotly.js library, so it works offline.
//! 3. [`save_line_plot`] draws static SVG and PNG images with the pure-Rust
//!    [plotters](https://github.com/plotters-rs/plotters) crate.
//! 4. [`plot_t_traces`], [`plot_max_t_values`], and [`plot_series`] write the files that
//!    the `tvla` and `plot` programs produce: HTML, SVG, and JSON.
//!
//! # No browser, no network, no system fonts
//!
//! Static images are drawn by plotters. They need no browser, no chromedriver, and no
//! system font. The font (DejaVu Sans) is embedded in the binary. See
//! `fonts/DejaVu-LICENSE.txt` for the license of the font.
//!
//! The `plotly` dependency must **not** enable the features `static_export_default`,
//! `static_export_chromedriver`, `static_export_geckodriver`, `kaleido`, or
//! `plotly_image`. Use only `plotly_embed_js`. The features for static export start a
//! browser driver and download a driver binary. A test in this module checks this.
//!
//! # Example
//!
//! ```no_run
//! use scasim::plot::*;
//!
//! # fn main() -> Result<(), PlotError> {
//! let t_values: Vec<f64> = vec![0.0; 1_000_000]; // one trace of t-values
//! let series = [Series::indexed("d=1", &t_values)];
//! let options = LineOptions {
//!     thresholds: vec![Threshold::new(4.5, "±4.5"), Threshold::new(6.0, "adjusted ±6.0")],
//!     ..LineOptions::t_values()
//! };
//!
//! // Interactive: one HTML file, plotly.js embedded.
//! let figure = line_figure(&series, &options)?;
//! write_html("t_test_d1.html", "t-test d=1", &[&figure], JsSource::Embedded)?;
//! write_json("t_test_d1.json", &figure)?;
//!
//! // Static: the file extension selects SVG or PNG.
//! save_line_plot("t_test_d1.svg", &series, &options, (1200, 600))?;
//!
//! // All three files at once: `chi2.html`, `chi2.svg`, and `chi2.json`.
//! plot_series("chi2", &series, &options, std::path::Path::new("."), false)?;
//! # Ok(())
//! # }
//! ```

mod envelope;
mod html;
mod static_plots;
mod types;

#[cfg(test)]
mod test_data;
#[cfg(test)]
mod tests;

use std::path::Path;

use log::info;
use ndarray::ArrayView2;

pub use envelope::{bucket_bounds, envelope};
pub use html::{JsSource, configuration, html_page, line_figure, write_html, write_json};
pub use static_plots::save_line_plot;
pub use types::{LineOptions, PlotError, Series, Threshold};

/// The size of the SVG files of [`plot_series`], [`plot_t_traces`], and
/// [`plot_max_t_values`], in pixels.
const SVG_SIZE: (u32, u32) = (1200, 600);

/// The threshold list for an optional threshold. With `abs_values`, only `+t` is drawn, so
/// the label has no `±`.
fn single_threshold(t: Option<f64>, abs_values: bool) -> Vec<Threshold> {
    t.map(|t| {
        let label = if abs_values {
            format!("{t}")
        } else {
            format!("±{t}")
        };
        Threshold::new(t, label)
    })
    .into_iter()
    .collect()
}

/// Checks that `name_stem` is a plain file name stem without a directory part.
fn check_stem(name_stem: &str) -> Result<(), PlotError> {
    if name_stem.is_empty() || name_stem.contains(['/', '\\']) {
        return Err(PlotError::Input(format!(
            "invalid file name stem {name_stem:?}: it must not be empty or contain a path separator"
        )));
    }
    Ok(())
}

/// Plots the series in one figure and writes `{name_stem}.html`, `{name_stem}.svg`, and
/// `{name_stem}.json` to `output_dir`.
///
/// This is the general function for line plots. The options decide what the plot shows. For
/// example:
///
/// * max |t| or max χ² versus the number of traces: [`Series::with_x`] with the numbers of
///   traces as `x`, and the options [`LineOptions::max_t`] or a copy with another
///   `y_label`;
/// * −log10(p) per sample: [`Series::indexed`], `y_label` `"-log10(p)"`, `symmetric: false`,
///   and the thresholds 5 and the Bonferroni value (a list of [`Threshold`] values).
///
/// The HTML file has the plotly.js library and works offline. The SVG file is drawn by
/// plotters. A long series without x values is reduced with an envelope. See
/// [`line_figure`]. If `show` is `true`, the figure is also shown in the web browser.
///
/// The directory `output_dir` must exist.
///
/// # Errors
///
/// Returns [`PlotError::Input`] for invalid data or options (see [`line_figure`]) and for a
/// `name_stem` that is empty or has a path separator. Returns [`PlotError::Io`] or
/// [`PlotError::Draw`] if a file cannot be written.
pub fn plot_series(
    name_stem: &str,
    series: &[Series<'_>],
    opts: &LineOptions,
    output_dir: &Path,
    show: bool,
) -> Result<(), PlotError> {
    check_stem(name_stem)?;
    let figure = line_figure(series, opts)?;
    let path = |ext: &str| output_dir.join(format!("{name_stem}.{ext}"));
    info!("Writing {}", path("html").display());
    write_html(path("html"), name_stem, &[&figure], JsSource::Embedded)?;
    info!("Writing {}", path("svg").display());
    save_line_plot(path("svg"), series, opts, SVG_SIZE)?;
    info!("Writing {}", path("json").display());
    write_json(path("json"), &figure)?;
    if show {
        figure.show();
    }
    Ok(())
}

/// Plots the t-values and writes the files to `output_dir`.
///
/// `t_values` has one row per t-test order: row `k - 1` has the t-values of order `k`.
/// For each order `d`, the function writes `t_test_d{d}.html`, `t_test_d{d}.svg`, and
/// `t_test_d{d}.json` (see [`plot_series`]). It also writes `all_t_values.html` with all
/// orders in one figure.
///
/// With `t_threshold = Some(t)`, the plots have threshold lines at `t` (and at `-t`, if
/// `abs_values` is `false`) and a fixed y range, also if all t-values are zero or positive.
/// With `None`,
/// the y axis scales automatically. With `abs_values`, the plots show `|t|`. NaN and
/// infinite t-values are plotted as 0. The x axis is the sample index.
///
/// If `show_plots` is `true`, the figures are also shown in the web browser.
///
/// # Errors
///
/// Returns [`PlotError::Input`] if `t_values` has no row or no column, or if the threshold
/// is not finite and positive. Returns [`PlotError::Io`] or [`PlotError::Draw`] if a file
/// cannot be written.
pub fn plot_t_traces(
    t_values: ArrayView2<'_, f64>,
    t_threshold: Option<f64>,
    abs_values: bool,
    output_dir: &Path,
    show_plots: bool,
) -> Result<(), PlotError> {
    let (orders, samples) = t_values.dim();
    if orders == 0 || samples == 0 {
        return Err(PlotError::Input(format!(
            "the t-values have {orders} orders and {samples} samples: both must be at least 1"
        )));
    }
    let opts = LineOptions {
        y_label: if abs_values { "|t|" } else { "t-value" }.into(),
        thresholds: single_threshold(t_threshold, abs_values),
        abs_values,
        ..LineOptions::t_values()
    };
    let rows: Vec<Vec<f64>> = t_values.rows().into_iter().map(|r| r.to_vec()).collect();
    for (i, row) in rows.iter().enumerate() {
        let d = i + 1;
        info!("Plotting t-values for d={d}");
        let series = [Series::indexed(format!("d={d}"), row)];
        plot_series(
            &format!("t_test_d{d}"),
            &series,
            &opts,
            output_dir,
            show_plots,
        )?;
    }
    let all: Vec<Series<'_>> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| Series::indexed(format!("d={}", i + 1), row))
        .collect();
    let figure = line_figure(&all, &opts)?;
    let path = output_dir.join("all_t_values.html");
    info!("Writing {}", path.display());
    write_html(&path, "t-test, all orders", &[&figure], JsSource::Embedded)?;
    if show_plots {
        figure.show();
    }
    Ok(())
}

/// The largest finite value, or `None` if there is none.
fn max_finite(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(None, |m, v| Some(m.map_or(v, |m: f64| m.max(v))))
}

/// Plots the maximum of |t| versus the number of traces and writes `max_t_values.html`,
/// `max_t_values.svg`, and `max_t_values.json` to `output_dir`.
///
/// `max_t_values[k - 1]` is the series for order `k`. Every series has one value for each
/// entry of `num_traces_so_far`. The function also prints the largest finite value of each
/// series to the standard output. A value that is NaN or infinite is plotted as a gap and
/// is ignored in the printed maximum.
///
/// # Errors
///
/// Returns [`PlotError::Input`] if there is no series, a series is empty or has another
/// length than `num_traces_so_far`, or the threshold is not finite and positive. Returns
/// [`PlotError::Io`] or [`PlotError::Draw`] if a file cannot be written.
pub fn plot_max_t_values(
    max_t_values: &[Vec<f64>],
    num_traces_so_far: &[usize],
    t_threshold: Option<f64>,
    output_dir: &Path,
    show_plots: bool,
) -> Result<(), PlotError> {
    if let Some((i, v)) = max_t_values
        .iter()
        .enumerate()
        .find(|(_, v)| v.len() != num_traces_so_far.len())
    {
        return Err(PlotError::Input(format!(
            "the max |t| series of order {} has {} values, but there are {} trace counts",
            i + 1,
            v.len(),
            num_traces_so_far.len()
        )));
    }
    let opts = LineOptions {
        thresholds: single_threshold(t_threshold, true),
        non_finite_as_zero: false,
        y_label: "max(|t|), descriptive repeated looks".into(),
        ..LineOptions::max_t()
    };
    let x: Vec<f64> = num_traces_so_far.iter().map(|&n| n as f64).collect();
    let series: Vec<Series<'_>> = max_t_values
        .iter()
        .enumerate()
        .map(|(i, v)| Series::with_x(format!("d={}", i + 1), &x, v))
        .collect();
    for (i, values) in max_t_values.iter().enumerate() {
        match max_finite(values) {
            Some(m) => println!("Max t-value for d={}: {m:.03}", i + 1),
            None => println!("Max t-value for d={}: no finite value", i + 1),
        }
    }
    plot_series("max_t_values", &series, &opts, output_dir, show_plots)
}
