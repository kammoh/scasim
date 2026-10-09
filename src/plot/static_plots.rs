//! Static SVG and PNG images with plotters.
//!
//! The text is drawn with the `ab_glyph` feature of plotters. That is a pure-Rust font
//! engine. It does not use FreeType, fontconfig, or any system font. One font (DejaVu
//! Sans) is embedded in the binary and registered under the family name `sans-serif`.

use std::ops::Range;
use std::path::Path;
use std::sync::OnceLock;

use plotters::coord::Shift;
use plotters::prelude::*;

use super::types::{
    LineOptions, PALETTE, PlotError, Prepared, Series, layout, prepare, threshold_color,
};

/// DejaVu Sans, license in `fonts/DejaVu-LICENSE.txt`.
const FONT_BYTES: &[u8] = include_bytes!("fonts/DejaVuSans.ttf");

/// The text and zero line color of plotly.js.
pub(crate) const TEXT: RGBColor = RGBColor(68, 68, 68);
/// The grid line color of plotly.js.
const GRID: RGBColor = RGBColor(238, 238, 238);

impl<E: std::error::Error + Send + Sync> From<DrawingAreaErrorKind<E>> for PlotError {
    fn from(e: DrawingAreaErrorKind<E>) -> Self {
        PlotError::Draw(e.to_string())
    }
}

/// Registers the embedded font once. Without it, every text draw call fails.
pub(crate) fn ensure_font() -> Result<(), PlotError> {
    static REGISTERED: OnceLock<bool> = OnceLock::new();
    let ok = *REGISTERED.get_or_init(|| {
        plotters::style::register_font("sans-serif", FontStyle::Normal, FONT_BYTES).is_ok()
    });
    if ok { Ok(()) } else { Err(PlotError::Font) }
}

/// Formats an axis tick: an integer if the value is one, otherwise up to three decimals.
pub(crate) fn fmt_tick(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Something that draws an image on any plotters back end.
pub(crate) trait Drawer {
    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Result<(), PlotError>;
}

/// Draws an image to `path`. The extension (`.svg` or `.png`, case-insensitive) selects
/// the back end.
pub(crate) fn render(path: &Path, size: (u32, u32), drawer: &impl Drawer) -> Result<(), PlotError> {
    ensure_font()?;
    if size.0 < 100 || size.1 < 100 {
        return Err(PlotError::Input(format!(
            "image size {}x{} is too small (minimum 100x100)",
            size.0, size.1
        )));
    }
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("svg") => {
            let root = SVGBackend::new(path, size).into_drawing_area();
            drawer.draw(&root)?;
            root.present()?;
        }
        Some("png") => {
            let root = BitMapBackend::new(path, size).into_drawing_area();
            drawer.draw(&root)?;
            root.present()?;
        }
        _ => return Err(PlotError::Extension(path.to_path_buf())),
    }
    Ok(())
}

/// Returns the index ranges of consecutive samples where `x`, `values`, and `also` (if given)
/// are finite. All slices have the same length.
fn finite_runs(x: &[f64], values: &[f64], also: Option<&[f64]>) -> Vec<Range<usize>> {
    let ok = |i: usize| {
        x[i].is_finite() && values[i].is_finite() && also.is_none_or(|a| a[i].is_finite())
    };
    let mut runs = Vec::new();
    let mut start = None;
    for i in 0..values.len() {
        match (ok(i), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                runs.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push(s..values.len());
    }
    runs
}

struct LineDrawer<'a> {
    prepared: Vec<Prepared>,
    options: &'a LineOptions,
}

impl Drawer for LineDrawer<'_> {
    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Result<(), PlotError> {
        let opts = self.options;
        let lay = layout(&self.prepared, opts);
        let (x0, x1) = lay.x_range;
        let (y0, y1) = lay.y_range.unwrap_or_else(|| {
            // Automatic range: the data range with 5 percent of padding.
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for p in &self.prepared {
                for &v in p.lo.iter().chain(p.hi.iter().flatten()) {
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            if lo >= hi {
                return (lo.min(0.0) - 0.5, hi.max(0.0) + 0.5);
            }
            let pad = 0.05 * (hi - lo);
            (lo - pad, hi + pad)
        });

        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(root)
            .margin(14)
            .x_label_area_size(50)
            .y_label_area_size(72)
            .build_cartesian_2d(x0..x1, y0..y1)?;
        let label_font = ("sans-serif", 14).into_font().color(&TEXT);
        let desc_font = ("sans-serif", 16).into_font().color(&TEXT);
        chart
            .configure_mesh()
            .x_desc(&opts.x_label)
            .y_desc(&opts.y_label)
            .x_labels(10)
            .y_labels(8)
            .x_label_formatter(&|v| fmt_tick(*v))
            .y_label_formatter(&|v| fmt_tick(*v))
            .label_style(label_font)
            .axis_desc_style(desc_font)
            .light_line_style(TRANSPARENT)
            .bold_line_style(GRID.stroke_width(1))
            .axis_style(TRANSPARENT)
            .draw()?;

        // The zero line, as in plotly.js.
        if y0 < 0.0 && y1 > 0.0 {
            chart.draw_series(LineSeries::new(
                vec![(x0, 0.0), (x1, 0.0)],
                TEXT.stroke_width(1),
            ))?;
        }

        // Data series.
        for (i, p) in self.prepared.iter().enumerate() {
            let (r, g, b) = PALETTE[i % PALETTE.len()];
            let color = RGBColor(r, g, b);
            let stroke = opts.line_width.round().max(1.0) as u32;
            let legend =
                move |(x, y)| PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(3));
            match &p.hi {
                Some(hi) => {
                    // Envelope: a filled band plus thin max and min lines.
                    let runs = finite_runs(&p.x, &p.lo, Some(hi));
                    for run in &runs {
                        let upper = run.clone().map(|k| (p.x[k], hi[k]));
                        let lower = run.clone().rev().map(|k| (p.x[k], p.lo[k]));
                        let outline: Vec<(f64, f64)> = upper.chain(lower).collect();
                        chart.draw_series(std::iter::once(Polygon::new(
                            outline,
                            color.mix(0.35).filled(),
                        )))?;
                    }
                    for (k, run) in runs.iter().enumerate() {
                        let top: Vec<_> = run.clone().map(|j| (p.x[j], hi[j])).collect();
                        let bottom: Vec<_> = run.clone().map(|j| (p.x[j], p.lo[j])).collect();
                        let annotation =
                            chart.draw_series(LineSeries::new(top, color.stroke_width(1)))?;
                        if k == 0 {
                            annotation.label(&p.name).legend(legend);
                        }
                        chart.draw_series(LineSeries::new(bottom, color.stroke_width(1)))?;
                    }
                }
                None => {
                    for (k, run) in finite_runs(&p.x, &p.lo, None).iter().enumerate() {
                        let points: Vec<_> = run.clone().map(|j| (p.x[j], p.lo[j])).collect();
                        let annotation = if points.len() == 1 {
                            chart.draw_series(std::iter::once(Circle::new(
                                points[0],
                                2,
                                color.filled(),
                            )))?
                        } else {
                            chart
                                .draw_series(LineSeries::new(points, color.stroke_width(stroke)))?
                        };
                        if k == 0 {
                            annotation.label(&p.name).legend(legend);
                        }
                    }
                }
            }
        }

        // Threshold lines.
        if lay.y_range.is_some() {
            for (i, t) in opts.thresholds.iter().enumerate() {
                let (r, g, b) = threshold_color(i);
                let color = RGBColor(r, g, b);
                // Dotted for the first threshold, dashed for the others.
                let (dash, gap) = if i == 0 { (2, 4) } else { (9, 5) };
                let signs: &[f64] = if lay.negative_lines {
                    &[1.0, -1.0]
                } else {
                    &[1.0]
                };
                for (k, sign) in signs.iter().enumerate() {
                    let y = sign * t.value;
                    let annotation = chart.draw_series(DashedLineSeries::new(
                        vec![(x0, y), (x1, y)],
                        dash,
                        gap,
                        color.stroke_width(1),
                    ))?;
                    if k == 0 {
                        annotation.label(&t.label).legend(move |(x, y)| {
                            PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(2))
                        });
                    }
                }
            }
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.85))
            .border_style(TEXT.mix(0.4))
            .label_font(("sans-serif", 14).into_font().color(&TEXT))
            .draw()?;
        Ok(())
    }
}

/// Draws a line plot to an SVG or PNG file.
///
/// The file extension selects the format: `.svg` or `.png` (any letter case). The
/// directory must exist. The plot has the same content as the figure from
/// [`line_figure`](super::line_figure): the same colors, the same threshold lines, the
/// same y range, and the same envelope band for long traces. The look follows the
/// defaults of plotly.js (white background, light gray grid, zero line), like the plots
/// that scasim made before.
///
/// `size` is the width and the height in pixels (for SVG, in user units). Use for
/// example `(1200, 600)`.
///
/// # Errors
///
/// Returns [`PlotError::Input`] for invalid data or a size below 100 pixels,
/// [`PlotError::Extension`] for another extension, and [`PlotError::Draw`] if the back
/// end cannot write the file.
pub fn save_line_plot(
    path: impl AsRef<Path>,
    series: &[Series<'_>],
    opts: &LineOptions,
    size: (u32, u32),
) -> Result<(), PlotError> {
    let prepared = prepare(series, opts)?;
    render(
        path.as_ref(),
        size,
        &LineDrawer {
            prepared,
            options: opts,
        },
    )
}
