//! Input data, options, and errors shared by the HTML and the static back ends.

use super::envelope::envelope;

/// An error from a plot function.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum PlotError {
    /// A file could not be written.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The input data are not valid. The text explains why.
    #[error("invalid plot input: {0}")]
    Input(String),
    /// The drawing back end failed. The text is the message of the back end.
    #[error("drawing error: {0}")]
    Draw(String),
    /// The embedded font could not be loaded.
    #[error("the embedded font could not be loaded")]
    Font,
    /// The file extension is not `.svg` or `.png`.
    #[error("unsupported image file name {0:?}: use the extension .svg or .png")]
    Extension(std::path::PathBuf),
}

/// One data series (one curve) in a line plot.
///
/// The values are borrowed, so a series does not copy a long trace.
#[derive(Clone, Debug)]
pub struct Series<'a> {
    /// The name in the legend.
    pub name: String,
    /// The x values. `None` means "use the sample index 0, 1, 2, ...". Only a series
    /// without x values is reduced with an envelope (see [`LineOptions::buckets`]).
    pub x: Option<&'a [f64]>,
    /// The y values.
    pub y: &'a [f64],
}

impl<'a> Series<'a> {
    /// A series whose x values are the sample indices. Use it for a t-value trace.
    pub fn indexed(name: impl Into<String>, y: &'a [f64]) -> Self {
        Self {
            name: name.into(),
            x: None,
            y,
        }
    }

    /// A series with explicit x values, for example the number of traces on the x axis
    /// of a max-|t| plot. `x` and `y` must have the same length.
    pub fn with_x(name: impl Into<String>, x: &'a [f64], y: &'a [f64]) -> Self {
        Self {
            name: name.into(),
            x: Some(x),
            y,
        }
    }
}

/// A horizontal threshold line, for example the standard threshold 4.5 or an adjusted
/// threshold that accounts for the number of samples.
///
/// The first threshold in a list is drawn as a red dotted line (as in the old plots).
/// The second is a black dashed line. More thresholds are gray dash-dotted lines.
/// A line is also drawn at `-value` if the plot shows negative values.
#[derive(Clone, Debug, PartialEq)]
pub struct Threshold {
    /// The value on the y axis. It must be finite and positive.
    pub value: f64,
    /// The text in the legend, for example `"±4.5"` or `"adjusted ±6.1"`.
    pub label: String,
}

impl Threshold {
    /// Creates a threshold.
    pub fn new(value: f64, label: impl Into<String>) -> Self {
        Self {
            value,
            label: label.into(),
        }
    }
}

/// Options for line plots (t-values per sample, or max |t| versus number of traces).
///
/// Use [`LineOptions::t_values`] or [`LineOptions::max_t`] and change single fields
/// with the struct update syntax:
///
/// ```
/// use scasim::plot::{LineOptions, Threshold};
///
/// let options = LineOptions {
///     thresholds: vec![Threshold::new(4.5, "±4.5"), Threshold::new(6.2, "adjusted ±6.2")],
///     ..LineOptions::t_values()
/// };
/// assert_eq!(options.buckets, 4000);
/// ```
#[derive(Clone, Debug)]
pub struct LineOptions {
    /// The title of the x axis.
    pub x_label: String,
    /// The title of the y axis.
    pub y_label: String,
    /// The threshold lines. If the list is empty, the y axis scales automatically.
    /// Otherwise the y range is fixed, like in the old plots: from `-m` to `m`, or from
    /// 0 to `m` if all values are not negative, where `m` is
    /// `max(largest threshold * 1.5, largest |value|) + 0.5`.
    pub thresholds: Vec<Threshold>,
    /// Plot `|value|` instead of `value`.
    pub abs_values: bool,
    /// The maximum number of points per series. A series without x values that is longer
    /// than this is reduced with [`envelope`]: the plot shows a band between the
    /// minimum and the maximum of each bucket, so no spike is lost. Use 2000 to 4000.
    pub buckets: usize,
    /// If `true` (as in the old plots), NaN and infinite values are plotted as 0. If
    /// `false`, they are plotted as gaps.
    pub non_finite_as_zero: bool,
    /// HTML and JSON only: round the plotted values to this number of decimal places.
    /// This makes the file about half as large. `None` keeps all digits.
    pub decimals: Option<u32>,
    /// The width of the data lines in pixels.
    pub line_width: f64,
}

impl LineOptions {
    /// The preset for t-values per sample: threshold ±4.5, y axis "t-value".
    pub fn t_values() -> Self {
        Self {
            x_label: "Time (cycles)".into(),
            y_label: "t-value".into(),
            thresholds: vec![Threshold::new(4.5, "±4.5")],
            abs_values: false,
            buckets: 4000,
            non_finite_as_zero: true,
            decimals: Some(4),
            line_width: 2.0,
        }
    }

    /// The preset for max |t| versus the number of traces: threshold 4.5.
    pub fn max_t() -> Self {
        Self {
            x_label: "Number of traces".into(),
            y_label: "max(|t|)".into(),
            thresholds: vec![Threshold::new(4.5, "4.5")],
            line_width: 1.0,
            ..Self::t_values()
        }
    }
}

/// A series after cleaning and reduction. Shared by the HTML and the static back ends.
pub(crate) struct Prepared {
    pub name: String,
    pub x: Vec<f64>,
    /// The values, or the minimum of each bucket if `hi` is `Some`.
    pub lo: Vec<f64>,
    /// The maximum of each bucket. `None` for a series that is drawn as a plain line.
    pub hi: Option<Vec<f64>>,
}

/// The ranges and the extra lines of a prepared plot.
pub(crate) struct Layout2d {
    pub x_range: (f64, f64),
    /// `None` means automatic scaling.
    pub y_range: Option<(f64, f64)>,
    /// `true` if the threshold lines at `-value` are in the y range.
    pub negative_lines: bool,
}

fn clean(v: f64, opts: &LineOptions) -> f64 {
    let v = if v.is_finite() {
        v
    } else if opts.non_finite_as_zero {
        0.0
    } else {
        f64::NAN
    };
    if opts.abs_values { v.abs() } else { v }
}

/// Validates the input, cleans the values, and reduces long traces.
pub(crate) fn prepare(
    series: &[Series<'_>],
    opts: &LineOptions,
) -> Result<Vec<Prepared>, PlotError> {
    if series.is_empty() {
        return Err(PlotError::Input("no series".into()));
    }
    if opts.buckets == 0 {
        return Err(PlotError::Input("buckets must be at least 1".into()));
    }
    if let Some(t) = opts
        .thresholds
        .iter()
        .find(|t| !(t.value.is_finite() && t.value > 0.0))
    {
        return Err(PlotError::Input(format!(
            "threshold {:?} has the value {}: it must be finite and positive",
            t.label, t.value
        )));
    }
    series
        .iter()
        .map(|s| {
            if s.y.is_empty() {
                return Err(PlotError::Input(format!("series {:?} is empty", s.name)));
            }
            if let Some(x) = s.x
                && x.len() != s.y.len()
            {
                return Err(PlotError::Input(format!(
                    "series {:?}: x has {} values but y has {}",
                    s.name,
                    x.len(),
                    s.y.len()
                )));
            }
            let y: Vec<f64> = s.y.iter().map(|&v| clean(v, opts)).collect();
            Ok(match s.x {
                Some(x) => Prepared {
                    name: s.name.clone(),
                    x: x.to_vec(),
                    lo: y,
                    hi: None,
                },
                None if y.len() > opts.buckets => {
                    let (x, lo, hi) = envelope(&y, opts.buckets);
                    Prepared {
                        name: s.name.clone(),
                        x,
                        lo,
                        hi: Some(hi),
                    }
                }
                None => Prepared {
                    name: s.name.clone(),
                    x: (0..y.len()).map(|i| i as f64).collect(),
                    lo: y,
                    hi: None,
                },
            })
        })
        .collect()
}

fn finite_min_max(it: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    it.filter(|v| v.is_finite()).fold(None, |acc, v| match acc {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

/// Computes the axis ranges.
pub(crate) fn layout(prepared: &[Prepared], opts: &LineOptions) -> Layout2d {
    let x = finite_min_max(prepared.iter().flat_map(|p| p.x.iter().copied())).unwrap_or((0.0, 1.0));
    // Avoid an empty range, e.g., for a series with one point.
    let x_range = if x.0 < x.1 { x } else { (x.0 - 0.5, x.0 + 0.5) };
    let y = finite_min_max(
        prepared
            .iter()
            .flat_map(|p| p.lo.iter().chain(p.hi.iter().flatten()).copied()),
    );
    let t_top = opts
        .thresholds
        .iter()
        .map(|t| t.value)
        .fold(f64::NAN, f64::max);
    if t_top.is_nan() {
        return Layout2d {
            x_range,
            y_range: None,
            negative_lines: false,
        };
    }
    let (y_min, y_max) = y.unwrap_or((0.0, 0.0));
    let top = (y_max.abs().max(y_min.abs())).max(t_top).max(1.5 * t_top) + 0.5;
    let non_negative = y_min >= 0.0;
    Layout2d {
        x_range,
        y_range: Some((if non_negative { 0.0 } else { -top }, top)),
        negative_lines: !non_negative,
    }
}

/// The default color cycle of plotly.js (D3 category 10), so that the HTML and the static
/// plots look like the plots that scasim made before.
pub(crate) const PALETTE: [(u8, u8, u8); 10] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
];

/// The color of the `i`-th threshold line: red, black, then gray.
pub(crate) fn threshold_color(i: usize) -> (u8, u8, u8) {
    match i {
        0 => (255, 0, 0),
        1 => (0, 0, 0),
        _ => (100, 100, 100),
    }
}
