//! Tests for the plot module. They write their files to `target/plot-test-out/` (or to the
//! directory in the environment variable `PLOT_TEST_OUT`) and print file sizes and CPU
//! times. Run `cargo test --release -- --nocapture` to see them.

use std::path::{Path, PathBuf};

use cpu_time::ThreadTime;
use plotly::Plot;

use super::test_data::{Leak, max_t_curve, t_trace};
use super::*;

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("PLOT_TEST_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/plot-test-out"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn size(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}

/// Runs `f` and returns its result and the CPU time of this thread in milliseconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = ThreadTime::now();
    let out = f();
    (out, start.elapsed().as_secs_f64() * 1e3)
}

const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";

fn assert_png(path: &Path, expected: (u32, u32)) -> image::RgbImage {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.starts_with(PNG_MAGIC), "{path:?} is not a PNG file");
    let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .unwrap()
        .to_rgb8();
    assert_eq!(image.dimensions(), expected);
    image
}

fn assert_svg(path: &Path, expected: (u32, u32)) -> String {
    let text = std::fs::read_to_string(path).unwrap();
    let text = text.trim();
    assert!(
        text.starts_with("<svg "),
        "{path:?} does not start with <svg"
    );
    assert!(
        text.ends_with("</svg>"),
        "{path:?} does not end with </svg>"
    );
    assert!(text.contains(&format!(
        "width=\"{}\" height=\"{}\"",
        expected.0, expected.1
    )));
    text.to_string()
}

/// The three planted leaks of the long test trace.
const SPIKE: usize = 777_777;

fn long_trace() -> Vec<f64> {
    t_trace(
        1_000_000,
        1,
        &[
            Leak {
                start: 123_456,
                len: 1,
                amplitude: 14.0,
            },
            Leak {
                start: SPIKE,
                len: 1,
                amplitude: -11.0,
            },
        ],
    )
}

// ---------------------------------------------------------------------------------------
// envelope
// ---------------------------------------------------------------------------------------

#[test]
fn envelope_keeps_a_single_sample_spike_in_a_million_samples() {
    let mut trace = vec![0.0; 1_000_000];
    trace[SPIKE] = 40.0;
    trace[250_001] = -30.0;
    let (x, min, max) = envelope(&trace, 2000);
    assert_eq!((x.len(), min.len(), max.len()), (2000, 2000, 2000));

    let argmax = max
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert_eq!(*argmax.1, 40.0);
    let bucket = bucket_bounds(1_000_000, 2000, argmax.0);
    assert!(
        bucket.contains(&SPIKE),
        "spike {SPIKE} is not in bucket {bucket:?}"
    );
    // The x value is the center of the bucket.
    assert!(x[argmax.0] >= bucket.start as f64 && x[argmax.0] < bucket.end as f64);
    // The negative spike survives in the minimum, and only there.
    let argmin = min
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert_eq!(*argmin.1, -30.0);
    assert!(bucket_bounds(1_000_000, 2000, argmin.0).contains(&250_001));
    // All other buckets are flat.
    assert_eq!(max.iter().filter(|&&v| v != 0.0).count(), 1);
    assert_eq!(min.iter().filter(|&&v| v != 0.0).count(), 1);
}

#[test]
fn envelope_matches_a_brute_force_reduction() {
    let trace = t_trace(100_003, 5, &[]);
    for buckets in [1, 2, 7, 1000, 4000, 100_002] {
        let (_, min, max) = envelope(&trace, buckets);
        assert_eq!(min.len(), buckets);
        for i in 0..buckets {
            let r = bucket_bounds(trace.len(), buckets, i);
            let slice = &trace[r];
            let lo = slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert_eq!((min[i], max[i]), (lo, hi), "bucket {i} of {buckets}");
        }
    }
}

#[test]
fn envelope_ignores_nan_and_keeps_all_nan_buckets_as_nan() {
    let nan = f64::NAN;
    // 12 samples in 4 buckets of 3.
    let values = [
        1.0,
        nan,
        3.0, // some NaN: min 1, max 3
        nan,
        nan,
        nan, // all NaN
        nan,
        -2.0,
        nan, // one finite value
        f64::INFINITY,
        5.0,
        f64::NEG_INFINITY, // infinite values count as values
    ];
    let (_, min, max) = envelope(&values, 4);
    assert_eq!((min[0], max[0]), (1.0, 3.0));
    assert!(min[1].is_nan() && max[1].is_nan());
    assert_eq!((min[2], max[2]), (-2.0, -2.0));
    assert_eq!((min[3], max[3]), (f64::NEG_INFINITY, f64::INFINITY));
}

#[test]
fn envelope_returns_short_input_unchanged() {
    let same = |a: &[f64], b: &[f64]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    for (len, buckets) in [(0, 5), (1, 5), (4, 5), (5, 5), (2000, 2000), (1, 1)] {
        let values: Vec<f64> = (0..len)
            .map(|i| {
                if i % 3 == 2 {
                    f64::NAN
                } else {
                    i as f64 * 0.5 - 1.0
                }
            })
            .collect();
        let (x, min, max) = envelope(&values, buckets);
        assert!(same(&min, &values), "min, len {len}");
        assert!(same(&max, &values), "max, len {len}");
        let index: Vec<f64> = (0..len).map(|i| i as f64).collect();
        assert!(same(&x, &index), "x, len {len}");
    }
    // One more sample than buckets is reduced.
    let (x, _, _) = envelope(&[1.0, 2.0, 3.0], 2);
    assert_eq!(x.len(), 2);
}

#[test]
fn bucket_bounds_cover_every_sample_exactly_once() {
    let cases = [
        (1, 1),
        (7, 3),
        (10, 10),
        (11, 10),
        (19, 10),
        (2000, 1999),
        (1_000_000, 2000),
        (1_000_003, 2000),
        (1_000_000, 3999),
        (123_457, 4000),
    ];
    for (n, buckets) in cases {
        let mut count = vec![0u8; n];
        let mut expected_start = 0;
        let (mut smallest, mut largest) = (usize::MAX, 0);
        for i in 0..buckets {
            let r = bucket_bounds(n, buckets, i);
            assert_eq!(
                r.start, expected_start,
                "n={n} buckets={buckets} i={i}: gap or overlap"
            );
            assert!(!r.is_empty(), "n={n} buckets={buckets} i={i}: empty bucket");
            smallest = smallest.min(r.len());
            largest = largest.max(r.len());
            for c in &mut count[r.clone()] {
                *c += 1;
            }
            expected_start = r.end;
        }
        assert_eq!(expected_start, n);
        assert!(count.iter().all(|&c| c == 1), "n={n} buckets={buckets}");
        assert!(
            largest - smallest <= 1,
            "n={n} buckets={buckets}: uneven buckets"
        );
    }
}

#[test]
#[should_panic(expected = "at least 1")]
fn envelope_with_zero_buckets_panics() {
    envelope(&[1.0], 0);
}

// ---------------------------------------------------------------------------------------
// interactive HTML
// ---------------------------------------------------------------------------------------

fn options_with_adjusted() -> LineOptions {
    LineOptions {
        thresholds: vec![
            Threshold::new(4.5, "±4.5"),
            Threshold::new(5.9, "adjusted ±5.9"),
        ],
        ..LineOptions::t_values()
    }
}

fn parsed(plot: &Plot) -> serde_json::Value {
    serde_json::from_str(&plot.to_json()).unwrap()
}

#[test]
fn html_for_a_million_samples_has_embedded_js_and_stays_small() {
    let dir = out_dir();
    let trace = long_trace();
    let series = [Series::indexed("d=1", &trace)];
    let options = options_with_adjusted();

    let (figure, cpu_build) = timed(|| line_figure(&series, &options).unwrap());
    let path = dir.join("test_d1_envelope.html");
    let (_, cpu_write) =
        timed(|| write_html(&path, "t-test d=1", &[&figure], JsSource::Embedded).unwrap());
    let html = std::fs::read_to_string(&path).unwrap();
    assert!(html.starts_with("<!doctype html>"));
    assert!(
        html.contains("plotly.js v3.0.1"),
        "plotly.js is not embedded"
    );
    assert!(html.contains("Plotly.newPlot"));
    assert!(html.contains("<title>t-test d=1</title>"));
    assert!(
        !html.contains("<script src="),
        "an embedded page must not load a script"
    );
    let with_envelope = size(&path);
    assert!(
        with_envelope < 5_000_000,
        "HTML with envelope is {with_envelope} bytes"
    );

    let json_path = dir.join("test_d1_envelope.json");
    write_json(&json_path, &figure).unwrap();
    let json_size = size(&json_path);

    // The same figure without an envelope (every sample), for the size report.
    let full_options = LineOptions {
        buckets: usize::MAX,
        ..options.clone()
    };
    let (full_figure, cpu_full) = timed(|| line_figure(&series, &full_options).unwrap());
    let full_path = dir.join("test_d1_full.html");
    write_html(&full_path, "full", &[&full_figure], JsSource::Embedded).unwrap();
    let without_envelope = size(&full_path);
    assert!(without_envelope > 4 * with_envelope);

    // The stock writer of the plotly crate embeds MathJax, too.
    let stock_path = dir.join("test_d1_envelope_stock.html");
    figure.write_html(&stock_path);
    let stock = size(&stock_path);
    assert!(stock > with_envelope + 2_000_000);

    let cdn_path = dir.join("test_d1_envelope_cdn.html");
    write_html(&cdn_path, "cdn", &[&figure], JsSource::Cdn).unwrap();
    let cdn = size(&cdn_path);
    assert!(cdn < 400_000, "CDN page is {cdn} bytes");
    assert!(
        std::fs::read_to_string(&cdn_path)
            .unwrap()
            .contains("cdn.plot.ly/plotly-3.0.1.min.js")
    );

    println!("HTML sizes for one trace of 10^6 samples:");
    println!("  with envelope (4000 buckets), our writer, embedded JS: {with_envelope} bytes");
    println!("  with envelope, our writer, CDN link:                   {cdn} bytes");
    println!("  with envelope, stock Plot::write_html (JS + MathJax):  {stock} bytes");
    println!("  without envelope, our writer, embedded JS:             {without_envelope} bytes");
    println!("  JSON with envelope:                                    {json_size} bytes");
    println!(
        "  CPU: build figure {cpu_build:.1} ms, write HTML {cpu_write:.1} ms, build full figure {cpu_full:.1} ms"
    );
}

#[test]
fn html_figure_has_envelope_thresholds_and_hover() {
    let trace = long_trace();
    let figure = line_figure(&[Series::indexed("d=1", &trace)], &options_with_adjusted()).unwrap();
    let json = parsed(&figure);
    let data = json["data"].as_array().unwrap();
    // max line, min line, two threshold pairs.
    assert_eq!(data.len(), 2 + 4);
    assert_eq!(data[0]["x"].as_array().unwrap().len(), 4000);
    assert_eq!(data[1]["fill"], "tonexty");
    // The spike survives in the data of the max trace.
    let top = data[0]["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .fold(f64::MIN, f64::max);
    assert!(top > 12.0, "max of the envelope is {top}");
    // Hover is enabled for data traces and for the whole figure.
    assert_eq!(json["layout"]["hovermode"], "x");
    for trace in &data[..2] {
        assert!(trace.get("hoverinfo").is_none());
        assert!(trace["hovertemplate"].as_str().unwrap().contains("%{y"));
    }
    // Threshold lines at +-4.5 and +-5.9, in the legend once each.
    let level = |i: usize| data[i]["y"][0].as_f64().unwrap();
    assert_eq!(
        [level(2), level(3), level(4), level(5)],
        [4.5, -4.5, 5.9, -5.9]
    );
    assert_eq!(data[2]["name"], "±4.5");
    assert_eq!(data[3]["showlegend"], false);
    assert_eq!(data[4]["name"], "adjusted ±5.9");
    // The old y range rule: m = max(1.5 * 5.9, max |t|) + 0.5.
    let range = &json["layout"]["yaxis"]["range"];
    let m = range[1].as_f64().unwrap();
    assert!((m - (top.max(1.5 * 5.9) + 0.5)).abs() < 0.6, "m = {m}");
    assert_eq!(range[0].as_f64().unwrap(), -m);
    assert_eq!(json["config"]["displayModeBar"], "hover");
    assert_eq!(json["config"]["typesetMath"], false);
}

#[test]
fn html_short_trace_has_no_envelope_and_no_negative_lines_for_abs() {
    let t = t_trace(500, 2, &[]);
    let options = LineOptions {
        abs_values: true,
        y_label: "|t|".into(),
        ..LineOptions::t_values()
    };
    let figure = line_figure(&[Series::indexed("d=1", &t)], &options).unwrap();
    let json = parsed(&figure);
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "one data trace and one threshold line");
    assert_eq!(data[0]["y"].as_array().unwrap().len(), 500);
    assert!(
        data[0]["y"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.as_f64().unwrap() >= 0.0)
    );
    assert_eq!(json["layout"]["yaxis"]["range"][0].as_f64().unwrap(), 0.0);
}

#[test]
fn html_max_t_plot_and_multi_figure_page() {
    let dir = out_dir();
    let traces: Vec<f64> = (1..=50).map(|k| k as f64 * 1000.0).collect();
    let m1 = max_t_curve(&traces, 0.03, 3.2, 1);
    let m2 = max_t_curve(&traces, 0.0, 3.4, 2);
    let series = [
        Series::with_x("d=1", &traces, &m1),
        Series::with_x("d=2", &traces, &m2),
    ];
    let max_figure = line_figure(&series, &LineOptions::max_t()).unwrap();
    let t = t_trace(3000, 3, &[]);
    let t_figure = line_figure(&[Series::indexed("d=1", &t)], &LineOptions::t_values()).unwrap();

    let path = dir.join("test_page_two_figures.html");
    write_html(
        &path,
        "two <figures> & more",
        &[&t_figure, &max_figure],
        JsSource::Embedded,
    )
    .unwrap();
    let html = std::fs::read_to_string(&path).unwrap();
    // Two plots, and one call inside the library text itself is not counted here: the
    // calls we generate have the form `Plotly.newPlot("figure-`.
    assert_eq!(html.matches("Plotly.newPlot(\"figure-").count(), 2);
    assert!(html.contains("id=\"figure-0\"") && html.contains("id=\"figure-1\""));
    assert!(html.contains("<title>two &lt;figures&gt; &amp; more</title>"));
    // One copy of the library, not two.
    assert!(size(&path) < 5_000_000);
    assert_eq!(
        parsed(&max_figure)["data"][0]["x"]
            .as_array()
            .unwrap()
            .len(),
        50
    );
}

#[test]
fn html_gaps_become_null_in_json() {
    let mut t = t_trace(100, 4, &[]);
    t[10] = f64::NAN;
    t[11] = f64::INFINITY;
    let gaps = LineOptions {
        non_finite_as_zero: false,
        ..LineOptions::t_values()
    };
    let json = parsed(&line_figure(&[Series::indexed("d=1", &t)], &gaps).unwrap());
    assert!(json["data"][0]["y"][10].is_null());
    assert!(json["data"][0]["y"][11].is_null());
    let zeros =
        parsed(&line_figure(&[Series::indexed("d=1", &t)], &LineOptions::t_values()).unwrap());
    assert_eq!(zeros["data"][0]["y"][10].as_f64(), Some(0.0));
    assert_eq!(zeros["data"][0]["y"][11].as_f64(), Some(0.0));
}

#[test]
fn invalid_input_is_an_error_not_a_panic() {
    let t = [1.0, 2.0, 3.0];
    let o = LineOptions::t_values();
    assert!(matches!(line_figure(&[], &o), Err(PlotError::Input(_))));
    assert!(matches!(
        line_figure(&[Series::indexed("a", &[])], &o),
        Err(PlotError::Input(_))
    ));
    assert!(matches!(
        line_figure(&[Series::with_x("a", &[1.0], &t)], &o),
        Err(PlotError::Input(_))
    ));
    let zero = LineOptions {
        buckets: 0,
        ..LineOptions::t_values()
    };
    assert!(matches!(
        line_figure(&[Series::indexed("a", &t)], &zero),
        Err(PlotError::Input(_))
    ));
    let dir = out_dir();
    assert!(matches!(
        save_line_plot(
            dir.join("x.jpg"),
            &[Series::indexed("a", &t)],
            &o,
            (400, 300)
        ),
        Err(PlotError::Extension(_))
    ));
    assert!(matches!(
        save_line_plot(dir.join("x.png"), &[Series::indexed("a", &t)], &o, (10, 10)),
        Err(PlotError::Input(_))
    ));
}

/// Runs `cargo tree` for one package and returns the output, or `None` if cargo cannot be
/// run (then the test is skipped). `-i <package> -e features` lists the features that are
/// enabled for the package. `-p <package> -e normal` lists the crates it depends on.
fn cargo_tree(args: [&str; 3]) -> Option<String> {
    let output = std::process::Command::new(env!("CARGO"))
        .args(["tree", "--offline", "--prefix", "none"])
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "cargo tree {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(String::from_utf8(output.stdout).unwrap())
}

/// The plotly features `static_export_*`, `kaleido`, and `plotly_image` start a browser
/// driver, download one, or need the `image` crate. `cargo tree` lists the features and the
/// crates that are really enabled. The checks look at the two packages only, so they also
/// work inside a larger project.
#[test]
fn no_static_export_or_system_font_crates_are_in_the_dependency_tree() {
    let trees = [
        cargo_tree(["-i", "plotly", "-e=features"]),
        cargo_tree(["-p", "plotly", "-e=normal"]),
        cargo_tree(["-i", "plotters", "-e=features"]),
        cargo_tree(["-p", "plotters", "-e=normal"]),
    ];
    let [
        Some(plotly_features),
        Some(plotly_crates),
        Some(plotters_features),
        Some(plotters_crates),
    ] = trees
    else {
        println!("cannot run cargo: skipped");
        return;
    };
    assert!(plotly_features.contains("plotly feature \"plotly_embed_js\""));
    for forbidden in [
        "static_export",
        "plotly_static",
        "plotly_kaleido",
        "kaleido",
        "plotly_image",
        "webdriver",
        "fantoccini",
        "chromedriver",
        "geckodriver",
    ] {
        for tree in [&plotly_features, &plotly_crates] {
            assert!(!tree.contains(forbidden), "plotly enables {forbidden:?}");
        }
    }
    assert!(plotters_features.contains("plotters feature \"ab_glyph\""));
    for forbidden in [
        "feature \"ttf\"",
        "feature \"font-kit\"",
        "freetype",
        "fontconfig",
    ] {
        for tree in [&plotters_features, &plotters_crates] {
            assert!(!tree.contains(forbidden), "plotters enables {forbidden:?}");
        }
    }
    assert!(plotters_crates.contains("ab_glyph v"));

    // The plotly crate must not bring the `image` or HTTP crates. (The `image` crate in
    // the tree of scasim comes from the PNG encoder of plotters.)
    for forbidden in ["image v", "reqwest v", "fantoccini v", "tokio v"] {
        assert!(
            !plotly_crates.contains(forbidden),
            "plotly depends on {forbidden:?}"
        );
    }

    // The whole tree of scasim has no browser driver, no static export, and no system font
    // library.
    if let Some(all) = cargo_tree(["-p", "scasim", "-e=normal"]) {
        for forbidden in [
            "plotly_static",
            "fantoccini",
            "chromedriver",
            "geckodriver",
            "kaleido",
            "font-kit",
            "freetype",
            "fontconfig",
        ] {
            assert!(
                !all.contains(forbidden),
                "the dependency tree of scasim has {forbidden:?}"
            );
        }
    }
}

/// SCALib was removed. Its crates (and the C++ build of `geigen`) must not come back.
#[test]
fn scalib_and_the_crates_it_needed_are_not_in_the_dependency_tree() {
    let Some(all) = cargo_tree(["-p", "scasim", "-e=normal"]) else {
        println!("cannot run cargo: skipped");
        return;
    };
    for forbidden in ["scalib", "geigen", "nshare", "plotly_static", "fantoccini"] {
        assert!(
            !all.contains(forbidden),
            "the dependency tree of scasim has {forbidden:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// static images (plotters)
// ---------------------------------------------------------------------------------------

/// Counts the pixels in `rows` that are not white.
fn non_white(image: &image::RgbImage, rows: std::ops::Range<u32>) -> usize {
    rows.flat_map(|y| (0..image.width()).map(move |x| (x, y)))
        .filter(|&(x, y)| image.get_pixel(x, y).0 != [255, 255, 255])
        .count()
}

#[test]
fn static_t_value_plots_svg_and_png() {
    let dir = out_dir();
    let trace = long_trace();
    let series = [Series::indexed("d=1", &trace)];
    let options = options_with_adjusted();
    let dims = (1200, 600);

    let svg = dir.join("test_t_values.svg");
    let png = dir.join("test_t_values.png");
    let (r, cpu_svg) = timed(|| save_line_plot(&svg, &series, &options, dims));
    r.unwrap();
    let (r, cpu_png) = timed(|| save_line_plot(&png, &series, &options, dims));
    r.unwrap();
    let text = assert_svg(&svg, dims);
    assert!(text.contains("Time (cycles)") && text.contains("t-value"));
    assert!(text.contains("<polygon"), "the envelope band is a polygon");
    let image = assert_png(&png, dims);
    // The x axis title and tick labels are drawn, so the font works.
    assert!(
        non_white(&image, 540..600) > 300,
        "no text at the bottom of the PNG"
    );
    // The planted spike (t = 14 at x = 123456) reaches high in the plot: some pixel above
    // the 5.9 line has the series color (31, 119, 180) near the spike.
    let spike_x = 86 + (123_456.0 / 1_000_000.0 * 1100.0) as u32;
    let found = (0..image.height()).any(|y| {
        (spike_x - 3..spike_x + 3).any(|x| {
            let p = image.get_pixel(x, y).0;
            p[2] as i32 > p[0] as i32 + 60 && y < 100
        })
    });
    assert!(found, "the spike is not visible");

    println!(
        "static t-value plot, 10^6 samples: SVG {} bytes ({cpu_svg:.1} ms CPU), PNG {} bytes ({cpu_png:.1} ms CPU)",
        size(&svg),
        size(&png)
    );
}

#[test]
fn static_max_t_plots_svg_and_png() {
    let dir = out_dir();
    let traces: Vec<f64> = (1..=100).map(|k| k as f64 * 1000.0).collect();
    let curves: Vec<Vec<f64>> = [0.03, 0.01, 0.0]
        .iter()
        .enumerate()
        .map(|(i, &g)| max_t_curve(&traces, g, 3.3, 20 + i as u64))
        .collect();
    let series: Vec<Series> = curves
        .iter()
        .enumerate()
        .map(|(i, c)| Series::with_x(format!("d={}", i + 1), &traces, c))
        .collect();
    let options = LineOptions {
        thresholds: vec![
            Threshold::new(4.5, "4.5"),
            Threshold::new(5.9, "adjusted 5.9"),
        ],
        ..LineOptions::max_t()
    };
    let dims = (1200, 500);
    let svg = dir.join("test_max_t.svg");
    let png = dir.join("test_max_t.png");
    let (r, cpu_svg) = timed(|| save_line_plot(&svg, &series, &options, dims));
    r.unwrap();
    let (r, cpu_png) = timed(|| save_line_plot(&png, &series, &options, dims));
    r.unwrap();
    let text = assert_svg(&svg, dims);
    assert!(text.contains("Number of traces") && text.contains("max(|t|)"));
    assert!(text.contains("adjusted 5.9"));
    let image = assert_png(&png, dims);
    assert!(non_white(&image, 440..500) > 300);
    println!(
        "static max-|t| plot: SVG {} bytes ({cpu_svg:.1} ms CPU), PNG {} bytes ({cpu_png:.1} ms CPU)",
        size(&svg),
        size(&png)
    );
}

#[test]
fn static_plots_handle_gaps_and_degenerate_series() {
    let dir = out_dir();
    let mut with_nan = t_trace(10_000, 8, &[]);
    for v in &mut with_nan[3000..3500] {
        *v = f64::NAN;
    }
    with_nan[100] = f64::NAN; // a single gap in the middle of the line
    let gaps = LineOptions {
        non_finite_as_zero: false,
        buckets: 500,
        ..LineOptions::t_values()
    };
    let short_gaps = LineOptions {
        buckets: 100_000,
        ..gaps.clone()
    };
    let one_point = [1.5];
    let constant = vec![2.0; 50];
    let all_nan = vec![f64::NAN; 20];
    for (name, series, options) in [
        (
            "gaps_envelope",
            vec![Series::indexed("d=1", &with_nan)],
            &gaps,
        ),
        (
            "gaps_lines",
            vec![Series::indexed("d=1", &with_nan)],
            &short_gaps,
        ),
        (
            "one_point",
            vec![Series::indexed("d=1", &one_point)],
            &LineOptions::t_values(),
        ),
        (
            "constant",
            vec![Series::indexed("d=1", &constant)],
            &LineOptions::t_values(),
        ),
        ("all_nan", vec![Series::indexed("d=1", &all_nan)], &gaps),
        (
            "no_thresholds",
            vec![Series::indexed("d=1", &constant)],
            &LineOptions {
                thresholds: vec![],
                ..LineOptions::t_values()
            },
        ),
    ] {
        for ext in ["svg", "png"] {
            let path = dir.join(format!("test_degenerate_{name}.{ext}"));
            save_line_plot(&path, &series, options, (600, 300))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(size(&path) > 500);
        }
        line_figure(&series, options).unwrap();
    }
}

// ---------------------------------------------------------------------------------------
// the file-writing functions of tvla and plot
// ---------------------------------------------------------------------------------------

/// A fresh output directory for one test.
fn test_dir(name: &str) -> PathBuf {
    let dir = out_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn plot_t_traces_writes_the_files_of_every_order() {
    let dir = test_dir("t_traces");
    let t_values = ndarray::Array2::from_shape_vec(
        (2, 3000),
        [t_trace(3000, 1, &[]), t_trace(3000, 2, &[])].concat(),
    )
    .unwrap();
    plot_t_traces(t_values.view(), Some(4.5), false, &dir, false).unwrap();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "all_t_values.html",
            "t_test_d1.html",
            "t_test_d1.json",
            "t_test_d1.svg",
            "t_test_d2.html",
            "t_test_d2.json",
            "t_test_d2.svg"
        ]
    );
    let all = std::fs::read_to_string(dir.join("all_t_values.html")).unwrap();
    assert!(all.contains("\"name\":\"d=1\"") && all.contains("\"name\":\"d=2\""));
    let json = read_json(&dir.join("t_test_d2.json"));
    assert_eq!(json["layout"]["yaxis"]["title"]["text"], "t-value");
    assert_eq!(json["layout"]["xaxis"]["title"]["text"], "Time (cycles)");
    // Data, +4.5, and -4.5.
    assert_eq!(json["data"].as_array().unwrap().len(), 3);
    let svg = assert_svg(&dir.join("t_test_d1.svg"), (1200, 600));
    assert!(svg.contains("t-value") && svg.contains("±4.5"));
}

#[test]
fn plot_t_traces_with_abs_values_and_without_threshold() {
    let dir = test_dir("t_traces_abs");
    let t_values = ndarray::Array2::from_shape_vec((1, 100), t_trace(100, 3, &[])).unwrap();
    plot_t_traces(t_values.view(), Some(4.5), true, &dir, false).unwrap();
    let json = read_json(&dir.join("t_test_d1.json"));
    assert_eq!(json["layout"]["yaxis"]["title"]["text"], "|t|");
    assert_eq!(json["layout"]["yaxis"]["range"][0], 0.0);
    // Data and +4.5 only.
    assert_eq!(json["data"].as_array().unwrap().len(), 2);

    let dir = test_dir("t_traces_free");
    plot_t_traces(t_values.view(), None, false, &dir, false).unwrap();
    let json = read_json(&dir.join("t_test_d1.json"));
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
    assert_eq!(json["layout"]["yaxis"]["autorange"], true);
}

#[test]
fn plot_t_traces_returns_errors_for_empty_and_invalid_input() {
    let dir = test_dir("t_traces_errors");
    let no_samples = ndarray::Array2::<f64>::zeros((2, 0));
    let no_orders = ndarray::Array2::<f64>::zeros((0, 10));
    let one = ndarray::Array2::<f64>::zeros((1, 10));
    for (t, threshold) in [
        (&no_samples, Some(4.5)),
        (&no_orders, Some(4.5)),
        (&one, Some(f64::NAN)),
        (&one, Some(-1.0)),
        (&one, Some(f64::INFINITY)),
    ] {
        let r = plot_t_traces(t.view(), threshold, false, &dir, false);
        assert!(matches!(r, Err(PlotError::Input(_))), "{t:?} {threshold:?}");
    }
    // A directory that does not exist is an I/O error.
    let r = plot_t_traces(one.view(), Some(4.5), false, &dir.join("missing"), false);
    assert!(matches!(r, Err(PlotError::Io(_))), "{r:?}");
}

#[test]
fn plot_t_traces_of_all_nan_and_all_zero_values_does_not_panic() {
    let dir = test_dir("t_traces_degenerate");
    for value in [f64::NAN, 0.0, f64::INFINITY] {
        let t = ndarray::Array2::from_elem((2, 50), value);
        plot_t_traces(t.view(), Some(4.5), false, &dir, false).unwrap();
        let json = read_json(&dir.join("t_test_d1.json"));
        assert_eq!(json["data"][0]["y"][0], 0.0, "{value}");
    }
}

#[test]
fn plot_max_t_values_does_not_panic_on_nan() {
    let dir = test_dir("max_t_nan");
    let counts = [0, 100, 200, 300];
    // The first series is all NaN (no finite t-value in any batch). The second has a NaN
    // in the middle. The old code panicked in `partial_cmp().unwrap()`.
    let values = vec![
        vec![f64::NAN; 4],
        vec![0.0, 3.0, f64::NAN, 5.0],
        vec![0.0, f64::INFINITY, 1.0, 2.0],
    ];
    plot_max_t_values(&values, &counts, Some(4.5), &dir, false).unwrap();
    for name in ["max_t_values.html", "max_t_values.svg", "max_t_values.json"] {
        assert!(size(&dir.join(name)) > 1000, "{name}");
    }
    let json = read_json(&dir.join("max_t_values.json"));
    // Three series and one threshold line. NaN and infinity are gaps.
    assert_eq!(json["data"].as_array().unwrap().len(), 4);
    assert!(json["data"][0]["y"][0].is_null());
    assert!(json["data"][1]["y"][2].is_null());
    assert!(json["data"][2]["y"][1].is_null());
    assert_eq!(
        json["data"][1]["x"],
        serde_json::json!([0.0, 100.0, 200.0, 300.0])
    );
    assert_eq!(json["layout"]["xaxis"]["title"]["text"], "Number of traces");
    assert_eq!(
        json["layout"]["yaxis"]["title"]["text"],
        "max(|t|), descriptive repeated looks"
    );
    assert_svg(&dir.join("max_t_values.svg"), (1200, 600));
}

#[test]
fn plot_max_t_values_returns_errors_for_invalid_input() {
    let dir = test_dir("max_t_errors");
    let r = plot_max_t_values(&[], &[0, 1], Some(4.5), &dir, false);
    assert!(matches!(r, Err(PlotError::Input(_))), "{r:?}");
    let r = plot_max_t_values(&[vec![]], &[], Some(4.5), &dir, false);
    assert!(matches!(r, Err(PlotError::Input(_))), "{r:?}");
    let r = plot_max_t_values(&[vec![1.0, 2.0]], &[0, 1, 2], Some(4.5), &dir, false);
    assert!(matches!(r, Err(PlotError::Input(_))), "{r:?}");
    let r = plot_max_t_values(&[vec![1.0]], &[5], Some(0.0), &dir, false);
    assert!(matches!(r, Err(PlotError::Input(_))), "{r:?}");
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "no file is written"
    );
}

#[test]
fn plot_max_t_values_with_one_point_works() {
    let dir = test_dir("max_t_one_point");
    plot_max_t_values(&[vec![0.0]], &[0], None, &dir, false).unwrap();
    plot_max_t_values(&[vec![7.0]], &[1000], Some(4.5), &dir, false).unwrap();
}

#[test]
fn max_finite_ignores_non_finite_values() {
    use super::max_finite;
    assert_eq!(max_finite(&[]), None);
    assert_eq!(max_finite(&[f64::NAN, f64::INFINITY]), None);
    assert_eq!(
        max_finite(&[1.0, f64::NAN, 3.5, f64::INFINITY, 2.0]),
        Some(3.5)
    );
    assert_eq!(max_finite(&[-3.0, -1.0]), Some(-1.0));
}

#[test]
fn plot_series_serves_the_chi2_plots() {
    let dir = test_dir("series");
    // max chi2 versus number of traces: x values, one threshold.
    let traces: Vec<f64> = (1..=50).map(|k| k as f64 * 1000.0).collect();
    let max_chi2 = max_t_curve(&traces, 0.02, 3.0, 5);
    let opts = LineOptions {
        y_label: "max(-log10 p)".into(),
        thresholds: vec![Threshold::new(5.0, "5")],
        ..LineOptions::max_t()
    };
    plot_series(
        "max_chi2",
        &[Series::with_x("pearson", &traces, &max_chi2)],
        &opts,
        &dir,
        false,
    )
    .unwrap();
    // -log10 p per sample: indexed series, two thresholds. The values are not negative, so
    // there are no lines at -5.
    let trace: Vec<f64> = t_trace(20_000, 6, &[]).iter().map(|v| v.abs()).collect();
    let opts = LineOptions {
        x_label: "Sample".into(),
        y_label: "-log10(p)".into(),
        symmetric: false,
        thresholds: vec![
            Threshold::new(5.0, "5"),
            Threshold::new(7.3, "Bonferroni 7.3"),
        ],
        buckets: 1000,
        ..LineOptions::t_values()
    };
    plot_series(
        "chi2",
        &[Series::indexed("pearson", &trace)],
        &opts,
        &dir,
        false,
    )
    .unwrap();
    for stem in ["max_chi2", "chi2"] {
        for ext in ["html", "svg", "json"] {
            assert!(
                size(&dir.join(format!("{stem}.{ext}"))) > 1000,
                "{stem}.{ext}"
            );
        }
    }
    let json = read_json(&dir.join("chi2.json"));
    // Max line, min line (envelope), and two thresholds.
    assert_eq!(json["data"].as_array().unwrap().len(), 4);
    assert_eq!(json["data"][2]["name"], "5");
    assert_eq!(json["data"][3]["name"], "Bonferroni 7.3");
    assert_eq!(json["layout"]["yaxis"]["title"]["text"], "-log10(p)");
    let svg = std::fs::read_to_string(dir.join("chi2.svg")).unwrap();
    assert!(svg.contains("Bonferroni 7.3") && svg.contains("-log10(p)"));
}

#[test]
fn plot_series_rejects_bad_names_and_input() {
    let dir = test_dir("series_errors");
    let y = [1.0, 2.0];
    let s = [Series::indexed("a", &y)];
    let o = LineOptions::t_values();
    for stem in ["", "a/b", "..\\x"] {
        let r = plot_series(stem, &s, &o, &dir, false);
        assert!(matches!(r, Err(PlotError::Input(_))), "{stem:?}");
    }
    assert!(matches!(
        plot_series("x", &[], &o, &dir, false),
        Err(PlotError::Input(_))
    ));
    let bad = LineOptions {
        thresholds: vec![Threshold::new(f64::NAN, "nan")],
        ..LineOptions::t_values()
    };
    assert!(matches!(
        plot_series("x", &s, &bad, &dir, false),
        Err(PlotError::Input(_))
    ));
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
}

#[test]
fn static_plot_with_non_finite_x_values_does_not_panic() {
    let dir = test_dir("nan_x");
    let x = [0.0, 1.0, f64::NAN, 3.0, f64::INFINITY];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let s = [Series::with_x("a", &x, &y)];
    save_line_plot(dir.join("x.svg"), &s, &LineOptions::max_t(), (400, 300)).unwrap();
    save_line_plot(dir.join("x.png"), &s, &LineOptions::max_t(), (400, 300)).unwrap();
}

// ---------------------------------------------------------------------------------------
// symmetric t-value plots
// ---------------------------------------------------------------------------------------

/// The number of y values of the `data` traces of a figure that are the constant `y`
/// (the threshold lines).
fn lines_at(json: &serde_json::Value, y: f64) -> usize {
    json["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| {
            t["y"]
                .as_array()
                .is_some_and(|v| v.len() == 2 && v[0] == y && v[1] == y)
        })
        .count()
}

#[test]
fn t_plots_of_all_zero_and_all_positive_values_are_symmetric_with_two_threshold_lines() {
    for (name, value) in [("zero", 0.0), ("positive", 2.0)] {
        let dir = test_dir(&format!("symmetric_{name}"));
        let t = ndarray::Array2::from_elem((1, 60), value);
        plot_t_traces(t.view(), Some(4.5), false, &dir, false).unwrap();
        // HTML and JSON: range [-m, m], m = 1.5 * 4.5 + 0.5 = 7.25, and both lines.
        let json = read_json(&dir.join("t_test_d1.json"));
        assert_eq!(
            json["layout"]["yaxis"]["range"],
            serde_json::json!([-7.25, 7.25]),
            "{name}"
        );
        assert_eq!(lines_at(&json, 4.5), 1, "{name}");
        assert_eq!(lines_at(&json, -4.5), 1, "{name}");
        // SVG: the y axis has negative tick labels, and the threshold line is drawn twice.
        let svg = std::fs::read_to_string(dir.join("t_test_d1.svg")).unwrap();
        assert!(
            svg.lines()
                .any(|l| l.starts_with('-') && l[1..].parse::<f64>().is_ok()),
            "{name}: no negative tick label in the SVG"
        );
        let red = svg.matches("#FF0000").count();
        let one_sided = {
            let o = LineOptions {
                symmetric: false,
                ..LineOptions::t_values()
            };
            let p = dir.join("one_sided.svg");
            let y = [value; 60];
            save_line_plot(&p, &[Series::indexed("d=1", &y)], &o, (1200, 600)).unwrap();
            std::fs::read_to_string(p)
                .unwrap()
                .matches("#FF0000")
                .count()
        };
        assert!(
            red > one_sided + one_sided / 2,
            "{name}: {red} red items, one-sided {one_sided}"
        );
    }
}

#[test]
fn chi2_style_plots_stay_non_negative() {
    let dir = test_dir("non_negative");
    let y = [0.5, 2.0, 7.0, 1.0];
    let opts = LineOptions {
        symmetric: false,
        thresholds: vec![Threshold::new(5.0, "5")],
        ..LineOptions::t_values()
    };
    plot_series(
        "chi2",
        &[Series::indexed("pearson", &y)],
        &opts,
        &dir,
        false,
    )
    .unwrap();
    let json = read_json(&dir.join("chi2.json"));
    assert_eq!(json["layout"]["yaxis"]["range"][0], 0.0);
    assert_eq!(lines_at(&json, 5.0), 1);
    assert_eq!(lines_at(&json, -5.0), 0);
    // max |t| plots are non-negative, too.
    let max = LineOptions::max_t();
    assert!(!max.symmetric);
    let x = [1.0, 2.0];
    let v = [1.0, 3.0];
    let fig = parsed(&line_figure(&[Series::with_x("d=1", &x, &v)], &max).unwrap());
    assert_eq!(fig["layout"]["yaxis"]["range"][0], 0.0);
    // A t-value series with negative values is symmetric with or without the option.
    let signed = [-3.0, 1.0];
    let sym =
        parsed(&line_figure(&[Series::indexed("a", &signed)], &LineOptions::t_values()).unwrap());
    assert_eq!(sym["layout"]["yaxis"]["range"][0], -7.25);
}

/// The lengths of the dashes that the SVG draws in `color`: horizontal two-point polylines.
fn dash_lengths(svg: &str, color: &str) -> std::collections::BTreeSet<i64> {
    let needle = format!("stroke=\"{color}\" stroke-width=\"1\" points=\"");
    svg.lines()
        .filter_map(|l| {
            let rest = &l[l.find(&needle)? + needle.len()..];
            let points = rest.split('"').next()?;
            let xs: Vec<(i64, i64)> = points
                .split_whitespace()
                .map(|p| {
                    let (x, y) = p.split_once(',').unwrap();
                    (x.parse().unwrap(), y.parse().unwrap())
                })
                .collect();
            (xs.len() == 2 && xs[0].1 == xs[1].1).then(|| xs[1].0 - xs[0].0)
        })
        .collect()
}

#[test]
fn static_threshold_lines_match_the_html_dash_styles() {
    let dir = test_dir("dash_styles");
    let y = [0.0, 1.0, -1.0, 0.5];
    let opts = LineOptions {
        thresholds: vec![
            Threshold::new(2.0, "first"),
            Threshold::new(3.0, "second"),
            Threshold::new(4.0, "third"),
            Threshold::new(5.0, "fourth"),
        ],
        ..LineOptions::t_values()
    };
    let svg_path = dir.join("dashes.svg");
    save_line_plot(&svg_path, &[Series::indexed("a", &y)], &opts, (1200, 600)).unwrap();
    let svg = std::fs::read_to_string(svg_path).unwrap();
    // Pixel rounding changes a length by one, so the test only tells short (dot) from long
    // (dash) pieces. First: red dotted. Second: black dashed. Third and later: gray dash-dot.
    let kinds = |color: &str| {
        let lengths = dash_lengths(&svg, color);
        (
            lengths.iter().any(|&l| l <= 3),
            lengths.iter().any(|&l| l >= 8),
        )
    };
    assert_eq!(kinds("#FF0000"), (true, false), "dotted");
    assert_eq!(kinds("#000000"), (false, true), "dashed");
    assert_eq!(kinds("#646464"), (true, true), "dash-dot");
    // The HTML uses the same styles.
    let json = parsed(&line_figure(&[Series::indexed("a", &y)], &opts).unwrap());
    let dashes: Vec<&str> = json["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["line"]["dash"].as_str())
        .collect();
    assert_eq!(
        &dashes[..6],
        ["dot", "dot", "dash", "dash", "dashdot", "dashdot"]
    );
}
