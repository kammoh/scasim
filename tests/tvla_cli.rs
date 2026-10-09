//! Runs the `tvla` binary on a tiny batch: a VCD waveform, a legacy `meta.json`, and a
//! `traces.npz` that differs from the traces of the waveform. The tests check when `tvla` reads
//! and writes `traces.npz`, what it prints, and how `--list-signals` splits its output.

mod common;

use common::*;
use ndarray::{Array1, Array2};
use ndarray_npz::{NpzReader, NpzWriter};
use scasim::stats::{Binning, HistAccumulator, TestOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, SystemTime};

/// The toggles of `tb.s0` (four bits) at the times 10, 20, ..., 80.
const TOGGLES: [u32; 8] = [1, 2, 3, 4, 2, 2, 4, 3];

/// The steps at which `tb.s1` (one bit) toggles. They lie in the segments of class 1.
const S1_STEPS: [usize; 4] = [2, 3, 6, 7];

/// Four segments of two time points each, with the labels 0, 1, 0, 1.
const MARKERS: &str = "[[10, 30, 0], [30, 50, 1], [50, 70, 0], [70, 90, 1]]";
const LABELS: [u16; 4] = [0, 1, 0, 1];

/// The traces of `tb.s0` alone, and of `tb.s0` and `tb.s1` together, for the default `TOGGLES`.
fn s0_traces() -> Array2<f32> {
    Array2::from_shape_vec((4, 2), vec![1., 2., 3., 4., 2., 2., 4., 3.]).unwrap()
}
fn s0_s1_traces() -> Array2<f32> {
    Array2::from_shape_vec((4, 2), vec![1., 2., 4., 5., 2., 2., 5., 4.]).unwrap()
}

/// Traces for `traces.npz` that no selection of the waveform gives.
fn cached_traces() -> Array2<f32> {
    Array2::from_shape_vec((4, 2), vec![10., 20., 30., 40., 50., 60., 70., 90.]).unwrap()
}

/// The format of the waveform file of a batch.
#[derive(Clone, Copy, Debug)]
enum Format {
    Vcd,
    Fst,
}

const FORMATS: [Format; 2] = [Format::Vcd, Format::Fst];

impl Format {
    fn file_name(self) -> &'static str {
        match self {
            Format::Vcd => "tvla.vcd",
            Format::Fst => "tvla.fst",
        }
    }
}

/// A temporary batch directory.
struct Batch {
    dir: tempfile::TempDir,
}

impl Batch {
    /// A batch with a VCD waveform. See `new_in`.
    fn new(toggles: [u32; 8], clock_period: Option<u64>) -> Batch {
        Batch::new_in(Format::Vcd, toggles, clock_period)
    }

    /// Writes the waveform (older than any file that `write_cache` writes) and `meta.json`. The
    /// signals are `tb.s0`, `tb.s1`, and `aux.a`. Only `tb.s0` and `tb.s1` toggle.
    fn new_in(format: Format, toggles: [u32; 8], clock_period: Option<u64>) -> Batch {
        let dir = tempfile::tempdir().unwrap();
        let mut fx = Fixture::flat(&[4, 1, 1]);
        fx.signals[2].scope = "aux".into();
        fx.signals[2].name = "a".into();
        let mut s0 = 0u32;
        for (step, k) in toggles.iter().enumerate() {
            s0 ^= (1 << k) - 1;
            let mut changes = vec![(0, format!("{s0:04b}"))];
            if S1_STEPS.contains(&step) {
                changes.push((1, if step % 2 == 0 { "1" } else { "0" }.into()));
            }
            fx.steps.push((10 * (step as u64 + 1), changes));
        }
        let waveform = dir.path().join(format.file_name());
        match format {
            Format::Vcd => write_vcd(&waveform, &fx),
            Format::Fst => write_fst(&waveform, &fx),
        }
        set_modified(&waveform, hours_ago(1));
        let clock = clock_period.map_or(String::new(), |c| format!(r#""clock_period": {c}, "#));
        std::fs::write(
            dir.path().join("meta.json"),
            format!(
                r#"{{"trace_filename": "{}", {clock}"markers": {MARKERS}}}"#,
                format.file_name()
            ),
        )
        .unwrap();
        Batch { dir }
    }

    fn meta(&self) -> PathBuf {
        self.dir.path().join("meta.json")
    }

    fn npz(&self) -> PathBuf {
        self.dir.path().join("traces.npz")
    }

    /// Writes `traces.npz` with the cached traces. A fresh file is newer than the waveform. A
    /// stale file is older.
    fn write_cache(&self, fresh: bool) {
        self.write_cache_with(&[0, 1, 2, 3], true, fresh.then(SystemTime::now));
    }

    /// Writes `traces.npz` with the entries `trace_<i>` for the given `indices`, in this order
    /// in the archive. `labels` come first or last. `time` is the modification time, or two
    /// hours ago if it is `None`.
    fn write_cache_with(&self, indices: &[usize], labels_last: bool, time: Option<SystemTime>) {
        let traces = cached_traces();
        let labels = Array1::from_vec(LABELS.to_vec());
        let mut npz = NpzWriter::new(std::fs::File::create(self.npz()).unwrap());
        if !labels_last {
            npz.add_array("labels", &labels).unwrap();
        }
        for &i in indices {
            npz.add_array(format!("trace_{i}"), &traces.row(i)).unwrap();
        }
        if labels_last {
            npz.add_array("labels", &labels).unwrap();
        }
        npz.finish().unwrap();
        set_modified(&self.npz(), time.unwrap_or_else(|| hours_ago(2)));
    }

    /// Writes a fresh `traces.npz` with these traces and labels.
    fn write_cache_data(&self, traces: &Array2<f32>, labels: &[u16]) {
        let mut npz = NpzWriter::new(std::fs::File::create(self.npz()).unwrap());
        for (i, row) in traces.outer_iter().enumerate() {
            npz.add_array(format!("trace_{i}"), &row).unwrap();
        }
        npz.add_array("labels", &Array1::from_vec(labels.to_vec()))
            .unwrap();
        npz.finish().unwrap();
        set_modified(&self.npz(), SystemTime::now());
    }

    /// The traces and labels in `traces.npz`.
    fn read_cache(&self) -> (Array2<f32>, Array1<u16>) {
        let mut npz = NpzReader::new(std::fs::File::open(self.npz()).unwrap()).unwrap();
        let rows: Vec<Array1<f32>> = (0..4)
            .map(|i| npz.by_name(&format!("trace_{i}")).unwrap())
            .collect();
        let traces = Array2::from_shape_vec((4, 2), rows.into_iter().flatten().collect()).unwrap();
        (traces, npz.by_name("labels").unwrap())
    }

    /// Runs `tvla` with `args` after the fixed arguments. Returns the output of a run that
    /// fails as well.
    fn run_any(&self, args: &[&str]) -> Output {
        self.run_order("1", args)
    }

    /// Like `run_any`, with the highest t-test order `order`.
    fn run_order(&self, order: &str, args: &[&str]) -> Output {
        let out = self.dir.path().join("out");
        Command::new(env!("CARGO_BIN_EXE_tvla"))
            .env_remove("RUST_LOG")
            .arg("--meta-json")
            .arg(self.meta())
            .arg("--ttest-output-dir")
            .arg(&out)
            .args(["--plot=false", "-d", order])
            .args(args)
            .output()
            .unwrap()
    }

    /// Runs `tvla` with `args` after the fixed arguments. The run must succeed.
    fn run(&self, args: &[&str]) -> Output {
        let output = self.run_any(args);
        assert!(output.status.success(), "{}", text(&output.stderr));
        output
    }

    /// The first-order t-values that the last run wrote.
    fn t_values(&self) -> Vec<f64> {
        let file = std::fs::File::open(self.dir.path().join("out/t_values.npz")).unwrap();
        let t: Array2<f64> = NpzReader::new(file).unwrap().by_name("t_values").unwrap();
        assert_eq!(t.nrows(), 1);
        t.row(0).to_vec()
    }
}

fn hours_ago(hours: u64) -> SystemTime {
    SystemTime::now() - Duration::from_secs(3600 * hours)
}

fn set_modified(path: &Path, time: SystemTime) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(time)
        .unwrap();
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The Welch t-value of each sample, from the traces of the labels 0 and 1. Like `scalib`, it
/// subtracts the mean of class 1 from the mean of class 0, and it divides the sum of squares by
/// the number of traces of a class (not by that number minus one).
fn welch_t(traces: &Array2<f32>, labels: &[u16]) -> Vec<f64> {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let variance = |v: &[f64]| {
        let m = mean(v);
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
    };
    (0..traces.ncols())
        .map(|sample| {
            let class = |label: u16| -> Vec<f64> {
                labels
                    .iter()
                    .zip(traces.column(sample))
                    .filter(|(l, _)| **l == label)
                    .map(|(_, v)| f64::from(*v))
                    .collect()
            };
            let (a, b) = (class(0), class(1));
            (mean(&a) - mean(&b))
                / (variance(&a) / a.len() as f64 + variance(&b) / b.len() as f64).sqrt()
        })
        .collect()
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
    }
}

#[test]
fn the_expected_traces_differ_so_that_the_tests_can_tell_the_sources_apart() {
    let t = |traces: &Array2<f32>| welch_t(traces, &LABELS);
    for (a, b) in [
        (t(&s0_traces()), t(&s0_s1_traces())),
        (t(&s0_traces()), t(&cached_traces())),
        (t(&s0_s1_traces()), t(&cached_traces())),
    ] {
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() > 0.1));
    }
}

#[test]
fn rules_neither_read_nor_overwrite_a_fresh_traces_npz() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        batch.write_cache(true);
        let before = std::fs::read(batch.npz()).unwrap();
        // `--exclude` after `--include`: the order decides, so `tb.s1` stays out of the selection.
        let output = batch.run(&["--include", "scope:tb", "--exclude", "signal:tb.s1"]);
        let stdout = text(&output.stdout);
        assert!(stdout.contains("Computing power traces from"), "{stdout}");
        assert!(!stdout.contains("Using existing"), "{stdout}");
        assert_eq!(std::fs::read(batch.npz()).unwrap(), before, "traces.npz");
        assert_close(&batch.t_values(), &welch_t(&s0_traces(), &LABELS));
        let stderr = text(&output.stderr);
        assert!(stderr.contains("1 signals selected"), "{stderr}");
        assert!(!stderr.contains("WARN"), "{stderr}");
        assert!(!stderr.contains("top-level scopes"), "{stderr}");
    }
}

#[test]
fn rules_do_not_overwrite_a_stale_traces_npz() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        batch.write_cache(false);
        let before = std::fs::read(batch.npz()).unwrap();
        let output = batch.run(&["--include", "scope:tb", "--exclude", "signal:tb.s1"]);
        assert!(!text(&output.stdout).contains("Saving traces"));
        assert_eq!(std::fs::read(batch.npz()).unwrap(), before, "traces.npz");
        assert_close(&batch.t_values(), &welch_t(&s0_traces(), &LABELS));
    }
}

#[test]
fn without_rules_a_fresh_traces_npz_is_used() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        batch.write_cache(true);
        let before = std::fs::read(batch.npz()).unwrap();
        let output = batch.run(&[]);
        assert!(text(&output.stdout).contains("Using existing traces and labels from"));
        assert_eq!(std::fs::read(batch.npz()).unwrap(), before, "traces.npz");
        assert_close(&batch.t_values(), &welch_t(&cached_traces(), &LABELS));
    }
}

#[test]
fn without_rules_a_stale_traces_npz_is_replaced() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        batch.write_cache(false);
        let output = batch.run(&[]);
        let stdout = text(&output.stdout);
        assert!(stdout.contains("Computing power traces from"), "{stdout}");
        assert!(!stdout.contains("Using existing"), "{stdout}");
        let (traces, labels) = batch.read_cache();
        assert_eq!(traces, s0_s1_traces());
        assert_eq!(labels.to_vec(), LABELS);
        assert_close(&batch.t_values(), &welch_t(&s0_s1_traces(), &LABELS));
    }
}

#[test]
fn use_existing_false_computes_the_traces_even_if_the_cache_is_fresh() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        batch.write_cache(true);
        let output = batch.run(&["--use-existing=false"]);
        assert!(text(&output.stdout).contains("Computing power traces from"));
        assert_eq!(batch.read_cache().0, s0_s1_traces());
    }
}

#[test]
fn the_run_reports_unmatched_rules_several_top_scopes_and_dropped_sampling() {
    for format in FORMATS {
        // Sampling at multiples of 20 keeps the time points 20, 40, 60, and 80: 7 of 23 toggles.
        let batch = Batch::new_in(format, [4, 1, 4, 2, 4, 2, 4, 2], Some(20));
        let output = batch.run(&[
            "--include",
            "signal:tb.s0",
            "--include",
            "signal:aux.a",
            "--exclude",
            "signal:tb.nope",
        ]);
        let stderr = text(&output.stderr);
        assert!(stderr.contains("keeps only 7 of 23 toggles"), "{stderr}");
        assert!(
            stderr.contains("the selection spans 2 top-level scopes: aux, tb"),
            "{stderr}"
        );
        assert!(
            stderr.contains("the rule -signal:tb.nope matches no signal"),
            "{stderr}"
        );
    }
}

#[test]
fn sampling_that_keeps_half_of_the_toggles_gives_no_warning() {
    // The same sampling keeps 11 of 21 toggles of the default toggles of `tb.s0`.
    let batch = Batch::new(TOGGLES, Some(20));
    let output = batch.run(&["--include", "signal:tb.s0"]);
    let stderr = text(&output.stderr);
    assert!(stderr.contains("1 signals selected"), "{stderr}");
    assert!(!stderr.contains("keeps only"), "{stderr}");
}

#[test]
fn list_signals_prints_only_data_lines_on_stdout() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        let output = batch.run(&[
            "--list-signals",
            "--include",
            "scope:tb",
            "--exclude",
            "signal:tb.s1",
            "--exclude",
            "signal:tb.nope",
        ]);
        let lines: Vec<String> = text(&output.stdout).lines().map(String::from).collect();
        assert_eq!(lines, ["0\tyes\ttb.s0", "1\tno\ttb.s1", "2\tno\taux.a"]);
        let stderr = text(&output.stderr);
        assert!(
            stderr.contains("1 of 3 selectable signals are selected"),
            "{stderr}"
        );
        assert!(
            stderr.contains("warning: the rule -signal:tb.nope matches no signal"),
            "{stderr}"
        );
    }
}

/// Asserts that a run failed with a message and without a panic.
fn assert_fails_with(output: &Output, expected: &[&str]) {
    // miette wraps long lines and draws a bar at the start of each wrapped line.
    let stderr = text(&output.stderr)
        .replace('│', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!output.status.success(), "the run succeeded: {stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(!stderr.contains("RUST_BACKTRACE"), "{stderr}");
    for part in expected {
        assert!(stderr.contains(part), "missing {part:?} in: {stderr}");
    }
}

/// A shortened path text: miette wraps long lines, so tests look for the file name only.
fn name_of(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().into_owned()
}

#[test]
fn an_empty_selection_is_an_error_that_lists_the_unmatched_rules() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        let output = batch.run_any(&["--include", "signal:does.not.exist"]);
        assert_fails_with(&output, &["selects no signals", "+signal:does.not.exist"]);
    }
}

#[test]
fn missing_meta_options_are_a_usage_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .output()
        .unwrap();
    assert_fails_with(&output, &["--meta-json", "--meta-list"]);
    assert!(!text(&output.stderr).contains("NPZ"));
}

#[test]
fn a_missing_metadata_file_is_an_error_that_names_the_file() {
    let batch = Batch::new(TOGGLES, None);
    let missing = batch.dir.path().join("no_such_meta.json");
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-json")
        .arg(&missing)
        .args(["--plot=false", "--ttest-output-dir"])
        .arg(batch.dir.path().join("out"))
        .output()
        .unwrap();
    assert_fails_with(&output, &["no_such_meta.json"]);
}

#[test]
fn a_missing_batch_in_the_meta_list_is_an_error_that_names_the_file() {
    let batch = Batch::new(TOGGLES, None);
    let list = batch.dir.path().join("meta.list");
    std::fs::write(&list, "meta.json\ngone/meta.json.gz\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-list")
        .arg(&list)
        .args(["--plot=false", "--ttest-output-dir"])
        .arg(batch.dir.path().join("out"))
        .output()
        .unwrap();
    assert_fails_with(&output, &["meta.json.gz"]);
}

#[test]
fn a_missing_waveform_is_an_error_that_names_the_file() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        let waveform = batch.dir.path().join(format.file_name());
        std::fs::remove_file(&waveform).unwrap();
        let output = batch.run_any(&[]);
        assert_fails_with(&output, &[&name_of(&waveform)]);
    }
}

#[test]
fn bad_metadata_is_an_error_that_names_the_file() {
    let batch = Batch::new(TOGGLES, None);
    std::fs::write(batch.meta(), "{ not json").unwrap();
    assert_fails_with(&batch.run_any(&[]), &["meta.json", "not valid JSON"]);
    let gz = batch.dir.path().join("meta.json.gz");
    std::fs::write(&gz, "not gzip").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-json")
        .arg(&gz)
        .args(["--plot=false", "--ttest-output-dir"])
        .arg(batch.dir.path().join("out"))
        .output()
        .unwrap();
    assert_fails_with(&output, &["meta.json.gz"]);
}

#[test]
fn a_batch_with_one_trace_is_an_error() {
    let batch = Batch::new(TOGGLES, None);
    std::fs::write(
        batch.meta(),
        r#"{"trace_filename": "tvla.vcd", "markers": [[10, 30, 0]]}"#,
    )
    .unwrap();
    assert_fails_with(&batch.run_any(&[]), &["at least two traces", "meta.json"]);
}

#[test]
fn traces_without_any_toggle_do_not_panic() {
    // `aux.a` never toggles, so all traces are zero and every t-value is NaN.
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        let output = batch.run(&["--include", "signal:aux.a"]);
        let stderr = text(&output.stderr);
        assert!(stderr.contains("not finite"), "{stderr}");
        assert_eq!(stderr.matches("not finite").count(), 1, "{stderr}");
        assert!(batch.t_values().iter().all(|t| !t.is_finite()));
    }
}

#[test]
fn a_second_order_test_with_two_traces_per_class_does_not_panic() {
    let batch = Batch::new(TOGGLES, None);
    let output = batch.run_order("2", &[]);
    assert!(output.status.success(), "{}", text(&output.stderr));
}

#[test]
fn the_cache_is_read_in_the_order_of_the_trace_indices() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        batch.write_cache_with(&[3, 1, 0, 2], false, Some(SystemTime::now()));
        let output = batch.run(&[]);
        assert!(text(&output.stdout).contains("Using existing traces"));
        assert_close(&batch.t_values(), &welch_t(&cached_traces(), &LABELS));
    }
}

#[test]
fn a_cache_with_missing_or_extra_trace_indices_is_an_error() {
    let batch = Batch::new(TOGGLES, None);
    // The index 2 is missing, and the index 4 has no place.
    batch.write_cache_with(&[0, 1, 3], true, Some(SystemTime::now()));
    assert_fails_with(&batch.run_any(&[]), &["traces.npz", "trace_2"]);
    // Four labels but three traces.
    batch.write_cache_with(&[0, 1, 2], true, Some(SystemTime::now()));
    assert_fails_with(&batch.run_any(&[]), &["traces.npz", "labels"]);
}

#[test]
fn a_cache_without_traces_is_an_error() {
    let batch = Batch::new(TOGGLES, None);
    batch.write_cache_with(&[], true, Some(SystemTime::now()));
    assert_fails_with(&batch.run_any(&[]), &["traces.npz", "no traces"]);
}

#[test]
fn a_metadata_file_newer_than_the_cache_causes_a_recompute() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        // The cache is newer than the waveform (one hour old) and older than the metadata file.
        batch.write_cache_with(
            &[0, 1, 2, 3],
            true,
            Some(hours_ago(0) - Duration::from_secs(1800)),
        );
        let before = std::fs::read(batch.npz()).unwrap();
        let output = batch.run(&[]);
        assert!(text(&output.stdout).contains("Computing power traces from"));
        assert_ne!(std::fs::read(batch.npz()).unwrap(), before, "traces.npz");
        assert_eq!(batch.read_cache().0, s0_s1_traces());
        // An old metadata file does not cause a recompute.
        set_modified(&batch.meta(), hours_ago(3));
        batch.write_cache(true);
        let before = std::fs::read(batch.npz()).unwrap();
        assert!(text(&batch.run(&[]).stdout).contains("Using existing traces"));
        assert_eq!(std::fs::read(batch.npz()).unwrap(), before, "traces.npz");
    }
}

#[test]
fn a_cache_is_used_if_the_waveform_is_gone_even_if_the_metadata_file_is_newer() {
    let batch = Batch::new(TOGGLES, None);
    std::fs::remove_file(batch.dir.path().join("tvla.vcd")).unwrap();
    batch.write_cache_with(
        &[0, 1, 2, 3],
        true,
        Some(hours_ago(0) - Duration::from_secs(1800)),
    );
    let output = batch.run(&[]);
    assert!(text(&output.stdout).contains("Using existing traces"));
}

#[test]
fn a_marker_accepts_other_u16_labels() {
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        std::fs::write(
            batch.meta(),
            format!(
                r#"{{"trace_filename": "{}", "markers": [[10, 30, 0], [30, 50, 1], [50, 70, 2]]}}"#,
                format.file_name()
            ),
        )
        .unwrap();
        let output = batch.run_any(&[]);
        assert!(output.status.success(), "{}", text(&output.stderr));
    }
}

#[test]
fn a_cache_accepts_other_u16_labels() {
    let batch = Batch::new(TOGGLES, None);
    let mut npz = NpzWriter::new(std::fs::File::create(batch.npz()).unwrap());
    for (i, row) in cached_traces().outer_iter().enumerate() {
        npz.add_array(format!("trace_{i}"), &row).unwrap();
    }
    npz.add_array("labels", &Array1::from_vec(vec![0u16, 1, 0, 2]))
        .unwrap();
    npz.finish().unwrap();
    set_modified(&batch.npz(), SystemTime::now());
    assert!(batch.run_any(&[]).status.success());
}

#[test]
fn order_zero_is_a_usage_error() {
    let batch = Batch::new(TOGGLES, None);
    let output = batch.run_order("0", &[]);
    assert_fails_with(&output, &["invalid value '0'"]);
}

/// Runs `tvla` with the default `--plot` (true). `PATH` has no program, so a browser driver
/// (chromedriver, geckodriver) cannot be found or started. Returns the output directory.
fn run_with_plots(batch: &Batch, order: &str, args: &[&str]) -> (Output, PathBuf) {
    let out = batch.dir.path().join("plots");
    let no_programs = batch.dir.path().join("no-programs");
    std::fs::create_dir_all(&no_programs).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .env("PATH", &no_programs)
        .arg("--meta-json")
        .arg(batch.meta())
        .arg("--ttest-output-dir")
        .arg(&out)
        .args(["-d", order])
        .args(args)
        .output()
        .unwrap();
    (output, out)
}

/// The files that `tvla` writes with the default `--plot` for the t-test orders 1..=`order`.
fn plot_files(order: usize) -> Vec<String> {
    let mut files = vec!["all_t_values.html".to_string()];
    for d in 1..=order {
        for ext in ["html", "svg", "json"] {
            files.push(format!("t_test_d{d}.{ext}"));
        }
    }
    for ext in ["html", "svg", "json"] {
        files.push(format!("max_t_values.{ext}"));
    }
    files
}

fn assert_plot_files(out: &Path, order: usize) {
    for name in plot_files(order) {
        let path = out.join(&name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!bytes.is_empty(), "{name} is empty");
        let text = String::from_utf8_lossy(&bytes);
        match name.rsplit('.').next().unwrap() {
            "svg" => assert!(text.contains("<svg"), "{name} is not an SVG file"),
            "html" => assert!(text.contains("plotly"), "{name} has no plotly"),
            "json" => {
                let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert!(json["data"].is_array(), "{name} has no data");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn the_default_plots_are_written_without_a_browser_driver() {
    let batch = Batch::new(TOGGLES, None);
    let (output, out) = run_with_plots(&batch, "2", &[]);
    let stderr = text(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert_plot_files(&out, 2);
    assert!(out.join("t_values.npz").exists());
    assert!(!stderr.to_lowercase().contains("driver"), "{stderr}");
}

#[test]
fn plots_of_traces_without_any_toggle_do_not_panic() {
    // `aux.a` never toggles, so all traces are zero, every t-value is NaN, and so is every
    // max |t|. The old code panicked when it printed the maximum.
    for format in FORMATS {
        let batch = Batch::new_in(format, TOGGLES, None);
        let (output, out) = run_with_plots(&batch, "2", &["--include", "signal:aux.a"]);
        let stderr = text(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert_plot_files(&out, 2);
    }
}

// ---------------------------------------------------------------------------------------------
// Chi-squared output, streaming windows, and integer traces
// ---------------------------------------------------------------------------------------------

/// Integer traces with a leak: `traces` samples with `n` rows each. The labels alternate. Sample 2
/// has a different spread in class 1 (a leak that the mean does not show). `seed` changes the
/// values.
fn leaky_traces(n: usize, samples: usize, seed: u64) -> (Array2<f32>, Vec<u16>) {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let labels: Vec<u16> = (0..n).map(|i| (i % 2) as u16).collect();
    let traces = Array2::from_shape_fn((n, samples), |(i, j)| {
        let base = (next() % 8 + next() % 8) as f32;
        if j == 2 && labels[i] == 1 {
            base * 2.0
        } else {
            base
        }
    });
    (traces, labels)
}

fn read_f64(path: &Path, name: &str) -> Array2<f64> {
    NpzReader::new(std::fs::File::open(path).unwrap())
        .unwrap()
        .by_name(name)
        .unwrap()
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn chi2_npz_equals_a_direct_histogram_computation() {
    let batch = Batch::new(TOGGLES, None);
    let (traces, labels) = leaky_traces(300, 5, 1);
    batch.write_cache_data(&traces, &labels);
    batch.run_order("2", &[]);

    let mut hist = HistAccumulator::new(5, Binning::Exact);
    hist.update(traces.view(), Array1::from_vec(labels).view())
        .unwrap();
    let want = hist.test_pair(0, 1, &TestOptions::default()).unwrap();

    let file = batch.dir.path().join("out/chi2.npz");
    let mut npz = NpzReader::new(std::fs::File::open(file).unwrap()).unwrap();
    let f64s = |npz: &mut NpzReader<std::fs::File>, name: &str| -> Vec<f64> {
        let a: Array1<f64> = npz.by_name(name).unwrap();
        a.to_vec()
    };
    let u32s = |npz: &mut NpzReader<std::fs::File>, name: &str| -> Vec<u32> {
        let a: Array1<u32> = npz.by_name(name).unwrap();
        a.to_vec()
    };
    let got_p = f64s(&mut npz, "neg_log10_p");
    let got_s = f64s(&mut npz, "statistic");
    let got_min = f64s(&mut npz, "min_expected");
    let got_dof = u32s(&mut npz, "dof");
    let got_columns = u32s(&mut npz, "columns");
    let got_merged = u32s(&mut npz, "merged");
    let got_n: Array1<u64> = npz.by_name("n").unwrap();
    assert_eq!(got_p.len(), 5);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(got_p[i].to_bits(), w.neg_log10_p.to_bits());
        assert_eq!(got_s[i].to_bits(), w.statistic.to_bits());
        assert_eq!(got_min[i].to_bits(), w.min_expected.to_bits());
        assert_eq!(got_dof[i], w.dof);
        assert_eq!(got_columns[i], w.columns);
        assert_eq!(got_merged[i], w.merged);
        assert_eq!(got_n[i], w.n);
    }
    // The planted leak is in sample 2.
    assert!(got_p[2] > 5.0, "{got_p:?}");
}

#[test]
fn chi2_false_writes_no_chi2_files() {
    let batch = Batch::new(TOGGLES, None);
    let (output, out) = run_with_plots(&batch, "1", &["--chi2=false"]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(out.join("t_values.npz").exists());
    assert!(out.join("max_t_values.html").exists());
    let names: Vec<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.iter().all(|n| !n.contains("chi2")),
        "unexpected files: {names:?}"
    );
    assert!(
        !text(&output.stderr).contains("chi2"),
        "{}",
        text(&output.stderr)
    );
}

#[test]
fn chi2_is_on_by_default_and_writes_data_and_plots() {
    let batch = Batch::new(TOGGLES, None);
    let (output, out) = run_with_plots(&batch, "1", &[]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    for stem in ["chi2", "max_chi2"] {
        for ext in ["html", "svg", "json"] {
            let path = out.join(format!("{stem}.{ext}"));
            assert!(std::fs::metadata(&path).unwrap().len() > 0, "{path:?}");
        }
    }
    assert!(out.join("chi2.npz").exists());
    // With `--plot=false` the data file is still written, and the plots are not.
    batch.run(&[]);
    let out = batch.dir.path().join("out");
    assert!(out.join("chi2.npz").exists());
    assert!(!out.join("chi2.html").exists());
}

/// Three cached batches with different sizes, and the path of their meta list.
fn three_batches() -> (Vec<Batch>, PathBuf, tempfile::TempDir) {
    let batches: Vec<Batch> = [(60, 11), (40, 12), (50, 13)]
        .iter()
        .map(|&(n, seed)| {
            let batch = Batch::new(TOGGLES, None);
            let (traces, labels) = leaky_traces(n, 5, seed);
            batch.write_cache_data(&traces, &labels);
            batch
        })
        .collect();
    let list_dir = tempfile::tempdir().unwrap();
    let list = list_dir.path().join("meta.list");
    let lines: Vec<String> = batches
        .iter()
        .map(|b| b.meta().to_string_lossy().into_owned())
        .collect();
    std::fs::write(&list, lines.join("\n")).unwrap();
    (batches, list, list_dir)
}

fn run_list(list: &Path, out: &Path, threads: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-list")
        .arg(list)
        .arg("--ttest-output-dir")
        .arg(out)
        .args(["--plot=false", "-d", "3", "--num-threads", threads])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", text(&output.stderr));
}

#[test]
fn the_window_size_does_not_change_the_results_bit_for_bit() {
    let (_batches, list, _keep) = three_batches();
    let out = tempfile::tempdir().unwrap();
    let mut results = Vec::new();
    // 1 thread: windows of one batch. 2 threads: a window of two, then one. 4 threads: one window.
    for threads in ["1", "2", "4"] {
        let dir = out.path().join(threads);
        run_list(&list, &dir, threads);
        results.push((
            bits(&read_f64(&dir.join("t_values.npz"), "t_values")),
            bits(&read_f64_1d(&dir.join("chi2.npz"), "neg_log10_p")),
            bits(&read_f64_1d(&dir.join("chi2.npz"), "statistic")),
        ));
    }
    assert!(results[0].0.iter().any(|&b| f64::from_bits(b).abs() > 1.0));
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[2]);
}

fn read_f64_1d(path: &Path, name: &str) -> Array2<f64> {
    let a: Array1<f64> = NpzReader::new(std::fs::File::open(path).unwrap())
        .unwrap()
        .by_name(name)
        .unwrap();
    a.insert_axis(ndarray::Axis(0))
}

#[test]
fn several_batches_give_the_same_bits_as_one_batch_with_all_traces() {
    let (_batches, list, _keep) = three_batches();
    let out = tempfile::tempdir().unwrap();
    run_list(&list, &out.path().join("many"), "2");

    let parts = [(60, 11), (40, 12), (50, 13)].map(|(n, seed)| leaky_traces(n, 5, seed));
    let views: Vec<_> = parts.iter().map(|(t, _)| t.view()).collect();
    let all = ndarray::concatenate(ndarray::Axis(0), &views).unwrap();
    let labels: Vec<u16> = parts.iter().flat_map(|(_, l)| l.clone()).collect();
    let one = Batch::new(TOGGLES, None);
    one.write_cache_data(&all, &labels);
    let output = one.run_order("3", &[]);
    assert!(output.status.success());

    let many = read_f64(&out.path().join("many/t_values.npz"), "t_values");
    let single = read_f64(&one.dir.path().join("out/t_values.npz"), "t_values");
    assert_eq!(bits(&many), bits(&single));
    let many = read_f64_1d(&out.path().join("many/chi2.npz"), "neg_log10_p");
    let single = read_f64_1d(&one.dir.path().join("out/chi2.npz"), "neg_log10_p");
    assert_eq!(bits(&many), bits(&single));
}

#[test]
fn traces_that_are_not_integers_are_an_error_that_names_the_file() {
    let batch = Batch::new(TOGGLES, None);
    let mut traces = cached_traces();
    traces[[1, 0]] = 30.5;
    batch.write_cache_data(&traces, &LABELS);
    let output = batch.run_any(&[]);
    assert_fails_with(&output, &["traces.npz", "integer"]);
    assert!(!batch.dir.path().join("out/t_values.npz").exists());
}

#[test]
fn the_summary_reports_both_tests_and_the_memory() {
    let batch = Batch::new(TOGGLES, None);
    let (traces, labels) = leaky_traces(300, 5, 1);
    batch.write_cache_data(&traces, &labels);
    let output = batch.run_order("2", &[]);
    let stderr = text(&output.stderr);
    for part in [
        "d=1: max |t|",
        "d=2: max |t|",
        "above 4.5",
        "Bonferroni",
        "m = 10",
        "chi2: max -log10(p)",
        "min expected count",
        "memory",
    ] {
        assert!(stderr.contains(part), "missing {part:?} in:\n{stderr}");
    }
    let output = batch.run_order("2", &["--chi2=false"]);
    let stderr = text(&output.stderr);
    assert!(stderr.contains("d=1: max |t|"), "{stderr}");
    assert!(!stderr.contains("chi2"), "{stderr}");
}
