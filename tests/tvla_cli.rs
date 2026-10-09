//! Runs the `tvla` binary on a tiny batch: a VCD waveform, a legacy `meta.json`, and a
//! `traces.npz` that differs from the traces of the waveform. The tests check when `tvla` reads
//! and writes `traces.npz`, what it prints, and how `--list-signals` splits its output.

mod common;

use common::*;
use ndarray::{Array1, Array2};
use ndarray_npz::{NpzReader, NpzWriter};
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
        let mut npz = NpzWriter::new(std::fs::File::create(self.npz()).unwrap());
        for (i, row) in cached_traces().outer_iter().enumerate() {
            npz.add_array(format!("trace_{i}"), &row).unwrap();
        }
        npz.add_array("labels", &Array1::from_vec(LABELS.to_vec()))
            .unwrap();
        npz.finish().unwrap();
        set_modified(
            &self.npz(),
            if fresh {
                SystemTime::now()
            } else {
                hours_ago(2)
            },
        );
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
        let out = self.dir.path().join("out");
        Command::new(env!("CARGO_BIN_EXE_tvla"))
            .env_remove("RUST_LOG")
            .arg("--meta-json")
            .arg(self.meta())
            .arg("--ttest-output-dir")
            .arg(&out)
            .args(["--plot=false", "-d", "1"])
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
