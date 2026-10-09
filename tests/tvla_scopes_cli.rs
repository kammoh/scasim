//! Runs the `tvla` binary with `--per-scope` on batches with a planted leak in one scope.

mod common;

use common::*;
use ndarray::{Array1, Array2};
use ndarray_npz::NpzReader;
use std::path::Path;
use std::process::{Command, Output};

fn tvla(meta: &Path, out: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-json")
        .arg(meta)
        .arg("--ttest-output-dir")
        .arg(out)
        .args(["-d", "2"])
        .args(args)
        .output()
        .unwrap()
}

/// The error output, with the line breaks of the error report joined.
fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).replace("\n  │ ", " ")
}

/// The rows of `channels.tsv` without the header, each split at the tabs.
fn tsv(out: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(out.join("channels.tsv")).unwrap();
    let mut lines = text.lines();
    let header = lines
        .next()
        .unwrap()
        .split('\t')
        .map(String::from)
        .collect();
    let rows = lines
        .map(|l| l.split('\t').map(String::from).collect())
        .collect();
    (header, rows)
}

fn column(header: &[String], name: &str) -> usize {
    header.iter().position(|h| h == name).unwrap()
}

fn read_t(path: &Path, name: &str) -> Array2<f64> {
    NpzReader::new(std::fs::File::open(path).unwrap())
        .unwrap()
        .by_name(name)
        .unwrap()
}

const EDGES: [&str; 4] = ["--clock", "tb.clk", "--include", "scope:tb.dut"];

#[test]
fn the_scope_with_the_planted_leak_ranks_first() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let mut args = vec!["--plot=false", "--per-scope", "tb.dut"];
    args.extend(EDGES);
    let output = tvla(&batch.meta, &out, &args);
    assert!(output.status.success(), "{}", stderr(&output));
    let (header, rows) = tsv(&out);
    assert_eq!(
        header,
        [
            "rank",
            "channel",
            "handles",
            "infinite_t",
            "max_abs_t_d1",
            "sample_d1",
            "max_abs_t_d2",
            "sample_d2",
            "max_neg_log10_p",
            "sample_chi2"
        ]
    );
    assert_eq!(rows.len(), 2);
    let (t1, s1) = (
        column(&header, "max_abs_t_d1"),
        column(&header, "sample_d1"),
    );
    // Rank 1: the scope with the leak, at the planted sample.
    assert_eq!(&rows[0][..3], ["1", "tb.dut.a", "1"]);
    assert!(rows[0][t1].parse::<f64>().unwrap() > 4.5, "{rows:?}");
    assert_eq!(rows[0][s1], "2");
    // Rank 2: the other scope has no exceedance.
    assert_eq!(&rows[1][..3], ["2", "tb.dut.b", "1"]);
    assert!(rows[1][t1].parse::<f64>().unwrap() < 4.5, "{rows:?}");
    let t2 = column(&header, "max_abs_t_d2");
    assert!(rows[1][t2].parse::<f64>().unwrap() < 4.5, "{rows:?}");
    let chi2 = column(&header, "max_neg_log10_p");
    assert!(rows[0][chi2].parse::<f64>().unwrap() > 5.0, "{rows:?}");
    let log = stderr(&output);
    assert!(
        log.contains("best channels by max |t|: 1. tb.dut.a |t|"),
        "{log}"
    );
    assert!(log.contains("accumulators of the channels"), "{log}");
}

#[test]
fn the_files_of_the_channels_and_the_unchanged_outputs() {
    let batch = write_leak_batch(&LeakSpec::default());
    let plain = batch.dir.path().join("plain");
    let scoped = batch.dir.path().join("scoped");
    let mut args = vec!["--plot=false"];
    args.extend(EDGES);
    assert!(tvla(&batch.meta, &plain, &args).status.success());
    args.extend(["--per-scope", "tb.dut"]);
    assert!(tvla(&batch.meta, &scoped, &args).status.success());
    // The outputs of the whole selection do not change, bit for bit.
    let bits = |dir: &Path, file: &str, name: &str| -> Vec<u64> {
        read_t(&dir.join(file), name)
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(
        bits(&plain, "t_values.npz", "t_values"),
        bits(&scoped, "t_values.npz", "t_values")
    );
    let chi2_bits = |dir: &Path| -> Vec<u64> {
        let file = std::fs::File::open(dir.join("chi2.npz")).unwrap();
        let column: Array1<f64> = NpzReader::new(file)
            .unwrap()
            .by_name("neg_log10_p")
            .unwrap();
        column.iter().map(|v| v.to_bits()).collect()
    };
    assert_eq!(chi2_bits(&plain), chi2_bits(&scoped));
    assert!(!plain.join("channels.tsv").exists());
    // The channel files: names in index order, one array each.
    let names = std::fs::read_to_string(scoped.join("channels.txt")).unwrap();
    assert_eq!(names, "tb.dut.a\ntb.dut.b\n");
    let file = scoped.join("t_values_channels.npz");
    let (a, b) = (read_t(&file, "t_0"), read_t(&file, "t_1"));
    assert_eq!(a.dim(), (2, 6));
    assert_eq!(b.dim(), (2, 6));
    assert!(a[[0, 2]].abs() > 4.5 && b[[0, 2]].abs() < 4.5);
    let chi2: Array1<f64> =
        NpzReader::new(std::fs::File::open(scoped.join("chi2_channels.npz")).unwrap())
            .unwrap()
            .by_name("chi2_0")
            .unwrap();
    assert_eq!(chi2.len(), 6);
    // The channel with the planted leak has the chi-squared maximum at the planted sample.
    let best = chi2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(best, 2);
}

#[test]
fn without_chi2_there_is_no_chi2_channel_file() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let mut args = vec!["--plot=false", "--chi2=false", "--per-scope", "tb.dut"];
    args.extend(EDGES);
    assert!(tvla(&batch.meta, &out, &args).status.success());
    assert!(out.join("channels.tsv").exists());
    assert!(!out.join("chi2_channels.npz").exists());
    let (header, _) = tsv(&out);
    assert!(!header.contains(&"max_neg_log10_p".to_string()));
}

#[test]
fn only_the_best_channel_is_plotted() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 40,
        ..LeakSpec::default()
    });
    let out = batch.dir.path().join("out");
    let no_programs = batch.dir.path().join("no-programs");
    std::fs::create_dir_all(&no_programs).unwrap();
    let mut args = vec!["--per-scope", "tb.dut"];
    args.extend(EDGES);
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .env("PATH", &no_programs)
        .arg("--meta-json")
        .arg(&batch.meta)
        .arg("--ttest-output-dir")
        .arg(&out)
        .args(["-d", "2"])
        .args(&args)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    for d in 1..=2 {
        for ext in ["html", "svg", "json"] {
            assert!(out.join(format!("top_channel/t_test_d{d}.{ext}")).exists());
        }
    }
    // The plots of the whole selection are in the output directory as before.
    assert!(out.join("t_test_d1.svg").exists());
    assert!(!out.join("top_channel/max_t_values.svg").exists());
    assert!(
        stderr(&output).contains("best channel tb.dut.a"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn aliases_belong_to_one_channel_and_are_counted() {
    // `tb.dut.b.x_alias` is a second name of `tb.dut.a.x`. Both scopes are equally deep, so the
    // smaller path, `tb.dut.a.x`, decides.
    let batch = write_leak_batch(&LeakSpec {
        aliases: vec![("tb.dut.b", "x_alias", 0)],
        ..LeakSpec::default()
    });
    let out = batch.dir.path().join("out");
    let mut args = vec!["--plot=false", "--per-scope", "tb.dut"];
    args.extend(EDGES);
    let output = tvla(&batch.meta, &out, &args);
    assert!(output.status.success(), "{}", stderr(&output));
    let (_, rows) = tsv(&out);
    assert_eq!(&rows[0][..3], ["1", "tb.dut.a", "1"]);
    assert_eq!(&rows[1][..3], ["2", "tb.dut.b", "1"]);
    let log = stderr(&output);
    assert!(log.contains("1 signals with several names"), "{log}");
}

#[test]
fn a_depth_of_two_splits_the_scopes_below_the_first_level() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    // Without a selection, `tb.clk` is in the scope `tb` itself. The signals below `tb.dut` are
    // folded into `tb.dut.a` and `tb.dut.b`, two levels below `tb`.
    let output = tvla(
        &batch.meta,
        &out,
        &[
            "--plot=false",
            "--clock",
            "tb.clk",
            "--per-scope",
            "tb",
            "--depth",
            "2",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let names = std::fs::read_to_string(out.join("channels.txt")).unwrap();
    assert_eq!(names, "tb\ntb.dut.a\ntb.dut.b\n");
    let (_, rows) = tsv(&out);
    assert_eq!(&rows[0][..2], ["1", "tb.dut.a"]);
}

#[test]
fn a_scope_without_selected_signals_is_an_error() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 20,
        ..LeakSpec::default()
    });
    let out = batch.dir.path().join("out");
    let output = tvla(
        &batch.meta,
        &out,
        &["--plot=false", "--per-scope", "tb.nope"],
    );
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("no selected signal is in the scope tb.nope"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn depth_needs_per_scope() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let output = tvla(&batch.meta, &out, &["--depth", "2"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--per-scope"),
        "{}",
        stderr(&output)
    );
    let output = tvla(&batch.meta, &out, &["--per-scope", "tb", "--depth", "0"]);
    assert!(!output.status.success());
}

#[test]
fn the_legacy_sampling_works_with_channels_and_leaves_traces_npz_alone() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 40,
        ..LeakSpec::default()
    });
    let out = batch.dir.path().join("out");
    let output = tvla(
        &batch.meta,
        &out,
        &[
            "--plot=false",
            "--per-scope",
            "tb.dut",
            "--include",
            "scope:tb.dut",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(out.join("channels.tsv").exists());
    assert!(!batch.dir.path().join("traces.npz").exists());
}
