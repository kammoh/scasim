//! Runs the `tvla` binary with `--traces-out` and checks the per-channel traces file.

mod common;

use common::*;
use ndarray::{Array1, Array2};
use ndarray_npz::NpzReader;
use scasim::stats::{Binning, HistAccumulator};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn tvla(meta: &Path, out: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-json")
        .arg(meta)
        .arg("--ttest-output-dir")
        .arg(out)
        .args(["--plot=false", "-d", "2"])
        .args(args)
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).replace("\n  │ ", " ")
}

fn ok(output: &Output) {
    assert!(output.status.success(), "{}", stderr(output));
}

/// The content of a traces file.
struct Traces {
    meta: serde_json::Value,
    arrays: BTreeMap<String, Array2<u32>>,
    labels: Array1<u16>,
    groups: Array1<u64>,
    segment_ids: Array1<u64>,
}

fn read_traces(path: &Path) -> Traces {
    let mut npz = NpzReader::new(std::fs::File::open(path).unwrap()).unwrap();
    let names = npz.names().unwrap();
    let meta_bytes: Array1<u8> = npz.by_name("meta.json").unwrap();
    let meta: serde_json::Value = serde_json::from_slice(meta_bytes.as_slice().unwrap()).unwrap();
    let mut arrays = BTreeMap::new();
    for name in names.iter().filter(|n| n.starts_with("t_")) {
        let name = name.trim_end_matches(".npy").to_string();
        arrays.insert(name.clone(), npz.by_name(&name).unwrap());
    }
    Traces {
        meta,
        arrays,
        labels: npz.by_name("labels").unwrap(),
        groups: npz.by_name("groups").unwrap(),
        segment_ids: npz.by_name("segment_ids").unwrap(),
    }
}

fn read_t(path: &Path, name: &str) -> Array2<f64> {
    NpzReader::new(std::fs::File::open(path).unwrap())
        .unwrap()
        .by_name(name)
        .unwrap()
}

fn bits(t: &Array2<f64>) -> Vec<u64> {
    t.iter().map(|v| v.to_bits()).collect()
}

/// The t-values of the rows of `traces` in `group` (all rows if `None`), with labels 0 and 1.
fn recompute(traces: &Array2<u32>, t: &Traces, group: Option<u64>) -> Array2<f64> {
    let rows: Vec<usize> = (0..traces.nrows())
        .filter(|&i| group.is_none_or(|g| t.groups[i] == g))
        .collect();
    let mut hist = HistAccumulator::new(traces.ncols(), Binning::Exact);
    hist.update(
        traces.select(ndarray::Axis(0), &rows).view(),
        t.labels.select(ndarray::Axis(0), &rows).view(),
    )
    .unwrap();
    hist.t_values(0, 1, 2).unwrap()
}

const EDGES: [&str; 4] = ["--clock", "tb.clk", "--include", "scope:tb.dut"];

fn names(t: &Traces) -> Vec<String> {
    t.meta["channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn legacy_traces_reproduce_the_t_values_bit_for_bit() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let file = batch.dir.path().join("traces-out.npz");
    let output = tvla(&batch.meta, &out, &["--traces-out", file.to_str().unwrap()]);
    ok(&output);
    let t = read_traces(&file);
    assert_eq!(t.arrays.keys().collect::<Vec<_>>(), ["t_0"]);
    let traces = &t.arrays["t_0"];
    assert_eq!(traces.nrows(), 300);
    assert_eq!(t.labels.to_vec(), batch.labels);
    assert_eq!(t.groups.to_vec(), vec![0; 300]);
    assert_eq!(t.segment_ids.to_vec(), (0..300).collect::<Vec<u64>>());
    let reference = read_t(&out.join("t_values.npz"), "t_values");
    assert_eq!(bits(&recompute(traces, &t, None)), bits(&reference));
    // The meta entry.
    assert_eq!(t.meta["format"], 1);
    assert_eq!(t.meta["samples"], traces.ncols());
    assert_eq!(t.meta["segments"], 300);
    assert_eq!(t.meta["shuffle_seed"], serde_json::Value::Null);
    assert_eq!(names(&t), ["total"]);
    assert_eq!(t.meta["channels"][0]["array"], "t_0");
    assert_eq!(t.meta["channels"][0]["is_total"], true);
    assert!(t.meta["channels"][0]["handles_hash"].is_string());
    assert!(t.meta["cache_key"]["common"]["settings"].is_object());
    assert!(t.meta["cache_key"]["batch"]["waveform_size"].is_number());
    // A run with --traces-out computes the traces again and does not touch `traces.npz`.
    assert!(!batch.dir.path().join("traces.npz").exists());
}

#[test]
fn edge_traces_reproduce_the_t_values_bit_for_bit() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let file = batch.dir.path().join("t.npz");
    let mut args = vec!["--traces-out", file.to_str().unwrap()];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &out, &args));
    let t = read_traces(&file);
    let traces = &t.arrays["t_0"];
    assert_eq!(traces.dim(), (300, 6));
    let reference = read_t(&out.join("t_values.npz"), "t_values");
    assert_eq!(bits(&recompute(traces, &t, None)), bits(&reference));
    assert_eq!(
        t.meta["cache_key"]["common"]["settings"]["sampling"],
        "edges"
    );
}

#[test]
fn traces_out_can_be_combined_with_stats_out() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let file = batch.dir.path().join("t.npz");
    let stats = batch.dir.path().join("batch.stats");
    let mut args = vec![
        "--traces-out",
        file.to_str().unwrap(),
        "--stats-out",
        stats.to_str().unwrap(),
    ];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &out, &args));
    assert!(stats.exists());
    // The same run without --traces-out writes the same cache key.
    let stats2 = batch.dir.path().join("batch2.stats");
    let mut args = vec!["--stats-out", stats2.to_str().unwrap()];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &batch.dir.path().join("out2"), &args));
    assert_eq!(
        std::fs::read(&stats).unwrap(),
        std::fs::read(&stats2).unwrap()
    );
    let t = read_traces(&file);
    assert_eq!(t.meta["cache_key"]["common"]["settings"]["clock"], "tb.clk");
}

#[test]
fn per_scope_traces_have_one_array_per_channel_and_the_total() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let file = batch.dir.path().join("t.npz");
    let mut args = vec![
        "--traces-out",
        file.to_str().unwrap(),
        "--per-scope",
        "tb.dut",
    ];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &out, &args));
    let t = read_traces(&file);
    assert_eq!(names(&t), ["total", "tb.dut.a", "tb.dut.b"]);
    assert_eq!(
        t.arrays.keys().cloned().collect::<Vec<_>>(),
        ["t_0", "t_1", "t_2"]
    );
    let total = read_t(&out.join("t_values.npz"), "t_values");
    assert_eq!(bits(&recompute(&t.arrays["t_0"], &t, None)), bits(&total));
    // The channel files of the run count scopes from 0, without the total.
    let channels = out.join("t_values_channels.npz");
    for (array, channel) in [("t_1", "t_0"), ("t_2", "t_1")] {
        let expected = read_t(&channels, channel);
        assert_eq!(
            bits(&recompute(&t.arrays[array], &t, None)),
            bits(&expected),
            "{array}"
        );
    }
    // The planted leak is in the channel a, at sample 2.
    assert_eq!(t.meta["channels"][1]["array"], "t_1");
    assert_eq!(t.meta["channels"][1]["is_total"], false);

    // Select channels: an exact name, a regex, and the total.
    let only = batch.dir.path().join("only.npz");
    let mut args = vec![
        "--traces-out",
        only.to_str().unwrap(),
        "--per-scope",
        "tb.dut",
        "--traces-channels",
        "tb.dut.b",
    ];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &batch.dir.path().join("o2"), &args));
    let s = read_traces(&only);
    assert_eq!(names(&s), ["tb.dut.b"]);
    assert_eq!(s.arrays.keys().collect::<Vec<_>>(), ["t_2"]);
    assert_eq!(s.arrays["t_2"], t.arrays["t_2"]);

    let both = batch.dir.path().join("both.npz");
    let mut args = vec![
        "--traces-out",
        both.to_str().unwrap(),
        "--per-scope",
        "tb.dut",
        "--traces-channels",
        "total",
        "regex:tb\\.dut\\.a",
    ];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &batch.dir.path().join("o3"), &args));
    let s = read_traces(&both);
    assert_eq!(names(&s), ["total", "tb.dut.a"]);
    assert_eq!(s.arrays["t_1"], t.arrays["t_1"]);
}

#[test]
fn a_channel_spec_that_matches_nothing_is_an_error_and_nothing_is_written() {
    let batch = write_leak_batch(&LeakSpec::default());
    let file = batch.dir.path().join("t.npz");
    for spec in ["nope", "regex:^zzz$", "tb.dut.a"] {
        let mut args = vec![
            "--traces-out",
            file.to_str().unwrap(),
            "--traces-channels",
            spec,
        ];
        args.extend(EDGES);
        let output = tvla(&batch.meta, &batch.dir.path().join("out"), &args);
        assert!(!output.status.success(), "{spec}");
        let log = stderr(&output);
        assert!(
            log.contains("--traces-channels") && log.contains(spec),
            "{log}"
        );
        assert!(!file.exists());
    }
    // No temporary file stays in the directory.
    let leftovers: Vec<_> = std::fs::read_dir(batch.dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn shuffled_runs_keep_the_raw_labels_and_the_seed() {
    let batch = write_leak_batch(&LeakSpec::default());
    let file = batch.dir.path().join("t.npz");
    let out = batch.dir.path().join("out");
    let mut args = vec![
        "--traces-out",
        file.to_str().unwrap(),
        "--shuffle-labels",
        "5",
    ];
    args.extend(EDGES);
    ok(&tvla(&batch.meta, &out, &args));
    let t = read_traces(&file);
    assert_eq!(t.labels.to_vec(), batch.labels);
    assert_eq!(t.meta["shuffle_seed"], 5);
    // The t-values of the file (raw labels) show the leak. The shuffled run does not.
    let raw = recompute(&t.arrays["t_0"], &t, None);
    assert!(raw[[0, 2]].abs() > 4.5, "{raw:?}");
    let shuffled = read_t(&out.join("t_values.npz"), "t_values");
    assert!(shuffled[[0, 2]].abs() < 4.5, "{shuffled:?}");
}

#[test]
fn traces_out_needs_one_meta_json_batch() {
    let batch = write_leak_batch(&LeakSpec::default());
    let list = batch.dir.path().join("meta.list");
    std::fs::write(&list, "meta.json\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .arg("--meta-list")
        .arg(&list)
        .args(["--traces-out", "x.npz", "--plot=false"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .arg("--meta-json")
        .arg(&batch.meta)
        .args(["--traces-channels", "total", "--plot=false"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "--traces-channels needs --traces-out"
    );
}

// Version 1 metadata, with two groups and three labels.

fn v1_batch() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let mut waveform = String::from(
        "$timescale 1ps $end\n$scope module tb $end\n$var wire 1 ! clk $end\n$scope module unit $end\n$var wire 4 \" data $end\n$upscope $end\n$upscope $end\n$enddefinitions $end\n#0\n0!\nb0000 \"\n",
    );
    let mut segments = Vec::new();
    for i in 0..24u64 {
        let label = (i % 3) as u16;
        for j in 0..2 {
            let t = 10 + i * 20 + j * 10;
            waveform.push_str(&format!(
                "#{t}\n1!\nb{:04b} \"\n#{}\n0!\n",
                (i * 7 + j * 5 + u64::from(label) * 3 + i / 3) % 16,
                t + 5
            ));
        }
        segments.push(serde_json::json!({
            "id": 100 + i, "start": 10 + i * 20, "end": 30 + i * 20, "label": label, "group": i / 12
        }));
    }
    waveform.push_str("#500\n1!\n");
    std::fs::write(dir.path().join("w.vcd"), waveform).unwrap();
    let json = serde_json::json!({
        "scasim_meta": 1, "waveform": "w.vcd", "time": {"mantissa": 1, "exponent": -12},
        "batch": {"id": "b7", "status": "committed", "seeds": {"base": 17},
                  "design_random": {"requested": "off", "applied": "off", "how": "hook"}},
        "segments": segments, "labels": {"0": "a", "1": "b", "2": "c"},
        "groups": {"0": "g0", "1": "g1"}, "extensions": {}
    });
    let meta = dir.path().join("meta.json");
    std::fs::write(&meta, serde_json::to_vec(&json).unwrap()).unwrap();
    (dir, meta)
}

#[test]
fn version_1_traces_hold_all_segments_with_groups_and_ids() {
    let (dir, meta) = v1_batch();
    let file = dir.path().join("t.npz");
    let out = dir.path().join("out");
    let args = [
        "--traces-out",
        file.to_str().unwrap(),
        "--clock",
        "tb.clk",
        "--group",
        "1",
    ];
    ok(&tvla(&meta, &out, &args));
    let t = read_traces(&file);
    let traces = &t.arrays["t_0"];
    // All 24 segments are in the file, whatever --group says.
    assert_eq!(traces.nrows(), 24);
    assert_eq!(t.segment_ids.to_vec(), (100..124).collect::<Vec<u64>>());
    assert_eq!(
        t.groups.to_vec(),
        (0..24).map(|i| i / 12).collect::<Vec<u64>>()
    );
    assert_eq!(
        t.labels.to_vec(),
        (0..24).map(|i| (i % 3) as u16).collect::<Vec<_>>()
    );
    let reference = read_t(&out.join("t_values.npz"), "t_values");
    assert_eq!(bits(&recompute(traces, &t, Some(1))), bits(&reference));
    assert_eq!(t.meta["batch_id"], "b7");
    // The other group.
    let out0 = dir.path().join("out0");
    ok(&tvla(&meta, &out0, &["--clock", "tb.clk", "--group", "0"]));
    let reference = read_t(&out0.join("t_values.npz"), "t_values");
    assert_eq!(bits(&recompute(traces, &t, Some(0))), bits(&reference));
}
