//! Runs the `tvla` binary with `--clock` on batches with a planted leak.

mod common;

use common::*;
use ndarray::Array2;
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
        .args(["--plot=false", "-d", "1"])
        .args(args)
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn t_values(out: &Path) -> Vec<f64> {
    let file = std::fs::File::open(out.join("t_values.npz")).unwrap();
    let t: Array2<f64> = NpzReader::new(file).unwrap().by_name("t_values").unwrap();
    t.row(0).to_vec()
}

/// The design is selected, the clock is not.
const DUT: [&str; 2] = ["--include", "scope:tb.dut"];

#[test]
fn clock_edges_find_the_planted_leak_at_its_sample() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let mut args = vec!["--clock", "tb.clk"];
    args.extend(DUT);
    let output = tvla(&batch.meta, &out, &args);
    assert!(output.status.success(), "{}", stderr(&output));
    let t = t_values(&out);
    assert_eq!(t.len(), 6);
    for (sample, t) in t.iter().enumerate() {
        if sample == 2 {
            assert!(t.abs() > 4.5, "the leak at sample 2 has t = {t}");
        } else {
            assert!(t.abs() < 4.5, "sample {sample} has t = {t}");
        }
    }
    let log = stderr(&output);
    assert!(
        log.contains("Sampling on the Rising edges of tb.clk"),
        "{log}"
    );
    assert!(log.contains("--exclude signal:tb.clk"), "{log}");
    // Conservation: the summary adds up the toggles inside and outside the bins.
    assert!(
        log.contains("clock edges: 1800 bins in 1 batches; toggles:"),
        "{log}"
    );
    assert!(log.contains("before the first edge"), "{log}");
}

#[test]
fn clock_mode_neither_reads_nor_writes_traces_npz() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let output = tvla(&batch.meta, &out, &["--clock", "tb.clk"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(!batch.dir.path().join("traces.npz").exists());
    // Without --clock, the legacy sampling writes the cache.
    let output = tvla(&batch.meta, &out, &[]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(batch.dir.path().join("traces.npz").exists());
}

#[test]
fn the_offset_and_the_edge_kind_are_accepted() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    // Falling edges are at 15 + 10k. With the offset -2, the bins start at 13 + 10k. Every
    // segment starts 3 ticks before its first bin. The data toggles at 13 + 10k, so the leak
    // stays at the sample 2.
    let output = tvla(
        &batch.meta,
        &out,
        &[
            "--clock",
            "tb.clk",
            "--edges",
            "falling",
            "--offset",
            "-2",
            "--include",
            "scope:tb.dut",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let log = stderr(&output);
    assert!(!log.contains("different places"), "{log}");
    assert!(log.contains("Sampling on the Falling edges"), "{log}");
    let t = t_values(&out);
    assert!(t[2].abs() > 4.5, "{t:?}");
    assert!(
        t.iter().enumerate().all(|(i, t)| i == 2 || t.abs() < 4.5),
        "{t:?}"
    );
}

#[test]
fn edges_and_offset_need_a_clock() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    for args in [["--edges", "both"], ["--offset", "3"]] {
        let output = tvla(&batch.meta, &out, &args);
        assert!(!output.status.success());
        assert!(stderr(&output).contains("--clock"), "{}", stderr(&output));
    }
}

#[test]
fn a_misaligned_segment_gives_a_warning() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 20,
        ..LeakSpec::default()
    });
    // Move the start of the second segment by one tick: its first bin opens 9 ticks later.
    let text = std::fs::read_to_string(&batch.meta).unwrap();
    let text = text.replacen("[70,", "[71,", 1);
    std::fs::write(&batch.meta, text).unwrap();
    let out = batch.dir.path().join("out");
    let output = tvla(
        &batch.meta,
        &out,
        &["--clock", "tb.clk", "--length-policy", "pad"],
    );
    let log = stderr(&output);
    assert!(
        log.contains("start at different places in the clock period"),
        "{log}"
    );
    assert!(log.contains("between 0 and 9 ticks"), "{log}");
}

#[test]
fn the_length_policy_defaults_to_error_with_a_clock_and_pad_without() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 20,
        ..LeakSpec::default()
    });
    // Make the second segment one period shorter.
    let text = std::fs::read_to_string(&batch.meta).unwrap();
    let text = text.replacen("130,", "120,", 1);
    std::fs::write(&batch.meta, text).unwrap();
    let out = batch.dir.path().join("out");
    let output = tvla(&batch.meta, &out, &["--clock", "tb.clk"]);
    assert!(!output.status.success());
    let log = stderr(&output);
    assert!(log.contains("different lengths"), "{log}");
    assert!(log.contains("5 samples x 1"), "{log}");
    for policy in ["pad", "truncate"] {
        let output = tvla(
            &batch.meta,
            &out,
            &["--clock", "tb.clk", "--length-policy", policy],
        );
        assert!(output.status.success(), "{policy}: {}", stderr(&output));
    }
    // The legacy sampling pads, as before.
    let output = tvla(&batch.meta, &out, &["--use-existing=false"]);
    assert!(output.status.success(), "{}", stderr(&output));
}

#[test]
fn a_policy_other_than_pad_leaves_traces_npz_alone() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 20,
        ..LeakSpec::default()
    });
    let out = batch.dir.path().join("out");
    let output = tvla(&batch.meta, &out, &["--length-policy", "error"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(!batch.dir.path().join("traces.npz").exists());
}

#[test]
fn different_alignments_in_different_batches_give_one_warning_at_the_end() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 20,
        ..LeakSpec::default()
    });
    // Batch A: every segment starts at an edge (offset 0). Batch B: every segment starts one
    // tick after an edge (offset 9). Each batch alone is aligned.
    batch.write_shifted_meta("a.json", 0);
    batch.write_shifted_meta("b.json", 1);
    let list = batch.dir.path().join("meta.list");
    std::fs::write(&list, "a.json\nb.json\n").unwrap();
    let out = batch.dir.path().join("out");
    let run = |list: &Path| {
        Command::new(env!("CARGO_BIN_EXE_tvla"))
            .env_remove("RUST_LOG")
            .arg("--meta-list")
            .arg(list)
            .arg("--ttest-output-dir")
            .arg(&out)
            .args(["--plot=false", "-d", "1", "--clock", "tb.clk"])
            .output()
            .unwrap()
    };
    let output = run(&list);
    assert!(output.status.success(), "{}", stderr(&output));
    let log = stderr(&output);
    assert_eq!(
        log.matches("different places in the clock period").count(),
        1,
        "{log}"
    );
    assert!(log.contains("across the batches"), "{log}");
    assert!(log.contains("between 0 and 9 ticks"), "{log}");
    // The same alignment in all batches: no warning.
    std::fs::write(&list, "a.json\na.json\n").unwrap();
    let log = stderr(&run(&list));
    assert!(!log.contains("different places"), "{log}");
}
