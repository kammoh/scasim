//! Runs the `tvla` binary with `--shuffle-labels` on a batch with a planted leak.

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
        .args(["--plot=false", "-d", "2", "--clock", "tb.clk"])
        .args(["--include", "scope:tb.dut"])
        .args(args)
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn t_values(out: &Path) -> Array2<f64> {
    let file = std::fs::File::open(out.join("t_values.npz")).unwrap();
    NpzReader::new(file).unwrap().by_name("t_values").unwrap()
}

fn bits(t: &Array2<f64>) -> Vec<u64> {
    t.iter().map(|v| v.to_bits()).collect()
}

fn max_abs(t: &Array2<f64>) -> f64 {
    t.iter()
        .filter(|v| v.is_finite())
        .fold(0.0, |m, v| v.abs().max(m))
}

#[test]
fn shuffled_labels_remove_a_planted_leak() {
    let batch = write_leak_batch(&LeakSpec::default());
    let plain = batch.dir.path().join("plain");
    let output = tvla(&batch.meta, &plain, &[]);
    assert!(output.status.success(), "{}", stderr(&output));
    let t = t_values(&plain);
    assert!(t[[0, 2]].abs() > 4.5, "the leak must be there: {t:?}");
    assert!(!stderr(&output).contains("shuffled"), "{}", stderr(&output));
    for seed in ["1", "2", "3"] {
        let out = batch.dir.path().join(format!("shuffled{seed}"));
        let output = tvla(&batch.meta, &out, &["--shuffle-labels", seed]);
        assert!(output.status.success(), "{}", stderr(&output));
        let t = t_values(&out);
        assert_eq!(t.dim(), (2, 6));
        assert!(max_abs(&t) < 4.5, "seed {seed}: {t:?}");
        let log = stderr(&output);
        assert!(
            log.contains(&format!("labels were shuffled with the seed {seed}")),
            "{log}"
        );
    }
}

#[test]
fn the_same_seed_gives_the_same_results_and_another_seed_does_not() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 100,
        ..LeakSpec::default()
    });
    let run = |name: &str, seed: &str| {
        let out = batch.dir.path().join(name);
        let output = tvla(&batch.meta, &out, &["--shuffle-labels", seed]);
        assert!(output.status.success(), "{}", stderr(&output));
        bits(&t_values(&out))
    };
    let a = run("a", "11");
    assert_eq!(a, run("b", "11"));
    assert_ne!(a, run("c", "12"));
}

#[test]
fn every_batch_is_shuffled_with_its_own_permutation() {
    // The same batch twice in the list. Each batch gets its own permutation. The run is still
    // reproducible, and the leak is gone.
    let batch = write_leak_batch(&LeakSpec::default());
    let list = batch.dir.path().join("meta.list");
    std::fs::write(&list, "meta.json\nmeta.json\n").unwrap();
    let run = |name: &str| {
        let out = batch.dir.path().join(name);
        let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
            .env_remove("RUST_LOG")
            .arg("--meta-list")
            .arg(&list)
            .arg("--ttest-output-dir")
            .arg(&out)
            .args(["--plot=false", "-d", "2", "--clock", "tb.clk"])
            .args(["--include", "scope:tb.dut", "--shuffle-labels", "5"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        t_values(&out)
    };
    let (first, second) = (run("x"), run("y"));
    assert_eq!(bits(&first), bits(&second));
    assert!(max_abs(&first) < 4.5, "{first:?}");
}

#[test]
fn the_seed_must_be_a_number() {
    let batch = write_leak_batch(&LeakSpec {
        traces: 20,
        ..LeakSpec::default()
    });
    let out = batch.dir.path().join("out");
    let output = tvla(&batch.meta, &out, &["--shuffle-labels", "abc"]);
    assert!(!output.status.success());
    let output = tvla(&batch.meta, &out, &["--shuffle-labels", "-1"]);
    assert!(!output.status.success());
}

#[test]
fn all_channels_see_the_same_shuffled_labels() {
    let batch = write_leak_batch(&LeakSpec::default());
    let out = batch.dir.path().join("out");
    let output = tvla(
        &batch.meta,
        &out,
        &["--per-scope", "tb.dut", "--shuffle-labels", "4"],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let tsv = std::fs::read_to_string(out.join("channels.tsv")).unwrap();
    // Read the columns by name: the layout has more columns than the |t| values.
    let header: Vec<&str> = tsv.lines().next().unwrap().split('\t').collect();
    let column = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let t_columns = [column("max_abs_t_d1"), column("max_abs_t_d2")];
    let infinite = column("infinite_t");
    for row in tsv.lines().skip(1) {
        let cells: Vec<&str> = row.split('\t').collect();
        assert_eq!(cells[infinite], "0", "{row}");
        for index in t_columns {
            assert!(cells[index].parse::<f64>().unwrap() < 4.5, "{row}");
        }
    }
    assert!(max_abs(&t_values(&out)) < 4.5);
}

#[test]
fn the_labels_of_a_cached_batch_are_shuffled_too() {
    let batch = write_leak_batch(&LeakSpec::default());
    // The legacy sampling with `clock_period` 1 keeps every time point, and it writes
    // `traces.npz`. The run that reads the cache must still shuffle the labels.
    let text = std::fs::read_to_string(&batch.meta).unwrap();
    std::fs::write(
        &batch.meta,
        text.replace("\"clock_period\": 10", "\"clock_period\": 1"),
    )
    .unwrap();
    let run = |name: &str, args: &[&str]| {
        let out = batch.dir.path().join(name);
        let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
            .env_remove("RUST_LOG")
            .arg("--meta-json")
            .arg(&batch.meta)
            .arg("--ttest-output-dir")
            .arg(&out)
            .args(["--plot=false", "-d", "1"])
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        (
            bits(&t_values(&out)),
            String::from_utf8_lossy(&output.stdout).into_owned() + &stderr(&output),
        )
    };
    let (plain, _) = run("plain", &[]);
    assert!(batch.dir.path().join("traces.npz").exists());
    assert!(plain.iter().any(|&b| f64::from_bits(b).is_finite()));
    let (shuffled, log) = run("shuffled", &["--shuffle-labels", "3"]);
    assert_ne!(plain, shuffled);
    assert!(log.contains("Using existing traces"), "{log}");
    assert!(
        log.contains("labels were shuffled with the seed 3"),
        "{log}"
    );
}

#[test]
fn the_shuffle_does_not_depend_on_where_the_batch_is_stored() {
    // The permutation comes from the batch's content, not its path. A copy of the batch in
    // another directory gives the same shuffled results, run after run.
    let batch = write_leak_batch(&LeakSpec::default());
    let copy = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(batch.dir.path()).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), copy.path().join(entry.file_name())).unwrap();
        }
    }
    let copy_meta = copy.path().join(batch.meta.file_name().unwrap());
    let run = |meta: &Path, out: &Path| {
        let output = tvla(meta, out, &["--shuffle-labels", "5"]);
        assert!(output.status.success(), "{}", stderr(&output));
        bits(&t_values(out))
    };
    let a = run(&batch.meta, &batch.dir.path().join("a"));
    let b = run(&copy_meta, &copy.path().join("b"));
    assert_eq!(a, b);
}
