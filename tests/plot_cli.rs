//! Runs the `plot` binary on `traces.npz` files. `plot ttest` must give the same `t_values.npz`
//! as `tvla` on the same batches, bit for bit, and bad input must give an error, not a panic.

use ndarray::{Array1, Array2};
use ndarray_npz::{NpzReader, NpzWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SAMPLES: usize = 6;

/// Integer-valued traces (small numbers) and alternating labels. The classes differ in some
/// samples. `seed` makes the batches differ.
fn batch_data(seed: u32, num_traces: usize) -> (Array2<f32>, Array1<u16>) {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) % 7
    };
    let labels: Vec<u16> = (0..num_traces).map(|i| (i % 2) as u16).collect();
    let mut data = Vec::new();
    for &label in &labels {
        for sample in 0..SAMPLES {
            let leak = if sample < 2 { 3 * u32::from(label) } else { 0 };
            data.push((next() + leak) as f32);
        }
    }
    (
        Array2::from_shape_vec((num_traces, SAMPLES), data).unwrap(),
        Array1::from_vec(labels),
    )
}

/// Writes a cache file. `order` lists the indices of the `trace_<i>` entries in archive order.
fn write_cache(path: &Path, traces: &Array2<f32>, labels: &Array1<u16>, order: &[usize]) {
    let mut npz = NpzWriter::new(std::fs::File::create(path).unwrap());
    npz.add_array("labels", labels).unwrap();
    for &i in order {
        npz.add_array(format!("trace_{i}"), &traces.row(i)).unwrap();
    }
    npz.finish().unwrap();
}

fn read_t_values(path: &Path) -> Array2<f64> {
    let file = std::fs::File::open(path).unwrap();
    NpzReader::new(file).unwrap().by_name("t_values").unwrap()
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|x| x.to_bits()).collect()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Two batch directories for `tvla`, each with `meta.json` (the waveform does not exist, so
/// the cache is the source) and `traces.npz`. Returns the metadata paths.
fn tvla_batches(root: &Path) -> Vec<PathBuf> {
    (0..2)
        .map(|b| {
            let dir = root.join(format!("batch{b}"));
            std::fs::create_dir_all(&dir).unwrap();
            let (traces, labels) = batch_data(b, 40 + 10 * b as usize);
            write_cache(
                &dir.join("traces.npz"),
                &traces,
                &labels,
                &(0..traces.nrows()).collect::<Vec<_>>(),
            );
            std::fs::write(
                dir.join("meta.json"),
                r#"{"trace_filename": "missing.vcd", "markers": [[0, 1, 0]]}"#,
            )
            .unwrap();
            dir.join("meta.json")
        })
        .collect()
}

fn run_plot(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_plot"))
        .env_remove("RUST_LOG")
        .args(args)
        .output()
        .unwrap()
}

fn assert_clean_failure(output: &Output) {
    let err = text(&output.stderr);
    assert!(!output.status.success(), "the run must fail");
    assert!(!err.contains("panicked"), "{err}");
    assert!(output.status.code().is_some(), "the run was killed: {err}");
}

#[test]
fn plot_ttest_gives_the_same_t_values_as_tvla_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let metas = tvla_batches(dir.path());
    let tvla_out = dir.path().join("tvla_out");
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-json")
        .arg(&metas[0])
        .arg("--meta-list")
        .arg({
            let list = dir.path().join("meta.list");
            std::fs::write(&list, "batch0/meta.json\nbatch1/meta.json\n").unwrap();
            list
        })
        .arg("--ttest-output-dir")
        .arg(&tvla_out)
        .args(["--plot=false", "--chi2=false", "-d", "3"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", text(&output.stderr));

    let plot_out = dir.path().join("plot_out");
    let output = run_plot(&[
        plot_out.to_str().unwrap(),
        "ttest",
        "-d",
        "3",
        "--filenames",
        dir.path().join("batch0/traces.npz").to_str().unwrap(),
        dir.path().join("batch1/traces.npz").to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", text(&output.stderr));

    let expected = read_t_values(&tvla_out.join("t_values.npz"));
    let actual = read_t_values(&plot_out.join("t_values.npz"));
    assert_eq!(expected.dim(), (3, SAMPLES));
    assert_eq!(bits(&expected), bits(&actual));
    // The max-|t| curve starts with the point (0, 0).
    let json = std::fs::read_to_string(plot_out.join("max_t_values.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json).unwrap();
    let first = json.pointer("/data/0").unwrap_or(&json);
    assert_eq!(first["x"][0], serde_json::json!(0.0));
    assert_eq!(first["x"][1], serde_json::json!(40.0));
    assert_eq!(first["y"][0], serde_json::json!(0.0));
    // The planted leak is found, so the comparison is not about NaN only.
    assert!(expected[[0, 0]].abs() > 4.5);
}

#[test]
fn a_list_file_with_empty_lines_and_relative_paths_works() {
    let dir = tempfile::tempdir().unwrap();
    tvla_batches(dir.path());
    let list = dir.path().join("npz.list");
    std::fs::write(&list, "\nbatch0/traces.npz\n\n  \nbatch1/traces.npz\n\n").unwrap();
    let out = dir.path().join("out");
    let output = run_plot(&[
        out.to_str().unwrap(),
        "ttest",
        "-d",
        "2",
        "--npz-list",
        list.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(read_t_values(&out.join("t_values.npz")).dim(), (2, SAMPLES));
}

#[test]
fn trace_entries_out_of_order_give_the_right_result() {
    let dir = tempfile::tempdir().unwrap();
    let (traces, labels) = batch_data(5, 30);
    let n = traces.nrows();
    let sorted = dir.path().join("sorted.npz");
    let shuffled = dir.path().join("shuffled.npz");
    write_cache(&sorted, &traces, &labels, &(0..n).collect::<Vec<_>>());
    // Entries in an order where the position differs from the index everywhere.
    let order: Vec<usize> = (0..n).rev().collect();
    write_cache(&shuffled, &traces, &labels, &order);

    let mut results = Vec::new();
    for (name, file) in [("a", &sorted), ("b", &shuffled)] {
        let out = dir.path().join(name);
        let output = run_plot(&[
            out.to_str().unwrap(),
            "ttest",
            "-d",
            "2",
            "--filenames",
            file.to_str().unwrap(),
        ]);
        assert!(output.status.success(), "{}", text(&output.stderr));
        results.push(read_t_values(&out.join("t_values.npz")));
    }
    assert_eq!(bits(&results[0]), bits(&results[1]));
}

#[test]
fn bad_input_is_an_error_and_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let out = out.to_str().unwrap();
    let good = dir.path().join("good.npz");
    let (traces, labels) = batch_data(1, 20);
    write_cache(&good, &traces, &labels, &(0..20).collect::<Vec<_>>());
    let good = good.to_str().unwrap();
    let garbage = dir.path().join("garbage.npz");
    std::fs::write(&garbage, b"this is not an npz file").unwrap();
    let garbage = garbage.to_str().unwrap();
    let missing = dir.path().join("missing.npz");
    let missing = missing.to_str().unwrap();
    let empty_list = dir.path().join("empty.list");
    std::fs::write(&empty_list, "\n  \n").unwrap();
    let empty_list = empty_list.to_str().unwrap();
    let one_trace = dir.path().join("one.npz");
    write_cache(
        &one_trace,
        &traces.slice(ndarray::s![..1, ..]).to_owned(),
        &labels.slice(ndarray::s![..1]).to_owned(),
        &[0],
    );
    let one_trace = one_trace.to_str().unwrap();

    let cases: Vec<Vec<&str>> = vec![
        vec![out, "ttest"],
        vec![out, "ttest", "--filenames", missing],
        vec![out, "ttest", "--filenames", garbage],
        vec![out, "ttest", "--filenames", good, garbage],
        vec![out, "ttest", "--filenames", one_trace],
        vec![out, "ttest", "--npz-list", empty_list],
        vec![out, "ttest", "--npz-list", missing],
        vec![out, "ttest", "-d", "0", "--filenames", good],
        vec![out, "plot-traces", "0", missing],
        vec![out, "plot-traces", "0", garbage],
        vec![out, "plot-traces", "20", good],
    ];
    for args in cases {
        let output = run_plot(&args);
        assert_clean_failure(&output);
        assert!(!text(&output.stderr).is_empty(), "{args:?}: no message");
    }
}

#[test]
fn plot_traces_reads_a_cache_with_entries_out_of_order() {
    let dir = tempfile::tempdir().unwrap();
    let (traces, labels) = batch_data(2, 10);
    let file = dir.path().join("t.npz");
    write_cache(&file, &traces, &labels, &(0..10).rev().collect::<Vec<_>>());
    let output = run_plot(&["plot-traces", "0", "9", file.to_str().unwrap()]);
    assert!(output.status.success(), "{}", text(&output.stderr));
}
