//! Compares the traces of a real batch with the `traces.npz` that an earlier version wrote.
//! Run with:
//!   SCASIM_REAL_BATCH=/path/to/batch cargo test --release --test real_data -- --ignored
//! The batch directory must contain the metadata file, the waveform, and `traces.npz`.
//! The test only reads the directory.

use ndarray::{Array1, Array2};
use ndarray_npz::NpzReader;
use scasim::hierarchy::Selection;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
#[ignore = "needs SCASIM_REAL_BATCH"]
fn real_batch_matches_reference_npz() {
    let dir = PathBuf::from(std::env::var("SCASIM_REAL_BATCH").expect("set SCASIM_REAL_BATCH"));
    let meta_path = ["meta.json.gz", "meta.json"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
        .expect("no meta.json(.gz) in SCASIM_REAL_BATCH");
    let meta = scasim::batch::read_batch_meta(&meta_path).unwrap();
    let (traces, labels, diagnostics) =
        scasim::batch::batch_traces(&meta, &Selection::all()).unwrap();
    eprintln!(
        "toggles: {} in total, {} kept, {} inside segments; {:?}",
        diagnostics.total_toggles,
        diagnostics.kept_toggles,
        diagnostics.segment_toggles,
        diagnostics.info
    );

    let mut npz = NpzReader::new(std::fs::File::open(dir.join("traces.npz")).unwrap()).unwrap();
    let names = npz.names().unwrap();
    let ref_labels: Array1<u16> = npz.by_name("labels").unwrap();
    assert_eq!(labels, ref_labels, "labels differ");
    let trace_names: Vec<_> = names
        .iter()
        .filter(|n| n.starts_with("trace_"))
        .cloned()
        .collect();
    assert_eq!(
        trace_names.len(),
        traces.nrows(),
        "number of traces differs"
    );
    for name in trace_names {
        let index: usize = name
            .trim_start_matches("trace_")
            .trim_end_matches(".npy")
            .parse()
            .unwrap();
        let reference: Array1<f32> = npz.by_name(&name).unwrap();
        assert_eq!(traces.row(index), reference, "{name} differs");
    }
}

// ---------------------------------------------------------------------------------------------
// t-value oracles. The files are in `tests/data/real_tvalues/`; the README there tells how
// they were made.
// ---------------------------------------------------------------------------------------------

/// The highest t-test order of the oracles.
const ORACLE_ORDER: &str = "4";

/// Relative tolerance on `|a - b| / max(1, |a|, |b|)`.
const T_TOLERANCE: f64 = 1e-9;

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/real_tvalues")
}

fn read_t_values(path: &Path) -> Array2<f64> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut npz = NpzReader::new(file).unwrap();
    npz.by_name("t_values").unwrap()
}

/// Compares two arrays of t-values and returns the largest relative difference of the finite
/// values.
///
/// The two arrays must have the same shape. A NaN must meet a NaN. An infinite value must meet
/// an infinite value of the same sign. Finite values must differ by at most `tolerance`
/// (relative, `|a - b| / max(1, |a|, |b|)`). The error message names the first element that
/// breaks a rule.
fn compare_t_values(
    actual: &Array2<f64>,
    oracle: &Array2<f64>,
    tolerance: f64,
) -> Result<f64, String> {
    if actual.dim() != oracle.dim() {
        return Err(format!(
            "shape differs: {:?} against {:?}",
            actual.dim(),
            oracle.dim()
        ));
    }
    let mut worst: f64 = 0.0;
    for ((index, &a), &b) in actual.indexed_iter().zip(oracle.iter()) {
        if a.is_nan() || b.is_nan() {
            if !(a.is_nan() && b.is_nan()) {
                return Err(format!("at {index:?}: {a} against oracle {b}"));
            }
        } else if a.is_infinite() || b.is_infinite() {
            if a != b {
                return Err(format!("at {index:?}: {a} against oracle {b}"));
            }
        } else {
            let error = (a - b).abs() / 1.0f64.max(a.abs()).max(b.abs());
            if error > tolerance {
                return Err(format!(
                    "at {index:?}: {a} against oracle {b} (relative difference {error:e})"
                ));
            }
            worst = worst.max(error);
        }
    }
    Ok(worst)
}

fn count_non_finite(t: &Array2<f64>) -> (usize, usize) {
    let nan = t.iter().filter(|v| v.is_nan()).count();
    let inf = t.iter().filter(|v| v.is_infinite()).count();
    (nan, inf)
}

/// Runs the `tvla` binary with the given extra arguments and returns its `t_values`.
fn run_tvla(out_dir: &Path, extra: &[&str]) -> Array2<f64> {
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .args(["-d", ORACLE_ORDER, "--plot=false", "--ttest-output-dir"])
        .arg(out_dir)
        .args(extra)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "tvla failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    read_t_values(&out_dir.join("t_values.npz"))
}

/// Checks that the t-values of one real batch equal the t-values of the SCALib-based `tvla`.
///
/// The test needs two environment variables:
///   SCASIM_REAL_TVALUES  the name of an oracle in `tests/data/real_tvalues/` (without `.npz`)
///   SCASIM_REAL_BATCH    the batch directory (metadata file and waveform; it is only read)
///
/// It copies the metadata file and the waveform to a temporary directory. It then runs `tvla`
/// with `-d 4 --plot=false --use-existing=false` and compares `t_values.npz` with the oracle.
///
/// Rules of the comparison: relative tolerance 1e-9, NaN equals NaN, and an infinite value
/// equals an infinite value of the same sign. SCALib gives NaN for 0/0 and for a class with
/// fewer than two traces, and +inf or -inf for x/0. The new engine must give a non-finite value
/// of the same kind at the same place (ruling A2 of the A2a design). The current oracles have
/// NaN but no infinite value.
///
/// Run with:
///   SCASIM_REAL_TVALUES=<name> SCASIM_REAL_BATCH=<batch dir> \
///     cargo test --release --test real_data tvla_t_values_match -- --ignored
#[test]
#[ignore = "needs SCASIM_REAL_TVALUES and SCASIM_REAL_BATCH"]
fn tvla_t_values_match_the_scalib_oracle() {
    let name = std::env::var("SCASIM_REAL_TVALUES").expect("set SCASIM_REAL_TVALUES");
    let batch = PathBuf::from(std::env::var("SCASIM_REAL_BATCH").expect("set SCASIM_REAL_BATCH"));
    let meta_path = ["meta.json.gz", "meta.json"]
        .iter()
        .map(|n| batch.join(n))
        .find(|p| p.exists())
        .expect("no meta.json(.gz) in SCASIM_REAL_BATCH");
    let meta = scasim::batch::read_batch_meta(&meta_path).unwrap();

    let work = tempfile::tempdir().unwrap();
    let meta_copy = work.path().join(meta_path.file_name().unwrap());
    std::fs::copy(&meta_path, &meta_copy).unwrap();
    // The waveform keeps its place relative to the metadata file.
    let relative = meta
        .trace_path
        .strip_prefix(&batch)
        .expect("the waveform must be inside the batch directory");
    let waveform_copy = work.path().join(relative);
    std::fs::create_dir_all(waveform_copy.parent().unwrap()).unwrap();
    std::fs::copy(&meta.trace_path, &waveform_copy).unwrap();

    let out = work.path().join("out");
    let actual = run_tvla(
        &out,
        &[
            "--use-existing=false",
            "--meta-json",
            meta_copy.to_str().unwrap(),
        ],
    );
    let oracle = read_t_values(&oracle_dir().join(format!("{name}.npz")));
    let worst = compare_t_values(&actual, &oracle, T_TOLERANCE)
        .unwrap_or_else(|e| panic!("t-values differ from the oracle {name}: {e}"));
    eprintln!(
        "{name}: shape {:?}, (NaN, inf) counts {:?}, worst relative difference {worst:e}",
        actual.dim(),
        count_non_finite(&actual)
    );
}

/// Checks the t-values of a run over many cached batches against the 50-batch oracle.
///
/// The test needs one environment variable:
///   SCASIM_REAL_MULTI  the directory of a multi run, with one subdirectory per batch
///
/// The batch list is `tests/data/real_tvalues/multi50.meta.list`. For each listed batch, the test
/// copies `meta.json.gz` and `traces.npz` to a temporary directory. It does not copy any
/// waveform, so `tvla` has to read the caches (`--use-existing=true`). The comparison rules are
/// the same as in `tvla_t_values_match_the_scalib_oracle`.
///
/// Run with:
///   SCASIM_REAL_MULTI=<multi run dir> \
///     cargo test --release --test real_data tvla_multi_batch -- --ignored
#[test]
#[ignore = "needs SCASIM_REAL_MULTI"]
fn tvla_multi_batch_t_values_match_the_scalib_oracle() {
    let base = PathBuf::from(std::env::var("SCASIM_REAL_MULTI").expect("set SCASIM_REAL_MULTI"));
    let list_path = oracle_dir().join("multi50.meta.list");
    let list = std::fs::read_to_string(&list_path).unwrap();
    let work = tempfile::tempdir().unwrap();
    let mut count = 0;
    for line in list.lines().filter(|l| !l.trim().is_empty()) {
        let meta_rel = Path::new(line.trim());
        let batch_rel = meta_rel.parent().unwrap();
        let target = work.path().join(batch_rel);
        std::fs::create_dir_all(&target).unwrap();
        for file in [meta_rel.file_name().unwrap(), "traces.npz".as_ref()] {
            std::fs::copy(base.join(batch_rel).join(file), target.join(file)).unwrap();
        }
        count += 1;
    }
    let list_copy = work.path().join("meta.list");
    std::fs::copy(&list_path, &list_copy).unwrap();

    let out = work.path().join("out");
    let actual = run_tvla(
        &out,
        &[
            "--use-existing=true",
            "--meta-list",
            list_copy.to_str().unwrap(),
        ],
    );
    let oracle = read_t_values(&oracle_dir().join("multi50.npz"));
    let worst = compare_t_values(&actual, &oracle, T_TOLERANCE)
        .unwrap_or_else(|e| panic!("t-values differ from the oracle multi50: {e}"));
    eprintln!(
        "multi50 ({count} batches): shape {:?}, (NaN, inf) counts {:?}, worst relative difference {worst:e}",
        actual.dim(),
        count_non_finite(&actual)
    );
}

#[test]
fn t_value_comparison_follows_its_rules() {
    use ndarray::array;
    let n = f64::NAN;
    let i = f64::INFINITY;
    let oracle = array![[1.0, n, i, -i, 1e6]];
    assert!(compare_t_values(&oracle, &oracle, 1e-9).is_ok());
    // A small relative difference passes, a large one does not.
    let near = array![[1.0 + 1e-11, n, i, -i, 1e6 * (1.0 + 1e-11)]];
    assert!(compare_t_values(&near, &oracle, 1e-9).is_ok());
    let far = array![[1.0 + 1e-8, n, i, -i, 1e6]];
    assert!(compare_t_values(&far, &oracle, 1e-9).is_err());
    // NaN and infinity must meet the same kind.
    assert!(compare_t_values(&array![[1.0, 1.0, i, -i, 1e6]], &oracle, 1e-9).is_err());
    assert!(compare_t_values(&array![[1.0, n, n, -i, 1e6]], &oracle, 1e-9).is_err());
    assert!(compare_t_values(&array![[1.0, n, -i, -i, 1e6]], &oracle, 1e-9).is_err());
    assert!(compare_t_values(&array![[1.0, i, i, -i, 1e6]], &oracle, 1e-9).is_err());
    // The shapes must match.
    assert!(compare_t_values(&array![[1.0]], &oracle, 1e-9).is_err());
}
