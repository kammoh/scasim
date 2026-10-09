//! Writes `tests/fixtures/stats/scalib_ttest.json`.
//!
//! The program makes deterministic traces and labels, feeds them to the SCALib `Ttest` in
//! batches, and saves the inputs together with the t-values that SCALib returns.
//! It only calls the public API of SCALib.
//!
//! Usage: `scalib-fixtures [OUTPUT_FILE]`. The default output file is
//! `tests/fixtures/stats/scalib_ttest.json`, relative to the root of the scasim repository.

use ndarray::{Array1, Array2, s};
use scalib::ttest::Ttest;
use serde_json::{Value, json};
use std::path::PathBuf;

/// Version of this generator: the data recipes, the cases, and the file format.
/// Change it when any of them changes.
const GENERATOR_VERSION: u32 = 1;

/// The Cargo lock file of this crate. It names the exact SCALib commit.
const LOCK_FILE: &str = include_str!("../Cargo.lock");

const SCALIB_REPOSITORY: &str = "https://github.com/kammoh/SCALib.git";

/// SplitMix64, a small deterministic random generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Two-class random data. Each sample has its own mean and standard deviation. The classes
/// differ in mean, variance, or skewness, so the t-values of all orders are more than noise.
/// The values are rounded to integers and are not negative. Labels are random.
fn gen_data(seed: u64, n: usize, ns: usize) -> (Array2<u32>, Array1<u16>) {
    let mut rng = Rng::new(seed);
    let mu: Vec<f64> = (0..ns).map(|_| 1000.0 + 30000.0 * rng.uniform()).collect();
    let sd: Vec<f64> = (0..ns).map(|_| 5.0 + 300.0 * rng.uniform()).collect();
    let labels = Array1::from_shape_fn(n, |_| (rng.next_u64() % 2) as u16);
    let mut x = Array2::zeros((n, ns));
    for i in 0..n {
        for j in 0..ns {
            let g = rng.normal();
            let v = match (labels[i], j % 4) {
                (0, _) | (_, 0) => g,
                (1, 1) => g + 0.15,
                (1, 2) => 1.3 * g,
                _ => (g * g - 1.0) / std::f64::consts::SQRT_2,
            };
            x[[i, j]] = (mu[j] + sd[j] * v).max(0.0).round() as u32;
        }
    }
    (x, labels)
}

/// Makes class 1 rare: every seventh trace.
fn rare_class_1(labels: &mut Array1<u16>) {
    for (i, l) in labels.iter_mut().enumerate() {
        *l = u16::from(i % 7 == 0);
    }
}

/// Small data with degenerate samples. SCALib gives NaN and infinite values for them.
/// Column 0 is constant in both classes; column 1 and 2 are constant per class with different
/// values; column 3 is zero; column 4 varies in class 0 only; column 5 is ordinary.
fn degenerate_data(seed: u64) -> (Array2<u32>, Array1<u16>) {
    let (n, ns) = (40, 6);
    let mut rng = Rng::new(seed);
    let labels = Array1::from_shape_fn(n, |i| (i % 2) as u16);
    let mut x = Array2::zeros((n, ns));
    for i in 0..n {
        let c = labels[i];
        x[[i, 0]] = 100;
        x[[i, 1]] = if c == 0 { 100 } else { 107 };
        x[[i, 2]] = if c == 0 { 107 } else { 100 };
        x[[i, 3]] = 0;
        x[[i, 4]] = if c == 0 {
            (rng.next_u64() % 50) as u32
        } else {
            25
        };
        x[[i, 5]] = (500.0 + 20.0 * rng.normal()).round() as u32;
    }
    (x, labels)
}

/// Ordinary data in which class 1 has exactly one trace.
fn one_trace_class_data(seed: u64) -> (Array2<u32>, Array1<u16>) {
    let (x, mut labels) = gen_data(seed, 30, 4);
    labels.fill(0);
    labels[11] = 1;
    (x, labels)
}

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    U32,
    F32,
}

impl Dtype {
    fn name(self) -> &'static str {
        match self {
            Dtype::U32 => "u32",
            Dtype::F32 => "f32",
        }
    }
}

struct Dataset {
    name: &'static str,
    dtype: Dtype,
    description: &'static str,
    traces: Array2<u32>,
    labels: Array1<u16>,
}

struct Case {
    dataset: usize,
    d: usize,
    batch_sizes: Vec<usize>,
}

/// Cuts `n` traces into batches of the given sizes, which repeat until all traces are used.
fn repeat_sizes(n: usize, sizes: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = n;
    let mut i = 0;
    while left > 0 {
        let size = sizes[i % sizes.len()].min(left);
        out.push(size);
        left -= size;
        i += 1;
    }
    out
}

/// Runs the SCALib t-test. Returns the t-values with shape `(d, ns)`.
fn run_scalib(dataset: &Dataset, d: usize, batch_sizes: &[usize]) -> Array2<f64> {
    let (n, ns) = dataset.traces.dim();
    assert_eq!(
        batch_sizes.iter().sum::<usize>(),
        n,
        "batches must cover all traces"
    );
    let mut ttest = Ttest::new(ns, d);
    let mut start = 0;
    for &len in batch_sizes {
        let end = start + len;
        let labels = dataset.labels.slice(s![start..end]);
        match dataset.dtype {
            Dtype::U32 => ttest.update(dataset.traces.slice(s![start..end, ..]), labels),
            Dtype::F32 => {
                let batch = dataset.traces.slice(s![start..end, ..]).mapv(|v| v as f32);
                ttest.update(batch.view(), labels);
            }
        }
        start = end;
    }
    ttest.get_ttest()
}

fn scalib_commit() -> String {
    let marker = format!("git+{SCALIB_REPOSITORY}?rev=");
    LOCK_FILE
        .lines()
        .find_map(|line| {
            let (_, tail) = line.split_once(&marker)?;
            let (_, commit) = tail.split_once('#')?;
            Some(commit.trim_end_matches('"').to_string())
        })
        .expect("Cargo.lock has no SCALib entry")
}

fn matrix_json(m: &Array2<f64>) -> (Value, Vec<Value>) {
    let mut non_finite = Vec::new();
    let rows: Vec<Value> = m
        .outer_iter()
        .enumerate()
        .map(|(k, row)| {
            Value::Array(
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        if v.is_finite() {
                            json!(v)
                        } else {
                            let kind = if v.is_nan() {
                                "nan"
                            } else if v > 0.0 {
                                "inf"
                            } else {
                                "-inf"
                            };
                            non_finite.push(json!({"order": k + 1, "sample": j, "kind": kind}));
                            Value::Null
                        }
                    })
                    .collect(),
            )
        })
        .collect();
    (Value::Array(rows), non_finite)
}

fn main() {
    let (u32_a, u32_a_labels) = gen_data(1, 600, 120);
    let (f32_a, f32_a_labels) = gen_data(2, 600, 120);
    let (small_a, small_a_labels) = gen_data(3, 600, 71);
    let mut unbalanced = Vec::new();
    for (seed, n, ns) in [(4u64, 300usize, 1usize), (5, 777, 33), (6, 50, 64)] {
        let (x, mut labels) = gen_data(seed, n, ns);
        rare_class_1(&mut labels);
        unbalanced.push((x, labels));
    }
    let (deg_x, deg_labels) = degenerate_data(7);
    let (one_x, one_labels) = one_trace_class_data(8);

    let mut datasets = vec![
        Dataset {
            name: "u32_600x120",
            dtype: Dtype::U32,
            description: "gen_data(seed 1, 600 traces, 120 samples), u32",
            traces: u32_a,
            labels: u32_a_labels,
        },
        Dataset {
            name: "f32_600x120",
            dtype: Dtype::F32,
            description: "gen_data(seed 2, 600 traces, 120 samples), integer values as f32",
            traces: f32_a,
            labels: f32_a_labels,
        },
        Dataset {
            name: "u32_600x71",
            dtype: Dtype::U32,
            description: "gen_data(seed 3, 600 traces, 71 samples), u32",
            traces: small_a,
            labels: small_a_labels,
        },
    ];
    let unbalanced_names = ["unbalanced_300x1", "unbalanced_777x33", "unbalanced_50x64"];
    for (name, (x, labels)) in unbalanced_names.into_iter().zip(unbalanced) {
        datasets.push(Dataset {
            name,
            dtype: Dtype::U32,
            description: "gen_data(seeds 4, 5, 6), then label 1 for every seventh trace, u32",
            traces: x,
            labels,
        });
    }
    datasets.push(Dataset {
        name: "degenerate_40x6",
        dtype: Dtype::U32,
        description: "constant, per-class constant, zero, and one-class-constant samples, u32",
        traces: deg_x,
        labels: deg_labels,
    });
    datasets.push(Dataset {
        name: "one_trace_class_30x4",
        dtype: Dtype::U32,
        description: "gen_data(seed 8, 30 traces, 4 samples), only trace 11 is in class 1, u32",
        traces: one_x,
        labels: one_labels,
    });

    let index = |name: &str| datasets.iter().position(|d| d.name == name).unwrap();
    let mut cases = Vec::new();
    // u32 traces, orders 1 to 4, several batch sizes (including 7).
    let ds = index("u32_600x120");
    for (label, sizes) in [
        ("one_batch", vec![600]),
        ("halves", vec![300, 300]),
        ("1_then_599", vec![1, 599]),
        ("batches_of_333", vec![333]),
        ("batches_of_7", vec![7]),
        ("mixed", vec![256, 100, 243, 1]),
    ] {
        cases.push((
            format!("u32_d4_{label}"),
            Case {
                dataset: ds,
                d: 4,
                batch_sizes: repeat_sizes(600, &sizes),
            },
        ));
    }
    // Integer-valued f32 traces.
    let ds = index("f32_600x120");
    for (label, sizes) in [
        ("one_batch", vec![600]),
        ("batches_of_200", vec![200]),
        ("299_then_301", vec![299, 301]),
    ] {
        cases.push((
            format!("f32_d4_{label}"),
            Case {
                dataset: ds,
                d: 4,
                batch_sizes: repeat_sizes(600, &sizes),
            },
        ));
    }
    // Smaller orders; a test of order k does not depend on the highest order d.
    let ds = index("u32_600x71");
    for d in 1..=3 {
        cases.push((
            format!("u32_d{d}_210_then_390"),
            Case {
                dataset: ds,
                d,
                batch_sizes: vec![210, 390],
            },
        ));
    }
    // Unbalanced classes and small ns.
    for name in unbalanced_names {
        let ds = index(name);
        let n = datasets[ds].traces.nrows();
        cases.push((
            format!("{name}_d3"),
            Case {
                dataset: ds,
                d: 3,
                batch_sizes: vec![n],
            },
        ));
    }
    // Degenerate samples and a class with one trace: the values show what SCALib returns.
    let ds = index("degenerate_40x6");
    cases.push((
        "degenerate_d4".to_string(),
        Case {
            dataset: ds,
            d: 4,
            batch_sizes: vec![40],
        },
    ));
    let ds = index("one_trace_class_30x4");
    cases.push((
        "one_trace_class_d4".to_string(),
        Case {
            dataset: ds,
            d: 4,
            batch_sizes: vec![30],
        },
    ));

    let mut case_values = Vec::new();
    for (name, case) in &cases {
        let dataset = &datasets[case.dataset];
        let t = run_scalib(dataset, case.d, &case.batch_sizes);
        let (t_values, non_finite) = matrix_json(&t);
        case_values.push(json!({
            "name": name,
            "dataset": dataset.name,
            "d": case.d,
            "ns": dataset.traces.ncols(),
            "batch_sizes": case.batch_sizes,
            "t_values": t_values,
            "non_finite": non_finite,
        }));
    }

    let dataset_values: Vec<Value> = datasets
        .iter()
        .map(|d| {
            json!({
                "name": d.name,
                "dtype": d.dtype.name(),
                "description": d.description,
                "n": d.traces.nrows(),
                "ns": d.traces.ncols(),
                "labels": d.labels.to_vec(),
                "traces": d.traces.outer_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
            })
        })
        .collect();

    let doc = json!({
        "format": 1,
        "generator": "scripts/scalib-fixtures",
        "generator_version": GENERATOR_VERSION,
        "rng": "SplitMix64, seeds in the dataset descriptions; the traces are stored below",
        "scalib": {"repository": SCALIB_REPOSITORY, "commit": scalib_commit()},
        "note": "t_values hold the SCALib results, shape (d, ns), classes 0 and 1. A null entry is non-finite; 'non_finite' lists its kind: nan, inf, or -inf. Traces are integers; for dtype f32, convert each value to f32 (exact).",
        "datasets": dataset_values,
        "cases": case_values,
    });

    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/stats/scalib_ttest.json")
        });
    std::fs::create_dir_all(output.parent().unwrap()).unwrap();
    let mut text = serde_json::to_string(&doc).unwrap();
    text.push('\n');
    std::fs::write(&output, &text).unwrap();
    eprintln!(
        "wrote {} ({} bytes, {} cases)",
        output.display(),
        text.len(),
        cases.len()
    );
}
