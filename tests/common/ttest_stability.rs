//! Numerical stability: traces with a large mean and a small variance.

use crate::common;

use common::{Rng, max_err, reference_t};
use ndarray::{Array1, Array2, s};
use scasim::stats::ttest::{MomentAccumulator, TraceSample};

/// The tolerance on `|a - b| / max(1, |a|, |b|)`.
const TOL: f64 = 1e-9;

/// Two classes with the given mean offset and standard deviation. Class 1 differs in
/// mean, variance, and skewness, so that every order has a nonzero t-value.
fn offset_data(
    seed: u64,
    n: usize,
    ns: usize,
    offset: f64,
    sigma: f64,
    round: bool,
) -> (Array2<f64>, Array1<u16>) {
    let mut rng = Rng::new(seed);
    let labels = Array1::from_shape_fn(n, |i| (i % 2) as u16);
    let x = Array2::from_shape_fn((n, ns), |(i, j)| {
        let g = rng.normal();
        let v = match (labels[i], j % 3) {
            (0, _) | (_, 0) => g,
            (_, 1) => 1.2 * g + 0.05,
            _ => (g * g - 1.0) / std::f64::consts::SQRT_2 + 0.03,
        };
        let v = offset + sigma * v;
        if round { v.round() } else { v }
    });
    (x, labels)
}

fn run<T: TraceSample>(x: &Array2<T>, labels: &Array1<u16>, d: usize, batch: usize) -> Array2<f64> {
    let mut acc = MomentAccumulator::new(x.ncols(), d).unwrap();
    let mut start = 0;
    while start < x.nrows() {
        let end = (start + batch).min(x.nrows());
        acc.update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
            .unwrap();
        start = end;
    }
    acc.t_values(0, 1)
}

fn check<T: TraceSample>(name: &str, x: &Array2<T>, labels: &Array1<u16>, batch: usize) -> f64 {
    let xr = x.mapv(|v| v.to_f64());
    let want = reference_t(&xr, &labels.to_vec(), 0, 1, 4);
    let got = run(x, labels, 4, batch);
    // The test must not be trivial.
    let big = want.iter().filter(|v| v.abs() > 3.0).count();
    assert!(
        big > 0,
        "{name}: no t-value is large; the data is not interesting"
    );
    let e = max_err(&got, &want);
    eprintln!("{name} (batch {batch}): error {e:.3e}");
    e
}

#[test]
fn offset_1e6_sigma_1_f64_and_f32() {
    let (x, labels) = offset_data(31, 20_000, 30, 1e6, 1.0, false);
    let mut worst: f64 = 0.0;
    for batch in [20_000, 1000, 77] {
        worst = worst.max(check("f64 1e6", &x, &labels, batch));
    }
    // In f32, the values are quantized to 1/16. The reference sees the quantized values.
    let xf: Array2<f32> = x.mapv(|v| v as f32);
    for batch in [20_000, 1000, 77] {
        worst = worst.max(check("f32 1e6", &xf, &labels, batch));
    }
    eprintln!("worst error at 1e6: {worst:.3e}");
    assert!(worst < TOL);
}

#[test]
fn offset_1e12_sigma_10_integers() {
    let (x, labels) = offset_data(32, 20_000, 30, 1e12, 10.0, true);
    let xi: Array2<i64> = x.mapv(|v| v as i64);
    let xu: Array2<u64> = x.mapv(|v| v as u64);
    let mut worst: f64 = 0.0;
    for batch in [20_000, 1000, 77] {
        worst = worst.max(check("i64 1e12", &xi, &labels, batch));
        worst = worst.max(check("u64 1e12", &xu, &labels, batch));
        worst = worst.max(check("f64 1e12", &x, &labels, batch));
    }
    eprintln!("worst error at 1e12: {worst:.3e}");
    assert!(worst < TOL);
}

#[test]
fn offset_1e12_sigma_10_float_noise() {
    // Not integers: the values have the precision of f64 at 1e12 (about 1.2e-4).
    let (x, labels) = offset_data(33, 20_000, 30, 1e12, 10.0, false);
    let e = check("f64 1e12 (not rounded)", &x, &labels, 500);
    assert!(e < TOL);
}

/// An exact reference for integer traces and orders 1 to 4. It uses 128-bit integer
/// arithmetic for all sums, and f64 only for the last divisions.
fn exact_integer_t(x: &Array2<i64>, labels: &[u16], a: u16, b: u16) -> Array2<f64> {
    let (n, ns) = x.dim();
    let mut out = Array2::from_elem((4, ns), f64::NAN);
    for j in 0..ns {
        let off = x[[0, j]] as i128;
        // For one class: n, S = sum z, and CM_p for p = 2..=8 from exact integer sums.
        let stats = |class: u16| {
            let z: Vec<i128> = (0..n)
                .filter(|&i| labels[i] == class)
                .map(|i| x[[i, j]] as i128 - off)
                .collect();
            let cnt = z.len() as i128;
            let sum: i128 = z.iter().sum();
            let mut cm = [0.0f64; 9];
            for (p, c) in cm.iter_mut().enumerate().skip(2) {
                let mut acc: i128 = 0;
                for &v in &z {
                    let dev = cnt * v - sum; // n * (z - mean), exact
                    acc = acc
                        .checked_add(dev.checked_pow(p as u32).expect("i128 overflow"))
                        .expect("i128 overflow");
                }
                // CM_p = acc / n^(p + 1)
                *c = acc as f64 / (cnt as f64).powi(p as i32 + 1);
            }
            (sum as f64 / cnt as f64, cm, cnt as f64)
        };
        let (ma, cma, na) = stats(a);
        let (mb, cmb, nb) = stats(b);
        let pre = |k: usize, cm: &[f64; 9]| -> (f64, f64) {
            if k == 2 {
                (cm[2], cm[4] - cm[2] * cm[2])
            } else {
                let s_k = cm[2].sqrt().powi(k as i32);
                (
                    cm[k] / s_k,
                    (cm[2 * k] - cm[k] * cm[k]) / cm[2].powi(k as i32),
                )
            }
        };
        for k in 1..=4usize {
            let (diff, va, vb) = if k == 1 {
                (ma - mb, cma[2], cmb[2])
            } else {
                let (pa, va) = pre(k, &cma);
                let (pb, vb) = pre(k, &cmb);
                (pa - pb, va, vb)
            };
            let t = diff / (va / na + vb / nb).sqrt();
            out[[k - 1, j]] = if t.is_finite() { t } else { f64::NAN };
        }
    }
    out
}

#[test]
fn matches_exact_integer_arithmetic_at_1e12() {
    // 800 traces keep all 128-bit sums (up to order 8) in range.
    let (x, labels) = offset_data(34, 800, 20, 1e12, 10.0, true);
    let xi: Array2<i64> = x.mapv(|v| v as i64);
    let want = exact_integer_t(&xi, &labels.to_vec(), 0, 1);
    let mut worst: f64 = 0.0;
    for batch in [800, 100, 33, 1] {
        let got = run(&xi, &labels, 4, batch);
        let e = max_err(&got, &want);
        eprintln!("exact integer reference, batch {batch}: error {e:.3e}");
        worst = worst.max(e);
    }
    assert!(worst < TOL);
}

#[test]
fn adding_a_large_constant_does_not_change_the_result() {
    // Small integers with mean near 0 (where f64 is very accurate) versus the same
    // integers plus 1e12.
    let (small, labels) = offset_data(35, 10_000, 30, 0.0, 10.0, true);
    let shifted = small.mapv(|v| v + 1e12);
    let xs: Array2<i64> = small.mapv(|v| v as i64);
    let xl: Array2<i64> = shifted.mapv(|v| v as i64);
    for batch in [10_000, 123] {
        let a = run(&xs, &labels, 4, batch);
        let b = run(&xl, &labels, 4, batch);
        let e = max_err(&a, &b);
        eprintln!("offset invariance, batch {batch}: {e:.3e}");
        assert!(e < TOL);
    }
}

#[test]
fn merging_accumulators_at_1e12_keeps_the_accuracy() {
    let (x, labels) = offset_data(36, 12_000, 20, 1e12, 10.0, true);
    let xi: Array2<i64> = x.mapv(|v| v as i64);
    let want = reference_t(&x, &labels.to_vec(), 0, 1, 4);
    // 12 accumulators of 1000 traces each, merged in a chain.
    let mut total = MomentAccumulator::new(20, 4).unwrap();
    for part in 0..12 {
        let r = part * 1000..(part + 1) * 1000;
        let mut a = MomentAccumulator::new(20, 4).unwrap();
        a.update(xi.slice(s![r.clone(), ..]), labels.slice(s![r]))
            .unwrap();
        total.merge(&a).unwrap();
    }
    let e = max_err(&total.t_values(0, 1), &want);
    eprintln!("12 merged accumulators at 1e12: {e:.3e}");
    assert!(e < TOL);
}
