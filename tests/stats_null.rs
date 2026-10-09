//! Under the null hypothesis (equal distributions), the p-values should be close to uniform.
//!
//! The values are Binomial(16, 1/2), drawn as the population count of 16 random bits. Both
//! classes have 2000 traces. The distribution is discrete and has sparse tails, so the p-values
//! of a chi-squared test are only approximately uniform. This test measures how close they are
//! with and without merging of sparse bins, for Pearson and G.
//!
//! The test is `#[ignore]`: it draws 40,000 tests per variant and is slow. Run it with
//! `cargo test --release --test stats_null -- --ignored --nocapture`. Runtime on a heavily
//! loaded machine: 9.1 s user CPU, 17 s wall (release build, 2026-10-08). The debug build is
//! about 20 times slower.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use scasim::stats::{Binning, HistAccumulator, Statistic, TestOptions};

/// Collects the p-values of `batches * samples` independent null tests.
fn null_p_values(opts: &TestOptions, per_class: usize, samples: usize, batches: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(20261008);
    let n = 2 * per_class;
    let labels = Array1::from_iter((0..n).map(|i| u16::from(i >= per_class)));
    let mut all = Vec::new();
    for _ in 0..batches {
        let mut traces = Array2::<u8>::zeros((n, samples));
        for v in traces.iter_mut() {
            *v = (rng.next_u32() & 0xFFFF).count_ones() as u8;
        }
        let mut acc = HistAccumulator::new(samples, Binning::Exact);
        acc.update(traces.view(), labels.view()).unwrap();
        all.extend(
            acc.test_pair(0, 1, opts)
                .unwrap()
                .iter()
                .filter(|r| r.is_valid())
                .map(|r| 10.0_f64.powf(-r.neg_log10_p)),
        );
    }
    all
}

fn fraction_below(p: &[f64], alpha: f64) -> f64 {
    p.iter().filter(|&&x| x <= alpha).count() as f64 / p.len() as f64
}

/// Kolmogorov-Smirnov distance to the uniform distribution.
fn ks(p: &[f64]) -> f64 {
    let mut s = p.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len() as f64;
    s.iter()
        .enumerate()
        .map(|(i, &x)| {
            ((i as f64 + 1.0) / n - x)
                .abs()
                .max((x - i as f64 / n).abs())
        })
        .fold(0.0, f64::max)
}

fn report(label: &str, p: &[f64]) -> Vec<f64> {
    let alphas = [0.5, 0.1, 0.05, 0.01, 0.001];
    let fractions: Vec<f64> = alphas.iter().map(|&a| fraction_below(p, a)).collect();
    eprintln!(
        "{label:<28} n = {}  P(p<=0.5) {:.4}  0.1: {:.4}  0.05: {:.4}  0.01: {:.5}  0.001: {:.5}  KS {:.4}",
        p.len(),
        fractions[0],
        fractions[1],
        fractions[2],
        fractions[3],
        fractions[4],
        ks(p)
    );
    fractions
}

#[test]
#[ignore = "slow statistical run; use --ignored"]
fn p_values_are_approximately_uniform() {
    let (per_class, samples, batches) = (2000, 4000, 10);
    for statistic in [Statistic::Pearson, Statistic::G] {
        for min_expected in [20.0, 5.0, 0.0] {
            let opts = TestOptions {
                statistic,
                min_expected,
            };
            let p = null_p_values(&opts, per_class, samples, batches);
            let f = report(&format!("{statistic:?}, min_expected {min_expected}"), &p);
            if min_expected > 0.0 && statistic == Statistic::Pearson {
                // 40000 tests: the standard error of a fraction near 0.05 is 0.0011, and of a
                // fraction near 0.01 is 0.0005. Allow four standard errors plus 5 percent
                // relative error for the discreteness of the tables.
                for (&alpha, &got) in [0.5, 0.1, 0.05, 0.01].iter().zip(&f) {
                    let se = (alpha * (1.0 - alpha) / p.len() as f64).sqrt();
                    assert!(
                        (got - alpha).abs() < 4.0 * se + 0.05 * alpha,
                        "Pearson with merging: P(p <= {alpha}) = {got}"
                    );
                }
            }
        }
    }
}
