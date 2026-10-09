//! Calibration of the chi-squared approximation: how often does a null table (equal class
//! distributions) give `-log10(p) >= k`? Under a perfect approximation the fraction is 10^-k.
//!
//! For each scenario (a bin probability vector and a number of traces per class), the example
//! draws many pairs of multinomial rows, runs each statistic with several merge thresholds, and
//! prints `observed / nominal` for k = 1..5 together with the number of events behind it.
//!
//! Usage: `cargo run --release --example chi2_validity -- [--sims 2000000]`

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Binomial, Distribution};
use rayon::prelude::*;
use scasim::stats::{Statistic, TestOptions, Workspace, test_table};

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn multinomial(rng: &mut SmallRng, n: u64, p: &[f64], out: &mut [u32]) -> Result<(), DynError> {
    let (mut rem_n, mut rem_p) = (n, 1.0_f64);
    for (j, &pj) in p.iter().enumerate() {
        if j + 1 == p.len() || rem_n == 0 {
            out[j] = rem_n as u32;
            rem_n = 0;
            continue;
        }
        let q = (pj / rem_p).clamp(0.0, 1.0);
        let k = Binomial::new(rem_n, q)?.sample(rng);
        out[j] = k as u32;
        rem_n -= k;
        rem_p -= pj;
    }
    Ok(())
}

fn normalized(w: Vec<f64>) -> Vec<f64> {
    let s: f64 = w.iter().sum();
    w.into_iter().map(|x| x / s).collect()
}

fn binom_pmf(n: u32, p: f64) -> Vec<f64> {
    let mut v = vec![0.0; n as usize + 1];
    let mut c = 1.0_f64; // binomial coefficient
    for k in 0..=n {
        v[k as usize] = c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
        c = c * f64::from(n - k) / f64::from(k + 1);
    }
    v
}

fn main() -> Result<(), DynError> {
    let args: Vec<String> = std::env::args().collect();
    let sims: usize = match args.iter().position(|a| a == "--sims") {
        None => 2_000_000,
        Some(i) => {
            let value = args.get(i + 1).ok_or("option --sims needs a value")?;
            value
                .parse()
                .map_err(|_| format!("option --sims: cannot read {value:?}"))?
        }
    };
    if sims == 0 {
        return Err("option --sims: need at least 1".into());
    }
    let scenarios: Vec<(&str, Vec<f64>)> = vec![
        ("Binomial(16,1/2), 17 bins", binom_pmf(16, 0.5)),
        (
            "geometric 0.6^j, 24 bins",
            normalized((0..24).map(|j| 0.6_f64.powi(j)).collect()),
        ),
        ("uniform, 64 bins", normalized(vec![1.0; 64])),
        (
            "Zipf 1/(j+1), 200 bins",
            normalized((0..200).map(|j| 1.0 / (j as f64 + 1.0)).collect()),
        ),
    ];
    let variants: Vec<(Statistic, f64)> = vec![
        (Statistic::Pearson, 0.0),
        (Statistic::Pearson, 5.0),
        (Statistic::Pearson, 20.0),
        (Statistic::G, 0.0),
        (Statistic::G, 5.0),
        (Statistic::G, 20.0),
    ];
    const LEVELS: usize = 5;
    println!(
        "{sims} null tables per scenario; entries are observed/nominal (events) for -log10 p >= 1..5"
    );
    for (name, pmf) in &scenarios {
        for &n in &[100_u64, 500, 2000] {
            let chunk = sims.min(20_000);
            let counts: Vec<Vec<[u64; LEVELS]>> = (0..sims / chunk)
                .into_par_iter()
                .map(|c| -> Result<_, DynError> {
                    let mut rng = SmallRng::seed_from_u64(0xC0FFEE + c as u64 * 7919 + n);
                    let mut a = vec![0_u32; pmf.len()];
                    let mut b = vec![0_u32; pmf.len()];
                    let mut ws = Workspace::default();
                    let mut acc = vec![[0_u64; LEVELS]; variants.len()];
                    for _ in 0..chunk {
                        multinomial(&mut rng, n, pmf, &mut a)?;
                        multinomial(&mut rng, n, pmf, &mut b)?;
                        for (v, &(statistic, min_expected)) in variants.iter().enumerate() {
                            let opts = TestOptions {
                                statistic,
                                min_expected,
                            };
                            let r = test_table(&[&a, &b], &opts, &mut ws)?;
                            for (k, count) in acc[v].iter_mut().enumerate() {
                                if r.is_valid() && r.neg_log10_p >= (k + 1) as f64 {
                                    *count += 1;
                                }
                            }
                        }
                    }
                    Ok(acc)
                })
                .collect::<Result<_, _>>()?;
            let total = (sims / chunk * chunk) as f64;
            println!(
                "\n{name}, n = {n} per class (expected count per cell in the mode: {:.1})",
                n as f64 * pmf.iter().cloned().fold(0.0, f64::max)
            );
            println!(
                "{:<22} {:>14} {:>14} {:>14} {:>14} {:>14}",
                "variant", "p<=1e-1", "1e-2", "1e-3", "1e-4", "1e-5"
            );
            for (v, &(statistic, min_expected)) in variants.iter().enumerate() {
                let mut line = format!("{:<22}", format!("{statistic:?} merge@{min_expected}"));
                for k in 0..LEVELS {
                    let events: u64 = counts.iter().map(|c| c[v][k]).sum();
                    let nominal = 10.0_f64.powi(-(k as i32 + 1));
                    line += &format!(" {:>8.2} ({:>5})", events as f64 / total / nominal, events);
                }
                println!("{line}");
            }
        }
    }
    Ok(())
}
