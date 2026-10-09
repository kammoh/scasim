//! Accuracy of the chi-squared test and of the special functions against external references.
//!
//! * SciPy `chi2_contingency(correction=False)` for the Pearson and G statistics, and mpmath
//!   (50 digits) for G (`scipy_tables.json`).
//! * mpmath for `-log10(p)` of the chi-squared survival function (`pvalue_mpmath.json`), and
//!   `statrs` as a linear-domain comparison.
//! * SciPy `norm.isf` for the inverse normal (`norm_isf.json`).
//! * The histogram accumulator against the table test, bit for bit.
//!
//! The fixtures are in `tests/fixtures/stats/`. Their generators are in `scripts/fixtures/`.

use ndarray::{Array1, Array2};
use scasim::stats::special::{chi2_ln_sf, ln_gamma, neg_log10_chi2_sf, normal_isf};
use scasim::stats::threshold::t_bonferroni;
use scasim::stats::{
    Binning, HistAccumulator, Statistic, TestOptions, TestResult, Workspace, test_table,
};
use serde_json::Value;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Reads a fixture file from `tests/fixtures/stats/`.
fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/stats/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

// ---- SciPy and mpmath tables --------------------------------------------------------------

struct Reference {
    stat: f64,
    dof: u32,
    p: f64,
}

struct Case {
    name: String,
    table: Vec<Vec<u32>>,
    g_mpmath: f64,
    pearson: Reference,
    g: Reference,
}

fn reference(v: &Value) -> Reference {
    Reference {
        stat: v["stat"].as_f64().unwrap(),
        dof: v["dof"].as_u64().unwrap() as u32,
        p: v["p"].as_f64().unwrap(),
    }
}

fn load_cases() -> Vec<Case> {
    let v = fixture("scipy_tables.json");
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().to_string(),
            table: c["table"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as u32)
                        .collect()
                })
                .collect(),
            g_mpmath: c["g_mpmath"].as_f64().unwrap(),
            pearson: reference(&c["pearson"]),
            g: reference(&c["g"]),
        })
        .collect()
}

fn run(table: &[Vec<u32>], statistic: Statistic) -> TestResult {
    let rows: Vec<&[u32]> = table.iter().map(Vec::as_slice).collect();
    let opts = TestOptions {
        statistic,
        min_expected: 0.0,
    };
    test_table(&rows, &opts, &mut Workspace::default()).expect("equal row lengths")
}

#[derive(Default)]
struct Worst {
    /// Relative error of the statistic.
    stat_rel: f64,
    /// Absolute error of the statistic, scaled by `1 + statistic`.
    stat_abs_scaled: f64,
    /// Error of `-log10(p)`: absolute, and relative to `max(1, value)`.
    nlp_abs: f64,
    nlp_rel: f64,
    n_p: usize,
}

fn compare(case: &Case, statistic: Statistic, reference: &Reference, worst: &mut Worst) {
    let got = run(&case.table, statistic);
    // NaN compares false and `f64::max` drops it, so check finiteness first.
    assert!(
        got.statistic.is_finite() && got.neg_log10_p.is_finite(),
        "{} {statistic:?}: {got:?}",
        case.name
    );
    assert!(
        reference.stat.is_finite(),
        "{}: reference {}",
        case.name,
        reference.stat
    );
    assert_eq!(got.dof, reference.dof, "{} {:?}", case.name, statistic);
    if reference.stat > 0.0 {
        worst.stat_rel = worst
            .stat_rel
            .max((got.statistic - reference.stat).abs() / reference.stat);
    }
    worst.stat_abs_scaled = worst
        .stat_abs_scaled
        .max((got.statistic - reference.stat).abs() / (1.0 + reference.stat));
    // SciPy underflows below about 1e-300, so compare only where it returns a usable p.
    if reference.p > 1.0e-300 && reference.dof > 0 {
        let want = -reference.p.log10();
        let abs = (got.neg_log10_p - want).abs();
        worst.nlp_abs = worst.nlp_abs.max(abs);
        worst.nlp_rel = worst.nlp_rel.max(abs / want.max(1.0));
        worst.n_p += 1;
    }
}

#[test]
fn pearson_matches_scipy() {
    let cases = load_cases();
    let mut worst = Worst::default();
    for case in &cases {
        compare(case, Statistic::Pearson, &case.pearson, &mut worst);
    }
    eprintln!(
        "Pearson vs SciPy over {} tables: statistic rel err {:.2e}; -log10 p over {} finite p: abs {:.2e}, rel {:.2e}",
        cases.len(),
        worst.stat_rel,
        worst.n_p,
        worst.nlp_abs,
        worst.nlp_rel
    );
    assert!(worst.stat_rel < 1.0e-12);
    assert!(worst.nlp_rel < 1.0e-11);
}

/// SciPy computes G as a float64 sum of `F * log(F / E)`. For near-null tables with large counts
/// this loses digits to cancellation (an absolute error near 5e-10 for N = 6e6). The crate uses
/// the cancellation-free form, so G is checked against mpmath (50 digits) with a tight tolerance,
/// and against SciPy with a tolerance that allows for SciPy's own error.
#[test]
fn g_matches_mpmath_and_scipy() {
    let cases = load_cases();
    let mut vs_scipy = Worst::default();
    let mut worst_mp = 0.0_f64;
    for case in &cases {
        compare(case, Statistic::G, &case.g, &mut vs_scipy);
        let got = run(&case.table, Statistic::G).statistic;
        assert!(got.is_finite(), "{}: G statistic {got}", case.name);
        if case.g_mpmath > 0.0 {
            worst_mp = worst_mp.max((got - case.g_mpmath).abs() / case.g_mpmath);
        }
    }
    eprintln!(
        "G vs mpmath: statistic rel err {worst_mp:.2e}. G vs SciPy: statistic abs err/(1+stat) {:.2e}, \
         -log10 p rel {:.2e} over {} finite p",
        vs_scipy.stat_abs_scaled, vs_scipy.nlp_rel, vs_scipy.n_p
    );
    assert!(worst_mp < 1.0e-13);
    assert!(vs_scipy.stat_abs_scaled < 2.0e-9);
    assert!(vs_scipy.nlp_rel < 1.0e-8);
}

/// Builds an accumulator with one sample from a table: row `i` is class `i`, column `j` is
/// value `j`, and each cell becomes that many traces.
fn accumulator_from(table: &[Vec<u32>], max_dense: usize) -> HistAccumulator {
    let mut values = Vec::new();
    let mut labels = Vec::new();
    for (i, row) in table.iter().enumerate() {
        for (j, &n) in row.iter().enumerate() {
            for _ in 0..n {
                values.push(j as u16);
                labels.push(i as u16);
            }
        }
    }
    let traces = Array2::from_shape_vec((values.len(), 1), values).unwrap();
    let mut acc = HistAccumulator::with_max_dense_bins(1, Binning::Exact, max_dense);
    acc.update(traces.view(), Array1::from(labels).view())
        .unwrap();
    acc
}

#[test]
fn accumulator_end_to_end_is_bit_identical_to_the_table_test() {
    let cases = load_cases();
    let mut checked = 0;
    for case in &cases {
        let total: u64 = case.table.iter().flatten().map(|&c| u64::from(c)).sum();
        if total > 120_000 {
            continue;
        }
        for max_dense in [4096, 0] {
            let acc = accumulator_from(&case.table, max_dense);
            for statistic in [Statistic::Pearson, Statistic::G] {
                let opts = TestOptions {
                    statistic,
                    min_expected: 0.0,
                };
                let got = acc.test_all(&opts).unwrap()[0];
                let want = run(&case.table, statistic);
                // Rows that are all zero never appear in the accumulator, so the row set is the
                // same after dropping. The statistic must not change by a single bit.
                assert_eq!(
                    got.statistic.to_bits(),
                    want.statistic.to_bits(),
                    "{}",
                    case.name
                );
                assert_eq!(got.dof, want.dof, "{}", case.name);
                assert_eq!(
                    got.neg_log10_p.to_bits(),
                    want.neg_log10_p.to_bits(),
                    "{}",
                    case.name
                );
            }
        }
        checked += 1;
    }
    eprintln!("{checked} tables checked through the accumulator (dense and sparse)");
    assert!(checked > 100);
}

#[test]
fn extreme_statistics_are_finite() {
    // Noise-free simulation: a fixed class with one value against a spread-out random class.
    let mut fixed = vec![0_u32; 33];
    fixed[16] = 1_000_000;
    let mut random = vec![0_u32; 33];
    for (j, c) in random.iter_mut().enumerate() {
        *c = 30_000 + (j as u32 * 37) % 500;
    }
    let r = run(&[fixed, random], Statistic::Pearson);
    eprintln!(
        "planted fixed-vs-random: statistic {:.3e}, -log10 p = {:.1}",
        r.statistic, r.neg_log10_p
    );
    assert!(r.neg_log10_p.is_finite() && r.neg_log10_p > 1.0e5);
}

// ---- p-values against mpmath ----------------------------------------------------------------

struct Point {
    dof: u32,
    x: f64,
    reference: f64,
}

fn load_points() -> Vec<Point> {
    let v = fixture("pvalue_mpmath.json");
    v["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Point {
            dof: p["dof"].as_u64().unwrap() as u32,
            x: p["x"].as_f64().unwrap(),
            reference: p["neg_log10_p"].as_str().unwrap().parse().unwrap(),
        })
        .collect()
}

#[test]
fn matches_mpmath() {
    let points = load_points();
    let mut worst = (0.0_f64, 0u32, 0.0_f64, 0.0_f64);
    for p in &points {
        let got = neg_log10_chi2_sf(p.x, p.dof);
        assert!(got.is_finite(), "dof {} x {}", p.dof, p.x);
        // Relative error, but never stricter than 1e-13 absolute for values near zero.
        let e = (got - p.reference).abs() / p.reference.abs().max(1.0);
        if e > worst.0 {
            worst = (e, p.dof, p.x, p.reference);
        }
    }
    eprintln!(
        "mine vs mpmath: {} points, worst error {:.3e} at dof {} x {} (want {})",
        points.len(),
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    assert!(worst.0 < 1.0e-12, "worst error {:?}", worst);
}

#[test]
fn statrs_comparison() {
    // statrs works with p itself, so it underflows when p < 1e-308 (-log10 p > 308).
    let points = load_points();
    let mut worst_ok = 0.0_f64;
    let mut n_ok = 0;
    let mut n_underflow = 0;
    let mut n_bad_before_underflow = 0;
    for p in &points {
        let d = ChiSquared::new(f64::from(p.dof)).unwrap();
        let sf_direct = d.sf(p.x);
        let got = -sf_direct.log10();
        if sf_direct == 0.0 || !got.is_finite() {
            n_underflow += 1;
            assert!(
                p.reference > 300.0,
                "statrs underflowed too early: {:?}",
                p.reference
            );
        } else {
            n_ok += 1;
            let e = (got - p.reference).abs() / p.reference.abs().max(1.0);
            worst_ok = worst_ok.max(e);
            if e > 1.0e-9 {
                n_bad_before_underflow += 1;
            }
        }
    }
    eprintln!(
        "statrs sf(): {n_ok} finite (worst error {worst_ok:.3e}, {n_bad_before_underflow} above 1e-9), \
         {n_underflow} underflowed to p = 0 or non-finite"
    );
}

#[test]
fn ln_gamma_matches_libm() {
    let mut worst = 0.0_f64;
    for i in 1..2000 {
        let z = f64::from(i) * 0.05;
        let want = libm::lgamma(z);
        let e = (ln_gamma(z) - want).abs() / want.abs().max(1.0);
        worst = worst.max(e);
    }
    eprintln!("ln_gamma vs libm::lgamma, z in (0, 100]: worst error {worst:.3e}");
    assert!(worst < 1.0e-14);
}

#[test]
fn sf_monotone_in_x() {
    for dof in [1u32, 2, 7, 64, 500] {
        let mut last = 0.0;
        for k in 0..2000 {
            let x = f64::from(k) * 0.5;
            let v = chi2_ln_sf(x, dof);
            assert!(v <= last + 1e-12, "dof {dof} x {x}: {v} > {last}");
            last = v;
        }
    }
}

// ---- inverse normal against SciPy ------------------------------------------------------------

fn load_isf() -> Vec<(f64, f64)> {
    let v = fixture("norm_isf.json");
    v["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                p["p"].as_str().unwrap().parse().unwrap(),
                p["z"].as_str().unwrap().parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn normal_isf_matches_scipy() {
    let points = load_isf();
    assert!(points.len() >= 200);
    let mut worst = (0.0_f64, 0.0_f64);
    for &(p, z) in &points {
        let got = normal_isf(p);
        assert!(got.is_finite(), "normal_isf({p:e}) = {got}");
        let e = (got - z).abs() / z.abs();
        if e > worst.0 {
            worst = (e, p);
        }
    }
    eprintln!(
        "normal_isf vs SciPy: {} points, worst relative error {:.3e} at p = {:e}",
        points.len(),
        worst.0,
        worst.1
    );
    assert!(worst.0 < 1.0e-14, "worst {worst:?}");
}

#[test]
fn t_bonferroni_matches_scipy_for_the_family_sizes_of_the_report() {
    let v = fixture("norm_isf.json");
    let points = load_isf();
    for m in v["bonferroni_m_values"].as_array().unwrap() {
        let m = m.as_u64().unwrap();
        let p = 1.0e-5 / (2.0 * m as f64);
        let &(_, z) = points
            .iter()
            .find(|(q, _)| *q == p)
            .unwrap_or_else(|| panic!("no fixture point for m = {m}"));
        let got = t_bonferroni(1.0e-5, m);
        assert!((got - z).abs() / z < 1.0e-14, "m = {m}: {got} versus {z}");
    }
    // The report case: alpha = 1e-5, 742 tests. SciPy gives 5.679903330994272.
    assert!((t_bonferroni(1.0e-5, 742) - 5.679_903_330_994_272).abs() < 1.0e-13);
}
