use ndarray::{Array1, Array2, s};
use scasim::stats::ttest::MomentAccumulator;
use scasim::stats::{Binning, HistAccumulator};

/// The bit patterns of all values, so that `-0.0` differs from `0.0` and NaN payloads count.
fn bits(a: &ndarray::Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

#[path = "common/ttest_data.rs"]
mod common;

fn feed_hist(acc: &mut HistAccumulator, x: &Array2<i64>, labels: &Array1<u16>, batch: usize) {
    for start in (0..x.nrows()).step_by(batch) {
        let end = (start + batch).min(x.nrows());
        acc.update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
            .unwrap();
    }
}

#[test]
fn matches_moment_accumulator_for_integer_data() {
    let mut worst: f64 = 0.0;
    for (n, ns) in [(300, 1), (777, 31), (1200, 33)] {
        let (raw, labels) = common::gen_data(n as u64, n, ns, 3);
        let x = raw.mapv(|v| v.round() as i64);
        let mut h = HistAccumulator::new(ns, Binning::Exact);
        feed_hist(&mut h, &x, &labels, 113);
        let mut m = MomentAccumulator::new(ns, 4).unwrap();
        m.update(x.view(), labels.view()).unwrap();
        for a in 0..3u16 {
            for b in 0..3u16 {
                if a == b {
                    continue;
                }
                let got = h.t_values(a, b, 4).unwrap();
                let want = m.t_values(a, b);
                let err = common::max_err(&got, &want);
                worst = worst.max(err);
            }
        }
    }
    eprintln!("histogram versus moments worst relative error: {worst:.3e}");
    assert!(worst <= 1e-11);
}

#[test]
fn merge_and_thread_count_are_bit_exact() {
    let (raw, labels) = common::gen_data(91, 2048, 35, 3);
    let x = raw.mapv(|v| v.round() as i64);
    let mut whole = HistAccumulator::new(35, Binning::Exact);
    whole.update(x.view(), labels.view()).unwrap();
    let want = whole.t_values(0, 2, 4).unwrap();
    for chunk in [1, 7, 128, 511] {
        let mut parts = Vec::new();
        for start in (0..x.nrows()).step_by(chunk) {
            let end = (start + chunk).min(x.nrows());
            let mut part = HistAccumulator::new(35, Binning::Exact);
            part.update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
                .unwrap();
            parts.push(part);
        }
        let mut forward = parts[0].clone();
        for p in &parts[1..] {
            forward.merge(p).unwrap();
        }
        assert_eq!(bits(&forward.t_values(0, 2, 4).unwrap()), bits(&want));
        let mut backward = parts.last().unwrap().clone();
        for p in parts[..parts.len() - 1].iter().rev() {
            backward.merge(p).unwrap();
        }
        assert_eq!(bits(&backward.t_values(0, 2, 4).unwrap()), bits(&want));
        let mut tree = parts.clone();
        while tree.len() > 1 {
            let mut next = Vec::new();
            for pair in tree.chunks(2) {
                let mut left = pair[0].clone();
                if let Some(right) = pair.get(1) {
                    left.merge(right).unwrap();
                }
                next.push(left);
            }
            tree = next;
        }
        assert_eq!(bits(&tree[0].t_values(0, 2, 4).unwrap()), bits(&want));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let one = pool.install(|| whole.t_values(0, 2, 4).unwrap());
    assert_eq!(bits(&one), bits(&want));
}

#[test]
fn rejects_fixed_bins_and_zero_order_and_handles_degenerate_data() {
    use scasim::stats::error::StatsError;
    let mut fixed = HistAccumulator::new(1, Binning::fixed(0.0, 1.0).unwrap());
    fixed
        .update(ndarray::array![[1i64]].view(), ndarray::array![0u16].view())
        .unwrap();
    assert_eq!(fixed.t_values(0, 1, 1), Err(StatsError::NotExactBinning));
    let mut exact = HistAccumulator::new(1, Binning::Exact);
    assert_eq!(exact.t_values(0, 1, 0), Err(StatsError::ZeroOrder));
    let x = ndarray::array![[1i64], [1], [2], [2]];
    let y = ndarray::array![0u16, 0, 1, 1];
    exact.update(x.view(), y.view()).unwrap();
    let t = exact.t_values(0, 1, 1).unwrap();
    assert_eq!(t[[0, 0]], f64::NEG_INFINITY);
    let same = exact.t_values(0, 0, 1).unwrap();
    assert!(same[[0, 0]].is_nan());
    let empty = HistAccumulator::new(1, Binning::Exact);
    assert!(empty.t_values(0, 1, 1).unwrap().iter().all(|v| v.is_nan()));
    assert!(exact.t_values(0, 99, 1).unwrap().iter().all(|v| v.is_nan()));
    let mut one = HistAccumulator::new(1, Binning::Exact);
    one.update(
        ndarray::array![[4i64], [4], [5]].view(),
        ndarray::array![0u16, 0, 1].view(),
    )
    .unwrap();
    assert!(one.t_values(0, 1, 1).unwrap()[[0, 0]].is_nan());
    let mut zeros = HistAccumulator::new(1, Binning::Exact);
    zeros
        .update(
            ndarray::array![[0i64], [0], [0], [0]].view(),
            ndarray::array![0u16, 0, 1, 1].view(),
        )
        .unwrap();
    assert!(zeros.t_values(0, 1, 4).unwrap().iter().all(|v| v.is_nan()));
}

#[test]
fn thread_count_does_not_change_bits() {
    let (raw, labels) = common::gen_data(192, 2048, 37, 3);
    let x = raw.mapv(|v| v.round() as i64);
    let mut h = HistAccumulator::new(37, Binning::Exact);
    h.update(x.view(), labels.view()).unwrap();
    let want = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| h.t_values(0, 2, 4).unwrap());
    for threads in [2, 3, 8] {
        let got = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| h.t_values(0, 2, 4).unwrap());
        assert_eq!(bits(&got), bits(&want), "{threads} threads");
    }
}

#[test]
fn matches_i128_reference_at_large_offset() {
    let (raw, labels) = common::gen_data(34, 800, 20, 2);
    let x = raw.mapv(|v| (1.0e12 + (v - 15000.0) / 20.0).round() as i64);
    let mut h = HistAccumulator::new(20, Binning::Exact);
    h.update(x.view(), labels.view()).unwrap();
    let got = h.t_values(0, 1, 4).unwrap();
    // The integer bins and their first centered sum are exact. This direct two-pass
    // reference widens every deviation and power to i128 before its final division.
    let mut want = Array2::from_elem((4, 20), f64::NAN);
    for j in 0..20 {
        let stats = |class: u16| {
            let values: Vec<i128> = (0..x.nrows())
                .filter(|&i| labels[i] == class)
                .map(|i| i128::from(x[[i, j]]))
                .collect();
            let n = values.len() as i128;
            let origin = values[0];
            let sum: i128 = values.iter().map(|&v| v - origin).sum();
            let mut cm = [0.0; 9];
            for (p, value) in cm.iter_mut().enumerate().skip(2) {
                let acc: i128 = values
                    .iter()
                    .map(|&v| (n * (v - origin) - sum).pow(p as u32))
                    .sum();
                *value = acc as f64 / (n as f64).powi(p as i32 + 1);
            }
            (n as f64, origin as f64, sum as f64 / n as f64, cm)
        };
        let (na, oa, xa, ca) = stats(0);
        let (nb, ob, xb, cb) = stats(1);
        for k in 1..=4 {
            let (diff, va, vb) = match k {
                1 => ((oa - ob) + (xa - xb), ca[2], cb[2]),
                2 => (ca[2] - cb[2], ca[4] - ca[2] * ca[2], cb[4] - cb[2] * cb[2]),
                _ => {
                    let f = |cm: &[f64; 9], k: usize| {
                        let m = cm[k] / cm[2].sqrt().powi(k as i32);
                        let v = (cm[2 * k] - cm[k] * cm[k]) / cm[2].powi(k as i32);
                        (m, v)
                    };
                    let (ya, va) = f(&ca, k);
                    let (yb, vb) = f(&cb, k);
                    (ya - yb, va, vb)
                }
            };
            want[[k - 1, j]] = diff / (va / na + vb / nb).sqrt();
        }
    }
    let err = common::max_err(&got, &want);
    eprintln!("i128 reference at 1e12 worst relative error: {err:.3e}");
    assert!(err <= 1e-12);
}

#[test]
fn planted_fourth_order_leak_and_variance_case_are_detected() {
    use scasim::stats::chi2::{Statistic, TestOptions};
    let n = 5000;
    let mut labels = Vec::with_capacity(2 * n);
    let mut values = Vec::with_capacity(2 * n);
    for i in 0..n {
        labels.push(0);
        values.push(if i % 2 == 0 { 2i64 } else { 6 });
        labels.push(1);
        values.push(match i % 8 {
            0 => 0,
            1..=6 => 4,
            _ => 8,
        });
    }
    let x = Array2::from_shape_vec((2 * n, 1), values).unwrap();
    let labels = Array1::from_vec(labels);
    let mut h = HistAccumulator::new(1, Binning::Exact);
    h.update(x.view(), labels.view()).unwrap();
    let t = h.t_values(0, 1, 4).unwrap();
    assert!(t.slice(s![..3, ..]).iter().all(|v| v.abs() < 4.5), "{t:?}");
    assert!(t[[3, 0]].abs() > 4.5, "{t:?}");
    let chi = h
        .test_pair(
            0,
            1,
            &TestOptions {
                statistic: Statistic::Pearson,
                min_expected: 20.0,
            },
        )
        .unwrap();
    assert!(chi[0].neg_log10_p > 5.0, "{}", chi[0].neg_log10_p);

    let mut labels_a = Vec::new();
    let mut values_a = Vec::new();
    for i in 0..2000 {
        labels_a.extend([0, 1]);
        values_a.extend([if i % 2 == 0 { 0i64 } else { 2 }, 1]);
    }
    let xa = Array2::from_shape_vec((4000, 1), values_a).unwrap();
    let la = Array1::from_vec(labels_a);
    let mut ha = HistAccumulator::new(1, Binning::Exact);
    ha.update(xa.view(), la.view()).unwrap();
    let ta = ha.t_values(0, 1, 2).unwrap();
    assert!(ta[[0, 0]].abs() < 4.5);
    assert!(ta[[1, 0]].abs() > 4.5);
}
