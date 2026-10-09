//! Tests against a naive two-pass reference, plus merge, batching, state round trip,
//! and edge cases.

use crate::common;

use common::{gen_data, max_err, reference_t, rounded};
use ndarray::{Array1, Array2, Axis, s};
use scasim::stats::error::StatsError;
use scasim::stats::ttest::{MomentAccumulator, TraceSample};

/// The tolerance on `|a - b| / max(1, |a|, |b|)` for comparisons with the reference.
const TOL: f64 = 1e-9;

fn feed<T: TraceSample>(
    acc: &mut MomentAccumulator,
    x: &Array2<T>,
    labels: &Array1<u16>,
    batch: usize,
) {
    let n = x.nrows();
    let mut start = 0;
    while start < n {
        let end = (start + batch).min(n);
        acc.update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
            .unwrap();
        start = end;
    }
}

fn to_f64<T: TraceSample>(x: &Array2<T>) -> Array2<f64> {
    x.mapv(|v| v.to_f64())
}

/// Compares all pairs of classes against the reference for the sample type `T`.
fn check_type<T: TraceSample>(name: &str, conv: impl Fn(f64) -> T) {
    let (n, ns, d) = (3000, 100, 4);
    let (x, labels) = gen_data(11, n, ns, 3);
    let xt: Array2<T> = x.mapv(&conv);
    let xr = to_f64(&xt);
    let lab: Vec<u16> = labels.to_vec();
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut acc, &xt, &labels, 700);
    let mut worst: f64 = 0.0;
    for a in 0..3u16 {
        for b in 0..3u16 {
            if a == b {
                continue;
            }
            let ours = acc.t_values(a, b);
            let reference = reference_t(&xr, &lab, a, b, d);
            let e = max_err(&ours, &reference);
            worst = worst.max(e);
        }
    }
    eprintln!("{name}: worst error against the reference {worst:.3e}");
    assert!(worst < TOL, "{name}: error {worst:e}");
}

#[test]
fn matches_naive_reference_for_all_sample_types() {
    check_type::<f64>("f64", |v| v);
    check_type::<f32>("f32", |v| v as f32);
    check_type::<u32>("u32", |v| v as u32);
    check_type::<u64>("u64", |v| v as u64);
    check_type::<i32>("i32", |v| v as i32 - 15000);
    check_type::<i64>("i64", |v| v as i64 - 15000);
    check_type::<u16>("u16", |v| v as u16);
    check_type::<i16>("i16", |v| v as i16 - 15000);
}

#[test]
fn matches_naive_reference_with_other_batch_sizes() {
    let (n, ns, d) = (2500, 70, 4);
    let (x, labels) = gen_data(12, n, ns, 3);
    let lab = labels.to_vec();
    for batch in [1usize, 5, 255, 256, 257, 2500] {
        let mut acc = MomentAccumulator::new(ns, d).unwrap();
        feed(&mut acc, &x, &labels, batch);
        let e = max_err(&acc.t_values(0, 2), &reference_t(&x, &lab, 0, 2, d));
        eprintln!("batch {batch}: error {e:.3e}");
        assert!(e < TOL);
    }
}

#[test]
fn order_k_does_not_depend_on_maximum_order() {
    let (n, ns) = (1500, 40);
    let (x, labels) = gen_data(13, n, ns, 2);
    let mut a4 = MomentAccumulator::new(ns, 4).unwrap();
    let mut a2 = MomentAccumulator::new(ns, 2).unwrap();
    feed(&mut a4, &x, &labels, 400);
    feed(&mut a2, &x, &labels, 400);
    let t4 = a4.t_values(0, 1);
    let t2 = a2.t_values(0, 1);
    let e = max_err(&t4.slice(s![..2, ..]).to_owned(), &t2);
    eprintln!("d=4 versus d=2: {e:.3e}");
    assert!(e < 1e-12);
}

#[test]
fn merging_two_halves_equals_one_pass() {
    let (n, ns, d) = (3000, 90, 4);
    let (x, labels) = gen_data(14, n, ns, 3);
    let mut whole = MomentAccumulator::new(ns, d).unwrap();
    whole.update(x.view(), labels.view()).unwrap();
    let mut first = MomentAccumulator::new(ns, d).unwrap();
    let mut second = MomentAccumulator::new(ns, d).unwrap();
    first
        .update(x.slice(s![..1234, ..]), labels.slice(s![..1234]))
        .unwrap();
    second
        .update(x.slice(s![1234.., ..]), labels.slice(s![1234..]))
        .unwrap();
    first.merge(&second).unwrap();
    for c in 0..3 {
        assert_eq!(first.count(c), whole.count(c));
    }
    let e = max_err(&first.t_values(0, 1), &whole.t_values(0, 1));
    eprintln!("merge halves: {e:.3e}");
    assert!(e < 1e-11);
}

#[test]
fn merge_is_associative_and_commutative_up_to_rounding() {
    let (n, ns, d) = (3000, 50, 3);
    let (x, labels) = gen_data(15, n, ns, 2);
    let parts: Vec<MomentAccumulator> = [0..700usize, 700..1100, 1100..3000]
        .into_iter()
        .map(|r| {
            let mut a = MomentAccumulator::new(ns, d).unwrap();
            a.update(x.slice(s![r.clone(), ..]), labels.slice(s![r]))
                .unwrap();
            a
        })
        .collect();
    // (A + B) + C
    let mut left = parts[0].clone();
    left.merge(&parts[1]).unwrap();
    left.merge(&parts[2]).unwrap();
    // A + (B + C)
    let mut bc = parts[1].clone();
    bc.merge(&parts[2]).unwrap();
    let mut right = parts[0].clone();
    right.merge(&bc).unwrap();
    // C + (B + A)
    let mut cba = parts[2].clone();
    let mut ba = parts[1].clone();
    ba.merge(&parts[0]).unwrap();
    cba.merge(&ba).unwrap();
    let e1 = max_err(&left.t_values(0, 1), &right.t_values(0, 1));
    let e2 = max_err(&left.t_values(0, 1), &cba.t_values(0, 1));
    eprintln!("associativity {e1:.3e}, commutativity {e2:.3e}");
    assert!(e1 < 1e-11 && e2 < 1e-11);
    // Merging also keeps the central sums right.
    let mut whole = MomentAccumulator::new(ns, d).unwrap();
    whole.update(x.view(), labels.view()).unwrap();
    for p in 2..=2 * d {
        let a = left.central_sum(0, p).unwrap();
        let b = whole.central_sum(0, p).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() <= 1e-10 * x.abs().max(y.abs()).max(1.0));
        }
    }
}

#[test]
fn batch_size_does_not_matter() {
    let (n, ns, d) = (3000, 45, 4);
    let (x, labels) = gen_data(16, n, ns, 2);
    let mut base = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut base, &x, &labels, 1000);
    let t_base = base.t_values(0, 1);
    for batch in [1usize, 7, 1000, 3000] {
        let mut acc = MomentAccumulator::new(ns, d).unwrap();
        feed(&mut acc, &x, &labels, batch);
        let e = max_err(&acc.t_values(0, 1), &t_base);
        eprintln!("batch size {batch}: {e:.3e}");
        assert!(e < 1e-11);
    }
}

#[test]
fn moments_round_trip() {
    let (n, ns, d) = (1200, 77, 3);
    let (x, labels) = gen_data(17, n, ns, 3);
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut acc, &x, &labels, 300);
    let m = acc.moments();
    assert_eq!(m.ns, ns);
    assert_eq!(m.d, d);
    assert_eq!(m.classes.len(), 3);
    assert_eq!(m.classes[0].data.len(), (2 * d + 1) * ns);
    assert_eq!(m.classes.iter().map(|c| c.count).sum::<u64>(), n as u64);
    let restored = MomentAccumulator::from_moments(m.clone()).unwrap();
    assert_eq!(restored.moments(), m);
    assert_eq!(restored.t_values(0, 1), acc.t_values(0, 1));
    // The restored accumulator keeps working: more traces and a merge give the same result.
    let (x2, l2) = gen_data(18, 500, ns, 3);
    let mut a1 = acc.clone();
    a1.update(x2.view(), l2.view()).unwrap();
    let mut a2 = restored;
    a2.update(x2.view(), l2.view()).unwrap();
    assert_eq!(a1.t_values(1, 2), a2.t_values(1, 2));
    // Mean and central sums are available through the plain data, too.
    let c0 = &m.classes[0];
    let mean = acc.mean(0).unwrap();
    for j in [0, 1, ns - 1] {
        assert_eq!(c0.mean(ns, j).unwrap(), mean[j]);
        assert_eq!(
            c0.central_sum(ns, 2, j).unwrap(),
            acc.central_sum(0, 2).unwrap()[j]
        );
    }
}

#[test]
fn moments_round_trip_through_json() {
    let (n, ns, d) = (900, 40, 2);
    let (x, labels) = gen_data(19, n, ns, 2);
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut acc, &x, &labels, 300);
    let text = serde_json::to_string(&acc.moments()).unwrap();
    let m: scasim::stats::ttest::Moments = serde_json::from_str(&text).unwrap();
    let restored = MomentAccumulator::from_moments(m).unwrap();
    assert_eq!(restored.moments(), acc.moments());
    assert_eq!(restored.t_values(0, 1), acc.t_values(0, 1));
}

#[test]
fn from_moments_rejects_invalid_state() {
    let ok = {
        let mut acc = MomentAccumulator::new(5, 2).unwrap();
        let x = Array2::<f64>::from_shape_fn((10, 5), |(i, j)| (i * j) as f64);
        let l = Array1::from_shape_fn(10, |i| (i % 2) as u16);
        acc.update(x.view(), l.view()).unwrap();
        acc.moments()
    };
    let mut bad = ok.clone();
    bad.d = 0;
    assert!(matches!(
        MomentAccumulator::from_moments(bad),
        Err(StatsError::ZeroOrder)
    ));
    let mut bad = ok.clone();
    bad.classes[1].data.pop();
    assert!(matches!(
        MomentAccumulator::from_moments(bad).unwrap_err(),
        StatsError::WrongMomentLength { label: 1, .. }
    ));
    let mut bad = ok.clone();
    bad.classes[1].label = 0;
    assert!(matches!(
        MomentAccumulator::from_moments(bad),
        Err(StatsError::DuplicateLabel(0))
    ));
    assert!(MomentAccumulator::from_moments(ok).is_ok());
}

#[test]
fn restore_then_merge_rejects_impossible_single_trace_moments() {
    use scasim::stats::ttest::{ClassMoments, Moments};

    let bad = Moments {
        ns: 1,
        d: 1,
        classes: vec![
            ClassMoments {
                label: 0,
                count: 1,
                data: vec![0.0, 0.0, 1.0],
            },
            ClassMoments {
                label: 1,
                count: 2,
                data: vec![1.0, 0.0, 0.0],
            },
        ],
    };
    assert!(MomentAccumulator::from_moments(bad).is_err());

    let valid = Moments {
        ns: 1,
        d: 1,
        classes: vec![
            ClassMoments {
                label: 0,
                count: 1,
                data: vec![0.0, 0.0, 0.0],
            },
            ClassMoments {
                label: 1,
                count: 2,
                data: vec![1.0, 0.0, 0.0],
            },
        ],
    };
    let mut restored = MomentAccumulator::from_moments(valid).unwrap();
    restored
        .update(ndarray::array![[0.0]].view(), ndarray::array![0u16].view())
        .unwrap();
    assert_eq!(restored.t_values(0, 1)[[0, 0]], f64::NEG_INFINITY);
}

#[test]
fn class_with_zero_or_one_trace_gives_nan() {
    let (ns, d) = (6, 3);
    let (x, labels) = gen_data(20, 200, ns, 2);
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    acc.update(x.view(), labels.view()).unwrap();
    // Class 2 was never seen.
    for t in [acc.t_values(0, 2), acc.t_values(2, 0), acc.t_values(2, 3)] {
        assert_eq!(t.dim(), (d, ns));
        assert!(t.iter().all(|v| v.is_nan()));
    }
    // Class 2 has exactly one trace.
    let one = x.slice(s![..1, ..]);
    acc.update(one, Array1::from_elem(1, 2u16).view()).unwrap();
    assert_eq!(acc.count(2), 1);
    for t in [acc.t_values(0, 2), acc.t_values(2, 1)] {
        assert!(t.iter().all(|v| v.is_nan()));
    }
    // The other classes are not affected.
    assert!(acc.t_values(0, 1).iter().all(|v| v.is_finite()));
    // A second trace makes the class usable.
    acc.update(x.slice(s![1..2, ..]), Array1::from_elem(1, 2u16).view())
        .unwrap();
    let t02 = acc.t_values(0, 2);
    assert!(t02.row(0).iter().all(|v| v.is_finite() || v.is_infinite()));
    // A fresh accumulator and an empty batch.
    let mut empty = MomentAccumulator::new(ns, d).unwrap();
    empty
        .update(x.slice(s![..0, ..]), labels.slice(s![..0]))
        .unwrap();
    assert!(empty.labels().is_empty());
    assert!(empty.t_values(0, 1).iter().all(|v| v.is_nan()));
}

#[test]
fn constant_sample_gives_nan_or_infinity_and_does_not_disturb_others() {
    let (n, ns, d) = (400, 8, 4);
    let (mut x, labels) = gen_data(21, n, ns, 2);
    // Sample 3 is constant. The value is not exactly representable in decimal.
    x.column_mut(3).fill(0.1);
    // Sample 4 is constant in both classes, with different values.
    for i in 0..n {
        x[[i, 4]] = if labels[i] == 0 { 2.0 } else { 3.0 };
    }
    // Sample 5 is constant in class 0 only, and class 1 varies.
    for i in 0..n {
        if labels[i] == 0 {
            x[[i, 5]] = 7.3;
        }
    }
    // Feed the traces in pieces, so that merges of constant data are part of the test.
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut acc, &x, &labels, 97);
    let t = acc.t_values(0, 1);
    for k in 0..d {
        assert!(t[[k, 3]].is_nan(), "order {}", k + 1);
    }
    assert_eq!(t[[0, 4]], f64::NEG_INFINITY);
    for k in 1..d {
        assert!(t[[k, 4]].is_nan(), "order {}", k + 1);
    }
    // The other samples agree with the reference. (The reference is not used for the
    // constant samples, because its mean of constant values is not exact.)
    let reference = reference_t(&x, &labels.to_vec(), 0, 1, d);
    let others = [0usize, 1, 2, 6, 7];
    let pick = |a: &Array2<f64>| a.select(Axis(1), &others);
    assert!(max_err(&pick(&t), &pick(&reference)) < TOL);
    for j in others {
        for k in 0..d {
            assert!(t[[k, j]].is_finite(), "sample {j}, order {}", k + 1);
        }
    }
    // Order 1 at sample 5: class 0 is constant, class 1 is not. The test is defined.
    assert!(t[[0, 5]].is_finite());
    assert!((t[[0, 5]] - reference[[0, 5]]).abs() < TOL * reference[[0, 5]].abs().max(1.0));
    // Order 2 is defined, too: the preprocessed variable of the constant class is 0.
    assert!(t[[1, 5]].is_finite());
    // From order 3 on, the standardized moments of a constant class are 0/0.
    for k in 2..d {
        assert!(t[[k, 5]].is_nan(), "order {}", k + 1);
    }
    // A constant f32 sample that is not exactly representable in decimal.
    let xf = Array2::<f32>::from_elem((300, 2), 1.0 / 3.0);
    let lf = Array1::from_shape_fn(300, |i| (i % 2) as u16);
    let mut accf = MomentAccumulator::new(2, 2).unwrap();
    feed(&mut accf, &xf, &lf, 50);
    assert!(accf.t_values(0, 1).iter().all(|v| v.is_nan()));
    assert!(accf.central_sum(0, 2).unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn other_labels_do_not_change_the_result() {
    let (n, ns, d) = (2000, 37, 3);
    let (x, labels) = gen_data(22, n, ns, 4);
    let mut all = MomentAccumulator::new(ns, d).unwrap();
    all.update(x.view(), labels.view()).unwrap();
    // Keep only the traces of classes 1 and 3.
    let keep: Vec<usize> = (0..n)
        .filter(|&i| labels[i] == 1 || labels[i] == 3)
        .collect();
    let xs = x.select(Axis(0), &keep);
    let ls = labels.select(Axis(0), &keep);
    let mut pair = MomentAccumulator::new(ns, d).unwrap();
    pair.update(xs.view(), ls.view()).unwrap();
    assert_eq!(all.t_values(1, 3), pair.t_values(1, 3));
    assert_eq!(all.t_values(3, 1), -&pair.t_values(1, 3));
    assert_eq!(all.labels(), vec![0, 1, 2, 3]);
}

#[test]
fn sizes_around_the_block_width_and_many_classes() {
    for ns in [1usize, 2, 31, 32, 33, 63, 64, 65, 100] {
        let (x, labels) = gen_data(23 + ns as u64, 600, ns, 2);
        let mut acc = MomentAccumulator::new(ns, 3).unwrap();
        feed(&mut acc, &x, &labels, 128);
        let e = max_err(
            &acc.t_values(0, 1),
            &reference_t(&x, &labels.to_vec(), 0, 1, 3),
        );
        assert!(e < TOL, "ns {ns}: {e:e}");
        let restored = MomentAccumulator::from_moments(acc.moments()).unwrap();
        assert_eq!(restored.t_values(0, 1), acc.t_values(0, 1));
    }
    // Many classes, each with a few traces.
    let (x, _) = gen_data(24, 5000, 10, 1);
    let labels = Array1::from_shape_fn(5000, |i| (i % 250) as u16 * 200);
    let mut acc = MomentAccumulator::new(10, 2).unwrap();
    acc.update(x.view(), labels.view()).unwrap();
    assert_eq!(acc.labels().len(), 250);
    assert_eq!(acc.count(200), 20);
    let t = acc.t_values(0, 200 * 249);
    let e = max_err(&t, &reference_t(&x, &labels.to_vec(), 0, 200 * 249, 2));
    assert!(e < TOL);
}

#[test]
fn accepts_views_with_unusual_strides() {
    let (n, ns, d) = (500, 20, 2);
    let (x, labels) = gen_data(25, n, ns, 2);
    let mut want = MomentAccumulator::new(ns, d).unwrap();
    want.update(x.view(), labels.view()).unwrap();
    // A transposed copy in Fortran order viewed through `t()`.
    let mut xt = Array2::<f64>::zeros((ns, n));
    xt.assign(&x.t());
    let mut a = MomentAccumulator::new(ns, d).unwrap();
    a.update(xt.t(), labels.view()).unwrap();
    assert_eq!(a.t_values(0, 1), want.t_values(0, 1));
    // A view of a wider array, so the row stride differs from the number of samples.
    let wide = Array2::from_shape_fn((n, ns + 9), |(i, j)| if j < ns { x[[i, j]] } else { 0.0 });
    let mut b = MomentAccumulator::new(ns, d).unwrap();
    b.update(wide.slice(s![.., ..ns]), labels.view()).unwrap();
    assert_eq!(b.t_values(0, 1), want.t_values(0, 1));
    // Labels with a stride.
    let doubled = Array1::from_shape_fn(2 * n, |i| if i % 2 == 0 { labels[i / 2] } else { 9 });
    let mut c = MomentAccumulator::new(ns, d).unwrap();
    c.update(x.view(), doubled.slice(s![..;2])).unwrap();
    assert_eq!(c.t_values(0, 1), want.t_values(0, 1));
}

#[test]
fn integer_traces_give_the_same_state_as_the_same_floats() {
    let (n, ns, d) = (800, 33, 3);
    let (x, labels) = gen_data(26, n, ns, 2);
    let xr = rounded(&x);
    let xi: Array2<i32> = xr.mapv(|v| v as i32);
    let xu: Array2<u64> = xr.mapv(|v| v as u64);
    let mut a = MomentAccumulator::new(ns, d).unwrap();
    let mut b = MomentAccumulator::new(ns, d).unwrap();
    let mut c = MomentAccumulator::new(ns, d).unwrap();
    a.update(xr.view(), labels.view()).unwrap();
    b.update(xi.view(), labels.view()).unwrap();
    c.update(xu.view(), labels.view()).unwrap();
    assert_eq!(a.moments(), b.moments());
    assert_eq!(a.moments(), c.moments());
}

#[test]
fn result_does_not_depend_on_the_number_of_threads() {
    let (n, ns, d) = (6000, 300, 3);
    let (x, labels) = gen_data(27, n, ns, 3);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut acc = MomentAccumulator::new(ns, d).unwrap();
            feed(&mut acc, &x, &labels, 2500);
            let mut other = MomentAccumulator::new(ns, d).unwrap();
            other
                .update(x.slice(s![..1000, ..]), labels.slice(s![..1000]))
                .unwrap();
            acc.merge(&other).unwrap();
            (acc.moments(), acc.t_values(0, 1), acc.t_values(2, 1))
        })
    };
    let base = run(1);
    for threads in [2, 3, 8] {
        assert!(run(threads) == base, "{threads} threads");
    }
}

#[test]
fn orders_above_four_use_the_general_code() {
    let (n, ns, d) = (2000, 50, 6);
    let (x, labels) = gen_data(28, n, ns, 2);
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut acc, &x, &labels, 600);
    let t = acc.t_values(0, 1);
    assert_eq!(t.dim(), (d, ns));
    // Order 5 and 6 are noisy but defined. Compare all six orders with the reference.
    let e = max_err(&t, &reference_t(&x, &labels.to_vec(), 0, 1, d));
    eprintln!("d=6: {e:.3e}");
    assert!(e < TOL);
}

#[test]
fn long_traces_and_many_traces_cross_range_and_wave_boundaries() {
    // More than 8192 sample points: the update handles them in two ranges.
    let (n, ns, d) = (300, 8192 + 100, 2);
    let (x, labels) = gen_data(29, n, ns, 2);
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    feed(&mut acc, &x, &labels, 120);
    let e = max_err(
        &acc.t_values(0, 1),
        &reference_t(&x, &labels.to_vec(), 0, 1, d),
    );
    eprintln!("long traces: {e:.3e}");
    assert!(e < TOL);
    // Many traces in one batch: more than 256 segments, so more than one wave.
    let (n, ns, d) = (80_000, 20, 3);
    let (x, labels) = gen_data(30, n, ns, 2);
    let mut acc = MomentAccumulator::new(ns, d).unwrap();
    acc.update(x.view(), labels.view()).unwrap();
    let e = max_err(
        &acc.t_values(0, 1),
        &reference_t(&x, &labels.to_vec(), 0, 1, d),
    );
    eprintln!("many traces: {e:.3e}");
    assert!(e < TOL);
}
