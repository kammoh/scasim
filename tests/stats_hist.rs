//! Tests of the histogram accumulator: merging, saved state, class slots, validation, and planted
//! differences that moment-based t-tests miss.
//!
//! * `merge_serde`: merging, serialization, determinism, and float binning.
//! * `planted`: the planted pairs of distributions. The in-test `welch_t` is a plain two-pass
//!   Welch t-test of orders 1..4. The later histogram-moment code compares with it.
//! * `class_slots`: class slots are sorted by label, so the state does not depend on the order
//!   of the batches (ruling A4).
//! * `validation`: `validate` and the accessors on inconsistent states (ruling A14).

mod merge_serde {
    //! Merging, serialization, determinism, and float binning.

    use ndarray::{Array1, Array2, s};
    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};
    use scasim::stats::{Binning, HistAccumulator, Statistic, StatsError, TestOptions, TestResult};

    const SAMPLES: usize = 70;
    const TRACES: usize = 1500;

    /// Random data with three classes and a class-dependent shift in a few samples.
    fn data(seed: u64) -> (Array2<u8>, Array1<u16>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let labels =
            Array1::from_iter((0..TRACES).map(|_| [3_u16, 7, 40000][rng.random_range(0..3)]));
        let traces = Array2::from_shape_fn((TRACES, SAMPLES), |(i, s)| {
            let base = (rng.next_u32() & 0x3F).count_ones() as u8;
            if s % 10 == 3 && labels[i] == 7 {
                base + 1
            } else {
                base
            }
        });
        (traces, labels)
    }

    fn same_results(a: &[TestResult], b: &[TestResult]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.statistic.to_bits(), y.statistic.to_bits());
            assert_eq!(x.neg_log10_p.to_bits(), y.neg_log10_p.to_bits());
            assert_eq!(
                (x.dof, x.n, x.columns, x.merged),
                (y.dof, y.n, y.columns, y.merged)
            );
        }
    }

    fn same_histograms(a: &HistAccumulator, b: &HistAccumulator) {
        assert_eq!(a.n_samples(), b.n_samples());
        for &label in &[3_u16, 7, 40000] {
            assert_eq!(a.class_count(label), b.class_count(label));
            for sample in 0..a.n_samples() {
                assert_eq!(a.histogram(sample, label), b.histogram(sample, label));
            }
        }
    }

    fn all_options() -> Vec<TestOptions> {
        let mut v = Vec::new();
        for statistic in [Statistic::Pearson, Statistic::G] {
            for min_expected in [0.0, 5.0, 20.0] {
                v.push(TestOptions {
                    statistic,
                    min_expected,
                });
            }
        }
        v
    }

    fn one_pass(traces: &Array2<u8>, labels: &Array1<u16>) -> HistAccumulator {
        let mut acc = HistAccumulator::new(SAMPLES, Binning::Exact);
        acc.update(traces.view(), labels.view()).unwrap();
        acc
    }

    #[test]
    fn merge_equals_one_pass_for_any_split_and_order() {
        let (t, l) = data(1);
        let reference = one_pass(&t, &l);
        // Split into uneven batches; merge in two different orders and as a tree.
        let cuts = [0, 130, 131, 700, 1100, TRACES];
        let parts: Vec<HistAccumulator> = cuts
            .windows(2)
            .map(|w| {
                one_pass(
                    &t.slice(s![w[0]..w[1], ..]).to_owned(),
                    &l.slice(s![w[0]..w[1]]).to_owned(),
                )
            })
            .collect();
        let mut forward = HistAccumulator::new(SAMPLES, Binning::Exact);
        for p in &parts {
            forward.merge(p).unwrap();
        }
        let mut backward = HistAccumulator::new(SAMPLES, Binning::Exact);
        for p in parts.iter().rev() {
            backward.merge(p).unwrap();
        }
        let mut left = parts[0].clone();
        left.merge(&parts[1]).unwrap();
        let mut right = parts[2].clone();
        right.merge(&parts[3]).unwrap();
        right.merge(&parts[4]).unwrap();
        left.merge(&right).unwrap();
        // Also a streaming accumulator that is updated batch by batch.
        let mut streamed = HistAccumulator::new(SAMPLES, Binning::Exact);
        for w in cuts.windows(2) {
            streamed
                .update(t.slice(s![w[0]..w[1], ..]), l.slice(s![w[0]..w[1]]))
                .unwrap();
        }
        for other in [&forward, &backward, &left, &streamed] {
            same_histograms(&reference, other);
            for opts in all_options() {
                same_results(
                    &reference.test_all(&opts).unwrap(),
                    &other.test_all(&opts).unwrap(),
                );
                same_results(
                    &reference.test_pair(3, 40000, &opts).unwrap(),
                    &other.test_pair(3, 40000, &opts).unwrap(),
                );
            }
        }
    }

    #[test]
    fn dense_and_sparse_layouts_give_identical_results() {
        let (t, l) = data(2);
        let dense = one_pass(&t, &l);
        let mut sparse = HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, 0);
        sparse.update(t.view(), l.view()).unwrap();
        assert_eq!(dense.n_sparse(), 0);
        assert_eq!(sparse.n_sparse(), SAMPLES);
        // A tiny dense window forces a mix: some samples convert while the data stream in.
        let mut tiny = HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, 3);
        tiny.update(t.view(), l.view()).unwrap();
        same_histograms(&dense, &sparse);
        same_histograms(&dense, &tiny);
        for opts in all_options() {
            let want = dense.test_all(&opts).unwrap();
            same_results(&want, &sparse.test_all(&opts).unwrap());
            same_results(&want, &tiny.test_all(&opts).unwrap());
        }
        // Merging a sparse accumulator into a dense one also gives the same counts.
        let mut mixed = HistAccumulator::new(SAMPLES, Binning::Exact);
        mixed.merge(&sparse).unwrap();
        same_histograms(&dense, &mixed);
    }

    #[test]
    fn serde_round_trip_is_exact() {
        let (t, l) = data(3);
        for max_dense in [4096, 0] {
            let mut acc = HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, max_dense);
            acc.update(t.view(), l.view()).unwrap();
            let before_bytes = acc.memory_bytes();
            acc.compact();
            assert!(acc.memory_bytes() <= before_bytes);
            let opts = TestOptions::default();
            let want = acc.test_all(&opts).unwrap();

            let json = serde_json::to_string(&acc).unwrap();
            let from_json: HistAccumulator = serde_json::from_str(&json).unwrap();
            from_json.validate().unwrap();
            same_histograms(&acc, &from_json);
            same_results(&want, &from_json.test_all(&opts).unwrap());

            let bytes = postcard::to_stdvec(&acc).unwrap();
            let from_bytes: HistAccumulator = postcard::from_bytes(&bytes).unwrap();
            from_bytes.validate().unwrap();
            same_histograms(&acc, &from_bytes);
            same_results(&want, &from_bytes.test_all(&opts).unwrap());
            // Serializing the loaded copy again gives the same bytes (deterministic format).
            assert_eq!(bytes, postcard::to_stdvec(&from_bytes).unwrap());
            eprintln!(
                "max_dense {max_dense}: {SAMPLES} samples x 3 classes x 7 bins: postcard {} bytes, json {} bytes, memory {} bytes",
                bytes.len(),
                json.len(),
                acc.memory_bytes()
            );
            // The loaded copy keeps accepting batches.
            let mut loaded = from_bytes;
            loaded.update(t.view(), l.view()).unwrap();
            assert_eq!(loaded.class_count(7), 2 * acc.class_count(7));
        }
    }

    #[test]
    fn validate_catches_corruption() {
        let (t, l) = data(4);
        let acc = one_pass(&t, &l);
        let mut v = serde_json::to_value(&acc).unwrap();
        v["class_counts"][0] = serde_json::json!(1);
        let bad: HistAccumulator = serde_json::from_value(v).unwrap();
        assert!(matches!(bad.validate(), Err(StatsError::InvalidState(_))));
    }

    #[test]
    fn counter_overflow_is_an_error_not_a_wraparound() {
        let (t, l) = data(5);
        let acc = one_pass(&t, &l);
        let mut v = serde_json::to_value(&acc).unwrap();
        v["class_counts"][0] = serde_json::json!(u64::from(u32::MAX) - 1);
        let mut near_full: HistAccumulator = serde_json::from_value(v).unwrap();
        let label = near_full.labels()[0];
        let before = near_full.class_count(label);
        let err = near_full.update(t.view(), l.view()).unwrap_err();
        assert_eq!(err, StatsError::CountOverflow { label });
        assert_eq!(near_full.class_count(label), before, "unchanged on error");
        assert_eq!(
            near_full.merge(&acc).unwrap_err(),
            StatsError::CountOverflow { label }
        );
    }

    #[test]
    fn results_do_not_depend_on_the_number_of_threads() {
        let (t, l) = data(6);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut acc = HistAccumulator::new(SAMPLES, Binning::Exact);
                for w in [0, 500, TRACES].windows(2) {
                    acc.update(t.slice(s![w[0]..w[1], ..]), l.slice(s![w[0]..w[1]]))
                        .unwrap();
                }
                acc.compact();
                let results = acc.test_all(&TestOptions::default()).unwrap();
                (postcard::to_stdvec(&acc).unwrap(), results)
            })
        };
        let (bytes1, r1) = run(1);
        for threads in [2, 7] {
            let (bytes, r) = run(threads);
            assert_eq!(bytes1, bytes);
            same_results(&r1, &r);
        }
    }

    #[test]
    fn float_traces_with_fixed_bins_match_integer_traces() {
        let (t, l) = data(7);
        let int = one_pass(&t, &l);
        // 0.25 * k is exactly representable, so the fixed rule with width 0.25 maps it to bin k.
        let tf = t.mapv(|v| f64::from(v) * 0.25 + 100.0);
        let binning = Binning::fixed(100.0, 0.25).unwrap();
        let mut fl = HistAccumulator::new(SAMPLES, binning);
        fl.update(tf.view(), l.view()).unwrap();
        // f32 input gives the same bins because the values are exact in f32.
        let tf32 = tf.mapv(|v| v as f32);
        let mut fl32 = HistAccumulator::new(SAMPLES, binning);
        fl32.update(tf32.view(), l.view()).unwrap();
        same_histograms(&int, &fl);
        same_histograms(&int, &fl32);
        let opts = TestOptions::default();
        same_results(&int.test_all(&opts).unwrap(), &fl.test_all(&opts).unwrap());
        // Values inside a bin are merged: shifting by less than a bin width changes nothing.
        let shifted = tf.mapv(|v| v + 0.1);
        let mut fl_shift = HistAccumulator::new(SAMPLES, binning);
        fl_shift.update(shifted.view(), l.view()).unwrap();
        same_histograms(&int, &fl_shift);
        // Under the exact rule, fractional values are rejected and counted.
        let mut exact = HistAccumulator::new(SAMPLES, Binning::Exact);
        let rejected = exact.update(shifted.view(), l.view()).unwrap();
        assert_eq!(rejected, (TRACES * SAMPLES) as u64);
    }

    #[test]
    fn negative_and_large_offsets_work() {
        let mut acc = HistAccumulator::new(2, Binning::Exact);
        let traces = Array2::from_shape_vec(
            (4, 2),
            vec![
                -5_i64, 1_000_000, -4, 1_000_001, -5, 1_000_000, -3, 1_000_003,
            ],
        )
        .unwrap();
        let labels = Array1::from(vec![0_u16, 0, 1, 1]);
        acc.update(traces.view(), labels.view()).unwrap();
        assert_eq!(acc.histogram(0, 0), vec![(-5, 1), (-4, 1)]);
        assert_eq!(acc.histogram(1, 1), vec![(1_000_000, 1), (1_000_003, 1)]);
        assert_eq!(
            acc.n_sparse(),
            0,
            "a narrow window far from zero stays dense"
        );
    }
}

mod planted {
    //! Planted distribution differences that moment-based t-tests miss.
    //!
    //! Two cases:
    //!
    //! * (a) The example from the task: class A takes the values {0, 2} with equal probability, and
    //!   class B is always 1. The means are equal (1), but the variances are not (1 versus 0). A
    //!   first-order t-test misses it. A second-order t-test (on the mean-free squared values) and the
    //!   chi-squared test both detect it. So this case does NOT have equal variances.
    //! * (b) A true equal-mean, equal-variance, equal-skewness pair: class A takes {2, 6} with equal
    //!   probability (mean 4, variance 4, kurtosis 1). Class B takes 4 with probability 3/4 and 0 or 8
    //!   with probability 1/8 each (mean 4, variance 4, kurtosis 4). Welch's t-tests of orders 1, 2,
    //!   and 3 all miss it. Only order 4 (and the chi-squared test) detect it.

    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scasim::stats::{Binning, HistAccumulator, Statistic, TestOptions, summarize};

    /// Welch's t statistic of order `d` for two sets of values, as in the TVLA methodology:
    /// each class is preprocessed with its own sample mean (and variance for d >= 3), then compared
    /// with a Welch t-test. This is a plain two-pass reference for the test module.
    fn welch_t(a: &[f64], b: &[f64], order: u32) -> f64 {
        fn prep(x: &[f64], order: u32) -> Vec<f64> {
            let n = x.len() as f64;
            let mean = x.iter().sum::<f64>() / n;
            match order {
                1 => x.to_vec(),
                2 => x.iter().map(|v| (v - mean).powi(2)).collect(),
                d => {
                    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                    let sd = var.sqrt();
                    x.iter().map(|v| ((v - mean) / sd).powi(d as i32)).collect()
                }
            }
        }
        fn mean_var(x: &[f64]) -> (f64, f64) {
            let n = x.len() as f64;
            let mean = x.iter().sum::<f64>() / n;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, var)
        }
        let (pa, pb) = (prep(a, order), prep(b, order));
        let ((ma, va), (mb, vb)) = (mean_var(&pa), mean_var(&pb));
        (ma - mb) / (va / pa.len() as f64 + vb / pb.len() as f64).sqrt()
    }

    fn chi2_neg_log10_p(a: &[f64], b: &[f64], statistic: Statistic) -> f64 {
        let n = a.len() + b.len();
        let values: Vec<u8> = a.iter().chain(b).map(|&v| v as u8).collect();
        let labels: Vec<u16> = (0..n).map(|i| u16::from(i >= a.len())).collect();
        let mut acc = HistAccumulator::new(1, Binning::Exact);
        acc.update(
            Array2::from_shape_vec((n, 1), values).unwrap().view(),
            Array1::from(labels).view(),
        )
        .unwrap();
        let opts = TestOptions {
            statistic,
            ..TestOptions::default()
        };
        acc.test_pair(0, 1, &opts).unwrap()[0].neg_log10_p
    }

    fn draw(rng: &mut StdRng, n: usize, choices: &[(f64, f64)]) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let u: f64 = rng.random();
                let mut acc = 0.0;
                for &(p, v) in choices {
                    acc += p;
                    if u < acc {
                        return v;
                    }
                }
                choices.last().unwrap().1
            })
            .collect()
    }

    #[test]
    fn case_a_equal_means_unequal_variances() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 2000;
        let a = draw(&mut rng, n, &[(0.5, 0.0), (0.5, 2.0)]);
        let b = vec![1.0; n];
        let t1 = welch_t(&a, &b, 1);
        // Class B has zero variance, so the second-order statistic can be infinite or NaN. Both
        // count as detected: the preprocessed class B is exactly constant.
        let t2 = welch_t(&a, &b, 2);
        let nlp = chi2_neg_log10_p(&a, &b, Statistic::Pearson);
        eprintln!("case (a), n = {n}: t1 = {t1:.2}, t2 = {t2:.3e}, chi2 -log10 p = {nlp:.1}");
        assert!(t1.abs() < 4.5, "first order must miss: {t1}");
        assert!(
            t2.is_nan() || t2.abs() >= 4.5,
            "second order must detect: {t2}"
        );
        assert!(nlp > 100.0);
    }

    #[test]
    fn case_b_equal_first_three_moments() {
        let mut rng = StdRng::seed_from_u64(2);
        let n = 5000;
        let a = draw(&mut rng, n, &[(0.5, 2.0), (0.5, 6.0)]);
        let b = draw(&mut rng, n, &[(0.125, 0.0), (0.75, 4.0), (0.125, 8.0)]);
        let t: Vec<f64> = (1..=4).map(|d| welch_t(&a, &b, d)).collect();
        let pearson = chi2_neg_log10_p(&a, &b, Statistic::Pearson);
        let g = chi2_neg_log10_p(&a, &b, Statistic::G);
        eprintln!(
            "case (b), n = {n}: t1 = {:.2}, t2 = {:.2}, t3 = {:.2}, t4 = {:.1}; chi2 -log10 p = {pearson:.1}, G = {g:.1}",
            t[0], t[1], t[2], t[3]
        );
        for (d, t) in t.iter().take(3).enumerate() {
            assert!(t.abs() < 4.5, "order {} must miss: t = {t}", d + 1);
        }
        assert!(t[3].abs() > 4.5, "order 4 must detect: t = {}", t[3]);
        assert!(pearson > 100.0 && g > 100.0);
    }

    /// How many traces per class does each test need to cross its threshold (case b)?
    #[test]
    fn case_b_chi2_detects_with_few_traces() {
        let mut rng = StdRng::seed_from_u64(3);
        let a = draw(&mut rng, 400, &[(0.5, 2.0), (0.5, 6.0)]);
        let b = draw(&mut rng, 400, &[(0.125, 0.0), (0.75, 4.0), (0.125, 8.0)]);
        let nlp = chi2_neg_log10_p(&a, &b, Statistic::Pearson);
        eprintln!("case (b), n = 400: chi2 -log10 p = {nlp:.1}");
        assert!(nlp > 5.0);
    }

    /// The leaking sample is found among many null samples, and the others stay below threshold.
    #[test]
    fn planted_sample_is_ranked_first() {
        let mut rng = StdRng::seed_from_u64(4);
        let (n, samples, leak) = (3000, 50, 17);
        let mut traces = Array2::<u8>::zeros((n, samples));
        let mut labels = Array1::<u16>::zeros(n);
        for i in 0..n {
            let class = u16::from(i % 2 == 1);
            labels[i] = class;
            for s in 0..samples {
                // Null: Binomial(8, 1/2), the same for both classes.
                let mut v: u8 = (0..8).map(|_| u8::from(rng.random::<bool>())).sum();
                if s == leak {
                    // Planted leak: same mean and variance in both classes, different kurtosis.
                    let u: f64 = rng.random();
                    v = if class == 0 {
                        if u < 0.5 { 2 } else { 6 }
                    } else if u < 0.125 {
                        0
                    } else if u < 0.875 {
                        4
                    } else {
                        8
                    };
                }
                traces[[i, s]] = v;
            }
        }
        let mut acc = HistAccumulator::new(samples, Binning::Exact);
        acc.update(traces.view(), labels.view()).unwrap();
        let results = acc.test_pair(0, 1, &TestOptions::default()).unwrap();
        let summary = summarize(&results, 5.0);
        eprintln!(
            "planted at sample {leak}: argmax = {}, max -log10 p = {:.1}, samples above 5: {}",
            summary.argmax, summary.max_neg_log10_p, summary.above
        );
        assert_eq!(summary.argmax, leak);
        assert_eq!(summary.above, 1);
    }
}

mod class_slots {
    //! Class slots are sorted by label (ruling A4). The state does not depend on batch order.

    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scasim::stats::{Binning, HistAccumulator, StatsError, TestOptions};

    const SAMPLES: usize = 12;

    /// A batch of `n` traces that all have class `label`. The value range depends on the sample
    /// and the label, so dense windows grow in different directions in different orders.
    fn batch(seed: u64, label: u16, n: usize, spread: u8) -> (Array2<u8>, Array1<u16>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let traces = Array2::from_shape_fn((n, SAMPLES), |(_, s)| {
            let centre = (s * 7 + usize::from(label) % 50) as u8;
            centre + rng.random_range(0..spread)
        });
        (traces, Array1::from_elem(n, label))
    }

    fn accumulate(parts: &[(Array2<u8>, Array1<u16>)], max_dense: usize) -> HistAccumulator {
        let mut acc = HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, max_dense);
        for (t, l) in parts {
            acc.update(t.view(), l.view()).unwrap();
        }
        acc
    }

    fn bytes(acc: &HistAccumulator) -> Vec<u8> {
        let mut acc = acc.clone();
        acc.compact();
        postcard::to_stdvec(&acc).unwrap()
    }

    fn parts() -> Vec<(Array2<u8>, Array1<u16>)> {
        // Labels sort opposite to the arrival order in the first run: 9, then 40000, then 5, 0.
        vec![
            batch(1, 9, 40, 6),
            batch(2, 40000, 25, 9),
            batch(3, 5, 31, 4),
            batch(4, 0, 17, 70),
            batch(5, 9, 22, 6),
        ]
    }

    #[test]
    fn labels_are_in_increasing_order() {
        let acc = accumulate(&parts(), 4096);
        assert_eq!(acc.labels(), &[0, 5, 9, 40000]);
        assert_eq!(acc.class_count(9), 62);
        assert_eq!(acc.class_count(40000), 25);
        acc.validate().unwrap();
    }

    #[test]
    fn batch_order_does_not_change_the_serialized_bytes() {
        let p = parts();
        // Dense windows, all-sparse, and a tiny window (some samples dense, some sparse).
        for max_dense in [4096, 0, 12] {
            let forward = accumulate(&p, max_dense);
            let backward: Vec<_> = p.iter().rev().cloned().collect();
            let backward = accumulate(&backward, max_dense);
            let shuffled: Vec<_> = [3, 0, 4, 2, 1].iter().map(|&i| p[i].clone()).collect();
            let shuffled = accumulate(&shuffled, max_dense);
            assert_eq!(forward.labels(), backward.labels());
            assert_eq!(bytes(&forward), bytes(&backward), "max_dense {max_dense}");
            assert_eq!(bytes(&forward), bytes(&shuffled), "max_dense {max_dense}");
            // One accumulator per batch, merged in different orders.
            let single = |i: usize| accumulate(std::slice::from_ref(&p[i]), max_dense);
            let mut merged_a =
                HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, max_dense);
            for i in [0, 1, 2, 3, 4] {
                merged_a.merge(&single(i)).unwrap();
            }
            let mut merged_b =
                HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, max_dense);
            for i in [4, 3, 2, 1, 0] {
                merged_b.merge(&single(i)).unwrap();
            }
            assert_eq!(bytes(&forward), bytes(&merged_a), "max_dense {max_dense}");
            assert_eq!(bytes(&forward), bytes(&merged_b), "max_dense {max_dense}");
            // The test results are bit-identical too.
            let opts = TestOptions::default();
            assert_eq!(
                forward
                    .test_all(&opts)
                    .unwrap()
                    .iter()
                    .map(|r| r.statistic.to_bits())
                    .collect::<Vec<_>>(),
                backward
                    .test_all(&opts)
                    .unwrap()
                    .iter()
                    .map(|r| r.statistic.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn a_label_in_the_middle_keeps_the_existing_counts() {
        // Class 9 first, then class 5 (inserted before it), then class 7 (inserted between).
        for max_dense in [4096, 0] {
            let (t9, l9) = batch(11, 9, 30, 8);
            let (t5, l5) = batch(12, 5, 20, 8);
            let (t7, l7) = batch(13, 7, 10, 8);
            let mut acc = HistAccumulator::with_max_dense_bins(SAMPLES, Binning::Exact, max_dense);
            let hist_9_before = {
                acc.update(t9.view(), l9.view()).unwrap();
                (0..SAMPLES)
                    .map(|s| acc.histogram(s, 9))
                    .collect::<Vec<_>>()
            };
            acc.update(t5.view(), l5.view()).unwrap();
            acc.update(t7.view(), l7.view()).unwrap();
            assert_eq!(acc.labels(), &[5, 7, 9]);
            for (s, before) in hist_9_before.iter().enumerate() {
                assert_eq!(&acc.histogram(s, 9), before, "sample {s}");
                assert_eq!(
                    acc.histogram(s, 5)
                        .iter()
                        .map(|x| u64::from(x.1))
                        .sum::<u64>(),
                    20
                );
                assert_eq!(
                    acc.histogram(s, 7)
                        .iter()
                        .map(|x| u64::from(x.1))
                        .sum::<u64>(),
                    10
                );
            }
            acc.validate().unwrap();
        }
    }

    #[test]
    fn duplicate_labels_in_a_test_are_an_error() {
        let acc = accumulate(&parts(), 4096);
        let opts = TestOptions::default();
        assert_eq!(
            acc.test_classes(&[5, 5], &opts).unwrap_err(),
            StatsError::DuplicateLabel(5)
        );
        assert_eq!(
            acc.test_pair(5, 77, &opts).unwrap_err(),
            StatsError::UnknownLabel(77)
        );
    }
}

mod validation {
    //! `validate` rejects inconsistent states, and no accessor panics on them (ruling A14).

    use ndarray::{Array1, Array2};
    use scasim::stats::{Binning, HistAccumulator, StatsError, TestOptions};
    use serde_json::{Value, json};

    /// Three classes, four samples; values 0..8 so that the histograms are dense. Sample 3
    /// has a wide range and is sparse when `max_dense` is 8.
    fn accumulator(max_dense: usize) -> HistAccumulator {
        let n = 30;
        let traces = Array2::from_shape_fn((n, 4), |(i, s)| {
            if s == 3 {
                (i * 40) as u16
            } else {
                ((i * (s + 1)) % 8) as u16
            }
        });
        let labels = Array1::from_iter((0..n).map(|i| [2_u16, 5, 9][i % 3]));
        let mut acc = HistAccumulator::with_max_dense_bins(4, Binning::Exact, max_dense);
        acc.update(traces.view(), labels.view()).unwrap();
        acc.validate().unwrap();
        acc
    }

    fn corrupt(acc: &HistAccumulator, edit: impl FnOnce(&mut Value)) -> HistAccumulator {
        let mut v = serde_json::to_value(acc).unwrap();
        edit(&mut v);
        serde_json::from_value(v).unwrap()
    }

    fn assert_invalid(acc: &HistAccumulator, what: &str) {
        match acc.validate() {
            Err(StatsError::InvalidState(msg)) => eprintln!("{what}: {msg}"),
            other => panic!("{what}: expected InvalidState, got {other:?}"),
        }
    }

    #[test]
    fn dense_slot_count_must_equal_the_label_count() {
        let acc = accumulator(8);
        let bad = corrupt(&acc, |v| v["hists"][0]["Dense"]["slots"] = json!(2));
        assert_invalid(&bad, "fewer slots than labels");
        let bad = corrupt(&acc, |v| v["hists"][0]["Dense"]["slots"] = json!(4));
        assert_invalid(&bad, "more slots than labels");
    }

    #[test]
    fn a_state_with_too_few_slots_gives_an_error_not_a_panic() {
        let acc = accumulator(8);
        let bad = corrupt(&acc, |v| {
            // Drop a class row from sample 0: slots 2, counts for two rows.
            let d = &mut v["hists"][0]["Dense"];
            let width = d["width"].as_u64().unwrap() as usize;
            d["slots"] = json!(2);
            let counts: Vec<Value> = d["counts"].as_array().unwrap()[..2 * width].to_vec();
            d["counts"] = Value::Array(counts);
        });
        assert_invalid(&bad, "too few slots");
        let opts = TestOptions::default();
        assert!(matches!(
            bad.test_all(&opts),
            Err(StatsError::InvalidState(_))
        ));
        assert!(matches!(
            bad.test_pair(2, 9, &opts),
            Err(StatsError::InvalidState(_))
        ));
    }

    #[test]
    fn the_number_of_histograms_must_equal_the_number_of_samples() {
        for max_dense in [8, 0] {
            let acc = accumulator(max_dense);
            let bad = corrupt(&acc, |v| {
                v["hists"].as_array_mut().unwrap().pop();
            });
            assert_invalid(&bad, "missing histogram");
            let bad = corrupt(&acc, |v| {
                let extra = v["hists"][0].clone();
                v["hists"].as_array_mut().unwrap().push(extra);
            });
            assert_invalid(&bad, "extra histogram");
        }
    }

    #[test]
    fn labels_must_be_increasing() {
        let acc = accumulator(8);
        let bad = corrupt(&acc, |v| v["labels"] = json!([5, 2, 9]));
        assert_invalid(&bad, "unsorted labels");
        let bad = corrupt(&acc, |v| v["labels"] = json!([2, 5, 5]));
        assert_invalid(&bad, "duplicate labels");
        let bad = corrupt(&acc, |v| v["labels"] = json!([2, 5]));
        assert_invalid(&bad, "labels and counts differ in length");
    }

    #[test]
    fn totals_must_be_consistent() {
        for max_dense in [8, 0] {
            let acc = accumulator(max_dense);
            // A class total that is lower than the counts in the histograms.
            let bad = corrupt(&acc, |v| v["class_counts"][1] = json!(3));
            assert_invalid(&bad, "class count too low");
            // A class total that is higher: the missing values are not explained by `rejected`.
            let bad = corrupt(&acc, |v| v["class_counts"][1] = json!(11));
            assert_invalid(&bad, "class count too high");
            // A wrong rejected counter.
            let bad = corrupt(&acc, |v| v["rejected"] = json!(1));
            assert_invalid(&bad, "rejected counter");
        }
    }

    #[test]
    fn rejected_values_keep_the_totals_consistent() {
        let mut acc = HistAccumulator::new(2, Binning::Exact);
        let traces =
            Array2::from_shape_vec((3, 2), vec![1.0, f64::NAN, 2.0, 2.5, 3.0, 4.0]).unwrap();
        let labels = Array1::from(vec![0_u16, 0, 1]);
        assert_eq!(acc.update(traces.view(), labels.view()).unwrap(), 2);
        // The traces with a rejected value still count in their class total.
        assert_eq!(acc.class_count(0), 2);
        acc.validate().unwrap();
    }

    #[test]
    fn sparse_entries_are_checked() {
        let acc = accumulator(8);
        assert!(acc.is_sparse(3));
        // An entry for a class slot that does not exist.
        // (The last entry has the largest bin, so the order of the entries stays valid.)
        let bad = corrupt(&acc, |v| {
            let last = v["hists"][3]["Sparse"].as_array().unwrap().len() - 1;
            v["hists"][3]["Sparse"][last][1] = json!(3);
        });
        assert_invalid(&bad, "unknown class slot");
    }

    #[test]
    fn unordered_or_repeated_sparse_entries_do_not_deserialize() {
        let acc = accumulator(8);
        let mut v = serde_json::to_value(&acc).unwrap();
        let first = v["hists"][3]["Sparse"][0].clone();
        v["hists"][3]["Sparse"].as_array_mut().unwrap().push(first);
        assert!(serde_json::from_value::<HistAccumulator>(v).is_err());
        let mut v = serde_json::to_value(&acc).unwrap();
        v["hists"][3]["Sparse"].as_array_mut().unwrap().swap(0, 1);
        assert!(serde_json::from_value::<HistAccumulator>(v).is_err());
        let mut v = serde_json::to_value(&acc).unwrap();
        v["hists"][3]["Sparse"][0][2] = json!(0);
        assert!(serde_json::from_value::<HistAccumulator>(v).is_err());
    }

    #[test]
    fn dense_windows_are_checked() {
        let acc = accumulator(8);
        let bad = corrupt(&acc, |v| v["hists"][0]["Dense"]["width"] = json!(9));
        assert_invalid(&bad, "counts do not match the window");
        let bad = corrupt(&acc, |v| v["hists"][0]["Dense"]["width"] = json!(u64::MAX));
        assert_invalid(&bad, "huge width");
        let bad = corrupt(&acc, |v| v["hists"][0]["Dense"]["base"] = json!(i64::MAX));
        assert_invalid(&bad, "window past the largest bin");
    }

    #[test]
    fn binning_is_checked() {
        let acc = accumulator(8);
        let bad = corrupt(&acc, |v| {
            v["binning"] = json!({"Fixed": {"origin": 0.0, "width": 0.0}});
        });
        assert_invalid(&bad, "zero bin width");
    }

    #[test]
    fn out_of_range_sample_indices_do_not_panic() {
        let acc = accumulator(8);
        assert!(acc.histogram(99, 2).is_empty());
        assert!(!acc.is_sparse(99));
    }

    #[test]
    fn empty_batches_are_fine() {
        let mut acc = accumulator(8);
        let before = serde_json::to_value(&acc).unwrap();
        let traces = Array2::<u8>::zeros((0, 4));
        let labels = Array1::<u16>::zeros(0);
        assert_eq!(acc.update(traces.view(), labels.view()).unwrap(), 0);
        assert_eq!(serde_json::to_value(&acc).unwrap(), before);
    }
}
