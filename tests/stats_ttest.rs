#[path = "common/ttest_data.rs"]
mod common;

#[path = "common/ttest_reference.rs"]
mod reference;
#[path = "common/ttest_stability.rs"]
mod stability;

#[test]
fn fallible_api_rejects_bad_shapes_and_moments() {
    use ndarray::{Array1, Array2};
    use scasim::stats::{
        error::StatsError,
        ttest::{ClassMoments, MomentAccumulator, Moments},
    };

    assert!(matches!(
        MomentAccumulator::new(4, 0),
        Err(StatsError::ZeroOrder)
    ));
    let mut acc = MomentAccumulator::new(4, 2).unwrap();
    assert!(matches!(
        acc.update(
            Array2::<f64>::zeros((2, 3)).view(),
            Array1::from_vec(vec![0, 1]).view()
        ),
        Err(StatsError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        acc.update(
            Array2::<f64>::zeros((2, 4)).view(),
            Array1::from_vec(vec![0]).view()
        ),
        Err(StatsError::LabelCountMismatch { .. })
    ));
    assert!(matches!(
        acc.merge(&MomentAccumulator::new(5, 2).unwrap()),
        Err(StatsError::IncompatibleAccumulators)
    ));
    assert_eq!(acc.central_sum(0, 1), None);
    let bad = Moments {
        ns: 1,
        d: 1,
        classes: vec![ClassMoments {
            label: 0,
            count: 1,
            data: vec![f64::NAN; 3],
        }],
    };
    assert!(matches!(
        MomentAccumulator::from_moments(bad),
        Err(StatsError::NonFiniteMoments { label: 0 })
    ));
    let empty_with_data = Moments {
        ns: 1,
        d: 1,
        classes: vec![ClassMoments {
            label: 0,
            count: 0,
            data: vec![0.0, 1.0, 0.0],
        }],
    };
    assert!(matches!(
        MomentAccumulator::from_moments(empty_with_data),
        Err(StatsError::EmptyClassHasData { label: 0 })
    ));
}

#[test]
fn class_moment_accessors_reject_bad_indexes() {
    use scasim::stats::ttest::ClassMoments;

    let c = ClassMoments {
        label: 0,
        count: 0,
        data: vec![],
    };
    assert_eq!(c.mean(1, 0), None);
    assert_eq!(c.mean(1, 1), None);
    assert_eq!(c.mean(usize::MAX, usize::MAX), None);
    assert_eq!(c.mean(usize::MAX, usize::MAX - 1), None);
    assert_eq!(c.central_sum(1, 1, 0), None);
    assert_eq!(c.central_sum(1, 2, 1), None);
    assert_eq!(c.central_sum(usize::MAX, usize::MAX, usize::MAX), None);
    assert_eq!(c.central_sum(usize::MAX, usize::MAX, 0), None);
}

#[test]
fn restored_moments_validate_block_size_and_even_moment_sign() {
    use scasim::stats::{
        error::StatsError,
        ttest::{ClassMoments, MomentAccumulator, Moments},
    };

    let too_wide = Moments {
        ns: 0,
        d: usize::MAX / 2,
        classes: vec![ClassMoments {
            label: 0,
            count: 0,
            data: vec![],
        }],
    };
    assert!(matches!(
        MomentAccumulator::from_moments(too_wide),
        Err(StatsError::InvalidMomentOrder { .. }) | Err(StatsError::CountOverflow)
    ));

    let negative_even = Moments {
        ns: 1,
        d: 1,
        classes: vec![ClassMoments {
            label: 0,
            count: 2,
            data: vec![0.0, 0.0, -1.0],
        }],
    };
    assert!(MomentAccumulator::from_moments(negative_even).is_err());
}

#[test]
fn maximum_t_test_order_is_documented_and_enforced() {
    use scasim::stats::{
        error::StatsError,
        ttest::{MAX_ORDER, MomentAccumulator},
    };
    assert!(MomentAccumulator::new(0, MAX_ORDER).is_ok());
    assert!(matches!(
        MomentAccumulator::new(0, MAX_ORDER + 1),
        Err(StatsError::InvalidMomentOrder { .. })
    ));
}

#[test]
fn t_values_are_scale_invariant_within_the_supported_range() {
    use ndarray::{Array1, Array2};
    use scasim::stats::ttest::MomentAccumulator;

    let labels = Array1::from_iter((0..32).map(|i| if i < 16 { 0u16 } else { 1u16 }));
    let evaluate = |exponent: i32| {
        let scale = 2.0f64.powi(exponent);
        let traces = Array2::from_shape_fn((32, 1), |(i, _)| {
            let is_one = if i < 16 { i == 15 } else { i >= 30 };
            if is_one { scale } else { 0.0 }
        });
        let mut acc = MomentAccumulator::new(1, 4).unwrap();
        acc.update(traces.view(), labels.view()).unwrap();
        acc.t_values(0, 1)
    };

    let base = evaluate(0);
    for exponent in [-120, -60, 0, 60, 120] {
        let actual = evaluate(exponent);
        for order in 0..4 {
            let expected = base[[order, 0]];
            let got = actual[[order, 0]];
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(f64::MIN_POSITIVE),
                "scale exponent {exponent}, order {}: {got} vs {expected}",
                order + 1
            );
        }
    }
}

#[test]
fn extreme_scale_inputs_fail_closed_outside_the_supported_range() {
    use ndarray::{Array1, Array2};
    use scasim::stats::ttest::MomentAccumulator;

    let labels = Array1::from_iter((0..32).map(|i| if i < 16 { 0u16 } else { 1u16 }));
    for exponent in [-500, 500] {
        let scale = 2.0f64.powi(exponent);
        let traces = Array2::from_shape_fn((32, 1), |(i, _)| {
            let is_one = if i < 16 { i == 15 } else { i >= 30 };
            if is_one { scale } else { 0.0 }
        });
        let mut acc = MomentAccumulator::new(1, 4).unwrap();
        acc.update(traces.view(), labels.view()).unwrap();
        let values = acc.t_values(0, 1);
        let first_invalid = 1;
        for order in first_invalid..4 {
            assert!(
                values[[order, 0]].is_nan(),
                "scale {exponent}, order {}: {values:?}",
                order + 1
            );
        }
    }
}

#[test]
fn mixed_scale_merge_preserves_moments_and_matches_one_pass() {
    use ndarray::{Array1, Array2};
    use scasim::stats::ttest::MomentAccumulator;

    let tiny = 2.0f64.powi(-100);
    let large = 2.0f64.powi(100);
    let traces = Array2::from_shape_vec(
        (16, 1),
        vec![
            0., 0., 0., tiny, 0., 0., tiny, 0., 0., 0., large, 0., large, 0., 0., large,
        ],
    )
    .unwrap();
    let labels = Array1::from_iter((0..16).map(|i| if i < 8 { 0u16 } else { 1u16 }));

    let mut all = MomentAccumulator::new(1, 4).unwrap();
    all.update(traces.view(), labels.view()).unwrap();
    let mut left = MomentAccumulator::new(1, 4).unwrap();
    let left_traces =
        Array2::from_shape_vec((8, 1), vec![0., 0., 0., tiny, 0., 0., 0., 0.]).unwrap();
    let left_labels = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);
    left.update(left_traces.view(), left_labels.view()).unwrap();
    let mut right = MomentAccumulator::new(1, 4).unwrap();
    let right_traces =
        Array2::from_shape_vec((8, 1), vec![0., tiny, 0., 0., large, 0., large, large]).unwrap();
    let right_labels = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);
    right
        .update(right_traces.view(), right_labels.view())
        .unwrap();
    left.merge(&right).unwrap();

    for order in 1..=4 {
        let expected = all.t_values(0, 1)[[order - 1, 0]];
        let actual = left.t_values(0, 1)[[order - 1, 0]];
        assert!((actual - expected).abs() <= 1e-11 * expected.abs().max(f64::MIN_POSITIVE));
    }
    assert!(left.central_sum(0, 2).unwrap()[0] > 0.0);
    assert!(left.central_sum(1, 2).unwrap()[0] > 0.0);
    for label in [0, 1] {
        for p in 2..=8 {
            assert_ne!(
                left.central_sum(label, p).unwrap()[0],
                0.0,
                "label {label}, order {p}"
            );
        }
    }
}

#[test]
fn tiny_counterexample_keeps_in_range_orders_and_rejects_lost_moments() {
    use ndarray::{Array1, Array2};
    use scasim::stats::ttest::MomentAccumulator;

    let labels = Array1::from_iter((0..32).map(|i| if i < 16 { 0u16 } else { 1u16 }));
    let evaluate = |exponent: i32| {
        let scale = 2.0f64.powi(exponent);
        let traces = Array2::from_shape_fn((32, 1), |(i, _)| {
            let is_one = if i < 16 { i == 15 } else { i >= 30 };
            if is_one { scale } else { 0.0 }
        });
        let mut acc = MomentAccumulator::new(1, 4).unwrap();
        acc.update(traces.view(), labels.view()).unwrap();
        acc.t_values(0, 1)
    };

    let base = evaluate(0);
    let tiny = evaluate(-133);
    for order in 0..3 {
        assert!((tiny[[order, 0]] - base[[order, 0]]).abs() <= 1e-12);
    }
    assert!(tiny[[3, 0]].is_nan());
}

#[test]
fn save_restore_is_exact_at_supported_extreme_scales() {
    use ndarray::{Array1, Array2};
    use scasim::stats::ttest::MomentAccumulator;

    for exponent in [-100, 100] {
        let scale = 2.0f64.powi(exponent);
        let traces = Array2::from_shape_fn((8, 1), |(i, _)| (i as f64 % 4.0) * scale);
        let labels = Array1::from_iter((0..8).map(|i| if i < 4 { 0u16 } else { 1u16 }));
        let mut acc = MomentAccumulator::new(1, 4).unwrap();
        acc.update(traces.view(), labels.view()).unwrap();
        let saved = acc.moments();
        let restored = MomentAccumulator::from_moments(saved.clone()).unwrap();
        assert_eq!(restored.moments(), saved, "scale {exponent}");
    }
}
