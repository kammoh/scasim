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
fn tiny_fourth_order_case_and_power_of_two_scales_keep_all_t_values() {
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
    for exponent in [-500, -133, 0, 133, 500] {
        let actual = evaluate(exponent);
        for order in 0..4 {
            let expected = base[[order, 0]];
            let got = actual[[order, 0]];
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "scale exponent {exponent}, order {}: {got} vs {expected}",
                order + 1
            );
        }
    }
    let tiny = evaluate(-133)[[3, 0]];
    assert!(
        (tiny - 0.557812321).abs() < 1e-8,
        "order four t-value was {tiny}"
    );
}
