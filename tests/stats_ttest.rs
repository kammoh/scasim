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
