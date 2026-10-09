use ndarray::{Array1, Array2, s};
use scasim::stats::ttest::MomentAccumulator;
use scasim::stats::{Binning, HistAccumulator};
use serde_json::Value;

#[derive(serde::Deserialize)]
struct Fixture {
    datasets: Vec<Dataset>,
    cases: Vec<Case>,
}
#[derive(serde::Deserialize)]
struct Dataset {
    name: String,
    traces: Vec<Vec<u32>>,
    labels: Vec<u16>,
}
#[derive(serde::Deserialize)]
struct Case {
    name: String,
    dataset: String,
    d: usize,
    ns: usize,
    batch_sizes: Vec<usize>,
    t_values: Vec<Vec<Value>>,
    non_finite: Vec<NonFinite>,
}
#[derive(serde::Deserialize)]
struct NonFinite {
    order: usize,
    sample: usize,
    kind: String,
}

fn check(name: &str, got: &Array2<f64>, case: &Case, path: &str, worst: &mut f64) {
    for ((order_sample, &value), fixture) in got
        .indexed_iter()
        .zip(case.t_values.iter().flat_map(|r| r.iter()))
    {
        let (order, sample) = order_sample;
        if let Some(nf) = case
            .non_finite
            .iter()
            .find(|v| v.order == order + 1 && v.sample == sample)
        {
            let same = match nf.kind.as_str() {
                "nan" => value.is_nan(),
                "inf" => value == f64::INFINITY,
                "-inf" => value == f64::NEG_INFINITY,
                _ => false,
            };
            assert!(
                same,
                "{path} {name} order {} sample {}: expected {}, got {value}",
                order + 1,
                sample,
                nf.kind
            );
            continue;
        }
        if let Some(expected) = fixture.as_f64() {
            if value.is_finite() {
                let err = (value - expected).abs() / 1.0f64.max(value.abs()).max(expected.abs());
                *worst = worst.max(err);
                assert!(
                    err <= 1e-9,
                    "{path} {name} order {} sample {}: {value} versus {expected}, error {err:e}",
                    order + 1,
                    sample
                );
            } else {
                // A2 returns NaN when a class has fewer than two traces, even if SCALib
                // reports infinity for a zero denominator.
                assert!(
                    value.is_nan(),
                    "{path} {name} order {} sample {}: {value}",
                    order + 1,
                    sample
                );
            }
        } else {
            assert!(
                value.is_nan(),
                "{path} {name} order {} sample {}: {value}",
                order + 1,
                sample
            );
        }
    }
}

#[test]
fn both_t_test_paths_match_every_scalib_fixture_case() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/stats/scalib_ttest.json")).unwrap();
    let mut worst_hist: f64 = 0.0;
    let mut worst_mom: f64 = 0.0;
    for case in &fixture.cases {
        let data = fixture
            .datasets
            .iter()
            .find(|d| d.name == case.dataset)
            .unwrap();
        assert_eq!(data.traces.len(), data.labels.len());
        assert_eq!(data.traces[0].len(), case.ns);
        let x = Array2::from_shape_vec(
            (data.traces.len(), case.ns),
            data.traces.iter().flatten().copied().collect(),
        )
        .unwrap();
        let labels = Array1::from_vec(data.labels.clone());
        let mut hist = HistAccumulator::new(case.ns, Binning::Exact);
        let mut moments = MomentAccumulator::new(case.ns, case.d).unwrap();
        let mut start = 0;
        for &size in &case.batch_sizes {
            let end = start + size;
            hist.update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
                .unwrap();
            moments
                .update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
                .unwrap();
            start = end;
        }
        assert_eq!(start, x.nrows(), "{}", case.name);
        check(
            &case.name,
            &hist.t_values(0, 1, case.d).unwrap(),
            case,
            "histogram",
            &mut worst_hist,
        );
        check(
            &case.name,
            &moments.t_values(0, 1),
            case,
            "moments",
            &mut worst_mom,
        );
    }
    eprintln!(
        "SCALib fixtures: histogram worst relative difference {worst_hist:.3e}; moments {worst_mom:.3e}"
    );
    assert!(worst_hist <= 1e-9);
    assert!(worst_mom <= 1e-9);
}
