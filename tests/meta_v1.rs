mod common;
use common::*;
use scasim::batch::{EdgeSampling, LengthPolicy, Sampling, compute_batch, read_batch_meta};
use scasim::hierarchy::Selection;
use scasim::power::{PowerPlan, edges::EdgeKind};

fn fixture(exponent: i8) -> (tempfile::TempDir, serde_json::Value) {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::flat(&[1]);
    fx.timescale_exponent = exponent;
    fx.steps = vec![
        (1, vec![(0, "1".into())]),
        (2, vec![(0, "0".into())]),
        (3, vec![(0, "1".into())]),
        (4, vec![(0, "0".into())]),
    ];
    write_fst(&dir.path().join("w.fst"), &fx);
    let json = serde_json::json!({"scasim_meta":1,"waveform":"w.fst","time":{"mantissa":1,"exponent":-12},"batch":{"id":"b0","seeds":{"base":17},"status":"committed","design_random":{"requested":"off","applied":"off","how":"hook"}},"segments":[{"id":7,"start":1000,"end":3000,"label":0},{"id":8,"start":3000,"end":4000,"label":1}],"labels":{"0":"fixed","1":"random"},"groups":{"0":"default"},"extensions":{"7":{"note":"kept"}},"future":true});
    (dir, json)
}
fn read(
    dir: &tempfile::TempDir,
    json: &serde_json::Value,
) -> miette::Result<scasim::batch::BatchMeta> {
    let path = dir.path().join("meta.json");
    std::fs::write(&path, serde_json::to_vec(json).unwrap()).unwrap();
    read_batch_meta(&path)
}
#[test]
fn v1_exact_ps_to_ns_and_ps_ticks() {
    for (exp, want) in [
        (-9, vec![(1, 3, 0), (3, 4, 1)]),
        (-12, vec![(1000, 3000, 0), (3000, 4000, 1)]),
    ] {
        let (dir, json) = fixture(exp);
        assert_eq!(read(&dir, &json).unwrap().markers, want);
    }
}
#[test]
fn v1_rejects_inexact_and_overflow_with_file_and_segment() {
    for (exp, start, unit) in [(-9, 1001, -12), (-12, 1, 0)] {
        let (dir, mut json) = fixture(exp);
        json["time"]["exponent"] = unit.into();
        json["segments"][0]["start"] = start.into();
        if unit == 0 {
            json["segments"][0]["end"] = u64::MAX.into();
        }
        let err = read(&dir, &json).unwrap_err().to_string();
        assert!(
            err.contains("meta.json") && err.contains("segment 7"),
            "{err}"
        );
    }
}
#[test]
fn v1_only_committed_and_needs_clock() {
    let (dir, mut json) = fixture(-9);
    for status in ["diagnostic", "running"] {
        json["batch"]["status"] = status.into();
        let err = read(&dir, &json).unwrap_err().to_string();
        assert!(err.contains("meta.json") && err.contains(status), "{err}");
    }
    json["batch"]["status"] = "committed".into();
    let meta = read(&dir, &json).unwrap();
    let plan = PowerPlan::toggles(Selection::all());
    let err = compute_batch(&meta, &plan, &Sampling::Legacy, LengthPolicy::Pad)
        .unwrap_err()
        .to_string();
    assert!(err.contains("--clock"), "{err}");
    compute_batch(
        &meta,
        &plan,
        &Sampling::Edges(EdgeSampling {
            clock: "tb.s0".into(),
            kind: EdgeKind::Both,
            offset: 0,
        }),
        LengthPolicy::Pad,
    )
    .unwrap();
}

#[test]
fn v1_preserves_names_seeds_extensions_and_optional_waveform() {
    let (dir, mut json) = fixture(-9);
    let meta = read(&dir, &json).unwrap();
    let v = meta.v1.unwrap();
    assert_eq!(v.batch.seeds["base"], 17);
    assert_eq!(v.labels[&1], "random");
    assert_eq!(v.extensions["7"]["note"], "kept");
    json.as_object_mut().unwrap().remove("waveform");
    assert!(read(&dir, &json).unwrap().v1.unwrap().waveform.is_none());
}

#[test]
fn time_units_handle_factors_and_large_exact_values() {
    use scasim::metadata::TimeUnit;
    let ps = TimeUnit {
        mantissa: 1,
        exponent: -12,
    };
    let two_ns = TimeUnit {
        mantissa: 2,
        exponent: -9,
    };
    assert_eq!(ps.ticks(4000, &two_ns), Some(2));
    assert_eq!(ps.ticks(3000, &two_ns), None);
    let huge = TimeUnit {
        mantissa: u64::MAX,
        exponent: -12,
    };
    assert_eq!(huge.ticks(u64::MAX, &huge), Some(u64::MAX));
    assert_eq!(
        TimeUnit {
            mantissa: 1,
            exponent: i32::MIN
        }
        .ticks(1, &ps),
        None
    );
    assert_eq!(
        TimeUnit {
            mantissa: 1,
            exponent: i32::MAX
        }
        .ticks(1, &ps),
        None
    );
}

#[test]
fn v1_legacy_mode_error_names_metadata_file() {
    let (dir, json) = fixture(-9);
    let meta = read(&dir, &json).unwrap();
    let plan = PowerPlan::toggles(Selection::all());
    let err = compute_batch(&meta, &plan, &Sampling::Legacy, LengthPolicy::Pad)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("meta.json") && err.contains("--clock"),
        "{err}"
    );
}
