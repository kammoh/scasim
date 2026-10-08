//! Tests of the reference path (`wellen`) on VCD files, and of what only it does.

mod common;

use common::*;
use scasim::power::reference::{activity_reference, activity_reference_binned};
use scasim::power::{Bins, PowerError, RunInfo, Totals, UnknownPolicy};

#[test]
fn reference_matches_the_oracle_on_a_vcd_file() {
    let fx = small_fixture();
    let (_d, path) = temp_vcd(&fx);
    for unknown in [
        UnknownPolicy::AsZero,
        UnknownPolicy::AsOne,
        UnknownPolicy::Half,
    ] {
        for full in [true, false] {
            let expected = expected_activity(&fx, &[("all", vec![0, 1, 2])], full, unknown);
            assert_eq!(
                activity_reference(&path, &plan_all(full, unknown)).unwrap(),
                expected,
                "{unknown:?}, full {full}"
            );
        }
    }
}

#[test]
fn reference_matches_the_hand_computed_values_on_a_vcd_file() {
    let (_d, path) = temp_vcd(&small_fixture());
    let got = activity_reference(&path, &plan_all(true, UnknownPolicy::Half)).unwrap();
    assert_eq!(got.times, SMALL_TIMES.to_vec());
    assert_eq!(got.timescale_exponent, Some(-12));
    let ch = &got.channels[0];
    assert_eq!(ch.toggles, SMALL_TOGGLES.to_vec());
    let f = ch.full.as_ref().unwrap();
    assert_eq!(f.rise, SMALL_RISE.to_vec());
    assert_eq!(f.fall, SMALL_FALL.to_vec());
    assert_eq!(f.hw_delta, SMALL_HW_DELTA_HALF.to_vec());
    assert_eq!(
        got.info,
        RunInfo {
            selected_handles: 3,
            top_scopes: vec!["tb".into()],
            unmatched_rules: vec![],
        }
    );
}

#[test]
fn bins_work_on_a_vcd_file() {
    let fx = small_fixture();
    let (_d, path) = temp_vcd(&fx);
    let bins = Bins::new(vec![20, 40], Some(50)).unwrap();
    let expected = rebin(
        &expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half),
        &bins,
    );
    let got =
        activity_reference_binned(&path, &plan_all(true, UnknownPolicy::Half), &bins).unwrap();
    assert_eq!(got, expected);
    assert_eq!(got.channels[0].before.hw_delta, 6);
}

#[test]
fn a_vcd_file_with_only_initial_values_has_one_time_point() {
    let mut fx = Fixture::flat(&[1, 8]);
    fx.initial = vec!["1".into(), "10101010".into()];
    let (_d, path) = temp_vcd(&fx);
    let got = activity_reference(&path, &plan_all(true, UnknownPolicy::Half)).unwrap();
    assert_eq!(got.times, vec![0]);
    assert_eq!(got.channels[0].toggles, vec![0]);
    assert_eq!(got.channels[0].full.as_ref().unwrap().hw_delta, vec![10]);
}

/// Some simulators (Questa, Riviera, VCS) write one variable per bit. `wellen` merges them into
/// one derived signal. The paths and the counts must still refer to the underlying signals.
#[test]
fn bit_blasted_vectors_count_every_bit_once() {
    let mut fx = Fixture::flat(&[1, 1, 1, 1]);
    for (i, s) in fx.signals.iter_mut().take(3).enumerate() {
        s.name = format!("d [{i}]");
    }
    fx.signals[3].name = "other".into();
    fx.steps = vec![
        (10, vec![(0, "1".into()), (1, "1".into()), (3, "1".into())]),
        (20, vec![(1, "0".into()), (2, "1".into())]),
    ];
    let (_d, path) = temp_vcd(&fx);
    let expected = expected_activity(
        &fx,
        &[("bus", vec![0, 1, 2]), ("other", vec![3])],
        true,
        UnknownPolicy::Half,
    );
    let p = plan(
        &[("bus", &["+signal:tb.d"]), ("other", &["+signal:tb.other"])],
        true,
        UnknownPolicy::Half,
    );
    let got = activity_reference(&path, &p).unwrap();
    assert_eq!(got.channels, expected.channels);
    assert_eq!(got.info.selected_handles, 4);
    let all = activity_reference(&path, &plan_all(true, UnknownPolicy::Half)).unwrap();
    assert_eq!(all.channels[0].toggles, vec![0, 3, 2]);
}

#[test]
fn timescales_that_are_not_powers_of_ten_are_unknown() {
    let vcd = |timescale: &str| {
        format!(
            "$timescale {timescale} $end\n$scope module tb $end\n$var wire 1 ! a $end\n\
             $upscope $end\n$enddefinitions $end\n#0\n$dumpvars\n0!\n$end\n#10\n1!\n"
        )
    };
    let dir = tempfile::tempdir().unwrap();
    for (text, expected) in [
        ("1 ps", Some(-12)),
        ("10 ns", Some(-8)),
        ("100 us", Some(-4)),
        ("1 s", Some(0)),
        ("3 ns", None),
        ("20 ps", None),
    ] {
        let path = dir.path().join("t.vcd");
        std::fs::write(&path, vcd(text)).unwrap();
        let got = activity_reference(&path, &plan_all(false, UnknownPolicy::Half)).unwrap();
        assert_eq!(got.timescale_exponent, expected, "timescale {text}");
    }
}

#[test]
fn only_bit_vector_variables_count() {
    let vcd = "$timescale 1ps $end\n\
               $scope module tb $end\n\
               $var wire 4 ! bits $end\n\
               $var real 64 \" r $end\n\
               $var event 1 # e $end\n\
               $upscope $end\n\
               $enddefinitions $end\n\
               #0\n$dumpvars\nb0000 !\nr0 \"\n$end\n\
               #5\nb0001 !\nr1.5 \"\n1#\n\
               #9\nb0011 !\nr2.5 \"\n1#\n";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("types.vcd");
    std::fs::write(&path, vcd).unwrap();
    let got = activity_reference(&path, &plan_all(true, UnknownPolicy::Half)).unwrap();
    assert_eq!(got.times, vec![0, 5, 9]);
    assert_eq!(got.channels[0].toggles, vec![0, 1, 1]);
    assert_eq!(got.info.selected_handles, 1);
}

#[test]
fn a_channel_that_selects_nothing_is_an_error() {
    let (_d, path) = temp_vcd(&small_fixture());
    let p = plan(&[("none", &["+scope:nowhere"])], false, UnknownPolicy::Half);
    assert!(matches!(
        activity_reference(&path, &p),
        Err(PowerError::EmptyChannel(name)) if name == "none"
    ));
}

#[test]
fn the_memory_guard_fails_before_loading_signals() {
    let (_d, path) = temp_vcd(&small_fixture());
    let mut p = plan_all(true, UnknownPolicy::Half);
    p.memory_limit = 10;
    assert!(matches!(
        activity_reference(&path, &p),
        Err(PowerError::Memory { what, .. }) if what.starts_with("the result")
    ));
    assert_eq!(Totals::default().toggles, 0);
}
