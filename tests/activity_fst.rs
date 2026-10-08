//! Tests of the fast FST path against hand-computed values, the independent oracle, and the
//! reference path.

mod common;

use common::*;
use proptest::prelude::*;
use scasim::hierarchy::SelectionError;
use scasim::power::fst::{activity_fst, activity_fst_binned};
use scasim::power::reference::{activity_reference, activity_reference_binned};
use scasim::power::{
    ActivityTrace, Bins, PowerError, RunInfo, Totals, UnknownPolicy, activity, activity_binned,
};
use std::panic::AssertUnwindSafe;
use std::path::Path;

const POLICIES: [UnknownPolicy; 3] = [
    UnknownPolicy::AsZero,
    UnknownPolicy::AsOne,
    UnknownPolicy::Half,
];

fn totals(toggles: u64, rise: u64, fall: u64, hw_delta: i64) -> Totals {
    Totals {
        toggles,
        rise,
        fall,
        hw_delta,
    }
}

// ---- Hand-computed values ----

#[test]
fn the_oracle_matches_the_hand_computed_values() {
    let got = expected_activity(
        &small_fixture(),
        &[("all", vec![0, 1, 2])],
        true,
        UnknownPolicy::Half,
    );
    assert_eq!(got.times, SMALL_TIMES.to_vec());
    assert_eq!(got.channels[0].toggles, SMALL_TOGGLES.to_vec());
    let f = got.channels[0].full.as_ref().unwrap();
    assert_eq!(f.rise, SMALL_RISE.to_vec());
    assert_eq!(f.fall, SMALL_FALL.to_vec());
    assert_eq!(f.hw_delta, SMALL_HW_DELTA_HALF.to_vec());
}

#[test]
fn both_paths_match_the_hand_computed_values() {
    let (_d, path) = temp_fst(&small_fixture());
    let plan = plan_all(true, UnknownPolicy::Half);
    for got in [
        activity_fst(&path, &plan).unwrap(),
        activity_reference(&path, &plan).unwrap(),
    ] {
        assert_eq!(got.times, SMALL_TIMES.to_vec());
        assert_eq!(got.timescale_exponent, Some(-12));
        let ch = &got.channels[0];
        assert_eq!(ch.name, "all");
        assert_eq!(ch.toggles, SMALL_TOGGLES.to_vec());
        let f = ch.full.as_ref().unwrap();
        assert_eq!(f.rise, SMALL_RISE.to_vec());
        assert_eq!(f.fall, SMALL_FALL.to_vec());
        assert_eq!(f.hw_delta, SMALL_HW_DELTA_HALF.to_vec());
        assert_eq!(
            (ch.before, ch.after),
            (Totals::default(), Totals::default())
        );
        assert_eq!(
            got.info,
            RunInfo {
                selected_handles: 3,
                top_scopes: vec!["tb".into()],
                unmatched_rules: vec![],
            }
        );
    }
}

#[test]
fn without_full_statistics_only_toggles_are_recorded() {
    let (_d, path) = temp_fst(&small_fixture());
    let plan = plan_all(false, UnknownPolicy::Half);
    for got in [
        activity_fst(&path, &plan).unwrap(),
        activity_reference(&path, &plan).unwrap(),
    ] {
        assert_eq!(got.channels[0].toggles, SMALL_TOGGLES.to_vec());
        assert!(got.channels[0].full.is_none());
        assert_eq!(got.channels[0].before, Totals::default());
        assert_eq!(got.channels[0].after, Totals::default());
    }
}

#[test]
fn the_unknown_policy_changes_only_the_hamming_weight() {
    let fx = small_fixture();
    let (_d, path) = temp_fst(&fx);
    // Hand values. s1 goes 1010 -> x01z -> 0000 at the times 40 and 50.
    // AsZero: the Hamming weight 4 -> 2 -> 0. AsOne: 4 -> 6 -> 0. Half: 4 -> 4 -> 0.
    let hand = [
        (UnknownPolicy::AsZero, [4, 2, 2, 0, -2, -2]),
        (UnknownPolicy::AsOne, [4, 2, 2, 0, 2, -6]),
        (UnknownPolicy::Half, [4, 2, 2, 0, 0, -4]),
    ];
    for (unknown, hw_delta) in hand {
        let plan = plan_all(true, unknown);
        let expected = expected_activity(&fx, &[("all", vec![0, 1, 2])], true, unknown);
        assert_eq!(
            expected.channels[0].full.as_ref().unwrap().hw_delta,
            hw_delta.to_vec(),
            "oracle {unknown:?}"
        );
        assert_both_paths(&path, &plan, &expected);
        assert_eq!(expected.channels[0].toggles, SMALL_TOGGLES.to_vec());
    }
}

#[test]
fn values_carry_over_section_boundaries() {
    let mut fx = small_fixture();
    fx.flush_before = vec![1, 2, 3, 4];
    let (_d, path) = temp_fst(&fx);
    let reader =
        fst_reader::FstReader::open(std::io::BufReader::new(std::fs::File::open(&path).unwrap()))
            .unwrap();
    assert_eq!(
        reader.sections().len(),
        5,
        "the fixture has several sections"
    );
    let expected = expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half);
    assert_both_paths(&path, &plan_all(true, UnknownPolicy::Half), &expected);
}

#[test]
fn an_aliased_signal_counts_once() {
    let mut fx = small_fixture();
    fx.aliases = vec![
        ("tb".into(), "alias0".into(), 0),
        ("tb".into(), "alias2".into(), 2),
    ];
    let (_d, path) = temp_fst(&fx);
    let expected = expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half);
    assert_both_paths(&path, &plan_all(true, UnknownPolicy::Half), &expected);
    assert_eq!(expected.info.selected_handles, 3);
}

#[test]
fn channels_follow_their_rules() {
    let fx = small_fixture();
    let (_d, path) = temp_fst(&fx);
    // One channel by an include rule and one by an exclude rule.
    let plan = plan(
        &[("wide", &["+signal:tb.s2"]), ("rest", &["-signal:tb.s2"])],
        true,
        UnknownPolicy::Half,
    );
    let expected = expected_activity(
        &fx,
        &[("wide", vec![2]), ("rest", vec![0, 1])],
        true,
        UnknownPolicy::Half,
    );
    assert_both_paths(&path, &plan, &expected);
}

#[test]
fn dispatch_uses_the_fast_path_for_fst_and_the_reference_for_vcd() {
    let fx = small_fixture();
    let (_d1, fst) = temp_fst(&fx);
    let (_d2, vcd) = temp_vcd(&fx);
    let plan = plan_all(true, UnknownPolicy::Half);
    let expected = expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half);
    assert_eq!(
        activity(&fst, &plan).unwrap(),
        activity_fst(&fst, &plan).unwrap()
    );
    assert_eq!(
        activity(&vcd, &plan).unwrap(),
        activity_reference(&vcd, &plan).unwrap()
    );
    assert_eq!(activity(&fst, &plan).unwrap(), expected);
    assert_eq!(activity(&vcd, &plan).unwrap(), expected);
    let bins = Bins::new(vec![5, 25], None).unwrap();
    let rebinned = rebin(&expected, &bins);
    assert_eq!(activity_binned(&fst, &plan, &bins).unwrap(), rebinned);
    assert_eq!(activity_binned(&vcd, &plan, &bins).unwrap(), rebinned);
}

#[test]
fn timescales_are_reported_as_powers_of_ten() {
    for exponent in [-15, -12, -9, -8, -6, -3, -1, 0] {
        let mut fx = small_fixture();
        fx.timescale_exponent = exponent;
        let (_d, path) = temp_fst(&fx);
        let got = activity_fst(&path, &plan_all(false, UnknownPolicy::Half)).unwrap();
        assert_eq!(got.timescale_exponent, Some(exponent), "fast path");
        let got = activity_reference(&path, &plan_all(false, UnknownPolicy::Half)).unwrap();
        assert_eq!(got.timescale_exponent, Some(exponent), "reference path");
    }
}

// ---- Edge cases ----

/// `run_tvla.py` allows signals of up to 16,384 bits.
#[test]
fn very_wide_signals() {
    let w = 16_384usize;
    let a = "01".repeat(w / 2);
    let b = "10".repeat(w / 2);
    let mut c = b.clone().into_bytes();
    c[0] = b'x';
    c[w - 1] = b'z';
    let c = String::from_utf8(c).unwrap();
    let mut fx = Fixture::flat(&[w as u32, 65]);
    fx.initial = vec![a.clone(), "0".repeat(65)];
    fx.steps = vec![
        (10, vec![(0, b.clone())]),              // all 16,384 bits change
        (20, vec![(0, c), (1, "1".repeat(65))]), // first and last bit change state, plus 65 rises
        (30, vec![(0, a)]),                      // all bits change again
    ];
    fx.flush_before = vec![2];
    let (_d, path) = temp_fst(&fx);
    let expected = expected_activity(&fx, &[("all", vec![0, 1])], true, UnknownPolicy::Half);
    assert_eq!(
        expected.channels[0].toggles,
        vec![0, 16_384, 2 + 65, 16_384]
    );
    assert_both_paths(&path, &plan_all(true, UnknownPolicy::Half), &expected);
}

/// Only initial values, no time step: the file has no time table, so the result is empty.
#[test]
fn no_steps_gives_an_empty_result() {
    let mut fx = Fixture::flat(&[1, 8]);
    fx.initial = vec!["1".into(), "10101010".into()];
    let (_d, path) = temp_fst(&fx);
    for full in [false, true] {
        let expected = expected_activity(&fx, &[("all", vec![0, 1])], full, UnknownPolicy::Half);
        assert!(expected.times.is_empty());
        assert_both_paths(&path, &plan_all(full, UnknownPolicy::Half), &expected);
    }
    // Bins on an empty file: the bins exist and have no activity.
    let bins = Bins::new(vec![0, 10], Some(20)).unwrap();
    let got = activity_fst_binned(&path, &plan_all(true, UnknownPolicy::Half), &bins).unwrap();
    assert_eq!(got.times, vec![0, 10]);
    assert_eq!(got.channels[0].toggles, vec![0, 0]);
    assert_eq!(got.channels[0].before, Totals::default());
    let reference =
        activity_reference_binned(&path, &plan_all(true, UnknownPolicy::Half), &bins).unwrap();
    assert_eq!(reference, got);
}

#[test]
fn a_step_without_changes_has_only_the_initial_hamming_weight() {
    let mut fx = Fixture::flat(&[1, 8]);
    fx.initial = vec!["1".into(), "10101010".into()];
    fx.steps = vec![(10, vec![])];
    let (_d, path) = temp_fst(&fx);
    let expected = expected_activity(&fx, &[("all", vec![0, 1])], true, UnknownPolicy::Half);
    assert_eq!(expected.times, vec![0, 10]);
    // "1" weighs 2 half-bits, "10101010" weighs 8.
    assert_eq!(
        expected.channels[0].full.as_ref().unwrap().hw_delta,
        vec![10, 0]
    );
    assert_both_paths(&path, &plan_all(true, UnknownPolicy::Half), &expected);
}

#[test]
fn changes_after_the_header_end_time_are_ignored() {
    let fx = small_fixture();
    let (_d, path) = temp_fst(&fx);
    patch_header_end_time(&path, 20);
    // `read_signals` drops the changes at 30 and later. The time points stay.
    let mut cut = fx.clone();
    for (time, changes) in &mut cut.steps {
        if *time > 20 {
            changes.clear();
        }
    }
    let expected = expected_activity(&cut, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half);
    assert_eq!(expected.times, SMALL_TIMES.to_vec());
    assert_eq!(expected.channels[0].toggles, vec![0, 3, 1, 0, 0, 0]);
    assert_both_paths(&path, &plan_all(true, UnknownPolicy::Half), &expected);

    // Also in bins: the bin at 30 exists and has no activity.
    let bins = Bins::new(vec![0, 30], Some(60)).unwrap();
    let rebinned = rebin(&expected, &bins);
    assert_eq!(rebinned.channels[0].toggles, vec![4, 0]);
    assert_both_paths_binned(
        &path,
        &plan_all(true, UnknownPolicy::Half),
        &bins,
        &rebinned,
    );
}

/// One signal changes several times at the same time. All changes count.
#[test]
fn glitches_inside_one_time_step_count() {
    let mut fx = Fixture::flat(&[1, 3]);
    fx.initial = vec!["0".into(), "000".into()];
    fx.steps = vec![(
        10,
        vec![
            (0, "1".into()),
            (0, "0".into()),
            (0, "1".into()),
            (1, "111".into()),
            (1, "010".into()),
        ],
    )];
    let (_d, path) = temp_fst(&fx);
    let plan = plan(
        &[("glitchy", &["+signal:tb.s0"]), ("bus", &["+signal:tb.s1"])],
        true,
        UnknownPolicy::Half,
    );
    let expected = expected_activity(
        &fx,
        &[("glitchy", vec![0]), ("bus", vec![1])],
        true,
        UnknownPolicy::Half,
    );
    // s0: 0 -> 1 -> 0 -> 1 is 3 toggles, 2 rises, 1 fall, and the Hamming weight +2.
    let s0 = &expected.channels[0];
    assert_eq!(s0.toggles, vec![0, 3]);
    let f = s0.full.as_ref().unwrap();
    assert_eq!(
        (f.rise.clone(), f.fall.clone(), f.hw_delta.clone()),
        (vec![0, 2], vec![0, 1], vec![0, 2])
    );
    // s1: 000 -> 111 (3 rises) -> 010 (2 falls): 5 toggles, Hamming weight +2.
    let s1 = &expected.channels[1];
    assert_eq!(s1.toggles, vec![0, 5]);
    assert_eq!(s1.full.as_ref().unwrap().hw_delta, vec![0, 2]);
    assert_both_paths(&path, &plan, &expected);
}

/// The characters `x z h u w l - ?` and their uppercase forms, on a 1-bit and an 8-bit signal.
#[test]
fn state_characters_follow_the_normalization_table() {
    for code in "xzhuwl-?XZHUWL".chars() {
        let mut fx = Fixture::flat(&[1, 8]);
        fx.initial = vec!["0".into(), "10100101".into()];
        let pattern = format!("{code}1{code}0{code}{code}01");
        fx.steps = vec![
            (10, vec![(0, code.to_string()), (1, pattern.clone())]),
            (20, vec![(0, "1".into()), (1, "01010101".into())]),
            (30, vec![(0, "0".into()), (1, pattern)]),
            (40, vec![(0, code.to_string()), (1, "00000000".into())]),
        ];
        let (_d, path) = temp_fst(&fx);
        for unknown in POLICIES {
            for full in [true, false] {
                let expected = expected_activity(&fx, &[("all", vec![0, 1])], full, unknown);
                let plan = plan_all(full, unknown);
                assert_eq!(
                    activity_fst(&path, &plan).unwrap(),
                    expected,
                    "fast path, code {code}, {unknown:?}"
                );
                // `wellen` cannot represent `?` and panics on it.
                if code != '?' {
                    assert_eq!(
                        activity_reference(&path, &plan).unwrap(),
                        expected,
                        "reference path, code {code}, {unknown:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn h_and_l_have_levels_but_are_different_characters() {
    let mut fx = Fixture::flat(&[8]);
    fx.initial = vec!["10101010".into()];
    fx.steps = vec![
        (10, vec![(0, "H0H0H0H0".into())]), // 1 -> H: same level, a toggle
        (20, vec![(0, "L0L0L0L0".into())]), // H -> L: 4 falls
        (30, vec![(0, "00000000".into())]), // L -> 0: same level, a toggle
        (40, vec![(0, "h0h0h0h0".into())]), // 0 -> h: 4 rises
        (50, vec![(0, "H0H0H0H0".into())]), // h -> H: only the case differs, no toggle
    ];
    let (_d, path) = temp_fst(&fx);
    let expected = expected_activity(&fx, &[("all", vec![0])], true, UnknownPolicy::Half);
    let ch = &expected.channels[0];
    assert_eq!(ch.toggles, vec![0, 4, 4, 4, 4, 0]);
    let f = ch.full.as_ref().unwrap();
    assert_eq!(f.rise, vec![0, 0, 0, 0, 4, 0]);
    assert_eq!(f.fall, vec![0, 0, 4, 0, 0, 0]);
    assert_eq!(f.hw_delta, vec![8, 0, -8, 0, 8, 0]);
    assert_both_paths(&path, &plan_all(true, UnknownPolicy::Half), &expected);
}

// ---- Selection ----

#[test]
fn selecting_an_alias_path_selects_the_handle() {
    let mut fx = small_fixture();
    fx.aliases = vec![("tb".into(), "alias0".into(), 0)];
    let (_d, path) = temp_fst(&fx);
    let expected_s0 = expected_activity(&fx, &[("a", vec![0])], true, UnknownPolicy::Half);
    for rules in [
        &["+signal:tb.alias0"][..],
        // Both names of the same signal: still one signal.
        &["+signal:tb.s0", "+signal:tb.alias0"][..],
        &["+regex:tb\\.alias.*"][..],
    ] {
        assert_both_paths(
            &path,
            &plan(&[("a", rules)], true, UnknownPolicy::Half),
            &expected_s0,
        );
    }
}

#[test]
fn an_exclude_by_any_name_removes_the_handle() {
    // The clock net `tb.dut.clk` has the alias `tb.clk`.
    let mut fx = Fixture::flat(&[1, 4]);
    fx.signals[0].scope = "tb.dut".into();
    fx.signals[0].name = "clk".into();
    fx.signals[1].scope = "tb.dut".into();
    fx.signals[1].name = "data".into();
    fx.aliases = vec![("tb".into(), "clk".into(), 0)];
    fx.steps = vec![
        (10, vec![(0, "1".into()), (1, "0011".into())]),
        (20, vec![(0, "0".into()), (1, "1100".into())]),
    ];
    let (_d, path) = temp_fst(&fx);
    let cases: [(&[&str], Vec<usize>); 5] = [
        (&["+scope:tb.dut", "-signal:tb.clk"], vec![1]),
        (&["+scope:tb.dut", "-signal:tb.dut.clk"], vec![1]),
        (&["-signal:tb.clk", "+scope:tb.dut"], vec![0, 1]),
        (&["+scope:tb.dut"], vec![0, 1]),
        (&["-scope:tb.dut", "+signal:tb.clk"], vec![0]),
    ];
    for (rules, members) in cases {
        let expected = expected_activity(&fx, &[("c", members)], true, UnknownPolicy::Half);
        assert_both_paths(
            &path,
            &plan(&[("c", rules)], true, UnknownPolicy::Half),
            &expected,
        );
    }
}

#[test]
fn the_run_info_reports_scopes_and_unmatched_rules() {
    let mut fx = Fixture::flat(&[1, 1]);
    fx.signals[1].scope = "other".into();
    fx.steps = vec![(10, vec![(0, "1".into()), (1, "1".into())])];
    let (_d, path) = temp_fst(&fx);
    let plan = plan(
        &[
            ("a", &["+scope:tb", "-signal:tb.nothing"]),
            ("b", &["+scope:other", "+regex:zzz"]),
        ],
        false,
        UnknownPolicy::Half,
    );
    let want = RunInfo {
        selected_handles: 2,
        top_scopes: vec!["other".into(), "tb".into()],
        unmatched_rules: vec!["-signal:tb.nothing".into(), "+regex:zzz".into()],
    };
    assert_eq!(activity_fst(&path, &plan).unwrap().info, want);
    assert_eq!(activity_reference(&path, &plan).unwrap().info, want);
}

#[test]
fn a_channel_that_selects_nothing_is_an_error() {
    let (_d, path) = temp_fst(&small_fixture());
    let plan = plan(&[("none", &["+scope:nowhere"])], false, UnknownPolicy::Half);
    assert!(matches!(
        activity_fst(&path, &plan),
        Err(PowerError::EmptyChannel(name)) if name == "none"
    ));
    assert!(matches!(
        activity_reference(&path, &plan),
        Err(PowerError::EmptyChannel(name)) if name == "none"
    ));
}

#[test]
fn module_rules_need_module_names() {
    let plan_core = plan(&[("core", &["+module:core"])], true, UnknownPolicy::Half);
    // No module names in this file.
    let mut fx = Fixture::flat(&[1, 1]);
    fx.signals[1].scope = "tb.dut".into();
    fx.steps = vec![(10, vec![(0, "1".into()), (1, "1".into())])];
    let (_d, path) = temp_fst(&fx);
    for result in [
        activity_fst(&path, &plan_core),
        activity_reference(&path, &plan_core),
    ] {
        assert!(matches!(
            result,
            Err(PowerError::Selection(SelectionError::NoModuleNames))
        ));
    }
    // With a module name for `tb.dut`, the rule selects the signals inside.
    fx.modules = vec![("tb.dut".into(), "core".into())];
    let (_d, path) = temp_fst(&fx);
    let expected = expected_activity(&fx, &[("core", vec![1])], true, UnknownPolicy::Half);
    assert_both_paths(&path, &plan_core, &expected);
}

// ---- Bins ----

#[test]
fn bins_collect_before_and_after_activity() {
    let fx = small_fixture();
    let (_d, path) = temp_fst(&fx);
    let plan = plan_all(true, UnknownPolicy::Half);
    // Bins [20, 40) and [40, 50). The times 0 and 10 are before; the time 50 is after.
    let bins = Bins::new(vec![20, 40], Some(50)).unwrap();
    let got = activity_fst_binned(&path, &plan, &bins).unwrap();
    assert_eq!(got.times, vec![20, 40]);
    let ch = &got.channels[0];
    assert_eq!(ch.toggles, vec![1, 2]);
    let f = ch.full.as_ref().unwrap();
    assert_eq!(f.rise, vec![1, 0]);
    assert_eq!(f.fall, vec![0, 0]);
    assert_eq!(f.hw_delta, vec![2, 0]);
    // The initial values (hw 4) and the first step (+2) are before the first bin: the Hamming
    // weight at the start of the first bin is 6 half-bits.
    assert_eq!(ch.before, totals(3, 2, 1, 6));
    assert_eq!(ch.after, totals(3, 0, 1, -4));
    let expected = rebin(
        &expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half),
        &bins,
    );
    assert_eq!(got, expected);
    assert_eq!(
        activity_reference_binned(&path, &plan, &bins).unwrap(),
        expected
    );
}

#[test]
fn an_open_last_bin_takes_everything_after_its_start() {
    let fx = small_fixture();
    let (_d, path) = temp_fst(&fx);
    // Bins [5, 25) and [25, infinity). The bins start between time points.
    let bins = Bins::new(vec![5, 25], None).unwrap();
    let expected = rebin(
        &expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half),
        &bins,
    );
    let ch = &expected.channels[0];
    assert_eq!(ch.toggles, vec![3 + 1, 2 + 3]);
    assert_eq!(ch.before, totals(0, 0, 0, 4));
    assert_eq!(ch.after, Totals::default());
    assert_both_paths_binned(
        &path,
        &plan_all(true, UnknownPolicy::Half),
        &bins,
        &expected,
    );
}

#[test]
fn binned_results_do_not_depend_on_the_sections() {
    let mut fx = small_fixture();
    fx.flush_before = vec![1, 2, 3, 4];
    let (_d, path) = temp_fst(&fx);
    let expected = expected_activity(&fx, &[("all", vec![0, 1, 2])], true, UnknownPolicy::Half);
    for bins in [
        Bins::new(vec![20, 40], Some(50)).unwrap(),
        Bins::new(vec![0], None).unwrap(),
        Bins::new(vec![15, 16, 17, 45], Some(100)).unwrap(),
        // Every time point is after the end.
        Bins::new(vec![1], Some(2)).unwrap(),
    ] {
        assert_both_paths_binned(
            &path,
            &plan_all(true, UnknownPolicy::Half),
            &bins,
            &rebin(&expected, &bins),
        );
    }
}

// ---- Memory guard ----

#[test]
fn the_memory_guard_names_what_is_too_large() {
    let (_d, path) = temp_fst(&small_fixture());
    let mut plan = plan_all(true, UnknownPolicy::Half);
    // The result needs 6 bins * (8 + 32) = 240 bytes.
    plan.memory_limit = 10;
    for result in [activity_fst(&path, &plan), activity_reference(&path, &plan)] {
        assert!(matches!(
            result,
            Err(PowerError::Memory { what, needed: 240, limit: 10 }) if what.starts_with("the result")
        ));
    }
    // The result fits, but the decode buffers of a section do not: at least 2 parts * 5 slots
    // * 32 bytes = 320 bytes.
    plan.memory_limit = 300;
    assert!(activity_reference(&path, &plan).is_ok());
    let err = activity_fst(&path, &plan).unwrap_err();
    assert!(
        matches!(&err, PowerError::Memory { what, limit: 300, .. } if what.starts_with("the decode buffers")),
        "{err}"
    );
}

// ---- Property test ----

/// A character for a state value. Mostly 0 and 1, sometimes any other state.
fn state_char(binary_only: bool) -> BoxedStrategy<u8> {
    if binary_only {
        prop_oneof![Just(b'0'), Just(b'1')].boxed()
    } else {
        prop_oneof![
            10 => Just(b'0'),
            10 => Just(b'1'),
            1 => Just(b'x'),
            1 => Just(b'z'),
            1 => Just(b'X'),
            1 => Just(b'Z'),
            1 => Just(b'h'),
            1 => Just(b'l'),
            1 => Just(b'u'),
            1 => Just(b'w'),
            1 => Just(b'-'),
        ]
        .boxed()
    }
}

/// A value of `width` bits. Three of four values are 2-state, so that wide signals also use the
/// packed representation.
fn value(width: u32) -> impl Strategy<Value = String> {
    (prop::bool::weighted(0.75), Just(width))
        .prop_flat_map(|(binary_only, width)| {
            prop::collection::vec(state_char(binary_only), width as usize)
        })
        .prop_map(|v| String::from_utf8(v).unwrap())
}

/// How to choose bins for a generated fixture.
#[derive(Debug, Clone)]
struct BinSpec {
    /// Which time points start a bin (repeated to cover all points).
    starts: Vec<bool>,
    /// Added to each chosen start, so that bins can start between time points.
    shifts: Vec<u64>,
    /// The end is this far after the last start.
    end_distance: Option<u64>,
}

impl BinSpec {
    fn bins(&self, fx: &Fixture) -> Option<Bins> {
        let times = std::iter::once(0).chain(fx.steps.iter().map(|s| s.0));
        let mut starts: Vec<u64> = times
            .enumerate()
            .filter(|(i, _)| self.starts[i % self.starts.len()])
            .map(|(i, t)| t + self.shifts[i % self.shifts.len()])
            .collect();
        starts.sort();
        starts.dedup();
        let last = *starts.last()?;
        Some(Bins::new(starts, self.end_distance.map(|d| last + d)).unwrap())
    }
}

#[derive(Debug, Clone)]
struct Case {
    fx: Fixture,
    /// Which signals are in the first channel; the others are in the second.
    split: Vec<bool>,
    full: bool,
    unknown: UnknownPolicy,
    bins: BinSpec,
}

fn case() -> impl Strategy<Value = Case> {
    prop::collection::vec(
        prop_oneof![
            Just(1u32),
            Just(2),
            Just(7),
            Just(8),
            Just(9),
            Just(63),
            Just(64),
            Just(65),
            Just(130)
        ],
        1..6,
    )
    .prop_flat_map(|widths| {
        let n = widths.len();
        let initial: Vec<_> = widths.iter().map(|&w| value(w)).collect();
        let ws = widths.clone();
        // A signal can change several times in one step.
        let step = (
            1u64..5,
            prop::collection::vec(0..n, 0..=n + 1).prop_flat_map(move |sigs| {
                let ws = ws.clone();
                sigs.into_iter()
                    .map(move |s| value(ws[s]).prop_map(move |v| (s, v)))
                    .collect::<Vec<_>>()
            }),
        );
        (
            Just(widths),
            prop::collection::vec(0..n, 0..3),
            initial,
            prop::collection::vec(step, 1..40),
            prop::collection::vec(1usize..40, 0..4),
            prop::collection::vec(any::<bool>(), n),
            any::<bool>(),
            prop_oneof![
                Just(UnknownPolicy::AsZero),
                Just(UnknownPolicy::AsOne),
                Just(UnknownPolicy::Half)
            ],
            (
                prop::collection::vec(any::<bool>(), 1..=41),
                prop::collection::vec(0u64..3, 1..=41),
                prop::option::of(1u64..40),
            ),
        )
    })
    .prop_map(
        |(widths, aliases, initial, deltas, flush_before, split, full, unknown, bins)| {
            let mut fx = Fixture::flat(&widths);
            fx.aliases = aliases
                .iter()
                .enumerate()
                .map(|(k, &t)| ("tb".into(), format!("a{k}"), t))
                .collect();
            fx.initial = initial;
            let mut time = 0;
            fx.steps = deltas
                .into_iter()
                .map(|(d, changes)| {
                    time += d;
                    (time, changes)
                })
                .collect();
            fx.flush_before = flush_before;
            Case {
                fx,
                split,
                full,
                unknown,
                bins: BinSpec {
                    starts: bins.0,
                    shifts: bins.1,
                    end_distance: bins.2,
                },
            }
        },
    )
}

/// Runs `f` in a rayon pool with `threads` threads.
fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    /// The oracle computes the expected result from the change list, not from a reader. Both
    /// paths must equal it, with identity bins and with random bins. The result of the fast path
    /// must not depend on the number of threads.
    #[test]
    fn both_paths_match_the_oracle(case in case()) {
        let Case { fx, split, full, unknown, bins } = case;
        let (_d, path) = temp_fst(&fx);
        prop_assume!(time_table_is_readable(&path, &fx));

        // Channel `all`, then the signals of the split as an include list and as an exclude list.
        let first: Vec<usize> = (0..split.len()).filter(|&i| split[i]).collect();
        let second: Vec<usize> = (0..split.len()).filter(|&i| !split[i]).collect();
        let rules = |sign: char, list: &[usize]| -> Vec<String> {
            list.iter().map(|i| format!("{sign}signal:tb.s{i}")).collect()
        };
        let mut specs: Vec<(&str, Vec<String>)> = vec![("all", vec![])];
        let mut members = vec![("all", (0..split.len()).collect::<Vec<_>>())];
        if !first.is_empty() {
            specs.push(("first", rules('+', &first)));
            members.push(("first", first.clone()));
        }
        if !second.is_empty() {
            // Excluding the signals of `first` selects the others.
            specs.push(("second", rules('-', &first)));
            members.push(("second", second.clone()));
        }
        let spec_refs: Vec<(&str, Vec<&str>)> = specs
            .iter()
            .map(|(name, rules)| (*name, rules.iter().map(String::as_str).collect()))
            .collect();
        let channel_args: Vec<(&str, &[&str])> =
            spec_refs.iter().map(|(name, rules)| (*name, rules.as_slice())).collect();
        let p = plan(&channel_args, full, unknown);

        let expected: ActivityTrace = expected_activity(&fx, &members, full, unknown);
        let fast = activity_fst(&path, &p).unwrap();
        prop_assert_eq!(&fast, &expected);
        prop_assert_eq!(&activity_reference(&path, &p).unwrap(), &expected);
        prop_assert_eq!(&with_threads(1, || activity_fst(&path, &p).unwrap()), &fast);
        prop_assert_eq!(&with_threads(4, || activity_fst(&path, &p).unwrap()), &fast);

        if let Some(bins) = bins.bins(&fx) {
            let expected_binned = rebin(&expected, &bins);
            let fast = activity_fst_binned(&path, &p, &bins).unwrap();
            prop_assert_eq!(&fast, &expected_binned);
            prop_assert_eq!(&activity_reference_binned(&path, &p, &bins).unwrap(), &expected_binned);
            prop_assert_eq!(
                &with_threads(1, || activity_fst_binned(&path, &p, &bins).unwrap()),
                &fast
            );
            prop_assert_eq!(
                &with_threads(4, || activity_fst_binned(&path, &p, &bins).unwrap()),
                &fast
            );
        }
    }
}

// ---- Corpus test ----

thread_local! {
    /// While true, panic messages of this thread are not printed.
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` and returns the message of a panic as an error, without printing it.
fn catch_quietly<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET.with(|quiet| quiet.get()) {
                default(info);
            }
        }));
    });
    QUIET.with(|quiet| quiet.set(true));
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    QUIET.with(|quiet| quiet.set(false));
    result.map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into())
    })
}

/// True if `FstReader::read_signals` reads the whole file without an error.
fn read_signals_succeeds(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(mut reader) =
        fst_reader::FstReader::open_and_read_time_table(std::io::BufReader::new(file))
    else {
        return false;
    };
    reader
        .read_signals(&fst_reader::FstFilter::all(), |_, _, _| Ok::<(), ()>(()))
        .is_ok()
}

/// Whether the first section of the file starts with a frame before the first time point.
fn starts_with_a_frame(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(mut reader) = fst_reader::FstReader::open(std::io::BufReader::new(file)) else {
        return false; // an incomplete file
    };
    if reader.sections().is_empty() {
        return false;
    }
    let section = reader.read_section(0).unwrap();
    section
        .time_table()
        .first()
        .is_none_or(|&t| t > section.info().start_time)
}

/// The result of checking one corpus file.
enum Checked {
    /// Both paths read the file and agree. `frame_first` is true if the file starts with a frame
    /// before the first time point.
    Compared {
        frame_first: bool,
    },
    Skipped(String),
}

fn check_corpus_file(path: &Path, name: &str) -> Checked {
    let full = plan_all(true, UnknownPolicy::Half);
    if path.ends_with("fst-writer/multi_vc_block.fst") {
        // The sections of this file overlap in time, so its time table decreases. `wellen`
        // panics on it. The fast path must report the decreasing time.
        assert!(
            matches!(
                activity_fst(path, &full),
                Err(PowerError::TimeTable {
                    previous: 3055,
                    next: 5
                })
            ),
            "{name}"
        );
        return Checked::Skipped("ill-formed upstream: the time table decreases".into());
    }
    let reference = match catch_quietly(|| activity_reference(path, &full)) {
        Ok(Ok(trace)) => Ok(trace),
        Ok(Err(e)) => Err(format!("the reference path fails: {e}")),
        Err(message) => Err(format!("wellen panics: {message}")),
    };
    // The fast path must read every file that `read_signals` reads. If only the reference path
    // reads a file, the fast path must agree with it.
    if let (false, Err(reason)) = (read_signals_succeeds(path), &reference) {
        return Checked::Skipped(reason.clone());
    }
    let fast = activity_fst(path, &full);
    if let Err(PowerError::EmptyChannel(_)) = &fast {
        // Acceptable only if the file has no signal with a bit-vector value.
        let index = scasim::power::hierarchy_index(path).unwrap();
        assert!(index.paths.iter().all(|p| p.is_empty()), "{name}");
        return Checked::Skipped("no selectable signal".into());
    }
    let fast = fast.unwrap_or_else(|e| panic!("{name}: the fast path failed: {e:?}"));
    let reference = match reference {
        Ok(reference) => reference,
        Err(reason) => return Checked::Skipped(reason),
    };
    assert_eq!(fast, reference, "{name}");
    // Every unknown policy, and a selection that excludes every path of the first selectable
    // signal. The second channel `rest` is only in the run with the default policy.
    let index = scasim::power::hierarchy_index(path).unwrap();
    let first = index.paths.iter().find(|p| !p.is_empty()).unwrap();
    let rules: Vec<String> = first
        .iter()
        .map(|p| format!("-signal:{}", p.path))
        .collect();
    let rule_refs: Vec<&str> = rules.iter().map(String::as_str).collect();
    for unknown in POLICIES {
        let mut channels: Vec<(&str, &[&str])> = vec![("all", &[])];
        if unknown == UnknownPolicy::Half {
            channels.push(("rest", &rule_refs));
        }
        let plan = plan(&channels, true, unknown);
        match (activity_fst(path, &plan), activity_reference(path, &plan)) {
            (Ok(f), Ok(r)) => assert_eq!(f, r, "{name}, {unknown:?}"),
            // The only signal is excluded: both paths must say so.
            (Err(PowerError::EmptyChannel(f)), Err(PowerError::EmptyChannel(r))) => {
                assert_eq!((f, r), ("rest".to_string(), "rest".to_string()), "{name}")
            }
            (f, r) => panic!(
                "{name}: fast {:?} and reference {:?} disagree",
                f.err(),
                r.err()
            ),
        }
    }
    Checked::Compared {
        frame_first: starts_with_a_frame(path),
    }
}

/// On every corpus file, the fast path must read what `read_signals` reads. Where the reference
/// path can read the file too, both must give the same result for all three unknown policies and
/// for a selection that excludes the first selectable signal.
#[test]
fn fast_path_matches_reference_on_corpus() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fst-reader/fsts");
    let (mut compared, mut frame_first) = (0, 0);
    let mut skipped = Vec::new();
    for path in corpus_files() {
        let name = path.strip_prefix(&root).unwrap().display().to_string();
        match check_corpus_file(&path, &name) {
            Checked::Compared { frame_first: f } => {
                compared += 1;
                frame_first += usize::from(f);
            }
            Checked::Skipped(reason) => skipped.push(format!("{name}: {reason}")),
        }
    }
    eprintln!(
        "compared {compared} corpus files; {frame_first} start with a frame before the first time point"
    );
    for line in &skipped {
        eprintln!("skipped {line}");
    }
    assert!(compared >= 20, "only {compared} corpus files compared");
}
