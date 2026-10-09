//! Probes: the value changes of one signal, on the fast path (FST) and the reference path (VCD).

mod common;

use common::*;
use scasim::power::probe::{ProbeTrace, probe_changes};

/// A clock `tb.clk` and a 4-bit signal `tb.d`. The clock has the states 0, 1, x, and z.
fn fixture() -> Fixture {
    let mut fx = Fixture::flat(&[1, 4]);
    fx.signals[0].name = "clk".into();
    fx.signals[1].name = "d".into();
    let clk = |v: &str| (0usize, v.to_string());
    fx.steps = vec![
        (5, vec![clk("1")]),
        (10, vec![clk("0"), (1, "0101".into())]),
        (15, vec![clk("1")]),
        (20, vec![clk("x")]),
        (25, vec![clk("1")]),
        (30, vec![clk("z")]),
        (35, vec![clk("0")]),
        // `d` changes, the clock does not.
        (40, vec![(1, "1111".into())]),
    ];
    fx
}

fn expected() -> ProbeTrace {
    ProbeTrace {
        path: "tb.clk".into(),
        initial: Some("0".into()),
        changes: vec![
            (5, "1".into()),
            (10, "0".into()),
            (15, "1".into()),
            (20, "x".into()),
            (25, "1".into()),
            (30, "z".into()),
            (35, "0".into()),
        ],
    }
}

#[test]
fn fst_and_vcd_give_the_same_changes() {
    let fx = fixture();
    let (_fst_dir, fst) = temp_fst(&fx);
    let (_vcd_dir, vcd) = temp_vcd(&fx);
    assert_eq!(probe_changes(&fst, "tb.clk").unwrap(), expected());
    assert_eq!(probe_changes(&vcd, "tb.clk").unwrap(), expected());
}

#[test]
fn a_vector_probe_returns_its_state_characters() {
    let fx = fixture();
    let (_dir, fst) = temp_fst(&fx);
    let (_vcd_dir, vcd) = temp_vcd(&fx);
    let want = ProbeTrace {
        path: "tb.d".into(),
        initial: Some("0000".into()),
        changes: vec![(10, "0101".into()), (40, "1111".into())],
    };
    assert_eq!(probe_changes(&fst, "tb.d").unwrap(), want);
    assert_eq!(probe_changes(&vcd, "tb.d").unwrap(), want);
}

#[test]
fn a_change_to_the_same_value_is_not_a_change() {
    let mut fx = Fixture::flat(&[1]);
    fx.steps = vec![
        (5, vec![(0, "0".into())]),
        (10, vec![(0, "1".into())]),
        (15, vec![(0, "1".into())]),
    ];
    let (_dir, fst) = temp_fst(&fx);
    let (_vcd_dir, vcd) = temp_vcd(&fx);
    for path in [fst, vcd] {
        let probe = probe_changes(&path, "tb.s0").unwrap();
        assert_eq!(probe.changes, vec![(10, "1".to_string())]);
    }
}

#[test]
fn an_unknown_path_is_an_error_with_at_most_ten_candidates() {
    let mut fx = Fixture::flat(&[1; 12]);
    for (i, s) in fx.signals.iter_mut().enumerate() {
        s.scope = format!("tb.u{i:02}");
        s.name = "clk".into();
    }
    fx.steps = vec![(5, vec![(0, "1".into())])];
    let (_dir, fst) = temp_fst(&fx);
    let (_vcd_dir, vcd) = temp_vcd(&fx);
    for path in [fst, vcd] {
        let message = probe_changes(&path, "tb.clk").unwrap_err().to_string();
        assert!(message.contains("tb.clk"), "{message}");
        assert_eq!(message.matches("tb.u").count(), 10, "{message}");
        assert!(message.contains("tb.u00.clk"), "{message}");
        assert!(!message.contains("tb.u11.clk"), "{message}");
    }
}
