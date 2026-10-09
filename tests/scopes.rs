//! Per-scope channels: grouping of the selected signals, aliases, and partition of the traces.

mod common;

use common::*;
use scasim::batch::{BatchMeta, LengthPolicy, Sampling, compute_batch};
use scasim::hierarchy::Selection;
use scasim::power::{PowerPlan, hierarchy_index};
use scasim::scopes::{ScopeGroups, group_by_scope, scope_plan};
use std::path::Path;

/// Signals (scope, name, width):
///   tb.clk
///   tb.dut.top_reg                direct member of tb.dut
///   tb.dut.a.x, tb.dut.a.y        scope a
///   tb.dut.a.sub.s                scope a.sub, two levels below tb.dut
///   tb.dut.b.z
///   tb.other.q                    outside tb.dut
/// Aliases: `x` is also `tb.dut.b.deep.x_alias` (a deeper scope, in b), and `y` is also
/// `tb.dut.b.y_alias` (the same depth as `tb.dut.a.y`; `a` sorts first).
fn fixture() -> Fixture {
    let mut sim = Sim::new(&[
        ("tb", "clk", 1),
        ("tb.dut", "top_reg", 4),
        ("tb.dut.a", "x", 4),
        ("tb.dut.a", "y", 4),
        ("tb.dut.a.sub", "s", 4),
        ("tb.dut.b", "z", 4),
        ("tb.other", "q", 4),
    ]);
    sim.clock(0, 10, 10, 6, false);
    for (k, sig) in [1usize, 2, 3, 4, 5, 6].into_iter().enumerate() {
        // Signal `sig` changes at the time 13 + 10 * k.
        sim.set(13 + 10 * k as u64, sig, &format!("{:04b}", k + 1));
    }
    sim.fx.aliases = vec![
        ("tb.dut.b.deep".into(), "x_alias".into(), 2),
        ("tb.dut.b".into(), "y_alias".into(), 3),
    ];
    sim.finish()
}

fn paths(fx: &Fixture) -> (tempfile::TempDir, std::path::PathBuf) {
    temp_vcd(fx)
}

fn groups(path: &Path, rules: &[&str], scope: &str, depth: usize) -> ScopeGroups {
    let index = hierarchy_index(path).unwrap();
    let selected = Selection::parse(rules)
        .unwrap()
        .resolve(&index)
        .unwrap()
        .selected;
    group_by_scope(&index, &selected, scope, depth)
}

fn summary(g: &ScopeGroups) -> Vec<(String, usize)> {
    g.channels
        .iter()
        .map(|c| (c.name.clone(), c.paths.len()))
        .collect()
}

#[test]
fn depth_one_gives_the_child_scopes_and_the_scope_itself() {
    let (_dir, path) = paths(&fixture());
    let g = groups(&path, &[], "tb.dut", 1);
    // tb.dut: top_reg. tb.dut.a: y and s (a.sub is folded into a at depth 1). tb.dut.b: z and
    // x (the alias of x in tb.dut.b.deep is the deepest path of x).
    assert_eq!(
        summary(&g),
        [
            ("(outside tb.dut)".to_string(), 2),
            ("tb.dut".to_string(), 1),
            ("tb.dut.a".to_string(), 2),
            ("tb.dut.b".to_string(), 2)
        ]
    );
    // clk and q have no path in tb.dut. They are in the channel `(outside tb.dut)`.
    assert_eq!(g.outside, 2);
}

#[test]
fn depth_two_splits_the_nested_scope() {
    let (_dir, path) = paths(&fixture());
    let g = groups(&path, &[], "tb.dut", 2);
    assert_eq!(
        summary(&g),
        [
            ("(outside tb.dut)".to_string(), 2),
            ("tb.dut".to_string(), 1),
            ("tb.dut.a".to_string(), 1),
            ("tb.dut.a.sub".to_string(), 1),
            ("tb.dut.b".to_string(), 1),
            ("tb.dut.b.deep".to_string(), 1)
        ]
    );
}

#[test]
fn a_handle_with_several_paths_belongs_to_the_channel_of_its_deepest_scope() {
    let (_dir, path) = paths(&fixture());
    let g = groups(&path, &[], "tb.dut", 1);
    // x: tb.dut.a.x (depth 3) and tb.dut.b.deep.x_alias (depth 4): the alias wins.
    // y: tb.dut.a.y and tb.dut.b.y_alias, both depth 3: the smaller path wins.
    let all: Vec<&str> = g
        .channels
        .iter()
        .flat_map(|c| &c.paths)
        .map(String::as_str)
        .collect();
    assert!(all.contains(&"tb.dut.b.deep.x_alias"), "{all:?}");
    assert!(!all.contains(&"tb.dut.a.x"), "{all:?}");
    assert!(all.contains(&"tb.dut.a.y"), "{all:?}");
    assert!(!all.contains(&"tb.dut.b.y_alias"), "{all:?}");
    // x and y have two paths in tb.dut each.
    assert_eq!(g.aliased, 2);
}

#[test]
fn every_selected_handle_in_the_scope_is_in_exactly_one_channel() {
    let (_dir, path) = paths(&fixture());
    let index = hierarchy_index(&path).unwrap();
    for depth in 1..=3 {
        let g = groups(&path, &[], "tb.dut", depth);
        let mut seen = std::collections::HashSet::new();
        for channel in &g.channels {
            for p in &channel.paths {
                let handle = index
                    .paths
                    .iter()
                    .position(|ps| ps.iter().any(|sp| &sp.path == p))
                    .unwrap();
                assert!(seen.insert(handle), "handle {handle} is in two channels");
            }
        }
        // top_reg, x, y, s, z, and the two signals outside the scope: clk and q.
        assert_eq!(seen.len(), 7, "depth {depth}");
    }
}

#[test]
fn a_scope_without_selected_signals_makes_no_channel() {
    let (_dir, path) = paths(&fixture());
    // Leave out z and the alias paths: scope b keeps no signal that has its deepest path there.
    let g = groups(
        &path,
        &[
            "-scope:tb.dut.b",
            "-signal:tb.dut.a.x",
            "-signal:tb.dut.a.y",
        ],
        "tb.dut",
        1,
    );
    let names: Vec<&str> = g.channels.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        ["(outside tb.dut)", "tb.dut", "tb.dut.a"],
        "{names:?}"
    );
    // With nothing selected outside the scope, there is no outside channel.
    let g = groups(&path, &["+scope:tb.dut", "-scope:tb.dut.b"], "tb.dut", 1);
    let names: Vec<&str> = g.channels.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["tb.dut", "tb.dut.a"], "{names:?}");
    // A scope that does not exist holds nothing: every selected signal is outside it.
    let g = groups(&path, &[], "tb.nope", 1);
    assert_eq!(summary(&g), [("(outside tb.nope)".to_string(), 7)]);
    // Nothing selected, no channel at all.
    assert!(
        groups(&path, &["+scope:tb.nothing"], "tb.dut", 1)
            .channels
            .is_empty()
    );
}

#[test]
fn the_channel_traces_add_up_to_the_trace_of_the_whole_selection_bit_for_bit() {
    let fx = fixture();
    let (_dir, path) = paths(&fx);
    let index = hierarchy_index(&path).unwrap();
    // The default selection includes `tb.clk` and `tb.other.q`, outside `tb.dut`.
    let base = Selection::all();
    let selected = base.resolve(&index).unwrap().selected;
    let g = group_by_scope(&index, &selected, "tb.dut", 1);
    let plan = scope_plan(base, &g);
    assert_eq!(plan.channels.len(), 1 + g.channels.len());
    assert!(plan.channels.iter().any(|c| c.name == "(outside tb.dut)"));
    assert_eq!(plan.channels[0].name, "total");
    let meta = BatchMeta {
        trace_path: path.clone(),
        clock_period: None,
        markers: vec![(10, 40, 0), (40, 70, 1)],
    };
    for sampling in [Sampling::Legacy, edges()] {
        let out = compute_batch(&meta, &plan, &sampling, LengthPolicy::Pad).unwrap();
        let total = &out.channels[0];
        let mut sum = ndarray::Array2::<f32>::zeros(total.dim());
        for channel in &out.channels[1..] {
            sum += channel;
        }
        assert_eq!(&sum, total);
        assert!(total.iter().any(|&v| v > 0.0));
    }
    // The same plan on the FST file.
    let (_fst_dir, fst) = temp_fst(&fx);
    let fst_meta = BatchMeta {
        trace_path: fst,
        ..meta
    };
    let out = compute_batch(&fst_meta, &plan, &edges(), LengthPolicy::Pad).unwrap();
    let mut sum = ndarray::Array2::<f32>::zeros(out.channels[0].dim());
    for channel in &out.channels[1..] {
        sum += channel;
    }
    assert_eq!(sum, out.channels[0]);
}

fn edges() -> Sampling {
    Sampling::Edges(scasim::batch::EdgeSampling {
        clock: "tb.clk".into(),
        kind: scasim::power::edges::EdgeKind::Rising,
        offset: 0,
    })
}

#[test]
fn the_plan_has_a_channel_for_each_group_and_resolves_to_its_signals() {
    let (_dir, path) = paths(&fixture());
    let index = hierarchy_index(&path).unwrap();
    let base = Selection::all();
    let selected = base.resolve(&index).unwrap().selected;
    let g = group_by_scope(&index, &selected, "tb.dut", 1);
    let plan = scope_plan(base, &g);
    for (channel, group) in plan.channels[1..].iter().zip(&g.channels) {
        assert_eq!(channel.name, group.name);
        let resolved = channel.selection.resolve(&index).unwrap();
        assert_eq!(
            resolved.selected.iter().filter(|&&s| s).count(),
            group.paths.len()
        );
    }
    let _: &PowerPlan = &plan;
}
