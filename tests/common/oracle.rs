//! The independent oracle: expected activity computed from the change list only, without any
//! waveform reader and without code from `src/` (except the result types). Also the plan
//! helpers and a small fixture with hand-computed results.

use super::fixture::*;
use scasim::hierarchy::Selection;
use scasim::power::{
    ActivityTrace, Bins, ChannelSpec, ChannelTrace, FullStats, PowerPlan, RunInfo, Totals,
    UnknownPolicy,
};

/// A plan with the given channels (name and rules, for example `+signal:tb.s0`).
pub fn plan(channels: &[(&str, &[&str])], full: bool, unknown: UnknownPolicy) -> PowerPlan {
    PowerPlan {
        channels: channels
            .iter()
            .map(|(name, rules)| ChannelSpec {
                name: name.to_string(),
                selection: Selection::parse(rules).unwrap(),
            })
            .collect(),
        full_stats: full,
        unknown,
        memory_limit: PowerPlan::DEFAULT_MEMORY_LIMIT,
    }
}

/// A plan with one channel `all` that selects every signal.
pub fn plan_all(full: bool, unknown: UnknownPolicy) -> PowerPlan {
    plan(&[("all", &[])], full, unknown)
}

// ---- The independent oracle ----

/// 0 or 1 for the levels, `None` for an unknown state. Written from the documented table.
fn oracle_level(c: u8) -> Option<u8> {
    match c.to_ascii_lowercase() {
        b'0' | b'l' => Some(0),
        b'1' | b'h' => Some(1),
        _ => None,
    }
}

/// Hamming weight in half-bit units.
fn oracle_weight(c: u8, unknown: UnknownPolicy) -> i64 {
    match oracle_level(c) {
        Some(0) => 0,
        Some(_) => 2,
        None => match unknown {
            UnknownPolicy::AsZero => 0,
            UnknownPolicy::AsOne => 2,
            UnknownPolicy::Half => 1,
        },
    }
}

/// The statistics of a change from `old` to `new`. `old == None` is a first value.
fn oracle_change(old: Option<&str>, new: &str, unknown: UnknownPolicy) -> Totals {
    let mut t = Totals::default();
    for (k, b) in new.bytes().enumerate() {
        t.hw_delta += oracle_weight(b, unknown);
        if let Some(old) = old {
            let a = old.as_bytes()[k];
            t.hw_delta -= oracle_weight(a, unknown);
            if !a.eq_ignore_ascii_case(&b) {
                t.toggles += 1;
            }
            if oracle_level(a) == Some(0) && oracle_level(b) == Some(1) {
                t.rise += 1;
            }
            if oracle_level(a) == Some(1) && oracle_level(b) == Some(0) {
                t.fall += 1;
            }
        }
    }
    t
}

/// The expected activity of `fx`, computed from the change list only. `channels` lists each
/// channel's name and the indices of the signals it contains. There is one time point for the
/// initial values at time 0 and one per step. A file without steps has no time point and so
/// no activity. The `unmatched_rules` of the result are empty.
pub fn expected_activity(
    fx: &Fixture,
    channels: &[(&str, Vec<usize>)],
    full: bool,
    unknown: UnknownPolicy,
) -> ActivityTrace {
    let n = if fx.steps.is_empty() {
        0
    } else {
        fx.steps.len() + 1
    };
    let mut times = Vec::new();
    if n > 0 {
        times.push(0);
        times.extend(fx.steps.iter().map(|(t, _)| *t));
    }
    let mut out_channels: Vec<ChannelTrace> = channels
        .iter()
        .map(|(name, _)| ChannelTrace {
            name: name.to_string(),
            toggles: vec![0; n],
            full: full.then(|| FullStats {
                rise: vec![0; n],
                fall: vec![0; n],
                hw_delta: vec![0; n],
            }),
            before: Totals::default(),
            after: Totals::default(),
        })
        .collect();
    if n > 0 {
        let mut add = |point: usize, sig: usize, t: Totals| {
            for (c, (_, members)) in channels.iter().enumerate() {
                if members.contains(&sig) {
                    let ch = &mut out_channels[c];
                    ch.toggles[point] += t.toggles;
                    if let Some(fs) = &mut ch.full {
                        fs.rise[point] += t.rise;
                        fs.fall[point] += t.fall;
                        fs.hw_delta[point] += t.hw_delta;
                    }
                }
            }
        };
        let mut current = fx.initial.clone();
        for (sig, v) in current.iter().enumerate() {
            add(0, sig, oracle_change(None, v, unknown));
        }
        for (k, (_, changes)) in fx.steps.iter().enumerate() {
            for (sig, v) in changes {
                add(k + 1, *sig, oracle_change(Some(&current[*sig]), v, unknown));
                current[*sig] = v.clone();
            }
        }
    }
    let mut selected: Vec<usize> = channels
        .iter()
        .flat_map(|(_, members)| members.iter().copied())
        .collect();
    selected.sort();
    selected.dedup();
    let mut top_scopes: Vec<String> = selected
        .iter()
        .flat_map(|&s| fx.paths(s))
        .map(|p| p.split('.').next().unwrap().to_string())
        .collect();
    top_scopes.sort();
    top_scopes.dedup();
    ActivityTrace {
        times,
        timescale_exponent: Some(fx.timescale_exponent),
        channels: out_channels,
        info: RunInfo {
            selected_handles: selected.len(),
            top_scopes,
            unmatched_rules: vec![],
        },
    }
}

/// The expected result in `bins`, computed from the expected result with one bin per time point
/// (`identity`). Every time point goes to the first matching place: before the first bin,
/// after the end, or the last bin that starts at or before it. This is a plain linear search.
pub fn rebin(identity: &ActivityTrace, bins: &Bins) -> ActivityTrace {
    let starts = bins.starts();
    let nbins = starts.len();
    #[derive(Clone, Copy)]
    enum Place {
        Before,
        Bin(usize),
        After,
    }
    let place = |t: u64| {
        if bins.end().is_some_and(|end| t >= end) {
            Place::After
        } else if nbins == 0 || t < starts[0] {
            Place::Before
        } else {
            // The last bin that starts at or before `t`.
            Place::Bin((0..nbins).rev().find(|&k| starts[k] <= t).unwrap())
        }
    };
    let channels = identity
        .channels
        .iter()
        .map(|ch| {
            let mut out = ChannelTrace {
                name: ch.name.clone(),
                toggles: vec![0; nbins],
                full: ch.full.as_ref().map(|_| FullStats {
                    rise: vec![0; nbins],
                    fall: vec![0; nbins],
                    hw_delta: vec![0; nbins],
                }),
                before: ch.before,
                after: ch.after,
            };
            for (i, &t) in identity.times.iter().enumerate() {
                let point = Totals {
                    toggles: ch.toggles[i],
                    rise: ch.full.as_ref().map_or(0, |f| f.rise[i]),
                    fall: ch.full.as_ref().map_or(0, |f| f.fall[i]),
                    hw_delta: ch.full.as_ref().map_or(0, |f| f.hw_delta[i]),
                };
                let add = |total: &mut Totals| {
                    total.toggles += point.toggles;
                    total.rise += point.rise;
                    total.fall += point.fall;
                    total.hw_delta += point.hw_delta;
                };
                match place(t) {
                    Place::Before => add(&mut out.before),
                    Place::After => add(&mut out.after),
                    Place::Bin(k) => {
                        out.toggles[k] += point.toggles;
                        if let Some(f) = &mut out.full {
                            f.rise[k] += point.rise;
                            f.fall[k] += point.fall;
                            f.hw_delta[k] += point.hw_delta;
                        }
                    }
                }
            }
            out
        })
        .collect();
    ActivityTrace {
        times: starts.to_vec(),
        timescale_exponent: identity.timescale_exponent,
        channels,
        info: identity.info.clone(),
    }
}

// ---- Small hand-computed fixture ----

/// Three signals (1, 4, and 70 bits) with 2- and 4-state changes and a nonzero initial value.
/// The hand-computed results with `UnknownPolicy::Half` are in the `SMALL_*` constants.
pub fn small_fixture() -> Fixture {
    let mut fx = Fixture::flat(&[1, 4, 70]);
    fx.initial = vec!["0".into(), "0110".into(), "0".repeat(70)];
    fx.steps = vec![
        // s0: 0 to 1 (1 rise). s1: 0110 to 1010 (1 rise, 1 fall). Hamming weight +2 +0.
        (10, vec![(0, "1".into()), (1, "1010".into())]),
        // s2: 1 rise, Hamming weight +2.
        (20, vec![(2, format!("1{}", "0".repeat(69)))]),
        // Nothing changes.
        (30, vec![]),
        // s1: 1010 to x01z: 2 other changes (1 to x, 0 to z). Hamming weight 4 to 1 + 2 + 1 = 4.
        (40, vec![(1, "x01z".into())]),
        // s1: x01z to 0000: 1 fall, 2 other changes. Hamming weight 4 to 0.
        (50, vec![(1, "0000".into())]),
    ];
    fx
}

pub const SMALL_TIMES: [u64; 6] = [0, 10, 20, 30, 40, 50];
pub const SMALL_TOGGLES: [u64; 6] = [0, 3, 1, 0, 2, 3];
pub const SMALL_RISE: [u64; 6] = [0, 2, 1, 0, 0, 0];
pub const SMALL_FALL: [u64; 6] = [0, 1, 0, 0, 0, 1];
pub const SMALL_HW_DELTA_HALF: [i64; 6] = [4, 2, 2, 0, 0, -4];
