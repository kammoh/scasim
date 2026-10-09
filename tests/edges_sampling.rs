//! Clock-edge sampling: the traces of `compute_batch` against an oracle that sums the planted
//! events per clock period. FST (fast path) and VCD (reference path) fixtures.

mod common;

use common::*;
use ndarray::{Array2, array};
use scasim::batch::{BatchMeta, EdgeSampling, LengthPolicy, Sampling, compute_batch};
use scasim::hierarchy::Selection;
use scasim::power::PowerPlan;
use scasim::power::edges::EdgeKind;
use scasim::power::power_trace;
use std::path::{Path, PathBuf};

/// A planted event: `bits` bits of `tb.data` toggle at `time`.
type Event = (u64, u32);

/// The signals: the clock `tb.clk` (index 0) and the 8-bit `tb.data` (index 1).
fn sim() -> Sim {
    Sim::new(&[("tb", "clk", 1), ("tb", "data", 8)])
}

/// Plants the events on `tb.data`. `events` must be in increasing time order.
fn plant(sim: &mut Sim, events: &[Event]) {
    let mut value = 0u32;
    for &(time, bits) in events {
        value ^= (1 << bits) - 1;
        sim.set(time, 1, &format!("{value:08b}"));
    }
}

/// The sum of the toggles of the events in each bin `[edges[k], edges[k + 1])`.
fn bin_sums(edges: &[u64], events: &[Event]) -> Vec<u64> {
    edges
        .windows(2)
        .map(|w| {
            events
                .iter()
                .filter(|e| w[0] <= e.0 && e.0 < w[1])
                .map(|e| u64::from(e.1))
                .sum()
        })
        .collect()
}

/// The toggles before the first edge and after the last edge.
fn outside(edges: &[u64], events: &[Event]) -> (u64, u64) {
    let sum = |keep: &dyn Fn(u64) -> bool| {
        events
            .iter()
            .filter(|e| keep(e.0))
            .map(|e| u64::from(e.1))
            .sum()
    };
    (
        sum(&|t| t < edges[0]),
        sum(&|t| t >= *edges.last().unwrap()),
    )
}

/// Both file formats of one fixture.
struct Files {
    _dirs: Vec<tempfile::TempDir>,
    paths: Vec<PathBuf>,
}

fn files(fx: &Fixture) -> Files {
    let (fst_dir, fst) = temp_fst(fx);
    let (vcd_dir, vcd) = temp_vcd(fx);
    Files {
        _dirs: vec![fst_dir, vcd_dir],
        paths: vec![fst, vcd],
    }
}

fn meta(path: &Path, clock_period: Option<u64>, markers: Vec<(u64, u64, u16)>) -> BatchMeta {
    BatchMeta {
        trace_path: path.to_path_buf(),
        clock_period,
        markers,
    }
}

fn edges_sampling(kind: EdgeKind, offset: i64) -> Sampling {
    Sampling::Edges(EdgeSampling {
        clock: "tb.clk".into(),
        kind,
        offset,
    })
}

/// A plan for the data signal only (the clock is left out).
fn data_plan() -> PowerPlan {
    PowerPlan::toggles(Selection::parse(&["-signal:tb.clk"]).unwrap())
}

/// Segments of `per` bins each, with the labels 0, 1, 0, ...
fn segments(edges: &[u64], per: usize) -> Vec<(u64, u64, u16)> {
    (0..(edges.len() - 1) / per)
        .map(|i| (edges[i * per], edges[(i + 1) * per], (i % 2) as u16))
        .collect()
}

/// The traces that the oracle expects for `segments` of `per` bins.
fn expected_traces(sums: &[u64], per: usize) -> Array2<f32> {
    let rows = sums.len() / per;
    Array2::from_shape_fn((rows, per), |(i, j)| sums[i * per + j] as f32)
}

/// The total toggles of the data signal, from a run that does not use any clock.
fn total_toggles(path: &Path) -> u64 {
    power_trace(path, &Selection::parse(&["-signal:tb.clk"]).unwrap())
        .unwrap()
        .0
        .total()
}

/// Events in many cycles: some at the edges, some inside the cycles, some before the first
/// edge, some after the last one. `first_edge` and `period` describe the rising edges.
fn events(first_edge: u64, period: u64, cycles: u64) -> Vec<Event> {
    let mut events = vec![(1, 2)];
    for k in 0..cycles {
        let edge = first_edge + k * period;
        if k % 2 == 1 {
            events.push((edge, 1));
        }
        events.push((edge + 1, (k % 4 + 1) as u32));
    }
    events.push((first_edge + (cycles - 1) * period + period - 1, 3));
    events.sort_unstable();
    events.dedup_by_key(|e| e.0);
    events
}

/// Runs the case on both formats and checks the traces, the labels, and the conservation.
fn check(mut sim: Sim, events: &[Event], kind: EdgeKind, offset: i64, edges: &[u64], per: usize) {
    plant(&mut sim, events);
    let fx = sim.finish();
    let files = files(&fx);
    let sums = bin_sums(edges, events);
    let marks = segments(edges, per);
    for path in &files.paths {
        let m = meta(path, None, marks.clone());
        let out = compute_batch(
            &m,
            &data_plan(),
            &edges_sampling(kind, offset),
            LengthPolicy::Error,
        )
        .unwrap();
        let name = path.display();
        assert_eq!(out.channels[0], expected_traces(&sums, per), "{name}");
        let labels: Vec<u16> = marks.iter().map(|m| m.2).collect();
        assert_eq!(out.labels.to_vec(), labels, "{name}");
        let report = out.diagnostics.edges.as_ref().expect("an edge report");
        let (before, after) = outside(edges, events);
        assert_eq!(report.summary.edges, edges.len());
        assert_eq!(report.before, before, "{name}");
        assert_eq!(report.after, after, "{name}");
        assert_eq!(report.inside, sums.iter().sum::<u64>(), "{name}");
        // Conservation: inside + before + after is the total of an independent run.
        assert_eq!(report.total(), total_toggles(path), "{name}");
        assert_eq!(
            report.total(),
            events.iter().map(|e| u64::from(e.1)).sum::<u64>(),
            "{name}"
        );
    }
}

fn rising(first: u64, period: u64, count: u64) -> Vec<u64> {
    (0..count).map(|k| first + k * period).collect()
}

#[test]
fn rising_edges_with_a_phase_offset() {
    let mut s = sim();
    s.clock(0, 3, 10, 13, false);
    check(
        s,
        &events(3, 10, 13),
        EdgeKind::Rising,
        0,
        &rising(3, 10, 13),
        4,
    );
}

#[test]
fn falling_edges_only() {
    let mut s = sim();
    s.clock(0, 3, 10, 13, false);
    // Falling edges: 8, 18, ..., 128.
    check(
        s,
        &events(3, 10, 13),
        EdgeKind::Falling,
        0,
        &rising(8, 10, 13),
        4,
    );
}

#[test]
fn both_edges() {
    let mut s = sim();
    s.clock(0, 3, 10, 13, false);
    let both: Vec<u64> = (0..26).map(|k| 3 + 5 * k).collect();
    check(s, &events(3, 10, 13), EdgeKind::Both, 0, &both, 5);
}

#[test]
fn x_on_the_clock_at_the_start_makes_no_edge() {
    let mut s = sim();
    s.initial(0, "x");
    s.set(2, 0, "0");
    s.clock(0, 7, 10, 13, false);
    check(
        s,
        &events(7, 10, 13),
        EdgeKind::Rising,
        0,
        &rising(7, 10, 13),
        4,
    );
}

#[test]
fn a_clock_that_starts_high_opens_its_first_bin_at_its_first_rising_edge() {
    let mut s = sim();
    s.clock(0, 15, 10, 13, true);
    // The clock is high at time 0 and falls at 10. The first rising edge is at 15.
    check(
        s,
        &events(15, 10, 13),
        EdgeKind::Rising,
        0,
        &rising(15, 10, 13),
        4,
    );
}

#[test]
fn the_offset_moves_the_bins() {
    let mut s = sim();
    s.clock(0, 3, 10, 13, false);
    // Edges at 3 + 10k, shifted by 2: the bins start at 5 + 10k.
    check(
        s,
        &events(3, 10, 13),
        EdgeKind::Rising,
        2,
        &rising(5, 10, 13),
        4,
    );
}

#[test]
fn activity_inside_a_cycle_counts_with_edges_and_is_dropped_by_the_legacy_sampling() {
    // Rising edges at 10, 20, ..., 130. Data toggles at 14, 24, ... (4 ticks after each edge).
    let mut s = sim();
    s.clock(0, 10, 10, 13, false);
    let planted: Vec<Event> = (0..12).map(|k| (14 + 10 * k, (k % 3 + 1) as u32)).collect();
    plant(&mut s, &planted);
    let fx = s.finish();
    let edges = rising(10, 10, 13);
    let marks = segments(&edges, 4);
    for path in &files(&fx).paths {
        let with_edges = compute_batch(
            &meta(path, None, marks.clone()),
            &data_plan(),
            &edges_sampling(EdgeKind::Rising, 0),
            LengthPolicy::Error,
        )
        .unwrap();
        assert_eq!(
            with_edges.channels[0],
            expected_traces(&bin_sums(&edges, &planted), 4)
        );
        assert!(with_edges.channels[0].iter().all(|&v| v > 0.0));
        // The legacy sampling keeps the time points 10, 20, ...: no data toggle is there.
        let legacy = compute_batch(
            &meta(path, Some(10), marks.clone()),
            &data_plan(),
            &Sampling::Legacy,
            LengthPolicy::Pad,
        )
        .unwrap();
        assert!(legacy.channels[0].iter().all(|&v| v == 0.0));
        assert_eq!(legacy.diagnostics.kept_toggles, 0);
        assert!(legacy.diagnostics.total_toggles > 0);
        assert_eq!(legacy.diagnostics.edges, None);
    }
}

#[test]
fn the_clock_is_in_the_selection_by_default() {
    let mut s = sim();
    s.clock(0, 10, 10, 6, false);
    let fx = s.finish();
    let edges = rising(10, 10, 6);
    let marks = vec![(10, 60, 0)];
    for path in &files(&fx).paths {
        let out = compute_batch(
            &meta(path, None, marks.clone()),
            &PowerPlan::toggles(Selection::all()),
            &edges_sampling(EdgeKind::Rising, 0),
            LengthPolicy::Error,
        )
        .unwrap();
        // Each bin [10k, 10k + 10) holds the rising edge at 10k and the falling edge at 10k + 5.
        assert_eq!(out.channels[0], array![[2.0, 2.0, 2.0, 2.0, 2.0]]);
        let report = out.diagnostics.edges.unwrap();
        // The bin of the last edge is not there: its rising edge is "after", and so is its fall.
        assert_eq!((report.inside, report.after), (10, 2));
        assert_eq!(report.before, 0);
        assert_eq!(edges.len(), report.summary.edges);
    }
}

#[test]
fn segments_that_start_at_different_places_in_the_clock_period_give_a_warning_value() {
    let mut s = sim();
    s.clock(0, 10, 10, 13, false);
    let fx = s.finish();
    let aligned = vec![(10, 50, 0), (50, 90, 1)];
    // The second segment starts 1 tick after its first edge: its first bin is the next one.
    let shifted = vec![(10, 50, 0), (51, 91, 1)];
    for path in &files(&fx).paths {
        let run = |marks: &Vec<(u64, u64, u16)>| {
            compute_batch(
                &meta(path, None, marks.clone()),
                &data_plan_with_clock(),
                &edges_sampling(EdgeKind::Rising, 0),
                LengthPolicy::Error,
            )
            .unwrap()
            .diagnostics
            .edges
            .unwrap()
        };
        let report = run(&aligned);
        assert_eq!((report.offset_min, report.offset_max), (0, 0));
        assert!(!report.offset_varies());
        // [51, 91) holds the bins that start at 60, 70, 80, and 90.
        let report = run(&shifted);
        assert_eq!((report.offset_min, report.offset_max), (0, 9));
        assert!(report.offset_varies());
    }
}

fn data_plan_with_clock() -> PowerPlan {
    PowerPlan::toggles(Selection::all())
}

#[test]
fn a_segment_without_a_bin_is_an_error() {
    let mut s = sim();
    s.clock(0, 10, 10, 13, false);
    let fx = s.finish();
    for path in &files(&fx).paths {
        let result = compute_batch(
            &meta(path, None, vec![(10, 50, 0), (52, 58, 1)]),
            &data_plan_with_clock(),
            &edges_sampling(EdgeKind::Rising, 0),
            LengthPolicy::Pad,
        );
        let message = format!("{:?}", result.expect_err("an error"));
        assert!(message.contains("no clock edge"), "{message}");
        assert!(message.contains("[52, 58)"), "{message}");
    }
}

#[test]
fn the_length_policies_apply_to_segments_of_different_length() {
    let mut s = sim();
    s.clock(0, 10, 10, 13, false);
    let fx = s.finish();
    // Bins start at 10, 20, ..., 120. Class 0 has 3 bins, class 1 has 4 and 2.
    let marks = vec![(10, 40, 0), (40, 80, 1), (80, 100, 1), (100, 130, 0)];
    for path in &files(&fx).paths {
        let run = |policy| {
            compute_batch(
                &meta(path, None, marks.clone()),
                &data_plan_with_clock(),
                &edges_sampling(EdgeKind::Rising, 0),
                policy,
            )
        };
        let pad = run(LengthPolicy::Pad).unwrap();
        assert_eq!(pad.channels[0].dim(), (4, 4));
        // Each bin holds the rising edge and the falling edge of the clock: 2 toggles.
        assert_eq!(pad.channels[0].row(0).to_vec(), [2.0, 2.0, 2.0, 0.0]);
        let truncate = run(LengthPolicy::Truncate).unwrap();
        assert_eq!(truncate.channels[0].dim(), (4, 2));
        let error = run(LengthPolicy::Error).expect_err("an error").to_string();
        assert!(error.contains("class 0: 3 samples x 2"), "{error}");
        assert!(
            error.contains("class 1: 2 samples x 1, 4 samples x 1"),
            "{error}"
        );
    }
}

#[test]
fn a_clock_path_that_does_not_exist_is_an_error_with_candidates() {
    let mut s = sim();
    s.clock(0, 10, 10, 4, false);
    let fx = s.finish();
    for path in &files(&fx).paths {
        let result = compute_batch(
            &meta(path, None, vec![(10, 30, 0)]),
            &data_plan_with_clock(),
            &Sampling::Edges(EdgeSampling {
                clock: "tb.nope.clk".into(),
                kind: EdgeKind::Rising,
                offset: 0,
            }),
            LengthPolicy::Error,
        );
        let message = format!("{:?}", result.expect_err("an error"));
        assert!(message.contains("tb.clk"), "{message}");
    }
}

/// A multi-section FST file gives the same traces as the VCD file.
#[test]
fn sections_do_not_change_the_edges() {
    let mut s = sim();
    s.clock(0, 3, 10, 13, false);
    let evs = events(3, 10, 13);
    plant(&mut s, &evs);
    let mut fx = s.finish();
    fx.flush_before = vec![3, 12, 24];
    let edges = rising(3, 10, 13);
    let marks = segments(&edges, 4);
    let (_dir, fst) = temp_fst(&fx);
    // `fst-writer` 0.3.1 sometimes writes a time table that nothing can read. The flush points
    // above give a readable file.
    assert!(time_table_is_readable(&fst, &fx));
    let reader = fst_reader::FstReader::open_and_read_time_table(std::io::BufReader::new(
        std::fs::File::open(&fst).unwrap(),
    ))
    .unwrap();
    assert_eq!(reader.sections().len(), 4);
    let out = compute_batch(
        &meta(&fst, None, marks),
        &data_plan(),
        &edges_sampling(EdgeKind::Rising, 0),
        LengthPolicy::Error,
    )
    .unwrap();
    assert_eq!(out.channels[0], expected_traces(&bin_sums(&edges, &evs), 4));
}
