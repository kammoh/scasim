//! Clock edges from a probe, and the bins between them.
//!
//! Rising means `0` to `1`, falling means `1` to `0`. A change to or from `x`, `z`, or another
//! state is not an edge. The first value of the signal is not a change, so a clock that starts
//! high has its first rising edge at its first `0` to `1` change.

use super::probe::{ProbeTrace, visit_values};
use super::{Bins, PowerError};

/// Which clock edges open a bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EdgeKind {
    #[default]
    Rising,
    Falling,
    Both,
}

/// The level of a state character: `Some(false)` for `0` and `l`, `Some(true)` for `1` and `h`
/// (in any case), and `None` for any other state.
fn level(value: u8) -> Option<bool> {
    match value.to_ascii_lowercase() {
        b'0' | b'l' => Some(false),
        b'1' | b'h' => Some(true),
        _ => None,
    }
}

/// Collects the edge times of a 1-bit signal from its values, one at a time. It keeps only the
/// previous level and the edge times, so its memory grows with the number of edges only.
pub struct EdgeCollector {
    path: String,
    kind: EdgeKind,
    offset: i64,
    started: bool,
    previous: Option<bool>,
    times: Vec<u64>,
    error: Option<PowerError>,
}

impl EdgeCollector {
    pub fn new(path: &str, kind: EdgeKind, offset: i64) -> Self {
        EdgeCollector {
            path: path.to_string(),
            kind,
            offset,
            started: false,
            previous: None,
            times: Vec::new(),
            error: None,
        }
    }

    fn fail(&mut self, reason: String) {
        self.error.get_or_insert(PowerError::Probe {
            path: self.path.clone(),
            reason,
        });
    }

    /// Adds the value `chars` (state characters) at `time`. The first value is the initial value
    /// and makes no edge.
    pub fn push(&mut self, time: u64, chars: &[u8]) {
        if self.error.is_some() {
            return;
        }
        if chars.len() != 1 {
            self.fail(format!(
                "the clock must be 1 bit wide, but this signal has {} bits",
                chars.len()
            ));
            return;
        }
        let now = level(chars[0]);
        let was_started = std::mem::replace(&mut self.started, true);
        let edge = match (self.previous, now) {
            _ if !was_started => false,
            (Some(false), Some(true)) => self.kind != EdgeKind::Falling,
            (Some(true), Some(false)) => self.kind != EdgeKind::Rising,
            _ => false,
        };
        self.previous = now;
        if !edge {
            return;
        }
        let Some(shifted) = time.checked_add_signed(self.offset) else {
            self.fail(format!(
                "the edge at time {time} moves to a time below 0 (or above the largest time) \
                 with the offset {}",
                self.offset
            ));
            return;
        };
        // A glitch has several edges at one time: they count once.
        if self.times.last() != Some(&shifted) {
            self.times.push(shifted);
        }
    }

    /// The edge times, strictly increasing. Fails if a value was not 1 bit wide, if a shifted
    /// time is below 0, if the signal never had a value, or if there are fewer than 2 edges.
    pub fn finish(mut self) -> Result<Vec<u64>, PowerError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.started {
            self.fail("the clock signal has no value".into());
        } else if self.times.len() < 2 {
            let reason = format!(
                "the clock has {} {:?} edges, but sampling needs at least 2",
                self.times.len(),
                self.kind
            );
            self.fail(reason);
        }
        match self.error {
            Some(error) => Err(error),
            None => Ok(self.times),
        }
    }
}

/// The times of the chosen edges of a 1-bit probe, each shifted by `offset` ticks. The times
/// are strictly increasing: edges at the same time (a glitch) count once. Fails if the probe is
/// not 1 bit wide, if a shifted time is below 0, or if there are fewer than 2 edges.
pub fn edge_times(probe: &ProbeTrace, kind: EdgeKind, offset: i64) -> Result<Vec<u64>, PowerError> {
    let mut collector = EdgeCollector::new(&probe.path, kind, offset);
    if let Some(initial) = &probe.initial {
        collector.push(0, initial.as_bytes());
    }
    for (time, value) in &probe.changes {
        collector.push(*time, value.as_bytes());
    }
    collector.finish()
}

/// Reads the signal with the exact path `signal_path` from a waveform file and returns its edge
/// times (see [`edge_times`]). The values go straight into the edge collector: the memory does
/// not grow with the number of value changes.
pub fn probe_edge_times(
    path: &std::path::Path,
    signal_path: &str,
    kind: EdgeKind,
    offset: i64,
) -> Result<Vec<u64>, PowerError> {
    let mut collector = EdgeCollector::new(signal_path, kind, offset);
    visit_values(path, signal_path, |time, chars| collector.push(time, chars))?;
    collector.finish()
}

/// The statistics of the clock periods, for the report.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeSummary {
    /// The number of edges. The number of bins is one less.
    pub edges: usize,
    pub first_edge: u64,
    pub min_period: u64,
    pub max_period: u64,
    pub mean_period: f64,
}

/// Summarizes edge times from [`edge_times`] (at least 2, strictly increasing).
pub fn summarize_edges(edges: &[u64]) -> EdgeSummary {
    let periods = edges.windows(2).map(|w| w[1] - w[0]);
    let (min_period, max_period, sum) = periods.fold((u64::MAX, 0, 0u128), |(min, max, sum), p| {
        (min.min(p), max.max(p), sum + u128::from(p))
    });
    let count = edges.len().saturating_sub(1).max(1);
    EdgeSummary {
        edges: edges.len(),
        first_edge: edges.first().copied().unwrap_or(0),
        min_period: if edges.len() < 2 { 0 } else { min_period },
        max_period,
        mean_period: sum as f64 / count as f64,
    }
}

/// The bins between the edges: bin `k` covers `[edges[k], edges[k + 1])`. The last edge only
/// closes the last bin. `edges` come from [`edge_times`].
pub fn edge_bins(edges: &[u64]) -> Result<Bins, PowerError> {
    match edges {
        [starts @ .., last] if !starts.is_empty() => Bins::new(starts.to_vec(), Some(*last)),
        _ => Err(PowerError::Bins(
            "edge bins need at least 2 edges".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(initial: &str, changes: &[(u64, &str)]) -> ProbeTrace {
        ProbeTrace {
            path: "tb.clk".into(),
            initial: Some(initial.into()),
            changes: changes.iter().map(|&(t, v)| (t, v.to_string())).collect(),
        }
    }

    /// A clock with a phase: rising at 3, 13, 23, 33; falling at 8, 18, 28.
    fn clock() -> ProbeTrace {
        probe(
            "0",
            &[
                (3, "1"),
                (8, "0"),
                (13, "1"),
                (18, "0"),
                (23, "1"),
                (28, "0"),
                (33, "1"),
            ],
        )
    }

    #[test]
    fn rising_falling_and_both_pick_the_right_changes() {
        let c = clock();
        assert_eq!(
            edge_times(&c, EdgeKind::Rising, 0).unwrap(),
            [3, 13, 23, 33]
        );
        assert_eq!(edge_times(&c, EdgeKind::Falling, 0).unwrap(), [8, 18, 28]);
        assert_eq!(
            edge_times(&c, EdgeKind::Both, 0).unwrap(),
            [3, 8, 13, 18, 23, 28, 33]
        );
    }

    #[test]
    fn the_offset_shifts_every_edge() {
        let c = clock();
        assert_eq!(
            edge_times(&c, EdgeKind::Rising, 2).unwrap(),
            [5, 15, 25, 35]
        );
        assert_eq!(
            edge_times(&c, EdgeKind::Rising, -3).unwrap(),
            [0, 10, 20, 30]
        );
        let err = edge_times(&c, EdgeKind::Rising, -4)
            .unwrap_err()
            .to_string();
        assert!(err.contains("below 0"), "{err}");
    }

    #[test]
    fn a_clock_that_starts_high_opens_its_first_bin_at_its_first_rising_edge() {
        let c = probe("1", &[(5, "0"), (10, "1"), (15, "0"), (20, "1")]);
        assert_eq!(edge_times(&c, EdgeKind::Rising, 0).unwrap(), [10, 20]);
        // The first value is not a transition, but the first change is a falling edge.
        assert_eq!(edge_times(&c, EdgeKind::Falling, 0).unwrap(), [5, 15]);
    }

    #[test]
    fn x_and_z_make_no_edge() {
        // x -> 0 and 1 -> x -> 1 are no edges. 0 -> 1 at 4 and at 14 are.
        let c = probe(
            "x",
            &[
                (2, "0"),
                (4, "1"),
                (6, "x"),
                (8, "1"),
                (10, "z"),
                (12, "0"),
                (14, "1"),
            ],
        );
        assert_eq!(edge_times(&c, EdgeKind::Rising, 0).unwrap(), [4, 14]);
        assert_eq!(edge_times(&c, EdgeKind::Both, 0).unwrap(), [4, 14]);
        // The clock never goes from 1 to 0 directly: 1 -> x, 1 -> z, and z -> 0 are no edges.
        assert!(edge_times(&c, EdgeKind::Falling, 0).is_err());
    }

    #[test]
    fn edges_at_the_same_time_count_once() {
        let c = probe("0", &[(5, "1"), (5, "0"), (5, "1"), (9, "0"), (12, "1")]);
        assert_eq!(edge_times(&c, EdgeKind::Rising, 0).unwrap(), [5, 12]);
        assert_eq!(edge_times(&c, EdgeKind::Both, 0).unwrap(), [5, 9, 12]);
    }

    #[test]
    fn fewer_than_two_edges_is_an_error() {
        let one = probe("0", &[(5, "1")]);
        let err = edge_times(&one, EdgeKind::Rising, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least 2"), "{err}");
        let none = probe("0", &[]);
        assert!(edge_times(&none, EdgeKind::Both, 0).is_err());
        let never = ProbeTrace {
            path: "tb.clk".into(),
            initial: None,
            changes: vec![],
        };
        assert!(edge_times(&never, EdgeKind::Both, 0).is_err());
    }

    #[test]
    fn a_probe_wider_than_one_bit_is_an_error() {
        let wide = probe("00", &[(5, "01"), (10, "10")]);
        let err = edge_times(&wide, EdgeKind::Both, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 bits"), "{err}");
    }

    #[test]
    fn bins_run_from_edge_to_edge_and_the_last_edge_closes_the_last_bin() {
        let bins = edge_bins(&[10, 20, 35]).unwrap();
        assert_eq!(bins.starts(), &[10, 20]);
        assert_eq!(bins.end(), Some(35));
        assert_eq!(bins.slot_of(9), 0);
        assert_eq!(bins.slot_of(10), 1);
        assert_eq!(bins.slot_of(34), 2);
        assert_eq!(bins.slot_of(35), 3);
        assert!(edge_bins(&[10]).is_err());
    }

    #[test]
    fn the_summary_has_the_period_statistics() {
        let s = summarize_edges(&[10, 20, 35, 45]);
        assert_eq!(s.edges, 4);
        assert_eq!(s.first_edge, 10);
        assert_eq!((s.min_period, s.max_period), (10, 15));
        assert!((s.mean_period - 35.0 / 3.0).abs() < 1e-12);
    }
}
