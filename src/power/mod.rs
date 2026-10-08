//! Switching activity per time bin and channel. Definitions are in
//! `docs/superpowers/specs/2026-10-08-fast-power-design.md`.

use crate::hierarchy::{HierarchyIndex, Selection, SelectionError};
use std::collections::BTreeSet;
use std::path::Path;

pub mod reference;
mod slots;
pub mod stats;

#[derive(Debug, thiserror::Error)]
pub enum PowerError {
    #[error("failed to read waveform with wellen")]
    Wellen(#[from] wellen::WellenError),
    #[error("failed to read FST file")]
    Fst(#[from] fst_reader::ReaderError),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Selection(#[from] SelectionError),
    #[error("channel `{0}` selects no signals")]
    EmptyChannel(String),
    #[error("the time table of the waveform decreases from {previous} to {next}")]
    TimeTable { previous: u64, next: u64 },
    #[error("invalid bins: {0}")]
    Bins(String),
    #[error(
        "{what} would need about {needed} bytes, more than the limit of {limit} bytes; \
         select fewer signals or channels, or raise the limit"
    )]
    Memory {
        /// What was too large.
        what: String,
        needed: u64,
        limit: u64,
    },
}

/// How a bit state other than `0`, `1`, `l`, and `h` counts in the Hamming weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownPolicy {
    AsZero,
    AsOne,
    #[default]
    Half,
}

/// Statistics of a set of value changes. `toggles` counts bit positions whose state changed;
/// `rise` (0 to 1) and `fall` (1 to 0) are subsets of them; `hw_delta` is the change of the
/// Hamming weight in half-bit units. See [`stats`] for the exact rules.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Totals {
    pub toggles: u64,
    pub rise: u64,
    pub fall: u64,
    pub hw_delta: i64,
}

/// Time bins. Bin `k` covers the times `[starts[k], starts[k + 1])`. The last bin covers
/// `[starts[last], end)`, or all later times if there is no `end`.
///
/// Changes before `starts[0]` count in the channel's `before` totals. Changes at or after `end`
/// count in its `after` totals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bins {
    starts: Vec<u64>,
    end: Option<u64>,
}

impl Bins {
    /// Creates bins. `starts` must be strictly increasing, and `end` must be greater than the
    /// last start. Empty `starts` are valid: then there is no bin at all.
    pub fn new(starts: Vec<u64>, end: Option<u64>) -> Result<Bins, PowerError> {
        if let Some(i) = (1..starts.len()).find(|&i| starts[i - 1] >= starts[i]) {
            return Err(PowerError::Bins(format!(
                "start {i} ({}) must be greater than the start before it ({})",
                starts[i],
                starts[i - 1]
            )));
        }
        if let (Some(end), Some(&last)) = (end, starts.last())
            && end <= last
        {
            return Err(PowerError::Bins(format!(
                "the end {end} must be greater than the last start {last}"
            )));
        }
        if starts.len() >= u32::MAX as usize - 2 {
            return Err(PowerError::Bins(format!(
                "{} bins are too many",
                starts.len()
            )));
        }
        Ok(Bins { starts, end })
    }

    /// One bin per distinct time of a waveform's time table, and no end. A time that repeats
    /// (a section boundary) gives one bin. Fails if the time table decreases.
    pub fn identity(time_table: &[u64]) -> Result<Bins, PowerError> {
        check_time_table(time_table)?;
        let mut starts = time_table.to_vec();
        starts.dedup();
        Bins::new(starts, None)
    }

    pub fn starts(&self) -> &[u64] {
        &self.starts
    }

    pub fn end(&self) -> Option<u64> {
        self.end
    }

    /// The number of bins.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// The slot for `time`: 0 before the first bin, `k + 1` in bin `k`, and `len() + 1` at or
    /// after the end.
    pub(crate) fn slot_of(&self, time: u64) -> usize {
        match self.end {
            Some(end) if time >= end => self.starts.len() + 1,
            _ => self.starts.partition_point(|&start| start <= time),
        }
    }
}

/// Fails if a time table decreases. Equal neighbors are valid.
pub(crate) fn check_time_table(time_table: &[u64]) -> Result<(), PowerError> {
    match time_table.windows(2).find(|w| w[0] > w[1]) {
        Some(w) => Err(PowerError::TimeTable {
            previous: w[0],
            next: w[1],
        }),
        None => Ok(()),
    }
}

/// A named group of signals.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: String,
    pub selection: Selection,
}

/// What to compute from a waveform.
#[derive(Debug, Clone)]
pub struct PowerPlan {
    pub channels: Vec<ChannelSpec>,
    /// Also record rise, fall, and Hamming-weight statistics. Without it, only toggles.
    pub full_stats: bool,
    pub unknown: UnknownPolicy,
    /// Upper limit for the memory of the result and of the decode buffers, in bytes.
    pub memory_limit: u64,
}

/// What a plan selects in one waveform file.
pub(crate) struct ResolvedPlan {
    /// For every handle index, the indices of the channels that select it.
    pub handle_channels: Vec<Vec<usize>>,
    pub info: RunInfo,
}

impl PowerPlan {
    pub const DEFAULT_MEMORY_LIMIT: u64 = 8 << 30;

    /// One channel named `all` with toggle counts of the selected signals.
    pub fn toggles(selection: Selection) -> Self {
        PowerPlan {
            channels: vec![ChannelSpec {
                name: "all".into(),
                selection,
            }],
            full_stats: false,
            unknown: UnknownPolicy::default(),
            memory_limit: Self::DEFAULT_MEMORY_LIMIT,
        }
    }

    /// Applies the selection of every channel. Fails if a channel selects no signal.
    pub(crate) fn resolve(&self, index: &HierarchyIndex) -> Result<ResolvedPlan, PowerError> {
        let mut handle_channels = vec![Vec::new(); index.paths.len()];
        let mut unmatched_rules = Vec::new();
        for (c, channel) in self.channels.iter().enumerate() {
            let resolution = channel.selection.resolve(index)?;
            if !resolution.selected.contains(&true) {
                return Err(PowerError::EmptyChannel(channel.name.clone()));
            }
            for (h, selected) in resolution.selected.into_iter().enumerate() {
                if selected {
                    handle_channels[h].push(c);
                }
            }
            for rule in resolution.unmatched_rules {
                if !unmatched_rules.contains(&rule) {
                    unmatched_rules.push(rule);
                }
            }
        }
        let selected: Vec<usize> = (0..handle_channels.len())
            .filter(|&h| !handle_channels[h].is_empty())
            .collect();
        // The first component of a path is the name of its top-level scope.
        let top_scopes: BTreeSet<String> = selected
            .iter()
            .flat_map(|&h| &index.paths[h])
            .filter_map(|p| p.path.split('.').next())
            .map(str::to_string)
            .collect();
        Ok(ResolvedPlan {
            handle_channels,
            info: RunInfo {
                selected_handles: selected.len(),
                top_scopes: top_scopes.into_iter().collect(),
                unmatched_rules,
            },
        })
    }

    /// Bytes that one channel needs for one bin: 8 for the toggles, or 32 with full statistics.
    fn bytes_per_channel_and_bin(&self) -> u64 {
        if self.full_stats { 32 } else { 8 }
    }

    /// Bytes needed for the result with `bins` bins: 8 bytes per bin for the time, plus the
    /// statistics of every channel.
    pub fn result_bytes(&self, bins: usize) -> u64 {
        let per_bin = 8 + self.channels.len() as u64 * self.bytes_per_channel_and_bin();
        (bins as u64).saturating_mul(per_bin)
    }

    /// Bytes needed for the decode buffers of a section that uses `section_bins` bins, with
    /// `threads` rayon threads. The fast path runs up to `2 * threads` parts at the same time.
    /// Each part has its own buffers for all channels.
    pub fn section_bytes(&self, section_bins: usize, threads: usize) -> u64 {
        let parts = 2 * threads as u64;
        parts
            .saturating_mul(self.channels.len() as u64)
            .saturating_mul(section_bins as u64)
            .saturating_mul(self.bytes_per_channel_and_bin())
    }

    /// Fails if the result for `bins` bins is larger than the memory limit.
    pub fn check_result_memory(&self, bins: usize) -> Result<(), PowerError> {
        self.check(
            self.result_bytes(bins),
            format!("the result ({bins} bins, {} channels)", self.channels.len()),
        )
    }

    /// Fails if the decode buffers of a section that uses `section_bins` bins (with the slots
    /// before the first bin and after the end) are larger than the memory limit.
    pub fn check_section_memory(
        &self,
        section_bins: usize,
        threads: usize,
    ) -> Result<(), PowerError> {
        self.check(
            self.section_bytes(section_bins, threads),
            format!(
                "the decode buffers of one section ({section_bins} bins, {} channels, \
                 {} parallel parts)",
                self.channels.len(),
                2 * threads
            ),
        )
    }

    fn check(&self, needed: u64, what: String) -> Result<(), PowerError> {
        if needed > self.memory_limit {
            return Err(PowerError::Memory {
                what,
                needed,
                limit: self.memory_limit,
            });
        }
        Ok(())
    }
}

/// Rise, fall, and Hamming-weight statistics per bin of one channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FullStats {
    pub rise: Vec<u64>,
    pub fall: Vec<u64>,
    /// Change of the Hamming weight in half-bit units.
    pub hw_delta: Vec<i64>,
}

/// Statistics per bin of one channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelTrace {
    pub name: String,
    /// Toggles per bin.
    pub toggles: Vec<u64>,
    /// Present if the plan asks for full statistics.
    pub full: Option<FullStats>,
    /// Totals of all changes before the first bin. `before.hw_delta` is the Hamming weight at
    /// the start of the first bin.
    pub before: Totals,
    /// Totals of all changes at or after the end of the last bin.
    pub after: Totals,
}

/// What a run selected, for messages to the user.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunInfo {
    /// The number of signal handles that at least one channel selects.
    pub selected_handles: usize,
    /// The first path component of every path of a selected handle, sorted, without duplicates.
    pub top_scopes: Vec<String>,
    /// The text of every selection rule that matches no signal.
    pub unmatched_rules: Vec<String>,
}

/// Switching activity of a waveform: one statistics trace per channel on one time axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivityTrace {
    /// The start time of every bin. Strictly increasing.
    pub times: Vec<u64>,
    /// The time unit of the waveform as a power of ten of seconds (for example `-12` for ps), if
    /// the file gives it as one.
    pub timescale_exponent: Option<i8>,
    pub channels: Vec<ChannelTrace>,
    pub info: RunInfo,
}

/// Toggle count per time point of one group of signals (used by `tvla`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PowerTrace {
    pub times: Vec<u64>,
    pub power: Vec<u64>,
}

impl PowerTrace {
    /// Legacy sampling of `tvla`: keeps only the time points where `time % period == 0` and drops
    /// all other activity. A2 replaces it with per-cycle binning.
    pub fn keep_multiples_of(&self, period: u64) -> PowerTrace {
        assert!(period > 0, "period must be positive");
        let (times, power) = self
            .times
            .iter()
            .zip(&self.power)
            .filter(|(t, _)| *t % period == 0)
            .map(|(t, p)| (*t, *p))
            .unzip();
        PowerTrace { times, power }
    }

    /// The sum of the toggles of all time points.
    pub fn total(&self) -> u64 {
        self.power.iter().sum()
    }
}

/// Computes the activity of a waveform file with one bin per distinct time of its time table.
/// All formats use the `wellen` reference path.
pub fn activity(path: &Path, plan: &PowerPlan) -> Result<ActivityTrace, PowerError> {
    reference::activity_reference(path, plan)
}

/// Computes the activity of a waveform file in the given bins.
pub fn activity_binned(
    path: &Path,
    plan: &PowerPlan,
    bins: &Bins,
) -> Result<ActivityTrace, PowerError> {
    reference::activity_reference_binned(path, plan, bins)
}

/// Toggle counts of the selected signals, as one trace.
pub fn power_trace(
    path: &Path,
    selection: &Selection,
) -> Result<(PowerTrace, RunInfo), PowerError> {
    let activity = activity(path, &PowerPlan::toggles(selection.clone()))?;
    let channel = activity
        .channels
        .into_iter()
        .next()
        .expect("the plan has one channel");
    let trace = PowerTrace {
        times: activity.times,
        power: channel.toggles,
    };
    Ok((trace, activity.info))
}

/// Reads the signal paths of a waveform file, without any values.
pub fn hierarchy_index(path: &Path) -> Result<HierarchyIndex, PowerError> {
    let header = wellen::viewers::read_header_from_file(path, &reference::load_options())?;
    Ok(HierarchyIndex::from_wellen(&header.hierarchy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_must_increase_strictly() {
        assert!(Bins::new(vec![0, 10, 20], None).is_ok());
        assert!(Bins::new(vec![], None).is_ok());
        assert!(matches!(
            Bins::new(vec![0, 10, 10], None),
            Err(PowerError::Bins(_))
        ));
        assert!(matches!(
            Bins::new(vec![5, 3], None),
            Err(PowerError::Bins(_))
        ));
    }

    #[test]
    fn the_end_must_be_greater_than_the_last_start() {
        assert!(Bins::new(vec![0, 10], Some(11)).is_ok());
        assert!(matches!(
            Bins::new(vec![0, 10], Some(10)),
            Err(PowerError::Bins(_))
        ));
        assert!(matches!(
            Bins::new(vec![0, 10], Some(3)),
            Err(PowerError::Bins(_))
        ));
    }

    #[test]
    fn slot_of_places_times_before_in_and_after_the_bins() {
        let bins = Bins::new(vec![10, 20, 30], Some(40)).unwrap();
        let slots: Vec<_> = [0, 9, 10, 19, 20, 39, 40, 99]
            .iter()
            .map(|&t| bins.slot_of(t))
            .collect();
        assert_eq!(slots, vec![0, 0, 1, 1, 2, 3, 4, 4]);
        // Without an end, the last bin is open.
        let open = Bins::new(vec![10, 20, 30], None).unwrap();
        assert_eq!(open.slot_of(1_000_000), 3);
    }

    #[test]
    fn identity_bins_merge_repeated_times_and_reject_decreasing_ones() {
        // A section boundary repeats its time.
        let bins = Bins::identity(&[0, 10, 10, 20, 20, 20, 30]).unwrap();
        assert_eq!(bins.starts(), &[0, 10, 20, 30]);
        assert_eq!(bins.end(), None);
        assert!(Bins::identity(&[]).unwrap().is_empty());
        assert!(matches!(
            Bins::identity(&[0, 10, 5]),
            Err(PowerError::TimeTable {
                previous: 10,
                next: 5
            })
        ));
    }

    #[test]
    fn memory_formulas() {
        let mut plan = PowerPlan::toggles(Selection::all());
        // 8 bytes for the time and 8 bytes for the toggles of the one channel.
        assert_eq!(plan.result_bytes(1000), 1000 * (8 + 8));
        // 2 * 10 parts, 1 channel, 50 bins, 8 bytes.
        assert_eq!(plan.section_bytes(50, 10), 20 * 50 * 8);
        plan.full_stats = true;
        plan.channels.push(plan.channels[0].clone());
        assert_eq!(plan.result_bytes(1000), 1000 * (8 + 2 * 32));
        assert_eq!(plan.section_bytes(50, 10), 20 * 2 * 50 * 32);
    }

    #[test]
    fn memory_errors_name_what_was_too_large() {
        let mut plan = PowerPlan::toggles(Selection::all());
        plan.memory_limit = 1000;
        assert!(plan.check_result_memory(62).is_ok());
        let err = plan.check_result_memory(63).unwrap_err();
        assert!(
            matches!(&err, PowerError::Memory { what, needed: 1008, limit: 1000 } if what.starts_with("the result")),
            "{err}"
        );
        assert!(plan.check_section_memory(6, 1).is_ok());
        let err = plan.check_section_memory(100, 4).unwrap_err();
        assert!(
            matches!(&err, PowerError::Memory { what, .. } if what.starts_with("the decode buffers")),
            "{err}"
        );
    }

    #[test]
    fn keep_multiples_of_keeps_only_multiples() {
        let trace = PowerTrace {
            times: vec![0, 5, 10, 15, 20],
            power: vec![1, 2, 3, 4, 5],
        };
        assert_eq!(
            trace.keep_multiples_of(10),
            PowerTrace {
                times: vec![0, 10, 20],
                power: vec![1, 3, 5]
            }
        );
        assert_eq!(trace.total(), 15);
    }

    #[test]
    fn resolve_reports_the_run_info() {
        use crate::hierarchy::SignalPath;
        let path = |p: &str, scope: &str| SignalPath {
            path: p.into(),
            scope: scope.into(),
            modules: vec![],
            is_alias: false,
        };
        let index = HierarchyIndex {
            paths: vec![
                vec![path("tb.dut.a", "tb.dut"), path("other.a", "other")],
                vec![path("tb.dut.b", "tb.dut")],
                vec![path("tb.c", "tb")],
            ],
            has_module_names: false,
        };
        let plan =
            PowerPlan::toggles(Selection::parse(&["+scope:tb.dut", "+scope:nowhere"]).unwrap());
        let resolved = plan.resolve(&index).unwrap();
        assert_eq!(resolved.handle_channels, vec![vec![0], vec![0], vec![]]);
        assert_eq!(resolved.info.selected_handles, 2);
        // Handle 0 is also visible as `other.a`.
        assert_eq!(resolved.info.top_scopes, vec!["other", "tb"]);
        assert_eq!(resolved.info.unmatched_rules, vec!["+scope:nowhere"]);
    }

    #[test]
    fn resolve_rejects_a_channel_without_signals() {
        let index = HierarchyIndex::default();
        let plan = PowerPlan::toggles(Selection::all());
        assert!(matches!(
            plan.resolve(&index),
            Err(PowerError::EmptyChannel(name)) if name == "all"
        ));
    }
}
