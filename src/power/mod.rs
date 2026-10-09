//! Switching activity per time bin and channel. Definitions are in
//! `docs/superpowers/specs/2026-10-08-fast-power-design.md`.

use crate::hierarchy::{HierarchyIndex, Selection, SelectionError};
use itertools::Itertools;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::Path;

pub mod fst;
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
        // The exact capacity: a copy of the whole time table would be larger if times repeat.
        let mut starts = Vec::with_capacity(count_distinct(time_table));
        starts.extend(time_table.iter().copied().dedup());
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

/// The number of distinct values in a list that does not decrease.
pub(crate) fn count_distinct(sorted: &[u64]) -> usize {
    match sorted.len() {
        0 => 0,
        _ => 1 + sorted.windows(2).filter(|w| w[0] != w[1]).count(),
    }
}

/// The bins of a run, and the bytes that the run holds in memory from its start to its end (the
/// result, and the bin starts if the run makes them). `given` are bins that the caller gives. If
/// there are none, the bins are one per distinct time of the time table. The memory check comes
/// before the allocation of those bins.
pub(crate) fn run_bins<'a>(
    plan: &PowerPlan,
    time_table: &[u64],
    given: Option<&'a Bins>,
) -> Result<(Cow<'a, Bins>, u64), PowerError> {
    check_time_table(time_table)?;
    match given {
        Some(bins) => {
            let held = plan.check_run_memory(bins.len(), false)?;
            Ok((Cow::Borrowed(bins), held))
        }
        None => {
            let held = plan.check_run_memory(count_distinct(time_table), true)?;
            Ok((Cow::Owned(Bins::identity(time_table)?), held))
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
    /// Limit for the estimated memory of the result and of the decode buffers, in bytes.
    ///
    /// This is a check before the allocation, not a hard limit on the memory of the process.
    /// Before each allocation that grows with the number of time points, bins, or channels, the
    /// run adds up the estimated memory that it holds at that point. It fails with
    /// [`PowerError::Memory`] if the sum is more than the limit. The sum counts:
    ///
    /// - the result: the bin starts, and `bins + 2` slots per channel and statistic (the two
    ///   extra slots are for the changes before the first bin and after the end);
    /// - the starts of identity bins (a run with given bins does not make them);
    /// - the placement of the time points of one section: 4 bytes for each time point;
    /// - the decode buffers of one section: one set of slots per channel for each parallel part
    ///   (at most `2 * threads` parts) in the fast path;
    /// - the largest signal that `wellen` loads in the reference path. The estimate assumes one
    ///   change at each time point. The reference path loads the signals in batches. The
    ///   estimates of the signals of one batch fit in the memory that is left.
    ///
    /// The estimate does not count:
    ///
    /// - what the readers allocate before the run can check anything: the time table of the
    ///   file, the compressed data of a section, its decoded time table, the hierarchy, and the
    ///   `wellen` body;
    /// - in the fast path, the decompressed signal chains and frames of a section (the fork of
    ///   `fst-reader` does not tell their size before it reads them): a 1024-bit vector with
    ///   100000 changes in one section decompresses at least 12.8 MB;
    /// - in the reference path, several changes at one time stamp (delta cycles), so a signal
    ///   can need more than the estimate: a 1-bit signal with 4 changes at each of 1000000
    ///   time stamps needs at least 20 MB, and the estimate is 6 MB;
    /// - the slack of `Vec` growth, the internal storage and the decompression buffers of
    ///   `wellen`, the last values of the signals, and the fixed size of the structures.
    ///
    /// An estimate that overflows counts as too large.
    pub memory_limit: u64,
}

/// What a plan selects in one waveform file.
pub(crate) struct ResolvedPlan {
    /// For every handle index, the indices of the channels that select it.
    pub handle_channels: Vec<Vec<usize>>,
    /// The handle indices that at least one channel selects, in increasing order.
    pub selected: Vec<usize>,
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
        let info = RunInfo {
            selected_handles: selected.len(),
            top_scopes: top_scopes.into_iter().collect(),
            unmatched_rules,
        };
        Ok(ResolvedPlan {
            handle_channels,
            selected,
            info,
        })
    }

    /// Bytes needed for the result with `bins` bins: 8 bytes per bin for the time, plus
    /// `bins + 2` slots for every channel (the bins, and the slots before the first bin and after
    /// the end). A slot is 8 bytes for the toggles, or 32 bytes with full statistics. Returns
    /// `u64::MAX` if the number does not fit in 64 bits.
    pub fn result_bytes(&self, bins: usize) -> u64 {
        estimate_result_bytes(self.channels.len(), self.full_stats, bins).unwrap_or(u64::MAX)
    }

    /// Bytes needed for the decode buffers of a section that uses `section_bins` slots, with
    /// `threads` rayon threads. The fast path runs up to `2 * threads` parts at the same time.
    /// Each part has its own buffers for all channels. Returns `u64::MAX` if the number does not
    /// fit in 64 bits.
    pub fn section_bytes(&self, section_bins: usize, threads: usize) -> u64 {
        self.buffer_bytes(section_bins, threads.saturating_mul(2))
            .unwrap_or(u64::MAX)
    }

    fn buffer_bytes(&self, slots: usize, parts: usize) -> Option<u64> {
        estimate_buffer_bytes(self.channels.len(), self.full_stats, slots, parts)
    }

    /// Fails if the result for `bins` bins is larger than the memory limit.
    pub fn check_result_memory(&self, bins: usize) -> Result<(), PowerError> {
        self.check_run_memory(bins, false).map(drop)
    }

    /// Fails if the result for `bins` bins, and the bin starts if `own_starts`, are larger than
    /// the memory limit. Returns the bytes that they need.
    pub(crate) fn check_run_memory(
        &self,
        bins: usize,
        own_starts: bool,
    ) -> Result<u64, PowerError> {
        let what = format!(
            "the result{} ({bins} bins, {} channels)",
            if own_starts {
                " and the bin starts"
            } else {
                ""
            },
            self.channels.len()
        );
        let starts = if own_starts {
            estimate_time_bytes(bins)
        } else {
            Some(0)
        };
        let terms = [
            estimate_result_bytes(self.channels.len(), self.full_stats, bins),
            starts,
        ];
        self.check_sum(&terms, what)
    }

    /// Fails if the decode buffers of a section that uses `section_bins` slots (with the slots
    /// before the first bin and after the end) are larger than the memory limit. Counts the
    /// buffers of `2 * threads` parts, and nothing else.
    pub fn check_section_memory(
        &self,
        section_bins: usize,
        threads: usize,
    ) -> Result<(), PowerError> {
        let parts = threads.saturating_mul(2);
        let terms = [self.buffer_bytes(section_bins, parts)];
        self.check_sum(&terms, self.decode_what(section_bins, parts))
            .map(drop)
    }

    /// Fails if the placement of a section with `time_points` time points, together with the
    /// `held` bytes of the run, is larger than the memory limit. Call it before the placement.
    pub(crate) fn check_placement_memory(
        &self,
        held: u64,
        time_points: usize,
    ) -> Result<(), PowerError> {
        let terms = [Some(held), estimate_placement_bytes(time_points)];
        let what = format!("the placement of one section ({time_points} time points)");
        self.check_sum(&terms, what).map(drop)
    }

    /// Fails if the placement and the decode buffers of a section, together with the `held` bytes
    /// of the run, are larger than the memory limit. The buffers have `slots` slots for each of
    /// `parts` parts. Call it before the buffers.
    pub(crate) fn check_decode_memory(
        &self,
        held: u64,
        time_points: usize,
        slots: usize,
        parts: usize,
    ) -> Result<(), PowerError> {
        let terms = [
            Some(held),
            estimate_placement_bytes(time_points),
            self.buffer_bytes(slots, parts),
        ];
        self.check_sum(&terms, self.decode_what(slots, parts))
            .map(drop)
    }

    /// Fails if the placement of the time points and the largest signal that `wellen` loads
    /// (`largest` bytes, `None` if the estimate overflows), together with the `held` bytes of the
    /// run, are larger than the memory limit. Call it before the signals load. Returns the bytes
    /// that are left for the signals of one batch.
    pub(crate) fn check_load_memory(
        &self,
        held: u64,
        signals: usize,
        time_points: usize,
        largest: Option<u64>,
    ) -> Result<u64, PowerError> {
        let placement = estimate_placement_bytes(time_points);
        let terms = [Some(held), placement, largest];
        let what = format!(
            "the largest signal that wellen loads, with the result and the placement \
             ({signals} signals, {time_points} time points)"
        );
        self.check_sum(&terms, what)?;
        // The sum of the first two terms is at most the limit, because the whole sum is.
        Ok(self.memory_limit - held - placement.unwrap_or(0))
    }

    fn decode_what(&self, slots: usize, parts: usize) -> String {
        format!(
            "the decode buffers of one section ({slots} slots, {} channels, {parts} parallel parts)",
            self.channels.len(),
        )
    }

    /// Fails if the sum of `terms` is more than the memory limit. A `None` term or a sum that
    /// overflows is too large. Returns the sum.
    fn check_sum(&self, terms: &[Option<u64>], what: String) -> Result<u64, PowerError> {
        let sum = terms
            .iter()
            .try_fold(0u64, |sum, term| sum.checked_add((*term)?));
        match sum {
            Some(needed) if needed <= self.memory_limit => Ok(needed),
            _ => Err(PowerError::Memory {
                what,
                needed: sum.unwrap_or(u64::MAX),
                limit: self.memory_limit,
            }),
        }
    }
}

/// Bytes of a list of `count` times (bin starts): 8 each. `None` if it overflows.
fn estimate_time_bytes(count: usize) -> Option<u64> {
    u64::try_from(count).ok()?.checked_mul(8)
}

/// Bytes that the placement of a section needs: one `u32` for each time point.
fn estimate_placement_bytes(time_points: usize) -> Option<u64> {
    u64::try_from(time_points).ok()?.checked_mul(4)
}

/// Bytes of one slot of one channel: 8 for the toggles, or 32 with full statistics.
fn slot_bytes(full_stats: bool) -> u64 {
    if full_stats { 32 } else { 8 }
}

/// Bytes of the result: the bin starts, and `bins + 2` slots for each of `channels` channels.
/// `None` if it overflows.
fn estimate_result_bytes(channels: usize, full_stats: bool, bins: usize) -> Option<u64> {
    let slots = u64::try_from(bins).ok()?.checked_add(2)?;
    let stats = u64::try_from(channels)
        .ok()?
        .checked_mul(slot_bytes(full_stats))?
        .checked_mul(slots)?;
    estimate_time_bytes(bins)?.checked_add(stats)
}

/// Bytes of the decode buffers: `slots` slots for each of `channels` channels, for each of `parts`
/// parts. `None` if it overflows.
fn estimate_buffer_bytes(
    channels: usize,
    full_stats: bool,
    slots: usize,
    parts: usize,
) -> Option<u64> {
    u64::try_from(parts)
        .ok()?
        .checked_mul(u64::try_from(channels).ok()?)?
        .checked_mul(slot_bytes(full_stats))?
        .checked_mul(u64::try_from(slots).ok()?)
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
/// FST files use the fast path; other formats (VCD, GHW) use the `wellen` reference path.
pub fn activity(path: &Path, plan: &PowerPlan) -> Result<ActivityTrace, PowerError> {
    if is_fst(path)? {
        fst::activity_fst(path, plan)
    } else {
        reference::activity_reference(path, plan)
    }
}

/// Computes the activity of a waveform file in the given bins.
pub fn activity_binned(
    path: &Path,
    plan: &PowerPlan,
    bins: &Bins,
) -> Result<ActivityTrace, PowerError> {
    if is_fst(path)? {
        fst::activity_fst_binned(path, plan, bins)
    } else {
        reference::activity_reference_binned(path, plan, bins)
    }
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
    if is_fst(path)? {
        let mut reader = fst::open_reader(path)?;
        Ok(HierarchyIndex::from_fst(&mut reader)?)
    } else {
        let header = wellen::viewers::read_header_from_file(path, &reference::load_options())?;
        Ok(HierarchyIndex::from_wellen(&header.hierarchy))
    }
}

fn is_fst(path: &Path) -> Result<bool, PowerError> {
    let mut file = std::fs::File::open(path)?;
    Ok(fst_reader::is_fst_file(&mut file))
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
        // 8 bytes for the time of each bin, and 8 bytes for each of the 1000 + 2 slots of the one
        // channel (the bins, the slot before the first bin, and the slot after the end).
        assert_eq!(plan.result_bytes(1000), 1000 * 8 + 1002 * 8);
        // 2 * 10 parts, 1 channel, 50 slots, 8 bytes.
        assert_eq!(plan.section_bytes(50, 10), 20 * 50 * 8);
        plan.full_stats = true;
        plan.channels.push(plan.channels[0].clone());
        assert_eq!(plan.result_bytes(1000), 1000 * 8 + 2 * 1002 * 32);
        assert_eq!(plan.section_bytes(50, 10), 20 * 2 * 50 * 32);
    }

    #[test]
    fn the_run_adds_up_what_it_holds() {
        let mut plan = PowerPlan::toggles(Selection::all());
        plan.memory_limit = 10_000;
        // The result of 100 bins: 800 + 102 * 8 = 1616 bytes. Identity bins add 800 bytes.
        assert_eq!(plan.check_run_memory(100, false).unwrap(), 1616);
        assert_eq!(plan.check_run_memory(100, true).unwrap(), 2416);
        // The placement needs 4 bytes for each time point: 1616 + 4 * 2000 = 9616 bytes.
        assert!(plan.check_placement_memory(1616, 2000).is_ok());
        let err = plan.check_placement_memory(1616, 2200).unwrap_err();
        assert!(
            matches!(&err, PowerError::Memory { what, needed: 10_416, limit: 10_000 } if what.starts_with("the placement")),
            "{err}"
        );
        // Placement, and 3 parts with 10 slots of 8 bytes: 1616 + 400 + 240 = 2256 bytes.
        assert!(plan.check_decode_memory(1616, 100, 10, 3).is_ok());
        plan.memory_limit = 2255;
        let err = plan.check_decode_memory(1616, 100, 10, 3).unwrap_err();
        assert!(
            matches!(&err, PowerError::Memory { what, needed: 2256, .. } if what.starts_with("the decode buffers")),
            "{err}"
        );
        // The same buffers alone are 240 bytes.
        assert!(plan.check_section_memory(10, 1).is_ok());
        // The reference path: the placement (400 bytes) and the largest signal (5000 bytes) add up
        // to 7016 bytes with the result. What is left is the budget of one batch.
        plan.memory_limit = 7016;
        assert_eq!(
            plan.check_load_memory(1616, 5, 100, Some(5000)).unwrap(),
            5000
        );
        plan.memory_limit = 9000;
        assert_eq!(
            plan.check_load_memory(1616, 5, 100, Some(5000)).unwrap(),
            6984
        );
        plan.memory_limit = 7015;
        let err = plan
            .check_load_memory(1616, 5, 100, Some(5000))
            .unwrap_err();
        assert!(
            matches!(&err, PowerError::Memory { what, needed: 7016, .. } if what.starts_with("the largest signal that wellen loads")),
            "{err}"
        );
        assert!(plan.check_load_memory(1616, 5, 100, None).is_err());
    }

    #[test]
    fn memory_errors_name_what_was_too_large() {
        let mut plan = PowerPlan::toggles(Selection::all());
        plan.memory_limit = 1000;
        // 16 bytes for each bin, and 16 for the two extra slots.
        assert!(plan.check_result_memory(61).is_ok());
        let err = plan.check_result_memory(62).unwrap_err();
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
    fn identity_bins_have_the_exact_capacity() {
        let bins = Bins::identity(&[0, 10, 10, 20, 20, 20, 30]).unwrap();
        assert_eq!(bins.starts.capacity(), 4);
        assert_eq!(count_distinct(&[]), 0);
        assert_eq!(count_distinct(&[5]), 1);
        assert_eq!(count_distinct(&[5, 5, 5]), 1);
        assert_eq!(count_distinct(&[5, 6, 6, 9]), 3);
    }

    #[test]
    fn the_estimate_functions_report_an_overflow_as_none() {
        assert_eq!(estimate_result_bytes(2, true, 10), Some(80 + 2 * 12 * 32));
        assert_eq!(estimate_result_bytes(usize::MAX / 2, true, 1000), None);
        assert_eq!(estimate_result_bytes(1, false, usize::MAX), None);
        assert_eq!(estimate_result_bytes(1, false, usize::MAX - 1), None);
        assert_eq!(estimate_result_bytes(usize::MAX / 2, false, 0), None);
        assert_eq!(estimate_buffer_bytes(3, false, 5, 2), Some(3 * 5 * 2 * 8));
        assert_eq!(estimate_buffer_bytes(usize::MAX / 2, true, 10, 4), None);
        assert_eq!(estimate_buffer_bytes(2, true, usize::MAX, 2), None);
        assert_eq!(estimate_buffer_bytes(2, true, 2, usize::MAX), None);
        assert_eq!(estimate_placement_bytes(usize::MAX), None);
        assert_eq!(estimate_time_bytes(usize::MAX / 4), None);
    }

    #[test]
    fn estimates_that_overflow_count_as_too_large() {
        // Even the largest limit does not accept an estimate that does not fit in 64 bits.
        let mut plan = PowerPlan::toggles(Selection::all());
        plan.memory_limit = u64::MAX;
        for result in [
            plan.check_section_memory(10, usize::MAX),
            plan.check_section_memory(usize::MAX, 1),
            plan.check_result_memory(usize::MAX),
        ] {
            assert!(
                matches!(
                    &result,
                    Err(PowerError::Memory {
                        needed: u64::MAX,
                        ..
                    })
                ),
                "{result:?}"
            );
        }
        assert_eq!(plan.result_bytes(usize::MAX), u64::MAX);
        assert_eq!(plan.section_bytes(usize::MAX, usize::MAX), u64::MAX);
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
