//! Reference path: activity with `wellen`'s viewer API. Loads the selected signals into memory.
//! Used for non-FST files and as an independent reference in tests: it compares values as state
//! characters and does not share the packed kernel of the fast path.
//!
//! Memory: `wellen` reads the whole body (and the time table) before the run knows anything
//! about the selection, so the memory limit does not count it. The run checks the memory limit
//! before it makes identity bins and before it loads the selected signals.
//!
//! `wellen` does not tell the size of a signal before it loads it. The run uses an estimate (see
//! [`signal_bytes`]) that is far above the real size when a signal changes rarely, and below it
//! when a signal has several changes at one time stamp. A check of the sum of all estimates would
//! refuse files that load without problems. Therefore the run loads the signals in batches. The
//! estimates of the signals of a batch add up to at most the memory that is left. The run fails
//! only if the estimate of one signal alone does not fit. The batches do not change the result:
//! the statistics of the signals are sums. [`PowerPlan::memory_limit`] lists what the estimate
//! does not count.

use super::slots::{Placement, assemble_trace, new_channel_stats};
use super::stats::{chars_delta, chars_first};
use super::{ActivityTrace, Bins, PowerError, PowerPlan, run_bins};
use crate::hierarchy::HierarchyIndex;
use std::path::Path;
use wellen::{SignalEncoding, SignalValueRef};

/// The options to read any waveform file with `wellen`.
pub(crate) fn load_options() -> wellen::LoadOptions {
    wellen::LoadOptions {
        multi_thread: true,
        remove_scopes_with_empty_name: false,
    }
}

/// The timescale as a power of ten of seconds, or `None` if the factor is not a power of ten or
/// the unit is unknown.
fn timescale_exponent(timescale: wellen::Timescale) -> Option<i8> {
    let unit = timescale.unit.to_exponent()?;
    let factor_exponent = timescale.factor.checked_ilog10()?;
    (10u32.pow(factor_exponent) == timescale.factor).then(|| unit + factor_exponent as i8)
}

/// An estimate of the bytes that `wellen` needs to hold the changes of the signal with the
/// handle index `handle`, for a time table of `time_points` entries. `None` if the estimate
/// overflows.
///
/// The estimate assumes that the signal changes once at every time point. A change needs 4
/// bytes for its time index, and its value:
///
/// - a bit vector of `w` bits: `ceil(w / 2)` bytes (4 bits per bit, nine states), and 1 meta byte;
/// - a real value: 8 bytes;
/// - a string, or an unknown type: 64 bytes (the `String` header and a short text).
///
/// The fixed size of a loaded signal is not counted.
fn signal_bytes(hierarchy: &wellen::Hierarchy, handle: usize, time_points: usize) -> Option<u64> {
    let value_bytes = match hierarchy.get_signal_tpe(wellen::SignalRef::from_index(handle)?) {
        Some(SignalEncoding::BitVector(width)) => u64::from(width).div_ceil(2) + 1,
        Some(SignalEncoding::Real) => 8,
        Some(SignalEncoding::String) | None => 64,
    };
    u64::try_from(time_points)
        .ok()?
        .checked_mul(4 + value_bytes)
}

/// Splits the signals into batches of consecutive signals. The `costs` of the signals of a batch
/// add up to at most `budget`. Every cost must be at most `budget`. A batch is never empty.
fn batches(costs: &[u64], budget: u64) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let (mut start, mut sum) = (0, 0u64);
    for (i, &cost) in costs.iter().enumerate() {
        if sum.checked_add(cost).is_none_or(|total| total > budget) && i > start {
            batches.push(start..i);
            (start, sum) = (i, 0);
        }
        sum += cost;
    }
    if start < costs.len() {
        batches.push(start..costs.len());
    }
    batches
}

/// Computes the activity of a waveform file with one bin per distinct time of its time table.
pub fn activity_reference(path: &Path, plan: &PowerPlan) -> Result<ActivityTrace, PowerError> {
    run(path, plan, None)
}

/// Computes the activity of a waveform file in the given bins.
pub fn activity_reference_binned(
    path: &Path,
    plan: &PowerPlan,
    bins: &Bins,
) -> Result<ActivityTrace, PowerError> {
    run(path, plan, Some(bins))
}

fn run(
    path: &Path,
    plan: &PowerPlan,
    given_bins: Option<&Bins>,
) -> Result<ActivityTrace, PowerError> {
    run_counting_batches(path, plan, given_bins).map(|(trace, _)| trace)
}

/// Like [`run`]. Also returns the number of calls to `load_signals` (the batches).
fn run_counting_batches(
    path: &Path,
    plan: &PowerPlan,
    given_bins: Option<&Bins>,
) -> Result<(ActivityTrace, usize), PowerError> {
    let header = wellen::viewers::read_header_from_file(path, &load_options())?;
    let hierarchy = header.hierarchy;
    let index = HierarchyIndex::from_wellen(&hierarchy);
    let resolved = plan.resolve(&index)?;
    let timescale_exponent = hierarchy.timescale().and_then(timescale_exponent);
    let body = wellen::viewers::read_body(header.body, &hierarchy, None)?;

    // Checks the memory of the result before it allocates identity bins.
    let (bins, held) = run_bins(plan, &body.time_table, given_bins)?;
    let bins: &Bins = &bins;
    let mut stats = new_channel_stats(plan, bins);
    if body.time_table.is_empty() {
        // `wellen` panics when it loads signals of an FST file without time points.
        let trace = assemble_trace(plan, bins, stats, timescale_exponent, resolved.info);
        return Ok((trace, 0));
    }

    // Load the selected signals. A derived signal never has channels, because the index maps
    // its variables to the underlying signals.
    let refs: Vec<_> = resolved
        .selected
        .iter()
        .map(|&h| wellen::SignalRef::from_index(h).expect("a valid signal index"))
        .collect();
    let time_points = body.time_table.len();
    let costs: Vec<Option<u64>> = resolved
        .selected
        .iter()
        .map(|&h| signal_bytes(&hierarchy, h, time_points))
        .collect();
    // The run needs at least the placement and the largest signal. The check returns the bytes
    // that are left for the signals of one batch.
    let largest = costs
        .iter()
        .try_fold(0u64, |largest, cost| cost.map(|cost| largest.max(cost)));
    let budget = plan.check_load_memory(held, refs.len(), time_points, largest)?;
    let costs: Vec<u64> = costs.into_iter().flatten().collect();

    let mut source = body.source;
    let placement = Placement::new(&body.time_table, bins, None);
    let (mut previous, mut current) = (Vec::new(), Vec::new());
    let mut batch_count = 0;
    for batch in batches(&costs, budget) {
        batch_count += 1;
        let signals = source.load_signals(&refs[batch], &hierarchy, true);
        for signal in &signals {
            let channels = &resolved.handle_channels[signal.signal_ref().index()];
            let mut has_previous = false;
            for (time_index, value) in signal.iter_changes() {
                let SignalValueRef::BitVec(bits) = value else {
                    continue;
                };
                current.clear();
                current.extend(bits.iter_msb_to_lsb().map(|b| b.as_ascii() as u8));
                let d = if has_previous {
                    chars_delta(&previous, &current, plan.unknown)
                } else {
                    chars_first(&current, plan.unknown)
                };
                std::mem::swap(&mut previous, &mut current);
                has_previous = true;
                let Some(slot) = placement.global_slot(time_index as usize) else {
                    continue;
                };
                for &c in channels {
                    if plan.full_stats {
                        stats[c].add::<true>(slot, &d);
                    } else {
                        stats[c].add::<false>(slot, &d);
                    }
                }
            }
        }
    }
    let trace = assemble_trace(plan, bins, stats, timescale_exponent, resolved.info);
    Ok((trace, batch_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::Selection;
    use std::io::Write;

    /// A VCD file with three signals of 1, 4, and 70 bits that change at 5 time points.
    fn write_vcd(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("t.vcd");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            "$timescale 1ps $end\n$scope module tb $end\n$var wire 1 ! a $end\n\
             $var wire 4 \" b $end\n$var wire 70 # c $end\n$upscope $end\n$enddefinitions $end\n\
             #0\n$dumpvars\n0!\nb0000 \"\nb{} #\n$end\n",
            "0".repeat(70)
        )
        .unwrap();
        for t in 1..=5u32 {
            writeln!(f, "#{}\n{}!\nb{:04b} \"\nb{:070b} #", t * 10, t % 2, t, t).unwrap();
        }
        path
    }

    /// The signals load in as few batches as the limit allows. The result does not depend on the
    /// number of batches.
    #[test]
    fn the_limit_decides_the_number_of_batches() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_vcd(dir.path());
        let mut plan = PowerPlan::toggles(Selection::all());
        plan.full_stats = true;
        // 6 time points. The estimates are 36, 42, and 240 bytes, 318 bytes together. With the
        // identity bins, the result needs 352 bytes and the placement 24 bytes.
        let (one, count) = run_counting_batches(&path, &plan, None).unwrap();
        assert_eq!(count, 1);
        plan.memory_limit = 352 + 24 + 318;
        let (same, count) = run_counting_batches(&path, &plan, None).unwrap();
        assert_eq!((count, &same), (1, &one));
        plan.memory_limit = 352 + 24 + 240;
        let (split, count) = run_counting_batches(&path, &plan, None).unwrap();
        assert_eq!((count, &split), (2, &one));
    }

    #[test]
    fn batches_group_consecutive_signals_up_to_the_budget() {
        assert_eq!(batches(&[], 10), Vec::<std::ops::Range<usize>>::new());
        assert_eq!(batches(&[3, 3, 3, 3], 10), vec![0..3, 3..4]);
        assert_eq!(batches(&[3, 3, 3, 3], 6), vec![0..2, 2..4]);
        // A signal that fills the budget is a batch of its own.
        assert_eq!(batches(&[4, 10, 4, 4], 10), vec![0..1, 1..2, 2..4]);
        assert_eq!(batches(&[10, 10], 10), vec![0..1, 1..2]);
        assert_eq!(batches(&[0, 0, 0], 0), vec![0..3]);
        // Large numbers do not overflow.
        assert_eq!(batches(&[u64::MAX, u64::MAX], u64::MAX), vec![0..1, 1..2]);
        let costs = [5, 1, 9, 2, 2, 7, 7, 1];
        for budget in 9..30 {
            let groups = batches(&costs, budget);
            assert_eq!(groups.first().map(|g| g.start), Some(0));
            assert_eq!(groups.last().map(|g| g.end), Some(costs.len()));
            for pair in groups.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
            for group in &groups {
                assert!(!group.is_empty());
                assert!(costs[group.clone()].iter().sum::<u64>() <= budget);
            }
        }
    }
}
