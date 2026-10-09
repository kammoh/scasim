//! Reference path: activity with `wellen`'s viewer API. Loads the selected signals into memory.
//! Used for non-FST files and as an independent reference in tests: it compares values as state
//! characters and does not share the packed kernel of the fast path.
//!
//! Memory: `wellen` reads the whole body (and the time table) before the run knows anything
//! about the selection, so the memory limit does not count it. The run checks the memory limit
//! before it makes identity bins and before it loads the selected signals.
//!
//! `wellen` does not tell the size of a signal before it loads it. The run uses an upper bound
//! (see [`signal_bytes`]) that is far above the real size when a signal changes rarely. A check
//! of the sum of all bounds would refuse files that load without problems. Therefore the run
//! loads the signals in batches. The bounds of the signals of a batch add up to at most the
//! memory that is left. The run fails only if the bound of one signal alone does not fit. The
//! batches do not change the result: the statistics of the signals are sums.

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

/// An upper bound for the bytes that `wellen` needs to hold the changes of the signal with the
/// handle index `handle`, for a time table of `time_points` entries. `None` if the bound
/// overflows.
///
/// The bound assumes the worst case: the signal changes at every time point. A change needs 4
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
        return Ok(assemble_trace(
            plan,
            bins,
            stats,
            timescale_exponent,
            resolved.info,
        ));
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
    for batch in batches(&costs, budget) {
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
    Ok(assemble_trace(
        plan,
        bins,
        stats,
        timescale_exponent,
        resolved.info,
    ))
}

#[cfg(test)]
mod tests {
    use super::batches;

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
