//! Reference path: activity with `wellen`'s viewer API. Loads the selected signals into memory.
//! Used for non-FST files and as an independent reference in tests: it compares values as state
//! characters and does not share the packed kernel of the fast path.

use super::slots::{Placement, assemble_trace, new_channel_stats};
use super::stats::{chars_delta, chars_first};
use super::{ActivityTrace, Bins, PowerError, PowerPlan, check_time_table};
use crate::hierarchy::HierarchyIndex;
use std::path::Path;
use wellen::SignalValueRef;

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

    let identity;
    let bins = match given_bins {
        Some(bins) => {
            check_time_table(&body.time_table)?;
            bins
        }
        None => {
            identity = Bins::identity(&body.time_table)?;
            &identity
        }
    };
    plan.check_result_memory(bins.len())?;
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
        .handle_channels
        .iter()
        .enumerate()
        .filter(|(_, channels)| !channels.is_empty())
        .map(|(h, _)| wellen::SignalRef::from_index(h).expect("a valid signal index"))
        .collect();
    let mut source = body.source;
    let signals = source.load_signals(&refs, &hierarchy, true);

    let placement = Placement::new(&body.time_table, bins, None);
    let (mut previous, mut current) = (Vec::new(), Vec::new());
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
    Ok(assemble_trace(
        plan,
        bins,
        stats,
        timescale_exponent,
        resolved.info,
    ))
}
