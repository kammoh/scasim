//! Fast path for FST files. Sections are processed in order. Inside a section, the selected
//! signals are decoded in parallel. Each signal starts from its last value in the previous
//! section, so the result equals a sequential pass over all changes. Unselected signals are never
//! decompressed.
//!
//! The path reports the same changes as [`fst_reader::FstReader::read_signals`]. This includes the
//! frame values of the first section. It excludes every change after the end time in the file
//! header.

use super::slots::{Placement, SlotStats, assemble_trace, merge_channel_stats, new_channel_stats};
use super::stats::{
    chars_delta, chars_first, packed_delta, packed_first, packed_to_chars, packed_toggles,
};
use super::{ActivityTrace, Bins, PowerError, PowerPlan, Totals, UnknownPolicy, check_time_table};
use crate::hierarchy::HierarchyIndex;
use fst_reader::{FstReader, FstSection, FstSignalHandle, FstValue};
use rayon::prelude::*;
use std::path::Path;

/// The last value of a signal, in the representation it last had.
#[derive(Debug, Clone, Default)]
enum Last {
    /// No value yet: the next value is the signal's first value.
    #[default]
    None,
    Packed(Vec<u8>),
    Chars(Vec<u8>),
}

/// Stores `value` as the signal's last value and returns the statistics of the change. With
/// `FULL == false`, only `toggles` is computed. Real and string values contribute nothing.
fn step<const FULL: bool>(
    last: &mut Last,
    value: FstValue<'_>,
    unknown: UnknownPolicy,
    scratch: &mut Vec<u8>,
) -> Totals {
    let chars = |old: &[u8], new: &[u8]| {
        let d = chars_delta(old, new, unknown);
        if FULL {
            d
        } else {
            Totals {
                toggles: d.toggles,
                ..Totals::default()
            }
        }
    };
    match value {
        FstValue::Packed { width, bytes } => match last {
            Last::Packed(prev) if prev.len() == bytes.len() => {
                let d = if FULL {
                    packed_delta(prev, bytes)
                } else {
                    Totals {
                        toggles: packed_toggles(prev, bytes),
                        ..Totals::default()
                    }
                };
                prev.copy_from_slice(bytes);
                d
            }
            Last::Chars(prev) => {
                packed_to_chars(width, bytes, scratch);
                let d = chars(prev, scratch);
                *last = Last::Packed(bytes.to_vec());
                d
            }
            _ => {
                *last = Last::Packed(bytes.to_vec());
                if FULL {
                    packed_first(bytes)
                } else {
                    Totals::default()
                }
            }
        },
        FstValue::Chars(new) => {
            let d = match last {
                Last::None if FULL => chars_first(new, unknown),
                Last::None => Totals::default(),
                Last::Chars(prev) => chars(prev, new),
                Last::Packed(prev) => {
                    packed_to_chars(new.len() as u32, prev, scratch);
                    chars(scratch, new)
                }
            };
            match last {
                Last::Chars(prev) if prev.len() == new.len() => prev.copy_from_slice(new),
                _ => *last = Last::Chars(new.to_vec()),
            }
            d
        }
        FstValue::VarLen(_) | FstValue::Real(_) => Totals::default(),
    }
}

/// Opens an FST file and reads its time table. Like `wellen`, falls back to an external
/// `.fst.hier` file when the hierarchy or geometry block is missing (an interrupted simulation).
pub(crate) fn open_reader(
    path: &Path,
) -> Result<FstReader<std::io::BufReader<std::fs::File>>, PowerError> {
    let open = || std::fs::File::open(path).map(std::io::BufReader::new);
    match FstReader::open_and_read_time_table(open()?) {
        Ok(reader) => Ok(reader),
        Err(
            fst_reader::ReaderError::MissingGeometry()
            | fst_reader::ReaderError::MissingHierarchy(),
        ) => {
            let mut hier = path.to_path_buf();
            hier.set_extension("fst.hier");
            let hier = std::io::BufReader::new(std::fs::File::open(hier)?);
            Ok(FstReader::open_incomplete_and_read_time_table(
                open()?,
                hier,
            )?)
        }
        Err(e) => Err(e.into()),
    }
}

/// Computes the activity of an FST file with one bin per distinct time of its time table.
pub fn activity_fst(path: &Path, plan: &PowerPlan) -> Result<ActivityTrace, PowerError> {
    if plan.full_stats {
        run::<true>(path, plan, None)
    } else {
        run::<false>(path, plan, None)
    }
}

/// Computes the activity of an FST file in the given bins.
pub fn activity_fst_binned(
    path: &Path,
    plan: &PowerPlan,
    bins: &Bins,
) -> Result<ActivityTrace, PowerError> {
    if plan.full_stats {
        run::<true>(path, plan, Some(bins))
    } else {
        run::<false>(path, plan, Some(bins))
    }
}

/// Adds the frame values of a section to slot `slot`. The frame values are the first values of
/// the signals: only their Hamming weight counts.
fn add_frame_values<const FULL: bool>(
    section: &FstSection,
    slot: usize,
    plan: &PowerPlan,
    handle_channels: &[Vec<usize>],
    last: &mut [Last],
    stats: &mut [SlotStats],
) -> Result<(), fst_reader::ReaderError> {
    let mut scratch = Vec::new();
    section.for_each_frame_value(|handle, value| {
        let h = handle.get_index();
        let Some(channels) = handle_channels.get(h).filter(|c| !c.is_empty()) else {
            return;
        };
        let d = step::<FULL>(&mut last[h], value, plan.unknown, &mut scratch);
        for &c in channels {
            stats[c].add::<FULL>(slot, &d);
        }
    })
}

/// Decodes the changes of all selected signals of one section, in parallel, and returns the
/// statistics of every channel for the slots of `placement`. `last` holds the last value of every
/// signal before the section, and holds it after the section.
fn decode_section<const FULL: bool>(
    section: &FstSection,
    placement: &Placement,
    plan: &PowerPlan,
    handle_channels: &[Vec<usize>],
    last: &mut [Last],
) -> Result<Vec<SlotStats>, fst_reader::ReaderError> {
    // About two parts per thread, so that rayon can balance the load.
    let part_len = last.len().div_ceil(2 * rayon::current_num_threads()).max(1);
    last.par_chunks_mut(part_len)
        .enumerate()
        .map(|(part, part_last)| {
            let first_handle = part * part_len;
            // The buffers of this part. They stay empty until the part has a selected signal.
            let mut local: Vec<SlotStats> = Vec::new();
            let mut scratch = Vec::new();
            for (offset, last) in part_last.iter_mut().enumerate() {
                let h = first_handle + offset;
                let Some(channels) = handle_channels.get(h).filter(|c| !c.is_empty()) else {
                    continue;
                };
                if local.is_empty() {
                    local = plan
                        .channels
                        .iter()
                        .map(|_| SlotStats::new(placement.slot_count, FULL))
                        .collect();
                }
                section.for_each_change(FstSignalHandle::from_index(h), |time_index, value| {
                    let Some(slot) = placement.local_slot(time_index) else {
                        return;
                    };
                    let d = step::<FULL>(last, value, plan.unknown, &mut scratch);
                    for &c in channels {
                        local[c].add::<FULL>(slot, &d);
                    }
                })?;
            }
            Ok(local)
        })
        .try_reduce(Vec::new, |a, b| Ok(merge_channel_stats(a, b)))
}

/// The activity computation. `FULL` selects the kernel: toggles only, or all statistics.
fn run<const FULL: bool>(
    path: &Path,
    plan: &PowerPlan,
    given_bins: Option<&Bins>,
) -> Result<ActivityTrace, PowerError> {
    let mut reader = open_reader(path)?;
    let header = reader.get_header();
    let (end_time, max_handle) = (header.end_time, header.max_handle as usize);
    let timescale_exponent = Some(header.timescale_exponent);

    let time_table = reader.get_time_table().unwrap_or_default();
    let identity;
    let bins = match given_bins {
        Some(bins) => {
            check_time_table(time_table)?;
            bins
        }
        None => {
            identity = Bins::identity(time_table)?;
            &identity
        }
    };
    let has_time_points = !time_table.is_empty();
    plan.check_result_memory(bins.len())?;

    let index = HierarchyIndex::from_fst(&mut reader)?;
    let resolved = plan.resolve(&index)?;
    let handle_channels = &resolved.handle_channels;
    // The statistics of all channels. The slots are: before the first bin, the bins, and after
    // the end.
    let mut stats = new_channel_stats(plan, bins);
    if has_time_points {
        // The last value of every signal. Only selected signals use theirs.
        let mut last = vec![Last::None; handle_channels.len().max(max_handle)];
        let mut first_section = true;
        for (section_index, info) in reader.sections().iter().enumerate() {
            // `read_signals` skips the sections that start after the header end time.
            if info.start_time > end_time {
                continue;
            }
            let section = reader.read_section(section_index)?;
            let times = section.time_table();

            // `read_signals` reports the frame of the first section at the section start time
            // when the first time point is later.
            let reports_frame = std::mem::take(&mut first_section)
                && times.first().is_none_or(|&t| t > info.start_time);
            if reports_frame {
                let slot = bins.slot_of(info.start_time);
                add_frame_values::<FULL>(
                    &section,
                    slot,
                    plan,
                    handle_channels,
                    &mut last,
                    &mut stats,
                )?;
            }

            let placement = Placement::new(times, bins, Some(end_time));
            if placement.slot_count == 0 {
                continue;
            }
            plan.check_section_memory(placement.slot_count, rayon::current_num_threads())?;
            let decoded =
                decode_section::<FULL>(&section, &placement, plan, handle_channels, &mut last)?;
            for (total, section_stats) in stats.iter_mut().zip(&decoded) {
                total.add_window(placement.first_slot, section_stats);
            }
        }
    }
    // A file without time points has no activity. Same as the reference path.
    Ok(assemble_trace(
        plan,
        bins,
        stats,
        timescale_exponent,
        resolved.info,
    ))
}
