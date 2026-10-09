//! Fast path for FST files. Sections are processed in order. Inside a section, the selected
//! signals are decoded in parallel. Each signal starts from its last value in the previous
//! section, so the result equals a sequential pass over all changes. Unselected signals are never
//! decompressed.
//!
//! The path reports the same changes as [`fst_reader::FstReader::read_signals`]. This includes the
//! frame values of the first section. It excludes every change after the end time in the file
//! header.
//!
//! Memory: the run checks the memory limit before every allocation that grows with the number of
//! time points, bins, or channels (see [`PowerPlan::memory_limit`]). The reader allocates some
//! memory before any check can run: the time table of the file, and, for each section, its
//! compressed data and its decoded time table. The fork of `fst-reader` does not tell their sizes
//! before it reads them, so they are not counted.

use super::slots::{Placement, SlotStats, assemble_trace, merge_channel_stats, new_channel_stats};
use super::stats::{
    chars_delta, chars_first, packed_delta, packed_first, packed_to_chars, packed_toggles,
};
use super::{ActivityTrace, Bins, PowerError, PowerPlan, Totals, UnknownPolicy, run_bins};
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
/// the signals: only their Hamming weight counts. `last[i]` is the last value of the signal
/// `selected[i]`.
fn add_frame_values<const FULL: bool>(
    section: &FstSection,
    slot: usize,
    plan: &PowerPlan,
    handle_channels: &[Vec<usize>],
    selected: &[usize],
    last: &mut [Last],
    stats: &mut [SlotStats],
) -> Result<(), fst_reader::ReaderError> {
    let mut scratch = Vec::new();
    section.for_each_frame_value(|handle, value| {
        let h = handle.get_index();
        // `selected` is in increasing order.
        let Ok(position) = selected.binary_search(&h) else {
            return;
        };
        let d = step::<FULL>(&mut last[position], value, plan.unknown, &mut scratch);
        for &c in &handle_channels[h] {
            stats[c].add::<FULL>(slot, &d);
        }
    })
}

/// The number of parts of the parallel decode of `selected` signals on `threads` threads: about
/// two per thread, so that rayon can balance the load, and never more than signals.
fn part_count(selected: usize, threads: usize) -> usize {
    selected.min(threads.saturating_mul(2).max(1))
}

/// One part of the parallel decode: selected handles, and the last value of each of them.
type Chunk<'a> = (&'a [usize], &'a mut [Last]);

/// Splits the selected handles and their last values into `parts` chunks (fewer if there are
/// fewer handles). The chunks have nearly equal lengths and keep the order of the handles. Every
/// chunk has at least one handle, and no two chunks share a last value, so that each part can
/// change the values of its handles without a lock.
///
/// The chunks cover the list of the selected handles, not the range of all handle numbers. A
/// selection of neighboring handles is split over all parts.
fn balanced_chunks<'a>(
    handles: &'a [usize],
    states: &'a mut [Last],
    parts: usize,
) -> Vec<Chunk<'a>> {
    assert_eq!(handles.len(), states.len());
    let len = handles.len();
    if len == 0 {
        return Vec::new();
    }
    let parts = parts.clamp(1, len);
    let (base, extra) = (len / parts, len % parts);
    let (mut handles, mut states) = (handles, states);
    let mut chunks = Vec::with_capacity(parts);
    for part in 0..parts {
        let chunk_len = base + usize::from(part < extra);
        let (chunk_handles, rest_handles) = handles.split_at(chunk_len);
        let (chunk_states, rest_states) = std::mem::take(&mut states).split_at_mut(chunk_len);
        chunks.push((chunk_handles, chunk_states));
        handles = rest_handles;
        states = rest_states;
    }
    chunks
}

/// Decodes the changes of all selected signals of one section, in parallel (one task for each
/// chunk), and returns the statistics of every channel for the slots of `placement`. The last
/// value of every signal before the section is in its chunk, and the chunk holds it again after
/// the section. The decoder stops at the first change after the time index `last_time_index`,
/// without parsing its value.
fn decode_section<const FULL: bool>(
    section: &FstSection,
    last_time_index: usize,
    placement: &Placement,
    plan: &PowerPlan,
    handle_channels: &[Vec<usize>],
    chunks: Vec<Chunk<'_>>,
) -> Result<Vec<SlotStats>, fst_reader::ReaderError> {
    chunks
        .into_par_iter()
        .map(|(handles, states)| {
            // The buffers of this chunk. `PowerPlan::check_decode_memory` counted them.
            let mut local: Vec<SlotStats> = plan
                .channels
                .iter()
                .map(|_| SlotStats::new(placement.slot_count, FULL))
                .collect();
            let mut scratch = Vec::new();
            for (&h, last) in handles.iter().zip(states.iter_mut()) {
                let channels = &handle_channels[h];
                section.for_each_change_until(
                    FstSignalHandle::from_index(h),
                    last_time_index,
                    |time_index, value| {
                        let Some(slot) = placement.local_slot(time_index) else {
                            return;
                        };
                        let d = step::<FULL>(last, value, plan.unknown, &mut scratch);
                        for &c in channels {
                            local[c].add::<FULL>(slot, &d);
                        }
                    },
                )?;
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
    let end_time = header.end_time;
    let timescale_exponent = Some(header.timescale_exponent);

    let time_table = reader.get_time_table().unwrap_or_default();
    // Checks the memory of the result before it allocates identity bins.
    let (bins, held) = run_bins(plan, time_table, given_bins)?;
    let bins: &Bins = &bins;
    let has_time_points = !time_table.is_empty();

    let index = HierarchyIndex::from_fst(&mut reader)?;
    let resolved = plan.resolve(&index)?;
    let handle_channels = &resolved.handle_channels;
    let selected = &resolved.selected;
    // The statistics of all channels. The slots are: before the first bin, the bins, and after
    // the end.
    let mut stats = new_channel_stats(plan, bins);
    if has_time_points {
        // The last value of every selected signal: `last[i]` belongs to `selected[i]`.
        let mut last = vec![Last::None; selected.len()];
        let parts = part_count(selected.len(), rayon::current_num_threads());
        let mut first_section = true;
        for (section_index, info) in reader.sections().iter().enumerate() {
            // `read_signals` skips the sections that start after the header end time.
            if info.start_time > end_time {
                continue;
            }
            // The compressed data and the time table of the section are not counted: see the
            // module documentation.
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
                    selected,
                    &mut last,
                    &mut stats,
                )?;
            }

            plan.check_placement_memory(held, times.len())?;
            let placement = Placement::new(times, bins, Some(end_time));
            if placement.slot_count == 0 {
                continue;
            }
            // `read_signals` stops at the first time point after the header end time, before it
            // parses the values there. The decoder must stop at the same place.
            let kept = times
                .iter()
                .position(|&t| t > end_time)
                .unwrap_or(times.len());
            let Some(last_time_index) = kept.checked_sub(1) else {
                continue;
            };
            plan.check_decode_memory(held, times.len(), placement.slot_count, parts)?;
            let chunks = balanced_chunks(selected, &mut last, parts);
            let decoded = decode_section::<FULL>(
                &section,
                last_time_index,
                &placement,
                plan,
                handle_channels,
                chunks,
            )?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes(chunks: &[(&[usize], &mut [Last])]) -> Vec<usize> {
        chunks
            .iter()
            .map(|(handles, states)| {
                assert_eq!(handles.len(), states.len());
                handles.len()
            })
            .collect()
    }

    /// Eight neighboring handles among 100000. The decode used to cut the whole handle range into
    /// parts, so all eight handles fell into one part and decoded on one thread.
    #[test]
    fn a_narrow_selection_is_split_over_all_threads() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let selected: Vec<usize> = (50_000..50_008).collect();
        let mut states = vec![Last::None; selected.len()];
        let parts = pool.install(|| part_count(selected.len(), rayon::current_num_threads()));
        let chunks = balanced_chunks(&selected, &mut states, parts);
        assert!(sizes(&chunks).iter().filter(|&&n| n > 0).count() >= 2);
        assert_eq!(sizes(&chunks), vec![1; 8]);
    }

    #[test]
    fn chunks_have_nearly_equal_lengths_and_keep_the_order() {
        for (len, parts) in [(17, 16), (100, 16), (5, 16), (16, 16), (33, 4), (1, 1)] {
            let selected: Vec<usize> = (0..len).map(|i| 3 * i + 7).collect();
            let mut states = vec![Last::None; len];
            let chunks = balanced_chunks(&selected, &mut states, parts);
            let sizes = sizes(&chunks);
            assert_eq!(sizes.len(), parts.min(len), "{len} handles, {parts} parts");
            let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
            assert!(max - min <= 1, "{sizes:?}");
            let joined: Vec<usize> = chunks.iter().flat_map(|(h, _)| h.iter().copied()).collect();
            assert_eq!(joined, selected);
        }
    }

    #[test]
    fn no_selected_handle_gives_no_chunk() {
        let mut states: Vec<Last> = Vec::new();
        assert!(balanced_chunks(&[], &mut states, 16).is_empty());
        assert_eq!(part_count(0, 8), 0);
        // A zero part count (it cannot happen with rayon) still works.
        assert_eq!(part_count(5, 0), 1);
    }

    /// Each chunk gets the state of exactly its handles.
    #[test]
    fn a_chunk_holds_the_states_of_its_handles() {
        let selected = vec![2, 5, 9, 10, 40];
        let mut states: Vec<Last> = selected
            .iter()
            .map(|&h| Last::Chars(vec![h as u8]))
            .collect();
        for (handles, chunk_states) in balanced_chunks(&selected, &mut states, 2) {
            for (&h, state) in handles.iter().zip(chunk_states.iter()) {
                assert!(matches!(state, Last::Chars(c) if c == &[h as u8]));
            }
        }
    }
}
