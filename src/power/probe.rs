//! Probes: the value changes of one signal, given by its exact path.
//!
//! A probe is a signal whose values the analysis needs, for example a clock. The first value of
//! the signal is its initial value. The initial value is not a change. On an FST file it is the
//! frame value of the first section if the first time point is later than the section start,
//! as in the activity run. Otherwise it is the first change. The probe reports the same changes
//! as the activity run sees, including the cut at the end time of the file header.
//!
//! The state characters are in lower case: `0`, `1`, `x`, `z`, and so on. A change to the value
//! that the signal already has is not a change.

use super::{PowerError, fst, is_fst, reference};
use crate::hierarchy::HierarchyIndex;
use crate::power::stats::packed_to_chars;
use fst_reader::{FstSignalHandle, FstValue};
use std::path::Path;
use wellen::SignalValueRef;

/// The most paths that an error message lists.
const MAX_CANDIDATES: usize = 10;

/// The value changes of one signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeTrace {
    /// The path that was asked for.
    pub path: String,
    /// The first value of the signal, or `None` if it never has a digital value.
    pub initial: Option<String>,
    /// The later changes: time and new value. Times do not decrease. Two changes can have the
    /// same time (a glitch).
    pub changes: Vec<(u64, String)>,
}

impl ProbeTrace {
    fn new(path: &str) -> Self {
        ProbeTrace {
            path: path.to_string(),
            initial: None,
            changes: Vec::new(),
        }
    }

    /// Records the value of the signal at `time`.
    fn push(&mut self, time: u64, chars: &[u8]) {
        let value: String = chars
            .iter()
            .map(|c| c.to_ascii_lowercase() as char)
            .collect();
        let last = self
            .changes
            .last()
            .map(|(_, v)| v)
            .or(self.initial.as_ref());
        match last {
            None => self.initial = Some(value),
            Some(last) if *last != value => self.changes.push((time, value)),
            Some(_) => {}
        }
    }
}

/// The handle of the one signal with this exact path. Fails if no signal has it, or if several
/// signals have it. The message lists up to 10 paths that contain the last name of the path.
fn resolve_handle(index: &HierarchyIndex, path: &str) -> Result<usize, PowerError> {
    let handles: Vec<usize> = index
        .paths
        .iter()
        .enumerate()
        .filter(|(_, paths)| paths.iter().any(|p| p.path == path))
        .map(|(h, _)| h)
        .collect();
    match handles[..] {
        [handle] => Ok(handle),
        [] => {
            let name = path.rsplit('.').next().unwrap_or(path);
            let mut candidates: Vec<&str> = index
                .paths
                .iter()
                .flatten()
                .map(|p| p.path.as_str())
                .filter(|p| p.contains(name))
                .collect();
            candidates.sort_unstable();
            candidates.dedup();
            let more = candidates.len().saturating_sub(MAX_CANDIDATES);
            candidates.truncate(MAX_CANDIDATES);
            let list = if candidates.is_empty() {
                format!("no signal path contains `{name}`")
            } else if more > 0 {
                format!("candidates: {} (and {more} more)", candidates.join(", "))
            } else {
                format!("candidates: {}", candidates.join(", "))
            };
            Err(PowerError::Probe {
                path: path.to_string(),
                reason: format!("no signal has this path; {list}"),
            })
        }
        _ => Err(PowerError::Probe {
            path: path.to_string(),
            reason: format!("{} signals have this path", handles.len()),
        }),
    }
}

/// Reads the value changes of the signal with the exact path `signal_path`. FST files use the
/// section API of the `fst-reader` fork. All other formats use `wellen`.
pub fn probe_changes(path: &Path, signal_path: &str) -> Result<ProbeTrace, PowerError> {
    if is_fst(path)? {
        probe_fst(path, signal_path)
    } else {
        probe_reference(path, signal_path)
    }
}

/// The characters of an FST value, or `None` for a value that is not a bit vector.
fn fst_chars(value: FstValue<'_>, scratch: &mut Vec<u8>) -> Option<Vec<u8>> {
    match value {
        FstValue::Packed { width, bytes } => {
            packed_to_chars(width, bytes, scratch);
            Some(scratch.clone())
        }
        FstValue::Chars(chars) => Some(chars.to_vec()),
        FstValue::VarLen(_) | FstValue::Real(_) => None,
    }
}

fn probe_fst(path: &Path, signal_path: &str) -> Result<ProbeTrace, PowerError> {
    let mut reader = fst::open_reader(path)?;
    let end_time = reader.get_header().end_time;
    let has_time_points = reader.get_time_table().is_some_and(|t| !t.is_empty());
    let index = HierarchyIndex::from_fst(&mut reader)?;
    let handle = resolve_handle(&index, signal_path)?;
    let mut probe = ProbeTrace::new(signal_path);
    if !has_time_points {
        return Ok(probe);
    }
    let mut scratch = Vec::new();
    let mut first_section = true;
    // The same sections, frame, and cut as `fst::run`.
    for (section_index, info) in reader.sections().iter().enumerate() {
        if info.start_time > end_time {
            continue;
        }
        let section = reader.read_section(section_index)?;
        let times = section.time_table();
        let reports_frame = std::mem::take(&mut first_section)
            && times.first().is_none_or(|&t| t > info.start_time);
        if reports_frame {
            section.for_each_frame_value(|h, value| {
                if h.get_index() == handle
                    && let Some(chars) = fst_chars(value, &mut scratch)
                {
                    probe.push(info.start_time, &chars);
                }
            })?;
        }
        let kept = times
            .iter()
            .position(|&t| t > end_time)
            .unwrap_or(times.len());
        let Some(last_time_index) = kept.checked_sub(1) else {
            continue;
        };
        section.for_each_change_until(
            FstSignalHandle::from_index(handle),
            last_time_index,
            |time_index, value| {
                if let Some(chars) = fst_chars(value, &mut scratch) {
                    probe.push(times[time_index], &chars);
                }
            },
        )?;
    }
    Ok(probe)
}

fn probe_reference(path: &Path, signal_path: &str) -> Result<ProbeTrace, PowerError> {
    let header = wellen::viewers::read_header_from_file(path, &reference::load_options())?;
    let hierarchy = header.hierarchy;
    let index = HierarchyIndex::from_wellen(&hierarchy);
    let handle = resolve_handle(&index, signal_path)?;
    let body = wellen::viewers::read_body(header.body, &hierarchy, None)?;
    let mut probe = ProbeTrace::new(signal_path);
    if body.time_table.is_empty() {
        // `wellen` panics when it loads signals of an FST file without time points.
        return Ok(probe);
    }
    let signal_ref = wellen::SignalRef::from_index(handle).expect("a valid signal index");
    let mut source = body.source;
    let signals = source.load_signals(&[signal_ref], &hierarchy, true);
    let mut chars = Vec::new();
    for signal in &signals {
        for (time_index, value) in signal.iter_changes() {
            let SignalValueRef::BitVec(bits) = value else {
                continue;
            };
            chars.clear();
            chars.extend(bits.iter_msb_to_lsb().map(|b| b.as_ascii() as u8));
            probe.push(body.time_table[time_index as usize], &chars);
        }
    }
    Ok(probe)
}
