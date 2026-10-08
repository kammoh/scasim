//! An oracle for real FST files, built from the event stream of
//! [`fst_reader::FstReader::read_signals`]. It uses no code from `src/power`. The state rules are
//! those of the fixture oracle (`oracle_change`). Events are placed into bins with a plain linear
//! search.
//!
//! `read_signals` is the specification of which changes a file has. This includes the frame values
//! of the first section, which it reports at the start time of that section when the first time
//! point is later. It reports no change after the end time in the file header.

use super::oracle::oracle_change;
use fst_reader::{FstFilter, FstReader, FstSignalValue, ReaderError};
use scasim::hierarchy::HierarchyIndex;
use scasim::power::{ActivityTrace, Bins, ChannelTrace, FullStats, RunInfo, Totals, UnknownPolicy};
use std::io::BufReader;
use std::path::Path;

/// What the bins of a file can depend on.
pub struct Layout {
    /// The distinct times of the time table, increasing.
    pub times: Vec<u64>,
    /// The start time of the first section that `read_signals` reads.
    pub first_start: u64,
}

/// The result of the event oracle for one file.
pub struct EventResult {
    /// The bins of each maker, in the order of the makers.
    pub bins: Vec<Bins>,
    /// `traces[p][k]` is the activity for the policy `p` and the bins `k`: one channel `all` with
    /// all selectable signals, with full statistics. `info` has only the number of selected
    /// handles.
    pub traces: Vec<Vec<ActivityTrace>>,
    /// The number of frame values of selectable signals. `read_signals` reports the frame values
    /// of the first section at the start time of that section, if the section has no time point
    /// at that time. The global time table has the start time as its first time point then.
    pub frame_events: usize,
    /// The number of selectable signals (handles).
    pub selected_handles: usize,
}

/// Opens an FST file like the fast path does: with the `.fst.hier` file when the file has no
/// hierarchy block.
fn open(path: &Path) -> Result<FstReader<BufReader<std::fs::File>>, ReaderError> {
    let file = || std::fs::File::open(path).map(BufReader::new);
    match FstReader::open_and_read_time_table(file()?) {
        Err(ReaderError::MissingGeometry() | ReaderError::MissingHierarchy()) => {
            let mut hier = path.to_path_buf();
            hier.set_extension("fst.hier");
            let hier = BufReader::new(std::fs::File::open(hier)?);
            FstReader::open_incomplete_and_read_time_table(file()?, hier)
        }
        other => other,
    }
}

/// Finds the slot of times with a linear search: 0 before the first bin, `k + 1` in bin `k`, and
/// `bins.len() + 1` at or after the end. The events come in the order of time, so the search
/// continues from the last position. It starts again at the beginning if a time is earlier.
struct Placer<'a> {
    bins: &'a Bins,
    /// The number of starts that are at or before the last time.
    passed: usize,
}

impl Placer<'_> {
    fn slot(&mut self, time: u64) -> usize {
        if self.bins.end().is_some_and(|end| time >= end) {
            return self.bins.len() + 1;
        }
        let starts = self.bins.starts();
        if self.passed > 0 && time < starts[self.passed - 1] {
            self.passed = 0;
        }
        while self.passed < starts.len() && starts[self.passed] <= time {
            self.passed += 1;
        }
        self.passed
    }
}

/// Computes the expected activity of an FST file with all selectable signals in one channel `all`
/// and full statistics, for each of the `policies` and for the bins of each of the `bin_makers`.
/// One pass over the events gives all results. A maker makes the bins from the layout of the file.
///
/// Selectable signals are those that `HierarchyIndex::from_fst` gives a path: not events, strings,
/// reals, or signals of width 0. Real values in the event stream are ignored. The first value of
/// a signal counts only for the Hamming weight.
///
/// Returns an error text if the file has no time points, if its time table decreases, or if
/// the reader fails.
pub fn expected_from_read_signals(
    path: &Path,
    policies: &[UnknownPolicy],
    bin_makers: &[&dyn Fn(&Layout) -> Bins],
) -> Result<EventResult, String> {
    let mut reader = open(path).map_err(|e| format!("cannot open the file: {e}"))?;
    let header = reader.get_header();
    let table = reader.get_time_table().unwrap_or_default().to_vec();
    if table.is_empty() {
        return Err("no time points".into());
    }
    if let Some(w) = table.windows(2).find(|w| w[0] > w[1]) {
        return Err(format!(
            "the time table decreases from {} to {}",
            w[0], w[1]
        ));
    }
    let mut times = table.clone();
    times.dedup();
    // `read_signals` skips the sections that start after the end time.
    let sections = reader.sections();
    let first_section = sections
        .iter()
        .position(|s| s.start_time <= header.end_time);
    let first_start = first_section.map_or(0, |i| sections[i].start_time);
    // The frame is reported if the first section has no time point at its start time.
    let has_frame = match first_section {
        Some(i) => {
            let section = reader
                .read_section(i)
                .map_err(|e| format!("cannot read the first section: {e}"))?;
            section
                .time_table()
                .first()
                .is_none_or(|&t| t > first_start)
        }
        None => false,
    };
    let layout = Layout { times, first_start };
    let bins: Vec<Bins> = bin_makers.iter().map(|make| make(&layout)).collect();

    let index = HierarchyIndex::from_fst(&mut reader).map_err(|e| format!("hierarchy: {e}"))?;
    let selectable: Vec<bool> = index.paths.iter().map(|p| !p.is_empty()).collect();

    // `slots[p][k]` are the totals for the policy `p` and the bins `k`, in the slots: before the
    // first bin, the bins, and after the end.
    let mut slots: Vec<Vec<Vec<Totals>>> = policies
        .iter()
        .map(|_| {
            bins.iter()
                .map(|b| vec![Totals::default(); b.len() + 2])
                .collect()
        })
        .collect();
    let mut placers: Vec<Placer> = bins.iter().map(|bins| Placer { bins, passed: 0 }).collect();
    let mut places = vec![0; bins.len()];
    let mut last: Vec<Option<String>> = vec![None; selectable.len()];
    let mut frame_events = 0;
    reader
        .read_signals(&FstFilter::all(), |time, handle, value| {
            let h = handle.get_index();
            if !selectable.get(h).copied().unwrap_or(false) {
                return Ok::<(), ()>(());
            }
            let FstSignalValue::String(chars) = value else {
                return Ok(());
            };
            let new = std::str::from_utf8(chars).expect("state characters are ASCII");
            // If the first section has a frame, the only events at its start time are frame
            // values.
            if has_frame && time == first_start {
                frame_events += 1;
            }
            for (place, placer) in places.iter_mut().zip(&mut placers) {
                *place = placer.slot(time);
            }
            for (p, &unknown) in policies.iter().enumerate() {
                let d = oracle_change(last[h].as_deref(), new, unknown);
                for (k, &place) in places.iter().enumerate() {
                    let slot = &mut slots[p][k][place];
                    slot.toggles += d.toggles;
                    slot.rise += d.rise;
                    slot.fall += d.fall;
                    slot.hw_delta += d.hw_delta;
                }
            }
            match &mut last[h] {
                Some(old) => {
                    old.clear();
                    old.push_str(new);
                }
                none => *none = Some(new.to_string()),
            }
            Ok(())
        })
        .map_err(|e| format!("read_signals: {e}"))?;

    let selected_handles = selectable.iter().filter(|&&s| s).count();
    let traces = slots
        .iter()
        .map(|per_bins| {
            per_bins
                .iter()
                .zip(&bins)
                .map(|(slots, bins)| {
                    let in_bins = &slots[1..=bins.len()];
                    let channel = ChannelTrace {
                        name: "all".into(),
                        toggles: in_bins.iter().map(|t| t.toggles).collect(),
                        full: Some(FullStats {
                            rise: in_bins.iter().map(|t| t.rise).collect(),
                            fall: in_bins.iter().map(|t| t.fall).collect(),
                            hw_delta: in_bins.iter().map(|t| t.hw_delta).collect(),
                        }),
                        before: slots[0],
                        after: slots[bins.len() + 1],
                    };
                    ActivityTrace {
                        times: bins.starts().to_vec(),
                        timescale_exponent: Some(header.timescale_exponent),
                        channels: vec![channel],
                        info: RunInfo {
                            selected_handles,
                            ..RunInfo::default()
                        },
                    }
                })
                .collect()
        })
        .collect();
    Ok(EventResult {
        bins,
        traces,
        frame_events,
        selected_handles,
    })
}

fn assert_same_values<T: PartialEq + std::fmt::Debug>(
    got: &[T],
    expected: &[T],
    what: &str,
    context: &str,
) {
    assert_eq!(got.len(), expected.len(), "{context}: length of {what}");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != expected[i]) {
        panic!(
            "{context}: {what} differ at bin {i}: got {:?}, expected {:?}",
            got[i], expected[i]
        );
    }
}

/// Asserts that two channels are equal. A failure names the first difference. It does not print
/// the vectors of a whole file.
pub fn assert_same_channel(got: &ChannelTrace, expected: &ChannelTrace, context: &str) {
    assert_eq!(got.name, expected.name, "{context}");
    assert_eq!(
        got.before, expected.before,
        "{context}: before the first bin"
    );
    assert_eq!(got.after, expected.after, "{context}: after the end");
    assert_same_values(&got.toggles, &expected.toggles, "toggles", context);
    let (got, expected) = (got.full.as_ref(), expected.full.as_ref());
    assert_eq!(
        got.is_some(),
        expected.is_some(),
        "{context}: full statistics"
    );
    if let (Some(g), Some(e)) = (got, expected) {
        assert_same_values(&g.rise, &e.rise, "rise", context);
        assert_same_values(&g.fall, &e.fall, "fall", context);
        assert_same_values(&g.hw_delta, &e.hw_delta, "hw_delta", context);
    }
}
