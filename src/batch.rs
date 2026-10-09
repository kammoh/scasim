//! One simulation batch: legacy metadata, power trace, and traces cut at the markers.

use crate::hierarchy::Selection;
use crate::power::{PowerTrace, RunInfo, power_trace};
use miette::{Context, IntoDiagnostic, miette};
use ndarray::{Array1, Array2, s};
use ndarray_npz::{NpzReader, NpzWriter};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Metadata of one batch in the legacy format written by the cocotb testbenches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchMeta {
    /// Waveform file, resolved relative to the metadata file.
    pub trace_path: PathBuf,
    /// If present, only time points at multiples of the clock period are kept (legacy sampling).
    /// It is greater than zero. `read_batch_meta` rejects a zero.
    pub clock_period: Option<u64>,
    /// One marker per trace: start time, end time (exclusive), and class label.
    pub markers: Vec<(u64, u64, u16)>,
}

/// Reads a `meta.json` or gzip-compressed `meta.json.gz` file.
pub fn read_batch_meta(meta_path: &Path) -> miette::Result<BatchMeta> {
    let file = std::fs::File::open(meta_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot open {}", meta_path.display()))?;
    let mut text = Vec::new();
    let read = if meta_path.extension().is_some_and(|e| e == "gz") {
        flate2::read::GzDecoder::new(file).read_to_end(&mut text)
    } else {
        std::io::BufReader::new(file).read_to_end(&mut text)
    };
    read.into_diagnostic()
        .wrap_err_with(|| format!("cannot read {}", meta_path.display()))?;
    let json: serde_json::Value = serde_json::from_slice(&text)
        .into_diagnostic()
        .wrap_err_with(|| format!("{} is not valid JSON", meta_path.display()))?;
    let trace_filename = json
        .get("trace_filename")
        .and_then(|v| v.as_str())
        .ok_or_else(|| miette!("{}: missing trace_filename", meta_path.display()))?;
    let clock_period = match json.get("clock_period") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(v.as_u64().filter(|&period| period > 0).ok_or_else(|| {
            miette!(
                "{}: clock_period must be a positive integer, got {v}",
                meta_path.display()
            )
        })?),
    };
    let markers = json
        .get("markers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| miette!("{}: missing markers", meta_path.display()))?
        .iter()
        .map(|marker| {
            let values = marker
                .as_array()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| miette!("{}: bad marker {marker}", meta_path.display()))?;
            let n = |i: usize| {
                values[i].as_u64().ok_or_else(|| {
                    miette!(
                        "{}: bad marker value {} in {marker}",
                        meta_path.display(),
                        values[i]
                    )
                })
            };
            let label = u16::try_from(n(2)?).map_err(|_| {
                miette!(
                    "{}: the label of the marker {marker} does not fit in 16 bits",
                    meta_path.display()
                )
            })?;
            Ok((n(0)?, n(1)?, label))
        })
        .collect::<miette::Result<Vec<_>>>()?;
    let dir = meta_path.parent().unwrap_or(Path::new("."));
    Ok(BatchMeta {
        trace_path: dir.join(trace_filename),
        clock_period,
        markers,
    })
}

/// Reads the traces and labels from a `traces.npz` cache file.
///
/// The file has the arrays `trace_<i>` and `labels`. The indices `i` must be exactly `0..n`, in
/// any order in the archive. The traces have the same length, and `labels` has `n` entries.
/// Any other file is an error, and the message names the file.
pub fn read_trace_cache(path: &Path) -> miette::Result<(Array2<f32>, Array1<u16>)> {
    let name = path.display();
    let file = std::fs::File::open(path)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot open {name}"))?;
    let mut npz = NpzReader::new(file)
        .into_diagnostic()
        .wrap_err_with(|| format!("{name} is not a valid npz file"))?;
    let labels: Array1<u16> = npz
        .by_name("labels")
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot read the array `labels` in {name}"))?;
    let names = npz
        .names()
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot list the arrays in {name}"))?;
    let trace_names: Vec<&String> = names.iter().filter(|n| n.starts_with("trace_")).collect();
    let order =
        trace_order(trace_names.iter().map(|n| n.as_str()), labels.len()).map_err(|problem| {
            miette!("{name}: {problem}. Delete the file or use --use-existing=false")
        })?;
    let mut rows = Vec::with_capacity(order.len());
    for &position in &order {
        let row_name = trace_names[position];
        let row: Array1<f32> = npz
            .by_name(row_name)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot read the array `{row_name}` in {name}"))?;
        rows.push(row);
    }
    let len = rows[0].len();
    if let Some(bad) = rows.iter().position(|r| r.len() != len) {
        return Err(miette!(
            "{name}: trace_{bad} has {} samples, but trace_0 has {len}. Delete the file or use \
             --use-existing=false",
            rows[bad].len()
        ));
    }
    let traces = Array2::from_shape_vec((rows.len(), len), rows.into_iter().flatten().collect())
        .into_diagnostic()?;
    Ok((traces, labels))
}

/// Sorts the names `trace_<i>` by the index `i`. Returns, for each index, the position of its
/// name in `names`. Fails if the indices are not exactly `0..n`, if there are none, or if there
/// are not `labels` entries for them.
fn trace_order<'a>(
    names: impl Iterator<Item = &'a str>,
    labels: usize,
) -> Result<Vec<usize>, String> {
    let mut indexed = Vec::new();
    for (position, name) in names.enumerate() {
        let index = name["trace_".len()..]
            .parse::<usize>()
            .map_err(|_| format!("the array name `{name}` has no trace index"))?;
        indexed.push((index, position));
    }
    if indexed.is_empty() {
        return Err("the file has no traces".into());
    }
    indexed.sort_unstable();
    if let Some(wrong) = indexed
        .iter()
        .enumerate()
        .find(|(i, (index, _))| i != index)
    {
        let expected = wrong.0;
        return Err(format!(
            "the trace indices are not 0..{}: trace_{expected} is missing or repeated",
            indexed.len()
        ));
    }
    if labels != indexed.len() {
        return Err(format!(
            "the file has {} traces, but {labels} labels",
            indexed.len()
        ));
    }
    Ok(indexed.into_iter().map(|(_, position)| position).collect())
}

/// Writes the traces and labels to a `traces.npz` cache file.
pub fn write_trace_cache(
    path: &Path,
    traces: &Array2<f32>,
    labels: &Array1<u16>,
) -> miette::Result<()> {
    let name = path.display();
    let file = std::fs::File::create(path)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot create {name}"))?;
    let mut npz = NpzWriter::new_compressed(file);
    for (i, trace) in traces.outer_iter().enumerate() {
        npz.add_array(format!("trace_{i}"), &trace)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot write {name}"))?;
    }
    npz.add_array("labels", labels)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot write {name}"))?;
    npz.finish()
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot write {name}"))?;
    Ok(())
}

/// The index range `[low, high)` of the time points that each marker covers, and its label.
/// A time that is not in the trace stands for the next later time point.
fn marker_ranges(trace: &PowerTrace, markers: &[(u64, u64, u16)]) -> Vec<(usize, usize, u16)> {
    let index = |t: u64| trace.times.binary_search(&t).unwrap_or_else(|i| i);
    markers
        .iter()
        .map(|&(start, end, label)| {
            let low = index(start);
            (low, index(end).max(low), label)
        })
        .collect()
}

/// Cuts one trace per marker. A marker covers the time points in `[start, end)`. Shorter traces
/// are padded with zeros to the length of the longest trace.
pub fn cut_traces(trace: &PowerTrace, markers: &[(u64, u64, u16)]) -> (Array2<f32>, Array1<u16>) {
    let ranges = marker_ranges(trace, markers);
    let max_len = ranges.iter().map(|(lo, hi, _)| hi - lo).max().unwrap_or(0);
    let mut traces = Array2::<f32>::zeros((ranges.len(), max_len));
    let mut labels = Array1::<u16>::zeros(ranges.len());
    for (i, &(lo, hi, label)) in ranges.iter().enumerate() {
        labels[i] = label;
        let values: Array1<f32> = trace.power[lo..hi].iter().map(|&p| p as f32).collect();
        traces.slice_mut(s![i, ..hi - lo]).assign(&values);
    }
    (traces, labels)
}

/// The toggles at the time points that at least one marker covers.
fn toggles_in_markers(trace: &PowerTrace, markers: &[(u64, u64, u16)]) -> u64 {
    let mut covered = vec![false; trace.times.len()];
    for (lo, hi, _) in marker_ranges(trace, markers) {
        covered[lo..hi].fill(true);
    }
    trace
        .power
        .iter()
        .zip(&covered)
        .filter(|(_, covered)| **covered)
        .map(|(p, _)| p)
        .sum()
}

/// Counts that show how much activity the legacy sampling and the markers leave out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchDiagnostics {
    /// Toggles of the selected signals in the whole waveform.
    pub total_toggles: u64,
    /// Toggles at the time points that the legacy sampling keeps.
    pub kept_toggles: u64,
    /// Toggles at kept time points that at least one marker covers.
    pub segment_toggles: u64,
    /// What the selection matched.
    pub info: RunInfo,
}

impl BatchDiagnostics {
    /// True if the legacy sampling drops more than half of the toggles. Then the traces probably
    /// miss most of the activity of the selected signals.
    pub fn sampling_drops_most(&self) -> bool {
        self.kept_toggles.saturating_mul(2) < self.total_toggles
    }

    /// True if the selected signals lie below more than one top-level scope. Then the selection
    /// probably includes more than the design under test, for example a testbench.
    pub fn spans_several_top_scopes(&self) -> bool {
        self.info.top_scopes.len() > 1
    }
}

/// Applies the legacy sampling and the markers to a power trace.
fn traces_from_power(
    trace: PowerTrace,
    meta: &BatchMeta,
    info: RunInfo,
) -> (Array2<f32>, Array1<u16>, BatchDiagnostics) {
    let total_toggles = trace.total();
    let kept = match meta.clock_period {
        Some(period) => trace.keep_multiples_of(period),
        None => trace,
    };
    let diagnostics = BatchDiagnostics {
        total_toggles,
        kept_toggles: kept.total(),
        segment_toggles: toggles_in_markers(&kept, &meta.markers),
        info,
    };
    let (traces, labels) = cut_traces(&kept, &meta.markers);
    (traces, labels, diagnostics)
}

/// Computes the traces, labels, and diagnostics of one batch from the selected signals.
pub fn batch_traces(
    meta: &BatchMeta,
    selection: &Selection,
) -> miette::Result<(Array2<f32>, Array1<u16>, BatchDiagnostics)> {
    let (trace, info) = power_trace(&meta.trace_path, selection)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot compute power of {}", meta.trace_path.display()))?;
    Ok(traces_from_power(trace, meta, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::power::PowerTrace;
    use ndarray::array;

    fn trace() -> PowerTrace {
        PowerTrace {
            times: vec![0, 10, 20, 30, 40, 50],
            power: vec![1, 2, 3, 4, 5, 6],
        }
    }

    #[test]
    fn cut_traces_pads_shorter_traces_with_zeros() {
        // [10, 30) -> indices 1..3 -> [2, 3]; [30, 60) -> indices 3..6 -> [4, 5, 6]
        let (traces, labels) = cut_traces(&trace(), &[(10, 30, 0), (30, 60, 1)]);
        assert_eq!(traces, array![[2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]);
        assert_eq!(labels, array![0u16, 1]);
    }

    #[test]
    fn cut_traces_uses_the_next_time_point_for_unknown_marker_times() {
        let (traces, _) = cut_traces(&trace(), &[(15, 40, 1)]);
        assert_eq!(traces, array![[3.0, 4.0]]);
    }

    #[test]
    fn read_batch_meta_reads_plain_and_gzip_json() {
        let json = r#"{"trace_filename": "tvla.fst", "clock_period": 10,
                       "markers": [[3710, 7420, 0], [7420, 11130, 1]], "num_tests": 2}"#;
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("meta.json");
        std::fs::write(&plain, json).unwrap();
        let gz = dir.path().join("meta.json.gz");
        let mut enc = flate2::write::GzEncoder::new(
            std::fs::File::create(&gz).unwrap(),
            flate2::Compression::default(),
        );
        std::io::Write::write_all(&mut enc, json.as_bytes()).unwrap();
        enc.finish().unwrap();
        for path in [plain, gz] {
            let meta = read_batch_meta(&path).unwrap();
            assert_eq!(meta.trace_path, dir.path().join("tvla.fst"));
            assert_eq!(meta.clock_period, Some(10));
            assert_eq!(meta.markers, vec![(3710, 7420, 0), (7420, 11130, 1)]);
        }
    }

    #[test]
    fn trace_order_sorts_by_the_parsed_index() {
        // A text sort puts `trace_10` before `trace_2`. The parsed index does not.
        let names: Vec<String> = [2, 10, 0, 1, 3, 4, 5, 6, 7, 8, 9]
            .iter()
            .map(|i| format!("trace_{i}"))
            .collect();
        let order = trace_order(names.iter().map(String::as_str), 11).unwrap();
        let sorted: Vec<&str> = order.iter().map(|&p| names[p].as_str()).collect();
        let want: Vec<String> = (0..11).map(|i| format!("trace_{i}")).collect();
        assert_eq!(sorted, want);
    }

    #[test]
    fn trace_order_rejects_gaps_repeats_and_other_problems() {
        let err = |names: &[&'static str], labels| trace_order(names.iter().copied(), labels);
        assert!(err(&["trace_0", "trace_1"], 2).is_ok());
        assert!(err(&[], 0).unwrap_err().contains("no traces"));
        assert!(
            err(&["trace_0", "trace_2"], 2)
                .unwrap_err()
                .contains("trace_1")
        );
        assert!(err(&["trace_0", "trace_0"], 2).is_err());
        assert!(
            err(&["trace_1", "trace_2"], 2)
                .unwrap_err()
                .contains("trace_0")
        );
        assert!(
            err(&["trace_0", "trace_x"], 2)
                .unwrap_err()
                .contains("trace_x")
        );
        assert!(
            err(&["trace_0", "trace_1"], 3)
                .unwrap_err()
                .contains("labels")
        );
    }

    #[test]
    fn the_trace_cache_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traces.npz");
        let traces = Array2::from_shape_fn((12, 3), |(i, j)| (10 * i + j) as f32);
        let labels = Array1::from_iter((0..12).map(|i| (i % 2) as u16));
        write_trace_cache(&path, &traces, &labels).unwrap();
        let (got_traces, got_labels) = read_trace_cache(&path).unwrap();
        assert_eq!(got_traces, traces);
        assert_eq!(got_labels, labels);
    }

    #[test]
    fn read_batch_meta_errors_name_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let bad_gzip = dir.path().join("bad.json.gz");
        std::fs::write(&bad_gzip, b"this is not gzip").unwrap();
        let bad_json = dir.path().join("bad.json");
        std::fs::write(&bad_json, b"{ not json").unwrap();
        for path in [bad_gzip, bad_json] {
            let error = read_batch_meta(&path).unwrap_err();
            let shown = error.to_string();
            assert!(shown.contains(&path.display().to_string()), "{shown}");
        }
    }

    #[test]
    fn read_batch_meta_rejects_a_non_integer_clock_period() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.json");
        std::fs::write(
            &path,
            r#"{"trace_filename": "a.fst", "clock_period": 10.5, "markers": []}"#,
        )
        .unwrap();
        assert!(read_batch_meta(&path).is_err());
    }

    #[test]
    fn read_batch_meta_without_clock_period() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.json");
        std::fs::write(&path, r#"{"trace_filename": "a.vcd", "markers": []}"#).unwrap();
        let meta = read_batch_meta(&path).unwrap();
        assert_eq!(meta.clock_period, None);
        assert!(meta.markers.is_empty());
    }

    fn meta(clock_period: Option<u64>, markers: Vec<(u64, u64, u16)>) -> BatchMeta {
        BatchMeta {
            trace_path: PathBuf::from("unused"),
            clock_period,
            markers,
        }
    }

    #[test]
    fn diagnostics_count_the_toggles_that_each_step_drops() {
        let trace = PowerTrace {
            times: vec![0, 5, 10, 15, 20],
            power: vec![0, 1, 3, 1, 1],
        };
        // Sampling keeps the times 0, 10, and 20. The markers cover the points at 0 and 10.
        let m = meta(Some(10), vec![(0, 10, 0), (10, 15, 1)]);
        let (traces, labels, d) = traces_from_power(trace, &m, RunInfo::default());
        assert_eq!(
            (d.total_toggles, d.kept_toggles, d.segment_toggles),
            (6, 4, 3)
        );
        assert_eq!(traces, array![[0.0], [3.0]]);
        assert_eq!(labels, array![0u16, 1]);
    }

    #[test]
    fn overlapping_markers_count_a_toggle_once() {
        let trace = PowerTrace {
            times: vec![0, 10, 20],
            power: vec![1, 2, 4],
        };
        let m = meta(None, vec![(0, 20, 0), (10, 30, 1)]);
        let (_, _, d) = traces_from_power(trace, &m, RunInfo::default());
        assert_eq!(
            (d.total_toggles, d.kept_toggles, d.segment_toggles),
            (7, 7, 7)
        );
    }

    fn diagnostics(total: u64, kept: u64, top_scopes: &[&str]) -> BatchDiagnostics {
        BatchDiagnostics {
            total_toggles: total,
            kept_toggles: kept,
            segment_toggles: 0,
            info: RunInfo {
                top_scopes: top_scopes.iter().map(|s| s.to_string()).collect(),
                ..RunInfo::default()
            },
        }
    }

    #[test]
    fn sampling_drops_most_when_it_keeps_less_than_half_of_the_toggles() {
        assert!(!diagnostics(10, 5, &[]).sampling_drops_most(), "half");
        assert!(diagnostics(10, 4, &[]).sampling_drops_most(), "one less");
        assert!(!diagnostics(11, 6, &[]).sampling_drops_most(), "odd total");
        assert!(diagnostics(11, 5, &[]).sampling_drops_most(), "odd total");
        assert!(diagnostics(10, 0, &[]).sampling_drops_most(), "none kept");
        assert!(!diagnostics(10, 10, &[]).sampling_drops_most(), "all kept");
        assert!(!diagnostics(0, 0, &[]).sampling_drops_most(), "no toggles");
    }

    #[test]
    fn selection_spans_several_top_scopes_from_two_scopes_on() {
        assert!(!diagnostics(0, 0, &[]).spans_several_top_scopes());
        assert!(!diagnostics(0, 0, &["tb"]).spans_several_top_scopes());
        assert!(diagnostics(0, 0, &["a", "b"]).spans_several_top_scopes());
        assert!(diagnostics(0, 0, &["a", "b", "c"]).spans_several_top_scopes());
    }

    #[test]
    fn cut_traces_gives_an_empty_trace_to_a_marker_that_ends_before_it_starts() {
        // [30, 10) has low index 3 and high index 1. The range must be empty, not negative.
        let (traces, labels) = cut_traces(&trace(), &[(30, 10, 1), (10, 30, 0)]);
        assert_eq!(traces, array![[0.0, 0.0], [2.0, 3.0]]);
        assert_eq!(labels, array![1u16, 0]);
        let (_, _, d) =
            traces_from_power(trace(), &meta(None, vec![(30, 10, 1)]), RunInfo::default());
        assert_eq!(d.segment_toggles, 0);
    }

    #[test]
    fn read_batch_meta_rejects_a_zero_clock_period() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.json");
        std::fs::write(
            &path,
            r#"{"trace_filename": "a.fst", "clock_period": 0, "markers": []}"#,
        )
        .unwrap();
        let message = read_batch_meta(&path).unwrap_err().to_string();
        assert!(message.contains(&path.display().to_string()), "{message}");
        assert!(message.contains("clock_period"), "{message}");
    }

    #[test]
    fn read_batch_meta_names_the_file_in_marker_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.json");
        for (markers, what) in [
            ("[[1, 2]]", "bad marker"),
            ("[[1, 2, \"x\"]]", "bad marker value"),
            ("[[1, 2, 70000]]", "label"),
            ("[[1, -2, 0]]", "bad marker value"),
        ] {
            std::fs::write(
                &path,
                format!(r#"{{"trace_filename": "a.fst", "markers": {markers}}}"#),
            )
            .unwrap();
            let message = read_batch_meta(&path).unwrap_err().to_string();
            assert!(message.contains(&path.display().to_string()), "{message}");
            assert!(message.contains(what), "{markers}: {message}");
        }
    }

    #[test]
    fn batch_traces_reads_a_waveform_and_reports_the_selection() {
        let vcd = "$timescale 1ps $end\n$scope module tb $end\n\
                   $var wire 1 ! clk $end\n$var wire 4 \" data $end\n$upscope $end\n\
                   $enddefinitions $end\n#0\n$dumpvars\n0!\nb0000 \"\n$end\n\
                   #5\n1!\n#10\n0!\nb1010 \"\n#15\n1!\n#20\n0!\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.vcd");
        std::fs::write(&path, vcd).unwrap();
        let m = BatchMeta {
            trace_path: path,
            clock_period: Some(10),
            markers: vec![(0, 10, 0), (10, 15, 1)],
        };
        let all = Selection::parse(&["+scope:tb", "-signal:tb.clk"]).unwrap();
        // Without the clock, only `data` toggles: twice at time 10.
        let (traces, _, d) = batch_traces(&m, &all).unwrap();
        assert_eq!(
            (d.total_toggles, d.kept_toggles, d.segment_toggles),
            (2, 2, 2)
        );
        assert_eq!(traces, array![[0.0], [2.0]]);
        assert_eq!(d.info.selected_handles, 1);
        let unmatched = Selection::parse(&["+scope:tb", "-signal:tb.nope"]).unwrap();
        let (_, _, d) = batch_traces(&m, &unmatched).unwrap();
        assert_eq!(d.info.unmatched_rules, vec!["-signal:tb.nope"]);
        assert_eq!(
            (d.total_toggles, d.kept_toggles, d.segment_toggles),
            (6, 4, 3)
        );
    }
}
