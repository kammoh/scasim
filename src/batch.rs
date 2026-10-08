//! One simulation batch: legacy metadata, power trace, and traces cut at the markers.

use crate::hierarchy::Selection;
use crate::power::{PowerTrace, RunInfo, power_trace};
use miette::{Context, IntoDiagnostic, miette};
use ndarray::{Array1, Array2, s};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Metadata of one batch in the legacy format written by the cocotb testbenches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchMeta {
    /// Waveform file, resolved relative to the metadata file.
    pub trace_path: PathBuf,
    /// If present, only time points at multiples of the clock period are kept (legacy sampling).
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
    if meta_path.extension().is_some_and(|e| e == "gz") {
        flate2::read::GzDecoder::new(file)
            .read_to_end(&mut text)
            .into_diagnostic()?;
    } else {
        std::io::BufReader::new(file)
            .read_to_end(&mut text)
            .into_diagnostic()?;
    }
    let json: serde_json::Value = serde_json::from_slice(&text).into_diagnostic()?;
    let trace_filename = json
        .get("trace_filename")
        .and_then(|v| v.as_str())
        .ok_or_else(|| miette!("{}: missing trace_filename", meta_path.display()))?;
    let clock_period = match json.get("clock_period") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| {
            miette!(
                "{}: clock_period must be a non-negative integer, got {v}",
                meta_path.display()
            )
        })?),
    };
    let markers = json
        .get("markers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| miette!("{}: missing markers", meta_path.display()))?
        .iter()
        .map(|m| {
            let m = m
                .as_array()
                .filter(|m| m.len() == 3)
                .ok_or_else(|| miette!("bad marker {m}"))?;
            let n = |i: usize| {
                m[i].as_u64()
                    .ok_or_else(|| miette!("bad marker value {}", m[i]))
            };
            Ok((n(0)?, n(1)?, u16::try_from(n(2)?).into_diagnostic()?))
        })
        .collect::<miette::Result<Vec<_>>>()?;
    let dir = meta_path.parent().unwrap_or(Path::new("."));
    Ok(BatchMeta {
        trace_path: dir.join(trace_filename),
        clock_period,
        markers,
    })
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

/// Applies the legacy sampling and the markers to a power trace.
fn traces_from_power(
    trace: PowerTrace,
    meta: &BatchMeta,
    info: RunInfo,
) -> (Array2<f32>, Array1<u16>, BatchDiagnostics) {
    let total_toggles = trace.total();
    let kept = match meta.clock_period {
        Some(period) if period > 0 => trace.keep_multiples_of(period),
        _ => trace,
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
