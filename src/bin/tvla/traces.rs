//! The per-channel traces file of one batch (`--traces-out`).
//!
//! The file is an `.npz` archive with these members:
//!
//! - `t_<i>`: the traces of the channel with index `i`, as `u32`, shape `(segments, samples)`.
//!   Index 0 is the channel `total`. With `--per-scope`, the scope channels follow in the order of
//!   `channels.txt` (their indices start at 1 here). The index does not change with
//!   `--traces-channels`. The values and the sample axis are the ones that go into the
//!   histograms, after the length policy of the batch.
//! - `labels` (`u16`): the labels of the segments, before `--shuffle-labels`.
//! - `groups` (`u64`) and `segment_ids` (`u64`): the group and the id of each segment. For
//!   legacy metadata, all groups are 0 and the ids are `0..segments`.
//! - `meta.json`: JSON text as a `u8` array. Fields: `format` (1), `kind`, `scasim` (version),
//!   `batch_id`, `segments`, `samples`, `shuffle_seed` (or `null`), `cache_key` (the key that
//!   `--stats-out` writes), and `channels`: a list of `{name, index, array, is_total, handles,
//!   handles_hash}` with `handles_hash` in hex.

use crate::cache::{CacheKey, ChannelIdentity};
use miette::{IntoDiagnostic, WrapErr, miette};
use ndarray::{Array1, Array2};
use ndarray_npz::NpzWriter;
use regex::Regex;
use scasim::batch::MAX_EXACT_COUNT;
use scasim::power::PowerPlan;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const FORMAT: u32 = 1;

/// What `--traces-out` and `--traces-channels` ask for.
#[derive(Debug, Clone)]
pub struct TracesOut {
    pub path: std::path::PathBuf,
    /// Exact channel names or `regex:PATTERN`. Empty means all channels.
    pub channels: Vec<String>,
}

/// The channels of the batch that the specs select, as sorted indices. Every spec must match at
/// least one channel. A `regex:` spec must match the whole name.
pub fn select_channels(specs: &[String], names: &[&str]) -> miette::Result<Vec<usize>> {
    if specs.is_empty() {
        return Ok((0..names.len()).collect());
    }
    let mut chosen = vec![false; names.len()];
    for spec in specs {
        let matches: Vec<bool> = if let Some(pattern) = spec.strip_prefix("regex:") {
            let regex = Regex::new(&format!("^(?:{pattern})$"))
                .into_diagnostic()
                .wrap_err_with(|| format!("invalid --traces-channels pattern {spec}"))?;
            names.iter().map(|n| regex.is_match(n)).collect()
        } else {
            names.iter().map(|n| n == spec).collect()
        };
        if !matches.contains(&true) {
            return Err(miette!(
                "--traces-channels {spec} matches no channel. The channels are: {}",
                names.join(", ")
            ));
        }
        for (c, m) in chosen.iter_mut().zip(matches) {
            *c |= m;
        }
    }
    Ok(chosen
        .iter()
        .enumerate()
        .filter(|(_, c)| **c)
        .map(|(i, _)| i)
        .collect())
}

/// Converts the traces to `u32`. The traces must hold exact integers: a value that is not an
/// integer, or is above 2^24 (an `f32` may have rounded it), or is above `u32::MAX`, is an error.
fn to_u32(traces: &Array2<f32>, what: &str) -> miette::Result<Array2<u32>> {
    let mut out = Array2::<u32>::zeros(traces.dim());
    for ((index, &value), cell) in traces.indexed_iter().zip(out.iter_mut()) {
        if !(value >= 0.0 && value.fract() == 0.0 && f64::from(value) <= u32::MAX as f64) {
            return Err(miette!(
                "{what}: the value {value} (segment {}, sample {}) is not an integer from 0 to \
                 {}",
                index.0,
                index.1,
                u32::MAX
            ));
        }
        if f64::from(value) > MAX_EXACT_COUNT as f64 {
            return Err(miette!(
                "{what}: the count {value} (segment {}, sample {}) is above {MAX_EXACT_COUNT} \
                 (2^24). An f32 trace may have rounded it, so the file would not hold the exact \
                 count. Select fewer signals or use a smaller clock period",
                index.0,
                index.1
            ));
        }
        *cell = value as u32;
    }
    Ok(out)
}

/// One channel of a batch.
pub struct ChannelData<'a> {
    /// The index of the channel in the plan (0 is the total).
    pub index: usize,
    pub identity: &'a ChannelIdentity,
    pub traces: &'a Array2<f32>,
}

/// What a traces file holds besides the traces.
pub struct BatchInfo<'a> {
    pub batch_id: &'a str,
    pub key: &'a CacheKey,
    /// The raw labels, before any shuffle.
    pub labels: &'a Array1<u16>,
    pub groups: &'a [u64],
    pub segment_ids: &'a [u64],
    pub shuffle_seed: Option<u64>,
    /// The bytes that the f32 traces of all channels of the batch hold now.
    pub held_bytes: u64,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Writes the traces file. It writes a temporary file in the same directory, then renames it.
pub fn write(
    path: &Path,
    channels: &[ChannelData<'_>],
    info: &BatchInfo<'_>,
) -> miette::Result<()> {
    let first = channels
        .first()
        .ok_or_else(|| miette!("there is no channel to write"))?;
    let (segments, samples) = first.traces.dim();
    // The u32 copies of the selected channels come on top of the f32 traces that the batch holds.
    let estimate = (segments as u64)
        .checked_mul(samples as u64)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_mul(channels.len() as u64))
        .and_then(|n| n.checked_add(info.held_bytes));
    let limit = PowerPlan::DEFAULT_MEMORY_LIMIT;
    match estimate {
        Some(n) if n <= limit => {}
        _ => {
            return Err(miette!(
                "the traces file needs about {} bytes (segments x samples x 4 bytes x channels, \
                 plus the traces in memory), more than the memory limit of {limit} bytes. Select \
                 fewer channels with --traces-channels",
                estimate.map_or("more than 2^64".into(), |n| n.to_string())
            ));
        }
    }
    let meta = serde_json::json!({
        "format": FORMAT,
        "kind": "scasim-traces",
        "scasim": env!("CARGO_PKG_VERSION"),
        "batch_id": info.batch_id,
        "segments": segments,
        "samples": samples,
        "shuffle_seed": info.shuffle_seed,
        "cache_key": info.key,
        "channels": channels.iter().map(|c| serde_json::json!({
            "name": c.identity.name,
            "index": c.index,
            "array": format!("t_{}", c.index),
            "is_total": c.identity.is_total,
            "handles": c.identity.handles,
            "handles_hash": hex(&c.identity.handles_hash),
        })).collect::<Vec<_>>(),
    });
    let meta_bytes = Array1::from(serde_json::to_vec(&meta).into_diagnostic()?);

    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let tmp = parent.join(format!(
        ".scasim-traces-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> miette::Result<()> {
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
            .into_diagnostic()?;
        let mut npz = NpzWriter::new_compressed(file);
        for channel in channels {
            let what = format!("channel {}", channel.identity.name);
            npz.add_array(
                format!("t_{}", channel.index),
                &to_u32(channel.traces, &what)?,
            )
            .into_diagnostic()?;
        }
        npz.add_array("labels", info.labels).into_diagnostic()?;
        npz.add_array("groups", &Array1::from(info.groups.to_vec()))
            .into_diagnostic()?;
        npz.add_array("segment_ids", &Array1::from(info.segment_ids.to_vec()))
            .into_diagnostic()?;
        npz.add_array("meta.json", &meta_bytes).into_diagnostic()?;
        let mut file = npz.finish().into_diagnostic()?;
        file.flush().into_diagnostic()?;
        file.sync_all().into_diagnostic()?;
        drop(file);
        std::fs::rename(&tmp, path).into_diagnostic()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.wrap_err_with(|| format!("cannot write the traces file {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn specs(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn channel_specs_select_by_name_or_whole_name_regex() {
        let names = ["total", "tb.dut.a", "tb.dut.ab"];
        assert_eq!(select_channels(&[], &names).unwrap(), [0, 1, 2]);
        assert_eq!(select_channels(&specs(&["tb.dut.a"]), &names).unwrap(), [1]);
        assert_eq!(
            select_channels(&specs(&["regex:tb\\.dut\\.a"]), &names).unwrap(),
            [1]
        );
        assert_eq!(
            select_channels(&specs(&["regex:tb.*", "total"]), &names).unwrap(),
            [0, 1, 2]
        );
        for bad in ["a", "regex:b", "regex:("] {
            assert!(select_channels(&specs(&[bad]), &names).is_err(), "{bad}");
        }
        let err = select_channels(&specs(&["total", "nope"]), &names).unwrap_err();
        assert!(err.to_string().contains("nope"), "{err}");
    }

    #[test]
    fn traces_convert_only_when_they_are_exact() {
        let exact = array![[0.0_f32, 1.0], [16777216.0, 5.0]];
        assert_eq!(
            to_u32(&exact, "c").unwrap(),
            array![[0u32, 1], [16777216, 5]]
        );
        for bad in [
            0.5_f32,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            16777218.0,
            5.0e9,
            1.0e20,
        ] {
            assert!(to_u32(&array![[bad]], "c").is_err(), "{bad}");
        }
    }
}
