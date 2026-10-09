//! Versioned batch histograms. Batch provenance differs; preprocessing must match.

use miette::{IntoDiagnostic, WrapErr, miette};
use scasim::batch::{EdgeReport, LengthPolicy};
use scasim::stats::{Binning, HistAccumulator};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const FORMAT: u32 = 1;
const ALGORITHM: u32 = 1;
const MAGIC: &[u8; 8] = b"SCASTATS";

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    pub sampling: String,
    pub clock: Option<String>,
    pub edges: String,
    pub offset: i64,
    pub rules: Vec<String>,
    pub per_scope: Option<String>,
    pub depth: usize,
    pub policy: LengthPolicy,
    pub shuffle_seed: Option<u64>,
    pub unknown: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            sampling: "legacy".into(),
            clock: None,
            edges: "rising".into(),
            offset: 0,
            rules: Vec::new(),
            per_scope: None,
            depth: 1,
            policy: LengthPolicy::Pad,
            shuffle_seed: None,
            unknown: "half".into(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommonKey {
    pub format: u32,
    pub algorithm: u32,
    pub scasim: String,
    pub settings: Settings,
}
impl Default for CommonKey {
    fn default() -> Self {
        Self {
            format: FORMAT,
            algorithm: ALGORITHM,
            scasim: env!("CARGO_PKG_VERSION").into(),
            settings: Settings::default(),
        }
    }
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchKey {
    pub metadata_sha256: [u8; 32],
    pub waveform_size: u64,
    pub waveform_mtime_seconds: u64,
    pub waveform_mtime_nanos: u32,
    pub waveform_sha256: [u8; 32],
    pub waveform_time: Option<(u64, i32)>,
    pub legacy_clock_period: Option<u64>,
}
impl BatchKey {
    pub fn new(meta: &scasim::batch::BatchMeta) -> miette::Result<Self> {
        // V1 unknown fields and extensions do not affect the current analyses.
        let content = if let Some(v1) = &meta.v1 {
            serde_json::json!({"segments":v1.segments,"time":v1.time,"labels":v1.labels,"groups":v1.groups,"seeds":v1.batch.seeds,"design":v1.design,"design_random":v1.batch.design_random})
        } else {
            serde_json::json!({"markers":meta.markers,"clock_period":meta.clock_period})
        };
        let metadata_sha256 = digest(&serde_json::to_vec(&content).into_diagnostic()?);
        let before = std::fs::metadata(&meta.trace_path).into_diagnostic()?;
        let modified = before
            .modified()
            .into_diagnostic()?
            .duration_since(std::time::UNIX_EPOCH)
            .into_diagnostic()?;
        let mut hasher = Sha256::new();
        let mut file = std::fs::File::open(&meta.trace_path).into_diagnostic()?;
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer).into_diagnostic()?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let after = file.metadata().into_diagnostic()?;
        if before.len() != after.len()
            || before.modified().into_diagnostic()? != after.modified().into_diagnostic()?
        {
            return Err(miette!(
                "{} changed while hashing",
                meta.trace_path.display()
            ));
        }
        let time = scasim::metadata::waveform_time_unit(&meta.trace_path)?;
        Ok(Self {
            metadata_sha256,
            waveform_size: before.len(),
            waveform_mtime_seconds: modified.as_secs(),
            waveform_mtime_nanos: modified.subsec_nanos(),
            waveform_sha256: hasher.finalize().into(),
            waveform_time: Some((time.mantissa, time.exponent)),
            legacy_clock_period: meta.clock_period,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub common: CommonKey,
    pub batch: BatchKey,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChannelIdentity {
    pub name: String,
    pub handles_hash: [u8; 32],
    pub handles: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub identity: ChannelIdentity,
    pub groups: BTreeMap<u64, HistAccumulator>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleAxis {
    pub relative_to_segment: bool,
    pub length: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cache {
    pub key: CacheKey,
    pub batch_id: String,
    pub axis: SampleAxis,
    pub counts: BTreeMap<u64, BTreeMap<u16, u64>>,
    pub labels: BTreeMap<u16, String>,
    pub groups: BTreeMap<u64, String>,
    pub channels: Vec<Channel>,
    // JSON is kept as text because postcard does not support deserialize_any.
    pub extensions: String,
    pub edges: Option<EdgeReport>,
    pub aliased: usize,
    pub outside: usize,
}
impl Cache {
    pub fn validate(&self) -> miette::Result<()> {
        if self.key.common.format != FORMAT
            || self.key.common.algorithm != ALGORITHM
            || self.key.common.scasim != env!("CARGO_PKG_VERSION")
        {
            return Err(miette!("unsupported statistics cache version"));
        }
        if self.batch_id.is_empty() || !self.axis.relative_to_segment || self.channels.is_empty() {
            return Err(miette!("invalid batch id, sample axis, or channel list"));
        }
        let mut names = BTreeSet::new();
        if self
            .channels
            .iter()
            .filter(|c| c.identity.name == "total")
            .count()
            != 1
        {
            return Err(miette!("the cache needs one total channel"));
        }
        for (g, counts) in &self.counts {
            if !self.groups.contains_key(g)
                || counts.is_empty()
                || counts.iter().any(|(l, &n)| {
                    !self.labels.contains_key(l) || n == 0 || n > u64::from(u32::MAX)
                })
            {
                return Err(miette!("invalid group or label counts"));
            }
        }
        for channel in &self.channels {
            if !names.insert(&channel.identity.name)
                || channel.identity.handles == 0
                || channel.groups.keys().ne(self.counts.keys())
            {
                return Err(miette!("invalid channel identity or group list"));
            }
            for (g, h) in &channel.groups {
                h.validate().into_diagnostic()?;
                if h.binning() != Binning::Exact
                    || h.rejected() != 0
                    || h.n_samples() != self.axis.length
                    || h.labels()
                        .iter()
                        .copied()
                        .ne(self.counts[g].keys().copied())
                    || h.labels()
                        .iter()
                        .any(|l| h.class_count(*l) != self.counts[g][l])
                {
                    return Err(miette!(
                        "channel {} has inconsistent histogram counts or samples",
                        channel.identity.name
                    ));
                }
            }
        }
        let _: serde_json::Value = serde_json::from_str(&self.extensions).into_diagnostic()?;
        Ok(())
    }

    pub fn write(&self, path: &Path) -> miette::Result<()> {
        self.validate()?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let tmp = parent.join(format!(
            ".scasim-stats-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| {
            let payload = postcard::to_stdvec(self).into_diagnostic()?;
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&tmp)
                .into_diagnostic()?;
            file.write_all(MAGIC).into_diagnostic()?;
            file.write_all(&digest(&payload)).into_diagnostic()?;
            file.write_all(&payload).into_diagnostic()?;
            file.sync_all().into_diagnostic()?;
            drop(file);
            let restored = read(&tmp)?;
            if postcard::to_stdvec(&restored).into_diagnostic()? != payload {
                return Err(miette!("statistics cache failed read-back validation"));
            }
            std::fs::rename(&tmp, path).into_diagnostic()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result.wrap_err_with(|| format!("cannot write statistics cache {}", path.display()))
    }

    /// Builds the histogram for one selected group or an explicit pool.
    pub fn selected(
        &self,
        index: usize,
        group: Option<u64>,
        pool: bool,
    ) -> miette::Result<HistAccumulator> {
        if group.is_none() && !pool && self.groups.len() > 1 {
            return Err(miette!(
                "more than one group exists; use --group G or --pool-groups"
            ));
        }
        let chosen = group.or_else(|| {
            (!pool)
                .then(|| self.groups.keys().next().copied())
                .flatten()
        });
        let mut hist = HistAccumulator::new(self.axis.length, Binning::Exact);
        for (g, h) in &self.channels[index].groups {
            if chosen.is_none_or(|c| c == *g) {
                hist.merge(h).into_diagnostic()?;
            }
        }
        Ok(hist)
    }
}

pub fn read(path: &Path) -> miette::Result<Cache> {
    let result = (|| {
        let bytes = std::fs::read(path).into_diagnostic()?;
        if bytes.len() < 40 || &bytes[..8] != MAGIC || bytes[8..40] != digest(&bytes[40..]) {
            return Err(miette!("invalid statistics cache header or checksum"));
        }
        let (cache, remaining): (Cache, &[u8]) =
            postcard::take_from_bytes(&bytes[40..]).into_diagnostic()?;
        if !remaining.is_empty() {
            return Err(miette!("trailing statistics cache data"));
        }
        cache.validate()?;
        Ok(cache)
    })();
    result.wrap_err_with(|| format!("cannot read statistics cache {}", path.display()))
}

#[derive(Default)]
pub struct MergeState {
    pub cache: Option<Cache>,
    seen: BTreeSet<String>,
}
impl MergeState {
    pub fn add(&mut self, other: Cache) -> miette::Result<()> {
        other.validate()?;
        if self.seen.contains(&other.batch_id) {
            return Err(miette!("duplicate batch id {}", other.batch_id));
        }
        if let Some(mine) = self.cache.as_ref() {
            if mine.key.common != other.key.common {
                return Err(miette!(
                    "statistics cache key mismatch (preprocessing or version)"
                ));
            }
            if mine.key.batch.waveform_time != other.key.batch.waveform_time
                || mine.key.batch.legacy_clock_period != other.key.batch.legacy_clock_period
            {
                return Err(miette!("statistics cache sample time mismatch"));
            }
            let identities = |c: &Cache| {
                c.channels
                    .iter()
                    .map(|c| c.identity.clone())
                    .collect::<BTreeSet<_>>()
            };
            if identities(mine) != identities(&other) {
                return Err(miette!("statistics cache channel identities differ"));
            }
            for (id, name) in &other.labels {
                if mine.labels.get(id).is_some_and(|n| n != name) {
                    return Err(miette!("label {id} has different names"));
                }
            }
            for (id, name) in &other.groups {
                if mine.groups.get(id).is_some_and(|n| n != name) {
                    return Err(miette!("group {id} has different names"));
                }
            }
            let n = match mine.key.common.settings.policy {
                LengthPolicy::Pad => mine.axis.length.max(other.axis.length),
                LengthPolicy::Truncate => mine.axis.length.min(other.axis.length),
                LengthPolicy::Error if mine.axis.length == other.axis.length => mine.axis.length,
                LengthPolicy::Error => {
                    return Err(miette!(
                        "statistics cache sample lengths differ; use --length-policy pad or truncate when writing the caches"
                    ));
                }
            };
            // Work on a copy. An overflow or validation error leaves the run unchanged.
            let mut next = mine.clone();
            let mut other = other;
            for cache in [&mut next, &mut other] {
                for channel in &mut cache.channels {
                    for h in channel.groups.values_mut() {
                        if h.n_samples() < n {
                            h.pad_samples(n).into_diagnostic()?;
                        } else {
                            h.truncate_samples(n).into_diagnostic()?;
                        }
                    }
                }
                cache.axis.length = n;
            }
            for (g, counts) in &other.counts {
                for (l, count) in counts {
                    let dest = next.counts.entry(*g).or_default().entry(*l).or_default();
                    *dest = dest
                        .checked_add(*count)
                        .filter(|&n| n <= u64::from(u32::MAX))
                        .ok_or_else(|| miette!("group {g} label {l}: count overflow"))?;
                }
            }
            next.labels.extend(other.labels);
            next.groups.extend(other.groups);
            for channel in &mut next.channels {
                let incoming = other
                    .channels
                    .iter()
                    .find(|c| c.identity == channel.identity)
                    .expect("channel identities were checked");
                for (g, h) in &incoming.groups {
                    if let Some(mine) = channel.groups.get_mut(g) {
                        mine.merge(h).into_diagnostic()?;
                    } else {
                        channel.groups.insert(*g, h.clone());
                    }
                }
            }
            next.validate()?;
            self.cache = Some(next);
            self.seen.insert(other.batch_id);
        } else {
            self.seen.insert(other.batch_id.clone());
            self.cache = Some(other);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    fn cache(id: &str, samples: usize, sparse: bool) -> Cache {
        let mut h = HistAccumulator::with_max_dense_bins(
            samples,
            Binning::Exact,
            if sparse { 0 } else { 4096 },
        );
        let data = ndarray::Array2::from_shape_fn((4, samples), |(i, j)| ((i * 3 + j) % 7) as u32);
        h.update(data.view(), array![0u16, 1, 0, 1].view()).unwrap();
        let channel = Channel {
            identity: ChannelIdentity {
                name: "total".into(),
                handles_hash: [7; 32],
                handles: 2,
            },
            groups: BTreeMap::from([(0, h)]),
        };
        Cache {
            key: CacheKey {
                common: CommonKey::default(),
                batch: BatchKey::default(),
            },
            batch_id: id.into(),
            axis: SampleAxis {
                relative_to_segment: true,
                length: samples,
            },
            counts: BTreeMap::from([(0, BTreeMap::from([(0, 2), (1, 2)]))]),
            labels: BTreeMap::from([(0, "a".into()), (1, "b".into())]),
            groups: BTreeMap::from([(0, "default".into())]),
            channels: vec![channel],
            extensions: "{}".into(),
            edges: None,
            aliased: 0,
            outside: 0,
        }
    }
    #[test]
    fn merge_lengths_dense_sparse_and_duplicates() {
        for (policy, n, accept) in [
            (LengthPolicy::Pad, 4, true),
            (LengthPolicy::Truncate, 2, true),
            (LengthPolicy::Error, 2, false),
        ] {
            let mut a = cache("a", 2, false);
            let mut b = cache("b", 4, true);
            a.key.common.settings.policy = policy;
            b.key.common.settings.policy = policy;
            let mut run = MergeState::default();
            run.add(a).unwrap();
            assert_eq!(run.add(b).is_ok(), accept);
            if accept {
                let c = run.cache.as_ref().unwrap();
                assert_eq!(c.axis.length, n);
                c.validate().unwrap();
                assert_eq!(c.counts[&0][&0], 4);
            }
            assert!(run.add(cache("a", 2, false)).is_err());
        }
    }
    #[test]
    fn merge_matches_channels_by_identity_and_checks_names_and_settings() {
        let mut a = cache("a", 2, false);
        let mut b = cache("b", 2, true);
        for c in [&mut a, &mut b] {
            let mut second = c.channels[0].clone();
            second.identity.name = "scope".into();
            c.channels.push(second);
        }
        b.channels.reverse();
        let mut run = MergeState::default();
        run.add(a).unwrap();
        run.add(b.clone()).unwrap();
        assert_eq!(
            run.cache.as_ref().unwrap().channels[0].groups[&0].class_count(0),
            4
        );
        b.batch_id = "c".into();
        b.labels.insert(0, "different".into());
        assert!(run.add(b.clone()).is_err());
        b.labels.insert(0, "a".into());
        b.key.common.settings.offset = 1;
        assert!(run.add(b).is_err());
    }
    #[test]
    fn count_overflow_leaves_merge_state_unchanged() {
        let mut c = cache("a", 0, false);
        let h = &c.channels[0].groups[&0];
        let mut json = serde_json::to_value(h).unwrap();
        json["class_counts"] = serde_json::json!([u32::MAX, 2]);
        c.channels[0]
            .groups
            .insert(0, serde_json::from_value(json).unwrap());
        c.counts.get_mut(&0).unwrap().insert(0, u64::from(u32::MAX));
        let mut run = MergeState::default();
        run.add(c).unwrap();
        assert!(run.add(cache("b", 0, false)).is_err());
        assert_eq!(
            run.cache.as_ref().unwrap().counts[&0][&0],
            u64::from(u32::MAX)
        );
    }

    #[test]
    fn corrupt_cache_and_atomic_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch.stats");
        let c = cache("a", 2, false);
        c.write(&path).unwrap();
        read(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(read(&path).is_err());
    }
}
