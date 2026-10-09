//! Streaming per-sample, per-class histograms of a channel.
//!
//! A channel is a set of traces with the same number of samples. For every sample (a column of
//! the trace matrix) and every class (a label), the accumulator counts how often each value bin
//! occurs. These counts are the contingency tables for the chi-squared and G tests, and they are
//! also sufficient for every moment-based statistic of the binned values.
//!
//! # Layout
//!
//! Each sample owns one histogram. A histogram is *dense* or *sparse*:
//!
//! * **Dense**: a window of consecutive bins `[base, base + width)` and one row of `width` `u32`
//!   counters per class (`counts[class * width + bin - base]`). The window grows on demand, with
//!   at most [`DEFAULT_MAX_DENSE_BINS`] bins by default. The memory is
//!   `4 * classes * width` bytes per sample. A hot update is one range check and one increment.
//! * **Sparse**: a hash map from `(bin, class)` to a counter. A histogram becomes sparse when its
//!   window would exceed the limit. Updates are slower (a hash lookup), and an entry costs about
//!   30 bytes, but only occupied bins use memory.
//!
//! Counters are `u32`. The accumulator enforces that no class has more than `u32::MAX` traces, so
//! a counter cannot overflow. Class totals and all sums in the tests use `u64`.
//!
//! # Class slots
//!
//! The classes are the distinct labels. The accumulator keeps them in increasing order of label:
//! the class with the smallest label has slot 0. A new label can arrive between two known labels.
//! Then the slots after it move up by one, and the accumulator re-lays out each histogram. A
//! dense histogram gets a new row of zeros, and a sparse histogram changes the slot in its keys.
//! This happens only when a batch brings a label that the accumulator has not seen. The state
//! depends only on the counts, not on the order of the batches. A lookup of a label is a binary
//! search.
//!
//! # Parallelism and determinism
//!
//! `update` splits the samples into blocks. Every block is processed by one thread, which loops
//! over all traces and updates the histograms of its samples. Reading a block of consecutive
//! samples from a row-major trace matrix is cache-friendly, and no two threads write to the same
//! histogram. The counts do not depend on the number of threads.
//!
//! Histograms merge by exact integer addition, so the merge of two accumulators has the same
//! counts as one pass over all traces, in any order and with any batch split. This is unlike the
//! floating-point moments of a t-test, whose merged result depends on the order of operations.
//! Test results are computed from the bins in increasing order of value, independent of the
//! internal layout, so equal counts give bit-identical test results.
//!
//! # Rejected values
//!
//! A value that the binning rule cannot bin is *rejected*: it is not counted in any bin, and
//! [`HistAccumulator::rejected`] counts it. The trace that holds the value still counts in the
//! total of its class, because its other samples are valid. So a class total can be larger than
//! the sum of the bins of one sample. A caller that cannot accept rejected values (like `tvla`)
//! must check `rejected() > 0` and treat it as an error.
//!
//! # Saved state
//!
//! The type implements `Serialize` and `Deserialize`. Deserialization calls
//! [`HistAccumulator::validate`], so a malformed saved state is an error and never an
//! accumulator: no method can receive an inconsistent state. (`validate` stays public for
//! states that you build in another way.)
//!
//! Call [`HistAccumulator::compact`] before saving. The window of a dense histogram depends on
//! the order in which the values arrived. After `compact`, it is exactly the occupied range. A
//! compacted state depends only on the counts, so two accumulators with the same counts give
//! the same bytes, whatever the order of the batches.

use std::collections::HashMap;

use ndarray::{ArrayView1, ArrayView2, s};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use serde::{Deserialize, Serialize};

use super::binning::{BinValue, Binning, fixed_bin};
use super::chi2::{TestOptions, TestResult, Workspace, test_table};
use super::error::StatsError;

/// Default limit on the number of bins in a dense window.
pub const DEFAULT_MAX_DENSE_BINS: usize = 4096;

/// Largest value of `max_dense_bins`. A larger limit is lowered to this value.
pub const MAX_DENSE_BINS_LIMIT: usize = 1 << 20;

/// Largest number of counters (`slots * width`) of one dense histogram: 2^28 counters, 1 GiB.
/// A histogram that would need more becomes sparse.
pub const MAX_DENSE_COUNTERS: usize = 1 << 28;

/// Smallest window allocated for a dense histogram.
const MIN_DENSE_WIDTH: usize = 8;

/// Number of consecutive samples processed together by one thread.
const BLOCK: usize = 32;

/// Largest bin magnitude that the fixed-width rule can produce (2^53).
const MAX_FIXED_BIN: i64 = 1 << 53;

/// One sparse entry: `(bin, class slot, count)`.
type SparseEntry = (i64, u16, u32);

fn invalid<T>(msg: impl Into<String>) -> Result<T, StatsError> {
    Err(StatsError::InvalidState(msg.into()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Dense {
    /// Bin index of the first column of the window.
    base: i64,
    /// Number of columns in the window. Zero while the histogram is empty.
    width: usize,
    /// Number of class rows that are allocated. It equals the number of labels.
    slots: usize,
    /// `slots * width` counters, class-major.
    counts: Vec<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(try_from = "Vec<SparseEntry>", into = "Vec<SparseEntry>")]
struct Sparse {
    map: FxHashMap<(i64, u16), u32>,
}

impl TryFrom<Vec<SparseEntry>> for Sparse {
    type Error = &'static str;

    /// Accepts only the canonical form: entries in strictly increasing `(bin, slot)` order, and
    /// no zero count. This is the form that `Into<Vec<SparseEntry>>` writes.
    fn try_from(entries: Vec<SparseEntry>) -> Result<Self, Self::Error> {
        let mut map: FxHashMap<(i64, u16), u32> =
            HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        let mut last = None;
        for (bin, slot, count) in entries {
            if count == 0 {
                return Err("sparse histogram: zero count");
            }
            if last.is_some_and(|l| l >= (bin, slot)) {
                return Err("sparse histogram: entries not in increasing order");
            }
            last = Some((bin, slot));
            map.insert((bin, slot), count);
        }
        Ok(Self { map })
    }
}

impl From<Sparse> for Vec<SparseEntry> {
    /// Entries in increasing `(bin, class)` order, so the serialized form is deterministic.
    fn from(sparse: Sparse) -> Self {
        let mut entries: Vec<SparseEntry> = sparse
            .map
            .into_iter()
            .map(|((bin, slot), count)| (bin, slot, count))
            .collect();
        entries.sort_unstable();
        entries
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum SampleHist {
    Dense(Dense),
    Sparse(Sparse),
}

impl SampleHist {
    fn empty() -> Self {
        Self::Dense(Dense {
            base: 0,
            width: 0,
            slots: 0,
            counts: Vec::new(),
        })
    }

    /// Makes room for `slots` class rows at the end. A dense histogram that would need more than
    /// `MAX_DENSE_COUNTERS` counters becomes sparse.
    fn ensure_slots(&mut self, slots: usize) {
        if let Self::Dense(d) = self
            && d.slots < slots
        {
            if slots.saturating_mul(d.width) > MAX_DENSE_COUNTERS {
                self.make_sparse();
            } else {
                d.counts.resize(slots * d.width, 0);
                d.slots = slots;
            }
        }
    }

    /// Converts a dense histogram into a sparse one with the same counts.
    fn make_sparse(&mut self) {
        let mut sparse = Sparse::default();
        self.for_each_nonzero(|s, b, c| {
            sparse.map.insert((b, s as u16), c);
        });
        *self = Self::Sparse(sparse);
    }

    /// Moves the class rows to their new slots. `old_to_new[i]` is the new slot of the old slot
    /// `i`, and `new_slots` is the new number of slots. The new slots that no old slot maps to
    /// are empty. The size of a new dense layout is checked before it is allocated.
    fn relabel(&mut self, old_to_new: &[u16], new_slots: usize) {
        let keeps_order = old_to_new
            .iter()
            .enumerate()
            .all(|(i, &n)| usize::from(n) == i);
        if keeps_order {
            self.ensure_slots(new_slots);
            return;
        }
        // Check the size of the new layout before any allocation. A dense histogram that would
        // need too many counters becomes sparse first, and its slots are then re-keyed.
        if let Self::Dense(d) = self
            && new_slots.saturating_mul(d.width) > MAX_DENSE_COUNTERS
        {
            self.make_sparse();
        }
        match self {
            Self::Dense(d) => {
                let mut counts = vec![0_u32; new_slots * d.width];
                for (old, &new) in old_to_new.iter().enumerate() {
                    let new = usize::from(new);
                    counts[new * d.width..(new + 1) * d.width]
                        .copy_from_slice(&d.counts[old * d.width..(old + 1) * d.width]);
                }
                d.slots = new_slots;
                d.counts = counts;
            }
            Self::Sparse(sp) => {
                let mut map: FxHashMap<(i64, u16), u32> =
                    HashMap::with_capacity_and_hasher(sp.map.len(), FxBuildHasher);
                for (&(bin, slot), &count) in &sp.map {
                    map.insert((bin, old_to_new[usize::from(slot)]), count);
                }
                sp.map = map;
            }
        }
    }

    /// Counts one occurrence. The class row `slot` must exist (see `ensure_slots`).
    #[inline(always)]
    fn add(&mut self, slot: usize, bin: i64, max_dense: usize) {
        if let Self::Dense(d) = self {
            // Wrapping subtraction and the unsigned compare also reject `bin < base`.
            let offset = bin.wrapping_sub(d.base) as u64;
            if offset < d.width as u64 {
                d.counts[slot * d.width + offset as usize] += 1;
                return;
            }
        }
        self.add_count(slot, bin, 1, max_dense);
    }

    /// Adds `count` occurrences; grows or converts the histogram when needed.
    fn add_count(&mut self, slot: usize, bin: i64, count: u32, max_dense: usize) {
        match self {
            Self::Dense(d) => {
                let offset = bin.wrapping_sub(d.base) as u64;
                if offset < d.width as u64 {
                    d.counts[slot * d.width + offset as usize] += count;
                    return;
                }
                // Size the new window from the occupied range, not from the old window. The
                // old window includes spare space, and counting it would make the window
                // double on every extension.
                let occupied = d.occupied();
                let (lo, hi) = occupied.map_or((bin, bin), |(a, b)| (a.min(bin), b.max(bin)));
                let span = i128::from(hi) - i128::from(lo) + 1;
                // The window must not be wider than the limit, and must not need too many counters.
                let too_big = span > max_dense as i128
                    || d.slots
                        .saturating_mul(Dense::width_for(span as usize, max_dense))
                        > MAX_DENSE_COUNTERS;
                if too_big {
                    self.make_sparse();
                    self.add_count(slot, bin, count, max_dense);
                } else {
                    d.relayout(occupied, lo, hi, max_dense);
                    d.counts[slot * d.width + (bin - d.base) as usize] += count;
                }
            }
            Self::Sparse(sp) => {
                *sp.map.entry((bin, slot as u16)).or_insert(0) += count;
            }
        }
    }

    /// Calls `f(class slot, bin, count)` for every non-zero counter, in no particular order.
    fn for_each_nonzero(&self, mut f: impl FnMut(usize, i64, u32)) {
        match self {
            Self::Dense(d) => {
                if d.width == 0 {
                    return;
                }
                for (slot, row) in d.counts.chunks_exact(d.width).enumerate() {
                    for (i, &c) in row.iter().enumerate() {
                        if c > 0 {
                            f(slot, d.base + i as i64, c);
                        }
                    }
                }
            }
            Self::Sparse(sp) => {
                for (&(bin, slot), &c) in &sp.map {
                    f(usize::from(slot), bin, c);
                }
            }
        }
    }

    /// Non-zero `(bin, count)` pairs of one class row, in increasing bin order.
    fn row(&self, slot: usize) -> Vec<(i64, u32)> {
        let mut out = Vec::new();
        self.for_each_nonzero(|s, b, c| {
            if s == slot {
                out.push((b, c));
            }
        });
        out.sort_unstable();
        out
    }

    /// Approximate heap and inline size in bytes.
    fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Dense(d) => d.counts.capacity() * std::mem::size_of::<u32>(),
                Self::Sparse(sp) => {
                    sp.map.capacity() * (std::mem::size_of::<((i64, u16), u32)>() + 1)
                }
            }
    }

    /// Shrinks a dense window to the occupied bins and releases spare capacity.
    fn compact(&mut self) {
        match self {
            Self::Dense(_) => {
                let (mut lo, mut hi) = (i64::MAX, i64::MIN);
                self.for_each_nonzero(|_, b, _| {
                    lo = lo.min(b);
                    hi = hi.max(b);
                });
                let Self::Dense(d) = self else { unreachable!() };
                if lo > hi {
                    *d = Dense {
                        base: 0,
                        width: 0,
                        slots: d.slots,
                        counts: Vec::new(),
                    };
                } else {
                    let new_width = (hi - lo + 1) as usize;
                    let mut counts = vec![0_u32; d.slots * new_width];
                    for slot in 0..d.slots {
                        let src = &d.counts[slot * d.width + (lo - d.base) as usize..][..new_width];
                        counts[slot * new_width..(slot + 1) * new_width].copy_from_slice(src);
                    }
                    d.base = lo;
                    d.width = new_width;
                    d.counts = counts;
                }
            }
            Self::Sparse(sp) => sp.map.shrink_to_fit(),
        }
    }

    /// Checks the layout against the number of labels, and adds the counts of each class slot
    /// to `totals`. Also checks that the bins are in the range of `binning`.
    fn check(
        &self,
        labels: usize,
        max_dense: usize,
        binning: Binning,
        totals: &mut [u64],
    ) -> Result<(), String> {
        match self {
            Self::Dense(d) => {
                if d.slots != labels {
                    return Err(format!("{} class rows for {labels} labels", d.slots));
                }
                match d.slots.checked_mul(d.width) {
                    Some(n) if n == d.counts.len() && n <= MAX_DENSE_COUNTERS => {}
                    _ => return Err("the counters do not fit the window".into()),
                }
                if d.width == 0 && d.base != 0 {
                    return Err("an empty window has a base".into());
                }
                if d.slots == 0 && d.width != 0 {
                    return Err("a window without class rows has columns".into());
                }
                if d.width > max_dense {
                    return Err(format!(
                        "window of {} bins is wider than the limit {max_dense}",
                        d.width
                    ));
                }
                if i128::from(d.base) + d.width as i128 > i128::from(i64::MAX) + 1 {
                    return Err("window goes past the largest bin".into());
                }
            }
            Self::Sparse(sp) => {
                if sp.map.keys().any(|&(_, s)| usize::from(s) >= labels) {
                    return Err("entry with an unknown class".into());
                }
            }
        }
        let mut bad_bin = None;
        self.for_each_nonzero(|slot, bin, count| {
            totals[slot] += u64::from(count);
            if matches!(binning, Binning::Fixed { .. })
                && bin.unsigned_abs() >= MAX_FIXED_BIN as u64
            {
                bad_bin = Some(bin);
            }
        });
        match bad_bin {
            Some(bin) => Err(format!("bin {bin} is out of range for fixed-width bins")),
            None => Ok(()),
        }
    }

    /// Calls `f` with one row per requested class slot, all of equal length, bins increasing.
    /// Returns an error if a slot does not exist in this histogram.
    fn with_rows<R>(
        &self,
        slots: &[usize],
        scratch: &mut Vec<u32>,
        f: impl FnOnce(&[&[u32]]) -> R,
    ) -> Result<R, StatsError> {
        match self {
            Self::Dense(d) => {
                let rows = slots
                    .iter()
                    .map(|&s| {
                        let start = s.checked_mul(d.width)?;
                        d.counts.get(start..start.checked_add(d.width)?)
                    })
                    .collect::<Option<Vec<&[u32]>>>();
                match rows {
                    Some(rows) => Ok(f(&rows)),
                    None => invalid("a histogram has fewer class rows than labels"),
                }
            }
            Self::Sparse(sp) => {
                let mut entries: Vec<(i64, usize, u32)> = sp
                    .map
                    .iter()
                    .filter_map(|(&(bin, slot), &c)| {
                        slots
                            .iter()
                            .position(|&s| s == usize::from(slot))
                            .map(|k| (bin, k, c))
                    })
                    .collect();
                entries.sort_unstable_by_key(|&(bin, k, _)| (bin, k));
                let mut columns = 0;
                let mut last = None;
                for &(bin, _, _) in &entries {
                    if last != Some(bin) {
                        columns += 1;
                        last = Some(bin);
                    }
                }
                scratch.clear();
                scratch.resize(slots.len() * columns, 0);
                let mut column = 0;
                let mut last = None;
                for &(bin, k, c) in &entries {
                    if last.is_some_and(|b| b != bin) {
                        column += 1;
                    }
                    last = Some(bin);
                    scratch[k * columns + column] = c;
                }
                let rows: Vec<&[u32]> = (0..slots.len())
                    .map(|k| &scratch[k * columns..(k + 1) * columns])
                    .collect();
                Ok(f(&rows))
            }
        }
    }
}

impl Dense {
    /// Smallest and largest bin that has a non-zero count in any class.
    fn occupied(&self) -> Option<(i64, i64)> {
        let (mut first, mut last) = (usize::MAX, 0);
        for slot in 0..self.slots {
            let row = &self.counts[slot * self.width..(slot + 1) * self.width];
            if let Some(a) = row.iter().position(|&c| c > 0) {
                first = first.min(a);
                last = last.max(row.iter().rposition(|&c| c > 0).unwrap_or(a));
            }
        }
        (first != usize::MAX).then(|| (self.base + first as i64, self.base + last as i64))
    }

    /// Width of a window for a span of `span` bins: the next power of two (at least
    /// `MIN_DENSE_WIDTH`, at most `max_dense`, and at least the span).
    fn width_for(span: usize, max_dense: usize) -> usize {
        span.next_power_of_two()
            .max(MIN_DENSE_WIDTH)
            .min(max_dense)
            .max(span)
    }

    /// Re-lays out the window so that it covers `[lo, hi]`, which contains the `occupied` range.
    /// The width is the next power of two (at least `MIN_DENSE_WIDTH`, at most `max_dense`, and
    /// at least the span). The spare space is split evenly below and above, so repeated
    /// extensions in either direction need only a logarithmic number of re-layouts.
    fn relayout(&mut self, occupied: Option<(i64, i64)>, lo: i64, hi: i64, max_dense: usize) {
        let span = (i128::from(hi) - i128::from(lo) + 1) as usize;
        let new_width = Self::width_for(span, max_dense);
        let base = i128::from(lo) - ((new_width - span) / 2) as i128;
        let base = base.clamp(
            i128::from(i64::MIN),
            i128::from(i64::MAX) - (new_width as i128 - 1),
        ) as i64;
        let mut counts = vec![0_u32; self.slots * new_width];
        if let Some((a, b)) = occupied {
            let len = (b - a + 1) as usize;
            let src = (a - self.base) as usize;
            let dst = (a - base) as usize;
            for slot in 0..self.slots {
                counts[slot * new_width + dst..][..len]
                    .copy_from_slice(&self.counts[slot * self.width + src..][..len]);
            }
        }
        self.base = base;
        self.width = new_width;
        self.counts = counts;
    }
}

/// The class labels of an accumulator after a batch or a merge is added.
#[derive(Debug)]
struct LabelPlan {
    /// All labels in increasing order.
    labels: Vec<u16>,
    /// The trace count of each label.
    class_counts: Vec<u64>,
    /// New slot of each old slot.
    old_to_new: Vec<u16>,
}

/// Streaming histograms of one channel: counts per sample, class, and value bin.
///
/// See the [module documentation](self) for the layout, and [`Binning`] for the binning rules.
/// The type implements `Serialize` and `Deserialize`; call [`compact`](Self::compact) before
/// saving to drop unused window space, and [`validate`](Self::validate) after loading.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "HistAccumulatorRaw")]
pub struct HistAccumulator {
    binning: Binning,
    max_dense_bins: usize,
    n_samples: usize,
    /// The class labels in strictly increasing order. The position is the class slot.
    labels: Vec<u16>,
    /// Number of traces seen for each class slot.
    class_counts: Vec<u64>,
    /// Number of sample values that could not be binned.
    rejected: u64,
    hists: Vec<SampleHist>,
}

/// The fields of [`HistAccumulator`] as they are saved. Deserialization reads this type and then
/// calls [`HistAccumulator::validate`], so a malformed state is an error, never an accumulator.
#[derive(Deserialize)]
struct HistAccumulatorRaw {
    binning: Binning,
    max_dense_bins: usize,
    n_samples: usize,
    labels: Vec<u16>,
    class_counts: Vec<u64>,
    rejected: u64,
    hists: Vec<SampleHist>,
}

impl TryFrom<HistAccumulatorRaw> for HistAccumulator {
    type Error = StatsError;

    fn try_from(raw: HistAccumulatorRaw) -> Result<Self, StatsError> {
        let acc = Self {
            binning: raw.binning,
            max_dense_bins: raw.max_dense_bins,
            n_samples: raw.n_samples,
            labels: raw.labels,
            class_counts: raw.class_counts,
            rejected: raw.rejected,
            hists: raw.hists,
        };
        acc.validate()?;
        Ok(acc)
    }
}

impl HistAccumulator {
    /// Creates an empty accumulator for traces with `n_samples` samples.
    pub fn new(n_samples: usize, binning: Binning) -> Self {
        Self::with_max_dense_bins(n_samples, binning, DEFAULT_MAX_DENSE_BINS)
    }

    /// Like [`new`](Self::new), with a custom limit for the width of a dense window. A histogram
    /// whose value range exceeds the limit becomes sparse. A limit of 0 makes all histograms
    /// sparse. A limit above [`MAX_DENSE_BINS_LIMIT`] is lowered to it. A histogram also becomes
    /// sparse if its dense layout would need more than [`MAX_DENSE_COUNTERS`] counters (many
    /// classes with a wide window).
    pub fn with_max_dense_bins(n_samples: usize, binning: Binning, max_dense_bins: usize) -> Self {
        Self {
            binning,
            max_dense_bins: max_dense_bins.min(MAX_DENSE_BINS_LIMIT),
            n_samples,
            labels: Vec::new(),
            class_counts: Vec::new(),
            rejected: 0,
            hists: (0..n_samples).map(|_| SampleHist::empty()).collect(),
        }
    }

    /// Number of samples per trace.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// The binning rule.
    pub fn binning(&self) -> Binning {
        self.binning
    }

    /// Class labels that have been seen, in increasing order.
    pub fn labels(&self) -> &[u16] {
        &self.labels
    }

    /// Number of traces seen for a class (0 for an unknown label).
    pub fn class_count(&self, label: u16) -> u64 {
        self.slot(label).map_or(0, |s| self.class_counts[s])
    }

    /// Fills `out` with the non-zero bins of one sample and class, in bin order.
    pub(crate) fn fill_bins(&self, sample: usize, slot: usize, out: &mut Vec<(i64, u32)>) {
        out.clear();
        if let Some(hist) = self.hists.get(sample) {
            hist.for_each_nonzero(|s, bin, count| {
                if s == slot {
                    out.push((bin, count));
                }
            });
        }
        out.sort_unstable_by_key(|&(bin, _)| bin);
    }

    /// Total number of sample values that could not be binned (NaN, infinity, fractions under
    /// [`Binning::Exact`], or out-of-range bins). They are not part of any histogram.
    ///
    /// A trace with a rejected value still counts in the total of its class
    /// ([`class_count`](Self::class_count)). A caller that treats rejected values as an error
    /// must check this counter after every [`update`](Self::update).
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Approximate memory used, in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .hists
                .iter()
                .map(SampleHist::memory_bytes)
                .sum::<usize>()
            + self.labels.capacity() * 2
            + self.class_counts.capacity() * 8
    }

    /// Number of samples whose histogram is sparse.
    pub fn n_sparse(&self) -> usize {
        self.hists
            .iter()
            .filter(|h| matches!(h, SampleHist::Sparse(_)))
            .count()
    }

    /// True if the histogram of `sample` is sparse. False for a sample that does not exist.
    pub fn is_sparse(&self, sample: usize) -> bool {
        matches!(self.hists.get(sample), Some(SampleHist::Sparse(_)))
    }

    fn slot(&self, label: u16) -> Option<usize> {
        self.labels.binary_search(&label).ok()
    }

    /// Merges the sorted `add` (label, trace count) pairs into the labels of this accumulator.
    /// Returns the new state of the class bookkeeping, or `CountOverflow` if a class would hold
    /// more than `u32::MAX` traces. This accumulator is not changed.
    fn plan_labels(&self, add: &[(u16, u64)]) -> Result<LabelPlan, StatsError> {
        let mut labels = Vec::with_capacity(self.labels.len() + add.len());
        let mut class_counts = Vec::with_capacity(labels.capacity());
        let mut old_to_new = Vec::with_capacity(self.labels.len());
        let (mut i, mut j) = (0, 0);
        while i < self.labels.len() || j < add.len() {
            let take_old = j == add.len() || (i < self.labels.len() && self.labels[i] <= add[j].0);
            let (label, mut count) = if take_old {
                (self.labels[i], self.class_counts[i])
            } else {
                (add[j].0, 0)
            };
            if take_old {
                old_to_new.push(labels.len() as u16);
                i += 1;
            }
            if j < add.len() && add[j].0 == label {
                count = count.saturating_add(add[j].1);
                j += 1;
            }
            if count > u64::from(u32::MAX) {
                return Err(StatsError::CountOverflow { label });
            }
            labels.push(label);
            class_counts.push(count);
        }
        Ok(LabelPlan {
            labels,
            class_counts,
            old_to_new,
        })
    }

    /// Adopts the new labels and counts of a plan, and moves the class rows of the histograms.
    fn apply_plan(&mut self, plan: LabelPlan) {
        let n_slots = plan.labels.len();
        self.labels = plan.labels;
        self.class_counts = plan.class_counts;
        let old_to_new = plan.old_to_new;
        self.hists
            .par_iter_mut()
            .for_each(|h| h.relabel(&old_to_new, n_slots));
    }

    /// Adds a batch of traces. `traces` has one row per trace and one column per sample, and
    /// `labels` has one class label per trace. Returns the number of values in this batch that
    /// were rejected by the binning rule. The traces that hold them still count in their class.
    ///
    /// The work is spread over the rayon thread pool. On error, the accumulator is unchanged.
    pub fn update<T: BinValue>(
        &mut self,
        traces: ArrayView2<T>,
        labels: ArrayView1<u16>,
    ) -> Result<u64, StatsError> {
        if traces.ncols() != self.n_samples() {
            return Err(StatsError::SampleCountMismatch {
                expected: self.n_samples(),
                got: traces.ncols(),
            });
        }
        if traces.nrows() != labels.len() {
            return Err(StatsError::LabelCountMismatch {
                traces: traces.nrows(),
                labels: labels.len(),
            });
        }

        // Count the traces of each label in the batch, then merge the labels in sorted order.
        let mut batch: FxHashMap<u16, u64> = FxHashMap::default();
        for &label in labels {
            *batch.entry(label).or_insert(0) += 1;
        }
        let mut add: Vec<(u16, u64)> = batch.into_iter().collect();
        add.sort_unstable();
        let plan = self.plan_labels(&add)?;
        self.apply_plan(plan);

        // The slot of each trace is the position of its label.
        let slots: Vec<u32> = labels
            .iter()
            .map(|&l| {
                // Every label of the batch is in `self.labels` after `apply_plan`.
                self.labels.binary_search(&l).unwrap_or(0) as u32
            })
            .collect();
        let n_slots = self.labels.len();
        let max_dense = self.max_dense_bins;
        let rejected = match self.binning {
            Binning::Exact => accumulate(
                &mut self.hists,
                traces,
                &slots,
                n_slots,
                max_dense,
                |v: T| v.exact_bin(),
            ),
            Binning::Fixed { origin, width } => accumulate(
                &mut self.hists,
                traces,
                &slots,
                n_slots,
                max_dense,
                move |v: T| fixed_bin(v.to_f64(), origin, width),
            ),
        };
        self.rejected = self.rejected.saturating_add(rejected);
        Ok(rejected)
    }

    /// Adds the counts of `other` to this accumulator.
    ///
    /// Both accumulators need the same number of samples and the same binning rule. The dense
    /// window limit may differ. The result has the same counts as one pass over the traces of
    /// both accumulators. On error, this accumulator is unchanged. `other` must be valid (see
    /// [`validate`](Self::validate)); `merge` returns an error for labels that are not in
    /// increasing order, but it does not check the histograms of `other`.
    pub fn merge(&mut self, other: &Self) -> Result<(), StatsError> {
        if self.n_samples() != other.n_samples() {
            return Err(StatsError::Incompatible(format!(
                "{} samples versus {}",
                self.n_samples(),
                other.n_samples()
            )));
        }
        if self.binning != other.binning {
            return Err(StatsError::Incompatible(format!(
                "binning {:?} versus {:?}",
                self.binning, other.binning
            )));
        }
        if other.labels.len() != other.class_counts.len()
            || other.hists.len() != other.n_samples
            || !other.labels.windows(2).all(|w| w[0] < w[1])
        {
            return invalid("the accumulator to merge is not valid");
        }
        let add: Vec<(u16, u64)> = other
            .labels
            .iter()
            .copied()
            .zip(other.class_counts.iter().copied())
            .collect();
        let plan = self.plan_labels(&add)?;
        self.apply_plan(plan);

        // Slot of each class of `other` in this accumulator.
        let slot_map: Vec<usize> = other
            .labels
            .iter()
            .map(|&l| self.labels.binary_search(&l).unwrap_or(0))
            .collect();
        self.rejected = self.rejected.saturating_add(other.rejected);
        let max_dense = self.max_dense_bins;
        self.hists
            .par_iter_mut()
            .zip(other.hists.par_iter())
            .for_each(|(mine, theirs)| {
                theirs.for_each_nonzero(|s, bin, count| {
                    // A slot beyond the labels of `other` cannot occur in a valid state.
                    if let Some(&slot) = slot_map.get(s) {
                        mine.add_count(slot, bin, count, max_dense);
                    }
                });
            });
        Ok(())
    }

    /// Shrinks every dense window to the occupied bins and releases spare memory. Call this
    /// before serializing: a compacted state depends only on the counts.
    pub fn compact(&mut self) {
        self.hists.par_iter_mut().for_each(SampleHist::compact);
    }

    /// Checks the internal invariants. Use it after deserializing untrusted or old data.
    ///
    /// The checks are:
    ///
    /// * the binning is valid, `max_dense_bins` is at most [`MAX_DENSE_BINS_LIMIT`], and
    ///   `labels` and `class_counts` have the same length;
    /// * the labels are in strictly increasing order, and no class count is above `u32::MAX`;
    /// * the number of histograms equals the number of samples;
    /// * every dense histogram has one row per label, counters that fit its window, and a
    ///   window that is not wider than `max_dense_bins` and has at most [`MAX_DENSE_COUNTERS`]
    ///   counters; a window without class rows (or without columns) is empty, with base 0;
    /// * every sparse entry belongs to a known class;
    /// * the bins are in the range of the binning rule;
    /// * the totals are consistent: for each sample and class, the bins hold at most as many
    ///   counts as the class has traces, and the missing counts of all samples add up to
    ///   [`rejected`](Self::rejected).
    pub fn validate(&self) -> Result<(), StatsError> {
        if self.max_dense_bins > MAX_DENSE_BINS_LIMIT {
            return invalid(format!(
                "max_dense_bins {} is above the limit {MAX_DENSE_BINS_LIMIT}",
                self.max_dense_bins
            ));
        }
        if let Binning::Fixed { origin, width } = self.binning
            && !(width.is_finite() && width > 0.0 && origin.is_finite())
        {
            return invalid(format!(
                "fixed binning with origin {origin} and width {width}"
            ));
        }
        if self.labels.len() != self.class_counts.len() {
            return invalid("labels and class_counts differ in length");
        }
        if !self.labels.windows(2).all(|w| w[0] < w[1]) {
            return invalid("class labels are not in strictly increasing order");
        }
        if self.class_counts.iter().any(|&c| c > u64::from(u32::MAX)) {
            return invalid("class count above u32::MAX");
        }
        if self.hists.len() != self.n_samples {
            return invalid(format!(
                "{} histograms for {} samples",
                self.hists.len(),
                self.n_samples
            ));
        }
        let mut missing: u128 = 0;
        let mut totals = vec![0_u64; self.labels.len()];
        for (i, h) in self.hists.iter().enumerate() {
            totals.fill(0);
            h.check(
                self.labels.len(),
                self.max_dense_bins,
                self.binning,
                &mut totals,
            )
            .or_else(|msg| invalid(format!("sample {i}: {msg}")))?;
            for (slot, (&total, &n)) in totals.iter().zip(&self.class_counts).enumerate() {
                if total > n {
                    return invalid(format!(
                        "sample {i}: class slot {slot} has more counts than traces"
                    ));
                }
                missing += u128::from(n - total);
            }
        }
        if missing != u128::from(self.rejected) {
            return invalid(format!(
                "{missing} values are missing from the histograms but {} are rejected",
                self.rejected
            ));
        }
        Ok(())
    }

    /// Non-zero `(bin, count)` pairs of one sample and class, in increasing bin order.
    /// Returns an empty vector for an unknown label or a sample that does not exist. Meant for
    /// reports, plots, and tests.
    pub fn histogram(&self, sample: usize, label: u16) -> Vec<(i64, u32)> {
        match (self.slot(label), self.hists.get(sample)) {
            (Some(s), Some(h)) => h.row(s),
            _ => Vec::new(),
        }
    }

    /// Runs the test on the given classes for every sample. One row of the contingency table is
    /// built per label, in the given order. Samples are processed in parallel.
    ///
    /// Returns `UnknownLabel` for a label without data, `DuplicateLabel` for a repeated label,
    /// and `InvalidState` if the accumulator is inconsistent (see [`validate`](Self::validate)).
    pub fn test_classes(
        &self,
        labels: &[u16],
        opts: &TestOptions,
    ) -> Result<Vec<TestResult>, StatsError> {
        let slots = labels
            .iter()
            .map(|&l| self.slot(l).ok_or(StatsError::UnknownLabel(l)))
            .collect::<Result<Vec<_>, _>>()?;
        for (i, l) in labels.iter().enumerate() {
            if labels[..i].contains(l) {
                return Err(StatsError::DuplicateLabel(*l));
            }
        }
        self.hists
            .par_iter()
            .map_init(
                || (Workspace::default(), Vec::<u32>::new()),
                |(ws, scratch), h| {
                    h.with_rows(&slots, scratch, |rows| test_table(rows, opts, ws))
                        .and_then(|result| result)
                },
            )
            .collect()
    }

    /// Two-class test for every sample: the traces with label `a` against those with label `b`.
    /// Traces of other classes are not part of the test.
    pub fn test_pair(
        &self,
        a: u16,
        b: u16,
        opts: &TestOptions,
    ) -> Result<Vec<TestResult>, StatsError> {
        self.test_classes(&[a, b], opts)
    }

    /// Test of all classes together for every sample (rows in increasing label order).
    pub fn test_all(&self, opts: &TestOptions) -> Result<Vec<TestResult>, StatsError> {
        self.test_classes(&self.labels, opts)
    }
}

/// Adds the batch to the histograms in parallel. Returns the number of rejected values.
fn accumulate<T, F>(
    hists: &mut [SampleHist],
    traces: ArrayView2<T>,
    slots: &[u32],
    n_slots: usize,
    max_dense: usize,
    bin_of: F,
) -> u64
where
    T: BinValue,
    F: Fn(T) -> Option<i64> + Sync,
{
    let ncols = traces.ncols();
    let flat = traces.as_slice();
    hists
        .par_chunks_mut(BLOCK)
        .enumerate()
        .map(|(b, block)| {
            let start = b * BLOCK;
            let len = block.len();
            let mut rejected = 0_u64;
            for h in block.iter_mut() {
                h.ensure_slots(n_slots);
            }
            let mut add_segment = |slot: u32, values: &mut dyn Iterator<Item = T>| {
                for (h, v) in block.iter_mut().zip(values) {
                    match bin_of(v) {
                        Some(bin) => h.add(slot as usize, bin, max_dense),
                        None => rejected += 1,
                    }
                }
            };
            for (i, &slot) in slots.iter().enumerate() {
                match flat {
                    // Fast path: the matrix is contiguous and row-major.
                    Some(data) => {
                        let seg = &data[i * ncols + start..i * ncols + start + len];
                        add_segment(slot, &mut seg.iter().copied());
                    }
                    None => {
                        let seg = traces.slice(s![i, start..start + len]);
                        add_segment(slot, &mut seg.iter().copied());
                    }
                }
            }
            rejected
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn small() -> (Array2<u8>, Array1<u16>) {
        let traces = Array2::from_shape_vec(
            (6, 3),
            vec![0, 1, 9, 0, 2, 9, 1, 2, 9, 1, 3, 8, 2, 3, 8, 2, 4, 7],
        )
        .unwrap();
        let labels = Array1::from(vec![5, 5, 5, 9, 9, 9]);
        (traces, labels)
    }

    #[test]
    fn counts_by_sample_and_class() {
        let (t, l) = small();
        let mut acc = HistAccumulator::new(3, Binning::Exact);
        assert_eq!(acc.update(t.view(), l.view()).unwrap(), 0);
        assert_eq!(acc.histogram(0, 5), vec![(0, 2), (1, 1)]);
        assert_eq!(acc.histogram(0, 9), vec![(1, 1), (2, 2)]);
        assert_eq!(acc.histogram(2, 5), vec![(9, 3)]);
        assert_eq!(acc.class_count(5), 3);
        assert_eq!(acc.class_count(9), 3);
        assert_eq!(acc.class_count(1), 0);
        acc.validate().unwrap();
    }

    #[test]
    fn errors_leave_the_accumulator_unchanged() {
        let (t, l) = small();
        let mut acc = HistAccumulator::new(3, Binning::Exact);
        assert_eq!(
            acc.update(t.view(), l.slice(s![..5])),
            Err(StatsError::LabelCountMismatch {
                traces: 6,
                labels: 5
            })
        );
        assert!(matches!(
            acc.update(t.slice(s![.., ..2]), l.view()),
            Err(StatsError::SampleCountMismatch { .. })
        ));
        assert!(acc.labels().is_empty());
    }

    #[test]
    fn window_grows_in_both_directions_and_goes_sparse() {
        let mut acc = HistAccumulator::with_max_dense_bins(1, Binning::Exact, 64);
        let values = [10_i64, 5, 12, -3, 40, 1000, 7];
        let traces = Array2::from_shape_vec((values.len(), 1), values.to_vec()).unwrap();
        let labels = Array1::from(vec![0_u16; values.len()]);
        acc.update(traces.view(), labels.view()).unwrap();
        let mut want: Vec<(i64, u32)> = values.iter().map(|&v| (v, 1)).collect();
        want.sort_unstable();
        assert_eq!(acc.histogram(0, 0), want);
        assert_eq!(acc.n_sparse(), 1);
        acc.validate().unwrap();
    }

    /// Repeated extension of the range in one or both directions must not inflate the window.
    #[test]
    fn window_does_not_drift() {
        let mut down = HistAccumulator::new(1, Binning::Exact);
        let mut pingpong = HistAccumulator::new(1, Binning::Exact);
        for i in 0..200_i64 {
            let one = |v: i64| Array2::from_elem((1, 1), v);
            let label = Array1::from(vec![0_u16]);
            down.update(one(1000 - i).view(), label.view()).unwrap();
            let v = if i % 2 == 0 { 1000 + i } else { 1000 - i };
            pingpong.update(one(v).view(), label.view()).unwrap();
        }
        for acc in [&down, &pingpong] {
            assert_eq!(acc.n_sparse(), 0);
            let SampleHist::Dense(d) = &acc.hists[0] else {
                unreachable!()
            };
            assert!(d.width <= 2 * 512, "width {}", d.width);
        }
    }

    #[test]
    fn rejects_unbinnable_values() {
        let mut acc = HistAccumulator::new(2, Binning::Exact);
        let traces = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 2.5, 3.0]).unwrap();
        let labels = Array1::from(vec![0_u16, 1]);
        assert_eq!(acc.update(traces.view(), labels.view()).unwrap(), 2);
        assert_eq!(acc.rejected(), 2);
        assert_eq!(acc.histogram(0, 0), vec![(1, 1)]);
        assert_eq!(acc.histogram(1, 1), vec![(3, 1)]);
        // The traces with a rejected value still count in their class.
        assert_eq!(acc.class_count(0), 1);
        assert_eq!(acc.class_count(1), 1);
        acc.validate().unwrap();
    }

    #[test]
    fn non_contiguous_input_works() {
        let (t, l) = small();
        let mut acc_a = HistAccumulator::new(3, Binning::Exact);
        acc_a.update(t.view(), l.view()).unwrap();
        // The transposed view of the transposed matrix has a non-standard layout.
        let tt = t.t().to_owned();
        let mut acc_b = HistAccumulator::new(3, Binning::Exact);
        acc_b.update(tt.t(), l.view()).unwrap();
        for s in 0..3 {
            for c in [5, 9] {
                assert_eq!(acc_a.histogram(s, c), acc_b.histogram(s, c));
            }
        }
    }

    #[test]
    fn merge_counts_equal_one_pass() {
        let (t, l) = small();
        let mut one = HistAccumulator::new(3, Binning::Exact);
        one.update(t.view(), l.view()).unwrap();
        let mut a = HistAccumulator::new(3, Binning::Exact);
        let mut b = HistAccumulator::new(3, Binning::Exact);
        a.update(t.slice(s![..4, ..]), l.slice(s![..4])).unwrap();
        b.update(t.slice(s![4.., ..]), l.slice(s![4..])).unwrap();
        a.merge(&b).unwrap();
        for s in 0..3 {
            for c in [5, 9] {
                assert_eq!(a.histogram(s, c), one.histogram(s, c));
            }
        }
        assert_eq!(a.class_count(9), 3);
    }

    #[test]
    fn merge_rejects_incompatible() {
        let a = HistAccumulator::new(3, Binning::Exact);
        let mut b = HistAccumulator::new(4, Binning::Exact);
        assert!(matches!(b.merge(&a), Err(StatsError::Incompatible(_))));
        let mut c = HistAccumulator::new(3, Binning::fixed(0.0, 2.0).unwrap());
        assert!(matches!(c.merge(&a), Err(StatsError::Incompatible(_))));
    }

    #[test]
    fn plan_labels_merges_in_sorted_order() {
        let mut acc = HistAccumulator::new(1, Binning::Exact);
        acc.labels = vec![3, 8];
        acc.class_counts = vec![10, 20];
        let plan = acc.plan_labels(&[(1, 1), (3, 5), (9, 2)]).unwrap();
        assert_eq!(plan.labels, vec![1, 3, 8, 9]);
        assert_eq!(plan.class_counts, vec![1, 15, 20, 2]);
        assert_eq!(plan.old_to_new, vec![1, 2]);
        let err = acc.plan_labels(&[(8, u64::from(u32::MAX))]).unwrap_err();
        assert_eq!(err, StatsError::CountOverflow { label: 8 });
    }

    // ---- Fix round 1, items 4 and 5: malformed states ------------------------------------

    /// A valid accumulator with 3 classes and 4 samples: samples 0 to 2 are dense, sample 3 is
    /// sparse (`max_dense_bins` is 8).
    fn valid_accumulator() -> HistAccumulator {
        let n = 30;
        let traces = Array2::from_shape_fn((n, 4), |(i, s)| {
            if s == 3 {
                (i * 40) as u16
            } else {
                ((i * (s + 1)) % 8) as u16
            }
        });
        let labels = Array1::from_iter((0..n).map(|i| [2_u16, 5, 9][i % 3]));
        let mut acc = HistAccumulator::with_max_dense_bins(4, Binning::Exact, 8);
        acc.update(traces.view(), labels.view()).unwrap();
        acc.validate().unwrap();
        assert!(!acc.is_sparse(0) && acc.is_sparse(3));
        acc
    }

    fn dense_mut(acc: &mut HistAccumulator, sample: usize) -> &mut Dense {
        match &mut acc.hists[sample] {
            SampleHist::Dense(d) => d,
            SampleHist::Sparse(_) => panic!("sample {sample} is sparse"),
        }
    }

    /// Every field that a method relies on, damaged one at a time.
    fn damaged_states() -> Vec<(&'static str, HistAccumulator)> {
        let mut out = Vec::new();
        let mut add = |name: &'static str, edit: &dyn Fn(&mut HistAccumulator)| {
            let mut acc = valid_accumulator();
            edit(&mut acc);
            out.push((name, acc));
        };
        add("labels not increasing", &|a| a.labels = vec![5, 2, 9]);
        add("duplicate labels", &|a| a.labels = vec![2, 5, 5]);
        add("labels longer than class_counts", &|a| {
            a.class_counts.pop().map(|_| ()).unwrap_or(())
        });
        add("labels and class_counts empty-mismatch", &|a| {
            a.labels = vec![0];
            a.class_counts = vec![];
            a.hists = (0..4).map(|_| SampleHist::empty()).collect();
        });
        add("class count above u32::MAX", &|a| {
            a.class_counts[1] = u64::from(u32::MAX) + 1
        });
        add("class count too low", &|a| a.class_counts[1] = 3);
        add("class count too high", &|a| a.class_counts[1] = 11);
        add("wrong rejected counter", &|a| a.rejected = 1);
        add("n_samples too large", &|a| a.n_samples = 5);
        add("missing histogram", &|a| {
            a.hists.pop();
        });
        add("dense slots too few", &|a| dense_mut(a, 0).slots = 2);
        add("dense slots too many", &|a| dense_mut(a, 0).slots = 4);
        add("dense counters do not fit the window", &|a| {
            dense_mut(a, 0).width = 9
        });
        add("dense window wider than the limit", &|a| {
            a.max_dense_bins = 4
        });
        add("dense window past the largest bin", &|a| {
            dense_mut(a, 0).base = i64::MAX
        });
        add("max_dense_bins above the limit", &|a| {
            a.max_dense_bins = MAX_DENSE_BINS_LIMIT + 1
        });
        add("sparse entry with an unknown class", &|a| {
            let SampleHist::Sparse(sp) = &mut a.hists[3] else {
                unreachable!()
            };
            sp.map.insert((i64::MAX, 3), 1);
        });
        add("fixed binning with zero width", &|a| {
            a.binning = Binning::Fixed {
                origin: 0.0,
                width: 0.0,
            };
        });
        add("fixed binning with a bin out of range", &|a| {
            a.binning = Binning::Fixed {
                origin: 0.0,
                width: 1.0,
            };
            let SampleHist::Sparse(sp) = &mut a.hists[3] else {
                unreachable!()
            };
            sp.map.insert((1 << 53, 0), 1);
            a.class_counts[0] += 1;
        });
        out
    }

    /// The case of the review: no classes, one dense histogram with a huge window.
    fn empty_with_a_huge_window() -> HistAccumulator {
        let mut acc = HistAccumulator::new(1, Binning::Exact);
        acc.max_dense_bins = usize::MAX;
        acc.hists[0] = SampleHist::Dense(Dense {
            base: i64::MIN,
            width: usize::MAX,
            slots: 0,
            counts: Vec::new(),
        });
        acc
    }

    #[test]
    fn a_damaged_state_fails_validate_and_both_deserializers() {
        let mut damaged = damaged_states();
        damaged.push(("empty with a huge window", empty_with_a_huge_window()));
        let mut dense_empty = valid_accumulator();
        dense_empty.labels.clear();
        dense_empty.class_counts.clear();
        dense_empty.hists = (0..4).map(|_| SampleHist::empty()).collect();
        dense_empty.rejected = 0;
        dense_empty.validate().unwrap();
        for (name, acc) in &damaged {
            assert!(
                matches!(acc.validate(), Err(StatsError::InvalidState(_))),
                "{name}: validate"
            );
            let json = serde_json::to_string(acc).unwrap();
            assert!(
                serde_json::from_str::<HistAccumulator>(&json).is_err(),
                "{name}: JSON"
            );
            let bytes = postcard::to_stdvec(acc).unwrap();
            assert!(
                postcard::from_bytes::<HistAccumulator>(&bytes).is_err(),
                "{name}: postcard"
            );
        }
        // A valid state still round-trips in both formats.
        let acc = valid_accumulator();
        let json = serde_json::to_string(&acc).unwrap();
        serde_json::from_str::<HistAccumulator>(&json)
            .unwrap()
            .validate()
            .unwrap();
        let bytes = postcard::to_stdvec(&acc).unwrap();
        postcard::from_bytes::<HistAccumulator>(&bytes)
            .unwrap()
            .validate()
            .unwrap();
    }

    /// An empty window must have no classes, no base, and no counters. Then adding the first
    /// class cannot try to allocate `usize::MAX` counters.
    #[test]
    fn an_empty_window_must_be_empty() {
        let mut acc = HistAccumulator::new(1, Binning::Exact);
        acc.max_dense_bins = MAX_DENSE_BINS_LIMIT;
        for (base, width) in [(i64::MIN, 0), (7, 0), (0, 16), (0, usize::MAX)] {
            acc.hists[0] = SampleHist::Dense(Dense {
                base,
                width,
                slots: 0,
                counts: Vec::new(),
            });
            assert!(acc.validate().is_err(), "base {base}, width {width}");
        }
    }

    #[test]
    fn the_dense_window_limit_is_capped() {
        let acc = HistAccumulator::with_max_dense_bins(1, Binning::Exact, usize::MAX);
        assert_eq!(acc.max_dense_bins, MAX_DENSE_BINS_LIMIT);
        let traces = Array2::from_shape_vec((2, 1), vec![i64::MIN, i64::MAX]).unwrap();
        let mut acc = acc;
        acc.update(traces.view(), Array1::from(vec![0_u16, 1]).view())
            .unwrap();
        assert!(acc.is_sparse(0));
        acc.validate().unwrap();
    }

    /// With many classes, a wide window needs too many counters. The histogram becomes sparse
    /// before it allocates them (`slots * width` above `MAX_DENSE_COUNTERS`).
    #[test]
    fn many_classes_with_a_wide_window_go_sparse() {
        let classes = MAX_DENSE_COUNTERS / MAX_DENSE_BINS_LIMIT + 1; // 257
        let mut acc = HistAccumulator::with_max_dense_bins(1, Binning::Exact, MAX_DENSE_BINS_LIMIT);
        let n = classes + 1;
        let values: Vec<i64> = (0..n)
            .map(|i| {
                if i == 0 {
                    0
                } else if i == 1 {
                    (MAX_DENSE_BINS_LIMIT - 1) as i64
                } else {
                    5
                }
            })
            .collect();
        let labels: Vec<u16> = (0..n).map(|i| i.saturating_sub(1) as u16).collect();
        let traces = Array2::from_shape_vec((n, 1), values).unwrap();
        acc.update(traces.view(), Array1::from(labels).view())
            .unwrap();
        assert_eq!(acc.labels().len(), classes);
        assert!(
            acc.is_sparse(0),
            "a dense window would need more than 2^28 counters"
        );
        acc.validate().unwrap();
        // The same data in two batches (the second adds the classes to a dense window).
        let mut acc2 =
            HistAccumulator::with_max_dense_bins(1, Binning::Exact, MAX_DENSE_BINS_LIMIT);
        let first =
            Array2::from_shape_vec((2, 1), vec![0_i64, (MAX_DENSE_BINS_LIMIT - 1) as i64]).unwrap();
        acc2.update(first.view(), Array1::from(vec![0_u16, 0]).view())
            .unwrap();
        assert!(!acc2.is_sparse(0));
        let rest = Array2::from_elem((classes - 1, 1), 5_i64);
        let rest_labels = Array1::from_iter((1..classes).map(|c| c as u16));
        acc2.update(rest.view(), rest_labels.view()).unwrap();
        assert!(
            acc2.is_sparse(0),
            "adding class rows must not allocate 2^28+ counters"
        );
        acc2.validate().unwrap();
    }

    /// If the state of a hand-built accumulator is wrong, a test returns an error. It does not panic.
    #[test]
    fn tests_on_a_state_with_too_few_rows_return_an_error() {
        let mut acc = valid_accumulator();
        dense_mut(&mut acc, 0).slots = 2;
        dense_mut(&mut acc, 0).counts.truncate(2 * 8);
        let opts = TestOptions::default();
        assert!(matches!(
            acc.test_all(&opts),
            Err(StatsError::InvalidState(_))
        ));
    }

    /// Fix round 2, item 1. A valid dense histogram has one class and a window of 2^20 bins
    /// (4 MiB). Adding labels that sort before it moves its row, and `relabel` must not
    /// allocate `new_slots * width` counters (here 258 * 2^20, above `MAX_DENSE_COUNTERS`). The
    /// sample becomes sparse first. The review used 65,536 labels (256 GiB).
    #[test]
    fn relabel_does_not_allocate_too_many_counters() {
        let wide = MAX_DENSE_BINS_LIMIT as i64 - 1;
        let first = |label: u16| {
            let mut acc =
                HistAccumulator::with_max_dense_bins(1, Binning::Exact, MAX_DENSE_BINS_LIMIT);
            let traces = Array2::from_shape_vec((2, 1), vec![0_i64, wide]).unwrap();
            acc.update(traces.view(), Array1::from(vec![label, label]).view())
                .unwrap();
            assert!(!acc.is_sparse(0), "a 2^20 window with one class is dense");
            acc
        };
        let classes = MAX_DENSE_COUNTERS / MAX_DENSE_BINS_LIMIT + 2; // 258 slots in total
        let low_labels: Vec<u16> = (0..classes as u16 - 1).collect();
        let check = |acc: &HistAccumulator| {
            assert!(
                acc.is_sparse(0),
                "258 * 2^20 counters must not be allocated"
            );
            assert!(
                acc.memory_bytes() < 64 << 20,
                "{} bytes",
                acc.memory_bytes()
            );
            assert_eq!(acc.labels().len(), classes);
            assert_eq!(acc.histogram(0, 65535), vec![(0, 1), (wide, 1)]);
            assert_eq!(acc.class_count(65535), 2);
            for &l in &low_labels {
                assert_eq!(acc.histogram(0, l), vec![(5, 1)], "label {l}");
            }
            acc.validate().unwrap();
        };
        // Merge: the labels of `other` all sort before label 65535.
        let mut other =
            HistAccumulator::with_max_dense_bins(1, Binning::Exact, MAX_DENSE_BINS_LIMIT);
        let five = Array2::from_elem((low_labels.len(), 1), 5_i64);
        other
            .update(five.view(), Array1::from(low_labels.clone()).view())
            .unwrap();
        let mut merged = first(65535);
        merged.merge(&other).unwrap();
        check(&merged);
        // Update with the same labels.
        let mut updated = first(65535);
        updated
            .update(five.view(), Array1::from(low_labels.clone()).view())
            .unwrap();
        check(&updated);
    }

    /// The slot arithmetic cannot overflow `usize`, even with a corrupt width on 32-bit targets.
    #[test]
    fn relabel_uses_saturating_arithmetic() {
        let mut h = SampleHist::Dense(Dense {
            base: 0,
            width: usize::MAX / 2,
            slots: 1,
            counts: Vec::new(),
        });
        // Not a valid state; `relabel` must switch to sparse and not allocate or wrap.
        h.relabel(&[1], 2);
        assert!(matches!(h, SampleHist::Sparse(_)));
    }
}
