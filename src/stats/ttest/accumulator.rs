//! The moment accumulator: the update and merge logic.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

use super::TraceSample;
use super::kernel::{
    MergeCoefs, W, block_count, block_len, block_width, copy_block, merge_block, rows_per_block,
    segment_block, segment_mean,
};
use super::moments::{ClassMoments, Moments};
use crate::stats::error::StatsError;

/// A batch with fewer values than this runs on the calling thread only.
const SEQUENTIAL_BELOW: usize = 1 << 14;

/// The number of traces of one class that one task reduces to a partial result.
///
/// A segment is small enough that its data stays in the cache between the two passes.
/// Reducing in segments also keeps the rounding error small, because the partial results
/// are combined in a tree-like way.
const SEGMENT: usize = 128;

/// The memory limit for the partial results that the update computes before it merges them.
///
/// The limit depends only on `ns` and `d`, never on the number of threads. The merge order
/// does not depend on it, so it does not change the result.
const WAVE_BYTES: usize = 16 << 20;

/// The range for the number of partial results that the update computes before it merges them.
const WAVE_TASKS: std::ops::RangeInclusive<usize> = 4..=256;

/// The number of blocks in one range of sample points.
///
/// The update handles the sample points range by range. This bounds the memory that the
/// partial results use, even for very long traces.
const RANGE_BLOCKS: usize = 256;

/// The maximum supported t-test order. The merge kernels support orders through 64.
pub const MAX_ORDER: usize = 64;

/// The state of one class.
#[derive(Clone, Debug)]
pub(super) struct ClassState {
    /// The number of traces.
    pub(super) n: u64,
    /// The blocks, one after the other. See the module `kernel` for the layout.
    pub(super) data: Vec<f64>,
}

impl ClassState {
    fn new(ns: usize, d: usize) -> Self {
        Self {
            n: 0,
            data: vec![0.0; block_count(ns) * block_len(d)],
        }
    }
}

/// The traces of one class in one batch, and the merge plan for them.
struct Group {
    label: u16,
    /// Row indices of the traces in the batch.
    ids: Vec<u32>,
    /// For each segment of `SEGMENT` traces: the merge coefficients, or `None` if the
    /// state is empty before that segment.
    plan: Vec<Option<MergeCoefs>>,
}

/// An accumulator of per-class moments for univariate t-tests of order 1 to `d`.
///
/// It processes `ns` sample points per trace. Each trace has a class label of type
/// `u16`. The accumulator creates the state of a class when it first sees the label.
///
/// See the [module documentation](super) for the method and the references.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use a2_ttest::ttest::MomentAccumulator;
///
/// let traces = array![[1.0f64], [2.0], [3.0], [10.0], [11.0], [14.0]];
/// let labels = array![0u16, 0, 0, 1, 1, 1];
///
/// // One pass over all traces.
/// let mut all = MomentAccumulator::new(1, 1).unwrap();
/// all.update(traces.view(), labels.view()).unwrap();
///
/// // Two batches, merged afterwards.
/// let mut first = MomentAccumulator::new(1, 1).unwrap();
/// first.update(traces.slice(ndarray::s![..4, ..]), labels.slice(ndarray::s![..4])).unwrap();
/// let mut second = MomentAccumulator::new(1, 1).unwrap();
/// second.update(traces.slice(ndarray::s![4.., ..]), labels.slice(ndarray::s![4..])).unwrap();
/// first.merge(&second).unwrap();
///
/// let (a, b) = (all.t_values(0, 1)[[0, 0]], first.t_values(0, 1)[[0, 0]]);
/// assert!((a - b).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct MomentAccumulator {
    pub(super) ns: usize,
    pub(super) d: usize,
    pub(super) classes: BTreeMap<u16, ClassState>,
}

impl MomentAccumulator {
    /// Creates an empty accumulator for traces with `ns` sample points.
    ///
    /// The accumulator can compute t-tests of order 1 to `d`.
    ///
    /// Returns an error if `d` is outside `1..=MAX_ORDER` or the internal layout size overflows.
    pub fn new(ns: usize, d: usize) -> Result<Self, StatsError> {
        if d == 0 {
            return Err(StatsError::ZeroOrder);
        }
        if d > MAX_ORDER {
            return Err(StatsError::InvalidMomentOrder {
                order: d,
                max: MAX_ORDER,
            });
        }
        let rows = d
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or(StatsError::SizeOverflow)?;
        rows.checked_mul(ns).ok_or(StatsError::SizeOverflow)?;
        rows.checked_mul(W)
            .and_then(|_| rows.checked_mul(ns.div_ceil(W)))
            .and_then(|v| v.checked_mul(W))
            .ok_or(StatsError::SizeOverflow)?;
        Ok(Self {
            ns,
            d,
            classes: BTreeMap::new(),
        })
    }

    /// The number of sample points per trace.
    pub fn ns(&self) -> usize {
        self.ns
    }

    /// The maximum t-test order.
    pub fn order(&self) -> usize {
        self.d
    }

    /// The labels of the classes that have at least one trace, in increasing order.
    pub fn labels(&self) -> Vec<u16> {
        self.classes.keys().copied().collect()
    }

    /// The number of traces in class `label` (zero for an unknown class).
    pub fn count(&self, label: u16) -> u64 {
        self.classes.get(&label).map_or(0, |c| c.n)
    }

    /// Adds a batch of traces.
    ///
    /// `traces` has the shape `[n_traces, ns]`, and `labels` has one class label per
    /// trace. The batch can have any size, and a batch can contain any classes.
    ///
    /// For each class, the function reduces the traces to their mean and central sums
    /// with a stable two-pass computation. Then it merges the result into the state with
    /// the formulas of Pébay. The work is split by sample point and runs on the rayon
    /// thread pool.
    ///
    /// Returns an error for a shape mismatch or a batch larger than `u32::MAX` traces.
    pub fn update<T: TraceSample>(
        &mut self,
        traces: ArrayView2<T>,
        labels: ArrayView1<u16>,
    ) -> Result<(), StatsError> {
        let (n_traces, ns) = traces.dim();
        if ns != self.ns {
            return Err(StatsError::SampleCountMismatch {
                expected: self.ns,
                got: ns,
            });
        }
        if labels.len() != n_traces {
            return Err(StatsError::LabelCountMismatch {
                traces: n_traces,
                labels: labels.len(),
            });
        }
        if u32::try_from(n_traces).is_err() {
            return Err(StatsError::BatchTooLarge);
        }
        if n_traces == 0 || ns == 0 {
            return Ok(());
        }
        let d = self.d;

        // Make sure that every row of the input is a contiguous slice.
        let std_layout: Array2<T>;
        let traces = if ns == 1 || traces.strides()[1] == 1 {
            traces.view()
        } else {
            std_layout = traces.as_standard_layout().into_owned();
            std_layout.view()
        };
        let rows: Vec<&[T]> = traces
            .outer_iter()
            .map(|r| r.to_slice().expect("rows are contiguous"))
            .collect();

        // Group the traces by class.
        let mut slot_of: BTreeMap<u16, usize> = BTreeMap::new();
        let mut groups: Vec<Group> = Vec::new();
        let mut last = (0u16, usize::MAX);
        for (i, &label) in labels.iter().enumerate() {
            if last.1 == usize::MAX || last.0 != label {
                let slot = *slot_of.entry(label).or_insert_with(|| {
                    groups.push(Group {
                        label,
                        ids: Vec::new(),
                        plan: Vec::new(),
                    });
                    groups.len() - 1
                });
                last = (label, slot);
            }
            groups[last.1].ids.push(i as u32);
        }
        groups.sort_by_key(|g| g.label);

        // Plan the merges. The coefficients depend only on the counts.
        for g in &mut groups {
            let mut na = self.classes.get(&g.label).map_or(0, |c| c.n);
            na.checked_add(g.ids.len() as u64)
                .ok_or(StatsError::SizeOverflow)?;
            for segment in g.ids.chunks(SEGMENT) {
                let nb = segment.len() as u64;
                g.plan.push((na > 0).then(|| MergeCoefs::new(d, na, nb)));
                na += nb;
            }
            self.classes
                .entry(g.label)
                .or_insert_with(|| ClassState::new(ns, d));
        }

        // The tasks: one for each segment of each class. They are in the order of the plan.
        let tasks: Vec<(usize, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(gi, g)| (0..g.plan.len()).map(move |si| (gi, si)))
            .collect();
        let parallel = n_traces * ns >= SEQUENTIAL_BELOW;
        let len = block_len(d);
        let n_blocks = block_count(ns);

        for b0 in (0..n_blocks).step_by(RANGE_BLOCKS) {
            let b1 = (b0 + RANGE_BLOCKS).min(n_blocks);
            let partial_bytes = (b1 - b0) * len * std::mem::size_of::<f64>();
            let wave_tasks =
                (WAVE_BYTES / partial_bytes).clamp(*WAVE_TASKS.start(), *WAVE_TASKS.end());
            for wave in tasks.chunks(wave_tasks) {
                // Stage 1: reduce each segment to a partial result.
                let reduce = |&(gi, si): &(usize, usize)| {
                    let ids = &groups[gi].ids;
                    let ids = &ids[si * SEGMENT..((si + 1) * SEGMENT).min(ids.len())];
                    segment_partial(&rows, ids, b0, b1, ns, d)
                };
                let partials: Vec<Vec<f64>> = if parallel {
                    wave.par_iter().map(reduce).collect()
                } else {
                    wave.iter().map(reduce).collect()
                };

                // Stage 2: merge the partial results into the states, in the order of
                // the plan. The result does not depend on the number of threads.
                let mut start = 0;
                while start < wave.len() {
                    let gi = wave[start].0;
                    let end = start + wave[start..].iter().take_while(|t| t.0 == gi).count();
                    let items: Vec<(&[f64], Option<&MergeCoefs>)> = (start..end)
                        .map(|k| (partials[k].as_slice(), groups[gi].plan[wave[k].1].as_ref()))
                        .collect();
                    let state = self
                        .classes
                        .get_mut(&groups[gi].label)
                        .expect("the class exists");
                    let range = &mut state.data[b0 * len..b1 * len];
                    let merge = |dp: &mut Vec<f64>, (bi, st): (usize, &mut [f64])| {
                        let w = block_width(ns, b0 + bi);
                        for (partial, coefs) in &items {
                            let pb = &partial[bi * len..(bi + 1) * len];
                            match coefs {
                                None => copy_block(st, pb, w, rows_per_block(d)),
                                Some(coefs) => merge_block(st, pb, w, coefs, dp),
                            }
                        }
                    };
                    if parallel {
                        range
                            .par_chunks_exact_mut(len)
                            .enumerate()
                            .for_each_init(|| vec![0.0; len], merge);
                    } else {
                        let mut dp = vec![0.0; len];
                        for item in range.chunks_exact_mut(len).enumerate() {
                            merge(&mut dp, item);
                        }
                    }
                    start = end;
                }
            }
        }

        for g in &groups {
            if let Some(state) = self.classes.get_mut(&g.label) {
                state.n += g.ids.len() as u64;
            }
        }
        Ok(())
    }

    /// Merges the state of `other` into `self`.
    ///
    /// Afterwards, `self` describes the union of the traces of both accumulators. The
    /// result does not depend on the order of the merges, except for rounding.
    ///
    /// Returns an error if dimensions differ or a class count would overflow.
    pub fn merge(&mut self, other: &Self) -> Result<(), StatsError> {
        if self.ns != other.ns || self.d != other.d {
            return Err(StatsError::IncompatibleAccumulators);
        }
        for (&label, b) in &other.classes {
            self.classes
                .get(&label)
                .map_or(0, |a| a.n)
                .checked_add(b.n)
                .ok_or(StatsError::SizeOverflow)?;
        }
        let (ns, d) = (self.ns, self.d);
        for (label, b) in &other.classes {
            if b.n == 0 {
                continue;
            }
            match self.classes.entry(*label) {
                Entry::Vacant(v) => {
                    v.insert(b.clone());
                }
                Entry::Occupied(mut o) => {
                    let a = o.get_mut();
                    if a.n == 0 {
                        *a = b.clone();
                        continue;
                    }
                    let coefs = MergeCoefs::new(d, a.n, b.n);
                    let len = block_len(d);
                    let run = |dp: &mut Vec<f64>, (c, (ab, bb)): (usize, (&mut [f64], &[f64]))| {
                        merge_block(ab, bb, block_width(ns, c), &coefs, dp);
                    };
                    if ns < SEQUENTIAL_BELOW / 8 {
                        let mut dp = vec![0.0; len];
                        for item in a
                            .data
                            .chunks_exact_mut(len)
                            .zip(b.data.chunks_exact(len))
                            .enumerate()
                        {
                            run(&mut dp, item);
                        }
                    } else {
                        a.data
                            .par_chunks_exact_mut(len)
                            .zip(b.data.par_chunks_exact(len))
                            .enumerate()
                            .for_each_init(|| vec![0.0; len], run);
                    }
                    a.n += b.n;
                }
            }
        }
        Ok(())
    }

    /// The mean at each sample point for class `label`, or `None` for an unknown class.
    pub fn mean(&self, label: u16) -> Option<Array1<f64>> {
        let c = self.classes.get(&label)?;
        Some(self.gather_row(c, 0) + self.gather_row(c, 1))
    }

    /// The central sum `sum_i (x_i - mean)^p` at each sample point for class `label`.
    ///
    /// Returns `None` for an unknown class.
    ///
    pub fn central_sum(&self, label: u16, p: usize) -> Option<Array1<f64>> {
        if !(2..=2 * self.d).contains(&p) {
            return None;
        }
        let c = self.classes.get(&label)?;
        Some(self.gather_row(c, p))
    }

    /// Exports the complete state as plain data.
    ///
    /// The result does not depend on the internal memory layout.
    pub fn moments(&self) -> Moments {
        let ns = self.ns;
        let rows = rows_per_block(self.d);
        let classes = self
            .classes
            .iter()
            .map(|(&label, c)| {
                let mut data = vec![0.0; rows * ns];
                for r in 0..rows {
                    data[r * ns..(r + 1) * ns].copy_from_slice(&self.gather_row(c, r).to_vec());
                }
                ClassMoments {
                    label,
                    count: c.n,
                    data,
                }
            })
            .collect();
        Moments {
            ns,
            d: self.d,
            classes,
        }
    }

    /// Restores an accumulator from a state that [`moments`](Self::moments) exported.
    ///
    /// Returns an error if the saved dimensions or class moments are invalid.
    pub fn from_moments(m: Moments) -> Result<Self, StatsError> {
        m.validate_shape()?;
        let (ns, d) = (m.ns, m.d);
        let rows = rows_per_block(d);
        let mut acc = Self::new(ns, d)?;
        for cm in m.classes {
            if cm.data.len() != rows * ns {
                return Err(StatsError::WrongMomentLength {
                    label: cm.label,
                    expected: rows * ns,
                    found: cm.data.len(),
                });
            }
            let mut state = ClassState::new(ns, d);
            state.n = cm.count;
            for c in 0..block_count(ns) {
                let w = block_width(ns, c);
                let base = c * block_len(d);
                for r in 0..rows {
                    let dst = base + r * W;
                    state.data[dst..dst + w].copy_from_slice(&cm.data[r * ns + c * W..][..w]);
                }
            }
            if acc.classes.insert(cm.label, state).is_some() {
                return Err(StatsError::DuplicateLabel(cm.label));
            }
        }
        Ok(acc)
    }

    /// Reads row `r` of the block layout as a vector over sample points.
    fn gather_row(&self, c: &ClassState, r: usize) -> Array1<f64> {
        let (ns, d) = (self.ns, self.d);
        let mut out = Vec::with_capacity(ns);
        for b in 0..block_count(ns) {
            let base = b * block_len(d) + r * W;
            out.extend_from_slice(&c.data[base..base + block_width(ns, b)]);
        }
        Array1::from_vec(out)
    }
}

/// Reduces the traces `ids` to their mean and central sums.
///
/// The result covers the blocks `b0 .. b1`. It has the same layout as the state of a class
/// for these blocks.
fn segment_partial<T: TraceSample>(
    rows: &[&[T]],
    ids: &[u32],
    b0: usize,
    b1: usize,
    ns: usize,
    d: usize,
) -> Vec<f64> {
    let len = block_len(d);
    let j0 = b0 * W;
    let j1 = (b1 * W).min(ns);
    let mut origin = vec![0.0; j1 - j0];
    let mut offset = vec![0.0; j1 - j0];
    segment_mean(rows, ids, j0, &mut origin, &mut offset);
    let mut out = vec![0.0; (b1 - b0) * len];
    for (bi, block) in out.chunks_exact_mut(len).enumerate() {
        let lo = bi * W;
        let hi = (lo + W).min(j1 - j0);
        segment_block(
            rows,
            ids,
            j0 + lo,
            d,
            &origin[lo..hi],
            &offset[lo..hi],
            block,
        );
    }
    out
}
