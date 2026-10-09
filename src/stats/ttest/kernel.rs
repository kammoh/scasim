//! Numeric kernels that work on one block of sample points.
//!
//! The state of one class is stored in blocks of [`W`] sample points. One block holds
//! `2d + 1` rows of [`W`] numbers each, in one contiguous piece of memory:
//!
//! | row | content                                          |
//! |-----|--------------------------------------------------|
//! | 0   | `origin` of the mean                             |
//! | 1   | `offset` of the mean (`mean = origin + offset`)  |
//! | `p` | central sum `CS_p`, for `p = 2, ..., 2d`         |
//!
//! A block is small enough to stay in the L1 cache while a thread works on it. Different
//! blocks are independent, so threads can process them in parallel without locks.
//! The last block of a class can be narrower than [`W`]. Its unused columns stay zero.

use super::TraceSample;

/// The number of sample points in one block.
pub(super) const W: usize = 32;

/// The number of rows in one block for maximum order `d`.
#[inline]
pub(super) fn rows_per_block(d: usize) -> usize {
    2 * d + 1
}

/// The number of `f64` values in one block for maximum order `d`.
#[inline]
pub(super) fn block_len(d: usize) -> usize {
    rows_per_block(d) * W
}

/// The number of blocks for `ns` sample points.
#[inline]
pub(super) fn block_count(ns: usize) -> usize {
    ns.div_ceil(W)
}

/// The number of real sample points in block `c`.
#[inline]
pub(super) fn block_width(ns: usize, c: usize) -> usize {
    W.min(ns - c * W)
}

/// Scalar coefficients of the two-set merge for one pair of counts.
///
/// They do not depend on the sample point. The caller computes them once and uses them
/// for all blocks.
pub(super) struct MergeCoefs {
    /// Maximum order `d`.
    d: usize,
    /// The weight `n_B / n` of the mean difference in the new mean.
    wb: f64,
    /// `C(p,k) (-n_B/n)^k`, at index `p * (2d + 1) + k`.
    fa: Vec<f64>,
    /// `C(p,k) (n_A/n)^k`, at index `p * (2d + 1) + k`.
    fb: Vec<f64>,
    /// The coefficient of `delta^p`, at index `p`.
    cst: Vec<f64>,
}

impl MergeCoefs {
    /// Computes the coefficients for sets of `na > 0` and `nb > 0` traces.
    pub(super) fn new(d: usize, na: u64, nb: u64) -> Self {
        debug_assert!(na > 0 && nb > 0);
        let m = 2 * d;
        let stride = m + 1;
        let (fna, fnb) = (na as f64, nb as f64);
        let n = fna + fnb;
        // Shares of the two sets in the merged set.
        let share_a = fna / n;
        let share_b = fnb / n;
        // Pascal's triangle up to row m.
        let mut binom = vec![0.0f64; stride * stride];
        for p in 0..=m {
            binom[p * stride] = 1.0;
            if p > 0 {
                for k in 1..=p {
                    binom[p * stride + k] =
                        binom[(p - 1) * stride + k - 1] + binom[(p - 1) * stride + k];
                }
            }
        }
        let mut fa = vec![0.0; stride * stride];
        let mut fb = vec![0.0; stride * stride];
        let mut cst = vec![0.0; stride];
        for p in 2..=m {
            let (mut pa, mut pb) = (1.0, 1.0);
            for k in 1..=p - 2 {
                pa *= -share_b;
                pb *= share_a;
                fa[p * stride + k] = binom[p * stride + k] * pa;
                fb[p * stride + k] = binom[p * stride + k] * pb;
            }
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            let q = p as i32 - 1;
            cst[p] = (fna * fnb / n) * (share_a.powi(q) + sign * share_b.powi(q));
        }
        Self {
            d,
            wb: share_b,
            fa,
            fb,
            cst,
        }
    }
}

/// Computes the mean of a segment of traces, for a range of sample points.
///
/// The function reads the traces row by row, so it reads contiguous memory. It finds the
/// mean as `origin + offset`. The origin is the first trace of the segment, and the offset
/// is the mean of the differences to the origin. These differences are small and exact
/// (for integer input), so the mean keeps its precision even if it is large.
///
/// `origin` and `offset` have one entry for each sample point `j0 .. j0 + origin.len()`.
pub(super) fn segment_mean<T: TraceSample>(
    rows: &[&[T]],
    ids: &[u32],
    j0: usize,
    origin: &mut [f64],
    offset: &mut [f64],
) {
    let width = origin.len();
    debug_assert!(!ids.is_empty() && offset.len() == width);
    for (o, x) in origin
        .iter_mut()
        .zip(&rows[ids[0] as usize][j0..j0 + width])
    {
        *o = x.to_f64();
    }
    offset.fill(0.0);
    for &i in &ids[1..] {
        let x = &rows[i as usize][j0..j0 + width];
        for ((s, o), x) in offset.iter_mut().zip(origin.iter()).zip(x) {
            *s += x.to_f64() - o;
        }
    }
    let nb = ids.len() as f64;
    for s in offset.iter_mut() {
        *s /= nb;
    }
}

/// Computes the block of a segment of traces: the mean and the central sums.
///
/// This is the second pass of the two-pass computation. It handles the `w` sample points
/// `j0 .. j0 + w` and writes one block to `out`. `origin` and `offset` are the results of
/// [`segment_mean`] for these sample points. The deviations from the mean are
/// `(x - origin) - offset`. The first term is an exact difference and the second term is
/// small, so the deviations are accurate even if the mean is large.
pub(super) fn segment_block<T: TraceSample>(
    rows: &[&[T]],
    ids: &[u32],
    j0: usize,
    d: usize,
    origin: &[f64],
    offset: &[f64],
    out: &mut [f64],
) {
    let w = origin.len();
    debug_assert!(w <= W && offset.len() == w);
    let (out_origin, rest) = out.split_at_mut(W);
    let (out_offset, sums) = rest.split_at_mut(W);
    out_origin[..w].copy_from_slice(origin);
    out_offset[..w].copy_from_slice(offset);
    // The number of central sums is 2d - 1. The compiler can unroll the loop over the
    // orders if it knows the number at compile time. Larger orders use the general code.
    match d {
        1 => sums_fixed::<T, 1>(rows, ids, j0, origin, offset, sums),
        2 => sums_fixed::<T, 3>(rows, ids, j0, origin, offset, sums),
        3 => sums_fixed::<T, 5>(rows, ids, j0, origin, offset, sums),
        4 => sums_fixed::<T, 7>(rows, ids, j0, origin, offset, sums),
        _ => sums_general(rows, ids, j0, d, origin, offset, sums),
    }
}

/// The central sums of orders 2 to `P + 1`, with `P` known at compile time.
fn sums_fixed<T: TraceSample, const P: usize>(
    rows: &[&[T]],
    ids: &[u32],
    j0: usize,
    origin: &[f64],
    offset: &[f64],
    sums: &mut [f64],
) {
    let w = origin.len();
    let mut acc = [[0.0f64; W]; P];
    for &i in ids {
        let x = &rows[i as usize][j0..j0 + w];
        for q in 0..w {
            let dev = (x[q].to_f64() - origin[q]) - offset[q];
            let mut pow = dev;
            for a in acc.iter_mut() {
                pow *= dev;
                a[q] += pow;
            }
        }
    }
    for (row, a) in sums.as_chunks_mut::<W>().0.iter_mut().zip(&acc) {
        row[..w].copy_from_slice(&a[..w]);
    }
}

/// The central sums of orders 2 to `2d`, for any `d`.
fn sums_general<T: TraceSample>(
    rows: &[&[T]],
    ids: &[u32],
    j0: usize,
    d: usize,
    origin: &[f64],
    offset: &[f64],
    sums: &mut [f64],
) {
    let w = origin.len();
    sums.fill(0.0);
    let mut dev = [0.0f64; W];
    let mut pow = [0.0f64; W];
    let (dev, pow) = (&mut dev[..w], &mut pow[..w]);
    let n_sums = 2 * d - 1;
    for &i in ids {
        let x = &rows[i as usize][j0..j0 + w];
        for (((dv, pw), x), (o, f)) in dev
            .iter_mut()
            .zip(pow.iter_mut())
            .zip(x)
            .zip(origin.iter().zip(offset.iter()))
        {
            *dv = (x.to_f64() - o) - f;
            *pw = *dv;
        }
        for row in sums.as_chunks_mut::<W>().0.iter_mut().take(n_sums) {
            for ((m, pw), dv) in row[..w].iter_mut().zip(pow.iter_mut()).zip(dev.iter()) {
                *pw *= dv;
                *m += *pw;
            }
        }
    }
}

/// Merges the block `blk` (set `B`) into the block `acc` (set `A`).
///
/// Both blocks cover the same `w` sample points. `dp` is scratch memory with
/// `(2d + 1) * W` values. The coefficients `c` must come from the counts of the two sets.
pub(super) fn merge_block(acc: &mut [f64], blk: &[f64], w: usize, c: &MergeCoefs, dp: &mut [f64]) {
    let m = 2 * c.d;
    let stride = m + 1;

    // dp row k holds delta^k, where delta = mean_B - mean_A.
    {
        let (d1, _) = dp[W..].split_at_mut(W);
        for q in 0..w {
            d1[q] = (blk[q] - acc[q]) + (blk[W + q] - acc[W + q]);
        }
    }
    for k in 2..=m {
        let (lo, hi) = dp.split_at_mut(k * W);
        let prev = &lo[(k - 1) * W..(k - 1) * W + w];
        let d1 = &lo[W..W + w];
        for ((h, p), d1) in hi[..w].iter_mut().zip(prev).zip(d1) {
            *h = p * d1;
        }
    }

    // Central sums, from the highest order to the lowest. The terms of order p need the
    // old sums of lower orders, so the order of the loop is important.
    for p in (2..=m).rev() {
        let (lo, hi) = acc.split_at_mut(p * W);
        let ap = &mut hi[..w];
        let bp = &blk[p * W..p * W + w];
        let dpp = &dp[p * W..p * W + w];
        let cst = c.cst[p];
        for ((a, b), dd) in ap.iter_mut().zip(bp).zip(dpp) {
            *a += b + cst * dd;
        }
        for k in 1..=p.saturating_sub(2) {
            let fa = c.fa[p * stride + k];
            let fb = c.fb[p * stride + k];
            let a_pk = &lo[(p - k) * W..(p - k) * W + w];
            let b_pk = &blk[(p - k) * W..(p - k) * W + w];
            let dk = &dp[k * W..k * W + w];
            for (((a, apk), bpk), dd) in ap.iter_mut().zip(a_pk).zip(b_pk).zip(dk) {
                *a += dd * (fa * apk + fb * bpk);
            }
        }
    }

    // The mean moves by the share of B in the merged set.
    let wb = c.wb;
    for (o, dd) in acc[W..W + w].iter_mut().zip(&dp[W..W + w]) {
        *o += wb * dd;
    }
}

/// Copies the first `w` columns of every row of block `src` to block `dst`.
pub(super) fn copy_block(dst: &mut [f64], src: &[f64], w: usize, rows: usize) {
    for r in 0..rows {
        dst[r * W..r * W + w].copy_from_slice(&src[r * W..r * W + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The code for a fixed order and the general code give the same bits.
    #[test]
    fn fixed_and_general_sums_agree() {
        let (n, w) = (50usize, 21usize);
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..w)
                    .map(|j| ((i * 37 + j * 11) % 101) as f32 * 0.25)
                    .collect()
            })
            .collect();
        let rows: Vec<&[f32]> = data.iter().map(|r| r.as_slice()).collect();
        let ids: Vec<u32> = (0..n as u32).collect();
        let mut origin = vec![0.0; w];
        let mut offset = vec![0.0; w];
        segment_mean(&rows, &ids, 0, &mut origin, &mut offset);
        for d in 1..=4 {
            let mut fixed = vec![0.0; block_len(d)];
            segment_block(&rows, &ids, 0, d, &origin, &offset, &mut fixed);
            let mut general = vec![0.0; block_len(d)];
            general[..W][..w].copy_from_slice(&origin);
            general[W..2 * W][..w].copy_from_slice(&offset);
            sums_general(&rows, &ids, 0, d, &origin, &offset, &mut general[2 * W..]);
            assert_eq!(fixed, general, "d = {d}");
        }
    }
}
