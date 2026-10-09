//! Min/max envelope reduction for long traces.
//!
//! A trace with 10^6 samples cannot be drawn point by point.  A plot is only a few
//! thousand pixels wide, and an HTML file with 10^6 points is tens of megabytes.
//! Plain decimation (keep every k-th sample) can drop a single-sample leakage spike.
//! The envelope keeps the minimum and the maximum of each bucket, so no spike is lost.

use std::ops::Range;

/// Returns the sample range of bucket `i` when `len` samples are split into `buckets`
/// buckets.
///
/// The bounds are `i * len / buckets` and `(i + 1) * len / buckets`, computed with
/// integer arithmetic. The buckets are contiguous, do not overlap, and together cover
/// every sample exactly once. If `buckets <= len`, every bucket has at least one
/// sample, and bucket sizes differ by at most one.
///
/// # Panics
///
/// Panics if `buckets == 0` or `i >= buckets`.
pub fn bucket_bounds(len: usize, buckets: usize, i: usize) -> Range<usize> {
    assert!(buckets > 0, "the number of buckets must be at least 1");
    assert!(i < buckets, "bucket index {i} is out of range 0..{buckets}");
    // Use u128 so the product cannot overflow, even on 32-bit targets.
    let bound = |k: usize| (k as u128 * len as u128 / buckets as u128) as usize;
    bound(i)..bound(i + 1)
}

/// Reduces `values` to at most `buckets` points and keeps the minimum and the maximum
/// of each bucket.
///
/// Returns `(x, min, max)`. All three vectors have the same length.
///
/// * If `values.len() <= buckets`, the input is returned unchanged: `x` is the sample
///   index (`0.0, 1.0, ...`), and `min` and `max` are both copies of `values`. NaN
///   values stay in place.
/// * Otherwise the samples are split into `buckets` buckets with [`bucket_bounds`]. For
///   each bucket, `x` is the center of the bucket, that is, the mean of the first and
///   the last sample index (so it can end in `.5`). `min` and `max` are the smallest
///   and the largest value in the bucket.
///
/// NaN values are ignored. A bucket that contains only NaN values gives NaN for both
/// `min` and `max`, so a plot shows a gap there. Infinite values are treated like any
/// other value.
///
/// # Panics
///
/// Panics if `buckets == 0`.
///
/// # Example
///
/// ```
/// use scasim::plot::envelope;
///
/// let mut trace = vec![0.0; 1_000_000];
/// trace[123_456] = 40.0; // a single-sample spike
/// let (x, min, max) = envelope(&trace, 2000);
/// assert_eq!(x.len(), 2000);
/// let top = max.iter().cloned().fold(f64::MIN, f64::max);
/// assert_eq!(top, 40.0);
/// assert_eq!(min.iter().cloned().fold(f64::MAX, f64::min), 0.0);
/// ```
pub fn envelope(values: &[f64], buckets: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert!(buckets > 0, "the number of buckets must be at least 1");
    let n = values.len();
    if n <= buckets {
        let x = (0..n).map(|i| i as f64).collect();
        return (x, values.to_vec(), values.to_vec());
    }
    let mut x = Vec::with_capacity(buckets);
    let mut min = Vec::with_capacity(buckets);
    let mut max = Vec::with_capacity(buckets);
    for i in 0..buckets {
        let range = bucket_bounds(n, buckets, i);
        // `f64::min` and `f64::max` return the other operand if one is NaN. Starting
        // from NaN therefore ignores NaN values, and an all-NaN bucket stays NaN.
        let (lo, hi) = values[range.clone()]
            .iter()
            .fold((f64::NAN, f64::NAN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        x.push((range.start + range.end - 1) as f64 / 2.0);
        min.push(lo);
        max.push(hi);
    }
    (x, min, max)
}
