//! Univariate TVLA t-tests of order 1 to `d` from one-pass central moments.
//!
//! This module implements fixed-versus-random (or any two-class) Welch t-tests on
//! side-channel traces. It handles each sample point of a trace independently.
//! For the test of order `k`, it uses the trace preprocessing of Schneider and Moradi.
//!
//! # Method
//!
//! For each class and each sample point, [`MomentAccumulator`] keeps:
//!
//! * the trace count `n` (it is the same for all sample points of one class),
//! * the mean, and
//! * the central sums `CS_p = sum_i (x_i - mean)^p` for `p = 2, ..., 2d`.
//!
//! The central moments are `CM_p = CS_p / n`. The module never stores raw power sums,
//! because they lose most of their accuracy at higher orders.
//!
//! The central sums are updated with the formulas of Pébay. Pébay's formulas combine the
//! sums of two disjoint sets of traces, `A` and `B`. A new batch of traces is first
//! reduced to its own mean and central sums with a stable two-pass computation. Then
//! it is merged into the stored state with the two-set formula. Two accumulators merge
//! in the same way. This gives the same result (up to rounding) for any split of the
//! traces into batches and for any merge order.
//!
//! For the two-set formula, let `delta = mean_B - mean_A` and `n = n_A + n_B`. Then
//!
//! ```text
//! CS_p = CS_p(A) + CS_p(B)
//!      + sum_{k=1}^{p-2} C(p,k) delta^k [ (-n_B/n)^k CS_{p-k}(A) + (n_A/n)^k CS_{p-k}(B) ]
//!      + (n_A n_B / n) delta^p [ (n_A/n)^(p-1) + (-1)^p (n_B/n)^(p-1) ]
//! ```
//!
//! This is the binomial expansion of `sum (x - mean)^p` around the merged mean. The terms
//! with `CS_1 = 0` vanish. The last term comes from the `CS_0 = n` terms. It equals the
//! last term of Pébay's equation.
//!
//! The mean is stored as the sum of two numbers, an `origin` and an `offset`. The origin
//! is a sample value of the first batch. This keeps the mean accurate when it is much
//! larger than the standard deviation (for example, when a trace has a large constant
//! offset). A single `f64` mean would lose about `log10(mean / sigma)` decimal digits.
//!
//! # The t-test of order `k`
//!
//! The test of order `k` compares the means of a preprocessed variable `Y` in the two
//! classes. With `s^2 = CM_2`, the definitions of Schneider and Moradi are:
//!
//! | order `k` | `Y`                       | mean of `Y` | variance of `Y`                      |
//! |-----------|---------------------------|-------------|--------------------------------------|
//! | 1         | `X`                       | `mean`      | `CM_2`                               |
//! | 2         | `(X - mean)^2`            | `CM_2`      | `CM_4 - CM_2^2`                      |
//! | `>= 3`    | `((X - mean) / s)^k`      | `CM_k / s^k`| `(CM_2k - CM_k^2) / s^(2k)`          |
//!
//! The statistic is the Welch t-value `(mean_a - mean_b) / sqrt(var_a / n_a + var_b / n_b)`.
//! All moments use the normalization `1 / n`, as in the paper. A test of order `k`
//! needs central moments up to order `2k`.
//!
//! # References
//!
//! * T. Schneider and A. Moradi, "Leakage Assessment Methodology - a clear roadmap for
//!   side-channel evaluations", CHES 2015, IACR ePrint 2015/207. See Section 4 and
//!   Appendix A.
//! * P. Pébay, "Formulas for Robust, One-Pass Parallel Computation of Covariances and
//!   Arbitrary-Order Statistical Moments", Sandia Report SAND2008-6212, 2008.
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use a2_ttest::ttest::MomentAccumulator;
//!
//! // Two sample points, order up to 2, classes 0 and 1.
//! let mut acc = MomentAccumulator::new(2, 2).unwrap();
//! let traces = array![[1.0f32, 5.0], [2.0, 6.0], [4.0, 5.5], [3.0, 9.0], [5.0, 1.0], [2.5, 4.0]];
//! let labels = array![0u16, 0, 0, 1, 1, 1];
//! acc.update(traces.view(), labels.view()).unwrap();
//! let t = acc.t_values(0, 1); // shape [2, 2]: [order - 1, sample]
//! assert_eq!(t.dim(), (2, 2));
//! ```

mod accumulator;
mod kernel;
mod moments;
mod sample;
mod tvalues;

pub use accumulator::MomentAccumulator;
pub use moments::{ClassMoments, Moments};
pub use sample::TraceSample;
