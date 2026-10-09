//! Thresholds on `-log10(p)` for a family of tests.
//!
//! A leakage evaluation tests many (channel, sample, pair) combinations. With `m` independent
//! tests at level `alpha`, the chance of at least one false alarm is about `m * alpha`. The
//! functions here return the per-test threshold as `-log10(p)`, so they compare directly with
//! the `neg_log10_p` of a test result.

use std::f64::consts::LN_10;

use super::special::normal_isf;

/// The conventional single-test threshold of the TVLA methodology, `p < 1e-5`.
pub const CONVENTIONAL: f64 = 5.0;

/// Bonferroni threshold `-log10(alpha / m)`.
///
/// It keeps the family-wise false-alarm probability below `alpha` for any dependence between the
/// tests. Neighboring samples of a trace are usually dependent, so the threshold is conservative.
pub fn bonferroni(alpha: f64, m: u64) -> f64 {
    -(alpha / m.max(1) as f64).log10()
}

/// Sidak threshold: the per-test level `1 - (1 - alpha)^(1/m)` as `-log10`.
///
/// It is exact for independent tests and differs from Bonferroni only for large `alpha`.
pub fn sidak(alpha: f64, m: u64) -> f64 {
    let per_test = -((-alpha).ln_1p() / m.max(1) as f64).exp_m1();
    -per_test.log10()
}

/// Threshold on `|t|` for a family of `m` two-sided t-tests, from the normal distribution.
///
/// Each test of the family uses the level `alpha / m`, so the family-wise false-alarm
/// probability stays below `alpha` (Bonferroni). A two-sided test at level `alpha / m` rejects
/// when `|t|` is above `z`, where `Q(z) = alpha / (2 m)` and `Q` is the upper tail of the
/// standard normal distribution. The result is `normal_isf(alpha / (2 m))`.
///
/// The normal distribution is a good model of the t statistic when the number of traces is
/// large, as in TVLA. A family with `m = 0` is treated as `m = 1`.
///
/// Example: `alpha = 1e-5` and `m = 742` give about 5.68. For `m = 1`, the result is 4.42, a
/// little lower than the conventional TVLA threshold 4.5.
pub fn t_bonferroni(alpha: f64, m: u64) -> f64 {
    normal_isf(alpha / (2.0 * m.max(1) as f64))
}

/// Number of tests in a family: channels, samples, and class pairs (or other test counts).
///
/// Returns `None` if the product does not fit in a `u64`.
pub fn family_size(channels: u64, samples: u64, tests_per_sample: u64) -> Option<u64> {
    channels.checked_mul(samples)?.checked_mul(tests_per_sample)
}

/// Converts `-log10(p)` to the natural-log p-value, for combining evidence in the log domain.
pub fn ln_p(neg_log10_p: f64) -> f64 {
    -neg_log10_p * LN_10
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bonferroni_values() {
        assert_relative_eq!(bonferroni(1e-5, 1), 5.0, epsilon = 1e-12);
        // 1000 channels * 371 samples: about 10.57.
        assert_relative_eq!(bonferroni(1e-5, 371_000), 10.569, epsilon = 1e-3);
    }

    #[test]
    fn sidak_close_to_bonferroni_for_small_alpha() {
        assert_relative_eq!(
            sidak(1e-5, 371_000),
            bonferroni(1e-5, 371_000),
            epsilon = 1e-4
        );
        assert!(sidak(0.5, 10) < bonferroni(0.5, 10));
    }

    #[test]
    fn t_bonferroni_values() {
        // SciPy: norm.isf(1e-5 / 1484) = 5.679903330994272.
        assert_relative_eq!(
            t_bonferroni(1e-5, 742),
            5.679_903_330_994_272,
            max_relative = 1e-14
        );
        // One test: norm.isf(5e-6) = 4.417173413469...
        assert_relative_eq!(
            t_bonferroni(1e-5, 1),
            4.417_173_413_469_023,
            max_relative = 1e-12
        );
        assert_eq!(t_bonferroni(1e-5, 0), t_bonferroni(1e-5, 1));
        // The threshold grows with the family size and shrinks with alpha.
        assert!(t_bonferroni(1e-5, 1484) > t_bonferroni(1e-5, 742));
        assert!(t_bonferroni(1e-3, 742) < t_bonferroni(1e-5, 742));
    }

    #[test]
    fn family_size_multiplies_and_detects_overflow() {
        assert_eq!(family_size(1000, 371, 1), Some(371_000));
        assert_eq!(family_size(0, u64::MAX, u64::MAX), Some(0));
        assert_eq!(family_size(u64::MAX, 2, 1), None);
        assert_eq!(family_size(1 << 32, 1 << 32, 1), None);
        assert_eq!(family_size(1 << 31, 1 << 32, 1), Some(1 << 63));
    }
}
