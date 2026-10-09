//! Welch t-statistics from per-class central moments.
//!
//! The preprocessing for each order follows Schneider and Moradi, CHES 2015,
//! ePrint 2015/207, Section 4 and Appendix A. The central sums use Pébay's
//! parallel moment formulas, SAND2008-6212.
//!
//! The accumulator stores raw central sums through order `2d` in `f64`. Any needed raw,
//! normalized, or scaled moment must be finite and either zero or normal. A subnormal moment,
//! a negative raw `CS_2`, or a positive `CS_2` that becomes zero or subnormal after division
//! returns NaN. Nonzero variance terms and the denominator must also be normal.
//! A zero-variance class keeps the zero-denominator rules below. The exact rule applies to the
//! moments used by the requested order: `p = 2` for order 1; `p = 2, 4` for order 2; and
//! `p = 2, k, 2k` for order `k >= 3`. In particular, the largest needed sum scales as
//! `n * sd^(2d)` and must stay in
//! the normal `f64` range. For `d = 4`, this gives an approximate standard-deviation range of
//! `1e-38` to `1e38` for ordinary class sizes. The scale stays at one when `2^-60 <= CM_2 <=
//! 2^60`, which preserves the original operation order and bit pattern in the normal range.
//! A non-finite scaled intermediate returns NaN.

/// Central moments for one class at one sample point.
///
/// `cs[(p - 2) * cs_stride]` is the central sum of order `p`.
#[derive(Clone, Copy, Debug)]
pub struct ClassSample<'a> {
    pub n: u64,
    pub origin: f64,
    pub offset: f64,
    pub cs: &'a [f64],
    /// Distance between consecutive orders in `cs`.
    pub cs_stride: usize,
}

/// Computes the Welch t-statistic of order `order` for one sample point.
///
/// A class with fewer than two traces gives NaN. A zero denominator gives
/// signed infinity for a nonzero numerator and NaN for zero over zero. Negative
/// variance from rounding gives NaN.
#[inline]
pub fn welch_t(order: usize, a: &ClassSample<'_>, b: &ClassSample<'_>) -> f64 {
    let Some(last_index) = order.checked_mul(2).and_then(|v| v.checked_sub(2)) else {
        return f64::NAN;
    };
    let Some(needed_a) = last_index
        .checked_mul(a.cs_stride)
        .and_then(|v| v.checked_add(1))
    else {
        return f64::NAN;
    };
    let Some(needed_b) = last_index
        .checked_mul(b.cs_stride)
        .and_then(|v| v.checked_add(1))
    else {
        return f64::NAN;
    };
    if order == 0
        || order > i32::MAX as usize
        || a.n < 2
        || b.n < 2
        || a.cs_stride == 0
        || b.cs_stride == 0
        || a.cs.len() < needed_a
        || b.cs.len() < needed_b
    {
        return f64::NAN;
    }
    let moment = |s: &ClassSample<'_>, p: usize| {
        let raw = s.cs[(p - 2) * s.cs_stride];
        if !in_range(raw) || (p == 2 && raw < 0.0) {
            return f64::NAN;
        }
        let normalized = raw / s.n as f64;
        if !in_range(normalized) || (p == 2 && raw > 0.0 && normalized == 0.0) {
            f64::NAN
        } else {
            normalized
        }
    };
    let (diff, va, vb) = if order <= 2 {
        let cm2a = moment(a, 2);
        let cm2b = moment(b, 2);
        if cm2a.is_nan() || cm2b.is_nan() {
            return f64::NAN;
        }
        let exponent = scale_exponent(cm2a.max(cm2b));
        if order == 1 {
            let diff = (scale_pow2(a.origin, -exponent) - scale_pow2(b.origin, -exponent))
                + (scale_pow2(a.offset, -exponent) - scale_pow2(b.offset, -exponent));
            let va = scale_pow2(cm2a, -2 * exponent);
            let vb = scale_pow2(cm2b, -2 * exponent);
            if !diff.is_finite() || ![va, vb].into_iter().all(in_range) {
                return f64::NAN;
            }
            (diff, va, vb)
        } else {
            let cm4a = moment(a, 4);
            let cm4b = moment(b, 4);
            if cm4a.is_nan() || cm4b.is_nan() {
                return f64::NAN;
            }
            let cm2a = scale_pow2(cm2a, -2 * exponent);
            let cm2b = scale_pow2(cm2b, -2 * exponent);
            let cm4a = scale_pow2(cm4a, -4 * exponent);
            let cm4b = scale_pow2(cm4b, -4 * exponent);
            if ![cm2a, cm2b, cm4a, cm4b].into_iter().all(in_range) {
                return f64::NAN;
            }
            let diff = cm2a - cm2b;
            let va = cm4a - cm2a * cm2a;
            let vb = cm4b - cm2b * cm2b;
            if !diff.is_finite() || ![va, vb].into_iter().all(in_range) {
                return f64::NAN;
            }
            (diff, va, vb)
        }
    } else {
        let (ma, va) = preprocessed(order, moment(a, 2), |p| moment(a, p));
        let (mb, vb) = preprocessed(order, moment(b, 2), |p| moment(b, p));
        (ma - mb, va, vb)
    };
    if va < 0.0 || vb < 0.0 {
        return f64::NAN;
    }
    if !diff.is_finite() {
        return f64::NAN;
    }
    let variance = va / a.n as f64 + vb / b.n as f64;
    let va_per_n = va / a.n as f64;
    let vb_per_n = vb / b.n as f64;
    let denominator = variance.sqrt();
    if variance < 0.0
        || !in_range(va_per_n)
        || !in_range(vb_per_n)
        || !in_range(variance)
        || (variance > 0.0 && !normal(denominator))
    {
        return f64::NAN;
    }
    if variance == 0.0 {
        return if diff > 0.0 {
            f64::INFINITY
        } else if diff < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    let result = diff / denominator;
    if !result.is_finite() {
        f64::NAN
    } else {
        result
    }
}

#[inline]
fn preprocessed(order: usize, cm2: f64, cm: impl Fn(usize) -> f64) -> (f64, f64) {
    if cm2 <= 0.0 || !cm2.is_finite() {
        if order == 2 {
            return (cm2, cm(4) - cm2 * cm2);
        }
        return (f64::NAN, f64::NAN);
    }

    // Scale moments by an exact power of two so the standardized values stay near one.
    // The exponent comes from CM_2 and is applied before any products or powers form.
    let scale_exponent = scale_exponent(cm2);
    let scaled = |p: usize| scale_pow2(cm(p), -scale_exponent * p as i32);
    let cm2_scaled = scaled(2);
    let cmk_raw = cm(order);
    let cm2k_raw = cm(2 * order);
    if !in_range(cmk_raw) || !in_range(cm2k_raw) {
        return (f64::NAN, f64::NAN);
    }
    if order == 2 {
        let cm4_scaled = scaled(4);
        if ![cm2_scaled, cm4_scaled].into_iter().all(in_range) {
            return (f64::NAN, f64::NAN);
        }
        (cm2_scaled, cm4_scaled - cm2_scaled * cm2_scaled)
    } else {
        let k = order as i32;
        let cmk = scale_pow2(cmk_raw, -scale_exponent * k);
        let cm2k = scale_pow2(cm2k_raw, -scale_exponent * 2 * k);
        let mean = cmk / cm2_scaled.sqrt().powi(k);
        let out_var = (cm2k - cmk * cmk) / cm2_scaled.powi(k);
        if ![cm2_scaled, cmk, cm2k, mean, out_var]
            .into_iter()
            .all(in_range)
        {
            return (f64::NAN, f64::NAN);
        }
        (mean, out_var)
    }
}

#[inline]
fn normal(value: f64) -> bool {
    value.is_finite() && value.abs() >= f64::MIN_POSITIVE
}

#[inline]
fn in_range(value: f64) -> bool {
    value.is_finite() && (value == 0.0 || normal(value))
}

#[inline]
fn scale_exponent(cm2: f64) -> i32 {
    if (2.0f64.powi(-60)..=2.0f64.powi(60)).contains(&cm2) {
        0
    } else if cm2 > 0.0 && cm2.is_finite() {
        (cm2.log2().floor() as i32).div_euclid(2)
    } else {
        0
    }
}

/// Multiplies by 2^exponent in bounded steps, including subnormal inputs and outputs.
#[inline]
fn scale_pow2(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 512 {
        value *= 2.0f64.powi(512);
        exponent -= 512;
    }
    while exponent < -512 {
        value *= 2.0f64.powi(-512);
        exponent += 512;
    }
    value * 2.0f64.powi(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: u64, origin: f64, offset: f64, sums: &[f64]) -> ClassSample<'_> {
        ClassSample {
            n,
            origin,
            offset,
            cs: sums,
            cs_stride: 1,
        }
    }

    #[test]
    fn zero_variance_uses_ieee_policy() {
        let sums = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a = sample(5, 10.0, 0.0, &sums);
        let b = sample(5, 9.0, 0.0, &sums);
        assert_eq!(welch_t(1, &a, &b), f64::INFINITY);
        assert_eq!(welch_t(1, &b, &a), f64::NEG_INFINITY);
        assert!(welch_t(1, &a, &a).is_nan());
    }

    #[test]
    fn too_few_traces_and_negative_variance_give_nan() {
        let sums = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a = sample(1, 10.0, 0.0, &sums);
        let b = sample(5, 9.0, 0.0, &sums);
        assert!(welch_t(1, &a, &b).is_nan());
        let negative = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a = sample(5, 10.0, 0.0, &negative);
        assert!(welch_t(1, &a, &b).is_nan());
    }

    #[test]
    fn subnormal_scaled_order_two_moment_gives_nan() {
        let large = 2.0f64.powi(100);
        let small = 3.0 * 2.0f64.powi(-168);
        let a_sums = [4.0 * large.powi(2), 0.0, 4.0 * large.powi(4)];
        let b_sums = [2.0 * small.powi(2), 0.0, 2.0 * small.powi(4)];
        let a = sample(4, 0.0, 0.0, &a_sums);
        let b = sample(4, 0.0, 0.0, &b_sums);

        assert!(welch_t(2, &a, &b).is_nan());
    }

    #[test]
    fn positive_raw_variance_that_divides_to_zero_gives_nan() {
        let raw_cs2 = 2.0f64.powi(-1074);
        let a_sums = [raw_cs2, 0.0];
        let b_sums = [0.0, 0.0];
        let a = sample(3, 0.0, 1.25 * 2.0f64.powi(-537), &a_sums);
        let b = sample(3, 0.0, 0.0, &b_sums);

        assert!(welch_t(1, &a, &b).is_nan());
    }

    #[test]
    fn negative_raw_variance_gives_nan_after_shared_scaling() {
        let a_sums = [4.0 * 2.0f64.powi(200), 0.0];
        let b_sums = [-4.0 * 2.0f64.powi(-1022), 0.0];
        let a = sample(4, 1.0, 0.0, &a_sums);
        let b = sample(4, 0.0, 0.0, &b_sums);

        assert!(welch_t(1, &a, &b).is_nan());
    }

    #[test]
    fn every_order_matches_direct_preprocessing_and_is_antisymmetric() {
        let xa = [-3.0, -1.0, 0.0, 1.0, 3.0, 4.0];
        let xb = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0];
        let moments = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let sums = (2..=8)
                .map(|p| x.iter().map(|v| (v - mean).powi(p)).sum::<f64>())
                .collect::<Vec<_>>();
            (mean, sums)
        };
        let (ma, sa) = moments(&xa);
        let (mb, sb) = moments(&xb);
        let a = ClassSample {
            n: xa.len() as u64,
            origin: ma,
            offset: 0.0,
            cs: &sa,
            cs_stride: 1,
        };
        let b = ClassSample {
            n: xb.len() as u64,
            origin: mb,
            offset: 0.0,
            cs: &sb,
            cs_stride: 1,
        };
        for k in 1..=4 {
            let direct = |x: &[f64]| {
                let mean = x.iter().sum::<f64>() / x.len() as f64;
                let dev: Vec<f64> = x.iter().map(|v| v - mean).collect();
                let variance = dev.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
                let y: Vec<f64> = match k {
                    1 => x.to_vec(),
                    2 => dev.iter().map(|v| v * v).collect(),
                    _ => dev
                        .iter()
                        .map(|v| (v / variance.sqrt()).powi(k as i32))
                        .collect(),
                };
                let ym = y.iter().sum::<f64>() / y.len() as f64;
                let yv = y.iter().map(|v| (v - ym).powi(2)).sum::<f64>() / y.len() as f64;
                (ym, yv)
            };
            let (ya, va) = direct(&xa);
            let (yb, vb) = direct(&xb);
            let expected = (ya - yb) / (va / xa.len() as f64 + vb / xb.len() as f64).sqrt();
            let got = welch_t(k, &a, &b);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "order {k}"
            );
            assert_eq!(welch_t(k, &b, &a), -got, "order {k}");
        }
    }

    #[test]
    fn standardized_orders_are_invariant_to_power_of_two_scale() {
        let base_a = [0.0, 0.0, 0.0, 1.0];
        let base_b = [0.0, 0.0, 1.0, 1.0, 1.0];
        let values = |samples: &[f64], e: i32| {
            let scale = 2.0f64.powi(e);
            let x: Vec<f64> = samples.iter().map(|v| v * scale).collect();
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let cs: Vec<f64> = (2..=8)
                .map(|p| x.iter().map(|v| (v - mean).powi(p)).sum())
                .collect();
            (mean, cs)
        };
        let reference = |e| {
            let (ma, ca) = values(&base_a, e);
            let (mb, cb) = values(&base_b, e);
            let a = sample(base_a.len() as u64, ma, 0.0, &ca);
            let b = sample(base_b.len() as u64, mb, 0.0, &cb);
            (1..=4).map(|k| welch_t(k, &a, &b)).collect::<Vec<_>>()
        };
        let expected = reference(0);
        for e in [-100, -50, 0, 50, 100] {
            let got = reference(e);
            for (k, (&actual, &want)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "scale exponent {e}, order {}: {actual} vs {want}",
                    k + 1
                );
            }
        }
    }

    #[test]
    fn normal_range_matches_frozen_18bc793_formula_bit_for_bit() {
        fn frozen(order: usize, a: &ClassSample<'_>, b: &ClassSample<'_>) -> f64 {
            let (diff, va, vb) = if order == 1 {
                (
                    (a.origin - b.origin) + (a.offset - b.offset),
                    a.cs[0] / a.n as f64,
                    b.cs[0] / b.n as f64,
                )
            } else {
                let cm = |s: &ClassSample<'_>, p: usize| s.cs[p - 2] / s.n as f64;
                let preprocess = |s: &ClassSample<'_>| {
                    let cm2 = cm(s, 2);
                    if order == 2 {
                        (cm2, cm(s, 4) - cm2 * cm2)
                    } else {
                        let k = order as i32;
                        (
                            cm(s, order) / cm2.sqrt().powi(k),
                            (cm(s, 2 * order) - cm(s, order).powi(2)) / cm2.powi(k),
                        )
                    }
                };
                let (ma, va) = preprocess(a);
                let (mb, vb) = preprocess(b);
                (ma - mb, va, vb)
            };
            let variance = va / a.n as f64 + vb / b.n as f64;
            diff / variance.sqrt()
        }

        let mut seed = 0x9e37_79b9_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as i32 % 2001) as f64 / 100.0
        };
        for _ in 0..128 {
            let xa: Vec<f64> = (0..19).map(|_| next()).collect();
            let xb: Vec<f64> = (0..23).map(|_| next()).collect();
            let moments = |x: &[f64]| {
                let mean = x.iter().sum::<f64>() / x.len() as f64;
                let sums = (2..=8)
                    .map(|p| x.iter().map(|v| (v - mean).powi(p)).sum::<f64>())
                    .collect::<Vec<_>>();
                (mean, sums)
            };
            let (ma, ca) = moments(&xa);
            let (mb, cb) = moments(&xb);
            let a = ClassSample {
                n: xa.len() as u64,
                origin: ma,
                offset: 0.0,
                cs: &ca,
                cs_stride: 1,
            };
            let b = ClassSample {
                n: xb.len() as u64,
                origin: mb,
                offset: 0.0,
                cs: &cb,
                cs_stride: 1,
            };
            for order in 1..=4 {
                assert_eq!(
                    welch_t(order, &a, &b).to_bits(),
                    frozen(order, &a, &b).to_bits()
                );
            }
        }
    }
}
