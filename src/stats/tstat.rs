//! Welch t-statistics from per-class central moments.
//!
//! The preprocessing for each order follows Schneider and Moradi, CHES 2015,
//! ePrint 2015/207, Section 4 and Appendix A. The central sums use Pébay's
//! parallel moment formulas, SAND2008-6212.

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
    let (diff, va, vb) = if order == 1 {
        (
            (a.origin - b.origin) + (a.offset - b.offset),
            a.cs[0] / a.n as f64,
            b.cs[0] / b.n as f64,
        )
    } else {
        let cm = |s: &ClassSample<'_>, p: usize| s.cs[(p - 2) * s.cs_stride] / s.n as f64;
        let (ma, va) = preprocessed(order, cm(a, 2), |p| cm(a, p));
        let (mb, vb) = preprocessed(order, cm(b, 2), |p| cm(b, p));
        (ma - mb, va, vb)
    };
    if va < 0.0 || vb < 0.0 {
        return f64::NAN;
    }
    let variance = va / a.n as f64 + vb / b.n as f64;
    if variance < 0.0 || variance.is_nan() {
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
    diff / variance.sqrt()
}

#[inline]
fn preprocessed(order: usize, cm2: f64, cm: impl Fn(usize) -> f64) -> (f64, f64) {
    if order == 2 {
        (cm2, cm(4) - cm2 * cm2)
    } else {
        let k = order as i32;
        let variance = cm2;
        let mean = cm(order) / variance.sqrt().powi(k);
        let out_var = (cm(2 * order) - cm(order).powi(2)) / variance.powi(k);
        (mean, out_var)
    }
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
    fn every_order_matches_direct_preprocessing_and_is_antisymmetric() {
        let xa = [-3.0, -1.0, 0.0, 1.0, 3.0, 4.0];
        let xb = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
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
}
