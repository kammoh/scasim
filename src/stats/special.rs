//! Special functions for p-values in the log domain.
//!
//! A leakage evaluation of a noise-free simulation can produce huge test statistics. The p-value
//! of such a statistic is far below the smallest positive `f64` (about 1e-308). A survival
//! function that returns `p` itself therefore returns 0, and `-log10(p)` becomes infinite. The
//! functions in this module return the natural logarithm of the survival function directly, so
//! the result stays finite and accurate for any statistic.
//!
//! The chi-squared survival function with `k` degrees of freedom is the regularized upper
//! incomplete gamma function `Q(k/2, x/2)`.
//!
//! # Method
//!
//! * For `x < a + 1`, the power series gives the lower function `P(a, x)`, and `ln Q = ln(1 - P)`.
//!   In this region `Q` is not small, so no precision is lost.
//! * For `x >= a + 1`, a continued fraction (modified Lentz algorithm) gives `Q` as a prefactor
//!   times a number of order 1. The prefactor `x^a e^-x / Gamma(a)` is computed in the log domain
//!   with Loader's saddle-point form. This form avoids the large cancellation between
//!   `a ln x`, `x`, and `ln Gamma(a)` for large `a`.
//!
//! The same saddle-point helper `bd0` is also used to compute the G statistic without
//! cancellation.
//!
//! # References
//!
//! * Loader, "Fast and accurate computation of binomial probabilities" (2000): `bd0` and the
//!   Stirling error.
//! * The structure of the series and the continued fraction follows the classic incomplete gamma
//!   routines (Numerical Recipes, `gser` and `gcf`).
//! * Wichura, "Algorithm AS 241: The percentage points of the normal distribution", Applied
//!   Statistics 37 (1988): `normal_isf`.

use std::f64::consts::{LN_10, PI};

/// `ln(sqrt(2 pi))`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Relative accuracy target for the series and the continued fraction.
const EPS: f64 = 1.0e-16;

/// Smallest magnitude allowed in the Lentz recurrences.
const TINY: f64 = 1.0e-300;

/// Maximum number of terms in the series and the continued fraction.
const MAX_ITER: usize = 200_000;

/// Natural logarithm of the gamma function for `z > 0`.
///
/// The argument is shifted up to at least 10 with the recurrence `Gamma(z + 1) = z Gamma(z)`.
/// Then Stirling's asymptotic series (seven terms) gives the result. The absolute error is about
/// `1e-15`.
pub fn ln_gamma(z: f64) -> f64 {
    debug_assert!(z > 0.0, "ln_gamma needs z > 0");
    let mut z = z;
    let mut product = 1.0;
    while z < 10.0 {
        product *= z;
        z += 1.0;
    }
    (z - 0.5) * z.ln() - z + LN_SQRT_2PI + stirling_series(z) - product.ln()
}

/// Stirling's series `1/(12 n) - 1/(360 n^3) + ...`, accurate for `n >= 10`.
fn stirling_series(n: f64) -> f64 {
    let inv = 1.0 / n;
    let inv2 = inv * inv;
    inv * (1.0 / 12.0
        - inv2
            * (1.0 / 360.0
                - inv2
                    * (1.0 / 1260.0
                        - inv2
                            * (1.0 / 1680.0
                                - inv2
                                    * (1.0 / 1188.0 - inv2 * (691.0 / 360_360.0 - inv2 / 156.0))))))
}

/// The Stirling error `ln Gamma(n + 1) - ((n + 1/2) ln n - n + ln sqrt(2 pi))` for `n > 0`.
///
/// For `n >= 10`, the asymptotic series is used directly, which has no cancellation.
pub fn stirling_error(n: f64) -> f64 {
    debug_assert!(n > 0.0);
    if n >= 10.0 {
        stirling_series(n)
    } else {
        ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - LN_SQRT_2PI
    }
}

/// Deviance term `bd0(x, np) = x ln(x / np) + np - x` for `x >= 0` and `np > 0`.
///
/// This is the algorithm of Loader (2000). It uses a series when `x` and `np` are close, so that
/// the result keeps full relative precision. It is non-negative, and it is zero only at `x = np`.
pub fn bd0(x: f64, np: f64) -> f64 {
    if x == 0.0 {
        return np;
    }
    let diff = x - np;
    if diff.abs() < 0.1 * (x + np) {
        let v = diff / (x + np);
        let mut s = diff * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        let v2 = v * v;
        for j in 1..1000 {
            ej *= v2;
            let s1 = s + ej / f64::from(2 * j + 1);
            if s1 == s {
                return s1;
            }
            s = s1;
        }
    }
    x * (x / np).ln() + np - x
}

/// `ln(x^a e^-x / Gamma(a))` for `a > 0`, `x > 0`, computed without large cancellation.
fn ln_prefactor(a: f64, x: f64) -> f64 {
    // x^a e^-x / Gamma(a) = a * Poisson(a; x), and the Poisson probability mass at a is
    // exp(-stirling_error(a) - bd0(a, x)) / sqrt(2 pi a).
    a.ln() - stirling_error(a) - bd0(a, x) - 0.5 * (2.0 * PI * a).ln()
}

/// Natural logarithm of the regularized upper incomplete gamma function `Q(a, x)`.
///
/// Requires `a > 0`. Returns 0 for `x <= 0` and `-inf` for `x = +inf`. The result is finite for
/// every finite `x`, even when `Q` is far below the smallest positive `f64`.
///
/// For very large `a` (above about `1e6`) and `x` close to `a`, the series and the continued
/// fraction need many terms (`O(sqrt(a))`). The cap is 200,000 terms.
pub fn ln_gamma_q(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0, "ln_gamma_q needs a > 0");
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return f64::NEG_INFINITY;
    }
    if x < a + 1.0 {
        // Series for P(a, x) = x^a e^-x / Gamma(a + 1) * sum_n x^n / ((a + 1) ... (a + n)).
        let ln_front = ln_prefactor(a, x) - a.ln();
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut denom = a;
        for _ in 0..MAX_ITER {
            denom += 1.0;
            term *= x / denom;
            sum += term;
            if term < sum * EPS {
                break;
            }
        }
        let p = (ln_front + sum.ln()).exp();
        (-p).ln_1p()
    } else {
        // Continued fraction (modified Lentz) for Gamma(a, x) e^x x^-a.
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITER {
            let i = i as f64;
            let an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < EPS {
                break;
            }
        }
        ln_prefactor(a, x) + h.ln()
    }
}

/// Natural logarithm of the chi-squared survival function `P(X >= x)` for `dof` degrees of
/// freedom. Returns 0 (that is, `p = 1`) for `dof = 0` or `x <= 0`.
pub fn chi2_ln_sf(x: f64, dof: u32) -> f64 {
    if dof == 0 {
        return 0.0;
    }
    ln_gamma_q(f64::from(dof) / 2.0, x / 2.0)
}

/// `-log10(p)` of a chi-squared statistic `x` with `dof` degrees of freedom.
///
/// This is the standard way to report leakage evidence: `-log10(p) > 5` corresponds to the
/// conventional threshold `p < 1e-5`. The value is non-negative, finite for finite `x`, and
/// exact in the log domain for statistics whose p-value would underflow `f64`.
pub fn neg_log10_chi2_sf(x: f64, dof: u32) -> f64 {
    let v = -chi2_ln_sf(x, dof) / LN_10;
    if v > 0.0 { v } else { 0.0 }
}

/// Upper-tail quantile of the standard normal distribution: the `z` with `Q(z) = p`, where
/// `Q(z) = P(Z > z)`.
///
/// Domain: `0 < p < 1`. The function returns `+inf` for `p = 0`, `-inf` for `p = 1`, and NaN for
/// any other input (NaN, negative, or above 1).
///
/// This is Wichura's algorithm AS 241 (PPND16), which has a relative error of about `1e-16`.
/// The tail branches use `p` itself, not `1 - p`, so a tiny `p` keeps its full relative
/// precision (`normal_isf(1e-300)` is about 37.05).
pub fn normal_isf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return f64::NEG_INFINITY;
    }
    // AS 241 works with the lower probability p' = 1 - p, and q = p' - 0.5 = 0.5 - p.
    let q = 0.5 - p;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * horner(&CENTRAL_NUM, r) / horner(&CENTRAL_DEN, r);
    }
    // Tail. AS 241 uses the smaller of p' and 1 - p'. For q > 0 (p < 0.5) this is p itself, so
    // a tiny p keeps its relative precision. For q < 0 it is 1 - p, which is exact for p > 0.5.
    let tail = if q > 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        horner(&NEAR_NUM, r) / horner(&NEAR_DEN, r)
    } else {
        let r = r - 5.0;
        horner(&FAR_NUM, r) / horner(&FAR_DEN, r)
    };
    // q > 0 is the upper tail: z is positive. Otherwise z is negative.
    if q > 0.0 { x } else { -x }
}

/// Evaluates a polynomial with Horner's rule. The coefficients start at the highest degree.
fn horner(coefficients: &[f64; 8], x: f64) -> f64 {
    coefficients.iter().fold(0.0, |acc, &c| acc * x + c)
}

// Coefficients of AS 241, highest degree first. The polynomials of the central branch are in
// `r = 0.180625 - q^2`, for `|q| <= 0.425`. The near tail is in `r - 1.6` for `r <= 5`, and the
// far tail is in `r - 5` for `r > 5`. Here `r = sqrt(-ln(tail probability))`.

/// Numerator of the central branch.
const CENTRAL_NUM: [f64; 8] = [
    2509.0809287301227,
    33430.57558358813,
    67265.7709270087,
    45921.95393154987,
    13731.69376550946,
    1971.5909503065513,
    133.14166789178438,
    3.3871328727963665,
];

/// Denominator of the central branch.
const CENTRAL_DEN: [f64; 8] = [
    5226.495278852854,
    28729.085735721943,
    39307.89580009271,
    21213.794301586597,
    5394.196021424751,
    687.1870074920579,
    42.31333070160091,
    1.0,
];

/// Numerator of the near tail.
const NEAR_NUM: [f64; 8] = [
    0.0007745450142783414,
    0.022723844989269184,
    0.2417807251774506,
    1.2704582524523684,
    3.6478483247632045,
    5.769497221460691,
    4.630337846156546,
    1.4234371107496835,
];

/// Denominator of the near tail.
const NEAR_DEN: [f64; 8] = [
    1.0507500716444169e-09,
    0.0005475938084995345,
    0.015198666563616457,
    0.14810397642748008,
    0.6897673349851,
    1.6763848301838038,
    2.053191626637759,
    1.0,
];

/// Numerator of the far tail.
const FAR_NUM: [f64; 8] = [
    2.0103343992922881e-07,
    2.7115555687434876e-05,
    0.0012426609473880784,
    0.026532189526576124,
    0.29656057182850487,
    1.7848265399172913,
    5.463784911164114,
    6.657904643501103,
];

/// Denominator of the far tail.
const FAR_DEN: [f64; 8] = [
    2.0442631033899397e-15,
    1.421511758316446e-07,
    1.8463183175100548e-05,
    0.0007868691311456133,
    0.014875361290850615,
    0.1369298809227358,
    0.599832206555888,
    1.0,
];

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ln_gamma_known_values() {
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-14);
        assert_relative_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-14);
        assert_relative_eq!(ln_gamma(0.5), 0.5 * PI.ln(), max_relative = 1e-14);
        // Gamma(10) = 9! = 362880.
        assert_relative_eq!(ln_gamma(10.0), 362_880.0_f64.ln(), max_relative = 1e-14);
    }

    #[test]
    fn q_exponential_case() {
        // Q(1, x) = exp(-x).
        for &x in &[0.1, 1.0, 2.5, 10.0, 700.0, 5000.0] {
            assert_relative_eq!(ln_gamma_q(1.0, x), -x, max_relative = 1e-13);
        }
    }

    #[test]
    fn q_erfc_case() {
        // Q(1/2, x) = erfc(sqrt(x)); at x = 1 this is 0.157299207050285...
        assert_relative_eq!(
            ln_gamma_q(0.5, 1.0).exp(),
            0.157_299_207_050_285_13,
            max_relative = 1e-13
        );
    }

    #[test]
    fn huge_statistic_stays_finite() {
        // ln Q is about -x/2 for x far above the mean, so -log10(p) is about x / (2 ln 10).
        let v = neg_log10_chi2_sf(1.0e6, 10);
        assert!(
            v.is_finite() && (v - 1.0e6 / (2.0 * LN_10)).abs() < 100.0,
            "{v}"
        );
    }

    #[test]
    fn degenerate_inputs() {
        assert_eq!(neg_log10_chi2_sf(5.0, 0), 0.0);
        assert_eq!(neg_log10_chi2_sf(0.0, 3), 0.0);
        assert_eq!(neg_log10_chi2_sf(-1.0, 3), 0.0);
    }

    /// Reference values of `norm.isf` from SciPy (also in `tests/fixtures/stats/norm_isf.json`).
    #[test]
    fn normal_isf_known_values() {
        assert_eq!(normal_isf(0.5), 0.0);
        assert_relative_eq!(
            normal_isf(0.025),
            1.959_963_984_540_054_5,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            normal_isf(1e-5),
            4.264_890_793_922_825,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            normal_isf(1e-300),
            37.047_096_299_361_2,
            max_relative = 1e-14
        );
    }

    #[test]
    fn normal_isf_domain() {
        assert_eq!(normal_isf(0.0), f64::INFINITY);
        assert_eq!(normal_isf(1.0), f64::NEG_INFINITY);
        for bad in [
            f64::NAN,
            -0.1,
            1.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0 - 1e-300,
        ] {
            assert!(normal_isf(bad).is_nan(), "{bad}");
        }
        // The smallest positive values stay finite.
        assert!(normal_isf(f64::MIN_POSITIVE).is_finite());
        assert!(normal_isf(5e-324).is_finite());
    }

    #[test]
    fn normal_isf_is_antisymmetric_and_decreasing() {
        // Q(-z) = 1 - Q(z), so isf(1 - p) = -isf(p). The value 1 - p is rounded, so the match is not exact.
        for k in 1..=200 {
            let p = f64::from(k) * 0.0025;
            assert_relative_eq!(normal_isf(1.0 - p), -normal_isf(p), max_relative = 1e-13);
        }
        let mut last = f64::INFINITY;
        for k in 1..1000 {
            let z = normal_isf(f64::from(k) / 1000.0);
            assert!(z < last);
            last = z;
        }
    }

    #[test]
    fn normal_isf_inverts_the_survival_function() {
        // Q(z) = erfc(z / sqrt(2)) / 2.
        for &p in &[0.4, 0.1, 1e-3, 1e-8, 1e-20, 1e-100] {
            let z = normal_isf(p);
            let q = 0.5 * libm::erfc(z / std::f64::consts::SQRT_2);
            assert_relative_eq!(q, p, max_relative = 1e-12);
        }
    }
}
