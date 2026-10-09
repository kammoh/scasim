//! Rules that map a sample value to an integer bin.
//!
//! # Rules
//!
//! * [`Binning::Exact`]: every integer value is its own bin. This is the right choice for toggle
//!   counts, Hamming weights, and any model with integer weights, because no information is lost
//!   and the bin index is the value. Floating-point input is accepted only if the value is an
//!   integer. Any other value (a fraction, NaN, or infinity) is *rejected*: it is not counted,
//!   and the accumulator reports it in its rejected counter. This avoids silent merging of
//!   unequal values.
//! * [`Binning::Fixed`]: bin `k` holds the values `x` with `origin + k * width <= x < origin +
//!   (k + 1) * width`. The bin index is `floor((x - origin) / width)`. Use this for weighted
//!   (float) models and for coarse bins of wide integer data. The rule is the same for integer and
//!   float input. NaN, infinity, and values whose bin index does not fit in 2^53 are rejected.
//!
//! Both rules are *streaming*: the bin of a value does not depend on other values, so batches can
//! be processed in any order and the histograms merge exactly. Adaptive rules (quantile bins,
//! bins fitted to the observed range) need the whole data set or a first pass, so they are not
//! part of the accumulator. [`Binning::fit`] derives a fixed rule from a known value range, for
//! example from a first batch or from the range of the model.

use serde::{Deserialize, Serialize};

use super::error::StatsError;

/// Largest bin magnitude that is exactly representable as an `f64` integer.
const MAX_EXACT: f64 = 9_007_199_254_740_992.0; // 2^53

/// How sample values are mapped to bins. See the [module documentation](self).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Binning {
    /// One bin per integer value.
    Exact,
    /// Equal-width bins `[origin + k * width, origin + (k + 1) * width)`.
    Fixed {
        /// Lower edge of bin 0.
        origin: f64,
        /// Width of each bin. Positive and finite.
        width: f64,
    },
}

impl Binning {
    /// Creates a fixed-width rule. `width` must be positive and finite, and `origin` finite.
    pub fn fixed(origin: f64, width: f64) -> Result<Self, StatsError> {
        if !(width.is_finite() && width > 0.0 && origin.is_finite()) {
            return Err(StatsError::InvalidBinning(format!(
                "origin {origin} and width {width} must be finite, and width must be positive"
            )));
        }
        Ok(Self::Fixed { origin, width })
    }

    /// Creates a fixed-width rule with `n_bins` bins that cover `[min, max]`, including `max`.
    pub fn fit(min: f64, max: f64, n_bins: usize) -> Result<Self, StatsError> {
        if n_bins == 0 || !(min.is_finite() && max.is_finite() && max >= min) {
            return Err(StatsError::InvalidBinning(format!(
                "cannot fit {n_bins} bins to [{min}, {max}]"
            )));
        }
        // A tiny widening keeps `max` inside the last bin.
        let width = ((max - min) / n_bins as f64).max(f64::MIN_POSITIVE) * (1.0 + 1.0e-12);
        Self::fixed(min, width)
    }
}

/// Bin index of `x` for the fixed-width rule, or `None` if `x` cannot be binned.
#[inline(always)]
pub fn fixed_bin(x: f64, origin: f64, width: f64) -> Option<i64> {
    let y = ((x - origin) / width).floor();
    // The comparison is false for NaN, so NaN is rejected too.
    if y.abs() < MAX_EXACT {
        Some(y as i64)
    } else {
        None
    }
}

/// A sample type that can be put into bins.
pub trait BinValue: Copy + Send + Sync {
    /// Bin index for [`Binning::Exact`], or `None` if the value is not an integer in range.
    fn exact_bin(self) -> Option<i64>;
    /// The value as `f64`, for [`Binning::Fixed`].
    fn to_f64(self) -> f64;
}

macro_rules! impl_bin_value_int {
    ($($t:ty),*) => {$(
        impl BinValue for $t {
            #[inline(always)]
            fn exact_bin(self) -> Option<i64> { i64::try_from(self).ok() }
            #[inline(always)]
            fn to_f64(self) -> f64 { self as f64 }
        }
    )*};
}
impl_bin_value_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

macro_rules! impl_bin_value_float {
    ($($t:ty),*) => {$(
        impl BinValue for $t {
            #[inline(always)]
            fn exact_bin(self) -> Option<i64> {
                let x = f64::from(self);
                // `fract` is NaN for infinity and NaN, so those are rejected.
                if x.fract() == 0.0 && x.abs() < MAX_EXACT { Some(x as i64) } else { None }
            }
            #[inline(always)]
            fn to_f64(self) -> f64 { f64::from(self) }
        }
    )*};
}
impl_bin_value_float!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integers_and_floats() {
        assert_eq!(7_u8.exact_bin(), Some(7));
        assert_eq!((-3_i32).exact_bin(), Some(-3));
        assert_eq!(u64::MAX.exact_bin(), None);
        assert_eq!(4.0_f64.exact_bin(), Some(4));
        assert_eq!((-0.0_f64).exact_bin(), Some(0));
        assert_eq!(4.5_f64.exact_bin(), None);
        assert_eq!(f64::NAN.exact_bin(), None);
        assert_eq!(f64::INFINITY.exact_bin(), None);
        assert_eq!(2.0_f32.exact_bin(), Some(2));
    }

    #[test]
    fn fixed_edges() {
        assert_eq!(fixed_bin(0.0, 0.0, 0.5), Some(0));
        assert_eq!(fixed_bin(0.49, 0.0, 0.5), Some(0));
        assert_eq!(fixed_bin(0.5, 0.0, 0.5), Some(1));
        assert_eq!(fixed_bin(-0.01, 0.0, 0.5), Some(-1));
        assert_eq!(fixed_bin(f64::NAN, 0.0, 1.0), None);
        assert_eq!(fixed_bin(1.0e300, 0.0, 1.0), None);
    }

    #[test]
    fn fit_includes_max() {
        let Binning::Fixed { origin, width } = Binning::fit(0.0, 10.0, 10).unwrap() else {
            unreachable!()
        };
        assert_eq!(fixed_bin(10.0, origin, width), Some(9));
        assert_eq!(fixed_bin(0.0, origin, width), Some(0));
        assert!(Binning::fixed(0.0, 0.0).is_err());
        assert!(Binning::fit(1.0, 0.0, 4).is_err());
    }
}
