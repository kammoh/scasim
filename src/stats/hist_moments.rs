//! Welch t-values from exact per-sample histograms.

use ndarray::{Array2, Axis};
use rayon::prelude::*;

use super::binning::Binning;
use super::error::StatsError;
use super::hist::HistAccumulator;
use super::tstat::{ClassSample, welch_t};
use super::ttest::MAX_ORDER;

struct Scratch {
    bins_a: Vec<(i64, u32)>,
    bins_b: Vec<(i64, u32)>,
    cs_a: Vec<f64>,
    cs_b: Vec<f64>,
}

impl Scratch {
    fn new(d: usize) -> Self {
        Self {
            bins_a: Vec::new(),
            bins_b: Vec::new(),
            cs_a: vec![0.0; 2 * d - 1],
            cs_b: vec![0.0; 2 * d - 1],
        }
    }
}

fn sample_moments(bins: &[(i64, u32)], cs: &mut [f64]) -> (u64, f64, f64) {
    cs.fill(0.0);
    let Some(&(origin, _)) = bins.first() else {
        return (0, 0.0, 0.0);
    };
    let n: u64 = bins.iter().map(|&(_, c)| u64::from(c)).sum();
    if n == 0 {
        return (0, 0.0, 0.0);
    }
    let s1: i128 = bins
        .iter()
        .map(|&(v, c)| i128::from(c) * (i128::from(v) - i128::from(origin)))
        .sum();
    let offset = s1 as f64 / n as f64;
    for &(value, count) in bins {
        let delta = (i128::from(value) - i128::from(origin)) as f64 - offset;
        let mut power = delta * delta;
        for sum in cs.iter_mut() {
            *sum += f64::from(count) * power;
            power *= delta;
        }
    }
    (n, origin as f64, offset)
}

impl HistAccumulator {
    /// Computes t-values of orders 1 through `d` for classes `a` and `b`.
    ///
    /// The result has shape `(d, n_samples)`. Row `k - 1` contains order `k`.
    /// Exact bins are sample values. Fixed-width bin indexes are not values, so this method
    /// returns [`StatsError::NotExactBinning`] for fixed binning. For each sample and class,
    /// `n` is the sum of that class's bin counts. This equals the class trace count when no
    /// values were rejected; a rejected value is absent from that sample's moments.
    ///
    /// Bin counts are exact. Sparse bins are sorted before summation, and all floating-point
    /// operations then follow increasing bin order. Equal counts therefore produce identical
    /// bits across merge orders, batch splits, and Rayon thread counts.
    pub fn t_values(&self, a: u16, b: u16, d: usize) -> Result<Array2<f64>, StatsError> {
        if d == 0 {
            return Err(StatsError::ZeroOrder);
        }
        if d > MAX_ORDER {
            return Err(StatsError::InvalidMomentOrder {
                order: d,
                max: MAX_ORDER,
            });
        }
        if self.binning() != Binning::Exact {
            return Err(StatsError::NotExactBinning);
        }
        let Some(slot_a) = self.labels().binary_search(&a).ok() else {
            return Ok(Array2::from_elem((d, self.n_samples()), f64::NAN));
        };
        let Some(slot_b) = self.labels().binary_search(&b).ok() else {
            return Ok(Array2::from_elem((d, self.n_samples()), f64::NAN));
        };
        let mut per_sample = Array2::zeros((self.n_samples(), d));
        per_sample
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each_init(
                || Scratch::new(d),
                |scratch, (sample, mut row)| {
                    self.fill_bins(sample, slot_a, &mut scratch.bins_a);
                    self.fill_bins(sample, slot_b, &mut scratch.bins_b);
                    let (na, oa, xa) = sample_moments(&scratch.bins_a, &mut scratch.cs_a);
                    let (nb, ob, xb) = sample_moments(&scratch.bins_b, &mut scratch.cs_b);
                    let a = ClassSample {
                        n: na,
                        origin: oa,
                        offset: xa,
                        cs: &scratch.cs_a,
                        cs_stride: 1,
                    };
                    let b = ClassSample {
                        n: nb,
                        origin: ob,
                        offset: xb,
                        cs: &scratch.cs_b,
                        cs_stride: 1,
                    };
                    for (i, t) in row.iter_mut().enumerate() {
                        *t = welch_t(i + 1, &a, &b);
                    }
                },
            );
        Ok(per_sample.reversed_axes().as_standard_layout().to_owned())
    }
}
