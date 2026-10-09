//! Computes t-values from the blocked accumulator state.

use ndarray::Array2;
use rayon::prelude::*;

use super::MomentAccumulator;
use super::kernel::{W, block_count, block_len, block_width};
use crate::stats::tstat::{ClassSample, welch_t};

impl MomentAccumulator {
    /// Computes Welch t-values of orders 1 through `d` between two classes.
    ///
    /// The result has shape `[d, ns]`. Infinite values mark deterministic
    /// differences. NaN marks undefined values.
    pub fn t_values(&self, class_a: u16, class_b: u16) -> Array2<f64> {
        let (d, ns) = (self.d, self.ns);
        let mut out = Array2::from_elem((d, ns), f64::NAN);
        let (Some(a), Some(b)) = (self.classes.get(&class_a), self.classes.get(&class_b)) else {
            return out;
        };
        if ns == 0 {
            return out;
        }
        let len = block_len(d);
        let mut buf = vec![f64::NAN; block_count(ns) * d * W];
        buf.par_chunks_exact_mut(d * W)
            .zip(
                a.data
                    .par_chunks_exact(len)
                    .zip(b.data.par_chunks_exact(len)),
            )
            .enumerate()
            .for_each(|(block, (target, (aa, bb)))| {
                let width = block_width(ns, block);
                for q in 0..width {
                    let sa = ClassSample {
                        n: a.n,
                        origin: aa[q],
                        offset: aa[W + q],
                        cs: &aa[2 * W + q..],
                        cs_stride: W,
                    };
                    let sb = ClassSample {
                        n: b.n,
                        origin: bb[q],
                        offset: bb[W + q],
                        cs: &bb[2 * W + q..],
                        cs_stride: W,
                    };
                    for k in 1..=d {
                        target[(k - 1) * W + q] = welch_t(k, &sa, &sb);
                    }
                }
            });
        for (block, values) in buf.chunks_exact(d * W).enumerate() {
            for k in 0..d {
                for q in 0..block_width(ns, block) {
                    out[[k, block * W + q]] = values[k * W + q];
                }
            }
        }
        out
    }
}
