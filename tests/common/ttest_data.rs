//! Helpers shared by the integration tests: a small random generator, test data,
//! a naive reference implementation, and comparison functions.
#![allow(dead_code)]

use ndarray::{Array1, Array2};

/// A small, fast, deterministic random generator (SplitMix64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Random data with `nclasses` classes. Each sample point has its own mean and
/// standard deviation, and the classes differ in mean, variance, or skewness, so that
/// the t-values of all orders are not just noise. Labels are random.
pub fn gen_data(seed: u64, n: usize, ns: usize, nclasses: u16) -> (Array2<f64>, Array1<u16>) {
    let mut rng = Rng::new(seed);
    let mu: Vec<f64> = (0..ns).map(|_| 1000.0 + 30000.0 * rng.uniform()).collect();
    let sd: Vec<f64> = (0..ns).map(|_| 5.0 + 300.0 * rng.uniform()).collect();
    let labels = Array1::from_shape_fn(n, |_| (rng.next_u64() % nclasses as u64) as u16);
    let mut x = Array2::zeros((n, ns));
    for i in 0..n {
        let c = labels[i];
        for j in 0..ns {
            let g = rng.normal();
            let v = match (c, j % 4) {
                (0, _) | (_, 0) => g,
                (1, 1) => g + 0.15,
                (1, 2) => 1.3 * g,
                (1, _) => (g * g - 1.0) / std::f64::consts::SQRT_2,
                (_, 1) => 0.8 * g - 0.1,
                (_, 2) => g + 0.05,
                (_, _) => 1.1 * g + 0.2,
            };
            x[[i, j]] = (mu[j] + sd[j] * v).max(0.0);
        }
    }
    (x, labels)
}

/// Rounds to integers (for tests with integer traces).
pub fn rounded(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(f64::round)
}

/// The naive reference: the Welch t-values of order 1 to `d` for classes `a` and `b`.
///
/// For each sample point and class, the function subtracts a common offset (the value of
/// trace 0), computes the mean, builds the preprocessed variable of each trace
/// explicitly, and computes its mean and variance in a second step. All moments use
/// the normalization 1/n.
pub fn reference_t(x: &Array2<f64>, labels: &[u16], a: u16, b: u16, d: usize) -> Array2<f64> {
    let ns = x.ncols();
    let mut out = Array2::from_elem((d, ns), f64::NAN);
    for j in 0..ns {
        let off = x[[0, j]];
        let stats = |class: u16, k: usize| -> (f64, f64, f64) {
            let z: Vec<f64> = (0..x.nrows())
                .filter(|&i| labels[i] == class)
                .map(|i| x[[i, j]] - off)
                .collect();
            let n = z.len() as f64;
            let mean = z.iter().sum::<f64>() / n;
            let dev: Vec<f64> = z.iter().map(|v| v - mean).collect();
            let sd = (dev.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
            let y: Vec<f64> = match k {
                1 => z.clone(),
                2 => dev.iter().map(|v| v * v).collect(),
                _ => dev.iter().map(|v| (v / sd).powi(k as i32)).collect(),
            };
            let ym = y.iter().sum::<f64>() / n;
            let yv = y.iter().map(|v| (v - ym) * (v - ym)).sum::<f64>() / n;
            (ym, yv, n)
        };
        for k in 1..=d {
            let (ma, va, na) = stats(a, k);
            let (mb, vb, nb) = stats(b, k);
            let t = (ma - mb) / (va / na + vb / nb).sqrt();
            out[[k - 1, j]] = if t.is_finite() { t } else { f64::NAN };
        }
    }
    out
}

/// The largest error `|a - b| / max(1, |a|, |b|)`.
///
/// Both arrays must have `NaN` at the same places. A `NaN` pair does not count.
pub fn max_err(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "shape mismatch");
    let mut worst: f64 = 0.0;
    for ((_, &x), &y) in a.indexed_iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            if (x.is_nan() && y.is_nan()) || x == y {
                continue;
            }
            return f64::INFINITY;
        }
        worst = worst.max((x - y).abs() / 1.0f64.max(x.abs()).max(y.abs()));
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::max_err;
    use ndarray::array;

    #[test]
    fn non_finite_values_match_only_when_equal_by_policy() {
        assert_eq!(
            max_err(&array![[f64::INFINITY]], &array![[f64::INFINITY]]),
            0.0
        );
        assert_eq!(max_err(&array![[f64::NAN]], &array![[f64::NAN]]), 0.0);
        assert!(max_err(&array![[f64::INFINITY]], &array![[1.0]]).is_infinite());
        assert!(max_err(&array![[f64::INFINITY]], &array![[f64::NEG_INFINITY]]).is_infinite());
    }
}
