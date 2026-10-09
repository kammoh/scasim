//! Synthetic t-test data for the tests and the examples. Not part of the public API.
//!
//! The data look like real t-values: noise with a standard deviation of 1, and a few
//! planted leakage regions.

/// A small deterministic random number generator (xorshift64*). It avoids a dependency on
/// the `rand` crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform number in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// A standard normal number (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A planted leakage: samples `start..start + len` get `amplitude` added.
#[derive(Clone, Copy)]
pub struct Leak {
    pub start: usize,
    pub len: usize,
    pub amplitude: f64,
}

/// A t-value trace: N(0, 1) noise plus the leaks.
pub fn t_trace(n: usize, seed: u64, leaks: &[Leak]) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut t: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    for leak in leaks {
        for v in &mut t[leak.start..(leak.start + leak.len).min(n)] {
            *v += leak.amplitude;
        }
    }
    t
}

/// Max |t| versus number of traces. A leaking order grows with the square root of the number
/// of traces. A non-leaking order stays around the noise maximum.
pub fn max_t_curve(
    num_traces: &[f64],
    leak_per_sqrt_trace: f64,
    noise_level: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    num_traces
        .iter()
        .map(|n| (noise_level + 0.15 * rng.normal()).max(leak_per_sqrt_trace * n.sqrt()))
        .collect()
}
