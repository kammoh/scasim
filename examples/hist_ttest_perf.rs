//! Measures update and t-value user CPU for histogram and moment accumulators.

use std::time::Instant;

use ndarray::{Array1, Array2};
use scasim::stats::ttest::MomentAccumulator;
use scasim::stats::{Binning, HistAccumulator};

fn user_cpu_seconds() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6
}

fn main() {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    for n in [2000, 20000] {
        for d in [2, 4] {
            for range in [65u32, 2001] {
                let x = Array2::from_shape_fn((n, 371), |_| {
                    if range == 65 {
                        (0..16).map(|_| next() & 1).sum::<u32>() * 4
                    } else {
                        next() % range
                    }
                });
                let labels = Array1::from_shape_fn(n, |i| (i % 2) as u16);
                rayon::broadcast(|_| ());
                let (c0, w0) = (user_cpu_seconds(), Instant::now());
                let mut h = HistAccumulator::new(371, Binning::Exact);
                h.update(x.view(), labels.view()).unwrap();
                let ht = h.t_values(0, 1, d).unwrap();
                let hist_cpu = user_cpu_seconds() - c0;
                let hist_wall = w0.elapsed().as_secs_f64();
                let (c0, w0) = (user_cpu_seconds(), Instant::now());
                let mut m = MomentAccumulator::new(371, d).unwrap();
                m.update(x.view(), labels.view()).unwrap();
                let mt = m.t_values(0, 1);
                let mom_cpu = user_cpu_seconds() - c0;
                let mom_wall = w0.elapsed().as_secs_f64();
                let checksum = ht.iter().filter(|v| v.is_finite()).sum::<f64>()
                    + mt.iter().filter(|v| v.is_finite()).sum::<f64>();
                println!(
                    "n={n} ns=371 d={d} values=0..{}: hist CPU {:.2} ms (wall {:.2}); moments CPU {:.2} ms (wall {:.2}); checksum {checksum:.6e}",
                    range - 1,
                    hist_cpu * 1e3,
                    hist_wall * 1e3,
                    mom_cpu * 1e3,
                    mom_wall * 1e3
                );
            }
        }
    }
}
