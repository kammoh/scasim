//! Measures the t-test accumulator with user CPU time.
//!
//! Usage: `ttest_perf <n_traces> <ns> <d> <reps> [batch]`
//!
//! Each repetition creates a new accumulator, feeds all traces (in batches of `batch`
//! traces, default: one batch), and computes the t-values after every batch (as the
//! `tvla` tool does). The program prints the user CPU time (all threads) and the wall time
//! of the best repetition.

use std::time::Instant;

use ndarray::{Array1, Array2, s};
use scasim::stats::ttest::MomentAccumulator;

fn user_cpu_seconds() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let ns: usize = args[2].parse().unwrap();
    let d: usize = args[3].parse().unwrap();
    let reps: usize = args[4].parse().unwrap();
    let batch: usize = args.get(5).map_or(n, |b| b.parse().unwrap());

    // Integer-valued f32 traces, like the traces that scasim produces. Two classes.
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let x = Array2::from_shape_fn((n, ns), |(i, _)| {
        let r = (next() % 200 + next() % 200) as f32;
        if i % 2 == 0 {
            r
        } else {
            r + (next() % 8) as f32
        }
    });
    let labels = Array1::from_shape_fn(n, |i| (i % 2) as u16);

    // Warm up the thread pool.
    rayon::broadcast(|_| ());

    let mut best_cpu = f64::MAX;
    let mut best_wall = f64::MAX;
    let mut checksum = 0.0;
    for _ in 0..reps {
        let (c0, w0) = (user_cpu_seconds(), Instant::now());
        let mut last = Array2::zeros((d, ns));
        let mut acc = MomentAccumulator::new(ns, d).unwrap();
        let mut start = 0;
        while start < n {
            let end = (start + batch).min(n);
            acc.update(x.slice(s![start..end, ..]), labels.slice(s![start..end]))
                .unwrap();
            last = acc.t_values(0, 1);
            start = end;
        }
        let (c1, w1) = (user_cpu_seconds(), w0.elapsed().as_secs_f64());
        best_cpu = best_cpu.min(c1 - c0);
        best_wall = best_wall.min(w1);
        checksum += last.iter().filter(|v| v.is_finite()).sum::<f64>();
    }
    println!(
        "n={n} ns={ns} d={d} batch={batch}: user CPU {:.2} ms, wall {:.2} ms (best of {reps}; checksum {checksum:.6e})",
        best_cpu * 1e3,
        best_wall * 1e3
    );
}
