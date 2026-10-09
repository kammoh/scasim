//! Performance harness: update and test times (wall and CPU), and memory.
//!
//! Usage: `cargo run --release --example chi2_perf -- [--traces N] [--samples S] [--channels C]
//!         [--reps R] [--classes K] [--range V] [--dtype u8|f64] [--uniform] [--fixed]`
//!
//! Values are drawn from a binomial-like distribution on `0..range` (or uniformly with
//! `--uniform`). Data generation is not timed. CPU time is user plus system time of the whole
//! process (all threads), from `getrusage`.

use std::time::Instant;

use ndarray::{Array1, Array2};
use scasim::stats::{Binning, HistAccumulator, Statistic, StatsError, TestOptions};

fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` fills the struct for `RUSAGE_SELF`.
    let usage = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init()
    };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1.0e-6;
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

fn max_rss_mib() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: as above.
    let usage = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init()
    };
    // macOS reports bytes, Linux reports kibibytes.
    let bytes = if cfg!(target_os = "macos") {
        usage.ru_maxrss as f64
    } else {
        usage.ru_maxrss as f64 * 1024.0
    };
    bytes / (1024.0 * 1024.0)
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Reads the value of the option `name`, or returns `default` if the option is absent.
fn arg<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == name) {
        None => Ok(default),
        Some(i) => {
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("option {name} needs a value"))?;
            value
                .parse()
                .map_err(|_| format!("option {name}: cannot read {value:?}"))
        }
    }
}

/// Runs `f` and returns its wall and CPU time in seconds.
fn time(f: impl FnOnce() -> Result<u64, StatsError>) -> Result<(f64, f64), StatsError> {
    let (w0, c0) = (Instant::now(), cpu_seconds());
    f()?;
    Ok((w0.elapsed().as_secs_f64(), cpu_seconds() - c0))
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn gen_values(rng: &mut XorShift, n: usize, range: u64, uniform: bool) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let r = rng.next();
            if uniform {
                (r % range) as u8
            } else {
                // Popcount of (range - 1) random bits: Binomial(range - 1, 1/2), on 0..range.
                let bits = (range - 1).min(63);
                ((r >> 1) & ((1_u64 << bits) - 1)).count_ones() as u8
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let traces: usize = arg("--traces", 2000)?;
    let samples: usize = arg("--samples", 371)?;
    let channels: usize = arg("--channels", 1)?;
    let reps: usize = arg("--reps", 20)?;
    let classes: usize = arg("--classes", 2)?;
    let range: u64 = arg("--range", 64)?;
    let dtype: String = arg("--dtype", "u8".to_string())?;
    if dtype != "u8" && dtype != "f64" {
        return Err(format!("option --dtype: expected u8 or f64, got {dtype:?}").into());
    }
    if classes == 0 || !(2..=256).contains(&range) {
        return Err(
            "options --classes and --range: need classes >= 1 and 2 <= range <= 256".into(),
        );
    }
    let uniform = flag("--uniform");
    let threads = rayon::current_num_threads();

    let labels = Array1::from_iter((0..traces).map(|i| (i % classes) as u16));
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    println!(
        "config: {channels} channel(s) x {samples} samples x {traces} traces, {classes} classes, values 0..{range} ({}), dtype {dtype}, {threads} rayon threads",
        if uniform { "uniform" } else { "binomial" }
    );

    // Update phase.
    let binning = if flag("--fixed") {
        Binning::fixed(0.0, 1.0)?
    } else {
        Binning::Exact
    };
    let mut accs: Vec<HistAccumulator> = Vec::new();
    let (mut upd_wall, mut upd_cpu) = (0.0, 0.0);
    let mut first_wall = 0.0;
    let mut n_updates = 0_usize;
    for c in 0..channels {
        let mut acc = HistAccumulator::new(samples, binning);
        let n_reps = if channels == 1 { reps } else { 1 };
        for rep in 0..n_reps {
            let values = gen_values(&mut rng, traces * samples, range, uniform);
            // Build the matrix first so that only the update is timed.
            let (wall, cpu) = if dtype == "u8" {
                let m = Array2::from_shape_vec((traces, samples), values)?;
                time(|| acc.update(m.view(), labels.view()))?
            } else {
                let v: Vec<f64> = values.iter().map(|&x| f64::from(x)).collect();
                let m = Array2::from_shape_vec((traces, samples), v)?;
                time(|| acc.update(m.view(), labels.view()))?
            };
            upd_wall += wall;
            upd_cpu += cpu;
            n_updates += 1;
            if rep == 0 && c == 0 {
                first_wall = wall;
            }
        }
        accs.push(acc);
    }
    let increments = (n_updates * traces * samples) as f64;
    println!(
        "update: {n_updates} calls, wall {upd_wall:.4} s, CPU {upd_cpu:.4} s ({:.2} ns CPU per value); first call wall {:.3} ms; mean wall per call {:.3} ms, mean CPU per call {:.3} ms",
        upd_cpu / increments * 1.0e9,
        first_wall * 1.0e3,
        upd_wall / n_updates as f64 * 1.0e3,
        upd_cpu / n_updates as f64 * 1.0e3,
    );

    // Test phase: all channels.
    for (name, opts) in [
        ("Pearson+merge", TestOptions::default()),
        (
            "G+merge",
            TestOptions {
                statistic: Statistic::G,
                min_expected: 5.0,
            },
        ),
    ] {
        let reps_t = if channels == 1 { 200 } else { 1 };
        let (w0, c0) = (Instant::now(), cpu_seconds());
        let mut max_nlp = 0.0_f64;
        for _ in 0..reps_t {
            for acc in &accs {
                let r = acc.test_pair(0, 1, &opts)?;
                max_nlp = max_nlp.max(r.iter().map(|x| x.neg_log10_p).fold(0.0, f64::max));
            }
        }
        let n_tests = (reps_t * accs.len() * samples) as f64;
        println!(
            "test {name:<14}: wall {:.4} s, CPU {:.4} s for {} tests = {:.0} ns CPU per sample test (max -log10 p {max_nlp:.2})",
            w0.elapsed().as_secs_f64(),
            cpu_seconds() - c0,
            n_tests as usize,
            (cpu_seconds() - c0) / n_tests * 1.0e9
        );
    }

    let mem: usize = accs.iter().map(HistAccumulator::memory_bytes).sum();
    println!(
        "memory: {:.1} KiB per channel (accumulator estimate), {:.1} MiB total; process max RSS {:.1} MiB",
        mem as f64 / accs.len() as f64 / 1024.0,
        mem as f64 / (1024.0 * 1024.0),
        max_rss_mib()
    );
    Ok(())
}
