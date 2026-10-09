//! The generator: events for one segment at a time, sorted by time, applied to the register
//! values, and written as value changes.

use super::writer::{Decl, FstSink, Sink, VcdSink, fst_time_table_ok};
use super::{Format, LeakKind, SynthError, SynthSpec, first_segment};
use crate::shuffle::{SplitMix64, mix};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The files that [`generate`] wrote, and the ground truth about them.
#[derive(Clone, Debug)]
pub struct SynthOutput {
    /// `meta.list`: the metadata file of each batch, relative to the directory of the list.
    pub meta_list: PathBuf,
    pub metas: Vec<PathBuf>,
    pub waves: Vec<PathBuf>,
    pub stats: SynthStats,
}

/// Counts of what the generator wrote, over all batches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthStats {
    /// The toggles (flipped bits) of the registers of each sub-scope `tb.dut.s<i>`.
    pub scope_toggles: Vec<u64>,
    /// The toggles of the testbench registers.
    pub tb_toggles: u64,
    /// The toggles of the clocks.
    pub clock_toggles: u64,
    /// The number of traces in each class.
    pub class_counts: [u64; 2],
    /// The largest number of events that the generator held at one time.
    pub peak_events: usize,
}

impl SynthStats {
    /// The toggles of all registers in `tb.dut`: the selection `--include scope:tb.dut`.
    pub fn dut_toggles(&self) -> u64 {
        self.scope_toggles.iter().sum()
    }
}

/// A change of the signal `sig` at `time`: its bits in `mask` flip.
#[derive(Clone, Copy)]
struct Event {
    time: u64,
    sig: usize,
    mask: u64,
}

/// A register that updates on the clock of its domain.
#[derive(Clone, Copy)]
struct Register {
    sig: usize,
    /// `(scope, register)` in `tb.dut`, or `None` for a testbench register.
    place: Option<(usize, usize)>,
}

/// The signals of the design and which registers belong to which clock.
struct Layout {
    decls: Vec<Decl>,
    /// The registers of each clock domain.
    domains: Vec<Vec<Register>>,
    /// The first signal of the testbench registers and of the registers of `tb.dut`.
    tb_first: usize,
    dut_first: usize,
}

impl Layout {
    fn new(spec: &SynthSpec) -> Layout {
        let tb = vec!["tb".to_string()];
        let mut decls = vec![Decl {
            scope: tb.clone(),
            name: "clk".into(),
            width: 1,
        }];
        if spec.clock2.is_some() {
            decls.push(Decl {
                scope: tb.clone(),
                name: "clk2".into(),
                width: 1,
            });
        }
        let tb_first = decls.len();
        let mut domains = vec![Vec::new(); spec.domains()];
        for k in 0..spec.tb_registers {
            domains[0].push(Register {
                sig: decls.len(),
                place: None,
            });
            decls.push(Decl {
                scope: tb.clone(),
                name: format!("t{k}"),
                width: spec.width,
            });
        }
        let dut_first = decls.len();
        for scope in 0..spec.scopes {
            for register in 0..spec.registers {
                domains[spec.domain_of(scope)].push(Register {
                    sig: decls.len(),
                    place: Some((scope, register)),
                });
                decls.push(Decl {
                    scope: vec!["tb".into(), "dut".into(), format!("s{scope}")],
                    name: format!("r{register}"),
                    width: spec.width,
                });
            }
        }
        Layout {
            decls,
            domains,
            tb_first,
            dut_first,
        }
    }
}

/// A generator with the 16-bit draws that the noise needs.
struct Rng {
    inner: SplitMix64,
    buffer: u64,
    left: u32,
}

impl Rng {
    fn new(seed: u64, batch: usize) -> Rng {
        Rng {
            inner: SplitMix64(mix(seed ^ 0xDA7A_5EED) ^ mix(batch as u64 + 1)),
            buffer: 0,
            left: 0,
        }
    }

    /// A stream for one leak, from the seed, the batch, the trace, the signal, and the cycle. It
    /// is separate from the noise stream, so the number of draws of a leak (which depends on the
    /// class) cannot shift the noise.
    fn for_leak(seed: u64, batch: usize, trace: usize, sig: usize, cycle: u64) -> Rng {
        let key = [batch as u64, trace as u64, sig as u64, cycle]
            .into_iter()
            .fold(mix(seed ^ 0x1EA4_5EED), |key, part| mix(key ^ part) ^ part);
        Rng {
            inner: SplitMix64(key),
            buffer: 0,
            left: 0,
        }
    }

    fn next16(&mut self) -> u32 {
        if self.left == 0 {
            self.buffer = self.inner.next();
            self.left = 4;
        }
        self.left -= 1;
        let value = (self.buffer & 0xFFFF) as u32;
        self.buffer >>= 16;
        value
    }

    /// A random value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.inner.below(n)
    }

    /// True with the probability `p` (a multiple of 2^-16 at most).
    fn chance(&mut self, p: f64) -> bool {
        (self.next16() as f64) < p * 65536.0
    }
}

fn low_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

/// The noise: every bit toggles with the probability `density`.
fn noise_mask(rng: &mut Rng, width: u32, density: f64) -> u64 {
    if density <= 0.0 {
        return 0;
    }
    (0..width).fold(0, |mask, bit| {
        mask | (u64::from(rng.chance(density)) << bit)
    })
}

/// The mask of a leak: the `T` lowest bits flip, with `T` from the distribution of the leak.
fn leak_mask(kind: LeakKind, strength: f64, class: u16, width: u32, rng: &mut Rng) -> u64 {
    let m = i64::from(width / 2);
    // Uniform on {-1, 0, +1}, for the kinds that use it.
    let u = match kind {
        LeakKind::Mean | LeakKind::Variance => rng.below(3) as i64 - 1,
        _ => 0,
    };
    let toggles = match kind {
        LeakKind::Mean => {
            let base = m + u;
            if class == 1 {
                let whole = strength.floor();
                let extra = whole as i64 + i64::from(rng.chance(strength - whole));
                base + extra
            } else {
                base
            }
        }
        LeakKind::Variance => {
            let a = if class == 1 {
                strength.round() as i64
            } else {
                1
            };
            m + a * u
        }
        LeakKind::Equal3 => {
            if class == 0 {
                [2, 6][rng.below(2) as usize]
            } else {
                // 0 and 8 with probability 1/8 each, 4 with probability 3/4.
                match rng.below(8) {
                    0 => 0,
                    1 => 8,
                    _ => 4,
                }
            }
        }
        LeakKind::Deterministic => {
            m + if class == 1 {
                strength.round() as i64
            } else {
                0
            }
        }
    };
    low_mask(toggles as u32)
}

/// Adds the events of one segment that starts at `base`.
fn segment_events(
    spec: &SynthSpec,
    layout: &Layout,
    (batch, trace): (usize, usize),
    base: u64,
    class: u16,
    rng: &mut Rng,
    events: &mut Vec<Event>,
) {
    let glitch = spec.glitch.map(|g| (g.delta, low_mask(g.bits)));
    for (domain, registers) in layout.domains.iter().enumerate() {
        let (period, phase) = spec.clock_of(domain);
        for cycle in 0..spec.own_cycles(domain) {
            if domain == 0 && spec.gated(cycle) {
                continue;
            }
            let rise = base + phase + cycle * period;
            for time in [rise, rise + period / 2] {
                events.push(Event {
                    time,
                    sig: domain,
                    mask: 1,
                });
            }
            for (k, offset) in spec.update_offsets(domain, cycle).into_iter().enumerate() {
                let time = base + offset;
                for register in registers {
                    // The leak is planted at the first update of the cycle.
                    let leak = register.place.and_then(|(scope, reg)| {
                        spec.leaks.iter().find(|l| {
                            k == 0 && (l.scope, l.register, l.cycle as u64) == (scope, reg, cycle)
                        })
                    });
                    // The noise is always drawn, so the noise stream does not depend on the
                    // leaks. The leak replaces the noise mask and draws from its own stream.
                    let noise = noise_mask(rng, spec.width, spec.density);
                    let mask = match leak {
                        Some(l) => {
                            let mut own =
                                Rng::for_leak(spec.seed, batch, trace, register.sig, cycle);
                            leak_mask(l.kind, l.strength, class, spec.width, &mut own)
                        }
                        None => noise,
                    };
                    if mask != 0 {
                        events.push(Event {
                            time,
                            sig: register.sig,
                            mask,
                        });
                    }
                    if let Some((delta, g)) = glitch {
                        for time in [time + delta, time + delta + 1] {
                            events.push(Event {
                                time,
                                sig: register.sig,
                                mask: g,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// The current values, and the toggles of each signal.
struct State {
    values: Vec<u64>,
    toggles: Vec<u64>,
    /// The group (time step) in which a signal was last touched.
    stamp: Vec<u64>,
    group: u64,
    touched: Vec<(usize, u64)>,
    /// The number of time steps written after time 0, and the last one.
    steps: u64,
    last: u64,
}

impl State {
    /// Sorts the events by time, applies them, and writes the net changes. Events at the same
    /// time apply together; a signal that returns to its old value in one step is not written.
    fn apply(&mut self, events: &mut [Event], sink: &mut impl Sink) -> Result<(), SynthError> {
        // The sort is stable, so events with equal times keep their order.
        events.sort_by_key(|e| e.time);
        sink.segment_start()?;
        let mut i = 0;
        while i < events.len() {
            let time = events[i].time;
            self.group += 1;
            self.touched.clear();
            while i < events.len() && events[i].time == time {
                let e = events[i];
                if self.stamp[e.sig] != self.group {
                    self.stamp[e.sig] = self.group;
                    self.touched.push((e.sig, self.values[e.sig]));
                }
                self.values[e.sig] ^= e.mask;
                i += 1;
            }
            let mut started = false;
            for &(sig, old) in &self.touched {
                let new = self.values[sig];
                if new == old {
                    continue;
                }
                if !started {
                    sink.time(time)?;
                    self.steps += 1;
                    self.last = time;
                    started = true;
                }
                sink.change(sig, new)?;
                self.toggles[sig] += u64::from((old ^ new).count_ones());
            }
        }
        Ok(())
    }
}

/// What a batch wrote: the toggles per signal, the classes, and the peak number of events.
struct BatchResult {
    toggles: Vec<u64>,
    classes: [u64; 2],
    peak_events: usize,
    steps: u64,
    last: u64,
}

/// Writes all segments of one batch to `sink`. `extra` ticks after the last event add one empty
/// time step (used to work around a bug of `fst-writer`).
fn write_batch(
    spec: &SynthSpec,
    layout: &Layout,
    batch: usize,
    extra: u64,
    sink: &mut impl Sink,
) -> Result<BatchResult, SynthError> {
    let mut rng = Rng::new(spec.seed, batch);
    let n = layout.decls.len();
    let mut state = State {
        values: vec![0; n],
        toggles: vec![0; n],
        stamp: vec![0; n],
        group: 0,
        touched: Vec::new(),
        steps: 0,
        last: 0,
    };
    let mut classes = [0u64; 2];
    let mut peak_events = 0;
    let mut events = Vec::new();
    let length = spec.segment_ticks();
    let start = first_segment(spec.period);
    for trace in 0..spec.traces {
        let class = spec.label(batch, trace);
        classes[class as usize] += 1;
        events.clear();
        let base = start + trace as u64 * length;
        segment_events(
            spec,
            layout,
            (batch, trace),
            base,
            class,
            &mut rng,
            &mut events,
        );
        peak_events = peak_events.max(events.len());
        state.apply(&mut events, sink)?;
    }
    // One more clock cycle closes the last bin of the last segment.
    let end = start + spec.traces as u64 * length + spec.phase;
    events.clear();
    for time in [end, end + spec.period / 2] {
        events.push(Event {
            time,
            sig: 0,
            mask: 1,
        });
    }
    state.apply(&mut events, sink)?;
    if extra > 0 {
        let time = state.last + extra;
        sink.time(time)?;
        state.steps += 1;
        state.last = time;
    }
    Ok(BatchResult {
        toggles: state.toggles,
        classes,
        peak_events,
        steps: state.steps,
        last: state.last,
    })
}

/// Writes the waveform of one batch. Retries with one more time step at the end if the FST
/// time table is unreadable (a bug of `fst-writer` 0.3.1).
fn write_wave(
    spec: &SynthSpec,
    layout: &Layout,
    batch: usize,
    path: &Path,
) -> Result<BatchResult, SynthError> {
    match spec.format {
        Format::Vcd => {
            let mut sink = VcdSink::create(path, &layout.decls)?;
            let result = write_batch(spec, layout, batch, 0, &mut sink)?;
            sink.finish()?;
            Ok(result)
        }
        Format::Fst => {
            for extra in 0..8 {
                let mut sink = FstSink::create(path, &layout.decls)?;
                let result = write_batch(spec, layout, batch, extra, &mut sink)?;
                sink.finish()?;
                if fst_time_table_ok(path, result.steps, result.last) {
                    return Ok(result);
                }
            }
            Err(SynthError::Fst(format!(
                "{} has an unreadable time table",
                path.display()
            )))
        }
    }
}

/// Writes the metadata of a batch: `trace_filename`, `clock_period`, and one marker
/// `[start, end, label]` for each trace.
fn write_meta(
    spec: &SynthSpec,
    batch: usize,
    wave_name: &str,
    path: &Path,
) -> Result<(), SynthError> {
    let mut out = flate2::write::GzEncoder::new(
        std::io::BufWriter::new(std::fs::File::create(path)?),
        flate2::Compression::default(),
    );
    write!(
        out,
        "{{\"trace_filename\": \"{wave_name}\", \"clock_period\": {}, \"markers\": [",
        spec.period
    )?;
    let length = spec.segment_ticks();
    for trace in 0..spec.traces {
        let start = first_segment(spec.period) + trace as u64 * length;
        let label = spec.label(batch, trace);
        let separator = if trace == 0 { "" } else { ", " };
        write!(out, "{separator}[{start}, {}, {label}]", start + length)?;
    }
    write!(out, "]}}")?;
    out.finish()?.flush()?;
    Ok(())
}

/// Writes the waveforms, the metadata, and `meta.list` of `spec` into `dir` (the batch `i` in
/// `dir/batch<i>/`). The files depend only on the spec.
pub fn generate(spec: &SynthSpec, dir: &Path) -> Result<SynthOutput, SynthError> {
    spec.validate()?;
    std::fs::create_dir_all(dir)?;
    let layout = Layout::new(spec);
    let wave_name = match spec.format {
        Format::Fst => "wave.fst",
        Format::Vcd => "wave.vcd",
    };
    let mut out = SynthOutput {
        meta_list: dir.join("meta.list"),
        metas: Vec::new(),
        waves: Vec::new(),
        stats: SynthStats {
            scope_toggles: vec![0; spec.scopes],
            tb_toggles: 0,
            clock_toggles: 0,
            class_counts: [0; 2],
            peak_events: 0,
        },
    };
    let mut list = String::new();
    for batch in 0..spec.batches {
        let batch_dir = dir.join(format!("batch{batch}"));
        std::fs::create_dir_all(&batch_dir)?;
        let wave = batch_dir.join(wave_name);
        let result = write_wave(spec, &layout, batch, &wave)?;
        let meta = batch_dir.join("meta.json.gz");
        write_meta(spec, batch, wave_name, &meta)?;
        list.push_str(&format!("batch{batch}/meta.json.gz\n"));
        let stats = &mut out.stats;
        stats.clock_toggles += result.toggles[..layout.tb_first].iter().sum::<u64>();
        stats.tb_toggles += result.toggles[layout.tb_first..layout.dut_first]
            .iter()
            .sum::<u64>();
        for (i, toggles) in result.toggles[layout.dut_first..].iter().enumerate() {
            stats.scope_toggles[i / spec.registers] += toggles;
        }
        for (total, n) in stats.class_counts.iter_mut().zip(result.classes) {
            *total += n;
        }
        stats.peak_events = stats.peak_events.max(result.peak_events);
        out.metas.push(meta);
        out.waves.push(wave);
    }
    std::fs::write(&out.meta_list, list)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::power::edges::EdgeKind;

    #[test]
    fn the_noise_mask_has_the_density_on_average() {
        let mut rng = Rng::new(3, 0);
        let bits: u32 = (0..20_000)
            .map(|_| noise_mask(&mut rng, 8, 0.25).count_ones())
            .sum();
        let rate = f64::from(bits) / (20_000.0 * 8.0);
        assert!((rate - 0.25).abs() < 0.01, "{rate}");
        assert_eq!(noise_mask(&mut rng, 8, 0.0), 0);
        assert_eq!(noise_mask(&mut rng, 8, 1.0), 0xFF);
    }

    #[test]
    fn the_leak_distributions_have_the_planted_moments() {
        let mut rng = Rng::new(4, 0);
        let mut moments = |kind, strength, class| {
            let n = 40_000;
            let t: Vec<f64> = (0..n)
                .map(|_| f64::from(leak_mask(kind, strength, class, 8, &mut rng).count_ones()))
                .collect();
            let mean = t.iter().sum::<f64>() / n as f64;
            let central = |k: i32| t.iter().map(|x| (x - mean).powi(k)).sum::<f64>() / n as f64;
            (mean, central(2), central(3), central(4))
        };
        // Equal3: mean 4, variance 4, third moment 0 in both classes; fourth 16 and 64.
        let (a, b) = (
            moments(LeakKind::Equal3, 0.0, 0),
            moments(LeakKind::Equal3, 0.0, 1),
        );
        for m in [a, b] {
            assert!((m.0 - 4.0).abs() < 0.05 && (m.1 - 4.0).abs() < 0.1 && m.2.abs() < 0.3);
        }
        assert!(
            (a.3 - 16.0).abs() < 0.5 && (b.3 - 64.0).abs() < 3.0,
            "{a:?} {b:?}"
        );
        // Variance: equal means, variances 2/3 and 2 a^2 / 3.
        let (a, b) = (
            moments(LeakKind::Variance, 3.0, 0),
            moments(LeakKind::Variance, 3.0, 1),
        );
        assert!((a.0 - 4.0).abs() < 0.02 && (b.0 - 4.0).abs() < 0.05);
        assert!(
            (a.1 - 2.0 / 3.0).abs() < 0.03 && (b.1 - 6.0).abs() < 0.15,
            "{a:?} {b:?}"
        );
        // The smallest valid strength, 2: the variances are 2/3 and 8/3.
        let (a, b) = (
            moments(LeakKind::Variance, 2.0, 0),
            moments(LeakKind::Variance, 2.0, 1),
        );
        assert!(
            (a.1 - 2.0 / 3.0).abs() < 0.03 && (b.1 - 8.0 / 3.0).abs() < 0.08,
            "{a:?} {b:?}"
        );
        // Mean: the means differ by the strength.
        let (a, b) = (
            moments(LeakKind::Mean, 1.5, 0),
            moments(LeakKind::Mean, 1.5, 1),
        );
        assert!((b.0 - a.0 - 1.5).abs() < 0.03, "{a:?} {b:?}");
        // Deterministic: no variance.
        let d = moments(LeakKind::Deterministic, 1.0, 1);
        assert_eq!((d.0, d.1), (5.0, 0.0));
    }

    #[test]
    fn the_fst_time_table_check_accepts_files_with_and_without_the_extra_step() {
        let spec = SynthSpec {
            traces: 5,
            ..SynthSpec::default()
        };
        let layout = Layout::new(&spec);
        let dir = tempfile::tempdir().unwrap();
        for extra in [0, 1] {
            let path = dir.path().join(format!("w{extra}.fst"));
            let mut sink = FstSink::create(&path, &layout.decls).unwrap();
            let result = write_batch(&spec, &layout, 0, extra, &mut sink).unwrap();
            sink.finish().unwrap();
            assert!(fst_time_table_ok(&path, result.steps, result.last));
            assert!(!fst_time_table_ok(&path, result.steps + 1, result.last));
        }
    }

    #[test]
    fn edge_kinds_have_their_update_offsets() {
        let spec = SynthSpec {
            edge: EdgeKind::Both,
            ..SynthSpec::default()
        };
        assert_eq!(spec.update_offsets(0, 2), [20, 25]);
    }
}
