//! Synthetic waveforms with planted leaks, for ground-truth tests of `tvla`.
//!
//! [`generate`] writes a waveform (FST or VCD) and the legacy metadata (`meta.json.gz`) for each
//! batch of a [`SynthSpec`], and a `meta.list` that names them. The design is `tb.dut.s<i>.r<j>`:
//! `scopes` sub-scopes of `registers` registers, each `width` bits wide. The clock `tb.clk` (and
//! with [`Clock2`], `tb.clk2`) is in the scope `tb`, with `tb_registers` testbench registers
//! outside `tb.dut`.
//!
//! Every register flips bits at its active clock edge, at the exact time of the edge, as a
//! zero-delay RTL simulation does. The number of flips is random (the noise, density
//! [`SynthSpec::density`] for each bit) unless a [`Leak`] plants a class-dependent distribution
//! for one register in one cycle. Glitches flip bits at edge + `delta` and flip them back one
//! tick later, inside the cycle.
//!
//! One segment (one trace) is `cycles` clock periods and starts at a rising edge of `tb.clk` that
//! is a multiple of `period`. The class of a trace is a function of the seed, the batch, and the
//! trace number. The generator keeps the events of one segment in memory, never all traces.
//!
//! This module needs the `synth` feature, because it writes FST files with `fst-writer`. The
//! tests and examples of this package turn the feature on by themselves. A normal build of the
//! library and of `tvla` does not need `fst-writer`.

mod generate;
mod writer;

pub use generate::{SynthOutput, SynthStats, generate};

use crate::power::edges::EdgeKind;
use crate::shuffle::mix;

/// The waveform format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Fst,
    Vcd,
}

/// A second clock `tb.clk2` for the odd-numbered scopes (`s1`, `s3`, ...). Its cycles and phase
/// are relative to the segment start, so every segment is the same. The segment length must be
/// a multiple of `period`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clock2 {
    pub period: u64,
    pub phase: u64,
}

/// A gated clock: in every segment, the clock `tb.clk` has no edge for `stop` cycles in every
/// `every` cycles (the last `stop` cycles of each group). The pattern restarts at each segment,
/// so all traces have the same number of edges. It applies to `tb.clk` only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gating {
    pub stop: u64,
    pub every: u64,
}

/// A glitch: after each active edge, every register flips its `bits` lowest bits at edge +
/// `delta` ticks and flips them back one tick later. The value after the glitch is the value
/// before. A sampling at multiples of the clock period drops these toggles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glitch {
    pub delta: u64,
    pub bits: u32,
}

/// The distribution of the number of toggles `T` of the leaking register in the leak cycle.
/// Let `m = width / 2`, and `u` uniform on {-1, 0, +1}. The noise of the register is replaced
/// in this cycle by:
///
/// - `Mean`: `T = m + u` in class 0 and `T = m + u + x` in class 1, where `x` is `strength`
///   rounded down plus 1 with the probability of the fraction. Order 1 finds it. The strength
///   must be above 0.
/// - `Variance`: `T = m + u` in class 0 and `T = m + a * u` in class 1, with `a = round(strength)`
///   and `a >= 2` (with `a = 1` the classes would be equal). The means are equal. Order 2 finds
///   it, order 1 does not.
/// - `Equal3`: class 0: `T` is 2 or 6, each with probability 1/2. Class 1: `T` is 0 with 1/8,
///   4 with 3/4, and 8 with 1/8. The mean, variance, and skewness are equal, the fourth moment
///   is not. Order 4 and the chi-squared test find it. `strength` is not used. Needs `width >= 8`.
/// - `Deterministic`: `T = m` in class 0 and `T = m + round(strength)` in class 1, without any
///   randomness. `round(strength)` must be at least 1. The t-statistic is infinite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeakKind {
    Mean,
    Variance,
    Equal3,
    Deterministic,
}

/// A planted leak: the register `register` of the sub-scope `scope` (`tb.dut.s<scope>`), in the
/// cycle `cycle` of the segment (counted in cycles of the clock of that scope, from 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leak {
    pub scope: usize,
    pub register: usize,
    pub cycle: usize,
    pub kind: LeakKind,
    pub strength: f64,
}

impl Leak {
    /// The same leak in another cycle.
    pub fn at_cycle(self, cycle: usize) -> Leak {
        Leak { cycle, ..self }
    }
}

/// An error of the generator.
#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    #[error("invalid synth spec: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("cannot write the FST file: {0}")]
    Fst(String),
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, SynthError> {
    Err(SynthError::Invalid(reason.into()))
}

/// The parameters of a synthetic design, its clocks, its noise, and its leaks. All fields have
/// defaults ([`Default`]).
#[derive(Clone, Debug)]
pub struct SynthSpec {
    /// S: the number of sub-scopes `tb.dut.s<i>`.
    pub scopes: usize,
    /// R: the number of registers `r<j>` in each sub-scope.
    pub registers: usize,
    /// W: the width of a register in bits (1 to 64).
    pub width: u32,
    /// The number of testbench registers `tb.t<k>` outside `tb.dut`. They have noise only.
    pub tb_registers: usize,
    /// The period of `tb.clk` in ticks (even).
    pub period: u64,
    /// The offset of the rising edge in the period, in ticks. Below `period / 2`.
    pub phase: u64,
    /// The edges where registers update.
    pub edge: EdgeKind,
    pub clock2: Option<Clock2>,
    pub gating: Option<Gating>,
    pub glitch: Option<Glitch>,
    /// The probability that a bit toggles at an update (the noise).
    pub density: f64,
    /// L: the clock periods in a segment (a trace).
    pub cycles: usize,
    /// N: the traces (segments) in each batch.
    pub traces: usize,
    pub batches: usize,
    pub seed: u64,
    pub leaks: Vec<Leak>,
    pub format: Format,
}

impl Default for SynthSpec {
    fn default() -> Self {
        SynthSpec {
            scopes: 2,
            registers: 2,
            width: 8,
            tb_registers: 1,
            period: 10,
            phase: 0,
            edge: EdgeKind::Rising,
            clock2: None,
            gating: None,
            glitch: None,
            density: 0.5,
            cycles: 6,
            traces: 1000,
            batches: 1,
            seed: 1,
            leaks: Vec::new(),
            format: Format::Fst,
        }
    }
}

/// The time of the first segment, in ticks. It is a multiple of the period, so no edge is at
/// time 0.
pub(crate) fn first_segment(period: u64) -> u64 {
    period
}

impl SynthSpec {
    /// The path of the clock `tb.clk`, for `--clock`.
    pub const CLOCK: &'static str = "tb.clk";

    /// The path of the scope `tb.dut.s<scope>`, for `--per-scope`.
    pub fn scope_path(scope: usize) -> String {
        format!("tb.dut.s{scope}")
    }

    /// The clock domain of a sub-scope: 1 for the odd scopes if there is a second clock.
    pub fn domain_of(&self, scope: usize) -> usize {
        usize::from(self.clock2.is_some() && scope % 2 == 1)
    }

    /// The number of clock domains.
    pub(crate) fn domains(&self) -> usize {
        1 + usize::from(self.clock2.is_some())
    }

    /// The length of a segment in ticks.
    pub fn segment_ticks(&self) -> u64 {
        self.cycles as u64 * self.period
    }

    /// `(period, phase)` of the clock of a domain.
    pub(crate) fn clock_of(&self, domain: usize) -> (u64, u64) {
        match (domain, self.clock2) {
            (1, Some(c)) => (c.period, c.phase),
            _ => (self.period, self.phase),
        }
    }

    /// The cycles of a domain in a segment.
    pub(crate) fn own_cycles(&self, domain: usize) -> u64 {
        self.segment_ticks() / self.clock_of(domain).0
    }

    /// True if the cycle `cycle` of `tb.clk` in a segment has no edge.
    pub(crate) fn gated(&self, cycle: u64) -> bool {
        self.gating
            .is_some_and(|g| cycle % g.every >= g.every - g.stop)
    }

    /// The offsets from the segment start of the update edges in cycle `cycle` of a domain.
    pub(crate) fn update_offsets(&self, domain: usize, cycle: u64) -> Vec<u64> {
        let (period, phase) = self.clock_of(domain);
        let rise = phase + cycle * period;
        match self.edge {
            EdgeKind::Rising => vec![rise],
            EdgeKind::Falling => vec![rise + period / 2],
            EdgeKind::Both => vec![rise, rise + period / 2],
        }
    }

    /// The offsets from the segment start of the edges of `tb.clk` that `tvla --clock tb.clk
    /// --edges <kind>` samples on, in one segment.
    fn sampled_edges(&self, kind: EdgeKind) -> Vec<u64> {
        let mut edges = Vec::new();
        for cycle in (0..self.cycles as u64).filter(|&c| !self.gated(c)) {
            let rise = self.phase + cycle * self.period;
            if kind != EdgeKind::Falling {
                edges.push(rise);
            }
            if kind != EdgeKind::Rising {
                edges.push(rise + self.period / 2);
            }
        }
        edges
    }

    /// The number of samples in a trace in edges mode (`--clock tb.clk --edges <kind>`).
    pub fn samples(&self, kind: EdgeKind) -> usize {
        self.sampled_edges(kind).len()
    }

    /// The sample of a trace that holds the update of a leak, in edges mode with `tb.clk`. A
    /// toggle at the time of an edge belongs to the bin that starts there. `None` if the update
    /// is before the first sampled edge of the segment (then it is in the previous trace).
    pub fn leak_sample(&self, leak: &Leak, kind: EdgeKind) -> Option<usize> {
        let domain = self.domain_of(leak.scope);
        let update = self.update_offsets(domain, leak.cycle as u64)[0];
        let before = self
            .sampled_edges(kind)
            .iter()
            .filter(|&&e| e <= update)
            .count();
        before.checked_sub(1)
    }

    /// The class (0 or 1) of trace `trace` of batch `batch`.
    pub fn label(&self, batch: usize, trace: usize) -> u16 {
        let key = mix(self.seed ^ 0x001A_BE15) ^ ((batch as u64) << 40);
        u16::from(mix(key.wrapping_add(trace as u64)) >> 63 == 1)
    }

    /// Checks the parameters. [`generate`] calls this.
    pub fn validate(&self) -> Result<(), SynthError> {
        let positive = [
            ("scopes", self.scopes),
            ("registers", self.registers),
            ("cycles", self.cycles),
            ("traces", self.traces),
            ("batches", self.batches),
        ];
        for (name, value) in positive {
            if value == 0 {
                return invalid(format!("{name} must be at least 1"));
            }
        }
        if !(1..=64).contains(&self.width) {
            return invalid("width must be from 1 to 64");
        }
        if !(0.0..=1.0).contains(&self.density) {
            return invalid("density must be from 0 to 1");
        }
        for domain in 0..self.domains() {
            let (period, phase) = self.clock_of(domain);
            if period < 2 || period % 2 != 0 {
                return invalid(format!("the period {period} must be even and at least 2"));
            }
            if phase >= period / 2 {
                return invalid(format!("the phase {phase} must be below period / 2"));
            }
            if !self.segment_ticks().is_multiple_of(period) {
                return invalid(format!(
                    "the segment ({} ticks) must be a multiple of the period {period}",
                    self.segment_ticks()
                ));
            }
            if let Some(g) = self.glitch {
                let latest = phase
                    + if self.edge == EdgeKind::Rising {
                        0
                    } else {
                        period / 2
                    };
                if g.delta < 1 || latest + g.delta + 1 >= period {
                    return invalid("a glitch must end inside the clock period");
                }
                if g.bits < 1 || g.bits > self.width {
                    return invalid("glitch bits must be from 1 to the width");
                }
            }
        }
        if let Some(g) = self.gating
            && (g.stop < 1 || g.stop >= g.every)
        {
            return invalid("gating needs 1 <= stop < every");
        }
        for (i, leak) in self.leaks.iter().enumerate() {
            self.validate_leak(leak)?;
            if self.leaks[..i]
                .iter()
                .any(|o| (o.scope, o.register, o.cycle) == (leak.scope, leak.register, leak.cycle))
            {
                return invalid(format!("two leaks at the same place: {leak:?}"));
            }
        }
        Ok(())
    }

    fn validate_leak(&self, leak: &Leak) -> Result<(), SynthError> {
        if leak.scope >= self.scopes || leak.register >= self.registers {
            return invalid(format!("the leak {leak:?} is outside the design"));
        }
        let domain = self.domain_of(leak.scope);
        if leak.cycle as u64 >= self.own_cycles(domain) {
            return invalid(format!("the leak {leak:?} is after the end of the segment"));
        }
        if domain == 0 && self.gated(leak.cycle as u64) {
            return invalid(format!(
                "the leak {leak:?} is in a cycle without clock edge"
            ));
        }
        if !leak.strength.is_finite() || leak.strength < 0.0 {
            return invalid("the strength of a leak must be finite and not negative");
        }
        let (width, m) = (i64::from(self.width), i64::from(self.width / 2));
        let s = leak.strength;
        let fits = match leak.kind {
            LeakKind::Mean => s > 0.0 && m >= 1 && m + 1 + s.ceil() as i64 <= width,
            // With a = 1 both classes have the same distribution.
            LeakKind::Variance => {
                let a = s.round() as i64;
                a >= 2 && m >= a && m + a <= width
            }
            LeakKind::Equal3 => width >= 8,
            LeakKind::Deterministic => {
                let d = s.round() as i64;
                d >= 1 && m + d <= width
            }
        };
        if !fits {
            return invalid(format!(
                "the leak {leak:?} does not fit in {} bits",
                self.width
            ));
        }
        Ok(())
    }
}
