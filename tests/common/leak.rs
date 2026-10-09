//! A batch with a planted leak, for the tests of `tvla`.

use super::fixture::*;
use super::sim::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

/// A batch: a clock `tb.clk` and some 8-bit data signals. Each trace is `cycles` clock periods
/// of 10 ticks. In every period, each data signal toggles 0 to 3 bits at random, whatever the
/// class. The leak: in the period `leak_cycle`, the signal `leak_signal` toggles 0 or 1 bits in
/// class 0 and 2 or 3 bits in class 1.
pub struct LeakSpec {
    /// `(scope, name)` of the data signals, sorted by scope. `tb.clk` is always there.
    pub signals: Vec<(&'static str, &'static str)>,
    pub traces: usize,
    pub cycles: u64,
    pub leak_cycle: u64,
    /// Index into `signals`, or `None` for no leak.
    pub leak_signal: Option<usize>,
    pub seed: u64,
    /// Extra variables (scope, name, index into `signals`): aliases of data signals.
    pub aliases: Vec<(&'static str, &'static str, usize)>,
}

impl Default for LeakSpec {
    fn default() -> Self {
        LeakSpec {
            signals: vec![("tb.dut.a", "x"), ("tb.dut.b", "y")],
            traces: 300,
            cycles: 6,
            leak_cycle: 2,
            leak_signal: Some(0),
            seed: 1,
            aliases: vec![],
        }
    }
}

pub struct LeakBatch {
    pub dir: tempfile::TempDir,
    pub meta: PathBuf,
    /// The class of each trace.
    pub labels: Vec<u16>,
    /// The fixture that was written.
    pub fixture: Fixture,
    cycles: u64,
}

impl LeakBatch {
    /// Writes another metadata file `name` next to `meta.json` for the same waveform. Every
    /// segment starts and ends `shift` ticks later. The last segment is left out, because a
    /// shifted one would end after the last clock edge.
    pub fn write_shifted_meta(&self, name: &str, shift: u64) -> PathBuf {
        let markers: Vec<String> = self
            .labels
            .iter()
            .enumerate()
            .take(self.labels.len() - 1)
            .map(|(i, class)| {
                let start = 10 + i as u64 * self.cycles * 10 + shift;
                format!("[{start}, {}, {class}]", start + self.cycles * 10)
            })
            .collect();
        let path = self.dir.path().join(name);
        std::fs::write(
            &path,
            format!(
                r#"{{"trace_filename": "tvla.vcd", "clock_period": 10, "markers": [{}]}}"#,
                markers.join(", ")
            ),
        )
        .unwrap();
        path
    }
}

/// Writes `tvla.vcd` and `meta.json` into a new directory.
pub fn write_leak_batch(spec: &LeakSpec) -> LeakBatch {
    let mut all: Vec<(&str, &str, u32)> = vec![("tb", "clk", 1)];
    all.extend(spec.signals.iter().map(|&(scope, name)| (scope, name, 8)));
    let mut sim = Sim::new(&all);
    let mut rng = SmallRng::seed_from_u64(spec.seed);
    let periods = spec.traces as u64 * spec.cycles;
    // The clock has one more rising edge than there are periods: it closes the last bin.
    sim.clock(0, 10, 10, periods + 1, false);
    let mut values = vec![0u32; spec.signals.len()];
    let mut labels = Vec::new();
    for trace in 0..spec.traces as u64 {
        let class: u16 = rng.random_range(0..2);
        labels.push(class);
        for cycle in 0..spec.cycles {
            let edge = 10 + (trace * spec.cycles + cycle) * 10;
            for (k, value) in values.iter_mut().enumerate() {
                let bits: u32 = if spec.leak_signal == Some(k) && cycle == spec.leak_cycle {
                    rng.random_range(0..2) + 2 * u32::from(class)
                } else {
                    rng.random_range(0..4)
                };
                if bits > 0 {
                    *value ^= (1 << bits) - 1;
                    sim.set(edge + 3, k + 1, &format!("{:08b}", *value));
                }
            }
        }
    }
    sim.fx.aliases = spec
        .aliases
        .iter()
        .map(|&(scope, name, target)| (scope.to_string(), name.to_string(), target + 1))
        .collect();
    let fixture = sim.finish();
    let dir = tempfile::tempdir().unwrap();
    write_vcd(&dir.path().join("tvla.vcd"), &fixture);
    let markers: Vec<String> = labels
        .iter()
        .enumerate()
        .map(|(i, class)| {
            let start = 10 + i as u64 * spec.cycles * 10;
            format!("[{start}, {}, {class}]", start + spec.cycles * 10)
        })
        .collect();
    let meta = dir.path().join("meta.json");
    std::fs::write(
        &meta,
        format!(
            r#"{{"trace_filename": "tvla.vcd", "clock_period": 10, "markers": [{}]}}"#,
            markers.join(", ")
        ),
    )
    .unwrap();
    LeakBatch {
        dir,
        meta,
        labels,
        fixture,
        cycles: spec.cycles,
    }
}
