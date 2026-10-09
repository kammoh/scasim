//! A small builder for fixtures with a clock and data events.

use super::fixture::*;
use std::collections::BTreeMap;

/// Builds a [`Fixture`] from clock and data events at given times.
pub struct Sim {
    pub fx: Fixture,
    events: BTreeMap<u64, Vec<(usize, String)>>,
}

impl Sim {
    /// Signals as `(scope, name, width)`, sorted by scope.
    pub fn new(signals: &[(&str, &str, u32)]) -> Sim {
        let mut fx = Fixture::flat(&signals.iter().map(|s| s.2).collect::<Vec<_>>());
        for (s, &(scope, name, _)) in fx.signals.iter_mut().zip(signals) {
            s.scope = scope.into();
            s.name = name.into();
        }
        Sim {
            fx,
            events: BTreeMap::new(),
        }
    }

    /// Sets the value of a signal at time 0.
    pub fn initial(&mut self, signal: usize, value: &str) {
        self.fx.initial[signal] = value.into();
    }

    /// Sets the value of a signal at `time` (greater than 0).
    pub fn set(&mut self, time: u64, signal: usize, value: &str) {
        assert!(time > 0);
        self.events
            .entry(time)
            .or_default()
            .push((signal, value.into()));
    }

    /// A clock with a 50 % duty cycle. Rising edges are at `phase + k * period` for `k` in
    /// `0..cycles`, falling edges half a period later. With `start_high`, the clock is high at
    /// time 0 and falls at `phase - period / 2` first (`phase` must be at least `period / 2`).
    pub fn clock(&mut self, signal: usize, phase: u64, period: u64, cycles: u64, start_high: bool) {
        if start_high {
            self.initial(signal, "1");
            self.set(phase - period / 2, signal, "0");
        }
        for k in 0..cycles {
            self.set(phase + k * period, signal, "1");
            self.set(phase + k * period + period / 2, signal, "0");
        }
    }

    pub fn finish(mut self) -> Fixture {
        self.fx.steps = self.events.into_iter().collect();
        self.fx
    }
}
