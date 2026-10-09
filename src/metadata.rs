//! Version 1 metadata and exact time conversion.

use miette::{IntoDiagnostic, WrapErr, miette};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeUnit {
    pub mantissa: u64,
    pub exponent: i32,
}

impl TimeUnit {
    /// Converts an integer time without rounding. Both units must be positive.
    pub fn ticks(&self, time: u64, waveform: &TimeUnit) -> Option<u64> {
        if self.mantissa == 0 || waveform.mantissa == 0 {
            return None;
        }
        if time == 0 {
            return Some(0);
        }
        let mut numerator = u128::from(time).checked_mul(u128::from(self.mantissa))?;
        let mut denominator = u128::from(waveform.mantissa);
        // Cancel factors before scaling, so an exact quotient need not overflow first.
        let gcd = |mut a: u128, mut b: u128| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let divisor = gcd(numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
        let delta = i64::from(self.exponent) - i64::from(waveform.exponent);
        if delta >= 0 {
            // Divide powers of ten by denominator factors as we go.
            for _ in 0..delta.min(128) {
                let divisor = gcd(10, denominator);
                denominator /= divisor;
                numerator = numerator.checked_mul(10 / divisor)?;
            }
            if delta > 128 {
                return None;
            }
        } else {
            for _ in 0..(-delta).min(128) {
                let divisor = gcd(numerator, 10);
                numerator /= divisor;
                denominator = denominator.checked_mul(10 / divisor)?;
            }
            if delta < -128 {
                return None;
            }
        }
        if numerator % denominator != 0 {
            return None;
        }
        u64::try_from(numerator / denominator).ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub id: u64,
    pub start: u64,
    pub end: u64,
    pub label: u16,
    #[serde(default)]
    pub group: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchIdentity {
    pub id: String,
    pub seeds: BTreeMap<String, u64>,
    pub status: String,
    pub design_random: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataV1 {
    pub scasim_meta: u8,
    pub batch: BatchIdentity,
    #[serde(default)]
    pub design: serde_json::Value,
    pub waveform: Option<String>,
    pub time: TimeUnit,
    pub segments: Vec<Segment>,
    pub labels: BTreeMap<u16, String>,
    pub groups: BTreeMap<u64, String>,
    #[serde(default)]
    pub extensions: serde_json::Value,
}

impl MetadataV1 {
    pub fn validate(&self, path: &Path) -> miette::Result<()> {
        if self.batch.status != "committed" {
            return Err(miette!(
                "{}: batch status is {}; only committed batches can be analyzed",
                path.display(),
                self.batch.status
            ));
        }
        if self.batch.id.is_empty() || self.time.mantissa == 0 {
            return Err(miette!(
                "{}: batch id and a positive time mantissa are required",
                path.display()
            ));
        }
        let mut ids = BTreeSet::new();
        let mut last = None;
        for s in &self.segments {
            if s.end <= s.start
                || last.is_some_and(|t| s.start < t)
                || !ids.insert(s.id)
                || !self.labels.contains_key(&s.label)
                || !self.groups.contains_key(&s.group)
            {
                return Err(miette!(
                    "{}: segment {} has invalid times, a duplicate id, or an undeclared label or group",
                    path.display(),
                    s.id
                ));
            }
            last = Some(s.start);
        }
        Ok(())
    }

    pub fn markers(
        &self,
        path: &Path,
        waveform: &TimeUnit,
    ) -> miette::Result<Vec<(u64, u64, u16)>> {
        self.segments
            .iter()
            .map(|s| {
                let convert = |t| {
                    self.time.ticks(t, waveform).ok_or_else(|| {
                        miette!(
                            "{}: segment {} time {t} is inexact or overflows waveform ticks",
                            path.display(),
                            s.id
                        )
                    })
                };
                Ok((convert(s.start)?, convert(s.end)?, s.label))
            })
            .collect()
    }
}

/// Reads the waveform time unit without loading signal values. VCD can have a scale factor.
pub fn waveform_time_unit(path: &Path) -> miette::Result<TimeUnit> {
    if crate::power::is_fst(path).into_diagnostic()? {
        let reader = crate::power::fst::open_reader(path).into_diagnostic()?;
        return Ok(TimeUnit {
            mantissa: 1,
            exponent: i32::from(reader.get_header().timescale_exponent),
        });
    }
    let header = wellen::viewers::read_header_from_file(path, &wellen::LoadOptions::default())
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot read {}", path.display()))?;
    let time = header
        .hierarchy
        .timescale()
        .ok_or_else(|| miette!("{}: missing waveform timescale", path.display()))?;
    let exponent = time
        .unit
        .to_exponent()
        .ok_or_else(|| miette!("{}: unknown waveform time unit", path.display()))?;
    Ok(TimeUnit {
        mantissa: u64::from(time.factor),
        exponent: i32::from(exponent),
    })
}
