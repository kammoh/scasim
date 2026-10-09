//! The analysis state that the `tvla` and `plot` binaries share: the fold of batches into a
//! histogram accumulator, the max-|t| curve, and the `t_values.npz` file.

use crate::batch::LengthPolicy;
use crate::stats::threshold::CONVENTIONAL;
use crate::stats::{Binning, HistAccumulator, TestOptions, TestResult};

use miette::{IntoDiagnostic, WrapErr, miette};
use ndarray::{Array1, Array2};
use ndarray_npz::NpzWriter;
use std::path::{Path, PathBuf};

/// The largest |t| in a row of t-values, or NaN if no value is finite. Values that are not
/// finite (NaN or infinite) are skipped.
pub fn max_abs_finite(row: impl IntoIterator<Item = f64>) -> f64 {
    row.into_iter()
        .filter(|x| x.is_finite())
        .map(f64::abs)
        .fold(f64::NAN, f64::max)
}

/// The largest |t|, including infinity. Only undefined values are skipped.
pub fn max_abs_defined(row: impl IntoIterator<Item = f64>) -> f64 {
    row.into_iter()
        .filter(|x| !x.is_nan())
        .map(f64::abs)
        .fold(f64::NAN, f64::max)
}

/// The traces and labels of one batch, and where they came from.
pub struct Loaded {
    /// The metadata file of the batch.
    pub metadata: PathBuf,
    /// The file that the traces were read from or computed from: `traces.npz` or the waveform.
    pub source: PathBuf,
    pub traces: Array2<f32>,
    pub labels: Array1<u16>,
}

/// The chi-squared results for the classes 0 and 1. A class without traces gives results that
/// are not valid tests, so a batch with one class only is not an error.
pub fn chi2_results(hist: &HistAccumulator) -> miette::Result<Vec<TestResult>> {
    chi2_pair_results(hist, [0, 1])
}

fn chi2_pair_results(hist: &HistAccumulator, pair: [u16; 2]) -> miette::Result<Vec<TestResult>> {
    if hist.class_count(pair[0]) == 0 || hist.class_count(pair[1]) == 0 {
        let none = TestResult {
            statistic: 0.0,
            dof: 0,
            neg_log10_p: 0.0,
            n: 0,
            rows: 0,
            columns: 0,
            merged: 0,
            min_expected: 0.0,
        };
        return Ok(vec![none; hist.n_samples()]);
    }
    hist.test_pair(pair[0], pair[1], &TestOptions::default())
        .into_diagnostic()
        .wrap_err("cannot compute the chi-squared test")
}

// Consumers borrow the same histogram. The pipeline updates it once per batch.
struct AnalysisPlan {
    pair: [u16; 2],
    order: usize,
}
trait AnalysisConsumer {
    type Report;
    fn init(plan: AnalysisPlan) -> Self;
    fn update(&self, hist: &mut HistAccumulator, batch: &Loaded) -> miette::Result<()> {
        hist.update(batch.traces.view(), batch.labels.view())
            .into_diagnostic()?;
        Ok(())
    }
    fn merge(&self, hist: &mut HistAccumulator, other: &HistAccumulator) -> miette::Result<()> {
        hist.merge(other).into_diagnostic()
    }
    fn finalize(&self, hist: &HistAccumulator) -> miette::Result<Self::Report>;
}
struct Tvla(AnalysisPlan);
struct ChiSquared(AnalysisPlan);
impl AnalysisConsumer for Tvla {
    type Report = Array2<f64>;
    fn init(plan: AnalysisPlan) -> Self {
        Self(plan)
    }
    fn finalize(&self, hist: &HistAccumulator) -> miette::Result<Self::Report> {
        hist.t_values(self.0.pair[0], self.0.pair[1], self.0.order)
            .into_diagnostic()
    }
}
impl AnalysisConsumer for ChiSquared {
    type Report = Vec<TestResult>;
    fn init(plan: AnalysisPlan) -> Self {
        Self(plan)
    }
    fn finalize(&self, hist: &HistAccumulator) -> miette::Result<Self::Report> {
        chi2_pair_results(hist, self.0.pair)
    }
}

/// The state of the analysis. Batches are added one by one, in the order of the meta list.
pub struct Fold {
    pub order: usize,
    pub chi2: bool,
    /// What to do with a batch whose traces have another length than the batches before it.
    /// See [`LengthPolicy`]. With `Truncate`, `samples` is the shortest length so far.
    pub policy: LengthPolicy,
    /// True if the fold computes the results and the curves after every batch.
    curves: bool,
    curve_every: usize,
    batches: usize,
    pub pair: [u16; 2],
    /// Trace counts at curve checkpoints. They include only the selected pair.
    pub curve_traces: Vec<usize>,
    /// Undefined t-values per order at each checkpoint.
    pub undefined_t: Vec<Vec<usize>>,
    pub hist: Option<HistAccumulator>,
    /// The current normalized number of samples per trace.
    pub samples: usize,
    /// Max |t| per order after each batch, starting with 0.0 for no trace.
    pub max_t: Vec<Vec<f64>>,
    /// Max -log10(p) after each batch, starting with 0.0 for no trace.
    pub max_chi2: Vec<f64>,
    /// The number of traces of the selected pair after each batch, starting with 0.
    pub num_traces: Vec<usize>,
    pub t_values: Option<Array2<f64>>,
    pub chi2_results: Option<Vec<TestResult>>,
}

impl Fold {
    pub fn new(order: usize, chi2: bool) -> Self {
        Fold {
            order,
            chi2,
            policy: LengthPolicy::Pad,
            curves: true,
            curve_every: 1,
            batches: 0,
            pair: [0, 1],
            curve_traces: vec![0],
            undefined_t: vec![vec![0]; order],
            hist: None,
            samples: 0,
            max_t: vec![vec![0.0]; order],
            max_chi2: vec![0.0],
            num_traces: vec![0],
            t_values: None,
            chi2_results: None,
        }
    }

    /// A fold that only collects the batches. It has no curves, so it does less work for each
    /// batch. Call [`Fold::finish`] to get `t_values` and `chi2_results`.
    pub fn without_curves(order: usize, chi2: bool) -> Self {
        Fold {
            curves: false,
            ..Fold::new(order, chi2)
        }
    }

    /// Sets the curve checkpoints: every batch, every K batches, or final results only.
    pub fn set_curve(&mut self, value: &str) -> miette::Result<()> {
        match value {
            "final" => self.curves = false,
            "every" => {
                self.curves = true;
                self.curve_every = 1;
            }
            _ => {
                let k = value
                    .strip_prefix("every:")
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&k| k > 0)
                    .ok_or_else(|| miette!("--curve must be every, every:K (K > 0), or final"))?;
                self.curves = true;
                self.curve_every = k;
            }
        }
        Ok(())
    }

    /// Adds one batch to the shared histogram.
    pub fn add(&mut self, batch: Loaded) -> miette::Result<()> {
        if batch.labels.len() != batch.traces.nrows() {
            return Err(miette!(
                "{}: traces and labels differ in length",
                batch.metadata.display()
            ));
        }
        let mut hist = HistAccumulator::new(batch.traces.ncols(), Binning::Exact);
        let consumer = Tvla::init(AnalysisPlan {
            pair: self.pair,
            order: self.order,
        });
        consumer
            .update(&mut hist, &batch)
            .wrap_err_with(|| format!("cannot add the batch {}", batch.metadata.display()))?;
        if hist.rejected() > 0 {
            return Err(miette!(
                "the traces from {} (batch {}) have {} values that are not integers or are 2^53 or larger. The statistics need integer-valued traces",
                batch.source.display(),
                batch.metadata.display(),
                hist.rejected()
            ));
        }
        self.add_histogram(hist)
            .map_err(|e| miette!("cannot add the batch {}: {e}", batch.metadata.display()))
    }

    /// Adds a validated batch histogram. Normal runs and cache merges share this path.
    pub fn add_histogram(&mut self, mut other: HistAccumulator) -> miette::Result<()> {
        other.validate().into_diagnostic()?;
        if other.rejected() != 0 {
            return Err(miette!("the histogram has rejected values"));
        }
        if let Some(hist) = self.hist.as_mut() {
            let next = other.n_samples();
            let current = hist.n_samples();
            if current != next {
                match self.policy {
                    LengthPolicy::Error => {
                        return Err(miette!(
                            "the batch has {next} samples per trace, but the first batch has {current}. Use --length-policy pad or truncate to accept this"
                        ));
                    }
                    LengthPolicy::Truncate => {
                        let n = current.min(next);
                        hist.truncate_samples(n).into_diagnostic()?;
                        other.truncate_samples(n).into_diagnostic()?;
                    }
                    LengthPolicy::Pad => {
                        let n = current.max(next);
                        hist.pad_samples(n).into_diagnostic()?;
                        other.pad_samples(n).into_diagnostic()?;
                    }
                }
            }
            Tvla::init(AnalysisPlan {
                pair: self.pair,
                order: self.order,
            })
            .merge(hist, &other)?;
        } else {
            self.hist = Some(other);
        }
        let hist = self.hist.as_ref().expect("the histogram was set");
        self.samples = hist.n_samples();
        let count = hist
            .class_count(self.pair[0])
            .checked_add(hist.class_count(self.pair[1]))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| miette!("trace count overflow"))?;
        self.num_traces.push(count);
        self.batches += 1;
        self.t_values = None;
        self.chi2_results = None;
        if self.curves && self.batches.is_multiple_of(self.curve_every) {
            self.compute(true)?;
        }
        Ok(())
    }

    /// Computes final results and records the last checkpoint if it is due.
    pub fn finish(&mut self) -> miette::Result<()> {
        if self.hist.is_some() && self.t_values.is_none() {
            self.compute(self.curves)?;
        }
        Ok(())
    }

    /// Computes the t-values and the chi-squared results from the accumulator. With `record`,
    /// it also appends the maxima to the curves.
    fn compute(&mut self, record: bool) -> miette::Result<()> {
        let hist = self.hist.as_ref().expect("a batch was added");
        let t_values = Tvla::init(AnalysisPlan {
            pair: self.pair,
            order: self.order,
        })
        .finalize(hist)
        .wrap_err("cannot compute the t-values")?;
        if record {
            for (max_t, t_row) in self.max_t.iter_mut().zip(t_values.rows()) {
                max_t.push(max_abs_defined(t_row.iter().copied()));
            }
        }
        if record {
            self.curve_traces
                .push(*self.num_traces.last().unwrap_or(&0));
            for (undefined, row) in self.undefined_t.iter_mut().zip(t_values.rows()) {
                undefined.push(row.iter().filter(|t| t.is_nan()).count());
            }
        }
        self.t_values = Some(t_values);
        if self.chi2 {
            let results = ChiSquared::init(AnalysisPlan {
                pair: self.pair,
                order: self.order,
            })
            .finalize(hist)?;
            if record {
                self.max_chi2
                    .push(crate::stats::summarize(&results, CONVENTIONAL).max_neg_log10_p);
            }
            self.chi2_results = Some(results);
        }
        Ok(())
    }
}

/// Reads file paths from a list file, one per line. Relative paths are relative to the
/// directory of the list file. Empty lines are skipped.
pub fn read_path_list(list_path: &Path) -> miette::Result<Vec<PathBuf>> {
    let root = list_path.parent().unwrap_or(Path::new(""));
    let text = std::fs::read_to_string(list_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot read the list file {}", list_path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let path = PathBuf::from(line);
            if path.is_absolute() {
                path
            } else {
                root.join(path)
            }
        })
        .collect())
}

/// Creates the output directory if it does not exist.
pub fn create_output_dir(dir: &Path) -> miette::Result<()> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot create the directory {}", dir.display()))?;
    }
    Ok(())
}

/// Writes the t-values (shape `(d, samples)`) to `t_values.npz` in `output_dir`, as the array
/// `t_values`. The directory must exist. Returns the path of the file.
pub fn write_t_values_npz(output_dir: &Path, t_values: &Array2<f64>) -> miette::Result<PathBuf> {
    let npz_path = output_dir.join("t_values.npz");
    log::info!("Saving t-test results to {}", npz_path.display());
    let mut npz = NpzWriter::new_compressed(
        std::fs::File::create(&npz_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot create {}", npz_path.display()))?,
    );
    npz.add_array("t_values", t_values)
        .into_diagnostic()
        .wrap_err("cannot write the t-values")?;
    npz.finish()
        .into_diagnostic()
        .wrap_err("cannot write the t-values")?;
    log::info!("Saved t_values to {}", npz_path.display());
    Ok(npz_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    fn loaded(samples: usize) -> Loaded {
        Loaded {
            metadata: PathBuf::from("m.json"),
            source: PathBuf::from("w.fst"),
            traces: Array2::from_shape_fn((4, samples), |(i, j)| (i * 3 + j) as f32),
            labels: Array1::from_vec(vec![0, 1, 0, 1]),
        }
    }

    #[test]
    fn the_length_policy_applies_to_a_batch_of_another_length() {
        // Pad follows the longest batch. Truncate follows the shortest batch so far.
        // Error accepts only the length of the first batch.
        for (policy, longer, shorter, samples_after) in [
            (LengthPolicy::Pad, true, true, 5),
            (LengthPolicy::Truncate, true, true, 2),
            (LengthPolicy::Error, false, false, 3),
        ] {
            let mut fold = Fold::new(1, false);
            fold.policy = policy;
            fold.add(loaded(3)).unwrap();
            assert_eq!(fold.add(loaded(5)).is_ok(), longer, "{policy:?} longer");
            assert_eq!(fold.add(loaded(2)).is_ok(), shorter, "{policy:?} shorter");
            assert_eq!(fold.samples, samples_after, "{policy:?}");
            assert!(fold.add(loaded(samples_after)).is_ok(), "{policy:?} same");
            assert!(fold.add(loaded(4)).is_ok() || policy == LengthPolicy::Error);
        }
        let mut fold = Fold::new(1, false);
        fold.policy = LengthPolicy::Error;
        fold.add(loaded(3)).unwrap();
        let message = fold.add(loaded(5)).unwrap_err().to_string();
        assert!(message.contains("5 samples per trace"), "{message}");
        assert!(message.contains("first batch has 3"), "{message}");
    }

    /// 40 traces with two classes and values that depend on the sample and the class.
    fn varied(samples: usize, seed: usize) -> Loaded {
        Loaded {
            metadata: PathBuf::from("m.json"),
            source: PathBuf::from("w.fst"),
            traces: Array2::from_shape_fn((40, samples), |(i, j)| {
                ((i * 31 + j * 17 + seed * 13 + (i % 2) * j * (seed + 1)) % 7) as f32
            }),
            labels: Array1::from_iter((0..40).map(|i| (i % 2) as u16)),
        }
    }

    /// The final t-values and chi-squared results as bits, for an exact comparison.
    fn final_bits(fold: &Fold) -> (Vec<u64>, Vec<u64>) {
        let t = fold.t_values.as_ref().unwrap().iter().map(|v| v.to_bits());
        let chi2 = fold.chi2_results.as_ref().unwrap();
        let p = chi2.iter().map(|r| r.neg_log10_p.to_bits());
        (t.collect(), p.collect())
    }

    #[test]
    fn truncate_gives_the_same_results_in_every_order_of_the_batches() {
        // Batches of 5, 3, and 4 samples.
        let batches = [(5, 1), (3, 2), (4, 3)];
        let run = |order: &[usize]| {
            let mut fold = Fold::new(2, true);
            fold.policy = LengthPolicy::Truncate;
            for &i in order {
                fold.add(varied(batches[i].0, batches[i].1)).unwrap();
            }
            fold
        };
        // The reference: every batch cut to the global minimum of 3 samples.
        let mut reference = Fold::new(2, true);
        for (samples, seed) in batches {
            let mut b = varied(samples, seed);
            b.traces = b.traces.slice(s![.., ..3]).to_owned();
            reference.add(b).unwrap();
        }
        let want = final_bits(&reference);
        assert!(want.0.iter().any(|&b| f64::from_bits(b).is_finite()));
        for order in [
            [0, 1, 2],
            [1, 0, 2],
            [0, 2, 1],
            [2, 1, 0],
            [1, 2, 0],
            [2, 0, 1],
        ] {
            let fold = run(&order);
            assert_eq!(fold.samples, 3, "{order:?}");
            assert_eq!(final_bits(&fold), want, "{order:?}");
        }
    }

    #[test]
    fn a_fold_without_curves_gives_the_same_final_results() {
        let mut full = Fold::new(2, true);
        let mut lean = Fold::without_curves(2, true);
        for samples in [3, 3, 3] {
            full.add(loaded(samples)).unwrap();
            lean.add(loaded(samples)).unwrap();
        }
        assert!(lean.t_values.is_none() && lean.chi2_results.is_none());
        lean.finish().unwrap();
        let bits = |f: &Fold| -> Vec<u64> {
            f.t_values
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| t.to_bits())
                .collect()
        };
        assert_eq!(bits(&lean), bits(&full));
        assert!(bits(&lean).len() > 1);
        assert_eq!(lean.chi2_results, full.chi2_results);
        assert_eq!(lean.num_traces, full.num_traces);
        // Finishing a fold with curves, or an empty fold, does nothing.
        full.finish().unwrap();
        Fold::without_curves(1, false).finish().unwrap();
    }

    #[test]
    fn curve_checkpoints_and_infinite_maxima() {
        let mut full = Fold::new(2, true);
        let mut spaced = Fold::new(2, true);
        spaced.set_curve("every:2").unwrap();
        for _ in 0..5 {
            full.add(loaded(3)).unwrap();
            spaced.add(loaded(3)).unwrap();
        }
        spaced.finish().unwrap();
        assert_eq!(spaced.curve_traces, vec![0, 8, 16, 20]);
        assert_eq!(spaced.max_t[0].len(), 4);
        assert_eq!(final_bits(&full), final_bits(&spaced));
        assert_eq!(
            max_abs_defined([f64::NAN, f64::NEG_INFINITY]),
            f64::INFINITY
        );
    }

    #[test]
    fn pad_grows_with_zero_counts() {
        let mut fold = Fold::new(2, true);
        fold.add(varied(2, 1)).unwrap();
        fold.add(varied(4, 2)).unwrap();
        assert_eq!(fold.samples, 4);
        assert!(
            fold.hist
                .as_ref()
                .unwrap()
                .histogram(3, 0)
                .iter()
                .any(|&(bin, count)| bin == 0 && count >= 20)
        );
    }

    #[test]
    fn max_abs_finite_skips_values_that_are_not_finite() {
        assert_eq!(max_abs_finite([1.0, -3.0, 2.0]), 3.0);
        assert_eq!(max_abs_finite([f64::NAN, -2.0, f64::INFINITY]), 2.0);
        assert!(max_abs_finite([f64::NAN, f64::NEG_INFINITY]).is_nan());
        assert!(max_abs_finite([]).is_nan());
    }
}
