//! The analysis state that the `tvla` and `plot` binaries share: the fold of batches into a
//! histogram accumulator, the max-|t| curve, and the `t_values.npz` file.

use crate::batch::LengthPolicy;
use crate::stats::threshold::CONVENTIONAL;
use crate::stats::{Binning, HistAccumulator, TestOptions, TestResult};
use log::{error, warn};
use miette::{IntoDiagnostic, WrapErr, miette};
use ndarray::{Array1, Array2, s};
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
    if hist.class_count(0) == 0 || hist.class_count(1) == 0 {
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
    hist.test_pair(0, 1, &TestOptions::default())
        .into_diagnostic()
        .wrap_err("cannot compute the chi-squared test")
}

/// The state of the analysis. Batches are added one by one, in the order of the meta list.
pub struct Fold {
    pub order: usize,
    pub chi2: bool,
    /// What to do with a batch whose traces have another length than those of the first batch.
    /// See [`LengthPolicy`].
    pub policy: LengthPolicy,
    /// True if the fold computes the results and the curves after every batch.
    curves: bool,
    pub hist: Option<HistAccumulator>,
    /// The number of samples per trace, set by the first batch.
    pub samples: usize,
    /// Max |t| per order after each batch, starting with 0.0 for no trace.
    pub max_t: Vec<Vec<f64>>,
    /// Max -log10(p) after each batch, starting with 0.0 for no trace.
    pub max_chi2: Vec<f64>,
    /// The number of traces after each batch, starting with 0.
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

    /// Adds one batch and records the results so far (the curves of the maxima, and the
    /// t-values and chi-squared results of all batches so far).
    pub fn add(&mut self, batch: Loaded) -> miette::Result<()> {
        let Loaded {
            metadata,
            source,
            traces,
            labels,
        } = batch;
        let (num_traces, cur_samples_per_trace) = traces.dim();
        if num_traces <= 1 {
            return Err(miette!(
                "the batch {} has {num_traces} traces; a t-test needs at least two traces",
                metadata.display()
            ));
        }
        if labels.len() != num_traces {
            return Err(miette!(
                "the batch {} has {num_traces} traces but {} labels",
                metadata.display(),
                labels.len()
            ));
        }
        if self.samples == 0 {
            // The first batch sets the number of samples per trace.
            self.samples = cur_samples_per_trace;
        }
        let traces = if self.samples == cur_samples_per_trace {
            traces
        } else {
            match self.policy {
                LengthPolicy::Error => {
                    return Err(miette!(
                        "the batch {} has {cur_samples_per_trace} samples per trace, but the first \
                         batch has {}. Use --length-policy pad or truncate to accept this",
                        metadata.display(),
                        self.samples
                    ));
                }
                LengthPolicy::Truncate if cur_samples_per_trace < self.samples => {
                    return Err(miette!(
                        "the batch {} has {cur_samples_per_trace} samples per trace, fewer than \
                         the {} of the first batch. The policy truncate cannot make traces \
                         longer. Use --length-policy pad to pad them with zeros",
                        metadata.display(),
                        self.samples
                    ));
                }
                LengthPolicy::Truncate => traces.slice(s![.., ..self.samples]).to_owned(),
                LengthPolicy::Pad => {
                    error!(
                        "Inconsistent number of samples per trace: expected {}, found {}",
                        self.samples, cur_samples_per_trace
                    );
                    if cur_samples_per_trace > self.samples {
                        warn!(
                            "Using the first {} samples of the longer trace",
                            self.samples
                        );
                        traces.slice(s![.., ..self.samples]).to_owned()
                    } else {
                        warn!(
                            "padding the traces with {cur_samples_per_trace} samples with zeros up to {}",
                            self.samples
                        );
                        let mut t = Array2::<f32>::zeros((num_traces, self.samples));
                        for (i, row) in traces.outer_iter().enumerate() {
                            t.slice_mut(s![i, ..row.len()]).assign(&row);
                        }
                        t
                    }
                }
            }
        };
        self.num_traces
            .push(self.num_traces.last().copied().unwrap_or(0) + num_traces);

        let samples = self.samples;
        let hist = self
            .hist
            .get_or_insert_with(|| HistAccumulator::new(samples, Binning::Exact));
        hist.update(traces.view(), labels.view())
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot add the batch {}", metadata.display()))?;
        if hist.rejected() > 0 {
            return Err(miette!(
                "the traces from {} (batch {}) have {} values that are not integers or are \
                 2^53 or larger. The statistics need integer-valued traces",
                source.display(),
                metadata.display(),
                hist.rejected()
            ));
        }

        if self.curves {
            self.compute(true)?;
        }
        Ok(())
    }

    /// Computes the results of all batches added so far. Call it once after the last batch if
    /// the fold was made with [`Fold::without_curves`]. A fold with curves has them already.
    pub fn finish(&mut self) -> miette::Result<()> {
        if !self.curves && self.hist.is_some() {
            self.compute(false)?;
        }
        Ok(())
    }

    /// Computes the t-values and the chi-squared results from the accumulator. With `record`,
    /// it also appends the maxima to the curves.
    fn compute(&mut self, record: bool) -> miette::Result<()> {
        let hist = self.hist.as_ref().expect("a batch was added");
        let t_values = hist
            .t_values(0, 1, self.order)
            .into_diagnostic()
            .wrap_err("cannot compute the t-values")?;
        if record {
            for (max_t, t_row) in self.max_t.iter_mut().zip(t_values.rows()) {
                max_t.push(max_abs_finite(t_row.iter().copied()));
            }
        }
        self.t_values = Some(t_values);
        if self.chi2 {
            let results = chi2_results(hist)?;
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
        for (policy, longer, shorter) in [
            (LengthPolicy::Pad, true, true),
            (LengthPolicy::Truncate, true, false),
            (LengthPolicy::Error, false, false),
        ] {
            let mut fold = Fold::new(1, false);
            fold.policy = policy;
            fold.add(loaded(3)).unwrap();
            assert_eq!(fold.add(loaded(5)).is_ok(), longer, "{policy:?} longer");
            assert_eq!(fold.add(loaded(2)).is_ok(), shorter, "{policy:?} shorter");
            assert_eq!(fold.samples, 3);
            assert!(fold.add(loaded(3)).is_ok(), "{policy:?} same");
        }
        let mut fold = Fold::new(1, false);
        fold.policy = LengthPolicy::Error;
        fold.add(loaded(3)).unwrap();
        let message = fold.add(loaded(5)).unwrap_err().to_string();
        assert!(message.contains("5 samples per trace"), "{message}");
        assert!(message.contains("first batch has 3"), "{message}");
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
    fn max_abs_finite_skips_values_that_are_not_finite() {
        assert_eq!(max_abs_finite([1.0, -3.0, 2.0]), 3.0);
        assert_eq!(max_abs_finite([f64::NAN, -2.0, f64::INFINITY]), 2.0);
        assert!(max_abs_finite([f64::NAN, f64::NEG_INFINITY]).is_nan());
        assert!(max_abs_finite([]).is_nan());
    }
}
