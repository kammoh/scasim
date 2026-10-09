//! Pearson chi-squared and G tests of independence on a classes-by-bins contingency table.
//!
//! The table has one row per class and one column per value bin. The null hypothesis is that the
//! class and the bin are independent, that is, all classes have the same value distribution.
//!
//! # Rules
//!
//! 1. **Empty rows and columns are dropped.** A class with no trace in this sample is not part of
//!    the test. A bin with a zero count in every class adds a degree of freedom but no
//!    information, so it is removed (Moradi et al., TCHES 2018, Section 3.1.1).
//! 2. **Optional merging of sparse bins** (`TestOptions::min_expected`). The chi-squared
//!    approximation needs enough expected counts per cell. Let `R_min` be the smallest row total
//!    and `N` the table total. A group of bins must have a column total of at least
//!    `min_expected * N / R_min`. Then every cell has an expected count of at least
//!    `min_expected`. The bins are scanned in increasing order of value. Adjacent bins are merged
//!    until the group is large enough. A small remainder at the end joins the last group.
//!    Merging adjacent bins keeps the order of the values. The rule uses only the row and column
//!    totals, which do not depend on the class assignment under the null hypothesis. Therefore
//!    the merging does not change the null distribution of the statistic.
//! 3. **Degrees of freedom** are `(rows - 1) * (groups - 1)` after rules 1 and 2. If fewer than two
//!    rows or two groups remain, the test is not valid: the statistic is 0, `dof` is 0, and
//!    `-log10(p)` is 0.
//!
//! # Calibration (why the default merge threshold is 20)
//!
//! The chi-squared distribution is only an approximation of the null distribution of the
//! statistic, and the tail matters here (the usual threshold is `p < 1e-5`). `examples/chi2_validity.rs`
//! draws two million null tables per scenario (Binomial, geometric, uniform, and Zipf bin
//! probabilities; 100, 500, and 2000 traces per class) and counts how often `-log10(p) >= k`.
//! The ratio of observed to nominal frequency was:
//!
//! | statistic and merge threshold | observed / nominal |
//! |---|---|
//! | Pearson, no merging | 0.0 to 0.99 (conservative; misses real leaks in sparse tables) |
//! | G, no merging | 0.75 to 700 (many false alarms in sparse tables) |
//! | Pearson, merge at 5 | 0.32 to 1.0 |
//! | G, merge at 5 | 0.95 to 4.2 |
//! | Pearson, merge at 20 | 0.50 to 1.01 |
//! | G, merge at 20 | 0.80 to 1.45 |
//!
//! (Ranges cover the cells of the experiment with at least 10 observed events.)
//!
//! Pearson never gives more false alarms than nominal once sparse bins are merged. G is
//! anti-conservative in sparse tables unless the merge threshold is 20 or more. Therefore the
//! default is Pearson with `min_expected = 20`. The usual rule "expected counts of at least 5"
//! is not enough for the tail of the distribution.
//!
//! The Pearson statistic is `sum (F - E)^2 / E`. The G statistic is `2 sum F ln(F / E)`. Both
//! have the same asymptotic chi-squared distribution under the null hypothesis. Here `E` is the
//! expected count `R_i * C_j / N`.

use serde::{Deserialize, Serialize};

use super::error::StatsError;
use super::special::{bd0, neg_log10_chi2_sf};

/// Which test statistic to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Statistic {
    /// Pearson's chi-squared statistic.
    Pearson,
    /// The likelihood-ratio (G) statistic, as used by PROLEAD.
    G,
}

/// Options of a contingency-table test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestOptions {
    /// The statistic to compute.
    pub statistic: Statistic,
    /// Minimum expected count per cell after merging adjacent bins. Use 0 to disable merging.
    /// See the calibration table in the [module documentation](self).
    pub min_expected: f64,
}

impl Default for TestOptions {
    /// Pearson's statistic with merging at an expected count of 20.
    fn default() -> Self {
        Self {
            statistic: Statistic::Pearson,
            min_expected: 20.0,
        }
    }
}

/// Result of one contingency-table test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestResult {
    /// The value of the test statistic.
    pub statistic: f64,
    /// Degrees of freedom after dropping and merging. Zero if the test is not valid.
    pub dof: u32,
    /// Evidence against the null hypothesis as `-log10(p)`. Computed in the log domain, so it is
    /// finite even when `p` is below the smallest positive `f64`. It is NaN only if the p-value
    /// computation did not converge, which cannot happen for any `u32` degrees of freedom (see
    /// [`ln_gamma_q`](super::special::ln_gamma_q)).
    pub neg_log10_p: f64,
    /// Total count `N` of the rows that took part in the test.
    pub n: u64,
    /// Number of rows (classes) that took part in the test.
    pub rows: u32,
    /// Number of columns (groups of bins) after merging.
    pub columns: u32,
    /// Number of non-empty bins that were merged away (`non-empty bins - columns`).
    pub merged: u32,
    /// Smallest expected cell count after merging. A value below 5 means that the chi-squared
    /// approximation is questionable.
    pub min_expected: f64,
}

impl TestResult {
    /// True if the table had at least two rows and two columns after dropping and merging.
    pub fn is_valid(&self) -> bool {
        self.dof >= 1
    }

    fn invalid(n: u64, rows: usize, nonzero_columns: u32) -> Self {
        Self {
            statistic: 0.0,
            dof: 0,
            neg_log10_p: 0.0,
            n,
            rows: rows as u32,
            columns: nonzero_columns.min(1),
            merged: nonzero_columns.saturating_sub(1),
            min_expected: 0.0,
        }
    }
}

/// Reusable scratch memory for [`test_table`]. Create one per thread.
#[derive(Debug, Default)]
pub struct Workspace {
    live: Vec<usize>,
    row_total: Vec<u64>,
    col_total: Vec<u64>,
    col_group: Vec<u32>,
    cells: Vec<u64>,
}

/// Runs the chi-squared or G test on a contingency table.
///
/// `rows` has one slice per class. All slices must have the same length (the number of bins), and
/// the bins must be in increasing order of value. Rows or columns that are all zero are ignored.
///
/// Returns [`StatsError::RowLengthMismatch`] if the rows have different lengths. The check comes
/// before any total is computed.
pub fn test_table(
    rows: &[&[u32]],
    opts: &TestOptions,
    ws: &mut Workspace,
) -> Result<TestResult, StatsError> {
    let width = rows.first().map_or(0, |r| r.len());
    if let Some((row, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != width) {
        return Err(StatsError::RowLengthMismatch {
            row,
            expected: width,
            got: r.len(),
        });
    }
    Ok(test_equal_rows(rows, width, opts, ws))
}

/// The body of [`test_table`]. All rows have `width` columns.
fn test_equal_rows(
    rows: &[&[u32]],
    width: usize,
    opts: &TestOptions,
    ws: &mut Workspace,
) -> TestResult {
    // Rule 1: drop empty rows.
    ws.live.clear();
    ws.row_total.clear();
    for (i, row) in rows.iter().enumerate() {
        let total: u64 = row.iter().map(|&c| u64::from(c)).sum();
        if total > 0 {
            ws.live.push(i);
            ws.row_total.push(total);
        }
    }
    let r = ws.live.len();
    let n: u64 = ws.row_total.iter().sum();

    // Column totals over the live rows.
    ws.col_total.clear();
    ws.col_total.resize(width, 0);
    for &i in &ws.live {
        for (t, &c) in ws.col_total.iter_mut().zip(rows[i]) {
            *t += u64::from(c);
        }
    }
    let nonzero = ws.col_total.iter().filter(|&&c| c > 0).count() as u32;
    if r < 2 || nonzero < 2 {
        return TestResult::invalid(n, r, nonzero);
    }

    // Rule 2: group adjacent bins until each group reaches the column-total threshold.
    let min_row = *ws.row_total.iter().min().expect("at least two rows");
    let nf = n as f64;
    let threshold = if opts.min_expected > 0.0 {
        opts.min_expected * nf / min_row as f64
    } else {
        0.0
    };
    ws.col_group.clear();
    ws.col_group.resize(width, u32::MAX);
    let mut closed = 0_u32;
    let mut open_total = 0_u64;
    for (group, &c) in ws.col_group.iter_mut().zip(&ws.col_total) {
        if c == 0 {
            continue;
        }
        *group = closed;
        open_total += c;
        if open_total as f64 >= threshold {
            closed += 1;
            open_total = 0;
        }
    }
    let groups = if open_total > 0 && closed == 0 {
        1
    } else {
        if open_total > 0 {
            // The remainder joins the last closed group.
            for group in ws.col_group.iter_mut().filter(|g| **g == closed) {
                *group = closed - 1;
            }
        }
        closed
    };
    if groups < 2 {
        return TestResult::invalid(n, r, nonzero);
    }
    let groups = groups as usize;

    // Merged cell counts, laid out as [group][live row].
    ws.cells.clear();
    ws.cells.resize(groups * r, 0);
    for (k, &i) in ws.live.iter().enumerate() {
        for (&c, &g) in rows[i].iter().zip(&ws.col_group) {
            if c > 0 {
                ws.cells[g as usize * r + k] += u64::from(c);
            }
        }
    }

    // Statistic.
    let raw = if r == 2 {
        two_row_statistic(&ws.cells, &ws.row_total, opts.statistic)
    } else {
        general_statistic(&ws.cells, &ws.row_total, r, nf, opts.statistic)
    };
    let min_col = ws
        .cells
        .chunks_exact(r)
        .map(|g| g.iter().sum::<u64>())
        .min()
        .expect("at least two groups");
    let dof = ((r as u64 - 1) * (groups as u64 - 1)).min(u64::from(u32::MAX)) as u32;
    TestResult {
        statistic: raw,
        dof,
        neg_log10_p: neg_log10_chi2_sf(raw, dof),
        n,
        rows: r as u32,
        columns: groups as u32,
        merged: nonzero - groups as u32,
        min_expected: min_row as f64 * min_col as f64 / nf,
    }
}

/// Statistic for a table with exactly two rows. `cells` is `[group][2]`.
///
/// Pearson uses the closed form `sum_j (R1 a_j - R0 b_j)^2 / (C_j R0 R1)`. The numerator is an
/// exact integer, so there is no cancellation for large counts.
fn two_row_statistic(cells: &[u64], row_total: &[u64], statistic: Statistic) -> f64 {
    let (r0, r1) = (row_total[0], row_total[1]);
    let n = (r0 + r1) as f64;
    match statistic {
        Statistic::Pearson => {
            let (r0i, r1i) = (i128::from(r0), i128::from(r1));
            let mut sum = 0.0;
            for pair in cells.as_chunks::<2>().0 {
                let (a, b) = (pair[0], pair[1]);
                let numerator = (r1i * i128::from(a) - r0i * i128::from(b)) as f64;
                sum += numerator * numerator / (a + b) as f64;
            }
            sum / (r0 as f64 * r1 as f64)
        }
        Statistic::G => {
            let (r0f, r1f) = (r0 as f64, r1 as f64);
            let mut sum = 0.0;
            for pair in cells.as_chunks::<2>().0 {
                let (a, b) = (pair[0] as f64, pair[1] as f64);
                let c = a + b;
                sum += bd0(a, r0f * c / n) + bd0(b, r1f * c / n);
            }
            2.0 * sum
        }
    }
}

/// Statistic for a table with any number of rows. `cells` is `[group][row]`.
fn general_statistic(
    cells: &[u64],
    row_total: &[u64],
    r: usize,
    n: f64,
    statistic: Statistic,
) -> f64 {
    let mut sum = 0.0;
    for group in cells.chunks_exact(r) {
        let c = group.iter().sum::<u64>() as f64;
        for (&f, &rt) in group.iter().zip(row_total) {
            let e = rt as f64 * c / n;
            let f = f as f64;
            sum += match statistic {
                Statistic::Pearson => (f - e) * (f - e) / e,
                // The sum of (F - E) over the table is zero, so bd0 = F ln(F/E) + E - F gives
                // the same total as F ln(F/E), without cancellation.
                Statistic::G => 2.0 * bd0(f, e),
            };
        }
    }
    sum
}

/// Summary of per-sample results for a report, against one threshold.
///
/// [`Summary2`] counts two thresholds in one pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// Largest `-log10(p)` over the valid samples (0 if there is none).
    pub max_neg_log10_p: f64,
    /// Index of the sample with the largest `-log10(p)`.
    pub argmax: usize,
    /// Number of samples with a valid test.
    pub valid: usize,
    /// Number of samples with `-log10(p)` at or above the threshold.
    pub above: usize,
}

/// Summarizes per-sample results against a `-log10(p)` threshold.
pub fn summarize(results: &[TestResult], threshold: f64) -> Summary {
    let s = summarize2(results, threshold, threshold);
    Summary {
        max_neg_log10_p: s.max_neg_log10_p,
        argmax: s.argmax,
        valid: s.valid,
        above: s.above[0],
    }
}

/// Summary of per-sample results against two thresholds.
///
/// The report of a leakage run needs two counts: the samples above the conventional threshold
/// ([`threshold::CONVENTIONAL`](super::threshold::CONVENTIONAL)) and the samples above the
/// Bonferroni threshold of the whole family of tests. [`summarize2`] gives both in one pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary2 {
    /// Largest `-log10(p)` over the valid samples (0 if there is none).
    pub max_neg_log10_p: f64,
    /// Index of the first sample with the largest `-log10(p)`.
    pub argmax: usize,
    /// Number of samples with a valid test.
    pub valid: usize,
    /// For each threshold, the number of valid samples with `-log10(p)` at or above it.
    pub above: [usize; 2],
}

/// Summarizes per-sample results against two `-log10(p)` thresholds, `t1` and `t2`.
///
/// Samples with an invalid test (fewer than two rows or two columns) are not counted, even if
/// a threshold is zero or negative. A NaN threshold counts nothing.
pub fn summarize2(results: &[TestResult], t1: f64, t2: f64) -> Summary2 {
    let mut s = Summary2 {
        max_neg_log10_p: 0.0,
        argmax: 0,
        valid: 0,
        above: [0, 0],
    };
    for (i, r) in results.iter().enumerate() {
        if !r.is_valid() {
            continue;
        }
        s.valid += 1;
        s.above[0] += usize::from(r.neg_log10_p >= t1);
        s.above[1] += usize::from(r.neg_log10_p >= t2);
        if r.neg_log10_p > s.max_neg_log10_p {
            s.max_neg_log10_p = r.neg_log10_p;
            s.argmax = i;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The worked example of Moradi et al. (TCHES 2018, Section 2.3).
    #[test]
    fn paper_example() {
        let a = [24, 59, 28, 9];
        let b = [23, 57, 20, 0];
        let opts = TestOptions {
            statistic: Statistic::Pearson,
            min_expected: 0.0,
        };
        let r = test_table(&[&a, &b], &opts, &mut Workspace::default()).unwrap();
        assert_eq!(r.dof, 3);
        assert_relative_eq!(r.statistic, 8.64, epsilon = 0.01);
        // p is about 0.0345.
        assert_relative_eq!(10.0_f64.powf(-r.neg_log10_p), 0.0345, epsilon = 0.0005);
    }

    #[test]
    fn empty_rows_and_columns_are_dropped() {
        let a = [10, 0, 20, 0, 30];
        let b = [20, 0, 20, 0, 20];
        let empty = [0; 5];
        let opts = TestOptions {
            statistic: Statistic::Pearson,
            min_expected: 0.0,
        };
        let mut ws = Workspace::default();
        let with = test_table(&[&a, &empty, &b], &opts, &mut ws).unwrap();
        let without = test_table(&[&[10, 20, 30], &[20, 20, 20]], &opts, &mut ws).unwrap();
        assert_eq!(with.dof, 2);
        assert_eq!(with.statistic.to_bits(), without.statistic.to_bits());
    }

    #[test]
    fn merging_keeps_a_valid_table_and_min_expected() {
        // The tail bins have tiny counts and are merged into their neighbor.
        let a = [100, 400, 400, 90, 8, 2];
        let b = [90, 410, 380, 100, 15, 5];
        let opts = TestOptions::default();
        let r = test_table(&[&a, &b], &opts, &mut Workspace::default()).unwrap();
        assert!(r.is_valid());
        assert!(r.min_expected >= 20.0, "{r:?}");
        assert!(r.merged > 0);
    }

    #[test]
    fn single_column_is_not_valid() {
        let r = test_table(
            &[&[0, 5, 0], &[0, 7, 0]],
            &TestOptions::default(),
            &mut Workspace::default(),
        )
        .unwrap();
        assert!(!r.is_valid());
        assert_eq!(r.neg_log10_p, 0.0);
    }

    #[test]
    fn rows_of_unequal_length_are_an_error() {
        let opts = TestOptions::default();
        let mut ws = Workspace::default();
        // A longer second row: the extra column must not reach the row totals.
        let err = test_table(&[&[1, 1], &[1, 1, 100]], &opts, &mut ws).unwrap_err();
        assert_eq!(
            err,
            StatsError::RowLengthMismatch {
                row: 1,
                expected: 2,
                got: 3
            }
        );
        // A shorter second row.
        let err = test_table(&[&[1, 1, 100], &[1, 1]], &opts, &mut ws).unwrap_err();
        assert!(matches!(err, StatsError::RowLengthMismatch { row: 1, .. }));
        // The workspace still works after an error.
        assert!(test_table(&[&[10, 20], &[20, 10]], &opts, &mut ws).is_ok());
        // No rows at all is an empty, invalid table, not an error.
        let r = test_table(&[], &opts, &mut ws).unwrap();
        assert!(!r.is_valid());
    }

    /// Merging by hand. Row totals 100 and 300 (N = 400), `min_expected` 20: the threshold on
    /// a group's column total is 20 * 400 / 100 = 80. Column totals are [3, 4, 8, 135, 250].
    /// Columns 0..=3 reach 150 and close group 0; column 4 (250) is group 1.
    /// Merged table [[50, 50], [100, 200]], expected [[37.5, 62.5], [112.5, 187.5]].
    /// Pearson = 156.25 * (1/37.5 + 1/62.5 + 1/112.5 + 1/187.5) = 8.8889 (= 80 / 9).
    #[test]
    fn merging_with_unequal_row_totals_by_hand() {
        let a = [2, 3, 5, 40, 50];
        let b = [1, 1, 3, 95, 200];
        let r = test_table(
            &[&a, &b],
            &TestOptions::default(),
            &mut Workspace::default(),
        )
        .unwrap();
        assert_eq!((r.rows, r.columns, r.merged, r.dof, r.n), (2, 2, 3, 1, 400));
        assert_relative_eq!(r.statistic, 80.0 / 9.0, max_relative = 1e-14);
        assert_relative_eq!(r.min_expected, 37.5, max_relative = 1e-14);
    }

    /// The remainder at the end joins the last closed group. Row totals 100 and 200 (N = 300),
    /// threshold 20 * 300 / 100 = 60, column totals [160, 120, 10, 10]. Column 0 closes group 0,
    /// column 1 closes group 1, columns 2 and 3 (total 20) join group 1.
    /// Merged table [[60, 40], [100, 100]], column totals [160, 140].
    /// Pearson = 2000^2 / (160 * 100 * 200) + 2000^2 / (140 * 100 * 200) = 1.25 + 1.428571...
    #[test]
    fn a_small_remainder_joins_the_last_group_by_hand() {
        let a = [60, 30, 6, 4];
        let b = [100, 90, 4, 6];
        let r = test_table(
            &[&a, &b],
            &TestOptions::default(),
            &mut Workspace::default(),
        )
        .unwrap();
        assert_eq!((r.rows, r.columns, r.merged, r.dof, r.n), (2, 2, 2, 1, 300));
        assert_relative_eq!(r.statistic, 1.25 + 10.0 / 7.0, max_relative = 1e-14);
        assert_relative_eq!(r.min_expected, 100.0 * 140.0 / 300.0, max_relative = 1e-14);
    }

    fn result(neg_log10_p: f64, dof: u32) -> TestResult {
        TestResult {
            statistic: 0.0,
            dof,
            neg_log10_p,
            n: 100,
            rows: 2,
            columns: 2,
            merged: 0,
            min_expected: 50.0,
        }
    }

    #[test]
    fn summarize2_counts_two_thresholds() {
        let r = [
            result(1.0, 1),
            result(5.0, 3),
            result(7.5, 3),
            result(0.0, 0),
            result(6.0, 2),
            result(7.5, 3),
        ];
        let s = summarize2(&r, 5.0, 7.0);
        assert_eq!(s.max_neg_log10_p, 7.5);
        assert_eq!(s.argmax, 2, "the first maximum wins");
        assert_eq!(s.valid, 5);
        assert_eq!(s.above, [4, 2]);
        // `summarize` is the one-threshold view of the same count.
        let one = summarize(&r, 5.0);
        assert_eq!((one.valid, one.above, one.argmax), (5, 4, 2));
        assert_eq!(one.max_neg_log10_p, 7.5);
    }

    #[test]
    fn summarize2_ignores_invalid_tests_and_empty_input() {
        let s = summarize2(&[], 5.0, 6.0);
        assert_eq!((s.valid, s.above, s.argmax), (0, [0, 0], 0));
        assert_eq!(s.max_neg_log10_p, 0.0);
        // An invalid test is never counted, even with a threshold of 0.
        let s = summarize2(&[result(0.0, 0), result(0.0, 1)], 0.0, f64::NAN);
        assert_eq!((s.valid, s.above), (1, [1, 0]));
    }
}
