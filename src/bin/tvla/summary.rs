//! The end-of-run summary of `tvla`: counts and maxima of the t-values and of the chi-squared
//! results. The functions here only compute and format. They do not log.

use crate::channels::RankRow;
use ndarray::ArrayView2;
use scasim::batch::EdgeReport;
use scasim::stats::TestResult;
use scasim::stats::chi2::Summary2;

/// Counts and maxima of the t-values of one order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderSummary {
    /// The largest finite |t|, or NaN if no value is finite.
    pub max_abs: f64,
    /// The first sample with that |t|. 0 if no value is finite.
    pub argmax: usize,
    /// Samples with |t| above the conventional threshold. Infinite values are included.
    pub above_conventional: usize,
    /// Samples with |t| above the Bonferroni threshold. Infinite values are included.
    pub above_bonferroni: usize,
    /// Samples where |t| is infinite (a deterministic difference between the classes).
    pub infinite: usize,
    /// Samples where t is undefined (NaN).
    pub undefined: usize,
}

/// Summarizes one row of t-values.
pub fn summarize_order(
    row: impl IntoIterator<Item = f64>,
    conventional: f64,
    bonferroni: f64,
) -> OrderSummary {
    let mut s = OrderSummary {
        max_abs: f64::NAN,
        argmax: 0,
        above_conventional: 0,
        above_bonferroni: 0,
        infinite: 0,
        undefined: 0,
    };
    for (i, t) in row.into_iter().enumerate() {
        let a = t.abs();
        if t.is_nan() {
            s.undefined += 1;
            continue;
        }
        if a.is_infinite() {
            s.infinite += 1;
        } else if s.max_abs.is_nan() || a > s.max_abs {
            s.max_abs = a;
            s.argmax = i;
        }
        s.above_conventional += usize::from(a > conventional);
        s.above_bonferroni += usize::from(a > bonferroni);
    }
    s
}

/// The chi-squared part of the summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chi2Report {
    pub summary: Summary2,
    /// The two thresholds of `summary.above`: the conventional one and the Bonferroni one.
    pub thresholds: [f64; 2],
    /// Number of samples (the size of the Bonferroni family).
    pub samples: usize,
    /// The result at the sample with the largest -log10(p). `None` if no test is valid.
    pub at_max: Option<TestResult>,
    /// The smallest expected count over the valid tests. `None` if no test is valid.
    pub min_expected: Option<f64>,
}

/// Combines a [`Summary2`] with the details that the report needs.
pub fn chi2_report(results: &[TestResult], thresholds: [f64; 2]) -> Chi2Report {
    let summary = scasim::stats::summarize2(results, thresholds[0], thresholds[1]);
    let at_max = (summary.valid > 0).then(|| results[summary.argmax]);
    let min_expected = results
        .iter()
        .filter(|r| r.is_valid())
        .map(|r| r.min_expected)
        .reduce(f64::min);
    Chi2Report {
        summary,
        thresholds,
        samples: results.len(),
        at_max,
        min_expected,
    }
}

/// How the toggles split at the clock edges, summed over all batches (edges mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeTotals {
    pub batches: usize,
    /// The number of bins (clock periods) in all batches.
    pub bins: usize,
    pub inside: u64,
    pub before: u64,
    pub after: u64,
    /// The smallest and the largest distance from a segment start to its first bin, over all
    /// segments of all batches. `None` before the first batch.
    offsets: Option<(u64, u64)>,
}

impl EdgeTotals {
    /// The smallest and the largest segment offset over all batches added so far.
    pub fn offset_range(&self) -> Option<(u64, u64)> {
        self.offsets
    }

    /// True if the segments do not all start at the same place in the clock period.
    pub fn offsets_differ(&self) -> bool {
        self.offsets.is_some_and(|(min, max)| min != max)
    }

    /// Adds the report of one batch.
    pub fn add(&mut self, e: &EdgeReport) {
        self.batches += 1;
        self.bins += e.summary.edges - 1;
        self.inside += e.inside;
        self.before += e.before;
        self.after += e.after;
        self.offsets = Some(match self.offsets {
            None => (e.offset_min, e.offset_max),
            Some((min, max)) => (min.min(e.offset_min), max.max(e.offset_max)),
        });
    }

    /// The fraction of the toggles outside all bins. 0 if there are no toggles.
    pub fn outside_fraction(&self) -> f64 {
        let total = self.inside + self.before + self.after;
        if total == 0 {
            0.0
        } else {
            (self.before + self.after) as f64 / total as f64
        }
    }
}

/// The per-scope channels (`--per-scope`).
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelsSummary {
    pub scope: String,
    pub depth: usize,
    pub count: usize,
    /// Signals with several names, and selected signals outside the scope.
    pub aliased: usize,
    pub outside: usize,
    /// The bytes of the accumulators of the channels.
    pub memory_bytes: usize,
    /// The best channels, one text for each.
    pub top: Vec<String>,
}

/// One line of the best channels: rank, name, and the largest |t|.
pub fn describe_rank(row: &RankRow) -> String {
    if row.infinite > 0 {
        return format!(
            "{}. {} {} infinite |t| (a class has no variance)",
            row.rank, row.name, row.infinite
        );
    }
    match row.best() {
        Some((d, t, sample)) => format!(
            "{}. {} |t| {t:.3} (d={d}, sample {sample})",
            row.rank, row.name
        ),
        None => format!("{}. {} no finite t-value", row.rank, row.name),
    }
}

/// The inputs of the summary text.
pub struct SummaryInput<'a> {
    pub t_values: ArrayView2<'a, f64>,
    pub conventional: f64,
    pub alpha: f64,
    /// The Bonferroni threshold of the t-values and the family size it comes from.
    pub bonferroni: f64,
    pub family: u64,
    pub chi2: Option<Chi2Report>,
    pub memory_bytes: usize,
    /// Present in edges mode.
    pub edges: Option<EdgeTotals>,
    /// Present with `--per-scope`.
    pub channels: Option<ChannelsSummary>,
    /// The seed of `--shuffle-labels`, if the labels were shuffled.
    pub shuffle_seed: Option<u64>,
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Formats the summary as one block of lines.
pub fn render(input: &SummaryInput<'_>) -> String {
    let mut lines = vec!["Summary".to_string()];
    if let Some(seed) = input.shuffle_seed {
        lines.push(format!(
            "  null run: labels were shuffled with the seed {seed}; no leak is expected"
        ));
    }
    let orders: Vec<OrderSummary> = input
        .t_values
        .rows()
        .into_iter()
        .map(|row| summarize_order(row.iter().copied(), input.conventional, input.bonferroni))
        .collect();
    lines.push(format!(
        "  t-test: Bonferroni threshold {:.3} for alpha = {:e} and m = {} (samples x orders)",
        input.bonferroni, input.alpha, input.family
    ));
    for (i, s) in orders.iter().enumerate() {
        let max = if s.max_abs.is_nan() {
            "max |t| none (no finite value)".to_string()
        } else {
            format!("max |t| {:.3} at sample {}", s.max_abs, s.argmax)
        };
        lines.push(format!(
            "  d={}: {max}; {} above {}; {} above {:.3} (Bonferroni)",
            i + 1,
            s.above_conventional,
            input.conventional,
            s.above_bonferroni,
            input.bonferroni
        ));
    }
    let infinite: usize = orders.iter().map(|s| s.infinite).sum();
    let undefined: usize = orders.iter().map(|s| s.undefined).sum();
    if infinite > 0 {
        lines.push(format!(
            "  {infinite} t-values with infinite |t| (deterministic difference between the classes)"
        ));
    }
    if undefined > 0 {
        lines.push(format!("  {undefined} t-values are undefined (NaN)"));
    }
    if let Some(c) = &input.chi2 {
        lines.push(format!(
            "  chi2 (Pearson): Bonferroni threshold {:.3} for alpha = {:e} and m = {} samples",
            c.thresholds[1], input.alpha, c.samples
        ));
        match (&c.at_max, c.min_expected) {
            (Some(r), Some(min_expected)) => lines.push(format!(
                "  chi2: max -log10(p) {:.3} at sample {} (dof {}, {} bins merged); {} above {}; \
                 {} above {:.3} (Bonferroni); min expected count {:.2}; n {}",
                c.summary.max_neg_log10_p,
                c.summary.argmax,
                r.dof,
                r.merged,
                c.summary.above[0],
                c.thresholds[0],
                c.summary.above[1],
                c.thresholds[1],
                min_expected,
                r.n
            )),
            _ => lines.push(
                "  chi2: no valid test (every sample has fewer than two classes or two bins)"
                    .to_string(),
            ),
        }
        if c.summary.failed > 0 {
            lines.push(format!(
                "  chi2: {} p-values failed to converge",
                c.summary.failed
            ));
        }
    }
    if let Some(e) = &input.edges {
        lines.push(format!(
            "  clock edges: {} bins in {} batches; toggles: {} inside the bins + {} before the \
             first edge + {} after the last edge = {} in total; {:.2}% outside",
            e.bins,
            e.batches,
            e.inside,
            e.before,
            e.after,
            e.inside + e.before + e.after,
            100.0 * e.outside_fraction()
        ));
    }
    if let Some(c) = &input.channels {
        lines.push(format!(
            "  channels: {} below {} (depth {}); {} signals with several names; {} selected \
             signals outside the scope; accumulators of the channels: {:.2} MiB",
            c.count,
            c.scope,
            c.depth,
            c.aliased,
            c.outside,
            mib(c.memory_bytes)
        ));
        lines.push(format!("  best channels by max |t|: {}", c.top.join("; ")));
    }
    lines.push(format!(
        "  accumulator memory: {:.2} MiB",
        mib(input.memory_bytes)
    ));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn an_order_counts_maxima_thresholds_and_non_finite_values() {
        let row = [1.0, -6.0, f64::NAN, f64::INFINITY, 4.6, f64::NEG_INFINITY];
        let s = summarize_order(row, 4.5, 5.5);
        assert_eq!(s.max_abs, 6.0);
        assert_eq!(s.argmax, 1);
        // -6, +inf, 4.6, -inf are above 4.5. -6, +inf, -inf are above 5.5.
        assert_eq!(s.above_conventional, 4);
        assert_eq!(s.above_bonferroni, 3);
        assert_eq!(s.infinite, 2);
        assert_eq!(s.undefined, 1);
    }

    #[test]
    fn a_row_without_finite_values_has_no_maximum() {
        let s = summarize_order([f64::NAN, f64::INFINITY], 4.5, 5.5);
        assert!(s.max_abs.is_nan());
        assert_eq!(s.argmax, 0);
        assert_eq!((s.infinite, s.undefined), (1, 1));
    }

    #[test]
    fn the_first_of_equal_maxima_wins() {
        let s = summarize_order([2.0, -3.0, 3.0], 4.5, 5.5);
        assert_eq!((s.max_abs, s.argmax), (3.0, 1));
    }

    #[test]
    fn the_text_has_one_line_per_order_and_prints_non_finite_counts_once() {
        let t = array![[1.0, f64::NAN], [f64::INFINITY, 7.0]];
        let text = render(&SummaryInput {
            t_values: t.view(),
            conventional: 4.5,
            alpha: 1e-5,
            bonferroni: 5.0,
            family: 4,
            chi2: None,
            memory_bytes: 3 << 20,
            edges: None,
            channels: None,
            shuffle_seed: None,
        });
        assert_eq!(text.matches("d=1: max |t| 1.000 at sample 0").count(), 1);
        assert_eq!(text.matches("d=2: max |t| 7.000 at sample 1").count(), 1);
        assert_eq!(text.matches("1 t-values with infinite |t|").count(), 1);
        assert_eq!(text.matches("1 t-values are undefined").count(), 1);
        assert!(text.contains("m = 4"));
        assert!(text.contains("3.00 MiB"));
        assert!(!text.contains("chi2"));
    }

    #[test]
    fn the_chi2_report_uses_the_valid_tests_only() {
        let result = |p: f64, dof: u32, min_expected: f64| TestResult {
            statistic: 1.0,
            dof,
            neg_log10_p: p,
            n: 100,
            rows: 2,
            columns: dof + 1,
            merged: 2,
            min_expected,
        };
        let results = [
            result(0.0, 0, 0.0),
            result(6.0, 3, 25.0),
            result(2.0, 4, 21.0),
            result(f64::NAN, 4, 1.0),
        ];
        let c = chi2_report(&results, [5.0, 5.6]);
        assert_eq!(c.summary.argmax, 1);
        assert_eq!(c.summary.above, [1, 1]);
        assert_eq!(c.summary.failed, 1);
        assert_eq!(c.min_expected, Some(21.0));
        assert_eq!(c.at_max.unwrap().dof, 3);
        assert_eq!(c.samples, 4);
        let t = array![[1.0]];
        let text = render(&SummaryInput {
            t_values: t.view(),
            conventional: 4.5,
            alpha: 1e-5,
            bonferroni: 4.4,
            family: 1,
            chi2: Some(c),
            memory_bytes: 0,
            edges: None,
            channels: None,
            shuffle_seed: None,
        });
        assert!(text.contains("max -log10(p) 6.000 at sample 1 (dof 3, 2 bins merged)"));
        assert!(text.contains("1 p-values failed"));
    }

    #[test]
    fn the_totals_track_the_range_of_the_segment_offsets_over_all_batches() {
        use scasim::power::edges::EdgeSummary;
        let report = |min, max| EdgeReport {
            summary: EdgeSummary {
                edges: 3,
                first_edge: 0,
                min_period: 1,
                max_period: 1,
                mean_period: 1.0,
            },
            inside: 1,
            before: 0,
            after: 0,
            offset_min: min,
            offset_max: max,
        };
        let mut totals = EdgeTotals::default();
        assert_eq!(totals.offset_range(), None);
        totals.add(&report(0, 0));
        assert_eq!(totals.offset_range(), Some((0, 0)));
        assert!(!totals.offsets_differ());
        totals.add(&report(9, 9));
        assert_eq!(totals.offset_range(), Some((0, 9)));
        assert!(totals.offsets_differ());
    }

    #[test]
    fn the_summary_reports_the_split_of_the_toggles_at_the_clock_edges() {
        let t = array![[1.0]];
        let edges = EdgeTotals {
            batches: 2,
            bins: 20,
            inside: 90,
            before: 6,
            after: 4,
            ..EdgeTotals::default()
        };
        assert_eq!(edges.outside_fraction(), 0.1);
        let text = render(&SummaryInput {
            t_values: t.view(),
            conventional: 4.5,
            alpha: 1e-5,
            bonferroni: 4.4,
            family: 1,
            chi2: None,
            memory_bytes: 0,
            edges: Some(edges),
            channels: None,
            shuffle_seed: None,
        });
        assert!(
            text.contains("20 bins in 2 batches; toggles: 90 inside the bins + 6 before"),
            "{text}"
        );
        assert!(text.contains("= 100 in total; 10.00% outside"), "{text}");
    }

    #[test]
    fn the_summary_lists_the_best_channels() {
        let row = |rank, name: &str, orders: Vec<(f64, usize)>| RankRow {
            index: 0,
            rank,
            name: name.into(),
            handles: 1,
            orders,
            infinite: 0,
            chi2: None,
        };
        let top = vec![
            describe_rank(&row(1, "tb.a", vec![(2.0, 1), (12.5, 4)])),
            describe_rank(&row(2, "tb.b", vec![(f64::NAN, 0)])),
            describe_rank(&RankRow {
                infinite: 3,
                ..row(3, "tb.c", vec![(1.0, 0)])
            }),
        ];
        assert_eq!(top[2], "3. tb.c 3 infinite |t| (a class has no variance)");
        assert_eq!(top[0], "1. tb.a |t| 12.500 (d=2, sample 4)");
        assert_eq!(top[1], "2. tb.b no finite t-value");
        let t = array![[1.0]];
        let text = render(&SummaryInput {
            t_values: t.view(),
            conventional: 4.5,
            alpha: 1e-5,
            bonferroni: 4.4,
            family: 1,
            chi2: None,
            memory_bytes: 0,
            edges: None,
            channels: Some(ChannelsSummary {
                scope: "tb.dut".into(),
                depth: 1,
                count: 2,
                aliased: 3,
                outside: 4,
                memory_bytes: 2 << 20,
                top,
            }),
            shuffle_seed: Some(9),
        });
        assert!(
            text.contains("null run: labels were shuffled with the seed 9"),
            "{text}"
        );
        assert!(text.contains("channels: 2 below tb.dut (depth 1); 3 signals with several names; 4 selected signals outside the scope; accumulators of the channels: 2.00 MiB"), "{text}");
        assert!(
            text.contains("best channels by max |t|: 1. tb.a |t| 12.500 (d=2, sample 4); 2. tb.b"),
            "{text}"
        );
    }

    #[test]
    fn a_chi2_report_without_valid_tests_says_so() {
        let c = chi2_report(&[], [5.0, 5.0]);
        assert!(c.at_max.is_none() && c.min_expected.is_none());
    }
}
