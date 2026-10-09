//! The results of the per-scope channels: the ranking, `channels.tsv`, and the `.npz` files.

use crate::summary::summarize_order;
use miette::{IntoDiagnostic, WrapErr};
use ndarray::{Array1, Array2};
use ndarray_npz::NpzWriter;
use scasim::stats::TestResult;
use std::fs::File;
use std::path::Path;

/// The final results of one channel.
pub struct ChannelResult<'a> {
    pub name: &'a str,
    /// The number of signals in the channel.
    pub handles: usize,
    /// Shape `(orders, samples)`.
    pub t_values: &'a Array2<f64>,
    /// The chi-squared results of all samples, if the test ran.
    pub chi2: Option<&'a [TestResult]>,
}

/// One row of the ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankRow {
    /// The index of the channel in the list of the results (it names the arrays `t_<index>`).
    pub index: usize,
    /// 1 for the channel with the largest max |t|.
    pub rank: usize,
    pub name: String,
    pub handles: usize,
    /// For each order: the largest finite |t| (NaN if none) and its sample.
    pub orders: Vec<(f64, usize)>,
    /// The number of samples, over all orders, where |t| is infinite. This is a deterministic
    /// difference between the classes (a class has no variance), the strongest leak there is.
    pub infinite: usize,
    /// The largest -log10(p) and its sample, if the chi-squared test ran.
    pub chi2: Option<(f64, usize)>,
}

impl RankRow {
    /// The order (counted from 1), the |t|, and the sample of the largest finite |t|, or `None`
    /// if there is none. The first order wins a tie.
    pub fn best(&self) -> Option<(usize, f64, usize)> {
        let best = self.max_abs_t();
        self.orders
            .iter()
            .position(|(t, _)| *t == best)
            .map(|i| (i + 1, best, self.orders[i].1))
    }

    /// The largest finite |t| over all orders, or NaN if there is none.
    pub fn max_abs_t(&self) -> f64 {
        self.orders
            .iter()
            .map(|(t, _)| *t)
            .filter(|t| !t.is_nan())
            .fold(f64::NAN, f64::max)
    }
}

/// Ranks the channels by their largest |t| over all orders, largest first. A channel with an
/// infinite |t| comes before all others, and the channels with more infinite values come first
/// among them. A channel without any finite or infinite |t| comes last. Equal values keep the
/// order of the results.
pub fn rank_channels(results: &[ChannelResult<'_>]) -> Vec<RankRow> {
    let mut rows: Vec<RankRow> = results
        .iter()
        .enumerate()
        .map(|(index, r)| RankRow {
            index,
            rank: 0,
            name: r.name.to_string(),
            handles: r.handles,
            infinite: r.t_values.iter().filter(|t| t.is_infinite()).count(),
            orders: r
                .t_values
                .rows()
                .into_iter()
                .map(|row| {
                    let s = summarize_order(row.iter().copied(), f64::INFINITY, f64::INFINITY);
                    (s.max_abs, s.argmax)
                })
                .collect(),
            chi2: r.chi2.map(|results| {
                let s = scasim::stats::summarize(results, f64::INFINITY);
                (s.max_neg_log10_p, s.argmax)
            }),
        })
        .collect();
    let key = |row: &RankRow| {
        let t = row.max_abs_t();
        (row.infinite, if t.is_nan() { f64::NEG_INFINITY } else { t })
    };
    rows.sort_by(|a, b| {
        let (ka, kb) = (key(a), key(b));
        kb.0.cmp(&ka.0)
            .then(kb.1.total_cmp(&ka.1))
            .then(a.index.cmp(&b.index))
    });
    for (i, row) in rows.iter_mut().enumerate() {
        row.rank = i + 1;
    }
    rows
}

/// The ranking as tab-separated text with a header line. A number that does not exist is `NaN`, with the sample 0.
pub fn render_tsv(rows: &[RankRow]) -> String {
    let orders = rows.first().map_or(0, |r| r.orders.len());
    let with_chi2 = rows.first().is_some_and(|r| r.chi2.is_some());
    let mut header = vec![
        "rank".to_string(),
        "channel".into(),
        "handles".into(),
        "infinite_t".into(),
    ];
    for d in 1..=orders {
        header.push(format!("max_abs_t_d{d}"));
        header.push(format!("sample_d{d}"));
    }
    if with_chi2 {
        header.push("max_neg_log10_p".into());
        header.push("sample_chi2".into());
    }
    let mut text = header.join("\t") + "\n";
    for r in rows {
        let mut cells = vec![
            r.rank.to_string(),
            r.name.clone(),
            r.handles.to_string(),
            r.infinite.to_string(),
        ];
        for (t, sample) in &r.orders {
            cells.push(format!("{t:.4}"));
            cells.push(sample.to_string());
        }
        if let Some((p, sample)) = r.chi2 {
            cells.push(format!("{p:.4}"));
            cells.push(sample.to_string());
        }
        text += &(cells.join("\t") + "\n");
    }
    text
}

fn npz_writer(path: &Path) -> miette::Result<NpzWriter<File>> {
    Ok(NpzWriter::new_compressed(
        File::create(path)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot create {}", path.display()))?,
    ))
}

/// Writes the files of the channels into `dir`:
/// - `channels.tsv`: the ranking;
/// - `channels.txt`: the names of the channels, one per line, in the order of `results`;
/// - `t_values_channels.npz`: the array `t_<index>` of shape `(orders, samples)` for each channel;
/// - `chi2_channels.npz`: if the chi-squared test ran, the array `chi2_<index>` of -log10(p), one
///   value for each sample.
pub fn write_channel_files(dir: &Path, results: &[ChannelResult<'_>]) -> miette::Result<()> {
    let rows = rank_channels(results);
    let write = |name: &str, text: String| {
        let path = dir.join(name);
        std::fs::write(&path, text)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot write {}", path.display()))
    };
    write("channels.tsv", render_tsv(&rows))?;
    let names: String = results.iter().map(|r| format!("{}\n", r.name)).collect();
    write("channels.txt", names)?;

    let path = dir.join("t_values_channels.npz");
    let mut npz = npz_writer(&path)?;
    for (i, r) in results.iter().enumerate() {
        npz.add_array(format!("t_{i}"), r.t_values)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot write {}", path.display()))?;
    }
    npz.finish()
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot write {}", path.display()))?;

    if results.iter().all(|r| r.chi2.is_some()) && !results.is_empty() {
        let path = dir.join("chi2_channels.npz");
        let mut npz = npz_writer(&path)?;
        for (i, r) in results.iter().enumerate() {
            let column = Array1::from_iter(r.chi2.unwrap().iter().map(|c| c.neg_log10_p));
            npz.add_array(format!("chi2_{i}"), &column)
                .into_diagnostic()
                .wrap_err_with(|| format!("cannot write {}", path.display()))?;
        }
        npz.finish()
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn results<'a>(names: &'a [&'a str], t: &'a [Array2<f64>]) -> Vec<ChannelResult<'a>> {
        names
            .iter()
            .zip(t)
            .map(|(name, t)| ChannelResult {
                name,
                handles: 2,
                t_values: t,
                chi2: None,
            })
            .collect()
    }

    #[test]
    fn channels_are_ranked_by_the_largest_abs_t_over_all_orders() {
        let t = [
            array![[1.0, -2.0, 0.5], [0.1, 0.2, 3.0]],
            array![[0.1, 0.2, 0.3], [-9.0, 0.0, 0.0]],
            array![
                [f64::NAN, f64::NAN, f64::NAN],
                [f64::NAN, f64::INFINITY, f64::NAN]
            ],
            array![[-2.0, 1.0, 0.0], [0.0, 3.0, 0.0]],
        ];
        let rows = rank_channels(&results(&["a", "b", "c", "d"], &t));
        let order: Vec<(&str, usize)> = rows.iter().map(|r| (r.name.as_str(), r.rank)).collect();
        // c has an infinite |t| (no variance in a class): the strongest leak, so it is first.
        // b: 9. a and d tie at 3.0 (order 2): the earlier channel first.
        assert_eq!(order, [("c", 1), ("b", 2), ("a", 3), ("d", 4)]);
        assert_eq!(rows[0].infinite, 1);
        assert_eq!(rows[1].orders, [(0.3, 2), (9.0, 0)]);
        assert_eq!(rows[1].index, 1);
        assert!(rows[0].max_abs_t().is_nan());
    }

    #[test]
    fn the_tsv_has_a_header_and_one_row_for_each_channel() {
        let t = [
            array![[1.0, -2.0], [0.0, 5.0]],
            array![[0.0, 0.5], [0.0, 0.0]],
        ];
        let mut r = results(&["tb.a", "tb.b"], &t);
        let tsv = render_tsv(&rank_channels(&r));
        assert_eq!(
            tsv,
            "rank\tchannel\thandles\tinfinite_t\tmax_abs_t_d1\tsample_d1\tmax_abs_t_d2\tsample_d2\n\
             1\ttb.a\t2\t0\t2.0000\t1\t5.0000\t1\n\
             2\ttb.b\t2\t0\t0.5000\t1\t0.0000\t0\n"
        );
        let chi2 = [
            TestResult {
                statistic: 1.0,
                dof: 1,
                neg_log10_p: 7.5,
                n: 10,
                rows: 2,
                columns: 2,
                merged: 0,
                min_expected: 5.0,
            },
            TestResult {
                statistic: 1.0,
                dof: 1,
                neg_log10_p: 1.5,
                n: 10,
                rows: 2,
                columns: 2,
                merged: 0,
                min_expected: 5.0,
            },
        ];
        r[0].chi2 = Some(&chi2[..1]);
        r[1].chi2 = Some(&chi2[1..]);
        let tsv = render_tsv(&rank_channels(&r));
        assert!(tsv.starts_with("rank\tchannel\thandles\tinfinite_t\tmax_abs_t_d1\tsample_d1\tmax_abs_t_d2\tsample_d2\tmax_neg_log10_p\tsample_chi2\n"));
        assert!(
            tsv.contains("1\ttb.a\t2\t0\t2.0000\t1\t5.0000\t1\t7.5000\t0\n"),
            "{tsv}"
        );
    }

    #[test]
    fn the_files_have_the_arrays_and_the_names() {
        let t = [array![[1.0, -2.0]], array![[3.0, 0.0]]];
        let dir = tempfile::tempdir().unwrap();
        write_channel_files(dir.path(), &results(&["x", "y"], &t)).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("channels.txt")).unwrap(),
            "x\ny\n"
        );
        let mut npz = ndarray_npz::NpzReader::new(
            File::open(dir.path().join("t_values_channels.npz")).unwrap(),
        )
        .unwrap();
        let t1: Array2<f64> = npz.by_name("t_1").unwrap();
        assert_eq!(t1, t[1]);
        assert!(!dir.path().join("chi2_channels.npz").exists());
        let tsv = std::fs::read_to_string(dir.path().join("channels.tsv")).unwrap();
        assert!(tsv.contains("1\ty\t2\t0\t3.0000\t0\n"), "{tsv}");
    }
}
