//! Statistics engines for leakage evaluation.
//!
//! * [`special`]: log-gamma, the deviance term `bd0`, the chi-squared survival function in the
//!   log domain, and the inverse normal distribution.
//! * [`binning`] and [`hist`]: the rules that map sample values to bins, and the streaming
//!   per-sample, per-class histograms ([`HistAccumulator`]) that feed the chi-squared test.
//! * [`chi2`]: the Pearson chi-squared test and the G test on a classes-by-bins table.
//! * [`threshold`]: multiple-testing thresholds, on `-log10(p)` and on `|t|`.
//! * [`error`]: the error type [`StatsError`].
//!
//! The statistical methods follow Moradi, Richter, Schneider, and Standaert, "Leakage Detection
//! with the chi^2-Test" (TCHES 2018), and the PROLEAD tool for the G test.

pub mod binning;
pub mod chi2;
pub mod error;
pub mod hist;
pub mod special;
pub mod threshold;

pub use binning::{BinValue, Binning};
pub use chi2::{
    Statistic, Summary, Summary2, TestOptions, TestResult, Workspace, summarize, summarize2,
    test_table,
};
pub use error::StatsError;
pub use hist::HistAccumulator;
