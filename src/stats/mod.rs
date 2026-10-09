//! Statistics engines for leakage evaluation.
//!
//! * [`special`]: log-gamma, the deviance term `bd0`, the chi-squared survival function in the
//!   log domain, and the inverse normal distribution.
//! * [`chi2`]: the Pearson chi-squared test and the G test on a classes-by-bins table.
//! * [`threshold`]: multiple-testing thresholds, on `-log10(p)` and on `|t|`.
//! * [`error`]: the error type [`StatsError`].
//!
//! The statistical methods follow Moradi, Richter, Schneider, and Standaert, "Leakage Detection
//! with the chi^2-Test" (TCHES 2018), and the PROLEAD tool for the G test.

pub mod chi2;
pub mod error;
pub mod special;
pub mod threshold;

pub use chi2::{
    Statistic, Summary, Summary2, TestOptions, TestResult, Workspace, summarize, summarize2,
    test_table,
};
pub use error::StatsError;
