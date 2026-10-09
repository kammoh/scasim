//! Error type of the statistics engines.

/// Errors from building, updating, merging, and testing statistics accumulators.
///
/// The enum is `#[non_exhaustive]`: later changes add variants. A `match` on it needs a
/// wildcard arm.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatsError {
    /// The trace matrix has a different number of samples than the accumulator.
    #[error("traces have {got} samples per trace but the accumulator has {expected}")]
    SampleCountMismatch {
        /// Number of samples in the accumulator.
        expected: usize,
        /// Number of samples in the batch.
        got: usize,
    },
    /// The number of labels differs from the number of traces.
    #[error("{traces} traces but {labels} labels")]
    LabelCountMismatch {
        /// Number of traces in the batch.
        traces: usize,
        /// Number of labels in the batch.
        labels: usize,
    },
    /// The binning parameters are not usable.
    #[error("invalid binning: {0}")]
    InvalidBinning(String),
    /// A class would hold more than `u32::MAX` traces, which the `u32` bin counters cannot hold.
    #[error("class {label} would exceed {} traces", u32::MAX)]
    CountOverflow {
        /// The class label.
        label: u16,
    },
    /// A test asked for a class that has no data in the accumulator.
    #[error("class label {0} is not in the accumulator")]
    UnknownLabel(u16),
    /// A test lists the same class label twice.
    #[error("class label {0} is listed twice")]
    DuplicateLabel(u16),
    /// Two accumulators cannot be merged.
    #[error("incompatible accumulators: {0}")]
    Incompatible(String),
    /// A deserialized accumulator is inconsistent.
    #[error("invalid accumulator state: {0}")]
    InvalidState(String),
}
