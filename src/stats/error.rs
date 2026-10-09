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
    /// The rows of a contingency table have different lengths.
    #[error("row {row} has {got} columns but row 0 has {expected}")]
    RowLengthMismatch {
        /// Index of the first row with a wrong length.
        row: usize,
        /// Length of row 0.
        expected: usize,
        /// Length of that row.
        got: usize,
    },
    /// Two accumulators cannot be merged.
    #[error("incompatible accumulators: {0}")]
    Incompatible(String),
    /// A deserialized accumulator is inconsistent.
    #[error("invalid accumulator state: {0}")]
    InvalidState(String),
    /// The t-test order is zero.
    #[error("t-test order must be greater than zero")]
    ZeroOrder,
    /// A batch has more than `u32::MAX` traces.
    #[error("batch has more than u32::MAX traces")]
    BatchTooLarge,
    /// Two moment accumulators have different sample counts or orders.
    #[error("accumulators have different sample counts or orders")]
    IncompatibleAccumulators,
    /// A size or count computed from the inputs does not fit in `usize` or `u64`.
    #[error("a size computed from the inputs overflows")]
    SizeOverflow,
    /// Saved moments of a class have the wrong number of values.
    #[error("class {label} has {found} moment values, expected {expected}")]
    WrongMomentLength {
        /// The class label.
        label: u16,
        /// The expected number of values.
        expected: usize,
        /// The number of values found.
        found: usize,
    },
    /// Saved moments of an empty class are not zero.
    #[error("class {label} has nonzero moment data but a zero trace count")]
    EmptyClassHasData {
        /// The class label.
        label: u16,
    },
    /// Saved moments contain NaN or infinity.
    #[error("class {label} contains a non-finite moment value")]
    NonFiniteMoments {
        /// The class label.
        label: u16,
    },
    /// Saved moments cannot come from the class's trace count.
    #[error("class {label} contains moments that cannot come from its trace count")]
    InvalidMoments {
        /// The class label.
        label: u16,
    },
    /// A moment order is outside the supported range.
    #[error("moment order {order} is outside 2..={max}")]
    InvalidMomentOrder {
        /// The requested order.
        order: usize,
        /// The largest supported order.
        max: usize,
    },
}
