/// Errors returned by the statistics APIs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StatsError {
    #[error("t-test order must be greater than zero")]
    ZeroOrder,
    #[error("trace shape mismatch: expected {expected} samples, found {found}")]
    ShapeMismatch { expected: usize, found: usize },
    #[error("label count mismatch: expected {expected}, found {found}")]
    LabelCountMismatch { expected: usize, found: usize },
    #[error("batch has more than u32::MAX traces")]
    BatchTooLarge,
    #[error("accumulators have different sample counts or orders")]
    IncompatibleAccumulators,
    #[error("class trace count overflow")]
    CountOverflow,
    #[error("class {label} has {found} moment values, expected {expected}")]
    WrongMomentLength {
        label: u16,
        expected: usize,
        found: usize,
    },
    #[error("class {label} has nonzero moment data but a zero trace count")]
    EmptyClassHasData { label: u16 },
    #[error("class {label} contains a non-finite moment value")]
    NonFiniteMoments { label: u16 },
    #[error("class {label} contains moments that cannot come from its trace count")]
    InvalidMoments { label: u16 },
    #[error("class label {0} appears more than once")]
    DuplicateLabel(u16),
    #[error("moment order {order} is outside 2..={max}")]
    InvalidMomentOrder { order: usize, max: usize },
}
