//! The plain, serializable form of an accumulator state.

use crate::stats::error::StatsError;
use serde::{Deserialize, Serialize};

/// The state of one class: its trace count and its moments for all sample points.
///
/// The field `data` is a row-major matrix with `2 * d + 1` rows and `ns` columns.
/// Row 0 and row 1 hold the mean as the sum of two numbers: `mean = data[0][j] + data[1][j]`.
/// Row `p` for `p = 2, ..., 2d` holds the central sum `CS_p = sum_i (x_i - mean)^p`.
/// Use [`ClassMoments::mean`] and [`ClassMoments::central_sum`] to read them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassMoments {
    /// The class label.
    pub label: u16,
    /// The number of traces in the class.
    pub count: u64,
    /// The moments, row-major with shape `[2 * d + 1, ns]`.
    pub data: Vec<f64>,
}

impl ClassMoments {
    /// The mean at sample point `j`, for a state with `ns` sample points.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ns` or if `data` is too short.
    pub fn mean(&self, ns: usize, j: usize) -> f64 {
        assert!(j < ns);
        self.data[j] + self.data[ns + j]
    }

    /// The central sum of order `p` (`2 <= p <= 2d`) at sample point `j`.
    ///
    /// # Panics
    ///
    /// Panics if `p < 2`, `j >= ns`, or `data` is too short.
    pub fn central_sum(&self, ns: usize, p: usize, j: usize) -> f64 {
        assert!(p >= 2 && j < ns);
        self.data[p * ns + j]
    }
}

/// The complete state of a [`MomentAccumulator`](super::MomentAccumulator).
///
/// A value of this type is plain data. A program can save it to disk (with the `serde`
/// feature) and restore it later with
/// [`MomentAccumulator::from_moments`](super::MomentAccumulator::from_moments).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    /// The number of sample points per trace.
    pub ns: usize,
    /// The maximum t-test order `d`. The state holds central sums up to order `2d`.
    pub d: usize,
    /// The state of each class, in increasing order of label.
    pub classes: Vec<ClassMoments>,
}

impl Moments {
    pub(crate) fn validate_shape(&self) -> Result<(), StatsError> {
        if self.d == 0 {
            return Err(StatsError::ZeroOrder);
        }
        let rows = self
            .d
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or(StatsError::CountOverflow)?;
        let expected = rows.checked_mul(self.ns).ok_or(StatsError::CountOverflow)?;
        let mut seen = std::collections::BTreeSet::new();
        for class in &self.classes {
            if class.data.len() != expected {
                return Err(StatsError::WrongMomentLength {
                    label: class.label,
                    expected,
                    found: class.data.len(),
                });
            }
            if !seen.insert(class.label) {
                return Err(StatsError::DuplicateLabel(class.label));
            }
            if class.data.iter().any(|v| !v.is_finite()) {
                return Err(StatsError::NonFiniteMoments { label: class.label });
            }
            if class.count == 0 && class.data.iter().any(|&v| v != 0.0) {
                return Err(StatsError::EmptyClassHasData { label: class.label });
            }
        }
        Ok(())
    }
}
