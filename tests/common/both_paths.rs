//! Helpers that run the fast path and the reference path.

use scasim::power::fst::{activity_fst, activity_fst_binned};
use scasim::power::reference::{activity_reference, activity_reference_binned};
use scasim::power::{ActivityTrace, Bins, PowerPlan};
use std::path::Path;

/// Asserts that the fast path and the reference path both give `expected`.
pub fn assert_both_paths(path: &Path, plan: &PowerPlan, expected: &ActivityTrace) {
    assert_eq!(&activity_fst(path, plan).unwrap(), expected, "fast path");
    assert_eq!(
        &activity_reference(path, plan).unwrap(),
        expected,
        "reference path"
    );
}

/// Like [`assert_both_paths`], with the given bins.
pub fn assert_both_paths_binned(
    path: &Path,
    plan: &PowerPlan,
    bins: &Bins,
    expected: &ActivityTrace,
) {
    assert_eq!(
        &activity_fst_binned(path, plan, bins).unwrap(),
        expected,
        "fast path"
    );
    assert_eq!(
        &activity_reference_binned(path, plan, bins).unwrap(),
        expected,
        "reference path"
    );
}
