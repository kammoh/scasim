//! Shuffled labels for null runs. A run with shuffled labels shows how large |t| gets when the
//! classes carry no information. It calibrates the false-positive floor of a test.

use ndarray::Array1;

/// A SplitMix64 generator. It is small, fast, and its output is the same on every platform.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// A number in `0..n` without bias (Lemire's method). `n` must be greater than 0.
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        let product = |x: u64| u128::from(x) * u128::from(n);
        let mut m = product(self.next());
        if (m as u64) < n {
            // Reject the few values that would make some results more likely than others.
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = product(self.next());
            }
        }
        (m >> 64) as u64
    }
}

/// The finalizer of SplitMix64.
pub(crate) fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Shuffles the labels of batch number `batch` in place (a Fisher-Yates shuffle). The result
/// depends only on `seed`, `batch`, and the labels, so a run is reproducible. Different batches
/// get different permutations. The number of labels in each class does not change.
pub fn shuffle_labels(labels: &mut Array1<u16>, seed: u64, batch: usize) {
    // Mix the batch number into the seed, so that batches with the same seed do not share one
    // stream.
    let mut rng = SplitMix64(mix(seed) ^ mix(batch as u64 ^ 0xA5A5_A5A5_5A5A_5A5A));
    for i in (1..labels.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        labels.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(n: usize) -> Array1<u16> {
        Array1::from_iter((0..n).map(|i| (i % 3 == 0) as u16))
    }

    fn counts(labels: &Array1<u16>) -> [usize; 2] {
        [0, 1].map(|c| labels.iter().filter(|&&l| l == c).count())
    }

    #[test]
    fn the_class_counts_do_not_change() {
        for n in [0, 1, 2, 7, 100, 1999] {
            let original = labels(n);
            let mut shuffled = original.clone();
            shuffle_labels(&mut shuffled, 42, 3);
            assert_eq!(counts(&shuffled), counts(&original), "{n} labels");
        }
    }

    #[test]
    fn the_same_seed_and_batch_give_the_same_permutation() {
        let (mut a, mut b) = (labels(200), labels(200));
        shuffle_labels(&mut a, 7, 2);
        shuffle_labels(&mut b, 7, 2);
        assert_eq!(a, b);
        assert_ne!(a, labels(200), "the labels must move");
    }

    #[test]
    fn another_seed_or_another_batch_gives_another_permutation() {
        let mut base = labels(200);
        shuffle_labels(&mut base, 7, 2);
        for (seed, batch) in [(8, 2), (7, 3), (7, 0)] {
            let mut other = labels(200);
            shuffle_labels(&mut other, seed, batch);
            assert_ne!(other, base, "seed {seed}, batch {batch}");
        }
    }

    #[test]
    fn every_position_gets_each_class_with_its_share() {
        // Class 1 is a third of the labels. Over many seeds, position 0 is class 1 about a third
        // of the time.
        let trials = 3000;
        let hits = (0..trials)
            .filter(|&seed| {
                let mut l = labels(30);
                shuffle_labels(&mut l, seed, 0);
                l[0] == 1
            })
            .count();
        let share = hits as f64 / trials as f64;
        assert!((share - 1.0 / 3.0).abs() < 0.04, "{share}");
    }

    #[test]
    fn the_generator_matches_the_published_splitmix64_values() {
        // The first outputs of SplitMix64 with the seed 0 (reference implementation).
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = SplitMix64(1);
        for n in [1u64, 2, 3, 10, 1000, u64::MAX] {
            for _ in 0..100 {
                assert!(rng.below(n) < n);
            }
        }
    }
}
