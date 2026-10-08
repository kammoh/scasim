//! Statistics of one value change. Two value layouts exist:
//! - state characters: one ASCII character per bit, most significant bit first;
//! - packed 2-state bytes in FST layout: the most significant bit is bit 7 of byte 0, and the
//!   unused low bits of the last byte are zero.
//!
//! State characters are normalized first: every character is lowercased, then it has a level:
//! `0` and `l` are level 0, `1` and `h` are level 1, and every other character is unknown.
//!
//! - A position toggles if the lowercased characters differ. So `h` to `1` is a toggle, but not
//!   a rise.
//! - A position rises if its level goes from 0 to 1 and falls if its level goes from 1 to 0.
//! - The Hamming weight is counted in half-bit units: level 1 is 2, level 0 is 0, and an unknown
//!   character has the weight that the [`UnknownPolicy`] gives.

use super::{Totals, UnknownPolicy};

/// The level of a state character after lowercasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Zero,
    One,
    Unknown,
}

fn level(c: u8) -> Level {
    match c.to_ascii_lowercase() {
        b'0' | b'l' => Level::Zero,
        b'1' | b'h' => Level::One,
        _ => Level::Unknown,
    }
}

impl UnknownPolicy {
    /// Hamming weight of one state character, in half-bit units.
    pub fn half_weight(self, c: u8) -> i64 {
        match (level(c), self) {
            (Level::Zero, _) | (Level::Unknown, UnknownPolicy::AsZero) => 0,
            (Level::One, _) | (Level::Unknown, UnknownPolicy::AsOne) => 2,
            (Level::Unknown, UnknownPolicy::Half) => 1,
        }
    }
}

/// Statistics of a change between two values given as state characters.
pub fn chars_delta(old: &[u8], new: &[u8], unknown: UnknownPolicy) -> Totals {
    assert_eq!(old.len(), new.len(), "values have different widths");
    let mut d = Totals::default();
    for (&a, &b) in old.iter().zip(new) {
        if !a.eq_ignore_ascii_case(&b) {
            d.toggles += 1;
            match (level(a), level(b)) {
                (Level::Zero, Level::One) => d.rise += 1,
                (Level::One, Level::Zero) => d.fall += 1,
                _ => {}
            }
        }
        d.hw_delta += unknown.half_weight(b) - unknown.half_weight(a);
    }
    d
}

/// Statistics of the first value of a signal: only its Hamming weight counts.
pub fn chars_first(new: &[u8], unknown: UnknownPolicy) -> Totals {
    Totals {
        hw_delta: new.iter().map(|&c| unknown.half_weight(c)).sum(),
        ..Totals::default()
    }
}

/// Calls `f` with the bytes of `a` and `b` in 64-bit words. A shorter rest is passed as single
/// bytes.
fn words(a: &[u8], b: &[u8], mut f: impl FnMut(u64, u64)) {
    assert_eq!(a.len(), b.len(), "packed values have different lengths");
    let (a_words, a_rest) = a.as_chunks::<8>();
    let (b_words, b_rest) = b.as_chunks::<8>();
    for (x, y) in a_words.iter().zip(b_words) {
        f(u64::from_ne_bytes(*x), u64::from_ne_bytes(*y));
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        f(u64::from(*x), u64::from(*y));
    }
}

/// Number of differing bits between two packed 2-state values.
pub fn packed_toggles(a: &[u8], b: &[u8]) -> u64 {
    let mut toggles = 0;
    words(a, b, |x, y| toggles += u64::from((x ^ y).count_ones()));
    toggles
}

/// Full statistics of a change between two packed 2-state values.
pub fn packed_delta(a: &[u8], b: &[u8]) -> Totals {
    let mut d = Totals::default();
    words(a, b, |x, y| {
        d.rise += u64::from((!x & y).count_ones());
        d.fall += u64::from((x & !y).count_ones());
        d.hw_delta += 2 * (i64::from(y.count_ones()) - i64::from(x.count_ones()));
    });
    d.toggles = d.rise + d.fall;
    d
}

/// Statistics of the first value of a signal, packed.
pub fn packed_first(b: &[u8]) -> Totals {
    Totals {
        hw_delta: 2 * b.iter().map(|x| i64::from(x.count_ones())).sum::<i64>(),
        ..Totals::default()
    }
}

/// Expands a packed 2-state value to `width` state characters.
pub fn packed_to_chars(width: u32, bytes: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.extend((0..width as usize).map(|i| b'0' + ((bytes[i / 8] >> (7 - i % 8)) & 1)));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuple(d: Totals) -> (u64, u64, u64, i64) {
        (d.toggles, d.rise, d.fall, d.hw_delta)
    }

    #[test]
    fn packed_and_chars_agree() {
        // 12-bit values: 1011_0010_1100 -> 0110_1010_0011
        let a = [0b1011_0010, 0b1100_0000];
        let b = [0b0110_1010, 0b0011_0000];
        let (mut ac, mut bc) = (Vec::new(), Vec::new());
        packed_to_chars(12, &a, &mut ac);
        packed_to_chars(12, &b, &mut bc);
        assert_eq!(ac, b"101100101100");
        assert_eq!(bc, b"011010100011");
        let d = packed_delta(&a, &b);
        assert_eq!(d, chars_delta(&ac, &bc, UnknownPolicy::Half));
        assert_eq!(tuple(d), (8, 4, 4, 0));
        assert_eq!(packed_toggles(&a, &b), 8);
        assert_eq!(packed_first(&b), chars_first(&bc, UnknownPolicy::Half));
    }

    #[test]
    fn packed_works_across_word_boundaries() {
        let a = [0xFFu8; 11];
        let mut b = [0xFFu8; 11];
        b[0] = 0x0F; // 4 falls
        b[8] = 0xFE; // 1 fall in the remainder
        b[10] = 0x00; // 8 falls
        let d = packed_delta(&a, &b);
        assert_eq!(tuple(d), (13, 0, 13, -26));
    }

    #[test]
    fn levels_and_weights_follow_the_normalization_table() {
        use UnknownPolicy::*;
        // (character, level 0 or 1 or unknown, weights for AsZero, AsOne, Half)
        let table: [(u8, Level, [i64; 3]); 16] = [
            (b'0', Level::Zero, [0, 0, 0]),
            (b'l', Level::Zero, [0, 0, 0]),
            (b'L', Level::Zero, [0, 0, 0]),
            (b'1', Level::One, [2, 2, 2]),
            (b'h', Level::One, [2, 2, 2]),
            (b'H', Level::One, [2, 2, 2]),
            (b'x', Level::Unknown, [0, 2, 1]),
            (b'X', Level::Unknown, [0, 2, 1]),
            (b'z', Level::Unknown, [0, 2, 1]),
            (b'Z', Level::Unknown, [0, 2, 1]),
            (b'u', Level::Unknown, [0, 2, 1]),
            (b'U', Level::Unknown, [0, 2, 1]),
            (b'w', Level::Unknown, [0, 2, 1]),
            (b'W', Level::Unknown, [0, 2, 1]),
            (b'-', Level::Unknown, [0, 2, 1]),
            (b'?', Level::Unknown, [0, 2, 1]),
        ];
        for (c, expected_level, weights) in table {
            assert_eq!(level(c), expected_level, "level of {}", c as char);
            for (policy, weight) in [AsZero, AsOne, Half].into_iter().zip(weights) {
                assert_eq!(
                    policy.half_weight(c),
                    weight,
                    "{policy:?} weight of {}",
                    c as char
                );
            }
        }
    }

    #[test]
    fn a_toggle_needs_different_lowercased_characters() {
        let h = UnknownPolicy::Half;
        // `h` and `1` are the same level but different characters: a toggle, not a rise.
        assert_eq!(tuple(chars_delta(b"h", b"1", h)), (1, 0, 0, 0));
        assert_eq!(tuple(chars_delta(b"0", b"h", h)), (1, 1, 0, 2));
        assert_eq!(tuple(chars_delta(b"1", b"l", h)), (1, 0, 1, -2));
        assert_eq!(tuple(chars_delta(b"l", b"0", h)), (1, 0, 0, 0));
        // Case alone is no change.
        assert_eq!(tuple(chars_delta(b"xZhL", b"XzHl", h)), (0, 0, 0, 0));
        // Unknown states are other changes: no rise, no fall.
        assert_eq!(tuple(chars_delta(b"01xz", b"10zx", h)), (4, 1, 1, 0));
        assert_eq!(
            tuple(chars_delta(b"x", b"1", UnknownPolicy::AsZero)),
            (1, 0, 0, 2)
        );
        assert_eq!(
            tuple(chars_delta(b"x", b"1", UnknownPolicy::AsOne)),
            (1, 0, 0, 0)
        );
        assert_eq!(tuple(chars_delta(b"x", b"1", h)), (1, 0, 0, 1));
        assert_eq!(tuple(chars_delta(b"u", b"-", h)), (1, 0, 0, 0));
    }

    #[test]
    fn the_first_value_counts_only_the_hamming_weight() {
        assert_eq!(
            tuple(chars_first(b"1hxl0Z", UnknownPolicy::Half)),
            (0, 0, 0, 2 + 2 + 1 + 1)
        );
        assert_eq!(
            tuple(chars_first(b"x1", UnknownPolicy::AsZero)),
            (0, 0, 0, 2)
        );
        assert_eq!(
            tuple(chars_first(b"x1", UnknownPolicy::AsOne)),
            (0, 0, 0, 4)
        );
    }
}
