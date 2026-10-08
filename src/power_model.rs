pub trait Hamming {
    /// Get the Hamming weight of the value, i.e. number of bits set to `1`.
    fn hamming_weight(&self) -> u32;
    /// Get the Hamming distance between two values, i.e. number of bits that differ.
    fn hamming_distance(&self, other: &Self) -> u32;
}

#[inline(always)]
pub fn power_model<V: Hamming>(prev_value: &V, new_value: &V) -> f32 {
    // new_value.hamming_weight() as f32 * 0.1 + // static power
    new_value.hamming_distance(prev_value) as f32
}

impl Hamming for wellen::SignalValueRef<'_> {
    #[inline(always)]
    fn hamming_weight(&self) -> u32 {
        match self {
            wellen::SignalValueRef::BitVec(v) => match v.be_bytes() {
                Some(bytes) => {
                    let mut iter = bytes.iter();
                    let first = iter.next().map_or(0, |b| b & first_byte_mask(v.width()));
                    first.count_ones() + iter.map(|b| b.count_ones()).sum::<u32>()
                }
                // only bits that are a definite `1` count; X/Z/etc. contribute nothing
                None => v
                    .iter_lsb_to_msb()
                    .filter(|&b| u8::from(b) == 1) // numeric value of `Bit::ONE`
                    .count() as u32,
            },
            _ => 0,
        }
    }
    #[inline(always)]
    fn hamming_distance(&self, other: &Self) -> u32 {
        match (self, other) {
            (wellen::SignalValueRef::BitVec(a), wellen::SignalValueRef::BitVec(b)) => {
                assert_eq!(a.width(), b.width(), "Cannot compare different bit widths!");
                match (a.be_bytes(), b.be_bytes()) {
                    // fast path: both values are 2-state, XOR the packed bytes
                    (Some(a_bytes), Some(b_bytes)) => {
                        let mut iter = a_bytes.iter().zip(b_bytes.iter());
                        let first = iter
                            .next()
                            .map_or(0, |(x, y)| (x ^ y) & first_byte_mask(a.width()));
                        first.count_ones() + iter.map(|(x, y)| (x ^ y).count_ones()).sum::<u32>()
                    }
                    // slow path: 4- or 9-state values, count bits whose state changed
                    _ => a
                        .iter_lsb_to_msb()
                        .zip(b.iter_lsb_to_msb())
                        .filter(|&(x, y)| u8::from(x) != u8::from(y))
                        .count() as u32,
                }
            }
            _ => 0,
        }
    }
}

/// Mask of the valid bits in the most significant byte of a big-endian, 2-state bit vector.
#[inline(always)]
fn first_byte_mask(width: u32) -> u8 {
    match width % 8 {
        0 => u8::MAX,
        n => (1u8 << n) - 1,
    }
}
