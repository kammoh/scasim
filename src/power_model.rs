
pub trait Hamming {
    /// Get the Hamming weight of the value, i.e. number of bits set to `1`.
    fn hamming_weight(&self) -> u32;
    /// Get the Hamming distance between two values, i.e. number of bits that differ.
    fn hamming_distance(&self, other: &Self) -> u32;
}

#[inline(always)]
pub fn power_model<V: Hamming>(prev_value: &V, new_value: &V) -> f32 {
    // new_value.hamming_weight() as f32 * 0.1 + // static power
    new_value.hamming_distance(&prev_value) as f32
}

impl<'a> Hamming for wellen::SignalValue<'a> {
    #[inline(always)]
    fn hamming_weight(&self) -> u32 {
        match self {
            wellen::SignalValue::Binary(data, bits) => {
                if *bits == 0 {
                    panic!("Cannot compute hamming weight of empty signal!");
                }
                data.iter().map(|&x| x.count_ones()).sum()
            }
            _ => 0,
        }
    }
    #[inline(always)]
    fn hamming_distance(&self, other: &Self) -> u32 {
        match (self, other) {
            (
                wellen::SignalValue::Binary(self_data, self_bits),
                wellen::SignalValue::Binary(other_data, other_bits),
            ) => {
                if self_bits != other_bits {
                    panic!("Cannot compare different bit widths!");
                }
                self_data
                    .iter()
                    .zip(other_data.iter())
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum()
            }
            _ => 0,
        }
    }
}
