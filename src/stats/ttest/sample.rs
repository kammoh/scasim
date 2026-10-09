//! The sample types that the accumulator accepts.

/// A numeric type that a trace sample can have.
///
/// The accumulator converts every sample to `f64` before it does any arithmetic.
/// The conversion is exact for `f32`, `f64`, and for integers with at most 53
/// significant bits. A `u64` or `i64` value that needs more bits is rounded to the
/// nearest `f64` value.
pub trait TraceSample: Copy + Send + Sync + 'static {
    /// Converts the sample to `f64`.
    fn to_f64(self) -> f64;
}

macro_rules! impl_trace_sample {
    ($($t:ty),*) => {
        $(
            impl TraceSample for $t {
                #[inline(always)]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_trace_sample!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
