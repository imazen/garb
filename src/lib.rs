//! # garb
//!
//! *Dress your pixels for the occasion.*
//!
//! You can't show up to a function in the wrong style. Get with the times and
//! swap your BGR for your RGB, your ARGB for your RGBA, your BGRX for your
//! BGRA, and tie up loose ends like that unreliable alpha.
//!
//! SIMD-optimized pixel format conversions for row-level and whole-image
//! operations. Supports x86-64 AVX2, ARM NEON, and WASM SIMD128 with
//! automatic fallback to scalar code.
//!
//! ## Naming convention
//!
//! All functions follow the pattern `{src}_to_{dst}` for copy operations and
//! `{src}_to_{dst}_inplace` for in-place mutations. Append `_strided` for
//! multi-row buffers with stride.
//!
//! ## Feature flags
//!
//! - **`rgb`** — Type-safe conversions using [`rgb`] crate pixel types
//!   via bytemuck. Zero-copy in-place swaps return reinterpreted references.
//! - **`imgref`** — Multi-row conversions using `ImgRef` / `ImgRefMut`
//!   from the [`imgref`](https://docs.rs/imgref) crate. No allocation — caller owns all buffers.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "imgref")]
extern crate alloc;

mod swizzle;
pub use swizzle::*;

#[cfg(feature = "rgb")]
pub mod typed_rgb;

#[cfg(feature = "imgref")]
pub mod imgref;

/// Pixel buffer size or alignment error.
///
/// Returned when a buffer's length is not a multiple of the pixel size,
/// a destination buffer is too small, or stride/dimensions are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SizeError {
    /// Buffer length is not a multiple of the expected bytes per pixel (or is empty).
    NotPixelAligned,
    /// Destination buffer has fewer pixels than the source.
    PixelCountMismatch,
    /// Stride, dimensions, or total buffer size are inconsistent.
    InvalidStride,
}

impl core::fmt::Display for SizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPixelAligned => f.write_str("buffer length is not pixel-aligned"),
            Self::PixelCountMismatch => f.write_str("destination has fewer pixels than source"),
            Self::InvalidStride => f.write_str("stride, dimensions, or buffer size are inconsistent"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SizeError {}
