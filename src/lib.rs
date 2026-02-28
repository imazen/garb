//! # garb
//!
//! *Dress your pixels for the occasion.*
//!
//! You can't show up to a function in the wrong style! Get with the times and
//! swap your BGR for your RGB, your ARGB for your RGBA, your BGRA for your
//! BGRX, and tie up loose ends like that unreliable alpha.
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
//! - **`imgref`** — Multi-row conversions using [`imgref::ImgRef`] /
//!   [`imgref::ImgRefMut`]. No allocation — caller owns all buffers.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "imgref")]
extern crate alloc;

mod swizzle;
pub use swizzle::*;

#[cfg(feature = "rgb")]
pub mod typed;

#[cfg(feature = "imgref")]
pub mod img;

/// Pixel buffer size or alignment error.
///
/// Returned when a buffer's length is not a multiple of the pixel size,
/// or a destination buffer is too small for the source pixel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeError;

impl core::fmt::Display for SizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("pixel buffer size mismatch")
    }
}
