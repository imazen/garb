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
//! ## Modules
//!
//! - [`bytes`] — Core `&[u8]` conversions (contiguous and strided).
//! - [`typed_rgb`] — Type-safe wrappers using `rgb` crate pixel types (feature `rgb`).
//! - [`imgref`] — Whole-image conversions on `ImgVec` / `ImgRef` (feature `imgref`).
//!
//! ## Naming convention
//!
//! All functions follow the pattern `{src}_to_{dst}` for copy operations and
//! `{src}_to_{dst}_inplace` for in-place mutations. Append `_strided` for
//! multi-row buffers with stride.
//!
//! ## Strides
//!
//! A **stride** (also called "pitch" or "row pitch") is the distance between
//! the start of one row and the start of the next, measured in units of the
//! slice's element type:
//!
//! - For `&[u8]` functions: stride is in **bytes**.
//! - For typed APIs (`imgref`, `typed_rgb`): stride is in **elements** of the
//!   slice's item type — e.g. pixel count for `ImgRef<Rgba<u8>>`, but byte
//!   count for `ImgRef<u8>`.
//!
//! When `stride == width` (in the appropriate unit) the image is contiguous.
//! When `stride > width` the gap at the end of each row is padding — garb
//! never reads or writes it.
//!
//! The required buffer length (in elements) for a strided image is:
//! `(height - 1) * stride + width`
//!
//! All `_strided` functions take dimensions *before* strides:
//! - In-place: `(buf, width, height, stride)`
//! - Copy: `(src, dst, width, height, src_stride, dst_stride)`
//!
//! ## Feature flags
//!
//! - **`std`** — Enables `std` on dependencies (e.g. `archmage`).
//! - **`rgb`** — Type-safe conversions using [`rgb`] crate pixel types
//!   via bytemuck. Zero-copy in-place swaps return reinterpreted references.
//! - **`imgref`** — Multi-row conversions using `ImgRef` / `ImgRefMut`
//!   from the [`imgref`](https://docs.rs/imgref) crate. No allocation — caller owns all buffers.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "imgref")]
extern crate alloc;

pub mod bytes;

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
            Self::InvalidStride => {
                f.write_str("stride, dimensions, or buffer size are inconsistent")
            }
        }
    }
}

impl core::error::Error for SizeError {}
