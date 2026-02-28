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
//! ## Core operations (always available)
//!
//! All functions in the crate root operate on raw `&[u8]` / `&mut [u8]` slices
//! at row granularity. They are the SIMD-accelerated building blocks.
//!
//! ## Feature flags
//!
//! - **`rgb`** — Type-safe conversions using [`rgb`] crate pixel types
//!   (`Rgb<u8>`, `Rgba<u8>`, `Bgr<u8>`, `Bgra<u8>`, etc.) via bytemuck.
//! - **`imgref`** — Whole-image conversions using [`imgref`] types
//!   (`ImgRef`, `ImgVec`). Implies `rgb`.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

mod swizzle;

pub use swizzle::*;

#[cfg(feature = "rgb")]
pub mod typed;

#[cfg(feature = "imgref")]
pub mod img;
