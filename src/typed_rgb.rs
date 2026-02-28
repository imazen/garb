//! Type-safe pixel format conversions using [`rgb`] crate types via bytemuck.
//!
//! All conversions are zero-copy where possible (same-size in-place swaps)
//! and use the SIMD-optimized core operations internally.
//!
//! # In-place conversions (same pixel size)
//!
//! ```rust
//! use rgb::{Rgba, Bgra};
//! use garb::typed_rgb;
//!
//! let mut pixels: Vec<Rgba<u8>> = vec![Rgba::new(255, 0, 128, 255); 100];
//! let bgra: &mut [Bgra<u8>] = typed_rgb::rgba_to_bgra_mut(&mut pixels);
//! // bgra is now [Bgra { b: 128, g: 0, r: 255, a: 255 }; 100]
//! ```
//!
//! # Copy conversions (different pixel sizes)
//!
//! ```rust
//! use rgb::{Rgb, Bgra};
//! use garb::typed_rgb;
//!
//! let rgb_pixels: Vec<Rgb<u8>> = vec![Rgb::new(255, 0, 128); 100];
//! let mut bgra_buf: Vec<Bgra<u8>> = vec![Bgra::default(); 100];
//! typed_rgb::rgb_to_bgra_buf(&rgb_pixels, &mut bgra_buf).unwrap();
//! ```

use crate::SizeError;
use rgb::{Bgr, Bgra, Gray, GrayAlpha, Rgb, Rgba};

// ---------------------------------------------------------------------------
// In-place 4bpp swaps (RGBA ↔ BGRA) — zero-copy via bytemuck reinterpret
// ---------------------------------------------------------------------------

/// Convert `&mut [Rgba<u8>]` to `&mut [Bgra<u8>]` in-place by swapping R↔B.
///
/// Returns a bytemuck-reinterpreted reference to the same memory.
pub fn rgba_to_bgra_mut(pixels: &mut [Rgba<u8>]) -> &mut [Bgra<u8>] {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
    crate::bytes::rgba_to_bgra_inplace(bytes).expect("typed slice is always valid");
    bytemuck::cast_slice_mut(bytes)
}

/// Convert `&mut [Bgra<u8>]` to `&mut [Rgba<u8>]` in-place by swapping B↔R.
///
/// Returns a bytemuck-reinterpreted reference to the same memory.
pub fn bgra_to_rgba_mut(pixels: &mut [Bgra<u8>]) -> &mut [Rgba<u8>] {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
    crate::bytes::bgra_to_rgba_inplace(bytes).expect("typed slice is always valid");
    bytemuck::cast_slice_mut(bytes)
}

/// Convert `&mut [Rgb<u8>]` to `&mut [Bgr<u8>]` in-place by swapping R↔B.
pub fn rgb_to_bgr_mut(pixels: &mut [Rgb<u8>]) -> &mut [Bgr<u8>] {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
    crate::bytes::rgb_to_bgr_inplace(bytes).expect("typed slice is always valid");
    bytemuck::cast_slice_mut(bytes)
}

/// Convert `&mut [Bgr<u8>]` to `&mut [Rgb<u8>]` in-place by swapping B↔R.
pub fn bgr_to_rgb_mut(pixels: &mut [Bgr<u8>]) -> &mut [Rgb<u8>] {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
    crate::bytes::bgr_to_rgb_inplace(bytes).expect("typed slice is always valid");
    bytemuck::cast_slice_mut(bytes)
}

// ---------------------------------------------------------------------------
// Copy 4bpp swaps (RGBA → BGRA, BGRA → RGBA)
// ---------------------------------------------------------------------------

/// Copy `&[Rgba<u8>]` into `&mut [Bgra<u8>]`, swapping R↔B.
pub fn rgba_to_bgra_buf(src: &[Rgba<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::rgba_to_bgra(src_bytes, dst_bytes)
}

/// Copy `&[Bgra<u8>]` into `&mut [Rgba<u8>]`, swapping B↔R.
pub fn bgra_to_rgba_buf(src: &[Bgra<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::bgra_to_rgba(src_bytes, dst_bytes)
}

// ---------------------------------------------------------------------------
// 3→4 bpp expansions (RGB/BGR → RGBA/BGRA)
// ---------------------------------------------------------------------------

/// Copy `&[Rgb<u8>]` into `&mut [Bgra<u8>]`, reversing channels and adding alpha=255.
pub fn rgb_to_bgra_buf(src: &[Rgb<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::rgb_to_bgra(src_bytes, dst_bytes)
}

/// Copy `&[Rgb<u8>]` into `&mut [Rgba<u8>]`, adding alpha=255.
pub fn rgb_to_rgba_buf(src: &[Rgb<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::rgb_to_rgba(src_bytes, dst_bytes)
}

/// Copy `&[Bgr<u8>]` into `&mut [Rgba<u8>]`, reversing channels and adding alpha=255.
///
/// Same byte shuffle as [`rgb_to_bgra_buf`] (symmetric swap + alpha).
pub fn bgr_to_rgba_buf(src: &[Bgr<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::bgr_to_rgba(src_bytes, dst_bytes)
}

/// Copy `&[Bgr<u8>]` into `&mut [Bgra<u8>]`, adding alpha=255.
///
/// Same byte shuffle as [`rgb_to_rgba_buf`] (keep order + alpha).
pub fn bgr_to_bgra_buf(src: &[Bgr<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::bgr_to_bgra(src_bytes, dst_bytes)
}

// ---------------------------------------------------------------------------
// 1→4 and 2→4 bpp expansions (Gray/GrayAlpha → RGBA/BGRA)
// ---------------------------------------------------------------------------

/// Copy `&[Gray<u8>]` into `&mut [Rgba<u8>]`, broadcasting gray to RGB with alpha=255.
pub fn gray_to_rgba_buf(src: &[Gray<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::gray_to_rgba(src_bytes, dst_bytes)
}

/// Copy `&[Gray<u8>]` into `&mut [Bgra<u8>]`, broadcasting gray to BGR with alpha=255.
///
/// Output is identical to [`gray_to_rgba_buf`] since R=G=B.
pub fn gray_to_bgra_buf(src: &[Gray<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::gray_to_bgra(src_bytes, dst_bytes)
}

/// Copy `&[GrayAlpha<u8>]` into `&mut [Rgba<u8>]`, broadcasting gray to RGB.
pub fn gray_alpha_to_rgba_buf(
    src: &[GrayAlpha<u8>],
    dst: &mut [Rgba<u8>],
) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::gray_alpha_to_rgba(src_bytes, dst_bytes)
}

/// Copy `&[GrayAlpha<u8>]` into `&mut [Bgra<u8>]`, broadcasting gray to BGR.
///
/// Output is identical to [`gray_alpha_to_rgba_buf`] since R=G=B.
pub fn gray_alpha_to_bgra_buf(
    src: &[GrayAlpha<u8>],
    dst: &mut [Bgra<u8>],
) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::gray_alpha_to_bgra(src_bytes, dst_bytes)
}

// ---------------------------------------------------------------------------
// 4→3 bpp strip alpha
// ---------------------------------------------------------------------------

/// Copy `&[Rgba<u8>]` into `&mut [Rgb<u8>]`, dropping alpha.
pub fn rgba_to_rgb_buf(src: &[Rgba<u8>], dst: &mut [Rgb<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::rgba_to_rgb(src_bytes, dst_bytes)
}

/// Copy `&[Bgra<u8>]` into `&mut [Bgr<u8>]`, dropping alpha.
pub fn bgra_to_bgr_buf(src: &[Bgra<u8>], dst: &mut [Bgr<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::bgra_to_bgr(src_bytes, dst_bytes)
}

/// Copy `&[Bgra<u8>]` into `&mut [Rgb<u8>]`, dropping alpha and swapping B↔R.
pub fn bgra_to_rgb_buf(src: &[Bgra<u8>], dst: &mut [Rgb<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::bgra_to_rgb(src_bytes, dst_bytes)
}

/// Copy `&[Rgba<u8>]` into `&mut [Bgr<u8>]`, dropping alpha and swapping R↔B.
pub fn rgba_to_bgr_buf(src: &[Rgba<u8>], dst: &mut [Bgr<u8>]) -> Result<(), SizeError> {
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
    crate::bytes::rgba_to_bgr(src_bytes, dst_bytes)
}

// ---------------------------------------------------------------------------
// Alpha fill on typed slices
// ---------------------------------------------------------------------------

/// Set alpha to 255 for all pixels in a `&mut [Rgba<u8>]`.
pub fn fill_alpha_rgba(pixels: &mut [Rgba<u8>]) {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
    crate::bytes::fill_alpha_rgba(bytes).expect("typed slice is always valid");
}

/// Set alpha to 255 for all pixels in a `&mut [Bgra<u8>]`.
pub fn fill_alpha_bgra(pixels: &mut [Bgra<u8>]) {
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
    crate::bytes::fill_alpha_rgba(bytes).expect("typed slice is always valid");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;

    #[test]
    fn test_rgba_to_bgra_inplace() {
        let mut pixels = vec![Rgba::new(255u8, 128, 0, 200), Rgba::new(10, 20, 30, 40)];
        let bgra = rgba_to_bgra_mut(&mut pixels);
        assert_eq!(
            bgra[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 200
            }
        );
        assert_eq!(
            bgra[1],
            Bgra {
                b: 30,
                g: 20,
                r: 10,
                a: 40
            }
        );
    }

    #[test]
    fn test_bgra_to_rgba_inplace() {
        let mut pixels = vec![Bgra {
            b: 0u8,
            g: 128,
            r: 255,
            a: 200,
        }];
        let rgba = bgra_to_rgba_mut(&mut pixels);
        assert_eq!(rgba[0], Rgba::new(255, 128, 0, 200));
    }

    #[test]
    fn test_rgb_to_bgra_copy() {
        let src = vec![Rgb::new(255u8, 128, 0)];
        let mut dst = vec![Bgra::default(); 1];
        rgb_to_bgra_buf(&src, &mut dst).unwrap();
        assert_eq!(
            dst[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 255
            }
        );
    }

    #[test]
    fn test_rgb_to_rgba_copy() {
        let src = vec![Rgb::new(255u8, 128, 0)];
        let mut dst = vec![Rgba::default(); 1];
        rgb_to_rgba_buf(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgba::new(255, 128, 0, 255));
    }

    #[test]
    fn test_bgr_to_rgba_copy() {
        let src = vec![Bgr {
            b: 0u8,
            g: 128,
            r: 255,
        }];
        let mut dst = vec![Rgba::default(); 1];
        bgr_to_rgba_buf(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgba::new(255, 128, 0, 255));
    }

    #[test]
    fn test_gray_to_rgba() {
        let src = vec![Gray::new(100u8)];
        let mut dst = vec![Rgba::default(); 1];
        gray_to_rgba_buf(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgba::new(100, 100, 100, 255));
    }

    #[test]
    fn test_gray_alpha_to_rgba() {
        let src = vec![GrayAlpha::new(100u8, 50)];
        let mut dst = vec![Rgba::default(); 1];
        gray_alpha_to_rgba_buf(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgba::new(100, 100, 100, 50));
    }

    #[test]
    fn test_rgba_to_rgb_drop_alpha() {
        let src = vec![Rgba::new(255u8, 128, 0, 200)];
        let mut dst = vec![Rgb::default(); 1];
        rgba_to_rgb_buf(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgb::new(255, 128, 0));
    }

    #[test]
    fn test_bgra_to_rgb_drop_and_swap() {
        let src = vec![Bgra {
            b: 0u8,
            g: 128,
            r: 255,
            a: 200,
        }];
        let mut dst = vec![Rgb::default(); 1];
        bgra_to_rgb_buf(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgb::new(255, 128, 0));
    }

    #[test]
    fn test_fill_alpha() {
        let mut pixels = vec![Rgba::new(10u8, 20, 30, 0), Rgba::new(40, 50, 60, 100)];
        fill_alpha_rgba(&mut pixels);
        assert_eq!(pixels[0].a, 255);
        assert_eq!(pixels[1].a, 255);
    }

    #[test]
    fn test_rgb_bgr_roundtrip() {
        let original = vec![Rgb::new(10u8, 20, 30), Rgb::new(40, 50, 60)];
        let mut pixels = original.clone();
        let bgr = rgb_to_bgr_mut(&mut pixels);
        assert_eq!(
            bgr[0],
            Bgr {
                b: 30,
                g: 20,
                r: 10
            }
        );
        let rgb_again = bgr_to_rgb_mut(bgr);
        assert_eq!(rgb_again, original.as_slice());
    }

    #[test]
    fn test_size_mismatch_returns_error() {
        let src = vec![Rgba::new(1u8, 2, 3, 4); 3];
        let mut dst = vec![Bgra::default(); 2]; // wrong size
        assert_eq!(
            rgba_to_bgra_buf(&src, &mut dst),
            Err(SizeError::PixelCountMismatch)
        );
    }
}
