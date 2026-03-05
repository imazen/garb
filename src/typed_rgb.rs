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

#[cfg(feature = "experimental")]
mod experimental_typed {
    use super::*;

    // ---------------------------------------------------------------------------
    // Gray layout conversions
    // ---------------------------------------------------------------------------

    /// Copy `&[Gray<u8>]` into `&mut [Rgb<u8>]`, broadcasting gray to R=G=B.
    pub fn gray_to_rgb_buf(src: &[Gray<u8>], dst: &mut [Rgb<u8>]) -> Result<(), SizeError> {
        crate::bytes::gray_to_rgb(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Alias for [`gray_to_rgb_buf`] — R=G=B so output is identical.
    #[inline(always)]
    pub fn gray_to_bgr_buf(src: &[Gray<u8>], dst: &mut [Bgr<u8>]) -> Result<(), SizeError> {
        crate::bytes::gray_to_bgr(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Copy `&[GrayAlpha<u8>]` into `&mut [Rgb<u8>]`, broadcasting gray, dropping alpha.
    pub fn gray_alpha_to_rgb_buf(
        src: &[GrayAlpha<u8>],
        dst: &mut [Rgb<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::gray_alpha_to_rgb(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Alias for [`gray_alpha_to_rgb_buf`].
    #[inline(always)]
    pub fn gray_alpha_to_bgr_buf(
        src: &[GrayAlpha<u8>],
        dst: &mut [Bgr<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::gray_alpha_to_bgr(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Copy `&[Gray<u8>]` into `&mut [GrayAlpha<u8>]`, adding alpha=255.
    pub fn gray_to_gray_alpha_buf(
        src: &[Gray<u8>],
        dst: &mut [GrayAlpha<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::gray_to_gray_alpha(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Copy `&[GrayAlpha<u8>]` into `&mut [Gray<u8>]`, dropping alpha.
    pub fn gray_alpha_to_gray_buf(
        src: &[GrayAlpha<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::gray_alpha_to_gray(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Identity gray extraction: `&[Rgb<u8>]` → `&[Gray<u8>]` (takes R channel).
    pub fn rgb_to_gray_identity_buf(
        src: &[Rgb<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::rgb_to_gray_identity(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Identity gray extraction: `&[Rgba<u8>]` → `&[Gray<u8>]` (takes R channel).
    pub fn rgba_to_gray_identity_buf(
        src: &[Rgba<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::rgba_to_gray_identity(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
        )
    }

    /// Alias for [`rgb_to_gray_identity_buf`].
    #[inline(always)]
    pub fn bgr_to_gray_identity_buf(
        src: &[Bgr<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::bgr_to_gray_identity(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
    }

    /// Alias for [`rgba_to_gray_identity_buf`].
    #[inline(always)]
    pub fn bgra_to_gray_identity_buf(
        src: &[Bgra<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::bytes::bgra_to_gray_identity(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
        )
    }

    // ---------------------------------------------------------------------------
    // Weighted luma — RGB/RGBA → Gray
    // ---------------------------------------------------------------------------

    macro_rules! luma_typed {
    ($matrix:ident, $doc_matrix:expr) => {
        paste::paste! {
            #[doc = concat!("RGB → Gray using ", $doc_matrix, " luma weights.")]
            pub fn [<rgb_to_gray_ $matrix _buf>](src: &[Rgb<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
                crate::bytes::[<rgb_to_gray_ $matrix>](bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
            }
            #[doc = concat!("BGR → Gray using ", $doc_matrix, " luma weights.")]
            pub fn [<bgr_to_gray_ $matrix _buf>](src: &[Bgr<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
                crate::bytes::[<bgr_to_gray_ $matrix>](bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
            }
            #[doc = concat!("RGBA → Gray using ", $doc_matrix, " luma weights.")]
            pub fn [<rgba_to_gray_ $matrix _buf>](src: &[Rgba<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
                crate::bytes::[<rgba_to_gray_ $matrix>](bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
            }
            #[doc = concat!("BGRA → Gray using ", $doc_matrix, " luma weights.")]
            pub fn [<bgra_to_gray_ $matrix _buf>](src: &[Bgra<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
                crate::bytes::[<bgra_to_gray_ $matrix>](bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst))
            }
        }
    };
}

    luma_typed!(bt709, "BT.709");
    luma_typed!(bt601, "BT.601");
    luma_typed!(bt2020, "BT.2020");

    /// Default alias: RGB → Gray uses BT.709.
    #[inline(always)]
    pub fn rgb_to_gray_buf(src: &[Rgb<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
        rgb_to_gray_bt709_buf(src, dst)
    }

    /// Default alias: BGR → Gray uses BT.709.
    #[inline(always)]
    pub fn bgr_to_gray_buf(src: &[Bgr<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
        bgr_to_gray_bt709_buf(src, dst)
    }

    /// Default alias: RGBA → Gray uses BT.709.
    #[inline(always)]
    pub fn rgba_to_gray_buf(src: &[Rgba<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
        rgba_to_gray_bt709_buf(src, dst)
    }

    /// Default alias: BGRA → Gray uses BT.709.
    #[inline(always)]
    pub fn bgra_to_gray_buf(src: &[Bgra<u8>], dst: &mut [Gray<u8>]) -> Result<(), SizeError> {
        bgra_to_gray_bt709_buf(src, dst)
    }

    // ---------------------------------------------------------------------------
    // f32 alpha premultiplication
    // ---------------------------------------------------------------------------

    /// Premultiply alpha for `&mut [Rgba<f32>]` in-place: `C' = C * A`, alpha preserved.
    pub fn premultiply_rgba_f32(pixels: &mut [Rgba<f32>]) {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
        crate::bytes::premultiply_alpha_f32(bytes).expect("typed slice is always valid");
    }

    /// Premultiply alpha, copying from `&[Rgba<f32>]` to `&mut [Rgba<f32>]`.
    pub fn premultiply_rgba_f32_buf(
        src: &[Rgba<f32>],
        dst: &mut [Rgba<f32>],
    ) -> Result<(), SizeError> {
        crate::bytes::premultiply_alpha_f32_copy(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
        )
    }

    /// Unpremultiply alpha for `&mut [Rgba<f32>]` in-place: `C' = C / A`.
    /// Where alpha is zero, all channels are set to zero.
    pub fn unpremultiply_rgba_f32(pixels: &mut [Rgba<f32>]) {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
        crate::bytes::unpremultiply_alpha_f32(bytes).expect("typed slice is always valid");
    }

    /// Unpremultiply alpha, copying from `&[Rgba<f32>]` to `&mut [Rgba<f32>]`.
    /// Where alpha is zero, all channels are set to zero.
    pub fn unpremultiply_rgba_f32_buf(
        src: &[Rgba<f32>],
        dst: &mut [Rgba<f32>],
    ) -> Result<(), SizeError> {
        crate::bytes::unpremultiply_alpha_f32_copy(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
        )
    }
} // mod experimental_typed
#[cfg(feature = "experimental")]
pub use experimental_typed::*;

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
