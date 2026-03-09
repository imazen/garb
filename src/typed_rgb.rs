//! Type-safe pixel format conversions using [`rgb`] crate types via bytemuck.
//!
//! All conversions are zero-copy where possible (same-size in-place swaps)
//! and use the SIMD-optimized core operations internally.
//!
//! # Generic API
//!
//! ```rust
//! use rgb::{Rgba, Bgra, Rgb};
//! use garb::{convert, convert_inplace};
//!
//! // Copy: type-inferred from src/dst
//! let rgb = vec![Rgb::new(255u8, 0, 128); 2];
//! let mut bgra = vec![Bgra::default(); 2];
//! convert(&rgb, &mut bgra).unwrap();
//!
//! // In-place: returns reinterpreted reference
//! let mut pixels = vec![Rgba::new(255u8, 0, 128, 255); 2];
//! let bgra: &mut [Bgra<u8>] = convert_inplace(&mut pixels);
//! ```

use crate::{ConvertInplace, ConvertTo, SizeError};
use rgb::{Bgr, Bgra, Gray, GrayAlpha, Rgb, Rgba};

// ===========================================================================
// Macro for ConvertTo impls (copy conversions)
// ===========================================================================

macro_rules! impl_convert_to {
    ($src:ty, $dst:ty, $bytes_fn:path) => {
        impl ConvertTo<$dst> for $src {
            #[inline]
            fn convert_to(src: &[Self], dst: &mut [$dst]) -> Result<(), SizeError> {
                let s: &[u8] = bytemuck::cast_slice(src);
                let d: &mut [u8] = bytemuck::cast_slice_mut(dst);
                $bytes_fn(s, d)
            }
        }
    };
}

// ===========================================================================
// Macro for ConvertInplace impls (same-size in-place swaps)
// ===========================================================================

macro_rules! impl_convert_inplace {
    ($src:ty, $dst:ty, $bytes_fn:path) => {
        impl ConvertInplace<$dst> for $src {
            #[inline]
            fn convert_inplace(buf: &mut [Self]) -> &mut [$dst] {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(buf);
                $bytes_fn(bytes).expect("typed slice is always valid");
                bytemuck::cast_slice_mut(bytes)
            }
        }
    };
}

// ===========================================================================
// In-place 4bpp swaps
// ===========================================================================

impl_convert_inplace!(Rgba<u8>, Bgra<u8>, crate::bytes::rgba_to_bgra_inplace);
impl_convert_inplace!(Bgra<u8>, Rgba<u8>, crate::bytes::bgra_to_rgba_inplace);
impl_convert_inplace!(Rgb<u8>, Bgr<u8>, crate::bytes::rgb_to_bgr_inplace);
impl_convert_inplace!(Bgr<u8>, Rgb<u8>, crate::bytes::bgr_to_rgb_inplace);

// ===========================================================================
// Copy 4bpp swaps
// ===========================================================================

impl_convert_to!(Rgba<u8>, Bgra<u8>, crate::bytes::rgba_to_bgra);
impl_convert_to!(Bgra<u8>, Rgba<u8>, crate::bytes::bgra_to_rgba);

// ===========================================================================
// 3→4 bpp expansions
// ===========================================================================

impl_convert_to!(Rgb<u8>, Bgra<u8>, crate::bytes::rgb_to_bgra);
impl_convert_to!(Rgb<u8>, Rgba<u8>, crate::bytes::rgb_to_rgba);
impl_convert_to!(Bgr<u8>, Rgba<u8>, crate::bytes::bgr_to_rgba);
impl_convert_to!(Bgr<u8>, Bgra<u8>, crate::bytes::bgr_to_bgra);

// ===========================================================================
// 1→4 and 2→4 bpp expansions
// ===========================================================================

impl_convert_to!(Gray<u8>, Rgba<u8>, crate::bytes::gray_to_rgba);
impl_convert_to!(Gray<u8>, Bgra<u8>, crate::bytes::gray_to_bgra);
impl_convert_to!(GrayAlpha<u8>, Rgba<u8>, crate::bytes::gray_alpha_to_rgba);
impl_convert_to!(GrayAlpha<u8>, Bgra<u8>, crate::bytes::gray_alpha_to_bgra);

// ===========================================================================
// 4→3 bpp strip alpha
// ===========================================================================

impl_convert_to!(Rgba<u8>, Rgb<u8>, crate::bytes::rgba_to_rgb);
impl_convert_to!(Bgra<u8>, Bgr<u8>, crate::bytes::bgra_to_bgr);
impl_convert_to!(Bgra<u8>, Rgb<u8>, crate::bytes::bgra_to_rgb);
impl_convert_to!(Rgba<u8>, Bgr<u8>, crate::bytes::rgba_to_bgr);

// ===========================================================================
// Alpha fill (not a conversion pair — standalone functions)
// ===========================================================================

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

// ===========================================================================
// Deprecated named functions — use convert() / convert_inplace() instead
// ===========================================================================

/// Use [`convert_inplace`](crate::convert_inplace) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert_inplace() instead")]
pub fn rgba_to_bgra_mut(pixels: &mut [Rgba<u8>]) -> &mut [Bgra<u8>] {
    crate::convert_inplace(pixels)
}

/// Use [`convert_inplace`](crate::convert_inplace) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert_inplace() instead")]
pub fn bgra_to_rgba_mut(pixels: &mut [Bgra<u8>]) -> &mut [Rgba<u8>] {
    crate::convert_inplace(pixels)
}

/// Use [`convert_inplace`](crate::convert_inplace) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert_inplace() instead")]
pub fn rgb_to_bgr_mut(pixels: &mut [Rgb<u8>]) -> &mut [Bgr<u8>] {
    crate::convert_inplace(pixels)
}

/// Use [`convert_inplace`](crate::convert_inplace) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert_inplace() instead")]
pub fn bgr_to_rgb_mut(pixels: &mut [Bgr<u8>]) -> &mut [Rgb<u8>] {
    crate::convert_inplace(pixels)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn rgba_to_bgra_buf(src: &[Rgba<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn bgra_to_rgba_buf(src: &[Bgra<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn rgb_to_bgra_buf(src: &[Rgb<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn rgb_to_rgba_buf(src: &[Rgb<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn bgr_to_rgba_buf(src: &[Bgr<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn bgr_to_bgra_buf(src: &[Bgr<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn gray_to_rgba_buf(src: &[Gray<u8>], dst: &mut [Rgba<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn gray_to_bgra_buf(src: &[Gray<u8>], dst: &mut [Bgra<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn gray_alpha_to_rgba_buf(
    src: &[GrayAlpha<u8>],
    dst: &mut [Rgba<u8>],
) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn gray_alpha_to_bgra_buf(
    src: &[GrayAlpha<u8>],
    dst: &mut [Bgra<u8>],
) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn rgba_to_rgb_buf(src: &[Rgba<u8>], dst: &mut [Rgb<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn bgra_to_bgr_buf(src: &[Bgra<u8>], dst: &mut [Bgr<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn bgra_to_rgb_buf(src: &[Bgra<u8>], dst: &mut [Rgb<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

/// Use [`convert`](crate::convert) instead.
#[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
pub fn rgba_to_bgr_buf(src: &[Rgba<u8>], dst: &mut [Bgr<u8>]) -> Result<(), SizeError> {
    crate::convert(src, dst)
}

// ===========================================================================
// Experimental (feature = "experimental")
// ===========================================================================

#[cfg(feature = "experimental")]
mod experimental_typed {
    use super::*;

    // -----------------------------------------------------------------------
    // Trait impls for experimental conversions
    // -----------------------------------------------------------------------

    impl_convert_to!(Gray<u8>, Rgb<u8>, crate::bytes::gray_to_rgb);
    impl_convert_to!(Gray<u8>, Bgr<u8>, crate::bytes::gray_to_bgr);
    impl_convert_to!(GrayAlpha<u8>, Rgb<u8>, crate::bytes::gray_alpha_to_rgb);
    impl_convert_to!(GrayAlpha<u8>, Bgr<u8>, crate::bytes::gray_alpha_to_bgr);
    impl_convert_to!(Gray<u8>, GrayAlpha<u8>, crate::bytes::gray_to_gray_alpha);
    impl_convert_to!(GrayAlpha<u8>, Gray<u8>, crate::bytes::gray_alpha_to_gray);
    impl_convert_to!(Rgb<u8>, Gray<u8>, crate::bytes::rgb_to_gray_identity);
    impl_convert_to!(Rgba<u8>, Gray<u8>, crate::bytes::rgba_to_gray_identity);
    impl_convert_to!(Bgr<u8>, Gray<u8>, crate::bytes::bgr_to_gray_identity);
    impl_convert_to!(Bgra<u8>, Gray<u8>, crate::bytes::bgra_to_gray_identity);

    // -----------------------------------------------------------------------
    // Deprecated named functions
    // -----------------------------------------------------------------------

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    pub fn gray_to_rgb_buf(src: &[Gray<u8>], dst: &mut [Rgb<u8>]) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    #[inline(always)]
    pub fn gray_to_bgr_buf(src: &[Gray<u8>], dst: &mut [Bgr<u8>]) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    pub fn gray_alpha_to_rgb_buf(
        src: &[GrayAlpha<u8>],
        dst: &mut [Rgb<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    #[inline(always)]
    pub fn gray_alpha_to_bgr_buf(
        src: &[GrayAlpha<u8>],
        dst: &mut [Bgr<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    pub fn gray_to_gray_alpha_buf(
        src: &[Gray<u8>],
        dst: &mut [GrayAlpha<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    pub fn gray_alpha_to_gray_buf(
        src: &[GrayAlpha<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    pub fn rgb_to_gray_identity_buf(
        src: &[Rgb<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    pub fn rgba_to_gray_identity_buf(
        src: &[Rgba<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    #[inline(always)]
    pub fn bgr_to_gray_identity_buf(
        src: &[Bgr<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    /// Use [`convert`](crate::convert) instead.
    #[deprecated(since = "0.2.1", note = "use garb::convert() instead")]
    #[inline(always)]
    pub fn bgra_to_gray_identity_buf(
        src: &[Bgra<u8>],
        dst: &mut [Gray<u8>],
    ) -> Result<(), SizeError> {
        crate::convert(src, dst)
    }

    // -----------------------------------------------------------------------
    // Weighted luma — these use specific matrices, not a single trait pair
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // f32 alpha premultiplication (not convertible to trait pairs)
    // -----------------------------------------------------------------------

    /// Premultiply alpha for `&mut [Rgba<f32>]` in-place.
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

    /// Unpremultiply alpha for `&mut [Rgba<f32>]` in-place.
    pub fn unpremultiply_rgba_f32(pixels: &mut [Rgba<f32>]) {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
        crate::bytes::unpremultiply_alpha_f32(bytes).expect("typed slice is always valid");
    }

    /// Unpremultiply alpha, copying from `&[Rgba<f32>]` to `&mut [Rgba<f32>]`.
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use rgb::{Bgra, Rgba};

    #[test]
    fn test_convert_copy() {
        let src = vec![Rgba::new(255u8, 128, 0, 200), Rgba::new(10, 20, 30, 40)];
        let mut dst = vec![Bgra::default(); 2];
        crate::convert(&src, &mut dst).unwrap();
        assert_eq!(
            dst[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 200
            }
        );
    }

    #[test]
    fn test_convert_inplace() {
        let mut pixels = vec![Rgba::new(255u8, 128, 0, 200), Rgba::new(10, 20, 30, 40)];
        let bgra: &mut [Bgra<u8>] = crate::convert_inplace(&mut pixels);
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
    fn test_convert_inplace_roundtrip() {
        let original = vec![Rgba::new(255u8, 128, 0, 200)];
        let mut pixels = original.clone();
        let bgra: &mut [Bgra<u8>] = crate::convert_inplace(&mut pixels);
        let rgba: &mut [Rgba<u8>] = crate::convert_inplace(bgra);
        assert_eq!(rgba, original.as_slice());
    }

    #[test]
    fn test_convert_rgb_to_bgra() {
        use rgb::Rgb;
        let src = vec![Rgb::new(255u8, 128, 0)];
        let mut dst = vec![Bgra::default(); 1];
        crate::convert(&src, &mut dst).unwrap();
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
    fn test_convert_gray_to_rgba() {
        use rgb::Gray;
        let src = vec![Gray::new(100u8)];
        let mut dst = vec![Rgba::default(); 1];
        crate::convert(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgba::new(100, 100, 100, 255));
    }

    #[test]
    fn test_convert_strip_alpha() {
        use rgb::Rgb;
        let src = vec![Rgba::new(255u8, 128, 0, 200)];
        let mut dst = vec![Rgb::default(); 1];
        crate::convert(&src, &mut dst).unwrap();
        assert_eq!(dst[0], Rgb::new(255, 128, 0));
    }

    #[test]
    fn test_fill_alpha() {
        let mut pixels = vec![Rgba::new(10u8, 20, 30, 0), Rgba::new(40, 50, 60, 100)];
        super::fill_alpha_rgba(&mut pixels);
        assert_eq!(pixels[0].a, 255);
        assert_eq!(pixels[1].a, 255);
    }

    #[test]
    fn test_size_mismatch_returns_error() {
        let src = vec![Rgba::new(1u8, 2, 3, 4); 3];
        let mut dst = vec![Bgra::default(); 2];
        assert_eq!(
            crate::convert(&src, &mut dst),
            Err(crate::SizeError::PixelCountMismatch)
        );
    }

    #[test]
    fn test_rgb_bgr_roundtrip() {
        use rgb::{Bgr, Rgb};
        let original = vec![Rgb::new(10u8, 20, 30), Rgb::new(40, 50, 60)];
        let mut pixels = original.clone();
        let bgr: &mut [Bgr<u8>] = crate::convert_inplace(&mut pixels);
        assert_eq!(
            bgr[0],
            Bgr {
                b: 30,
                g: 20,
                r: 10
            }
        );
        let rgb_again: &mut [Rgb<u8>] = crate::convert_inplace(bgr);
        assert_eq!(rgb_again, original.as_slice());
    }

    #[allow(deprecated)]
    #[test]
    fn test_deprecated_fns_still_work() {
        let mut pixels = vec![Rgba::new(255u8, 128, 0, 200)];
        let bgra = super::rgba_to_bgra_mut(&mut pixels);
        assert_eq!(
            bgra[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 200
            }
        );
    }
}
