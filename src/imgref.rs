//! Whole-image pixel format conversions using [`imgref`] types.
//!
//! These functions handle strided image buffers using the SIMD-optimized
//! core operations. No allocation — caller owns all buffers.
//!
//! # Generic API
//!
//! ```rust
//! use rgb::{Rgba, Bgra, Rgb};
//! use ::imgref::{ImgVec, ImgRefMut};
//! use garb::{convert_imgref, convert_imgref_inplace};
//!
//! // In-place: consumes ImgVec, returns reinterpreted
//! let img = ImgVec::new(vec![Rgba::new(255u8, 0, 128, 200); 4], 2, 2);
//! let bgra_img: ImgVec<Bgra<u8>> = convert_imgref_inplace(img);
//!
//! // Copy: ImgRef → ImgRefMut
//! let src = ImgVec::new(vec![Rgb::new(255u8, 0, 128); 4], 2, 2);
//! let mut dst_buf = vec![Bgra::default(); 4];
//! let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
//! convert_imgref(src.as_ref(), dst).unwrap();
//! ```

use alloc::vec::Vec;

use imgref::{ImgRef, ImgRefMut, ImgVec};
use rgb::{Bgr, Bgra, Gray, GrayAlpha, Rgb, Rgba};

use crate::{ConvertImage, ConvertImageInplace, SizeError};

// ---------------------------------------------------------------------------
// Dimension check
// ---------------------------------------------------------------------------

fn check_dims(sw: usize, sh: usize, dw: usize, dh: usize) -> Result<(), SizeError> {
    if sw != dw || sh != dh {
        Err(SizeError::PixelCountMismatch)
    } else {
        Ok(())
    }
}

// ===========================================================================
// Macros for trait impls
// ===========================================================================

/// Implement ConvertImage (copy) for a (Src, Dst) pair using a bytes:: function.
macro_rules! impl_convert_image {
    ($src:ty, $dst:ty, $bytes_fn:path) => {
        impl ConvertImage<$dst> for $src {
            fn convert_image(
                src: ImgRef<'_, Self>,
                mut dst: ImgRefMut<'_, $dst>,
            ) -> Result<(), SizeError> {
                check_dims(src.width(), src.height(), dst.width(), dst.height())?;
                for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
                    let s: &[u8] = bytemuck::cast_slice(src_row);
                    let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
                    $bytes_fn(s, d)?;
                }
                Ok(())
            }
        }
    };
}

/// Implement ConvertImageInplace for a (Src, Dst) pair using an inplace bytes:: function.
macro_rules! impl_convert_image_inplace {
    ($src:ty, $dst:ty, $bytes_fn:path) => {
        impl ConvertImageInplace<$dst> for $src {
            fn convert_image_inplace(mut img: ImgVec<Self>) -> ImgVec<$dst> {
                let w = img.width();
                let h = img.height();
                let stride = img.stride();
                for row in img.rows_mut() {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(row);
                    $bytes_fn(bytes).expect("row is always valid");
                }
                let buf: Vec<$dst> = bytemuck::allocation::cast_vec(img.into_buf());
                ImgVec::new_stride(buf, w, h, stride)
            }
        }
    };
}

// ===========================================================================
// In-place 4bpp swaps
// ===========================================================================

impl_convert_image_inplace!(Rgba<u8>, Bgra<u8>, crate::bytes::rgba_to_bgra_inplace);
impl_convert_image_inplace!(Bgra<u8>, Rgba<u8>, crate::bytes::bgra_to_rgba_inplace);

// ===========================================================================
// Copy conversions
// ===========================================================================

// 4bpp ↔ 4bpp
impl_convert_image!(Rgba<u8>, Bgra<u8>, crate::bytes::rgba_to_bgra);
impl_convert_image!(Bgra<u8>, Rgba<u8>, crate::bytes::bgra_to_rgba);

// 3→4 bpp
impl_convert_image!(Rgb<u8>, Bgra<u8>, crate::bytes::rgb_to_bgra);
impl_convert_image!(Rgb<u8>, Rgba<u8>, crate::bytes::rgb_to_rgba);
impl_convert_image!(Bgr<u8>, Rgba<u8>, crate::bytes::bgr_to_rgba);
impl_convert_image!(Bgr<u8>, Bgra<u8>, crate::bytes::bgr_to_bgra);

// 1→4, 2→4
impl_convert_image!(Gray<u8>, Rgba<u8>, crate::bytes::gray_to_rgba);
impl_convert_image!(Gray<u8>, Bgra<u8>, crate::bytes::gray_to_bgra);
impl_convert_image!(GrayAlpha<u8>, Rgba<u8>, crate::bytes::gray_alpha_to_rgba);
impl_convert_image!(GrayAlpha<u8>, Bgra<u8>, crate::bytes::gray_alpha_to_bgra);

// 4→3 bpp
impl_convert_image!(Rgba<u8>, Rgb<u8>, crate::bytes::rgba_to_rgb);
impl_convert_image!(Bgra<u8>, Bgr<u8>, crate::bytes::bgra_to_bgr);
impl_convert_image!(Bgra<u8>, Rgb<u8>, crate::bytes::bgra_to_rgb);
impl_convert_image!(Rgba<u8>, Bgr<u8>, crate::bytes::rgba_to_bgr);

// ===========================================================================
// Deprecated named functions
// ===========================================================================

/// Use [`convert_imgref_inplace`](crate::convert_imgref_inplace) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref_inplace() instead")]
pub fn swap_rgba_to_bgra(img: ImgVec<Rgba<u8>>) -> ImgVec<Bgra<u8>> {
    crate::convert_imgref_inplace(img)
}

/// Use [`convert_imgref_inplace`](crate::convert_imgref_inplace) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref_inplace() instead")]
pub fn swap_bgra_to_rgba(img: ImgVec<Bgra<u8>>) -> ImgVec<Rgba<u8>> {
    crate::convert_imgref_inplace(img)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_rgba_to_bgra(
    src: ImgRef<'_, Rgba<u8>>,
    dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_bgra_to_rgba(
    src: ImgRef<'_, Bgra<u8>>,
    dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_rgb_to_bgra(
    src: ImgRef<'_, Rgb<u8>>,
    dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_rgb_to_rgba(
    src: ImgRef<'_, Rgb<u8>>,
    dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_bgr_to_rgba(
    src: ImgRef<'_, Bgr<u8>>,
    dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_bgr_to_bgra(
    src: ImgRef<'_, Bgr<u8>>,
    dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_gray_to_rgba(
    src: ImgRef<'_, Gray<u8>>,
    dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_gray_to_bgra(
    src: ImgRef<'_, Gray<u8>>,
    dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_gray_alpha_to_rgba(
    src: ImgRef<'_, GrayAlpha<u8>>,
    dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_gray_alpha_to_bgra(
    src: ImgRef<'_, GrayAlpha<u8>>,
    dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_rgba_to_rgb(
    src: ImgRef<'_, Rgba<u8>>,
    dst: ImgRefMut<'_, Rgb<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_bgra_to_rgb(
    src: ImgRef<'_, Bgra<u8>>,
    dst: ImgRefMut<'_, Rgb<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_bgra_to_bgr(
    src: ImgRef<'_, Bgra<u8>>,
    dst: ImgRefMut<'_, Bgr<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

/// Use [`convert_imgref`](crate::convert_imgref) instead.
#[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
pub fn convert_rgba_to_bgr(
    src: ImgRef<'_, Rgba<u8>>,
    dst: ImgRefMut<'_, Bgr<u8>>,
) -> Result<(), SizeError> {
    crate::convert_imgref(src, dst)
}

// ===========================================================================
// Experimental
// ===========================================================================

#[cfg(feature = "experimental")]
mod experimental_imgref {
    use super::*;

    // -----------------------------------------------------------------------
    // Trait impls
    // -----------------------------------------------------------------------

    impl_convert_image!(Gray<u8>, Rgb<u8>, crate::bytes::gray_to_rgb);
    impl_convert_image!(Gray<u8>, Bgr<u8>, crate::bytes::gray_to_bgr);
    impl_convert_image!(GrayAlpha<u8>, Rgb<u8>, crate::bytes::gray_alpha_to_rgb);
    impl_convert_image!(Gray<u8>, GrayAlpha<u8>, crate::bytes::gray_to_gray_alpha);
    impl_convert_image!(GrayAlpha<u8>, Gray<u8>, crate::bytes::gray_alpha_to_gray);

    // Identity gray
    impl_convert_image!(Rgb<u8>, Gray<u8>, crate::bytes::rgb_to_gray_identity);
    impl_convert_image!(Rgba<u8>, Gray<u8>, crate::bytes::rgba_to_gray_identity);

    // -----------------------------------------------------------------------
    // Deprecated named functions
    // -----------------------------------------------------------------------

    /// Use [`convert_imgref`](crate::convert_imgref) instead.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_gray_to_rgb(
        src: ImgRef<'_, Gray<u8>>,
        dst: ImgRefMut<'_, Rgb<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    /// Use [`convert_imgref`](crate::convert_imgref) instead.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_gray_to_bgr(
        src: ImgRef<'_, Gray<u8>>,
        dst: ImgRefMut<'_, Bgr<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    /// Use [`convert_imgref`](crate::convert_imgref) instead.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_gray_alpha_to_rgb(
        src: ImgRef<'_, GrayAlpha<u8>>,
        dst: ImgRefMut<'_, Rgb<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    /// Use [`convert_imgref`](crate::convert_imgref) instead.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_gray_to_gray_alpha(
        src: ImgRef<'_, Gray<u8>>,
        dst: ImgRefMut<'_, GrayAlpha<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    /// Use [`convert_imgref`](crate::convert_imgref) instead.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_gray_alpha_to_gray(
        src: ImgRef<'_, GrayAlpha<u8>>,
        dst: ImgRefMut<'_, Gray<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    /// Identity gray extraction from RGB image.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_rgb_to_gray_identity(
        src: ImgRef<'_, Rgb<u8>>,
        dst: ImgRefMut<'_, Gray<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    /// Identity gray extraction from RGBA image.
    #[deprecated(since = "0.3.0", note = "use garb::convert_imgref() instead")]
    pub fn convert_rgba_to_gray_identity(
        src: ImgRef<'_, Rgba<u8>>,
        dst: ImgRefMut<'_, Gray<u8>>,
    ) -> Result<(), SizeError> {
        crate::convert_imgref(src, dst)
    }

    // -----------------------------------------------------------------------
    // Weighted luma — keep as named functions (matrix is a parameter choice)
    // -----------------------------------------------------------------------

    macro_rules! luma_imgref {
    ($matrix:ident, $doc_matrix:expr) => {
        paste::paste! {
            #[doc = concat!("Convert `ImgRef<Rgb<u8>>` to `ImgRefMut<Gray<u8>>` using ", $doc_matrix, " luma.")]
            pub fn [<convert_rgb_to_gray_ $matrix>](
                src: ImgRef<'_, Rgb<u8>>,
                mut dst: ImgRefMut<'_, Gray<u8>>,
            ) -> Result<(), SizeError> {
                check_dims(src.width(), src.height(), dst.width(), dst.height())?;
                for (s, d) in src.rows().zip(dst.rows_mut()) {
                    crate::bytes::[<rgb_to_gray_ $matrix>](bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d))?;
                }
                Ok(())
            }
            #[doc = concat!("Convert `ImgRef<Bgr<u8>>` to `ImgRefMut<Gray<u8>>` using ", $doc_matrix, " luma.")]
            pub fn [<convert_bgr_to_gray_ $matrix>](
                src: ImgRef<'_, Bgr<u8>>,
                mut dst: ImgRefMut<'_, Gray<u8>>,
            ) -> Result<(), SizeError> {
                check_dims(src.width(), src.height(), dst.width(), dst.height())?;
                for (s, d) in src.rows().zip(dst.rows_mut()) {
                    crate::bytes::[<bgr_to_gray_ $matrix>](bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d))?;
                }
                Ok(())
            }
            #[doc = concat!("Convert `ImgRef<Rgba<u8>>` to `ImgRefMut<Gray<u8>>` using ", $doc_matrix, " luma.")]
            pub fn [<convert_rgba_to_gray_ $matrix>](
                src: ImgRef<'_, Rgba<u8>>,
                mut dst: ImgRefMut<'_, Gray<u8>>,
            ) -> Result<(), SizeError> {
                check_dims(src.width(), src.height(), dst.width(), dst.height())?;
                for (s, d) in src.rows().zip(dst.rows_mut()) {
                    crate::bytes::[<rgba_to_gray_ $matrix>](bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d))?;
                }
                Ok(())
            }
            #[doc = concat!("Convert `ImgRef<Bgra<u8>>` to `ImgRefMut<Gray<u8>>` using ", $doc_matrix, " luma.")]
            pub fn [<convert_bgra_to_gray_ $matrix>](
                src: ImgRef<'_, Bgra<u8>>,
                mut dst: ImgRefMut<'_, Gray<u8>>,
            ) -> Result<(), SizeError> {
                check_dims(src.width(), src.height(), dst.width(), dst.height())?;
                for (s, d) in src.rows().zip(dst.rows_mut()) {
                    crate::bytes::[<bgra_to_gray_ $matrix>](bytemuck::cast_slice(s), bytemuck::cast_slice_mut(d))?;
                }
                Ok(())
            }
        }
    };
}

    luma_imgref!(bt709, "BT.709");
    luma_imgref!(bt601, "BT.601");
    luma_imgref!(bt2020, "BT.2020");

    /// Default: RGB → Gray uses BT.709.
    #[inline(always)]
    pub fn convert_rgb_to_gray(
        src: ImgRef<'_, Rgb<u8>>,
        dst: ImgRefMut<'_, Gray<u8>>,
    ) -> Result<(), SizeError> {
        convert_rgb_to_gray_bt709(src, dst)
    }

    /// Default: RGBA → Gray uses BT.709.
    #[inline(always)]
    pub fn convert_rgba_to_gray(
        src: ImgRef<'_, Rgba<u8>>,
        dst: ImgRefMut<'_, Gray<u8>>,
    ) -> Result<(), SizeError> {
        convert_rgba_to_gray_bt709(src, dst)
    }

    // -----------------------------------------------------------------------
    // f32 alpha premultiplication (not a type conversion)
    // -----------------------------------------------------------------------

    /// Premultiply alpha for an `Rgba<f32>` image in-place.
    pub fn premultiply_rgba_f32(mut img: ImgRefMut<'_, Rgba<f32>>) {
        for row in img.rows_mut() {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(row);
            crate::bytes::premultiply_alpha_f32(bytes).expect("row is always valid");
        }
    }

    /// Unpremultiply alpha for an `Rgba<f32>` image in-place.
    pub fn unpremultiply_rgba_f32(mut img: ImgRefMut<'_, Rgba<f32>>) {
        for row in img.rows_mut() {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(row);
            crate::bytes::unpremultiply_alpha_f32(bytes).expect("row is always valid");
        }
    }
} // mod experimental_imgref
#[cfg(feature = "experimental")]
pub use experimental_imgref::*;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use imgref::{ImgRefMut, ImgVec};
    use rgb::{Bgra, Gray, Rgb, Rgba};

    #[test]
    fn test_convert_imgref_inplace() {
        let img = ImgVec::new(vec![Rgba::new(255u8, 128, 0, 200); 4], 2, 2);
        let bgra: ImgVec<Bgra<u8>> = crate::convert_imgref_inplace(img);
        assert_eq!(bgra.width(), 2);
        assert_eq!(bgra.height(), 2);
        assert_eq!(
            bgra.buf()[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 200
            }
        );
    }

    #[test]
    fn test_convert_imgref_copy() {
        let src = ImgVec::new(vec![Rgb::new(255u8, 128, 0); 4], 2, 2);
        let mut dst_buf = vec![Bgra::default(); 4];
        let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
        crate::convert_imgref(src.as_ref(), dst).unwrap();
        assert_eq!(
            dst_buf[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 255
            }
        );
    }

    #[test]
    fn test_convert_imgref_gray() {
        let src = ImgVec::new(vec![Gray::new(100u8); 4], 2, 2);
        let mut dst_buf = vec![Rgba::default(); 4];
        let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
        crate::convert_imgref(src.as_ref(), dst).unwrap();
        assert_eq!(dst_buf[0], Rgba::new(100, 100, 100, 255));
    }

    #[test]
    fn test_convert_imgref_strip_alpha() {
        let src = ImgVec::new(vec![Rgba::new(255u8, 128, 0, 200); 4], 2, 2);
        let mut dst_buf = vec![Rgb::default(); 4];
        let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
        crate::convert_imgref(src.as_ref(), dst).unwrap();
        assert_eq!(dst_buf[0], Rgb::new(255, 128, 0));
    }

    #[test]
    fn test_strided_image() {
        let buf = vec![
            Rgba::new(1u8, 2, 3, 4),
            Rgba::new(5, 6, 7, 8),
            Rgba::new(9, 10, 11, 12),
            Rgba::default(),
            Rgba::new(13, 14, 15, 16),
            Rgba::new(17, 18, 19, 20),
            Rgba::new(21, 22, 23, 24),
            Rgba::default(),
        ];
        let src = ImgVec::new_stride(buf, 3, 2, 4);
        let mut dst_buf = vec![Bgra::default(); 6];
        let dst = ImgRefMut::new(&mut dst_buf, 3, 2);
        crate::convert_imgref(src.as_ref(), dst).unwrap();
        assert_eq!(
            dst_buf[0],
            Bgra {
                b: 3,
                g: 2,
                r: 1,
                a: 4
            }
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let src = ImgVec::new(vec![Rgba::new(1u8, 2, 3, 4); 4], 2, 2);
        let mut dst_buf = vec![Bgra::default(); 6];
        let dst = ImgRefMut::new(&mut dst_buf, 3, 2);
        assert_eq!(
            crate::convert_imgref(src.as_ref(), dst),
            Err(crate::SizeError::PixelCountMismatch)
        );
    }

    #[allow(deprecated)]
    #[test]
    fn test_deprecated_fns_still_work() {
        let img = ImgVec::new(vec![Rgba::new(255u8, 128, 0, 200); 4], 2, 2);
        let bgra = super::swap_rgba_to_bgra(img);
        assert_eq!(
            bgra.buf()[0],
            Bgra {
                b: 0,
                g: 128,
                r: 255,
                a: 200
            }
        );
    }
}
