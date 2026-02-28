//! Whole-image pixel format conversions using [`imgref`] types.
//!
//! These functions handle strided image buffers using the SIMD-optimized
//! core operations. No allocation — caller owns all buffers.
//!
//! # In-place swaps (takes ownership, zero-copy reinterpret)
//!
//! ```rust
//! use rgb::{Rgba, Bgra};
//! use ::imgref::ImgVec;
//! use garb::imgref;
//!
//! let rgba_img = ImgVec::new(vec![Rgba::new(255u8, 0, 128, 200); 4], 2, 2);
//! let bgra_img: ImgVec<Bgra<u8>> = imgref::swap_rgba_to_bgra(rgba_img);
//! ```
//!
//! # Copy conversions (caller provides destination)
//!
//! ```rust
//! use rgb::{Rgb, Bgra};
//! use ::imgref::{ImgVec, ImgRefMut};
//! use garb::imgref;
//!
//! let src = ImgVec::new(vec![Rgb::new(255u8, 0, 128); 4], 2, 2);
//! let mut dst_buf = vec![Bgra::default(); 4];
//! let mut dst = ImgRefMut::new(&mut dst_buf, 2, 2);
//! imgref::convert_rgb_to_bgra(src.as_ref(), dst).unwrap();
//! ```

use alloc::vec::Vec;

use imgref::{ImgRef, ImgRefMut, ImgVec};
use rgb::{Bgr, Bgra, Gray, GrayAlpha, Rgb, Rgba};

use crate::SizeError;

// ---------------------------------------------------------------------------
// Dimension check
// ---------------------------------------------------------------------------

fn check_dims(sw: usize, sh: usize, dw: usize, dh: usize) -> Result<(), SizeError> {
    if sw != dw || sh != dh {
        Err(SizeError)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// In-place 4bpp swaps (RGBA ↔ BGRA) on ImgVec
// ---------------------------------------------------------------------------

/// Convert an `ImgVec<Rgba<u8>>` to `ImgVec<Bgra<u8>>` in-place.
///
/// Swaps R↔B channels and reinterprets the buffer. Zero-copy for the
/// buffer itself; only the pixel data is modified.
pub fn swap_rgba_to_bgra(mut img: ImgVec<Rgba<u8>>) -> ImgVec<Bgra<u8>> {
    let w = img.width();
    let h = img.height();
    let stride = img.stride();
    for row in img.rows_mut() {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(row);
        crate::rgba_to_bgra_inplace(bytes).expect("row is always valid");
    }
    let buf: Vec<Bgra<u8>> = bytemuck::allocation::cast_vec(img.into_buf());
    ImgVec::new_stride(buf, w, h, stride)
}

/// Convert an `ImgVec<Bgra<u8>>` to `ImgVec<Rgba<u8>>` in-place.
pub fn swap_bgra_to_rgba(mut img: ImgVec<Bgra<u8>>) -> ImgVec<Rgba<u8>> {
    let w = img.width();
    let h = img.height();
    let stride = img.stride();
    for row in img.rows_mut() {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(row);
        crate::bgra_to_rgba_inplace(bytes).expect("row is always valid");
    }
    let buf: Vec<Rgba<u8>> = bytemuck::allocation::cast_vec(img.into_buf());
    ImgVec::new_stride(buf, w, h, stride)
}

// ---------------------------------------------------------------------------
// Copy 4bpp swap (RGBA ↔ BGRA)
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Rgba<u8>>` to `ImgRefMut<Bgra<u8>>` by copying with R↔B swap.
pub fn convert_rgba_to_bgra(
    src: ImgRef<'_, Rgba<u8>>,
    mut dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgba_to_bgra(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Bgra<u8>>` to `ImgRefMut<Rgba<u8>>` by copying with B↔R swap.
pub fn convert_bgra_to_rgba(
    src: ImgRef<'_, Bgra<u8>>,
    mut dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::bgra_to_rgba(s, d)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 3→4 bpp conversions
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Rgb<u8>>` to `ImgRefMut<Bgra<u8>>` (reverse + alpha=255).
pub fn convert_rgb_to_bgra(
    src: ImgRef<'_, Rgb<u8>>,
    mut dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgb_to_bgra(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Rgb<u8>>` to `ImgRefMut<Rgba<u8>>` (keep order + alpha=255).
pub fn convert_rgb_to_rgba(
    src: ImgRef<'_, Rgb<u8>>,
    mut dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgb_to_rgba(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Bgr<u8>>` to `ImgRefMut<Rgba<u8>>` (reverse + alpha=255).
pub fn convert_bgr_to_rgba(
    src: ImgRef<'_, Bgr<u8>>,
    mut dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::bgr_to_rgba(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Bgr<u8>>` to `ImgRefMut<Bgra<u8>>` (keep order + alpha=255).
pub fn convert_bgr_to_bgra(
    src: ImgRef<'_, Bgr<u8>>,
    mut dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::bgr_to_bgra(s, d)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1→4 and 2→4 bpp expansions
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Gray<u8>>` to `ImgRefMut<Rgba<u8>>`.
pub fn convert_gray_to_rgba(
    src: ImgRef<'_, Gray<u8>>,
    mut dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_to_rgba(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Gray<u8>>` to `ImgRefMut<Bgra<u8>>`.
pub fn convert_gray_to_bgra(
    src: ImgRef<'_, Gray<u8>>,
    mut dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_to_bgra(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<GrayAlpha<u8>>` to `ImgRefMut<Rgba<u8>>`.
pub fn convert_gray_alpha_to_rgba(
    src: ImgRef<'_, GrayAlpha<u8>>,
    mut dst: ImgRefMut<'_, Rgba<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_alpha_to_rgba(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<GrayAlpha<u8>>` to `ImgRefMut<Bgra<u8>>`.
pub fn convert_gray_alpha_to_bgra(
    src: ImgRef<'_, GrayAlpha<u8>>,
    mut dst: ImgRefMut<'_, Bgra<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_alpha_to_bgra(s, d)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 4→3 bpp strip alpha
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Rgba<u8>>` to `ImgRefMut<Rgb<u8>>`, dropping alpha.
pub fn convert_rgba_to_rgb(
    src: ImgRef<'_, Rgba<u8>>,
    mut dst: ImgRefMut<'_, Rgb<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgba_to_rgb(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Bgra<u8>>` to `ImgRefMut<Rgb<u8>>`, dropping alpha and swapping B↔R.
pub fn convert_bgra_to_rgb(
    src: ImgRef<'_, Bgra<u8>>,
    mut dst: ImgRefMut<'_, Rgb<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::bgra_to_rgb(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Bgra<u8>>` to `ImgRefMut<Bgr<u8>>`, dropping alpha.
pub fn convert_bgra_to_bgr(
    src: ImgRef<'_, Bgra<u8>>,
    mut dst: ImgRefMut<'_, Bgr<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::bgra_to_bgr(s, d)?;
    }
    Ok(())
}

/// Convert `ImgRef<Rgba<u8>>` to `ImgRefMut<Bgr<u8>>`, dropping alpha and swapping R↔B.
pub fn convert_rgba_to_bgr(
    src: ImgRef<'_, Rgba<u8>>,
    mut dst: ImgRefMut<'_, Bgr<u8>>,
) -> Result<(), SizeError> {
    check_dims(src.width(), src.height(), dst.width(), dst.height())?;
    for (src_row, dst_row) in src.rows().zip(dst.rows_mut()) {
        let s: &[u8] = bytemuck::cast_slice(src_row);
        let d: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgba_to_bgr(s, d)?;
    }
    Ok(())
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
    fn test_swap_rgba_to_bgra_img() {
        let img = ImgVec::new(vec![Rgba::new(255u8, 128, 0, 200); 4], 2, 2);
        let bgra = swap_rgba_to_bgra(img);
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
    fn test_convert_rgb_to_bgra_img() {
        let src = ImgVec::new(vec![Rgb::new(255u8, 128, 0); 4], 2, 2);
        let mut dst_buf = vec![Bgra::default(); 4];
        let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
        convert_rgb_to_bgra(src.as_ref(), dst).unwrap();
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
    fn test_convert_gray_to_rgba_img() {
        let src = ImgVec::new(vec![Gray::new(100u8); 4], 2, 2);
        let mut dst_buf = vec![Rgba::default(); 4];
        let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
        convert_gray_to_rgba(src.as_ref(), dst).unwrap();
        assert_eq!(dst_buf[0], Rgba::new(100, 100, 100, 255));
    }

    #[test]
    fn test_convert_rgba_to_rgb_img() {
        let src = ImgVec::new(vec![Rgba::new(255u8, 128, 0, 200); 4], 2, 2);
        let mut dst_buf = vec![Rgb::default(); 4];
        let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
        convert_rgba_to_rgb(src.as_ref(), dst).unwrap();
        assert_eq!(dst_buf[0], Rgb::new(255, 128, 0));
    }

    #[test]
    fn test_strided_image() {
        // Stride > width: 3 pixels wide with stride 4
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
        convert_rgba_to_bgra(src.as_ref(), dst).unwrap();
        // First pixel: R=1,G=2,B=3,A=4 → B=3,G=2,R=1,A=4
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
        let dst = ImgRefMut::new(&mut dst_buf, 3, 2); // width mismatch
        assert_eq!(convert_rgba_to_bgra(src.as_ref(), dst), Err(SizeError));
    }
}
