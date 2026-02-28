//! Whole-image pixel format conversions using [`imgref`] types.
//!
//! These functions handle strided image buffers and produce [`ImgVec`] results.
//! They use the SIMD-optimized core operations row-by-row.
//!
//! ```rust
//! use rgb::{Rgb, Bgra};
//! use imgref::ImgVec;
//! use garb::img;
//!
//! let rgb_img = ImgVec::new(vec![Rgb::new(255u8, 0, 128); 100], 10, 10);
//! let bgra_img: ImgVec<Bgra<u8>> = img::convert_rgb_to_bgra(rgb_img.as_ref());
//! ```

use alloc::vec;
use alloc::vec::Vec;

use imgref::{ImgRef, ImgVec};
use rgb::{Bgr, Bgra, Gray, GrayAlpha, Rgb, Rgba};

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
        crate::swap_br_inplace(bytes);
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
        crate::swap_br_inplace(bytes);
    }
    let buf: Vec<Rgba<u8>> = bytemuck::allocation::cast_vec(img.into_buf());
    ImgVec::new_stride(buf, w, h, stride)
}

// ---------------------------------------------------------------------------
// 3→4 bpp conversions
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Rgb<u8>>` to `ImgVec<Bgra<u8>>` (reverse + alpha=255).
pub fn convert_rgb_to_bgra(img: ImgRef<'_, Rgb<u8>>) -> ImgVec<Bgra<u8>> {
    let w = img.width();
    let h = img.height();
    let buf: Vec<Bgra<u8>> = vec![Bgra::default(); w * h];
    let mut dst = ImgVec::new(buf, w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgb_to_bgra(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Rgb<u8>>` to `ImgVec<Rgba<u8>>` (keep order + alpha=255).
pub fn convert_rgb_to_rgba(img: ImgRef<'_, Rgb<u8>>) -> ImgVec<Rgba<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgba::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgb_to_rgba(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Bgr<u8>>` to `ImgVec<Rgba<u8>>` (reverse + alpha=255).
pub fn convert_bgr_to_rgba(img: ImgRef<'_, Bgr<u8>>) -> ImgVec<Rgba<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgba::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgb_to_bgra(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Bgr<u8>>` to `ImgVec<Bgra<u8>>` (keep order + alpha=255).
pub fn convert_bgr_to_bgra(img: ImgRef<'_, Bgr<u8>>) -> ImgVec<Bgra<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Bgra::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::rgb_to_rgba(src_bytes, dst_bytes);
    }
    dst
}

// ---------------------------------------------------------------------------
// 1→4 and 2→4 bpp expansions
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Gray<u8>>` to `ImgVec<Rgba<u8>>`.
pub fn convert_gray_to_rgba(img: ImgRef<'_, Gray<u8>>) -> ImgVec<Rgba<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgba::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_to_4bpp(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Gray<u8>>` to `ImgVec<Bgra<u8>>`.
pub fn convert_gray_to_bgra(img: ImgRef<'_, Gray<u8>>) -> ImgVec<Bgra<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Bgra::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_to_4bpp(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<GrayAlpha<u8>>` to `ImgVec<Rgba<u8>>`.
pub fn convert_gray_alpha_to_rgba(img: ImgRef<'_, GrayAlpha<u8>>) -> ImgVec<Rgba<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgba::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_alpha_to_4bpp(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<GrayAlpha<u8>>` to `ImgVec<Bgra<u8>>`.
pub fn convert_gray_alpha_to_bgra(img: ImgRef<'_, GrayAlpha<u8>>) -> ImgVec<Bgra<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Bgra::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::gray_alpha_to_4bpp(src_bytes, dst_bytes);
    }
    dst
}

// ---------------------------------------------------------------------------
// 4→3 bpp strip alpha
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Rgba<u8>>` to `ImgVec<Rgb<u8>>`, dropping alpha.
pub fn convert_rgba_to_rgb(img: ImgRef<'_, Rgba<u8>>) -> ImgVec<Rgb<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgb::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::drop_alpha_4to3(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Bgra<u8>>` to `ImgVec<Rgb<u8>>`, dropping alpha and swapping B↔R.
pub fn convert_bgra_to_rgb(img: ImgRef<'_, Bgra<u8>>) -> ImgVec<Rgb<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgb::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::drop_alpha_swap_4to3(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Bgra<u8>>` to `ImgVec<Bgr<u8>>`, dropping alpha.
pub fn convert_bgra_to_bgr(img: ImgRef<'_, Bgra<u8>>) -> ImgVec<Bgr<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Bgr::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::drop_alpha_4to3(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Rgba<u8>>` to `ImgVec<Bgr<u8>>`, dropping alpha and swapping R↔B.
pub fn convert_rgba_to_bgr(img: ImgRef<'_, Rgba<u8>>) -> ImgVec<Bgr<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Bgr::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::drop_alpha_swap_4to3(src_bytes, dst_bytes);
    }
    dst
}

// ---------------------------------------------------------------------------
// Copy 4bpp swap (RGBA ↔ BGRA) producing new ImgVec
// ---------------------------------------------------------------------------

/// Convert `ImgRef<Rgba<u8>>` to `ImgVec<Bgra<u8>>` by copying with R↔B swap.
pub fn convert_rgba_to_bgra(img: ImgRef<'_, Rgba<u8>>) -> ImgVec<Bgra<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Bgra::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::copy_swap_br(src_bytes, dst_bytes);
    }
    dst
}

/// Convert `ImgRef<Bgra<u8>>` to `ImgVec<Rgba<u8>>` by copying with B↔R swap.
pub fn convert_bgra_to_rgba(img: ImgRef<'_, Bgra<u8>>) -> ImgVec<Rgba<u8>> {
    let w = img.width();
    let h = img.height();
    let mut dst = ImgVec::new(vec![Rgba::default(); w * h], w, h);
    for (src_row, dst_row) in img.rows().zip(dst.rows_mut()) {
        let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst_row);
        crate::copy_swap_br(src_bytes, dst_bytes);
    }
    dst
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_rgba_to_bgra_img() {
        let img = ImgVec::new(
            vec![Rgba::new(255u8, 128, 0, 200); 4],
            2,
            2,
        );
        let bgra = swap_rgba_to_bgra(img);
        assert_eq!(bgra.width(), 2);
        assert_eq!(bgra.height(), 2);
        assert_eq!(bgra.buf()[0], Bgra { b: 0, g: 128, r: 255, a: 200 });
    }

    #[test]
    fn test_convert_rgb_to_bgra_img() {
        let img = ImgVec::new(
            vec![Rgb::new(255u8, 128, 0); 4],
            2,
            2,
        );
        let bgra = convert_rgb_to_bgra(img.as_ref());
        assert_eq!(bgra.buf()[0], Bgra { b: 0, g: 128, r: 255, a: 255 });
    }

    #[test]
    fn test_convert_gray_to_rgba_img() {
        let img = ImgVec::new(vec![Gray::new(100u8); 4], 2, 2);
        let rgba = convert_gray_to_rgba(img.as_ref());
        assert_eq!(rgba.buf()[0], Rgba::new(100, 100, 100, 255));
    }

    #[test]
    fn test_convert_rgba_to_rgb_img() {
        let img = ImgVec::new(
            vec![Rgba::new(255u8, 128, 0, 200); 4],
            2,
            2,
        );
        let rgb = convert_rgba_to_rgb(img.as_ref());
        assert_eq!(rgb.buf()[0], Rgb::new(255, 128, 0));
    }

    #[test]
    fn test_strided_image() {
        // Stride > width: 3 pixels wide with stride 4
        let buf = vec![
            Rgba::new(1u8, 2, 3, 4), Rgba::new(5, 6, 7, 8), Rgba::new(9, 10, 11, 12), Rgba::default(),
            Rgba::new(13, 14, 15, 16), Rgba::new(17, 18, 19, 20), Rgba::new(21, 22, 23, 24), Rgba::default(),
        ];
        let img = ImgVec::new_stride(buf, 3, 2, 4);
        let bgra = convert_rgba_to_bgra(img.as_ref());
        assert_eq!(bgra.width(), 3);
        assert_eq!(bgra.height(), 2);
        // First pixel: R=1,G=2,B=3,A=4 → B=3,G=2,R=1,A=4
        assert_eq!(bgra.buf()[0], Bgra { b: 3, g: 2, r: 1, a: 4 });
    }
}
