// ---------------------------------------------------------------------------
// Row-level pixel swizzle operations with SIMD dispatch.
//
// Architecture: #[rite] row functions contain the SIMD loops.
// #[arcane] wrappers dispatch via incant! — contiguous (single call)
// and strided (loop over rows, single dispatch).
// ---------------------------------------------------------------------------

use crate::SizeError;
use archmage::incant;

mod scalar;
use scalar::*;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
use avx2::*;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
use neon::*;

#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
use wasm::*;

#[cfg(test)]
mod tests;

// ===========================================================================
// Validation helpers
// ===========================================================================

#[inline]
fn check_inplace(len: usize, bpp: usize) -> Result<(), SizeError> {
    if len == 0 || !len.is_multiple_of(bpp) {
        Err(SizeError::NotPixelAligned)
    } else {
        Ok(())
    }
}

#[inline]
fn check_copy(
    src_len: usize,
    src_bpp: usize,
    dst_len: usize,
    dst_bpp: usize,
) -> Result<(), SizeError> {
    if src_len == 0 || !src_len.is_multiple_of(src_bpp) {
        return Err(SizeError::NotPixelAligned);
    }
    if dst_len < (src_len / src_bpp) * dst_bpp {
        return Err(SizeError::PixelCountMismatch);
    }
    Ok(())
}

#[inline]
fn check_strided(
    len: usize,
    width: usize,
    height: usize,
    stride: usize,
    bpp: usize,
) -> Result<(), SizeError> {
    if width == 0 || height == 0 {
        return Err(SizeError::InvalidStride);
    }
    let row_bytes = width.checked_mul(bpp).ok_or(SizeError::InvalidStride)?;
    if row_bytes > stride {
        return Err(SizeError::InvalidStride);
    }
    if height > 0 {
        let total = (height - 1)
            .checked_mul(stride)
            .ok_or(SizeError::InvalidStride)?
            .checked_add(row_bytes)
            .ok_or(SizeError::InvalidStride)?;
        if len < total {
            return Err(SizeError::InvalidStride);
        }
    }
    Ok(())
}

// ===========================================================================
// Utility
// ===========================================================================

#[inline(always)]
fn swap_br_u32(v: u32) -> u32 {
    (v & 0xFF00_FF00) | (v.rotate_left(16) & 0x00FF_00FF)
}

// ===========================================================================
// Public API — contiguous
// ===========================================================================

/// Swap B↔R channels in-place for 4bpp pixels (RGBA↔BGRA).
pub fn rgba_to_bgra_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(swap_br_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Copy 4bpp pixels, swapping B↔R (RGBA→BGRA or BGRA→RGBA).
pub fn rgba_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 4)?;
    incant!(copy_swap_br_impl(src, dst), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Set the alpha channel (byte 3) to 255 for every 4bpp pixel.
///
/// Works for any alpha-last layout (RGBA, BGRA).
pub fn fill_alpha_rgba(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(fill_alpha_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Alias for [`fill_alpha_rgba`].
pub fn fill_alpha_bgra(buf: &mut [u8]) -> Result<(), SizeError> {
    fill_alpha_rgba(buf)
}

/// RGB (3 bytes/px) → BGRA (4 bytes/px). Reverses channel order, alpha=255.
pub fn rgb_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 4)?;
    incant!(rgb_to_bgra_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// RGB (3 bytes/px) → RGBA (4 bytes/px). Keeps channel order, alpha=255.
pub fn rgb_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 4)?;
    incant!(rgb_to_rgba_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// Gray (1 byte/px) → RGBA/BGRA (4 bytes/px). R=G=B=gray, alpha=255.
pub fn gray_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 1, dst.len(), 4)?;
    incant!(gray_to_4bpp_impl(src, dst), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// GrayAlpha (2 bytes/px) → RGBA/BGRA (4 bytes/px). R=G=B=gray.
pub fn gray_alpha_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 2, dst.len(), 4)?;
    incant!(
        gray_alpha_to_4bpp_impl(src, dst),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

// ===========================================================================
// Public API — strided
// ===========================================================================

/// Swap B↔R in-place for a strided 4bpp image (RGBA↔BGRA).
///
/// `stride` is the distance in bytes between the start of consecutive rows.
/// Must be ≥ `width × 4`. Padding bytes between rows are never read or written.
/// The buffer must be at least `(height - 1) * stride + width * 4` bytes.
pub fn rgba_to_bgra_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 4)?;
    incant!(
        swap_br_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Copy 4bpp pixels between strided buffers, swapping B↔R (RGBA→BGRA or vice versa).
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows in the source and destination buffers respectively.
/// Padding bytes between rows are never read or written.
pub fn rgba_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        copy_swap_br_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Set alpha (byte 3) to 255 for every 4bpp pixel in a strided buffer.
///
/// Works for any alpha-last layout (RGBA, BGRA).
/// `stride` is the distance in bytes between the start of consecutive rows.
/// Must be ≥ `width × 4`. Padding bytes between rows are never read or written.
pub fn fill_alpha_rgba_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 4)?;
    incant!(
        fill_alpha_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Alias for [`fill_alpha_rgba_strided`].
pub fn fill_alpha_bgra_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    fill_alpha_rgba_strided(buf, width, height, stride)
}

/// RGB (3 bytes/px) → BGRA (4 bytes/px) between strided buffers. Alpha=255.
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn rgb_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 3)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        rgb_to_bgra_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// RGB (3 bytes/px) → RGBA (4 bytes/px) between strided buffers. Alpha=255.
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn rgb_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 3)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        rgb_to_rgba_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// Gray (1 byte/px) → RGBA (4 bytes/px) between strided buffers. R=G=B=gray, alpha=255.
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn gray_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 1)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        gray_to_4bpp_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// GrayAlpha (2 bytes/px) → RGBA (4 bytes/px) between strided buffers. R=G=B=gray.
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn gray_alpha_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 2)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        gray_alpha_to_4bpp_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

// ===========================================================================
// SIMD-dispatched 3bpp and 4→3 operations
// ===========================================================================

/// Swap R↔B in-place for 3bpp pixels (RGB↔BGR).
pub fn rgb_to_bgr_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 3)?;
    incant!(swap_bgr_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Copy 3bpp pixels, swapping R↔B (RGB→BGR or BGR→RGB).
pub fn rgb_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 3)?;
    incant!(copy_swap_bgr_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// 4bpp → 3bpp by dropping byte 3 (alpha). Keeps byte order.
pub fn rgba_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 3)?;
    incant!(rgba_to_rgb_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// 4bpp → 3bpp, dropping alpha and reversing bytes 0↔2 (BGRA→RGB, RGBA→BGR).
pub fn bgra_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 3)?;
    incant!(bgra_to_rgb_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// Swap R↔B in-place for a strided 3bpp image (RGB↔BGR).
///
/// `stride` is the distance in bytes between the start of consecutive rows.
/// Must be ≥ `width × 3`. Padding bytes between rows are never read or written.
pub fn rgb_to_bgr_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 3)?;
    incant!(
        swap_bgr_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Copy 3bpp pixels between strided buffers, swapping R↔B (RGB→BGR or vice versa).
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn rgb_to_bgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 3)?;
    check_strided(dst.len(), width, height, dst_stride, 3)?;
    incant!(
        copy_swap_bgr_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// RGBA (4 bytes/px) → RGB (3 bytes/px) between strided buffers, dropping alpha.
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn rgba_to_rgb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 3)?;
    incant!(
        rgba_to_rgb_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// BGRA (4 bytes/px) → RGB (3 bytes/px) between strided buffers, dropping alpha + swapping.
///
/// `src_stride` / `dst_stride` are the distances in bytes between the start of
/// consecutive rows. Padding bytes between rows are never read or written.
pub fn bgra_to_rgb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 3)?;
    incant!(
        bgra_to_rgb_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

// ===========================================================================
// ARGB/XRGB operations — contiguous
// ===========================================================================

/// Rotate each pixel's bytes left by 1: \[A,R,G,B\]→\[R,G,B,A\] (ARGB→RGBA).
///
/// Also converts ABGR→BGRA.
pub fn argb_to_rgba_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(rotate_left_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Copy 4bpp pixels, rotating each left by 1 byte: \[A,R,G,B\]→\[R,G,B,A\] (ARGB→RGBA).
pub fn argb_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 4)?;
    incant!(copy_rotate_left_impl(src, dst), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Rotate each pixel's bytes right by 1: \[R,G,B,A\]→\[A,R,G,B\] (RGBA→ARGB).
///
/// Also converts BGRA→ABGR.
pub fn rgba_to_argb_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(rotate_right_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Copy 4bpp pixels, rotating each right by 1 byte: \[R,G,B,A\]→\[A,R,G,B\] (RGBA→ARGB).
pub fn rgba_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 4)?;
    incant!(
        copy_rotate_right_impl(src, dst),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Reverse each pixel's 4 bytes: \[A,R,G,B\]→\[B,G,R,A\] (ARGB→BGRA).
///
/// Also converts BGRA→ARGB, ABGR→RGBA, RGBA→ABGR.
pub fn argb_to_bgra_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(reverse_4bpp_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Copy 4bpp pixels, reversing each pixel's bytes: \[A,R,G,B\]→\[B,G,R,A\] (ARGB→BGRA).
pub fn argb_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 4)?;
    incant!(
        copy_reverse_4bpp_impl(src, dst),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Set the alpha channel (byte 0) to 255 for every 4bpp pixel.
///
/// Works for any alpha-first layout (ARGB, ABGR). Converts XRGB→ARGB.
pub fn fill_alpha_argb(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(fill_alpha_first_impl(buf), [v3, neon, wasm128, scalar]);
    Ok(())
}

/// Alias for [`fill_alpha_argb`].
pub fn fill_alpha_xrgb(buf: &mut [u8]) -> Result<(), SizeError> {
    fill_alpha_argb(buf)
}

/// RGB (3 bytes/px) → ARGB (4 bytes/px). Alpha=255, prepended.
pub fn rgb_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 4)?;
    incant!(rgb_to_argb_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// RGB (3 bytes/px) → ABGR (4 bytes/px). Reverses channel order, alpha=255 prepended.
pub fn rgb_to_abgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 4)?;
    incant!(rgb_to_abgr_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// ARGB (4 bytes/px) → RGB (3 bytes/px). Drops the leading alpha byte.
pub fn argb_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 3)?;
    incant!(argb_to_rgb_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// ARGB (4 bytes/px) → BGR (3 bytes/px). Drops alpha, reverses channel order.
pub fn argb_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 3)?;
    incant!(argb_to_bgr_impl(src, dst), [v3, wasm128, scalar]);
    Ok(())
}

/// Gray (1 byte/px) → ARGB (4 bytes/px). A=255, R=G=B=gray.
pub fn gray_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 1, dst.len(), 4)?;
    incant!(
        gray_to_4bpp_alpha_first_impl(src, dst),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// GrayAlpha (2 bytes/px) → ARGB (4 bytes/px). R=G=B=gray, alpha first.
pub fn gray_alpha_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 2, dst.len(), 4)?;
    incant!(
        gray_alpha_to_4bpp_alpha_first_impl(src, dst),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

// ===========================================================================
// ARGB/XRGB operations — strided
// ===========================================================================

/// Rotate each pixel's bytes left by 1 in a strided buffer (ARGB→RGBA).
pub fn argb_to_rgba_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 4)?;
    incant!(
        rotate_left_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Copy 4bpp pixels between strided buffers, rotating left by 1 (ARGB→RGBA).
pub fn argb_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        copy_rotate_left_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Rotate each pixel's bytes right by 1 in a strided buffer (RGBA→ARGB).
pub fn rgba_to_argb_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 4)?;
    incant!(
        rotate_right_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Copy 4bpp pixels between strided buffers, rotating right by 1 (RGBA→ARGB).
pub fn rgba_to_argb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        copy_rotate_right_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Reverse each pixel's 4 bytes in a strided buffer (ARGB↔BGRA).
pub fn argb_to_bgra_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 4)?;
    incant!(
        reverse_4bpp_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Copy 4bpp pixels between strided buffers, reversing bytes (ARGB→BGRA).
pub fn argb_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        copy_reverse_4bpp_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Set alpha (byte 0) to 255 for every 4bpp pixel in a strided buffer (XRGB→ARGB).
pub fn fill_alpha_argb_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), width, height, stride, 4)?;
    incant!(
        fill_alpha_first_strided(buf, width, height, stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// Alias for [`fill_alpha_argb_strided`].
pub fn fill_alpha_xrgb_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    fill_alpha_argb_strided(buf, width, height, stride)
}

/// RGB → ARGB between strided buffers. Alpha=255, prepended.
pub fn rgb_to_argb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 3)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        rgb_to_argb_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// RGB → ABGR between strided buffers. Alpha=255, channels reversed.
pub fn rgb_to_abgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 3)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        rgb_to_abgr_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// ARGB → RGB between strided buffers. Drops the leading alpha.
pub fn argb_to_rgb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 3)?;
    incant!(
        argb_to_rgb_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// ARGB → BGR between strided buffers. Drops alpha, reverses channel order.
pub fn argb_to_bgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 4)?;
    check_strided(dst.len(), width, height, dst_stride, 3)?;
    incant!(
        argb_to_bgr_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, wasm128, scalar]
    );
    Ok(())
}

/// Gray → ARGB between strided buffers. A=255, R=G=B=gray.
pub fn gray_to_argb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 1)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        gray_to_4bpp_alpha_first_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

/// GrayAlpha → ARGB between strided buffers. Alpha first, R=G=B=gray.
pub fn gray_alpha_to_argb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), width, height, src_stride, 2)?;
    check_strided(dst.len(), width, height, dst_stride, 4)?;
    incant!(
        gray_alpha_to_4bpp_alpha_first_strided(src, dst, width, height, src_stride, dst_stride),
        [v3, neon, wasm128, scalar]
    );
    Ok(())
}

#[cfg(feature = "experimental")]
mod experimental_api {
    use super::*;
    use crate::SizeError;
    use archmage::incant;

    // ===========================================================================
    // Weighted luma conversions — RGB/RGBA → Gray
    // ===========================================================================

    macro_rules! luma_api {
    (
        $matrix:ident, $r:expr, $g:expr, $b:expr,
        $doc_matrix:expr
    ) => {
        paste::paste! {
            #[doc = concat!("RGB (3 bytes/px) → Gray (1 byte/px) using ", $doc_matrix, " luma weights [", stringify!($r), ", ", stringify!($g), ", ", stringify!($b), "].")]
            pub fn [<rgb_to_gray_ $matrix>](src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
                check_copy(src.len(), 3, dst.len(), 1)?;
                incant!([<rgb_to_gray_ $matrix _impl>](src, dst), [v3, scalar]);
                Ok(())
            }

            #[doc = concat!("BGR (3 bytes/px) → Gray (1 byte/px) using ", $doc_matrix, " luma weights.")]
            pub fn [<bgr_to_gray_ $matrix>](src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
                check_copy(src.len(), 3, dst.len(), 1)?;
                incant!([<bgr_to_gray_ $matrix _impl>](src, dst), [v3, scalar]);
                Ok(())
            }

            #[doc = concat!("RGBA (4 bytes/px) → Gray (1 byte/px) using ", $doc_matrix, " luma weights. Alpha ignored.")]
            pub fn [<rgba_to_gray_ $matrix>](src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
                check_copy(src.len(), 4, dst.len(), 1)?;
                incant!([<rgba_to_gray_ $matrix _impl>](src, dst), [v3, scalar]);
                Ok(())
            }

            #[doc = concat!("BGRA (4 bytes/px) → Gray (1 byte/px) using ", $doc_matrix, " luma weights. Alpha ignored.")]
            pub fn [<bgra_to_gray_ $matrix>](src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
                check_copy(src.len(), 4, dst.len(), 1)?;
                incant!([<bgra_to_gray_ $matrix _impl>](src, dst), [v3, scalar]);
                Ok(())
            }

            // Strided variants
            #[doc = concat!("RGB → Gray ", $doc_matrix, " between strided buffers.")]
            pub fn [<rgb_to_gray_ $matrix _strided>](
                src: &[u8], dst: &mut [u8], width: usize, height: usize, src_stride: usize, dst_stride: usize,
            ) -> Result<(), SizeError> {
                check_strided(src.len(), width, height, src_stride, 3)?;
                check_strided(dst.len(), width, height, dst_stride, 1)?;
                incant!([<rgb_to_gray_ $matrix _strided>](src, dst, width, height, src_stride, dst_stride), [v3, scalar]);
                Ok(())
            }

            #[doc = concat!("BGR → Gray ", $doc_matrix, " between strided buffers.")]
            pub fn [<bgr_to_gray_ $matrix _strided>](
                src: &[u8], dst: &mut [u8], width: usize, height: usize, src_stride: usize, dst_stride: usize,
            ) -> Result<(), SizeError> {
                check_strided(src.len(), width, height, src_stride, 3)?;
                check_strided(dst.len(), width, height, dst_stride, 1)?;
                incant!([<bgr_to_gray_ $matrix _strided>](src, dst, width, height, src_stride, dst_stride), [v3, scalar]);
                Ok(())
            }

            #[doc = concat!("RGBA → Gray ", $doc_matrix, " between strided buffers.")]
            pub fn [<rgba_to_gray_ $matrix _strided>](
                src: &[u8], dst: &mut [u8], width: usize, height: usize, src_stride: usize, dst_stride: usize,
            ) -> Result<(), SizeError> {
                check_strided(src.len(), width, height, src_stride, 4)?;
                check_strided(dst.len(), width, height, dst_stride, 1)?;
                incant!([<rgba_to_gray_ $matrix _strided>](src, dst, width, height, src_stride, dst_stride), [v3, scalar]);
                Ok(())
            }

            #[doc = concat!("BGRA → Gray ", $doc_matrix, " between strided buffers.")]
            pub fn [<bgra_to_gray_ $matrix _strided>](
                src: &[u8], dst: &mut [u8], width: usize, height: usize, src_stride: usize, dst_stride: usize,
            ) -> Result<(), SizeError> {
                check_strided(src.len(), width, height, src_stride, 4)?;
                check_strided(dst.len(), width, height, dst_stride, 1)?;
                incant!([<bgra_to_gray_ $matrix _strided>](src, dst, width, height, src_stride, dst_stride), [v3, scalar]);
                Ok(())
            }
        }
    };
}

    luma_api!(bt709, 54, 183, 19, "BT.709");
    luma_api!(bt601, 77, 150, 29, "BT.601");
    luma_api!(bt2020, 67, 174, 15, "BT.2020");

    /// Alias: `rgb_to_gray` defaults to BT.709.
    #[inline(always)]
    pub fn rgb_to_gray(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        rgb_to_gray_bt709(src, dst)
    }

    /// Alias: `bgr_to_gray` defaults to BT.709.
    #[inline(always)]
    pub fn bgr_to_gray(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        bgr_to_gray_bt709(src, dst)
    }

    /// Alias: `rgba_to_gray` defaults to BT.709.
    #[inline(always)]
    pub fn rgba_to_gray(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        rgba_to_gray_bt709(src, dst)
    }

    /// Alias: `bgra_to_gray` defaults to BT.709.
    #[inline(always)]
    pub fn bgra_to_gray(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        bgra_to_gray_bt709(src, dst)
    }

    // ===========================================================================
    // Depth conversions — element-level, channel-agnostic
    // ===========================================================================

    /// Convert u8 elements to u16 elements. Maps `[0,255]` → `[0,65535]` via `v * 257`.
    ///
    /// `src` contains u8 values. `dst` must have at least `src.len() * 2` bytes.
    pub fn convert_u8_to_u16(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        if src.is_empty() {
            return Err(SizeError::NotPixelAligned);
        }
        check_copy(src.len(), 1, dst.len(), 2)?;
        incant!(convert_u8_to_u16_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Convert u16 elements to u8 elements. Rounded: `(v * 255 + 32768) >> 16`.
    ///
    /// `src` contains u16 values (must be a multiple of 2 bytes).
    /// `dst` must have at least `src.len() / 2` bytes.
    pub fn convert_u16_to_u8(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 1)?;
        incant!(convert_u16_to_u8_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Convert u8 elements to f32 elements. Maps `[0,255]` → `[0.0, 1.0]` via `v / 255.0`.
    ///
    /// `src` contains u8 values. `dst` must have at least `src.len() * 4` bytes.
    pub fn convert_u8_to_f32(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        if src.is_empty() {
            return Err(SizeError::NotPixelAligned);
        }
        check_copy(src.len(), 1, dst.len(), 4)?;
        incant!(convert_u8_to_f32_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Convert f32 elements to u8 elements. Clamped to `[0,1]`, then `v * 255 + 0.5`.
    ///
    /// `src` contains f32 values (must be a multiple of 4 bytes).
    /// `dst` must have at least `src.len() / 4` bytes.
    pub fn convert_f32_to_u8(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 1)?;
        incant!(convert_f32_to_u8_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Convert u16 elements to f32 elements. Maps `[0,65535]` → `[0.0, 1.0]` via `v / 65535.0`.
    ///
    /// `src` contains u16 values (must be a multiple of 2 bytes).
    /// `dst` must have at least `src.len() * 2` bytes (u16→f32 doubles byte count).
    pub fn convert_u16_to_f32(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 4)?;
        incant!(convert_u16_to_f32_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Convert f32 elements to u16 elements. Clamped to `[0,1]`, then `v * 65535 + 0.5`.
    ///
    /// `src` contains f32 values (must be a multiple of 4 bytes).
    /// `dst` must have at least `src.len() / 2` bytes (f32→u16 halves byte count).
    pub fn convert_f32_to_u16(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 2)?;
        incant!(convert_f32_to_u16_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    // ===========================================================================
    // Gray layout conversions — no luma weights
    // ===========================================================================

    /// Gray (1 byte/px) → RGB/BGR (3 bytes/px). R=G=B=gray.
    pub fn gray_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        if src.is_empty() {
            return Err(SizeError::NotPixelAligned);
        }
        check_copy(src.len(), 1, dst.len(), 3)?;
        incant!(gray_to_rgb_impl(src, dst), [scalar]);
        Ok(())
    }

    /// GrayAlpha (2 bytes/px) → RGB/BGR (3 bytes/px). R=G=B=gray, alpha dropped.
    pub fn gray_alpha_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 3)?;
        incant!(gray_alpha_to_rgb_impl(src, dst), [scalar]);
        Ok(())
    }

    /// Gray (1 byte/px) → GrayAlpha (2 bytes/px). Alpha set to 255.
    pub fn gray_to_gray_alpha(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        if src.is_empty() {
            return Err(SizeError::NotPixelAligned);
        }
        check_copy(src.len(), 1, dst.len(), 2)?;
        incant!(gray_to_gray_alpha_impl(src, dst), [scalar]);
        Ok(())
    }

    /// GrayAlpha (2 bytes/px) → Gray (1 byte/px). Drops alpha.
    pub fn gray_alpha_to_gray(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 1)?;
        incant!(gray_alpha_to_gray_impl(src, dst), [scalar]);
        Ok(())
    }

    /// RGB (3 bytes/px) → Gray (1 byte/px). Identity extraction — takes the R channel.
    ///
    /// Use this for roundtripping gray data that was expanded to RGB (where R=G=B).
    /// For perceptual luminance conversion, use `rgb_to_gray_bt709` instead.
    pub fn rgb_to_gray_identity(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 3, dst.len(), 1)?;
        incant!(rgb_to_gray_identity_impl(src, dst), [scalar]);
        Ok(())
    }

    /// RGBA (4 bytes/px) → Gray (1 byte/px). Identity extraction — takes the R channel.
    ///
    /// Use this for roundtripping gray data that was expanded to RGBA (where R=G=B).
    /// For perceptual luminance conversion, use `rgba_to_gray_bt709` instead.
    pub fn rgba_to_gray_identity(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 1)?;
        incant!(rgba_to_gray_identity_impl(src, dst), [scalar]);
        Ok(())
    }

    /// Alias for [`gray_to_rgb`] — R=G=B so output is identical.
    #[inline(always)]
    pub fn gray_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        gray_to_rgb(src, dst)
    }

    /// Alias for [`gray_alpha_to_rgb`] — R=G=B so output is identical.
    #[inline(always)]
    pub fn gray_alpha_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        gray_alpha_to_rgb(src, dst)
    }

    /// Alias for [`rgb_to_gray_identity`] — R=G=B so any channel works.
    #[inline(always)]
    pub fn bgr_to_gray_identity(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        rgb_to_gray_identity(src, dst)
    }

    /// Alias for [`rgba_to_gray_identity`] — R=G=B so any channel works.
    #[inline(always)]
    pub fn bgra_to_gray_identity(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        rgba_to_gray_identity(src, dst)
    }

    // Strided gray layout conversions

    /// Gray (1 byte/px) → RGB (3 bytes/px) between strided buffers.
    pub fn gray_to_rgb_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 1)?;
        check_strided(dst.len(), width, height, dst_stride, 3)?;
        incant!(
            gray_to_rgb_strided(src, dst, width, height, src_stride, dst_stride),
            [scalar]
        );
        Ok(())
    }

    /// GrayAlpha (2 bytes/px) → RGB (3 bytes/px) between strided buffers.
    pub fn gray_alpha_to_rgb_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 3)?;
        incant!(
            gray_alpha_to_rgb_strided(src, dst, width, height, src_stride, dst_stride),
            [scalar]
        );
        Ok(())
    }

    /// Gray (1 byte/px) → GrayAlpha (2 bytes/px) between strided buffers.
    pub fn gray_to_gray_alpha_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 1)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        incant!(
            gray_to_gray_alpha_strided(src, dst, width, height, src_stride, dst_stride),
            [scalar]
        );
        Ok(())
    }

    /// GrayAlpha (2 bytes/px) → Gray (1 byte/px) between strided buffers.
    pub fn gray_alpha_to_gray_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 1)?;
        incant!(
            gray_alpha_to_gray_strided(src, dst, width, height, src_stride, dst_stride),
            [scalar]
        );
        Ok(())
    }

    /// RGB (3 bytes/px) → Gray (1 byte/px) identity extraction between strided buffers.
    pub fn rgb_to_gray_identity_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 3)?;
        check_strided(dst.len(), width, height, dst_stride, 1)?;
        incant!(
            rgb_to_gray_identity_strided(src, dst, width, height, src_stride, dst_stride),
            [scalar]
        );
        Ok(())
    }

    /// RGBA (4 bytes/px) → Gray (1 byte/px) identity extraction between strided buffers.
    pub fn rgba_to_gray_identity_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 1)?;
        incant!(
            rgba_to_gray_identity_strided(src, dst, width, height, src_stride, dst_stride),
            [scalar]
        );
        Ok(())
    }

    /// Alias for [`gray_to_rgb_strided`].
    #[inline(always)]
    pub fn gray_to_bgr_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        gray_to_rgb_strided(src, dst, width, height, src_stride, dst_stride)
    }

    /// Alias for [`gray_alpha_to_rgb_strided`].
    #[inline(always)]
    pub fn gray_alpha_to_bgr_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        gray_alpha_to_rgb_strided(src, dst, width, height, src_stride, dst_stride)
    }

    /// Alias for [`rgb_to_gray_identity_strided`].
    #[inline(always)]
    pub fn bgr_to_gray_identity_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        rgb_to_gray_identity_strided(src, dst, width, height, src_stride, dst_stride)
    }

    /// Alias for [`rgba_to_gray_identity_strided`].
    #[inline(always)]
    pub fn bgra_to_gray_identity_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        rgba_to_gray_identity_strided(src, dst, width, height, src_stride, dst_stride)
    }

    // Strided depth conversions

    /// Convert u8→u16 between strided buffers. `width_elements` is element count per row.
    pub fn convert_u8_to_u16_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 1)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        incant!(
            convert_u8_to_u16_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Convert u16→u8 between strided buffers. `width` is element count per row.
    pub fn convert_u16_to_u8_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 1)?;
        incant!(
            convert_u16_to_u8_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Convert u8→f32 between strided buffers. `width` is element count per row.
    pub fn convert_u8_to_f32_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 1)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        incant!(
            convert_u8_to_f32_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Convert f32→u8 between strided buffers. `width` is element count per row.
    pub fn convert_f32_to_u8_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 1)?;
        incant!(
            convert_f32_to_u8_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Convert u16→f32 between strided buffers. `width` is element count per row.
    pub fn convert_u16_to_f32_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        incant!(
            convert_u16_to_f32_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Convert f32→u16 between strided buffers. `width` is element count per row.
    pub fn convert_f32_to_u16_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        incant!(
            convert_f32_to_u16_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    // ===========================================================================
    // f32 alpha premultiplication
    // ===========================================================================

    /// Premultiply alpha for f32 RGBA pixels in-place: `C' = C * A`, alpha preserved.
    ///
    /// Each pixel is 4 × f32 (16 bytes). The buffer must be a multiple of 16 bytes.
    pub fn premultiply_alpha_f32(buf: &mut [u8]) -> Result<(), SizeError> {
        check_inplace(buf.len(), 16)?;
        incant!(premul_f32_impl(buf), [v3, scalar]);
        Ok(())
    }

    /// Premultiply alpha for f32 RGBA pixels, copying from `src` to `dst`.
    ///
    /// Each pixel is 4 × f32 (16 bytes). Both buffers must be a multiple of 16 bytes.
    pub fn premultiply_alpha_f32_copy(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 16, dst.len(), 16)?;
        incant!(premul_f32_copy_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Unpremultiply alpha for f32 RGBA pixels in-place: `C' = C / A`.
    ///
    /// Where alpha is zero, all channels are set to zero.
    /// Each pixel is 4 × f32 (16 bytes). The buffer must be a multiple of 16 bytes.
    pub fn unpremultiply_alpha_f32(buf: &mut [u8]) -> Result<(), SizeError> {
        check_inplace(buf.len(), 16)?;
        incant!(unpremul_f32_impl(buf), [v3, scalar]);
        Ok(())
    }

    /// Unpremultiply alpha for f32 RGBA pixels, copying from `src` to `dst`.
    ///
    /// Where alpha is zero, all channels are set to zero.
    /// Each pixel is 4 × f32 (16 bytes). Both buffers must be a multiple of 16 bytes.
    pub fn unpremultiply_alpha_f32_copy(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 16, dst.len(), 16)?;
        incant!(unpremul_f32_copy_impl(src, dst), [v3, scalar]);
        Ok(())
    }

    /// Alias for [`premultiply_alpha_f32`].
    #[inline(always)]
    pub fn premultiply_alpha_rgba_f32(buf: &mut [u8]) -> Result<(), SizeError> {
        premultiply_alpha_f32(buf)
    }

    /// Alias for [`premultiply_alpha_f32_copy`].
    #[inline(always)]
    pub fn premultiply_alpha_rgba_f32_copy(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        premultiply_alpha_f32_copy(src, dst)
    }

    /// Alias for [`unpremultiply_alpha_f32`].
    #[inline(always)]
    pub fn unpremultiply_alpha_rgba_f32(buf: &mut [u8]) -> Result<(), SizeError> {
        unpremultiply_alpha_f32(buf)
    }

    /// Alias for [`unpremultiply_alpha_f32_copy`].
    #[inline(always)]
    pub fn unpremultiply_alpha_rgba_f32_copy(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        unpremultiply_alpha_f32_copy(src, dst)
    }

    // Strided f32 premul

    /// Premultiply alpha for f32 RGBA pixels in a strided buffer.
    ///
    /// `width` is the number of pixels per row. `stride` is bytes between row starts.
    /// Must be ≥ `width × 16`. Each pixel is 4 × f32 (16 bytes).
    pub fn premultiply_alpha_f32_strided(
        buf: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(buf.len(), width, height, stride, 16)?;
        incant!(premul_f32_strided(buf, width, height, stride), [v3, scalar]);
        Ok(())
    }

    /// Premultiply alpha for f32 RGBA pixels between strided buffers.
    pub fn premultiply_alpha_f32_copy_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 16)?;
        check_strided(dst.len(), width, height, dst_stride, 16)?;
        incant!(
            premul_f32_copy_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Unpremultiply alpha for f32 RGBA pixels in a strided buffer.
    ///
    /// Where alpha is zero, all channels are set to zero.
    /// `width` is the number of pixels per row. `stride` is bytes between row starts.
    /// Must be ≥ `width × 16`. Each pixel is 4 × f32 (16 bytes).
    pub fn unpremultiply_alpha_f32_strided(
        buf: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(buf.len(), width, height, stride, 16)?;
        incant!(
            unpremul_f32_strided(buf, width, height, stride),
            [v3, scalar]
        );
        Ok(())
    }

    /// Unpremultiply alpha for f32 RGBA pixels between strided buffers.
    ///
    /// Where alpha is zero, all channels are set to zero.
    pub fn unpremultiply_alpha_f32_copy_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 16)?;
        check_strided(dst.len(), width, height, dst_stride, 16)?;
        incant!(
            unpremul_f32_copy_strided(src, dst, width, height, src_stride, dst_stride),
            [v3, scalar]
        );
        Ok(())
    }
    // ===========================================================================
    // u8 alpha premultiplication (RGBA/BGRA, alpha-last at byte 3)
    // ===========================================================================

    /// Premultiply alpha for u8 RGBA pixels in-place: `C' = round(C × A / 255)`.
    ///
    /// Each pixel is 4 bytes `[R, G, B, A]` with alpha at byte 3. Uses the
    /// exact integer formula — no precision loss beyond the inherent
    /// quantization to u8. The buffer must be a multiple of 4 bytes.
    pub fn premultiply_alpha_rgba_u8(buf: &mut [u8]) -> Result<(), SizeError> {
        check_inplace(buf.len(), 4)?;
        premul_u8_impl(buf);
        Ok(())
    }

    /// Premultiply alpha for u8 BGRA pixels in-place: `C' = round(C × A / 255)`.
    ///
    /// Each pixel is 4 bytes `[B, G, R, A]` with alpha at byte 3. Same
    /// operation as [`premultiply_alpha_rgba_u8`] — alpha position is identical.
    #[inline(always)]
    pub fn premultiply_alpha_bgra_u8(buf: &mut [u8]) -> Result<(), SizeError> {
        premultiply_alpha_rgba_u8(buf)
    }

    /// Premultiply alpha for u8 RGBA pixels, copying from `src` to `dst`.
    ///
    /// Each pixel is 4 bytes `[R, G, B, A]` with alpha at byte 3.
    pub fn premultiply_alpha_rgba_u8_copy(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 4)?;
        premul_u8_copy_impl(src, dst);
        Ok(())
    }

    /// Premultiply alpha for u8 BGRA pixels, copying from `src` to `dst`.
    ///
    /// Same operation as [`premultiply_alpha_rgba_u8_copy`] — alpha position
    /// is identical.
    #[inline(always)]
    pub fn premultiply_alpha_bgra_u8_copy(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        premultiply_alpha_rgba_u8_copy(src, dst)
    }

    /// Premultiply alpha for u8 RGBA pixels in a strided buffer.
    ///
    /// `width` is pixels per row. `stride` is bytes between row starts.
    /// Must be ≥ `width × 4`.
    pub fn premultiply_alpha_rgba_u8_strided(
        buf: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(buf.len(), width, height, stride, 4)?;
        premul_u8_strided_impl(buf, width, height, stride);
        Ok(())
    }

    /// Alias for [`premultiply_alpha_rgba_u8_strided`].
    #[inline(always)]
    pub fn premultiply_alpha_bgra_u8_strided(
        buf: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(), SizeError> {
        premultiply_alpha_rgba_u8_strided(buf, width, height, stride)
    }

    /// Premultiply alpha for u8 RGBA pixels between strided buffers.
    pub fn premultiply_alpha_rgba_u8_copy_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        premul_u8_copy_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// Alias for [`premultiply_alpha_rgba_u8_copy_strided`].
    #[inline(always)]
    pub fn premultiply_alpha_bgra_u8_copy_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        premultiply_alpha_rgba_u8_copy_strided(src, dst, width, height, src_stride, dst_stride)
    }

    // ===========================================================================
    // Packed pixel format expansion (2bpp → 4bpp, little-endian)
    // ===========================================================================
    //
    // These functions expand packed 16-bit pixel formats into standard 8-bit
    // RGBA or BGRA. The source data is a byte slice containing **little-endian**
    // u16 values (low byte first). This is the native byte order on x86, ARM,
    // WASM, and most modern platforms.
    //
    // ## Bit layouts
    //
    // **RGB565** — 2 bytes per pixel, no alpha:
    // ```text
    //   u16 bit:  15 14 13 12 11 | 10  9  8  7  6  5 |  4  3  2  1  0
    //   channel:  R4 R3 R2 R1 R0 | G5 G4 G3 G2 G1 G0 | B4 B3 B2 B1 B0
    // ```
    // Matches OpenGL `GL_UNSIGNED_SHORT_5_6_5`, Vulkan `VK_FORMAT_R5G6B5_UNORM_PACK16`,
    // Android `Bitmap.Config.RGB_565`, Direct3D `DXGI_FORMAT_B5G6R5_UNORM` (note:
    // D3D labels this B5G6R5 but the bit layout is identical — R in the high bits).
    //
    // **RGBA4444** — 2 bytes per pixel, with alpha:
    // ```text
    //   u16 bit:  15 14 13 12 | 11 10  9  8 |  7  6  5  4 |  3  2  1  0
    //   channel:  R3 R2 R1 R0 | G3 G2 G1 G0 | B3 B2 B1 B0 | A3 A2 A1 A0
    // ```
    // Matches OpenGL `GL_UNSIGNED_SHORT_4_4_4_4`, Vulkan `VK_FORMAT_R4G4B4A4_UNORM_PACK16`.
    //
    // ## Big-endian source data
    //
    // If your source data stores the u16 values in big-endian byte order (high
    // byte first), swap each byte pair before calling:
    // ```rust,ignore
    // for pair in src.chunks_exact_mut(2) { pair.swap(0, 1); }
    // ```
    // Then call the conversion function as normal. A dedicated `_be` variant
    // may be added in a future release if there is demand.
    //
    // ## Channel expansion
    //
    // Sub-byte channels are expanded to 8 bits by replicating the MSBs into
    // the vacated LSBs. This maps the full source range exactly onto [0, 255]:
    // - 5-bit: `v << 3 | v >> 2` (0→0, 31→255)
    // - 6-bit: `v << 2 | v >> 4` (0→0, 63→255)
    // - 4-bit: `v << 4 | v`      (0→0, 15→255)

    /// RGB565 (little-endian u16, 2 bytes/px) → RGBA (4 bytes/px). Alpha set to 255.
    ///
    /// Source bit layout per u16: `R[15:11] G[10:5] B[4:0]`.
    /// See [module-level docs](self#packed-pixel-format-expansion-2bpp--4bpp-little-endian)
    /// for byte-order details.
    pub fn rgb565_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 4)?;
        rgb565_to_rgba_impl(src, dst);
        Ok(())
    }

    /// RGB565 (little-endian u16, 2 bytes/px) → BGRA (4 bytes/px). Alpha set to 255.
    ///
    /// Source bit layout per u16: `R[15:11] G[10:5] B[4:0]`.
    /// Output byte order: `[B, G, R, A]`.
    pub fn rgb565_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 4)?;
        rgb565_to_bgra_impl(src, dst);
        Ok(())
    }

    /// RGBA4444 (little-endian u16, 2 bytes/px) → RGBA (4 bytes/px).
    ///
    /// Source bit layout per u16: `R[15:12] G[11:8] B[7:4] A[3:0]`.
    pub fn rgba4444_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 4)?;
        rgba4444_to_rgba_impl(src, dst);
        Ok(())
    }

    /// RGBA4444 (little-endian u16, 2 bytes/px) → BGRA (4 bytes/px).
    ///
    /// Source bit layout per u16: `R[15:12] G[11:8] B[7:4] A[3:0]`.
    /// Output byte order: `[B, G, R, A]`.
    pub fn rgba4444_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 2, dst.len(), 4)?;
        rgba4444_to_bgra_impl(src, dst);
        Ok(())
    }

    // Strided packed format conversions

    /// RGB565 (LE, 2 bytes/px) → RGBA (4 bytes/px) between strided buffers.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn rgb565_to_rgba_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        rgb565_to_rgba_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// RGB565 (LE, 2 bytes/px) → BGRA (4 bytes/px) between strided buffers.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn rgb565_to_bgra_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        rgb565_to_bgra_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// RGBA4444 (LE, 2 bytes/px) → RGBA (4 bytes/px) between strided buffers.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn rgba4444_to_rgba_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        rgba4444_to_rgba_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// RGBA4444 (LE, 2 bytes/px) → BGRA (4 bytes/px) between strided buffers.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn rgba4444_to_bgra_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 2)?;
        check_strided(dst.len(), width, height, dst_stride, 4)?;
        rgba4444_to_bgra_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }
    // ===========================================================================
    // Packed pixel format compression (4bpp → 2bpp, little-endian)
    // ===========================================================================
    //
    // Compress standard 8-bit RGBA or BGRA pixels into packed 16-bit formats.
    // Output is **little-endian** u16 values, matching the expand functions.
    //
    // ## Lossy compression
    //
    // These conversions are lossy — 8-bit channels are rounded to fewer bits.
    // The rounding formula `(v * max_N + 128) >> 8` guarantees:
    // - **Perfect roundtrip** for values that originated from N bits:
    //   `expand(compress(expand(x))) == expand(x)` for all x in [0, max_N]
    // - **Nearest match** for arbitrary 8-bit values
    // - **Endpoints preserved**: 0→0, 255→max_N
    //
    // For RGB565, alpha is **dropped** (ignored). For RGBA4444, alpha is
    // compressed to 4 bits using the same rounding.
    //
    // ## Bit layouts
    //
    // Same as the expand functions:
    // - **RGB565**: `R[15:11] G[10:5] B[4:0]`
    // - **RGBA4444**: `R[15:12] G[11:8] B[7:4] A[3:0]`

    /// RGBA (4 bytes/px) → RGB565 (little-endian u16, 2 bytes/px). Alpha dropped.
    ///
    /// Lossy: 8-bit channels are rounded to 5/6/5 bits.
    /// Output bit layout per u16: `R[15:11] G[10:5] B[4:0]`.
    pub fn rgba_to_rgb565(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 2)?;
        rgba_to_rgb565_impl(src, dst);
        Ok(())
    }

    /// BGRA (4 bytes/px) → RGB565 (little-endian u16, 2 bytes/px). Alpha dropped.
    ///
    /// Lossy: 8-bit channels are rounded to 5/6/5 bits.
    /// Output bit layout per u16: `R[15:11] G[10:5] B[4:0]`.
    pub fn bgra_to_rgb565(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 2)?;
        bgra_to_rgb565_impl(src, dst);
        Ok(())
    }

    /// RGBA (4 bytes/px) → RGBA4444 (little-endian u16, 2 bytes/px).
    ///
    /// Lossy: 8-bit channels are rounded to 4 bits each.
    /// Output bit layout per u16: `R[15:12] G[11:8] B[7:4] A[3:0]`.
    pub fn rgba_to_rgba4444(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 2)?;
        rgba_to_rgba4444_impl(src, dst);
        Ok(())
    }

    /// BGRA (4 bytes/px) → RGBA4444 (little-endian u16, 2 bytes/px).
    ///
    /// Lossy: 8-bit channels are rounded to 4 bits each.
    /// Output bit layout per u16: `R[15:12] G[11:8] B[7:4] A[3:0]`.
    pub fn bgra_to_rgba4444(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
        check_copy(src.len(), 4, dst.len(), 2)?;
        bgra_to_rgba4444_impl(src, dst);
        Ok(())
    }

    // Strided compress conversions

    /// RGBA (4 bytes/px) → RGB565 (LE, 2 bytes/px) between strided buffers. Alpha dropped.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn rgba_to_rgb565_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        rgba_to_rgb565_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// BGRA (4 bytes/px) → RGB565 (LE, 2 bytes/px) between strided buffers. Alpha dropped.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn bgra_to_rgb565_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        bgra_to_rgb565_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// RGBA (4 bytes/px) → RGBA4444 (LE, 2 bytes/px) between strided buffers.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn rgba_to_rgba4444_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        rgba_to_rgba4444_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }

    /// BGRA (4 bytes/px) → RGBA4444 (LE, 2 bytes/px) between strided buffers.
    ///
    /// `width` is pixels per row. `src_stride`/`dst_stride` are bytes between row starts.
    pub fn bgra_to_rgba4444_strided(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), SizeError> {
        check_strided(src.len(), width, height, src_stride, 4)?;
        check_strided(dst.len(), width, height, dst_stride, 2)?;
        bgra_to_rgba4444_strided_impl(src, dst, width, height, src_stride, dst_stride);
        Ok(())
    }
} // mod experimental_api
#[cfg(feature = "experimental")]
pub use experimental_api::*;

// ===========================================================================
// Aliases — symmetric operations get both names
// ===========================================================================

/// Alias for [`rgba_to_bgra_inplace`] — same swap operation.
#[inline(always)]
pub fn bgra_to_rgba_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    rgba_to_bgra_inplace(buf)
}

/// Alias for [`rgba_to_bgra`] — same swap operation.
#[inline(always)]
pub fn bgra_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgba_to_bgra(src, dst)
}

/// Alias for [`rgb_to_bgr_inplace`].
#[inline(always)]
pub fn bgr_to_rgb_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    rgb_to_bgr_inplace(buf)
}

/// Alias for [`rgb_to_bgr`].
#[inline(always)]
pub fn bgr_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgb_to_bgr(src, dst)
}

/// BGR→RGBA = same byte shuffle as RGB→BGRA.
#[inline(always)]
pub fn bgr_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgb_to_bgra(src, dst)
}

/// BGR→BGRA = same byte shuffle as RGB→RGBA.
#[inline(always)]
pub fn bgr_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgb_to_rgba(src, dst)
}

/// Alias for [`gray_to_rgba`] — R=G=B so output is identical.
#[inline(always)]
pub fn gray_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    gray_to_rgba(src, dst)
}

/// Alias for [`gray_alpha_to_rgba`] — R=G=B so output is identical.
#[inline(always)]
pub fn gray_alpha_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    gray_alpha_to_rgba(src, dst)
}

/// BGRA→BGR = same as RGBA→RGB (drop alpha, keep order).
#[inline(always)]
pub fn bgra_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgba_to_rgb(src, dst)
}

/// RGBA→BGR = same as BGRA→RGB (drop alpha + swap).
#[inline(always)]
pub fn rgba_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bgra_to_rgb(src, dst)
}

// Strided aliases
/// Alias for [`rgba_to_bgra_inplace_strided`] — same swap operation.
#[inline(always)]
pub fn bgra_to_rgba_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    rgba_to_bgra_inplace_strided(buf, width, height, stride)
}
/// Alias for [`rgba_to_bgra_strided`] — same swap operation.
#[inline(always)]
pub fn bgra_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgba_to_bgra_strided(src, dst, width, height, src_stride, dst_stride)
}
/// BGR→RGBA = same byte shuffle as RGB→BGRA.
#[inline(always)]
pub fn bgr_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgb_to_bgra_strided(src, dst, width, height, src_stride, dst_stride)
}
/// BGR→BGRA = same byte shuffle as RGB→RGBA.
#[inline(always)]
pub fn bgr_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgb_to_rgba_strided(src, dst, width, height, src_stride, dst_stride)
}
/// Alias for [`gray_to_rgba_strided`] — R=G=B so output is identical.
#[inline(always)]
pub fn gray_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    gray_to_rgba_strided(src, dst, width, height, src_stride, dst_stride)
}
/// Alias for [`gray_alpha_to_rgba_strided`] — R=G=B so output is identical.
#[inline(always)]
pub fn gray_alpha_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    gray_alpha_to_rgba_strided(src, dst, width, height, src_stride, dst_stride)
}
/// BGRA→BGR = same as RGBA→RGB (drop alpha, keep order).
#[inline(always)]
pub fn bgra_to_bgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgba_to_rgb_strided(src, dst, width, height, src_stride, dst_stride)
}
/// RGBA→BGR = same as BGRA→RGB (drop alpha + swap).
#[inline(always)]
pub fn rgba_to_bgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    bgra_to_rgb_strided(src, dst, width, height, src_stride, dst_stride)
}
/// Alias for [`rgb_to_bgr_inplace_strided`].
#[inline(always)]
pub fn bgr_to_rgb_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    rgb_to_bgr_inplace_strided(buf, width, height, stride)
}
/// Alias for [`rgb_to_bgr_strided`].
#[inline(always)]
pub fn bgr_to_rgb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgb_to_bgr_strided(src, dst, width, height, src_stride, dst_stride)
}

// ARGB aliases — contiguous

/// Alias for [`argb_to_rgba_inplace`] — ABGR→BGRA uses the same rotate.
#[inline(always)]
pub fn abgr_to_bgra_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    argb_to_rgba_inplace(buf)
}

/// Alias for [`argb_to_rgba`] — ABGR→BGRA uses the same rotate.
#[inline(always)]
pub fn abgr_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    argb_to_rgba(src, dst)
}

/// Alias for [`rgba_to_argb_inplace`] — BGRA→ABGR uses the same rotate.
#[inline(always)]
pub fn bgra_to_abgr_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    rgba_to_argb_inplace(buf)
}

/// Alias for [`rgba_to_argb`] — BGRA→ABGR uses the same rotate.
#[inline(always)]
pub fn bgra_to_abgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgba_to_argb(src, dst)
}

/// Alias for [`argb_to_bgra_inplace`] — reverse is symmetric.
#[inline(always)]
pub fn bgra_to_argb_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgra_inplace(buf)
}

/// Alias for [`argb_to_bgra`] — reverse is symmetric.
#[inline(always)]
pub fn bgra_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgra(src, dst)
}

/// Alias for [`argb_to_bgra_inplace`] — ABGR↔RGBA is the same reverse.
#[inline(always)]
pub fn abgr_to_rgba_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgra_inplace(buf)
}

/// Alias for [`argb_to_bgra`] — ABGR↔RGBA is the same reverse.
#[inline(always)]
pub fn abgr_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgra(src, dst)
}

/// Alias for [`argb_to_bgra_inplace`] — RGBA↔ABGR is the same reverse.
#[inline(always)]
pub fn rgba_to_abgr_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgra_inplace(buf)
}

/// Alias for [`argb_to_bgra`] — RGBA↔ABGR is the same reverse.
#[inline(always)]
pub fn rgba_to_abgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgra(src, dst)
}

/// Alias for [`fill_alpha_argb`] — ABGR has alpha in the same position.
#[inline(always)]
pub fn fill_alpha_abgr(buf: &mut [u8]) -> Result<(), SizeError> {
    fill_alpha_argb(buf)
}

/// Alias for [`fill_alpha_argb`] — XBGR has alpha in the same position.
#[inline(always)]
pub fn fill_alpha_xbgr(buf: &mut [u8]) -> Result<(), SizeError> {
    fill_alpha_argb(buf)
}

/// BGR→ARGB = same byte shuffle as RGB→ABGR.
#[inline(always)]
pub fn bgr_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgb_to_abgr(src, dst)
}

/// BGR→ABGR = same byte shuffle as RGB→ARGB.
#[inline(always)]
pub fn bgr_to_abgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    rgb_to_argb(src, dst)
}

/// Alias for [`argb_to_rgb`] — ABGR→BGR drops the same leading byte.
#[inline(always)]
pub fn abgr_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    argb_to_rgb(src, dst)
}

/// Alias for [`argb_to_bgr`] — ABGR→RGB drops + reverses same as ARGB→BGR.
#[inline(always)]
pub fn abgr_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    argb_to_bgr(src, dst)
}

/// Alias for [`gray_to_argb`] — R=G=B so output is identical to ABGR.
#[inline(always)]
pub fn gray_to_abgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    gray_to_argb(src, dst)
}

/// Alias for [`gray_alpha_to_argb`] — R=G=B so output is identical to ABGR.
#[inline(always)]
pub fn gray_alpha_to_abgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    gray_alpha_to_argb(src, dst)
}

// ARGB aliases — strided

/// Alias for [`argb_to_rgba_inplace_strided`] — ABGR→BGRA uses the same rotate.
#[inline(always)]
pub fn abgr_to_bgra_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    argb_to_rgba_inplace_strided(buf, width, height, stride)
}

/// Alias for [`argb_to_rgba_strided`].
#[inline(always)]
pub fn abgr_to_bgra_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    argb_to_rgba_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`rgba_to_argb_inplace_strided`] — BGRA→ABGR uses the same rotate.
#[inline(always)]
pub fn bgra_to_abgr_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    rgba_to_argb_inplace_strided(buf, width, height, stride)
}

/// Alias for [`rgba_to_argb_strided`].
#[inline(always)]
pub fn bgra_to_abgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgba_to_argb_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`argb_to_bgra_inplace_strided`] — reverse is symmetric.
#[inline(always)]
pub fn bgra_to_argb_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgra_inplace_strided(buf, width, height, stride)
}

/// Alias for [`argb_to_bgra_strided`] — reverse is symmetric.
#[inline(always)]
pub fn bgra_to_argb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgra_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`argb_to_bgra_inplace_strided`] — ABGR↔RGBA is the same reverse.
#[inline(always)]
pub fn abgr_to_rgba_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgra_inplace_strided(buf, width, height, stride)
}

/// Alias for [`argb_to_bgra_strided`] — ABGR↔RGBA is the same reverse.
#[inline(always)]
pub fn abgr_to_rgba_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgra_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`argb_to_bgra_inplace_strided`] — RGBA↔ABGR is the same reverse.
#[inline(always)]
pub fn rgba_to_abgr_inplace_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgra_inplace_strided(buf, width, height, stride)
}

/// Alias for [`argb_to_bgra_strided`] — RGBA↔ABGR is the same reverse.
#[inline(always)]
pub fn rgba_to_abgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgra_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`fill_alpha_argb_strided`].
#[inline(always)]
pub fn fill_alpha_abgr_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    fill_alpha_argb_strided(buf, width, height, stride)
}

/// Alias for [`fill_alpha_argb_strided`].
#[inline(always)]
pub fn fill_alpha_xbgr_strided(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    fill_alpha_argb_strided(buf, width, height, stride)
}

/// BGR→ARGB = same byte shuffle as RGB→ABGR.
#[inline(always)]
pub fn bgr_to_argb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgb_to_abgr_strided(src, dst, width, height, src_stride, dst_stride)
}

/// BGR→ABGR = same byte shuffle as RGB→ARGB.
#[inline(always)]
pub fn bgr_to_abgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    rgb_to_argb_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`argb_to_rgb_strided`] — ABGR→BGR drops the same leading byte.
#[inline(always)]
pub fn abgr_to_bgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    argb_to_rgb_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`argb_to_bgr_strided`] — ABGR→RGB drops + reverses.
#[inline(always)]
pub fn abgr_to_rgb_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    argb_to_bgr_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`gray_to_argb_strided`] — R=G=B so identical to ABGR.
#[inline(always)]
pub fn gray_to_abgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    gray_to_argb_strided(src, dst, width, height, src_stride, dst_stride)
}

/// Alias for [`gray_alpha_to_argb_strided`] — R=G=B so identical to ABGR.
#[inline(always)]
pub fn gray_alpha_to_abgr_strided(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    gray_alpha_to_argb_strided(src, dst, width, height, src_stride, dst_stride)
}
