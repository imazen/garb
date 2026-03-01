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
// Depth conversions — element-level, channel-agnostic
// ===========================================================================

/// Convert u8 elements to u16 elements. Maps [0,255] → [0,65535] via `v * 257`.
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

/// Convert u8 elements to f32 elements. Maps [0,255] → [0.0, 1.0] via `v / 255.0`.
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

/// Convert f32 elements to u8 elements. Clamped to [0,1], then `v * 255 + 0.5`.
///
/// `src` contains f32 values (must be a multiple of 4 bytes).
/// `dst` must have at least `src.len() / 4` bytes.
pub fn convert_f32_to_u8(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 1)?;
    incant!(convert_f32_to_u8_impl(src, dst), [v3, scalar]);
    Ok(())
}

/// Convert u16 elements to f32 elements. Maps [0,65535] → [0.0, 1.0] via `v / 65535.0`.
///
/// `src` contains u16 values (must be a multiple of 2 bytes).
/// `dst` must have at least `src.len() * 2` bytes (u16→f32 doubles byte count).
pub fn convert_u16_to_f32(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 2, dst.len(), 4)?;
    incant!(convert_u16_to_f32_impl(src, dst), [v3, scalar]);
    Ok(())
}

/// Convert f32 elements to u16 elements. Clamped to [0,1], then `v * 65535 + 0.5`.
///
/// `src` contains f32 values (must be a multiple of 4 bytes).
/// `dst` must have at least `src.len() / 2` bytes (f32→u16 halves byte count).
pub fn convert_f32_to_u16(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 2)?;
    incant!(convert_f32_to_u16_impl(src, dst), [v3, scalar]);
    Ok(())
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
