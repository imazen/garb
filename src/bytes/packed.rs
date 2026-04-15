//! Packed pixel format conversions (RGB565, RGBA4444).

#![cfg_attr(target_arch = "x86", allow(unused_imports))]

use archmage::prelude::*;

use super::{check_copy, check_strided};
use crate::SizeError;

// ===========================================================================
// Scalar implementations — expansion (2bpp → 4bpp)
// ===========================================================================
//
// ## Bit layout (all formats are little-endian u16)
//
// **RGB565**: bits `[15:11]=R(5) [10:5]=G(6) [4:0]=B(5)`
//   Byte 0 = low byte, Byte 1 = high byte (standard little-endian).
//   This matches OpenGL `GL_UNSIGNED_SHORT_5_6_5`, Vulkan `R5G6B5_UNORM`,
//   and Android `ENCODING_RGB_565`.
//
// **RGBA4444**: bits `[15:12]=R(4) [11:8]=G(4) [7:4]=B(4) [3:0]=A(4)`
//   Byte 0 = low byte, Byte 1 = high byte (standard little-endian).
//   This matches OpenGL `GL_UNSIGNED_SHORT_4_4_4_4`.
//
// ## Big-endian data
//
// These functions expect **little-endian** u16 values in the byte slice.
// If your data is big-endian (e.g. from a network protocol or a big-endian
// file format), swap each pair of bytes before calling these functions.
// A simple pre-pass: `for pair in src.chunks_exact_mut(2) { pair.swap(0, 1); }`
//
// ## Expansion rounding
//
// Channels are expanded to 8 bits by replicating the top bits into the
// vacated low bits, which maps the full source range [0, max] to [0, 255]
// exactly (e.g. 5-bit 31 → `31<<3 | 31>>2` = 255, 6-bit 63 → `63<<2 | 63>>4` = 255).

// Packed format contiguous dispatchers (autoversion generates per-tier variants)
#[autoversion(v3, neon, wasm128)]
fn rgb565_to_rgba_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
        let v = u16::from_le_bytes([s[0], s[1]]);
        let r5 = (v >> 11) & 0x1F;
        let g6 = (v >> 5) & 0x3F;
        let b5 = v & 0x1F;
        d[0] = (r5 << 3 | r5 >> 2) as u8;
        d[1] = (g6 << 2 | g6 >> 4) as u8;
        d[2] = (b5 << 3 | b5 >> 2) as u8;
        d[3] = 0xFF;
    }
}
#[autoversion(v3, neon, wasm128)]
fn rgb565_to_bgra_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
        let v = u16::from_le_bytes([s[0], s[1]]);
        let r5 = (v >> 11) & 0x1F;
        let g6 = (v >> 5) & 0x3F;
        let b5 = v & 0x1F;
        d[0] = (b5 << 3 | b5 >> 2) as u8;
        d[1] = (g6 << 2 | g6 >> 4) as u8;
        d[2] = (r5 << 3 | r5 >> 2) as u8;
        d[3] = 0xFF;
    }
}
#[autoversion(v3, neon, wasm128)]
fn rgba4444_to_rgba_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
        let v = u16::from_le_bytes([s[0], s[1]]);
        let r4 = (v >> 12) & 0xF;
        let g4 = (v >> 8) & 0xF;
        let b4 = (v >> 4) & 0xF;
        let a4 = v & 0xF;
        d[0] = (r4 << 4 | r4) as u8;
        d[1] = (g4 << 4 | g4) as u8;
        d[2] = (b4 << 4 | b4) as u8;
        d[3] = (a4 << 4 | a4) as u8;
    }
}
#[autoversion(v3, neon, wasm128)]
fn rgba4444_to_bgra_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
        let v = u16::from_le_bytes([s[0], s[1]]);
        let r4 = (v >> 12) & 0xF;
        let g4 = (v >> 8) & 0xF;
        let b4 = (v >> 4) & 0xF;
        let a4 = v & 0xF;
        d[0] = (b4 << 4 | b4) as u8;
        d[1] = (g4 << 4 | g4) as u8;
        d[2] = (r4 << 4 | r4) as u8;
        d[3] = (a4 << 4 | a4) as u8;
    }
}

// Packed format strided dispatchers
#[autoversion(v3, neon, wasm128)]
fn rgb565_to_rgba_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 2]
            .chunks_exact(2)
            .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
        {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r5 = (v >> 11) & 0x1F;
            let g6 = (v >> 5) & 0x3F;
            let b5 = v & 0x1F;
            d[0] = (r5 << 3 | r5 >> 2) as u8;
            d[1] = (g6 << 2 | g6 >> 4) as u8;
            d[2] = (b5 << 3 | b5 >> 2) as u8;
            d[3] = 0xFF;
        }
    }
}
#[autoversion(v3, neon, wasm128)]
fn rgb565_to_bgra_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 2]
            .chunks_exact(2)
            .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
        {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r5 = (v >> 11) & 0x1F;
            let g6 = (v >> 5) & 0x3F;
            let b5 = v & 0x1F;
            d[0] = (b5 << 3 | b5 >> 2) as u8;
            d[1] = (g6 << 2 | g6 >> 4) as u8;
            d[2] = (r5 << 3 | r5 >> 2) as u8;
            d[3] = 0xFF;
        }
    }
}
#[autoversion(v3, neon, wasm128)]
fn rgba4444_to_rgba_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 2]
            .chunks_exact(2)
            .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
        {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r4 = (v >> 12) & 0xF;
            let g4 = (v >> 8) & 0xF;
            let b4 = (v >> 4) & 0xF;
            let a4 = v & 0xF;
            d[0] = (r4 << 4 | r4) as u8;
            d[1] = (g4 << 4 | g4) as u8;
            d[2] = (b4 << 4 | b4) as u8;
            d[3] = (a4 << 4 | a4) as u8;
        }
    }
}
#[autoversion(v3, neon, wasm128)]
fn rgba4444_to_bgra_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 2]
            .chunks_exact(2)
            .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
        {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r4 = (v >> 12) & 0xF;
            let g4 = (v >> 8) & 0xF;
            let b4 = (v >> 4) & 0xF;
            let a4 = v & 0xF;
            d[0] = (b4 << 4 | b4) as u8;
            d[1] = (g4 << 4 | g4) as u8;
            d[2] = (r4 << 4 | r4) as u8;
            d[3] = (a4 << 4 | a4) as u8;
        }
    }
}

// ===========================================================================
// Scalar implementations — compression (4bpp → 2bpp)
// ===========================================================================
//
// ## Rounding
//
// Channels are compressed from 8 bits using round-to-nearest:
//   compress_N(v) = (v * max_N + 128) >> 8
// where max_N is the maximum value for N bits (31, 63, or 15).
//
// This guarantees:
// - Perfect roundtrip: expand(compress(expand(x))) == expand(x) for all x
// - Nearest match for values not from N bits
// - Endpoints preserved: 0→0, 255→max
//
// Alpha is ignored (dropped) for RGB565, preserved for RGBA4444.
// Output is little-endian u16, matching the expand functions.

// Contiguous compress dispatchers

/// RGBA (4bpp) → RGB565 (LE u16, 2bpp). Alpha dropped.
#[autoversion(v3, neon, wasm128)]
fn rgba_to_rgb565_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
        let r5 = (s[0] as u16 * 31 + 128) >> 8;
        let g6 = (s[1] as u16 * 63 + 128) >> 8;
        let b5 = (s[2] as u16 * 31 + 128) >> 8;
        d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
    }
}

/// BGRA (4bpp) → RGB565 (LE u16, 2bpp). Alpha dropped.
#[autoversion(v3, neon, wasm128)]
fn bgra_to_rgb565_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
        let r5 = (s[2] as u16 * 31 + 128) >> 8;
        let g6 = (s[1] as u16 * 63 + 128) >> 8;
        let b5 = (s[0] as u16 * 31 + 128) >> 8;
        d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
    }
}

/// RGBA (4bpp) → RGBA4444 (LE u16, 2bpp).
#[autoversion(v3, neon, wasm128)]
fn rgba_to_rgba4444_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
        let r4 = (s[0] as u16 * 15 + 128) >> 8;
        let g4 = (s[1] as u16 * 15 + 128) >> 8;
        let b4 = (s[2] as u16 * 15 + 128) >> 8;
        let a4 = (s[3] as u16 * 15 + 128) >> 8;
        d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
    }
}

/// BGRA (4bpp) → RGBA4444 (LE u16, 2bpp).
#[autoversion(v3, neon, wasm128)]
fn bgra_to_rgba4444_impl(s: &[u8], d: &mut [u8]) {
    for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
        let r4 = (s[2] as u16 * 15 + 128) >> 8;
        let g4 = (s[1] as u16 * 15 + 128) >> 8;
        let b4 = (s[0] as u16 * 15 + 128) >> 8;
        let a4 = (s[3] as u16 * 15 + 128) >> 8;
        d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
    }
}

// Strided compress dispatchers

#[autoversion(v3, neon, wasm128)]
fn rgba_to_rgb565_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
        {
            let r5 = (s[0] as u16 * 31 + 128) >> 8;
            let g6 = (s[1] as u16 * 63 + 128) >> 8;
            let b5 = (s[2] as u16 * 31 + 128) >> 8;
            d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
        }
    }
}

#[autoversion(v3, neon, wasm128)]
fn bgra_to_rgb565_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
        {
            let r5 = (s[2] as u16 * 31 + 128) >> 8;
            let g6 = (s[1] as u16 * 63 + 128) >> 8;
            let b5 = (s[0] as u16 * 31 + 128) >> 8;
            d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
        }
    }
}

#[autoversion(v3, neon, wasm128)]
fn rgba_to_rgba4444_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
        {
            let r4 = (s[0] as u16 * 15 + 128) >> 8;
            let g4 = (s[1] as u16 * 15 + 128) >> 8;
            let b4 = (s[2] as u16 * 15 + 128) >> 8;
            let a4 = (s[3] as u16 * 15 + 128) >> 8;
            d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
        }
    }
}

#[autoversion(v3, neon, wasm128)]
fn bgra_to_rgba4444_strided_impl(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
        {
            let r4 = (s[2] as u16 * 15 + 128) >> 8;
            let g4 = (s[1] as u16 * 15 + 128) >> 8;
            let b4 = (s[0] as u16 * 15 + 128) >> 8;
            let a4 = (s[3] as u16 * 15 + 128) >> 8;
            d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
        }
    }
}

// ===========================================================================
// Public API — packed pixel format expansion (2bpp → 4bpp, little-endian)
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
/// Source bit layout per u16 (little-endian): `R[15:11] G[10:5] B[4:0]`.
/// Sub-byte channels are expanded to 8 bits by MSB replication.
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
// Public API — packed pixel format compression (4bpp → 2bpp, little-endian)
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
