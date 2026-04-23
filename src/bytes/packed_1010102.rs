//! Packed RGBA1010102 pixel format conversions.
//!
//! This module handles the bit-level packing only. Transfer functions (PQ,
//! HLG, sRGB) live in `linear-srgb` — `garb` is transfer-agnostic and just
//! moves bits around.
//!
//! ## Bit layout
//!
//! `RGBA1010102` is a 32-bit packed format with little-endian byte order:
//!
//! ```text
//!   u32 bit:  31 30 | 29 28 27 26 25 24 23 22 21 20 | 19 18 17 16 15 14 13 12 11 10 |  9  8  7  6  5  4  3  2  1  0
//!   channel:  A  A  | B  B  B  B  B  B  B  B  B  B  | G  G  G  G  G  G  G  G  G  G  | R  R  R  R  R  R  R  R  R  R
//! ```
//!
//! Equivalently:
//!
//! ```text
//!   packed = r | (g << 10) | (b << 20) | (a << 30)
//! ```
//!
//! This matches:
//! - DXGI / WIC `DXGI_FORMAT_R10G10B10A2_UNORM` (R in low bits, A in high bits)
//! - Vulkan `VK_FORMAT_A2B10G10R10_UNORM_PACK32` (Vulkan names the channels MSB-first
//!   but the in-memory layout is identical: bits `[0..10)` carry red)
//! - DRM `DRM_FORMAT_ABGR2101010` (DRM names channels LSB-first; "ABGR2101010" reads
//!   as A in MSB-most then B then G then R — bits `[0..10)` carry red)
//! - WGPU `Rgb10a2Unorm`
//! - The packing used by `ultrahdr-core::pack_rgba1010102`
//!
//! Other 1010102 variants (e.g. Apple `kCVPixelFormatType_ARGB2101010LEPacked`,
//! which puts A in the MSBs followed by R, G, B descending into the LSBs) are
//! NOT covered here. They can be added when there is a concrete consumer.
//!
//! ## Big-endian source data
//!
//! These functions expect the four packed bytes in **little-endian** order
//! (low byte first). If your data is big-endian, swap each group of four bytes
//! before calling these functions.
//!
//! ## Channel expansion
//!
//! Each channel is expanded into the low bits of a `u16` — 10-bit channels
//! occupy `[0..10)`, the high six bits are zero. Alpha (2 bits) is expanded to
//! 10 bits by **bit replication** so `0b00 → 0`, `0b01 → 341`, `0b10 → 682`,
//! `0b11 → 1023`. This matches how every graphics API expands 2-bit unorm
//! alpha and ensures perfect round-trip for the four legal alpha values.
//!
//! For the pack direction, 10-bit alpha is rounded back to 2 bits using
//! `(a_10 * 3 + 511) / 1023`, which is the standard rounded down-conversion
//! (endpoints preserved, nearest match for arbitrary inputs).
//!
//! ## Output layout (unpacked u16)
//!
//! The unpacked side is **interleaved RGBA u16**: four `u16` per pixel,
//! contiguous in memory, with each channel's value in `[0, 1023]`. This is the
//! natural input for downstream PQ/HLG transfer functions in `linear-srgb`.
//!
//! ## Channel-order swizzles (BGRA, ARGB, ...) are not handled here
//!
//! BGRA / ARGB / etc. swizzling is orthogonal to the bit-packing concern and
//! lives elsewhere in `garb` (see the byte-channel swizzle helpers). Callers
//! that need a different channel order should chain unpack-to-RGBA + a
//! separate swizzle pass.

#![cfg_attr(target_arch = "x86", allow(unused_imports))]

use archmage::prelude::*;

use crate::SizeError;

// ===========================================================================
// Channel expansion / compression helpers
// ===========================================================================

/// Expand a 2-bit unorm value to 10 bits via bit replication.
///
/// Maps `[0, 3]` to `[0, 1023]`:
/// - `0b00` → `0`
/// - `0b01` → `0b0101010101` = `341`
/// - `0b10` → `0b1010101010` = `682`
/// - `0b11` → `0b1111111111` = `1023`
#[inline(always)]
fn expand2_to_10(a: u32) -> u16 {
    // Replicate the 2-bit pattern five times into 10 bits.
    // a = 0bXY -> 0bXYXYXYXYXY
    //
    // 0x155 == 0b00_0101_0101 = a constant whose set bits are at positions
    // 0, 2, 4, 6, 8. Multiplying by `a` (where a in 0..=3) shifts that pattern
    // up by 0 or 1 and accumulates, which is exactly the bit replication:
    //   a=0 -> 0
    //   a=1 -> 0b0101010101 = 341
    //   a=2 -> 0b1010101010 = 682
    //   a=3 -> 0b1111111111 = 1023
    let a = a & 0x3;
    (a * 0x155) as u16 & 0x3FF
}

/// Compress a 10-bit unorm value to 2 bits with round-to-nearest.
///
/// Endpoints preserved (`0 → 0`, `1023 → 3`). Inverse of `expand2_to_10` for
/// the four legal expanded values.
#[inline(always)]
fn compress_10_to_2(a10: u16) -> u32 {
    let a = a10 as u32 & 0x3FF;
    (a * 3 + 511) / 1023
}

// ===========================================================================
// Scalar implementations — unpack (4bpp packed → 8bpp interleaved u16)
// ===========================================================================

#[inline(always)]
fn unpack_one_to_rgba16(src: &[u8; 4], dst: &mut [u16; 4]) {
    let v = u32::from_le_bytes(*src);
    dst[0] = (v & 0x3FF) as u16;
    dst[1] = ((v >> 10) & 0x3FF) as u16;
    dst[2] = ((v >> 20) & 0x3FF) as u16;
    dst[3] = expand2_to_10(v >> 30);
}

#[autoversion(v3, neon, wasm128)]
fn rgba1010102_to_rgba16_impl(src: &[u8], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let s4: &[u8; 4] = s.try_into().unwrap();
        let d4: &mut [u16; 4] = d.try_into().unwrap();
        unpack_one_to_rgba16(s4, d4);
    }
}

#[autoversion(v3, neon, wasm128)]
fn rgba1010102_to_rgba16_strided_impl(
    src: &[u8],
    dst: &mut [u16],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
        {
            let s4: &[u8; 4] = s.try_into().unwrap();
            let d4: &mut [u16; 4] = d.try_into().unwrap();
            unpack_one_to_rgba16(s4, d4);
        }
    }
}

// ===========================================================================
// Scalar implementations — pack (8bpp interleaved u16 → 4bpp packed)
// ===========================================================================

#[inline(always)]
fn pack_one_from_rgba16(src: &[u16; 4], dst: &mut [u8; 4]) {
    let r = (src[0] as u32) & 0x3FF;
    let g = (src[1] as u32) & 0x3FF;
    let b = (src[2] as u32) & 0x3FF;
    let a = compress_10_to_2(src[3]);
    let v = r | (g << 10) | (b << 20) | (a << 30);
    *dst = v.to_le_bytes();
}

#[autoversion(v3, neon, wasm128)]
fn rgba16_to_rgba1010102_impl(src: &[u16], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let s4: &[u16; 4] = s.try_into().unwrap();
        let d4: &mut [u8; 4] = d.try_into().unwrap();
        pack_one_from_rgba16(s4, d4);
    }
}

#[autoversion(v3, neon, wasm128)]
fn rgba16_to_rgba1010102_strided_impl(
    src: &[u16],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
        {
            let s4: &[u16; 4] = s.try_into().unwrap();
            let d4: &mut [u8; 4] = d.try_into().unwrap();
            pack_one_from_rgba16(s4, d4);
        }
    }
}

// ===========================================================================
// Validation helpers
// ===========================================================================

/// Validate a `&[u8]` packed source / `&mut [u16]` interleaved destination
/// pair. Each pixel is 4 packed bytes ↔ 4 u16 channels.
#[inline]
fn check_unpack(src_bytes: usize, dst_u16: usize) -> Result<(), SizeError> {
    if src_bytes == 0 || !src_bytes.is_multiple_of(4) {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = src_bytes / 4;
    if dst_u16 < pixels * 4 {
        return Err(SizeError::PixelCountMismatch);
    }
    Ok(())
}

/// Validate a `&[u16]` interleaved source / `&mut [u8]` packed destination pair.
#[inline]
fn check_pack(src_u16: usize, dst_bytes: usize) -> Result<(), SizeError> {
    if src_u16 == 0 || !src_u16.is_multiple_of(4) {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = src_u16 / 4;
    if dst_bytes < pixels * 4 {
        return Err(SizeError::PixelCountMismatch);
    }
    Ok(())
}

#[inline]
fn check_strided_bytes(
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
    let total = (height - 1)
        .checked_mul(stride)
        .ok_or(SizeError::InvalidStride)?
        .checked_add(row_bytes)
        .ok_or(SizeError::InvalidStride)?;
    if len < total {
        return Err(SizeError::InvalidStride);
    }
    Ok(())
}

// ===========================================================================
// Public API — contiguous unpack
// ===========================================================================

/// `RGBA1010102` (LE u32, 4 bytes/px) → interleaved `RGBA` u16 (4 channels/px).
///
/// Each output channel is a `u16` with the value in the low 10 bits (`[0, 1023]`).
/// Alpha is expanded from 2 bits to 10 bits by bit replication
/// (`0 → 0`, `1 → 341`, `2 → 682`, `3 → 1023`).
///
/// The destination must hold at least 4 `u16` per source pixel.
///
/// ```rust
/// use garb::bytes::rgba1010102_to_rgba16;
/// // Pure red max: r=1023, g=0, b=0, a=3 -> bytes 0xFF 0x03 0x00 0xC0
/// let src = [0xFF_u8, 0x03, 0x00, 0xC0];
/// let mut dst = [0u16; 4];
/// rgba1010102_to_rgba16(&src, &mut dst).unwrap();
/// assert_eq!(dst, [1023, 0, 0, 1023]);
/// ```
pub fn rgba1010102_to_rgba16(src: &[u8], dst: &mut [u16]) -> Result<(), SizeError> {
    check_unpack(src.len(), dst.len())?;
    rgba1010102_to_rgba16_impl(src, dst);
    Ok(())
}

// ===========================================================================
// Public API — contiguous pack
// ===========================================================================

/// Interleaved `RGBA` u16 (4 channels/px) → `RGBA1010102` (LE u32, 4 bytes/px).
///
/// Each input channel is read from the low 10 bits of a `u16`. Values above
/// `1023` are masked (`& 0x3FF`); callers should clamp first if they want
/// saturation rather than wraparound.
///
/// 10-bit alpha is rounded to 2 bits using `(a * 3 + 511) / 1023`. This is
/// the inverse of [`rgba1010102_to_rgba16`]'s bit-replication expansion for
/// the four legal alpha values (`0, 341, 682, 1023`).
///
/// ```rust
/// use garb::bytes::rgba16_to_rgba1010102;
/// let src: [u16; 4] = [1023, 0, 0, 1023]; // pure red, opaque
/// let mut dst = [0u8; 4];
/// rgba16_to_rgba1010102(&src, &mut dst).unwrap();
/// assert_eq!(dst, [0xFF, 0x03, 0x00, 0xC0]);
/// ```
pub fn rgba16_to_rgba1010102(src: &[u16], dst: &mut [u8]) -> Result<(), SizeError> {
    check_pack(src.len(), dst.len())?;
    rgba16_to_rgba1010102_impl(src, dst);
    Ok(())
}

// ===========================================================================
// Public API — strided unpack
// ===========================================================================

/// `RGBA1010102` → interleaved `RGBA` u16 between strided buffers.
///
/// `width` is pixels per row. `src_stride` is bytes between source row starts;
/// `dst_stride` is **u16 elements** between destination row starts (i.e. the
/// stride is in units of the slice's element type, matching garb's typed-API
/// convention).
pub fn rgba1010102_to_rgba16_strided(
    src: &[u8],
    dst: &mut [u16],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided_bytes(src.len(), width, height, src_stride, 4)?;
    check_strided_bytes(dst.len(), width, height, dst_stride, 4)?;
    rgba1010102_to_rgba16_strided_impl(src, dst, width, height, src_stride, dst_stride);
    Ok(())
}

// ===========================================================================
// Public API — strided pack
// ===========================================================================

/// Interleaved `RGBA` u16 → `RGBA1010102` between strided buffers.
///
/// `src_stride` is **u16 elements** between source row starts; `dst_stride` is
/// bytes between destination row starts.
pub fn rgba16_to_rgba1010102_strided(
    src: &[u16],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    check_strided_bytes(src.len(), width, height, src_stride, 4)?;
    check_strided_bytes(dst.len(), width, height, dst_stride, 4)?;
    rgba16_to_rgba1010102_strided_impl(src, dst, width, height, src_stride, dst_stride);
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;

    // ---- bit-level helpers ----

    #[test]
    fn expand2_to_10_table() {
        assert_eq!(expand2_to_10(0b00), 0);
        assert_eq!(expand2_to_10(0b01), 0b01_0101_0101);
        assert_eq!(expand2_to_10(0b10), 0b10_1010_1010);
        assert_eq!(expand2_to_10(0b11), 0b11_1111_1111);
        // Masking: only low 2 bits of input considered.
        assert_eq!(expand2_to_10(0xFFFF_FFFC), 0);
        assert_eq!(expand2_to_10(0xFFFF_FFFF), 1023);
    }

    #[test]
    fn compress_10_to_2_endpoints() {
        assert_eq!(compress_10_to_2(0), 0);
        assert_eq!(compress_10_to_2(1023), 3);
    }

    #[test]
    fn compress_inverts_expand_for_legal_values() {
        for a2 in 0..=3u32 {
            let expanded = expand2_to_10(a2);
            assert_eq!(
                compress_10_to_2(expanded),
                a2,
                "compress(expand({a2})) = compress({expanded}) != {a2}"
            );
        }
    }

    #[test]
    fn compress_rounds_to_nearest() {
        // 341 -> 1, 342 -> 1 (still closest to 341)
        assert_eq!(compress_10_to_2(341), 1);
        // Midpoint between 341 (a=1) and 682 (a=2) is 511.5 -> 512+ rounds to 2.
        assert_eq!(compress_10_to_2(511), 1);
        assert_eq!(compress_10_to_2(512), 2);
    }

    // ---- spot values against documented references ----

    #[test]
    fn unpack_known_values_rgba() {
        // All zero
        let mut dst = [0u16; 4];
        rgba1010102_to_rgba16(&[0u8; 4], &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 0, 0]);

        // Pure red max: packed = 0x000003FF -> bytes [FF, 03, 00, 00]
        rgba1010102_to_rgba16(&[0xFF, 0x03, 0x00, 0x00], &mut dst).unwrap();
        assert_eq!(dst, [1023, 0, 0, 0]);

        // Pure green max: packed = 0x000FFC00 -> bytes [00, FC, 0F, 00]
        rgba1010102_to_rgba16(&[0x00, 0xFC, 0x0F, 0x00], &mut dst).unwrap();
        assert_eq!(dst, [0, 1023, 0, 0]);

        // Pure blue max: packed = 0x3FF00000 -> bytes [00, 00, F0, 3F]
        rgba1010102_to_rgba16(&[0x00, 0x00, 0xF0, 0x3F], &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 1023, 0]);

        // Pure alpha max: packed = 0xC0000000 -> bytes [00, 00, 00, C0]
        rgba1010102_to_rgba16(&[0x00, 0x00, 0x00, 0xC0], &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 0, 1023]);

        // All max white opaque: packed = 0xFFFFFFFF
        rgba1010102_to_rgba16(&[0xFF, 0xFF, 0xFF, 0xFF], &mut dst).unwrap();
        assert_eq!(dst, [1023, 1023, 1023, 1023]);
    }

    #[test]
    fn pack_known_values_rgba() {
        let mut dst = [0u8; 4];
        // Pure red max -> 0x000003FF
        rgba16_to_rgba1010102(&[1023, 0, 0, 0], &mut dst).unwrap();
        assert_eq!(dst, [0xFF, 0x03, 0x00, 0x00]);
        // Pure green max -> 0x000FFC00
        rgba16_to_rgba1010102(&[0, 1023, 0, 0], &mut dst).unwrap();
        assert_eq!(dst, [0x00, 0xFC, 0x0F, 0x00]);
        // Pure blue max -> 0x3FF00000
        rgba16_to_rgba1010102(&[0, 0, 1023, 0], &mut dst).unwrap();
        assert_eq!(dst, [0x00, 0x00, 0xF0, 0x3F]);
        // Pure alpha max -> 0xC0000000
        rgba16_to_rgba1010102(&[0, 0, 0, 1023], &mut dst).unwrap();
        assert_eq!(dst, [0x00, 0x00, 0x00, 0xC0]);
        // White opaque -> 0xFFFFFFFF
        rgba16_to_rgba1010102(&[1023, 1023, 1023, 1023], &mut dst).unwrap();
        assert_eq!(dst, [0xFF, 0xFF, 0xFF, 0xFF]);
    }

    // ---- round-trip ----

    /// Tiny linear-congruential generator so round-trip tests are deterministic
    /// without a dev-dep on `rand`.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_F00D)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
    }

    #[test]
    fn forward_round_trip_random_packed() {
        // For every random packed u32, unpack then repack; result must equal
        // the original (the legal alpha values are exactly `expand2_to_10`'s
        // output, so the rounding is a no-op).
        let mut rng = Lcg::new(0xA110_1010_2222_3333);
        let n_pixels = 4096;
        let mut packed = vec![0u8; n_pixels * 4];
        for chunk in packed.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rng.next_u32().to_le_bytes());
        }

        let mut chans = vec![0u16; n_pixels * 4];
        rgba1010102_to_rgba16(&packed, &mut chans).unwrap();

        let mut repacked = vec![0u8; n_pixels * 4];
        rgba16_to_rgba1010102(&chans, &mut repacked).unwrap();
        assert_eq!(packed, repacked);
    }

    #[test]
    fn forward_round_trip_all_alpha_values() {
        // For all 4 alpha values × representative RGB samples, pack/unpack
        // is exact.
        let samples_per_chan: [u16; 8] = [0, 1, 256, 511, 512, 768, 1022, 1023];
        for &r in &samples_per_chan {
            for &g in &samples_per_chan {
                for &b in &samples_per_chan {
                    for a in 0..=3u32 {
                        let a10 = expand2_to_10(a);
                        let chans = [r, g, b, a10];
                        let mut packed = [0u8; 4];
                        rgba16_to_rgba1010102(&chans, &mut packed).unwrap();
                        let mut back = [0u16; 4];
                        rgba1010102_to_rgba16(&packed, &mut back).unwrap();
                        assert_eq!(
                            back, chans,
                            "round-trip failed for {chans:?} -> {packed:?} -> {back:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reverse_round_trip_all_packed_values_sample() {
        // Sample 65536 packed u32s; pack(unpack(x)) == x for every sampled value.
        let mut rng = Lcg::new(0xBEEF_F00D_F00D_BEEF);
        for _ in 0..65536 {
            let v = rng.next_u32();
            let bytes = v.to_le_bytes();
            let mut chans = [0u16; 4];
            rgba1010102_to_rgba16(&bytes, &mut chans).unwrap();
            let mut back = [0u8; 4];
            rgba16_to_rgba1010102(&chans, &mut back).unwrap();
            assert_eq!(back, bytes, "x = 0x{v:08X}");
        }
    }

    // ---- edge cases ----

    #[test]
    fn zero_length_rejected() {
        let mut dst = [0u16; 0];
        assert_eq!(
            rgba1010102_to_rgba16(&[], &mut dst),
            Err(SizeError::NotPixelAligned)
        );
        let mut dst_bytes = [0u8; 0];
        assert_eq!(
            rgba16_to_rgba1010102(&[], &mut dst_bytes),
            Err(SizeError::NotPixelAligned)
        );
    }

    #[test]
    fn unaligned_lengths_rejected() {
        // Source not multiple of 4 bytes
        let mut dst = [0u16; 4];
        assert_eq!(
            rgba1010102_to_rgba16(&[0, 0, 0], &mut dst),
            Err(SizeError::NotPixelAligned)
        );
        // Source not multiple of 4 u16
        let mut dst_bytes = [0u8; 4];
        assert_eq!(
            rgba16_to_rgba1010102(&[0u16, 0, 0], &mut dst_bytes),
            Err(SizeError::NotPixelAligned)
        );
    }

    #[test]
    fn dst_too_small_rejected() {
        let src = [0u8; 8]; // 2 pixels
        let mut dst = [0u16; 7]; // need 8
        assert_eq!(
            rgba1010102_to_rgba16(&src, &mut dst),
            Err(SizeError::PixelCountMismatch)
        );
        let src_chans = [0u16; 8]; // 2 pixels
        let mut dst_bytes = [0u8; 7]; // need 8
        assert_eq!(
            rgba16_to_rgba1010102(&src_chans, &mut dst_bytes),
            Err(SizeError::PixelCountMismatch)
        );
    }

    #[test]
    fn single_pixel_works() {
        let src = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut dst = [0u16; 4];
        rgba1010102_to_rgba16(&src, &mut dst).unwrap();
        assert_eq!(dst, [1023, 1023, 1023, 1023]);
    }

    #[test]
    fn pack_clamps_via_mask() {
        // Values > 1023 are masked. 0xFFFF & 0x3FF == 0x3FF (1023).
        let mut dst = [0u8; 4];
        rgba16_to_rgba1010102(&[0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF], &mut dst).unwrap();
        // alpha 0xFFFF -> compress_10_to_2(1023) = 3
        assert_eq!(dst, [0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn alpha_2bit_expansion_documented_values() {
        // Spot-check the four legal alpha values per the doc comment.
        let cases = [(0u8, 0u16), (1, 341), (2, 682), (3, 1023)];
        for (a2, want) in cases {
            // Build packed pixel: alpha in MSBs only.
            let v = (a2 as u32) << 30;
            let bytes = v.to_le_bytes();
            let mut dst = [0u16; 4];
            rgba1010102_to_rgba16(&bytes, &mut dst).unwrap();
            assert_eq!(dst[3], want, "alpha {a2}");
        }
    }

    // ---- strided ----

    #[test]
    fn strided_unpack_matches_contiguous() {
        let width = 7;
        let height = 5;
        let src_stride_bytes = width * 4 + 6; // 6 bytes padding per row
        let dst_stride_u16 = width * 4 + 3; // 3 u16 padding per row

        // Build source with distinct values per row+col so padding bugs would
        // misalign data.
        let mut src = vec![0u8; src_stride_bytes * height];
        for y in 0..height {
            for x in 0..width {
                let v = (y as u32 * 1000 + x as u32 * 7).wrapping_mul(0xDEAD);
                src[y * src_stride_bytes + x * 4..][..4].copy_from_slice(&v.to_le_bytes());
            }
        }

        let mut dst = vec![0u16; dst_stride_u16 * height];
        rgba1010102_to_rgba16_strided(
            &src,
            &mut dst,
            width,
            height,
            src_stride_bytes,
            dst_stride_u16,
        )
        .unwrap();

        // Compare each row against the contiguous unpacker.
        for y in 0..height {
            let s_row = &src[y * src_stride_bytes..][..width * 4];
            let mut want = vec![0u16; width * 4];
            rgba1010102_to_rgba16(s_row, &mut want).unwrap();
            assert_eq!(
                &dst[y * dst_stride_u16..][..width * 4],
                want.as_slice(),
                "row {y}"
            );
            // Padding bytes after row content must remain untouched (zero).
            for &p in &dst[y * dst_stride_u16 + width * 4..(y + 1) * dst_stride_u16] {
                assert_eq!(p, 0, "padding mutated in row {y}");
            }
        }
    }

    #[test]
    fn strided_pack_matches_contiguous() {
        let width = 3;
        let height = 4;
        let src_stride_u16 = width * 4 + 5;
        let dst_stride_bytes = width * 4 + 7;

        let mut src = vec![0u16; src_stride_u16 * height];
        for y in 0..height {
            for x in 0..width {
                let base = (y * 100 + x * 11) as u16;
                let off = y * src_stride_u16 + x * 4;
                src[off] = base & 0x3FF;
                src[off + 1] = (base.wrapping_mul(3)) & 0x3FF;
                src[off + 2] = (base.wrapping_mul(5)) & 0x3FF;
                src[off + 3] = expand2_to_10((base as u32) & 0x3);
            }
        }

        let mut dst = vec![0u8; dst_stride_bytes * height];
        rgba16_to_rgba1010102_strided(
            &src,
            &mut dst,
            width,
            height,
            src_stride_u16,
            dst_stride_bytes,
        )
        .unwrap();

        for y in 0..height {
            let s_row = &src[y * src_stride_u16..][..width * 4];
            let mut want = vec![0u8; width * 4];
            rgba16_to_rgba1010102(s_row, &mut want).unwrap();
            assert_eq!(
                &dst[y * dst_stride_bytes..][..width * 4],
                want.as_slice(),
                "row {y}"
            );
        }
    }

    #[test]
    fn strided_invalid_dimensions_rejected() {
        let mut dst = [0u16; 16];
        // Width 0
        assert_eq!(
            rgba1010102_to_rgba16_strided(&[0u8; 16], &mut dst, 0, 1, 4, 4),
            Err(SizeError::InvalidStride)
        );
        // Stride < row_bytes
        assert_eq!(
            rgba1010102_to_rgba16_strided(&[0u8; 16], &mut dst, 4, 1, 8, 16),
            Err(SizeError::InvalidStride)
        );
        // Buffer too small for stride×height
        assert_eq!(
            rgba1010102_to_rgba16_strided(&[0u8; 16], &mut dst, 1, 5, 4, 4),
            Err(SizeError::InvalidStride)
        );
    }

    #[test]
    fn round_trip_via_strided() {
        // Strided unpack then strided pack reproduces the input bytes exactly,
        // even with non-trivial strides.
        let width = 5;
        let height = 3;
        let src_stride_bytes = width * 4 + 2;
        let intermediate_stride_u16 = width * 4 + 4;
        let dst_stride_bytes = width * 4 + 6;

        let mut rng = Lcg::new(7);
        let mut src = vec![0u8; src_stride_bytes * height];
        for y in 0..height {
            for x in 0..width {
                let off = y * src_stride_bytes + x * 4;
                src[off..off + 4].copy_from_slice(&rng.next_u32().to_le_bytes());
            }
        }

        let mut chans = vec![0u16; intermediate_stride_u16 * height];
        rgba1010102_to_rgba16_strided(
            &src,
            &mut chans,
            width,
            height,
            src_stride_bytes,
            intermediate_stride_u16,
        )
        .unwrap();

        let mut back = vec![0u8; dst_stride_bytes * height];
        rgba16_to_rgba1010102_strided(
            &chans,
            &mut back,
            width,
            height,
            intermediate_stride_u16,
            dst_stride_bytes,
        )
        .unwrap();

        for y in 0..height {
            assert_eq!(
                &back[y * dst_stride_bytes..][..width * 4],
                &src[y * src_stride_bytes..][..width * 4],
                "row {y}"
            );
        }
    }
}
