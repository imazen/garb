//! Interleaved-RGB → planar-`f32` deinterleave (experimental).
//!
//! Hot-path primitive for image analyzers, ML preprocessing, and any pipeline
//! that wants planar `f32` input from packed RGB bytes/u16. Both 8-bit
//! (`RGB24`) and 16-bit (`RGB48`) sources are supported; the output is always
//! three contiguous `f32` planes whose values match the integer source
//! (no normalization, no transfer-function math — that lives in `linear-srgb`).
//!
//! The scalar fallback compiles to a long `vpinsrb` / `vpinsrw` chain on x86 —
//! 21 inserts per 8-pixel chunk for `RGB24`. The SIMD paths replace that with:
//!
//! - **x86-64 AVX2 (`v3`)** — two overlapping 16-byte loads, three `vpshufb`
//!   pairs to gather each plane's bytes into the low 8 bytes of an XMM,
//!   `vpmovzxbd` / `vpmovzxwd` to widen to `i32x8`, `vcvtdq2ps` to `f32x8`.
//! - **aarch64 NEON** — `vld3_u8` / `vld3q_u16` hardware structure-loads
//!   deinterleave in a single instruction, then `vmovl_*` widens and
//!   `vcvtq_f32_u32` converts.
//! - **Scalar** — explicit `array::from_fn` writes, exposed for benchmarking
//!   the unaccelerated path.

use crate::SizeError;
#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
use archmage::incant;
use archmage::prelude::*;

// ===========================================================================
// Public slice-level API
// ===========================================================================

/// Deinterleave packed RGB24 (`u8`) into three contiguous `f32` planes.
///
/// `src.len()` must be a multiple of 3 (one byte per channel). Each output
/// plane must hold at least `src.len() / 3` floats. The integer values
/// `0..=255` are converted directly to `f32` — no normalization.
///
/// # Errors
/// - [`SizeError::NotPixelAligned`] if `src.len() % 3 != 0` or `src` is empty.
/// - [`SizeError::PixelCountMismatch`] if any plane is shorter than the pixel
///   count.
pub fn rgb24_to_planes_f32(
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) -> Result<(), SizeError> {
    if src.is_empty() || !src.len().is_multiple_of(3) {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = src.len() / 3;
    if r.len() < pixels || g.len() < pixels || b.len() < pixels {
        return Err(SizeError::PixelCountMismatch);
    }
    let r = &mut r[..pixels];
    let g = &mut g[..pixels];
    let b = &mut b[..pixels];
    incant!(rgb24_to_planes_impl(src, r, g, b), [v3, neon, scalar]);
    Ok(())
}

/// Deinterleave packed RGB48 (`u16`) into three contiguous `f32` planes.
///
/// `src.len()` must be a multiple of 3 (one `u16` per channel). Each output
/// plane must hold at least `src.len() / 3` floats. The integer values
/// `0..=65535` are converted directly to `f32` — no normalization.
///
/// # Errors
/// - [`SizeError::NotPixelAligned`] if `src.len() % 3 != 0` or `src` is empty.
/// - [`SizeError::PixelCountMismatch`] if any plane is shorter than the pixel
///   count.
pub fn rgb48_to_planes_f32(
    src: &[u16],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) -> Result<(), SizeError> {
    if src.is_empty() || !src.len().is_multiple_of(3) {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = src.len() / 3;
    if r.len() < pixels || g.len() < pixels || b.len() < pixels {
        return Err(SizeError::PixelCountMismatch);
    }
    let r = &mut r[..pixels];
    let g = &mut g[..pixels];
    let b = &mut b[..pixels];
    incant!(rgb48_to_planes_impl(src, r, g, b), [v3, neon, scalar]);
    Ok(())
}

/// Scalar-only RGB24 → planar f32 (no validation, no SIMD dispatch). Exists
/// solely to give benchmarks a stable handle on the unaccelerated path.
///
/// Caller must ensure `src.len() % 3 == 0` and each plane has at least
/// `src.len() / 3` elements; the function panics on shorter slices because
/// it falls through to slice indexing.
#[doc(hidden)]
pub fn scalar_only_rgb24(src: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let pixels = src.len() / 3;
    rgb24_to_planes_loop_scalar(src, &mut r[..pixels], &mut g[..pixels], &mut b[..pixels]);
}

/// Scalar-only RGB48 → planar f32. See [`scalar_only_rgb24`].
#[doc(hidden)]
pub fn scalar_only_rgb48(src: &[u16], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let pixels = src.len() / 3;
    rgb48_to_planes_loop_scalar(src, &mut r[..pixels], &mut g[..pixels], &mut b[..pixels]);
}

/// Scalar source compiled under AVX2 target-feature so LLVM can autovectorize.
/// On non-x86_64 builds this falls back to the plain scalar path. Returns
/// `false` if AVX2 is unavailable at runtime so benchmarks can skip cleanly.
#[doc(hidden)]
pub fn autovec_avx2_rgb24(src: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32]) -> bool {
    let pixels = src.len() / 3;
    let r = &mut r[..pixels];
    let g = &mut g[..pixels];
    let b = &mut b[..pixels];
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = X64V3Token::summon() {
            x86::rgb24_to_planes_impl_v3_autovec(t, src, r, g, b);
            return true;
        }
    }
    rgb24_to_planes_loop_scalar(src, r, g, b);
    false
}

/// u16 counterpart to [`autovec_avx2_rgb24`].
#[doc(hidden)]
pub fn autovec_avx2_rgb48(src: &[u16], r: &mut [f32], g: &mut [f32], b: &mut [f32]) -> bool {
    let pixels = src.len() / 3;
    let r = &mut r[..pixels];
    let g = &mut g[..pixels];
    let b = &mut b[..pixels];
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = X64V3Token::summon() {
            x86::rgb48_to_planes_impl_v3_autovec(t, src, r, g, b);
            return true;
        }
    }
    rgb48_to_planes_loop_scalar(src, r, g, b);
    false
}

// ===========================================================================
// Scalar (fallback + benchmark baseline)
// ===========================================================================

#[inline(always)]
fn rgb24_chunk8_scalar(c: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..8 {
        r[i] = c[i * 3] as f32;
        g[i] = c[i * 3 + 1] as f32;
        b[i] = c[i * 3 + 2] as f32;
    }
    (r, g, b)
}

#[inline(always)]
fn rgb48_chunk8_scalar(c: &[u16; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..8 {
        r[i] = c[i * 3] as f32;
        g[i] = c[i * 3 + 1] as f32;
        b[i] = c[i * 3 + 2] as f32;
    }
    (r, g, b)
}

#[inline(always)]
fn rgb24_to_planes_loop_scalar(src: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let pixels = src.len() / 3;
    let n_chunks = pixels / 8;
    for ci in 0..n_chunks {
        let bs = ci * 24;
        let ps = ci * 8;
        let c: &[u8; 24] = src[bs..bs + 24].try_into().unwrap();
        let (rv, gv, bv) = rgb24_chunk8_scalar(c);
        r[ps..ps + 8].copy_from_slice(&rv);
        g[ps..ps + 8].copy_from_slice(&gv);
        b[ps..ps + 8].copy_from_slice(&bv);
    }
    for p in (n_chunks * 8)..pixels {
        r[p] = src[p * 3] as f32;
        g[p] = src[p * 3 + 1] as f32;
        b[p] = src[p * 3 + 2] as f32;
    }
}

#[inline(always)]
fn rgb48_to_planes_loop_scalar(src: &[u16], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let pixels = src.len() / 3;
    let n_chunks = pixels / 8;
    for ci in 0..n_chunks {
        let bs = ci * 24;
        let ps = ci * 8;
        let c: &[u16; 24] = src[bs..bs + 24].try_into().unwrap();
        let (rv, gv, bv) = rgb48_chunk8_scalar(c);
        r[ps..ps + 8].copy_from_slice(&rv);
        g[ps..ps + 8].copy_from_slice(&gv);
        b[ps..ps + 8].copy_from_slice(&bv);
    }
    for p in (n_chunks * 8)..pixels {
        r[p] = src[p * 3] as f32;
        g[p] = src[p * 3 + 1] as f32;
        b[p] = src[p * 3 + 2] as f32;
    }
}

pub(crate) fn rgb24_to_planes_impl_scalar(
    _t: ScalarToken,
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    rgb24_to_planes_loop_scalar(src, r, g, b);
}

pub(crate) fn rgb48_to_planes_impl_scalar(
    _t: ScalarToken,
    src: &[u16],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    rgb48_to_planes_loop_scalar(src, r, g, b);
}

// ===========================================================================
// x86-64 AVX2 (v3)
// ===========================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;

    // RGB24: per-plane shuffle masks. Two overlapping 16-byte loads cover the
    // 24-byte chunk; we shuffle each load to put the wanted plane bytes into
    // a contiguous run, then OR. The pair's 0x80 lanes guarantee a clean OR.

    // Layout of `lo` (bytes 0..16):
    //   R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    // Layout of `hi` (bytes 8..24):
    //   B2 R3 G3 B3 R4 G4 B4 R5 G5 B5 R6 G6 B6 R7 G7 B7

    const R_LO: [i8; 16] = [
        0, 3, 6, 9, 12, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const R_HI: [i8; 16] = [
        -128, -128, -128, -128, -128, 7, 10, 13, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const G_LO: [i8; 16] = [
        1, 4, 7, 10, 13, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const G_HI: [i8; 16] = [
        -128, -128, -128, -128, -128, 8, 11, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const B_LO: [i8; 16] = [
        2, 5, 8, 11, 14, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const B_HI: [i8; 16] = [
        -128, -128, -128, -128, -128, 9, 12, 15, -128, -128, -128, -128, -128, -128, -128, -128,
    ];

    // RGB48: three 16-byte loads cover the 48-byte chunk. v1 = bytes 0..16
    // (u16[0..8]), v2 = bytes 16..32 (u16[8..16]), v3 = bytes 32..48
    // (u16[16..24]). Each plane needs two bytes per element (u16 → output
    // 16 bytes). The masks below come from working out the byte index of
    // R/G/B[0..8] within each 16-byte load.

    const R16_V1: [i8; 16] = [
        0, 1, 6, 7, 12, 13, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const R16_V2: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, 2, 3, 8, 9, 14, 15, -128, -128, -128, -128,
    ];
    const R16_V3: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 4, 5, 10, 11,
    ];

    const G16_V1: [i8; 16] = [
        2, 3, 8, 9, 14, 15, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const G16_V2: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, 4, 5, 10, 11, -128, -128, -128, -128, -128, -128,
    ];
    const G16_V3: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 0, 1, 6, 7, 12, 13,
    ];

    const B16_V1: [i8; 16] = [
        4, 5, 10, 11, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const B16_V2: [i8; 16] = [
        -128, -128, -128, -128, 0, 1, 6, 7, 12, 13, -128, -128, -128, -128, -128, -128,
    ];
    const B16_V3: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 2, 3, 8, 9, 14, 15,
    ];

    #[rite]
    pub fn rgb24_chunk8_v3(_t: X64V3Token, c: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let lo16: &[u8; 16] = c[0..16].try_into().unwrap();
        let hi16: &[u8; 16] = c[8..24].try_into().unwrap();
        let lo = _mm_loadu_si128(lo16);
        let hi = _mm_loadu_si128(hi16);

        let r_lo = _mm_shuffle_epi8(lo, _mm_loadu_si128(&R_LO));
        let r_hi = _mm_shuffle_epi8(hi, _mm_loadu_si128(&R_HI));
        let r_u8 = _mm_or_si128(r_lo, r_hi);
        let g_lo = _mm_shuffle_epi8(lo, _mm_loadu_si128(&G_LO));
        let g_hi = _mm_shuffle_epi8(hi, _mm_loadu_si128(&G_HI));
        let g_u8 = _mm_or_si128(g_lo, g_hi);
        let b_lo = _mm_shuffle_epi8(lo, _mm_loadu_si128(&B_LO));
        let b_hi = _mm_shuffle_epi8(hi, _mm_loadu_si128(&B_HI));
        let b_u8 = _mm_or_si128(b_lo, b_hi);

        let r_i32 = _mm256_cvtepu8_epi32(r_u8);
        let g_i32 = _mm256_cvtepu8_epi32(g_u8);
        let b_i32 = _mm256_cvtepu8_epi32(b_u8);

        let r_f32 = _mm256_cvtepi32_ps(r_i32);
        let g_f32 = _mm256_cvtepi32_ps(g_i32);
        let b_f32 = _mm256_cvtepi32_ps(b_i32);

        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        _mm256_storeu_ps(&mut r_out, r_f32);
        _mm256_storeu_ps(&mut g_out, g_f32);
        _mm256_storeu_ps(&mut b_out, b_f32);
        (r_out, g_out, b_out)
    }

    #[rite]
    pub fn rgb48_chunk8_v3(_t: X64V3Token, c: &[u16; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
        // Reinterpret the 24-u16 chunk as 48 bytes via 3 × 16-byte loads.
        let bytes: &[u8; 48] = bytemuck::cast_ref(c);
        let v1b: &[u8; 16] = bytes[0..16].try_into().unwrap();
        let v2b: &[u8; 16] = bytes[16..32].try_into().unwrap();
        let v3b: &[u8; 16] = bytes[32..48].try_into().unwrap();
        let v1 = _mm_loadu_si128(v1b);
        let v2 = _mm_loadu_si128(v2b);
        let v3 = _mm_loadu_si128(v3b);

        let r_v1 = _mm_shuffle_epi8(v1, _mm_loadu_si128(&R16_V1));
        let r_v2 = _mm_shuffle_epi8(v2, _mm_loadu_si128(&R16_V2));
        let r_v3 = _mm_shuffle_epi8(v3, _mm_loadu_si128(&R16_V3));
        let r_u16 = _mm_or_si128(_mm_or_si128(r_v1, r_v2), r_v3);

        let g_v1 = _mm_shuffle_epi8(v1, _mm_loadu_si128(&G16_V1));
        let g_v2 = _mm_shuffle_epi8(v2, _mm_loadu_si128(&G16_V2));
        let g_v3 = _mm_shuffle_epi8(v3, _mm_loadu_si128(&G16_V3));
        let g_u16 = _mm_or_si128(_mm_or_si128(g_v1, g_v2), g_v3);

        let b_v1 = _mm_shuffle_epi8(v1, _mm_loadu_si128(&B16_V1));
        let b_v2 = _mm_shuffle_epi8(v2, _mm_loadu_si128(&B16_V2));
        let b_v3 = _mm_shuffle_epi8(v3, _mm_loadu_si128(&B16_V3));
        let b_u16 = _mm_or_si128(_mm_or_si128(b_v1, b_v2), b_v3);

        let r_i32 = _mm256_cvtepu16_epi32(r_u16);
        let g_i32 = _mm256_cvtepu16_epi32(g_u16);
        let b_i32 = _mm256_cvtepu16_epi32(b_u16);

        let r_f32 = _mm256_cvtepi32_ps(r_i32);
        let g_f32 = _mm256_cvtepi32_ps(g_i32);
        let b_f32 = _mm256_cvtepi32_ps(b_i32);

        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        _mm256_storeu_ps(&mut r_out, r_f32);
        _mm256_storeu_ps(&mut g_out, g_f32);
        _mm256_storeu_ps(&mut b_out, b_f32);
        (r_out, g_out, b_out)
    }

    /// Scalar loop body, but compiled inside the AVX2 target-feature region so
    /// LLVM is free to autovectorize. Identical source to
    /// [`super::rgb24_to_planes_loop_scalar`] (which is `#[inline(always)]`).
    #[arcane]
    pub(crate) fn rgb24_to_planes_impl_v3_autovec(
        _t: X64V3Token,
        src: &[u8],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        super::rgb24_to_planes_loop_scalar(src, r, g, b);
    }

    /// Same idea for u16: the scalar loop, autovectorized under AVX2.
    #[arcane]
    pub(crate) fn rgb48_to_planes_impl_v3_autovec(
        _t: X64V3Token,
        src: &[u16],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        super::rgb48_to_planes_loop_scalar(src, r, g, b);
    }

    #[arcane]
    pub(crate) fn rgb24_to_planes_impl_v3(
        t: X64V3Token,
        src: &[u8],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let pixels = src.len() / 3;
        let n_chunks = pixels / 8;
        for ci in 0..n_chunks {
            let bs = ci * 24;
            let ps = ci * 8;
            let c: &[u8; 24] = src[bs..bs + 24].try_into().unwrap();
            let (rv, gv, bv) = rgb24_chunk8_v3(t, c);
            r[ps..ps + 8].copy_from_slice(&rv);
            g[ps..ps + 8].copy_from_slice(&gv);
            b[ps..ps + 8].copy_from_slice(&bv);
        }
        for p in (n_chunks * 8)..pixels {
            r[p] = src[p * 3] as f32;
            g[p] = src[p * 3 + 1] as f32;
            b[p] = src[p * 3 + 2] as f32;
        }
    }

    #[arcane]
    pub(crate) fn rgb48_to_planes_impl_v3(
        t: X64V3Token,
        src: &[u16],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let pixels = src.len() / 3;
        let n_chunks = pixels / 8;
        for ci in 0..n_chunks {
            let bs = ci * 24;
            let ps = ci * 8;
            let c: &[u16; 24] = src[bs..bs + 24].try_into().unwrap();
            let (rv, gv, bv) = rgb48_chunk8_v3(t, c);
            r[ps..ps + 8].copy_from_slice(&rv);
            g[ps..ps + 8].copy_from_slice(&gv);
            b[ps..ps + 8].copy_from_slice(&bv);
        }
        for p in (n_chunks * 8)..pixels {
            r[p] = src[p * 3] as f32;
            g[p] = src[p * 3 + 1] as f32;
            b[p] = src[p * 3 + 2] as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
use x86::{rgb24_to_planes_impl_v3, rgb48_to_planes_impl_v3};

// ===========================================================================
// Chunk-level primitives (8 pixels, AVX2)
// ===========================================================================
//
// These are the inner kernels exposed for callers that already run inside
// a `#[target_feature(enable = "avx2,...")]` region (set up by `#[arcane]`,
// `#[rite]`, or `#[magetypes]`) and want to deinterleave one 8-pixel chunk
// at a time without paying any per-call dispatch cost.
//
// Inlining behavior: `#[rite]` injects `#[inline]` and the matching
// `#[target_feature]` attrs, so when called from another `#[rite]` /
// `#[arcane]` / `#[magetypes]`-decorated function with the same features,
// LLVM inlines the body and keeps the f32x8 results in YMM registers
// (the returned `[f32; 8]` arrays do NOT round-trip through stack memory
// when the consumer reads them with another `#[target_feature]` SIMD load).
//
// Output shape matches the natural f32x8 layout: 3 fixed-size arrays of
// 8 floats each (R, G, B). Caller wraps each with their preferred f32x8
// abstraction (magetypes f32x8, std::simd, raw `__m256`, etc.).

/// AVX2 chunk-level deinterleave: `&[u8; 24]` (8 RGB pixels, packed) →
/// 3×`[f32; 8]` (planar). Caller provides an `X64V3Token` proving AVX2
/// is available; calling from inside a non-AVX2 target_feature region
/// will fail to compile.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub use x86::rgb24_chunk8_v3 as rgb24_chunk8_to_planes_v3;

/// AVX2 chunk-level deinterleave for `RGB48` (`u16` per channel). See
/// [`rgb24_chunk8_to_planes_v3`]; same shape, just `&[u16; 24]` input.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub use x86::rgb48_chunk8_v3 as rgb48_chunk8_to_planes_v3;

/// Scalar fallback chunk-level deinterleave. Compiles on every target;
/// LLVM autovectorizes inside whatever target_feature region the caller
/// sets up, so this is also the right thing to call from a `NeonToken`,
/// `Wasm128Token`, or generic `ScalarToken` consumer that doesn't have a
/// hand-tuned chunk kernel yet.
#[doc(hidden)]
#[inline(always)]
pub fn rgb24_chunk8_to_planes_scalar(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    rgb24_chunk8_scalar(chunk)
}

#[doc(hidden)]
#[inline(always)]
pub fn rgb48_chunk8_to_planes_scalar(chunk: &[u16; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    rgb48_chunk8_scalar(chunk)
}

// ===========================================================================
// aarch64 NEON — vld3 hardware deinterleave
// ===========================================================================

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;

    #[arcane]
    pub(crate) fn rgb24_to_planes_impl_neon(
        _t: NeonToken,
        src: &[u8],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        // 16-pixel chunks (vld3q_u8 reads 48 bytes / 16 RGB pixels)
        let pixels = src.len() / 3;
        let n_chunks = pixels / 16;
        for ci in 0..n_chunks {
            let bs = ci * 48;
            let ps = ci * 16;
            let c: &[u8; 48] = src[bs..bs + 48].try_into().unwrap();

            // vld3q_u8: hardware structure-load → 3×u8x16 (planar)
            let uint8x16x3_t(r_u8x16, g_u8x16, b_u8x16) = vld3q_u8(c);

            // Widen each 16-byte plane to 16 f32 via u16x8 → u32x4 → f32x4 (×4)
            let r_u16_lo = vmovl_u8(vget_low_u8(r_u8x16));
            let r_u16_hi = vmovl_high_u8(r_u8x16);
            let g_u16_lo = vmovl_u8(vget_low_u8(g_u8x16));
            let g_u16_hi = vmovl_high_u8(g_u8x16);
            let b_u16_lo = vmovl_u8(vget_low_u8(b_u8x16));
            let b_u16_hi = vmovl_high_u8(b_u8x16);

            let r0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16_lo)));
            let r1 = vcvtq_f32_u32(vmovl_high_u16(r_u16_lo));
            let r2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16_hi)));
            let r3 = vcvtq_f32_u32(vmovl_high_u16(r_u16_hi));
            let g0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16_lo)));
            let g1 = vcvtq_f32_u32(vmovl_high_u16(g_u16_lo));
            let g2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16_hi)));
            let g3 = vcvtq_f32_u32(vmovl_high_u16(g_u16_hi));
            let b0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_lo)));
            let b1 = vcvtq_f32_u32(vmovl_high_u16(b_u16_lo));
            let b2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_hi)));
            let b3 = vcvtq_f32_u32(vmovl_high_u16(b_u16_hi));

            let r_chunk: &mut [f32; 16] = (&mut r[ps..ps + 16]).try_into().unwrap();
            let g_chunk: &mut [f32; 16] = (&mut g[ps..ps + 16]).try_into().unwrap();
            let b_chunk: &mut [f32; 16] = (&mut b[ps..ps + 16]).try_into().unwrap();
            vst1q_f32_x4(bytemuck::cast_mut(r_chunk), float32x4x4_t(r0, r1, r2, r3));
            vst1q_f32_x4(bytemuck::cast_mut(g_chunk), float32x4x4_t(g0, g1, g2, g3));
            vst1q_f32_x4(bytemuck::cast_mut(b_chunk), float32x4x4_t(b0, b1, b2, b3));
        }
        for p in (n_chunks * 16)..pixels {
            r[p] = src[p * 3] as f32;
            g[p] = src[p * 3 + 1] as f32;
            b[p] = src[p * 3 + 2] as f32;
        }
    }

    #[arcane]
    pub(crate) fn rgb48_to_planes_impl_neon(
        _t: NeonToken,
        src: &[u16],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let pixels = src.len() / 3;
        let n_chunks = pixels / 8;
        for ci in 0..n_chunks {
            let us = ci * 24;
            let ps = ci * 8;
            let c: &[u16; 24] = src[us..us + 24].try_into().unwrap();

            // vld3q_u16: structure-load 8 RGB pixels (24 u16s) → 3×u16x8
            let uint16x8x3_t(r_u16, g_u16, b_u16) = vld3q_u16(c);

            let r_lo_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16)));
            let r_hi_f = vcvtq_f32_u32(vmovl_high_u16(r_u16));
            let g_lo_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16)));
            let g_hi_f = vcvtq_f32_u32(vmovl_high_u16(g_u16));
            let b_lo_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16)));
            let b_hi_f = vcvtq_f32_u32(vmovl_high_u16(b_u16));

            let r_chunk: &mut [f32; 8] = (&mut r[ps..ps + 8]).try_into().unwrap();
            let g_chunk: &mut [f32; 8] = (&mut g[ps..ps + 8]).try_into().unwrap();
            let b_chunk: &mut [f32; 8] = (&mut b[ps..ps + 8]).try_into().unwrap();
            vst1q_f32_x2(bytemuck::cast_mut(r_chunk), float32x4x2_t(r_lo_f, r_hi_f));
            vst1q_f32_x2(bytemuck::cast_mut(g_chunk), float32x4x2_t(g_lo_f, g_hi_f));
            vst1q_f32_x2(bytemuck::cast_mut(b_chunk), float32x4x2_t(b_lo_f, b_hi_f));
        }
        for p in (n_chunks * 8)..pixels {
            r[p] = src[p * 3] as f32;
            g[p] = src[p * 3 + 1] as f32;
            b[p] = src[p * 3 + 2] as f32;
        }
    }
}

#[cfg(target_arch = "aarch64")]
use arm::{rgb24_to_planes_impl_neon, rgb48_to_planes_impl_neon};

// ===========================================================================
// f32 RGB / RGBA  ⇄  f32 planes (3 / 4 channel, identity)
// ===========================================================================
//
// Used by zenfilters' `scatter_srgb_passthrough` and the inline gather in
// pipeline.rs — both pure identity copies between AoS and SoA. f32→f32 has
// 1:1 memory ratio (no widen), so DRAM bandwidth dominates much sooner than
// the integer→f32 paths above. We currently ship scalar + an AVX2
// autovectorize wrapper (LLVM gets to pick the shuffles); benchmarks show
// the lift is large enough that hand-tuning hasn't been justified yet.

// --- Scalar loops ---------------------------------------------------------

#[inline(always)]
fn rgb_f32_to_planes_loop_scalar(src: &[f32], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let pixels = src.len() / 3;
    for i in 0..pixels {
        r[i] = src[i * 3];
        g[i] = src[i * 3 + 1];
        b[i] = src[i * 3 + 2];
    }
}

#[inline(always)]
fn rgba_f32_to_planes_loop_scalar(
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    a: &mut [f32],
) {
    let pixels = src.len() / 4;
    for i in 0..pixels {
        r[i] = src[i * 4];
        g[i] = src[i * 4 + 1];
        b[i] = src[i * 4 + 2];
        a[i] = src[i * 4 + 3];
    }
}

#[inline(always)]
fn planes_to_rgb_f32_loop_scalar(r: &[f32], g: &[f32], b: &[f32], dst: &mut [f32]) {
    let pixels = r.len();
    for i in 0..pixels {
        dst[i * 3] = r[i];
        dst[i * 3 + 1] = g[i];
        dst[i * 3 + 2] = b[i];
    }
}

#[inline(always)]
fn planes_to_rgba_f32_loop_scalar(r: &[f32], g: &[f32], b: &[f32], a: &[f32], dst: &mut [f32]) {
    let pixels = r.len();
    for i in 0..pixels {
        dst[i * 4] = r[i];
        dst[i * 4 + 1] = g[i];
        dst[i * 4 + 2] = b[i];
        dst[i * 4 + 3] = a[i];
    }
}

pub(crate) fn rgb_f32_to_planes_impl_scalar(
    _t: ScalarToken,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    rgb_f32_to_planes_loop_scalar(src, r, g, b);
}

pub(crate) fn rgba_f32_to_planes_impl_scalar(
    _t: ScalarToken,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    a: &mut [f32],
) {
    rgba_f32_to_planes_loop_scalar(src, r, g, b, a);
}

pub(crate) fn planes_to_rgb_f32_impl_scalar(
    _t: ScalarToken,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    dst: &mut [f32],
) {
    planes_to_rgb_f32_loop_scalar(r, g, b, dst);
}

pub(crate) fn planes_to_rgba_f32_impl_scalar(
    _t: ScalarToken,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    a: &[f32],
    dst: &mut [f32],
) {
    planes_to_rgba_f32_loop_scalar(r, g, b, a, dst);
}

// --- x86_64 AVX2 autovectorize wrappers ----------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_f32 {
    use super::*;

    #[arcane]
    pub(crate) fn rgb_f32_to_planes_impl_v3(
        _t: X64V3Token,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        super::rgb_f32_to_planes_loop_scalar(src, r, g, b);
    }

    #[arcane]
    pub(crate) fn rgba_f32_to_planes_impl_v3(
        _t: X64V3Token,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        a: &mut [f32],
    ) {
        super::rgba_f32_to_planes_loop_scalar(src, r, g, b, a);
    }

    #[arcane]
    pub(crate) fn planes_to_rgb_f32_impl_v3(
        _t: X64V3Token,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        dst: &mut [f32],
    ) {
        super::planes_to_rgb_f32_loop_scalar(r, g, b, dst);
    }

    #[arcane]
    pub(crate) fn planes_to_rgba_f32_impl_v3(
        _t: X64V3Token,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        a: &[f32],
        dst: &mut [f32],
    ) {
        super::planes_to_rgba_f32_loop_scalar(r, g, b, a, dst);
    }
}

#[cfg(target_arch = "x86_64")]
use x86_f32::{
    planes_to_rgb_f32_impl_v3, planes_to_rgba_f32_impl_v3, rgb_f32_to_planes_impl_v3,
    rgba_f32_to_planes_impl_v3,
};

// --- aarch64 NEON wrappers ------------------------------------------------
//
// NEON is the aarch64 baseline so the plain scalar function is already
// compiled with NEON enabled — LLVM autovectorizes. We still expose a NEON
// dispatch hook so future hand-tuned ld3q/st3q paths can plug in without
// touching call sites.

#[cfg(target_arch = "aarch64")]
mod arm_f32 {
    use super::*;

    #[arcane]
    pub(crate) fn rgb_f32_to_planes_impl_neon(
        _t: NeonToken,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        super::rgb_f32_to_planes_loop_scalar(src, r, g, b);
    }

    #[arcane]
    pub(crate) fn rgba_f32_to_planes_impl_neon(
        _t: NeonToken,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        a: &mut [f32],
    ) {
        super::rgba_f32_to_planes_loop_scalar(src, r, g, b, a);
    }

    #[arcane]
    pub(crate) fn planes_to_rgb_f32_impl_neon(
        _t: NeonToken,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        dst: &mut [f32],
    ) {
        super::planes_to_rgb_f32_loop_scalar(r, g, b, dst);
    }

    #[arcane]
    pub(crate) fn planes_to_rgba_f32_impl_neon(
        _t: NeonToken,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        a: &[f32],
        dst: &mut [f32],
    ) {
        super::planes_to_rgba_f32_loop_scalar(r, g, b, a, dst);
    }
}

#[cfg(target_arch = "aarch64")]
use arm_f32::{
    planes_to_rgb_f32_impl_neon, planes_to_rgba_f32_impl_neon, rgb_f32_to_planes_impl_neon,
    rgba_f32_to_planes_impl_neon,
};

// --- Public API ----------------------------------------------------------

/// Deinterleave packed `f32` RGB into three contiguous `f32` planes.
///
/// `src.len()` must be a multiple of 3. Each plane must hold at least
/// `src.len() / 3` floats. Pure identity per channel — no transfer math.
///
/// # Errors
/// - [`SizeError::NotPixelAligned`] if `src.len() % 3 != 0` or `src` is empty.
/// - [`SizeError::PixelCountMismatch`] if any plane is shorter than the pixel
///   count.
pub fn rgb_f32_to_planes_f32(
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) -> Result<(), SizeError> {
    if src.is_empty() || !src.len().is_multiple_of(3) {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = src.len() / 3;
    if r.len() < pixels || g.len() < pixels || b.len() < pixels {
        return Err(SizeError::PixelCountMismatch);
    }
    let r = &mut r[..pixels];
    let g = &mut g[..pixels];
    let b = &mut b[..pixels];
    incant!(rgb_f32_to_planes_impl(src, r, g, b), [v3, neon, scalar]);
    Ok(())
}

/// Deinterleave packed `f32` RGBA into four contiguous `f32` planes.
///
/// `src.len()` must be a multiple of 4. Each plane must hold at least
/// `src.len() / 4` floats. Pure identity per channel.
///
/// # Errors
/// - [`SizeError::NotPixelAligned`] if `src.len() % 4 != 0` or `src` is empty.
/// - [`SizeError::PixelCountMismatch`] if any plane is shorter than the pixel
///   count.
pub fn rgba_f32_to_planes_f32(
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    a: &mut [f32],
) -> Result<(), SizeError> {
    if src.is_empty() || !src.len().is_multiple_of(4) {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = src.len() / 4;
    if r.len() < pixels || g.len() < pixels || b.len() < pixels || a.len() < pixels {
        return Err(SizeError::PixelCountMismatch);
    }
    let r = &mut r[..pixels];
    let g = &mut g[..pixels];
    let b = &mut b[..pixels];
    let a = &mut a[..pixels];
    incant!(rgba_f32_to_planes_impl(src, r, g, b, a), [v3, neon, scalar]);
    Ok(())
}

/// Interleave three `f32` planes into packed `f32` RGB.
///
/// All three planes must be the same length. `dst.len()` must be at least
/// `3 × r.len()`. Pure identity per channel.
///
/// # Errors
/// - [`SizeError::NotPixelAligned`] if any plane is empty or planes have
///   different lengths.
/// - [`SizeError::PixelCountMismatch`] if `dst` is smaller than `3 × r.len()`.
pub fn planes_f32_to_rgb_f32(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    dst: &mut [f32],
) -> Result<(), SizeError> {
    if r.is_empty() || r.len() != g.len() || r.len() != b.len() {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = r.len();
    if dst.len() < pixels * 3 {
        return Err(SizeError::PixelCountMismatch);
    }
    let dst = &mut dst[..pixels * 3];
    incant!(planes_to_rgb_f32_impl(r, g, b, dst), [v3, neon, scalar]);
    Ok(())
}

/// Interleave four `f32` planes into packed `f32` RGBA.
///
/// All four planes must be the same length. `dst.len()` must be at least
/// `4 × r.len()`. Pure identity per channel.
///
/// # Errors
/// - [`SizeError::NotPixelAligned`] if any plane is empty or planes differ
///   in length.
/// - [`SizeError::PixelCountMismatch`] if `dst` is smaller than `4 × r.len()`.
pub fn planes_f32_to_rgba_f32(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    a: &[f32],
    dst: &mut [f32],
) -> Result<(), SizeError> {
    if r.is_empty() || r.len() != g.len() || r.len() != b.len() || r.len() != a.len() {
        return Err(SizeError::NotPixelAligned);
    }
    let pixels = r.len();
    if dst.len() < pixels * 4 {
        return Err(SizeError::PixelCountMismatch);
    }
    let dst = &mut dst[..pixels * 4];
    incant!(planes_to_rgba_f32_impl(r, g, b, a, dst), [v3, neon, scalar]);
    Ok(())
}

/// Scalar-only handle for benchmarking the f32 RGB deinterleave path.
///
/// `#[inline(always)]` so it inlines into a `#[target_feature]` region when
/// callers want to pull the whole-loop autovectorize comparison out to a
/// single dispatch boundary.
#[doc(hidden)]
#[inline(always)]
pub fn scalar_only_rgb_f32_to_planes(src: &[f32], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let pixels = src.len() / 3;
    rgb_f32_to_planes_loop_scalar(src, &mut r[..pixels], &mut g[..pixels], &mut b[..pixels]);
}

/// Scalar-only handle for benchmarking f32 RGBA deinterleave.
#[doc(hidden)]
#[inline(always)]
pub fn scalar_only_rgba_f32_to_planes(
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    a: &mut [f32],
) {
    let pixels = src.len() / 4;
    rgba_f32_to_planes_loop_scalar(
        src,
        &mut r[..pixels],
        &mut g[..pixels],
        &mut b[..pixels],
        &mut a[..pixels],
    );
}

/// Scalar-only handle for benchmarking the f32 planes→RGB interleave path.
#[doc(hidden)]
#[inline(always)]
pub fn scalar_only_planes_f32_to_rgb(r: &[f32], g: &[f32], b: &[f32], dst: &mut [f32]) {
    planes_to_rgb_f32_loop_scalar(r, g, b, &mut dst[..r.len() * 3]);
}

/// Scalar-only handle for benchmarking the f32 planes→RGBA interleave path.
#[doc(hidden)]
#[inline(always)]
pub fn scalar_only_planes_f32_to_rgba(r: &[f32], g: &[f32], b: &[f32], a: &[f32], dst: &mut [f32]) {
    planes_to_rgba_f32_loop_scalar(r, g, b, a, &mut dst[..r.len() * 4]);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;

    fn ref_planes_u8(
        src: &[u8],
    ) -> (
        alloc::vec::Vec<f32>,
        alloc::vec::Vec<f32>,
        alloc::vec::Vec<f32>,
    ) {
        let pixels = src.len() / 3;
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        for i in 0..pixels {
            r[i] = src[i * 3] as f32;
            g[i] = src[i * 3 + 1] as f32;
            b[i] = src[i * 3 + 2] as f32;
        }
        (r, g, b)
    }

    fn ref_planes_u16(
        src: &[u16],
    ) -> (
        alloc::vec::Vec<f32>,
        alloc::vec::Vec<f32>,
        alloc::vec::Vec<f32>,
    ) {
        let pixels = src.len() / 3;
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        for i in 0..pixels {
            r[i] = src[i * 3] as f32;
            g[i] = src[i * 3 + 1] as f32;
            b[i] = src[i * 3 + 2] as f32;
        }
        (r, g, b)
    }

    #[test]
    fn rgb24_round_trip_aligned() {
        let pixels = 8 * 64;
        let src: alloc::vec::Vec<u8> = (0..pixels * 3).map(|i| (i & 0xFF) as u8).collect();
        let (r_ref, g_ref, b_ref) = ref_planes_u8(&src);
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        rgb24_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r, r_ref);
        assert_eq!(g, g_ref);
        assert_eq!(b, b_ref);
    }

    #[test]
    fn rgb24_round_trip_with_tail() {
        // 67 pixels = 8 chunks × 8 + 3 tail pixels
        let pixels = 67;
        let src: alloc::vec::Vec<u8> = (0..pixels * 3)
            .map(|i: usize| (i.wrapping_mul(31) & 0xFF) as u8)
            .collect();
        let (r_ref, g_ref, b_ref) = ref_planes_u8(&src);
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        rgb24_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r, r_ref);
        assert_eq!(g, g_ref);
        assert_eq!(b, b_ref);
    }

    #[test]
    fn rgb48_round_trip_aligned() {
        let pixels = 8 * 32;
        let src: alloc::vec::Vec<u16> = (0..pixels * 3)
            .map(|i: usize| (i.wrapping_mul(257) & 0xFFFF) as u16)
            .collect();
        let (r_ref, g_ref, b_ref) = ref_planes_u16(&src);
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        rgb48_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r, r_ref);
        assert_eq!(g, g_ref);
        assert_eq!(b, b_ref);
    }

    #[test]
    fn rgb48_round_trip_with_tail() {
        let pixels = 51;
        let src: alloc::vec::Vec<u16> = (0..pixels * 3)
            .map(|i: usize| (i.wrapping_mul(8191) & 0xFFFF) as u16)
            .collect();
        let (r_ref, g_ref, b_ref) = ref_planes_u16(&src);
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        rgb48_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r, r_ref);
        assert_eq!(g, g_ref);
        assert_eq!(b, b_ref);
    }

    #[test]
    fn errors_rejected() {
        let mut r = vec![0.0; 4];
        let mut g = vec![0.0; 4];
        let mut b = vec![0.0; 4];
        // Empty src
        assert_eq!(
            rgb24_to_planes_f32(&[], &mut r, &mut g, &mut b),
            Err(SizeError::NotPixelAligned)
        );
        // Not multiple of 3
        let bad = [0u8; 7];
        assert_eq!(
            rgb24_to_planes_f32(&bad, &mut r, &mut g, &mut b),
            Err(SizeError::NotPixelAligned)
        );
        // Plane too short
        let src = [0u8; 24];
        let mut tiny = vec![0.0; 2];
        assert_eq!(
            rgb24_to_planes_f32(&src, &mut tiny, &mut g, &mut b),
            Err(SizeError::PixelCountMismatch)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgb24_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let pixels = 8 * 17;
            let src: alloc::vec::Vec<u8> = (0..pixels * 3).map(|i| (i & 0xFF) as u8).collect();
            let mut r_v = vec![0.0f32; pixels];
            let mut g_v = vec![0.0f32; pixels];
            let mut b_v = vec![0.0f32; pixels];
            let mut r_s = vec![0.0f32; pixels];
            let mut g_s = vec![0.0f32; pixels];
            let mut b_s = vec![0.0f32; pixels];
            x86::rgb24_to_planes_impl_v3(t, &src, &mut r_v, &mut g_v, &mut b_v);
            rgb24_to_planes_impl_scalar(
                ScalarToken::summon().unwrap(),
                &src,
                &mut r_s,
                &mut g_s,
                &mut b_s,
            );
            assert_eq!(r_v, r_s);
            assert_eq!(g_v, g_s);
            assert_eq!(b_v, b_s);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgb48_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let pixels = 8 * 17;
            let src: alloc::vec::Vec<u16> = (0..pixels * 3)
                .map(|i: usize| (i.wrapping_mul(257) & 0xFFFF) as u16)
                .collect();
            let mut r_v = vec![0.0f32; pixels];
            let mut g_v = vec![0.0f32; pixels];
            let mut b_v = vec![0.0f32; pixels];
            let mut r_s = vec![0.0f32; pixels];
            let mut g_s = vec![0.0f32; pixels];
            let mut b_s = vec![0.0f32; pixels];
            x86::rgb48_to_planes_impl_v3(t, &src, &mut r_v, &mut g_v, &mut b_v);
            rgb48_to_planes_impl_scalar(
                ScalarToken::summon().unwrap(),
                &src,
                &mut r_s,
                &mut g_s,
                &mut b_s,
            );
            assert_eq!(r_v, r_s);
            assert_eq!(g_v, g_s);
            assert_eq!(b_v, b_s);
        }
    }

    // ----- f32 RGB / RGBA round-trips ------------------------------------

    #[test]
    fn rgb_f32_round_trip() {
        let pixels = 67;
        let src: alloc::vec::Vec<f32> = (0..pixels * 3)
            .map(|i: usize| i as f32 * 0.5 - 100.0)
            .collect();
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        rgb_f32_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        for i in 0..pixels {
            assert_eq!(r[i], src[i * 3]);
            assert_eq!(g[i], src[i * 3 + 1]);
            assert_eq!(b[i], src[i * 3 + 2]);
        }

        let mut interleaved = vec![0.0f32; pixels * 3];
        planes_f32_to_rgb_f32(&r, &g, &b, &mut interleaved).unwrap();
        assert_eq!(interleaved, src);
    }

    #[test]
    fn rgba_f32_round_trip() {
        let pixels = 51;
        let src: alloc::vec::Vec<f32> = (0..pixels * 4)
            .map(|i: usize| i as f32 * 0.25 + 1.0)
            .collect();
        let mut r = vec![0.0f32; pixels];
        let mut g = vec![0.0f32; pixels];
        let mut b = vec![0.0f32; pixels];
        let mut a = vec![0.0f32; pixels];
        rgba_f32_to_planes_f32(&src, &mut r, &mut g, &mut b, &mut a).unwrap();
        for i in 0..pixels {
            assert_eq!(r[i], src[i * 4]);
            assert_eq!(g[i], src[i * 4 + 1]);
            assert_eq!(b[i], src[i * 4 + 2]);
            assert_eq!(a[i], src[i * 4 + 3]);
        }

        let mut interleaved = vec![0.0f32; pixels * 4];
        planes_f32_to_rgba_f32(&r, &g, &b, &a, &mut interleaved).unwrap();
        assert_eq!(interleaved, src);
    }

    #[test]
    fn f32_errors_rejected() {
        let mut r = vec![0.0; 4];
        let mut g = vec![0.0; 4];
        let mut b = vec![0.0; 4];
        let a = vec![0.0; 4];
        let mut dst = vec![0.0; 16];

        assert_eq!(
            rgb_f32_to_planes_f32(&[], &mut r, &mut g, &mut b),
            Err(SizeError::NotPixelAligned)
        );
        let bad = [0.0f32; 7];
        assert_eq!(
            rgb_f32_to_planes_f32(&bad, &mut r, &mut g, &mut b),
            Err(SizeError::NotPixelAligned)
        );
        let src = [0.0f32; 12]; // 4 pixels of RGB
        let mut tiny = vec![0.0f32; 2];
        assert_eq!(
            rgb_f32_to_planes_f32(&src, &mut tiny, &mut g, &mut b),
            Err(SizeError::PixelCountMismatch)
        );

        // Plane length mismatch on interleave side
        let r_short = vec![0.0; 3];
        assert_eq!(
            planes_f32_to_rgb_f32(&r_short, &g, &b, &mut dst),
            Err(SizeError::NotPixelAligned)
        );
        // dst too small
        let mut tiny_dst = vec![0.0; 5];
        assert_eq!(
            planes_f32_to_rgb_f32(&r, &g, &b, &mut tiny_dst),
            Err(SizeError::PixelCountMismatch)
        );

        // RGBA mismatch
        let a_short = vec![0.0; 3];
        assert_eq!(
            planes_f32_to_rgba_f32(&r, &g, &b, &a_short, &mut dst),
            Err(SizeError::NotPixelAligned)
        );
        let _ = &a;
    }
}
