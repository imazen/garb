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

// --- x86_64 AVX2 chunked SIMD wrappers ----------------------------------
//
// Each `*_impl_v3` chunks the input into 16-pixel groups (calling
// `*_chunk16_*_v3`), drops to chunk-8 / chunk-4 for the tail, then
// finishes with scalar pixel-by-pixel for the final < 4 pixels. The
// per-chunk SIMD functions are `#[rite]` so they fuse into this
// `#[arcane]` region with no `call` instruction at the chunk boundary.

#[cfg(target_arch = "x86_64")]
mod x86_f32 {
    use super::*;

    #[arcane]
    pub(crate) fn rgb_f32_to_planes_impl_v3(
        t: X64V3Token,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let pixels = src.len() / 3;
        let mut p = 0;
        // 16-pixel chunks
        while p + 16 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 48] = src[off_src..off_src + 48].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk16_to_planes_v3(t, chunk);
            r[p..p + 16].copy_from_slice(&rc);
            g[p..p + 16].copy_from_slice(&gc);
            b[p..p + 16].copy_from_slice(&bc);
            p += 16;
        }
        // 8-pixel
        while p + 8 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 24] = src[off_src..off_src + 24].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk8_to_planes_v3(t, chunk);
            r[p..p + 8].copy_from_slice(&rc);
            g[p..p + 8].copy_from_slice(&gc);
            b[p..p + 8].copy_from_slice(&bc);
            p += 8;
        }
        // 4-pixel
        while p + 4 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 12] = src[off_src..off_src + 12].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk4_to_planes_v3(t, chunk);
            r[p..p + 4].copy_from_slice(&rc);
            g[p..p + 4].copy_from_slice(&gc);
            b[p..p + 4].copy_from_slice(&bc);
            p += 4;
        }
        // scalar tail
        while p < pixels {
            r[p] = src[p * 3];
            g[p] = src[p * 3 + 1];
            b[p] = src[p * 3 + 2];
            p += 1;
        }
    }

    #[arcane]
    pub(crate) fn rgba_f32_to_planes_impl_v3(
        t: X64V3Token,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        a: &mut [f32],
    ) {
        let pixels = src.len() / 4;
        let mut p = 0;
        while p + 16 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 64] = src[off_src..off_src + 64].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk16_to_planes_v3(t, chunk);
            r[p..p + 16].copy_from_slice(&rc);
            g[p..p + 16].copy_from_slice(&gc);
            b[p..p + 16].copy_from_slice(&bc);
            a[p..p + 16].copy_from_slice(&ac);
            p += 16;
        }
        while p + 8 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 32] = src[off_src..off_src + 32].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk8_to_planes_v3(t, chunk);
            r[p..p + 8].copy_from_slice(&rc);
            g[p..p + 8].copy_from_slice(&gc);
            b[p..p + 8].copy_from_slice(&bc);
            a[p..p + 8].copy_from_slice(&ac);
            p += 8;
        }
        while p + 4 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 16] = src[off_src..off_src + 16].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk4_to_planes_v3(t, chunk);
            r[p..p + 4].copy_from_slice(&rc);
            g[p..p + 4].copy_from_slice(&gc);
            b[p..p + 4].copy_from_slice(&bc);
            a[p..p + 4].copy_from_slice(&ac);
            p += 4;
        }
        while p < pixels {
            r[p] = src[p * 4];
            g[p] = src[p * 4 + 1];
            b[p] = src[p * 4 + 2];
            a[p] = src[p * 4 + 3];
            p += 1;
        }
    }

    #[arcane]
    pub(crate) fn planes_to_rgb_f32_impl_v3(
        t: X64V3Token,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        dst: &mut [f32],
    ) {
        let pixels = r.len();
        let mut p = 0;
        while p + 16 <= pixels {
            let r_chunk: &[f32; 16] = r[p..p + 16].try_into().unwrap();
            let g_chunk: &[f32; 16] = g[p..p + 16].try_into().unwrap();
            let b_chunk: &[f32; 16] = b[p..p + 16].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk16_v3(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 48].copy_from_slice(&part);
            p += 16;
        }
        while p + 8 <= pixels {
            let r_chunk: &[f32; 8] = r[p..p + 8].try_into().unwrap();
            let g_chunk: &[f32; 8] = g[p..p + 8].try_into().unwrap();
            let b_chunk: &[f32; 8] = b[p..p + 8].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk8_v3(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 24].copy_from_slice(&part);
            p += 8;
        }
        while p + 4 <= pixels {
            let r_chunk: &[f32; 4] = r[p..p + 4].try_into().unwrap();
            let g_chunk: &[f32; 4] = g[p..p + 4].try_into().unwrap();
            let b_chunk: &[f32; 4] = b[p..p + 4].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk4_v3(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 12].copy_from_slice(&part);
            p += 4;
        }
        while p < pixels {
            dst[p * 3] = r[p];
            dst[p * 3 + 1] = g[p];
            dst[p * 3 + 2] = b[p];
            p += 1;
        }
    }

    #[arcane]
    pub(crate) fn planes_to_rgba_f32_impl_v3(
        t: X64V3Token,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        a: &[f32],
        dst: &mut [f32],
    ) {
        let pixels = r.len();
        let mut p = 0;
        while p + 16 <= pixels {
            let r_chunk: &[f32; 16] = r[p..p + 16].try_into().unwrap();
            let g_chunk: &[f32; 16] = g[p..p + 16].try_into().unwrap();
            let b_chunk: &[f32; 16] = b[p..p + 16].try_into().unwrap();
            let a_chunk: &[f32; 16] = a[p..p + 16].try_into().unwrap();
            let part = super::planes_to_rgba_f32_chunk16_v3(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 64].copy_from_slice(&part);
            p += 16;
        }
        while p + 8 <= pixels {
            let r_chunk: &[f32; 8] = r[p..p + 8].try_into().unwrap();
            let g_chunk: &[f32; 8] = g[p..p + 8].try_into().unwrap();
            let b_chunk: &[f32; 8] = b[p..p + 8].try_into().unwrap();
            let a_chunk: &[f32; 8] = a[p..p + 8].try_into().unwrap();
            let part = super::planes_to_rgba_f32_chunk8_v3(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 32].copy_from_slice(&part);
            p += 8;
        }
        while p + 4 <= pixels {
            let r_chunk: &[f32; 4] = r[p..p + 4].try_into().unwrap();
            let g_chunk: &[f32; 4] = g[p..p + 4].try_into().unwrap();
            let b_chunk: &[f32; 4] = b[p..p + 4].try_into().unwrap();
            let a_chunk: &[f32; 4] = a[p..p + 4].try_into().unwrap();
            let part = super::planes_to_rgba_f32_chunk4_v3(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 16].copy_from_slice(&part);
            p += 4;
        }
        while p < pixels {
            dst[p * 4] = r[p];
            dst[p * 4 + 1] = g[p];
            dst[p * 4 + 2] = b[p];
            dst[p * 4 + 3] = a[p];
            p += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
use x86_f32::{
    planes_to_rgb_f32_impl_v3, planes_to_rgba_f32_impl_v3, rgb_f32_to_planes_impl_v3,
    rgba_f32_to_planes_impl_v3,
};

// --- aarch64 NEON chunked SIMD wrappers ----------------------------------
//
// NEON has dedicated hardware structure-load/store instructions for f32:
//   `vld3q_f32` deinterleaves 4 RGB pixels in a single instruction
//   `vld4q_f32` deinterleaves 4 RGBA pixels in a single instruction
//   `vst3q_f32` / `vst4q_f32` interleave on the inverse path
//
// Each `*_impl_neon` chunks into 16 → 8 → 4 → scalar tail; the per-chunk
// `#[rite]` SIMD functions inline into this `#[arcane]` region.

#[cfg(target_arch = "aarch64")]
mod arm_f32 {
    use super::*;

    #[arcane]
    pub(crate) fn rgb_f32_to_planes_impl_neon(
        t: NeonToken,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let pixels = src.len() / 3;
        let mut p = 0;
        while p + 16 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 48] = src[off_src..off_src + 48].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk16_to_planes_neon(t, chunk);
            r[p..p + 16].copy_from_slice(&rc);
            g[p..p + 16].copy_from_slice(&gc);
            b[p..p + 16].copy_from_slice(&bc);
            p += 16;
        }
        while p + 8 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 24] = src[off_src..off_src + 24].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk8_to_planes_neon(t, chunk);
            r[p..p + 8].copy_from_slice(&rc);
            g[p..p + 8].copy_from_slice(&gc);
            b[p..p + 8].copy_from_slice(&bc);
            p += 8;
        }
        while p + 4 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 12] = src[off_src..off_src + 12].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk4_to_planes_neon(t, chunk);
            r[p..p + 4].copy_from_slice(&rc);
            g[p..p + 4].copy_from_slice(&gc);
            b[p..p + 4].copy_from_slice(&bc);
            p += 4;
        }
        while p < pixels {
            r[p] = src[p * 3];
            g[p] = src[p * 3 + 1];
            b[p] = src[p * 3 + 2];
            p += 1;
        }
    }

    #[arcane]
    pub(crate) fn rgba_f32_to_planes_impl_neon(
        t: NeonToken,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        a: &mut [f32],
    ) {
        let pixels = src.len() / 4;
        let mut p = 0;
        while p + 16 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 64] = src[off_src..off_src + 64].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk16_to_planes_neon(t, chunk);
            r[p..p + 16].copy_from_slice(&rc);
            g[p..p + 16].copy_from_slice(&gc);
            b[p..p + 16].copy_from_slice(&bc);
            a[p..p + 16].copy_from_slice(&ac);
            p += 16;
        }
        while p + 8 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 32] = src[off_src..off_src + 32].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk8_to_planes_neon(t, chunk);
            r[p..p + 8].copy_from_slice(&rc);
            g[p..p + 8].copy_from_slice(&gc);
            b[p..p + 8].copy_from_slice(&bc);
            a[p..p + 8].copy_from_slice(&ac);
            p += 8;
        }
        while p + 4 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 16] = src[off_src..off_src + 16].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk4_to_planes_neon(t, chunk);
            r[p..p + 4].copy_from_slice(&rc);
            g[p..p + 4].copy_from_slice(&gc);
            b[p..p + 4].copy_from_slice(&bc);
            a[p..p + 4].copy_from_slice(&ac);
            p += 4;
        }
        while p < pixels {
            r[p] = src[p * 4];
            g[p] = src[p * 4 + 1];
            b[p] = src[p * 4 + 2];
            a[p] = src[p * 4 + 3];
            p += 1;
        }
    }

    #[arcane]
    pub(crate) fn planes_to_rgb_f32_impl_neon(
        t: NeonToken,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        dst: &mut [f32],
    ) {
        let pixels = r.len();
        let mut p = 0;
        while p + 16 <= pixels {
            let r_chunk: &[f32; 16] = r[p..p + 16].try_into().unwrap();
            let g_chunk: &[f32; 16] = g[p..p + 16].try_into().unwrap();
            let b_chunk: &[f32; 16] = b[p..p + 16].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk16_neon(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 48].copy_from_slice(&part);
            p += 16;
        }
        while p + 8 <= pixels {
            let r_chunk: &[f32; 8] = r[p..p + 8].try_into().unwrap();
            let g_chunk: &[f32; 8] = g[p..p + 8].try_into().unwrap();
            let b_chunk: &[f32; 8] = b[p..p + 8].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk8_neon(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 24].copy_from_slice(&part);
            p += 8;
        }
        while p + 4 <= pixels {
            let r_chunk: &[f32; 4] = r[p..p + 4].try_into().unwrap();
            let g_chunk: &[f32; 4] = g[p..p + 4].try_into().unwrap();
            let b_chunk: &[f32; 4] = b[p..p + 4].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk4_neon(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 12].copy_from_slice(&part);
            p += 4;
        }
        while p < pixels {
            dst[p * 3] = r[p];
            dst[p * 3 + 1] = g[p];
            dst[p * 3 + 2] = b[p];
            p += 1;
        }
    }

    #[arcane]
    pub(crate) fn planes_to_rgba_f32_impl_neon(
        t: NeonToken,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        a: &[f32],
        dst: &mut [f32],
    ) {
        let pixels = r.len();
        let mut p = 0;
        while p + 16 <= pixels {
            let r_chunk: &[f32; 16] = r[p..p + 16].try_into().unwrap();
            let g_chunk: &[f32; 16] = g[p..p + 16].try_into().unwrap();
            let b_chunk: &[f32; 16] = b[p..p + 16].try_into().unwrap();
            let a_chunk: &[f32; 16] = a[p..p + 16].try_into().unwrap();
            let part =
                super::planes_to_rgba_f32_chunk16_neon(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 64].copy_from_slice(&part);
            p += 16;
        }
        while p + 8 <= pixels {
            let r_chunk: &[f32; 8] = r[p..p + 8].try_into().unwrap();
            let g_chunk: &[f32; 8] = g[p..p + 8].try_into().unwrap();
            let b_chunk: &[f32; 8] = b[p..p + 8].try_into().unwrap();
            let a_chunk: &[f32; 8] = a[p..p + 8].try_into().unwrap();
            let part = super::planes_to_rgba_f32_chunk8_neon(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 32].copy_from_slice(&part);
            p += 8;
        }
        while p + 4 <= pixels {
            let r_chunk: &[f32; 4] = r[p..p + 4].try_into().unwrap();
            let g_chunk: &[f32; 4] = g[p..p + 4].try_into().unwrap();
            let b_chunk: &[f32; 4] = b[p..p + 4].try_into().unwrap();
            let a_chunk: &[f32; 4] = a[p..p + 4].try_into().unwrap();
            let part = super::planes_to_rgba_f32_chunk4_neon(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 16].copy_from_slice(&part);
            p += 4;
        }
        while p < pixels {
            dst[p * 4] = r[p];
            dst[p * 4 + 1] = g[p];
            dst[p * 4 + 2] = b[p];
            dst[p * 4 + 3] = a[p];
            p += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
use arm_f32::{
    planes_to_rgb_f32_impl_neon, planes_to_rgba_f32_impl_neon, rgb_f32_to_planes_impl_neon,
    rgba_f32_to_planes_impl_neon,
};

// --- wasm32 SIMD128 chunked SIMD wrappers --------------------------------
//
// Mirrors the v3 / neon dispatchers above; uses the per-chunk
// `*_wasm128` `#[rite]` functions. `incant!` already accepts a
// `wasm128` tier so adding it as a slice-level path is purely a
// performance opt-in for callers on wasm32 with `+simd128`.

#[cfg(target_arch = "wasm32")]
mod wasm_f32 {
    use super::*;

    #[archmage::arcane]
    pub(crate) fn rgb_f32_to_planes_impl_wasm128(
        t: Wasm128Token,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let pixels = src.len() / 3;
        let mut p = 0;
        while p + 16 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 48] = src[off_src..off_src + 48].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk16_to_planes_wasm128(t, chunk);
            r[p..p + 16].copy_from_slice(&rc);
            g[p..p + 16].copy_from_slice(&gc);
            b[p..p + 16].copy_from_slice(&bc);
            p += 16;
        }
        while p + 8 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 24] = src[off_src..off_src + 24].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk8_to_planes_wasm128(t, chunk);
            r[p..p + 8].copy_from_slice(&rc);
            g[p..p + 8].copy_from_slice(&gc);
            b[p..p + 8].copy_from_slice(&bc);
            p += 8;
        }
        while p + 4 <= pixels {
            let off_src = p * 3;
            let chunk: &[f32; 12] = src[off_src..off_src + 12].try_into().unwrap();
            let (rc, gc, bc) = super::rgb_f32_chunk4_to_planes_wasm128(t, chunk);
            r[p..p + 4].copy_from_slice(&rc);
            g[p..p + 4].copy_from_slice(&gc);
            b[p..p + 4].copy_from_slice(&bc);
            p += 4;
        }
        while p < pixels {
            r[p] = src[p * 3];
            g[p] = src[p * 3 + 1];
            b[p] = src[p * 3 + 2];
            p += 1;
        }
    }

    #[archmage::arcane]
    pub(crate) fn rgba_f32_to_planes_impl_wasm128(
        t: Wasm128Token,
        src: &[f32],
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        a: &mut [f32],
    ) {
        let pixels = src.len() / 4;
        let mut p = 0;
        while p + 16 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 64] = src[off_src..off_src + 64].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk16_to_planes_wasm128(t, chunk);
            r[p..p + 16].copy_from_slice(&rc);
            g[p..p + 16].copy_from_slice(&gc);
            b[p..p + 16].copy_from_slice(&bc);
            a[p..p + 16].copy_from_slice(&ac);
            p += 16;
        }
        while p + 8 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 32] = src[off_src..off_src + 32].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk8_to_planes_wasm128(t, chunk);
            r[p..p + 8].copy_from_slice(&rc);
            g[p..p + 8].copy_from_slice(&gc);
            b[p..p + 8].copy_from_slice(&bc);
            a[p..p + 8].copy_from_slice(&ac);
            p += 8;
        }
        while p + 4 <= pixels {
            let off_src = p * 4;
            let chunk: &[f32; 16] = src[off_src..off_src + 16].try_into().unwrap();
            let (rc, gc, bc, ac) = super::rgba_f32_chunk4_to_planes_wasm128(t, chunk);
            r[p..p + 4].copy_from_slice(&rc);
            g[p..p + 4].copy_from_slice(&gc);
            b[p..p + 4].copy_from_slice(&bc);
            a[p..p + 4].copy_from_slice(&ac);
            p += 4;
        }
        while p < pixels {
            r[p] = src[p * 4];
            g[p] = src[p * 4 + 1];
            b[p] = src[p * 4 + 2];
            a[p] = src[p * 4 + 3];
            p += 1;
        }
    }

    #[archmage::arcane]
    pub(crate) fn planes_to_rgb_f32_impl_wasm128(
        t: Wasm128Token,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        dst: &mut [f32],
    ) {
        let pixels = r.len();
        let mut p = 0;
        while p + 16 <= pixels {
            let r_chunk: &[f32; 16] = r[p..p + 16].try_into().unwrap();
            let g_chunk: &[f32; 16] = g[p..p + 16].try_into().unwrap();
            let b_chunk: &[f32; 16] = b[p..p + 16].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk16_wasm128(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 48].copy_from_slice(&part);
            p += 16;
        }
        while p + 8 <= pixels {
            let r_chunk: &[f32; 8] = r[p..p + 8].try_into().unwrap();
            let g_chunk: &[f32; 8] = g[p..p + 8].try_into().unwrap();
            let b_chunk: &[f32; 8] = b[p..p + 8].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk8_wasm128(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 24].copy_from_slice(&part);
            p += 8;
        }
        while p + 4 <= pixels {
            let r_chunk: &[f32; 4] = r[p..p + 4].try_into().unwrap();
            let g_chunk: &[f32; 4] = g[p..p + 4].try_into().unwrap();
            let b_chunk: &[f32; 4] = b[p..p + 4].try_into().unwrap();
            let part = super::planes_to_rgb_f32_chunk4_wasm128(t, r_chunk, g_chunk, b_chunk);
            let off = p * 3;
            dst[off..off + 12].copy_from_slice(&part);
            p += 4;
        }
        while p < pixels {
            dst[p * 3] = r[p];
            dst[p * 3 + 1] = g[p];
            dst[p * 3 + 2] = b[p];
            p += 1;
        }
    }

    #[archmage::arcane]
    pub(crate) fn planes_to_rgba_f32_impl_wasm128(
        t: Wasm128Token,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        a: &[f32],
        dst: &mut [f32],
    ) {
        let pixels = r.len();
        let mut p = 0;
        while p + 16 <= pixels {
            let r_chunk: &[f32; 16] = r[p..p + 16].try_into().unwrap();
            let g_chunk: &[f32; 16] = g[p..p + 16].try_into().unwrap();
            let b_chunk: &[f32; 16] = b[p..p + 16].try_into().unwrap();
            let a_chunk: &[f32; 16] = a[p..p + 16].try_into().unwrap();
            let part =
                super::planes_to_rgba_f32_chunk16_wasm128(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 64].copy_from_slice(&part);
            p += 16;
        }
        while p + 8 <= pixels {
            let r_chunk: &[f32; 8] = r[p..p + 8].try_into().unwrap();
            let g_chunk: &[f32; 8] = g[p..p + 8].try_into().unwrap();
            let b_chunk: &[f32; 8] = b[p..p + 8].try_into().unwrap();
            let a_chunk: &[f32; 8] = a[p..p + 8].try_into().unwrap();
            let part =
                super::planes_to_rgba_f32_chunk8_wasm128(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 32].copy_from_slice(&part);
            p += 8;
        }
        while p + 4 <= pixels {
            let r_chunk: &[f32; 4] = r[p..p + 4].try_into().unwrap();
            let g_chunk: &[f32; 4] = g[p..p + 4].try_into().unwrap();
            let b_chunk: &[f32; 4] = b[p..p + 4].try_into().unwrap();
            let a_chunk: &[f32; 4] = a[p..p + 4].try_into().unwrap();
            let part =
                super::planes_to_rgba_f32_chunk4_wasm128(t, r_chunk, g_chunk, b_chunk, a_chunk);
            let off = p * 4;
            dst[off..off + 16].copy_from_slice(&part);
            p += 4;
        }
        while p < pixels {
            dst[p * 4] = r[p];
            dst[p * 4 + 1] = g[p];
            dst[p * 4 + 2] = b[p];
            dst[p * 4 + 3] = a[p];
            p += 1;
        }
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_f32::{
    planes_to_rgb_f32_impl_wasm128, planes_to_rgba_f32_impl_wasm128,
    rgb_f32_to_planes_impl_wasm128, rgba_f32_to_planes_impl_wasm128,
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
    incant!(
        rgb_f32_to_planes_impl(src, r, g, b),
        [v3, neon, wasm128, scalar]
    );
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
    incant!(
        rgba_f32_to_planes_impl(src, r, g, b, a),
        [v3, neon, wasm128, scalar]
    );
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
    incant!(
        planes_to_rgb_f32_impl(r, g, b, dst),
        [v3, neon, wasm128, scalar]
    );
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
    incant!(
        planes_to_rgba_f32_impl(r, g, b, a, dst),
        [v3, neon, wasm128, scalar]
    );
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
// Chunk-level f32 RGB / RGBA deinterleave + interleave primitives
// ---------------------------------------------------------------------------
// Mirrors `rgb24_chunk8_to_planes_scalar` at f32 input, across {4, 8, 16}
// pixel chunk widths × {RGB, RGBA} × {deinterleave, interleave}.
//
// Three flavors per shape:
//   `*_scalar`  — fixed-array literal; LLVM is free to autovectorize but
//                 isn't required to. Always available.
//   `*_v3`      — x86_64 AVX2 hand-rolled shuffles. Caller passes an
//                 `X64V3Token` (proves AVX2 available).
//   `*_neon`    — aarch64 NEON `vld3q_f32` / `vld4q_f32` (deinterleave) and
//                 `vst3q_f32` / `vst4q_f32` (interleave). One hardware
//                 structure-load per 4-pixel chunk.
//   `*_wasm128` — wasm32 SIMD128 shuffles via `i32x4_shuffle`-style ops.
//
// `#[rite]` on the per-arch variants — they inline into the caller's
// target_feature region so chunked loops fuse with no per-call overhead.
//
// The bare-name aliases (e.g. `rgb_f32_chunk4_to_planes`) point at
// `*_scalar` for backward compatibility with PR #5's published API.
// ===========================================================================

// --- Chunk-of-4 deinterleave / interleave (scalar) -----------------------

/// RGB f32 deinterleave, chunk-of-4-pixels (scalar). Loads 12 f32 values
/// from interleaved RGBRGB... layout and returns three planar `[f32; 4]`
/// arrays. The chunk-4 size matches the natural NEON / WASM128 vector lane
/// count and feeds directly into the `vld3q_f32` hardware structure-load.
#[inline]
pub fn rgb_f32_chunk4_to_planes_scalar(chunk: &[f32; 12]) -> ([f32; 4], [f32; 4], [f32; 4]) {
    (
        [chunk[0], chunk[3], chunk[6], chunk[9]],
        [chunk[1], chunk[4], chunk[7], chunk[10]],
        [chunk[2], chunk[5], chunk[8], chunk[11]],
    )
}

/// RGBA f32 deinterleave, chunk-of-4-pixels (scalar). Loads 16 f32 values
/// from interleaved RGBARGBA... layout and returns four planar `[f32; 4]`
/// arrays.
#[inline]
pub fn rgba_f32_chunk4_to_planes_scalar(
    chunk: &[f32; 16],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    (
        [chunk[0], chunk[4], chunk[8], chunk[12]],
        [chunk[1], chunk[5], chunk[9], chunk[13]],
        [chunk[2], chunk[6], chunk[10], chunk[14]],
        [chunk[3], chunk[7], chunk[11], chunk[15]],
    )
}

/// RGB f32 interleave, chunk-of-4-pixels (scalar). Stores three planar
/// `[f32; 4]` arrays into 12 f32 in interleaved RGBRGB... layout.
#[inline]
pub fn planes_to_rgb_f32_chunk4_scalar(r: &[f32; 4], g: &[f32; 4], b: &[f32; 4]) -> [f32; 12] {
    [
        r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2], r[3], g[3], b[3],
    ]
}

/// RGBA f32 interleave, chunk-of-4-pixels (scalar). Stores four planar
/// `[f32; 4]` arrays into 16 f32 in interleaved RGBARGBA... layout.
#[inline]
pub fn planes_to_rgba_f32_chunk4_scalar(
    r: &[f32; 4],
    g: &[f32; 4],
    b: &[f32; 4],
    a: &[f32; 4],
) -> [f32; 16] {
    [
        r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], r[3], g[3], b[3],
        a[3],
    ]
}

// --- Chunk-of-8 deinterleave / interleave (scalar) -----------------------

/// RGB f32 deinterleave, chunk-of-8-pixels (scalar). Loads 24 f32 values
/// from interleaved RGBRGB... layout and returns three planar `[f32; 8]`
/// arrays. Pairs naturally with AVX2 (`__m256` = 8×f32).
#[inline]
pub fn rgb_f32_chunk8_to_planes_scalar(chunk: &[f32; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    (
        [
            chunk[0], chunk[3], chunk[6], chunk[9], chunk[12], chunk[15], chunk[18], chunk[21],
        ],
        [
            chunk[1], chunk[4], chunk[7], chunk[10], chunk[13], chunk[16], chunk[19], chunk[22],
        ],
        [
            chunk[2], chunk[5], chunk[8], chunk[11], chunk[14], chunk[17], chunk[20], chunk[23],
        ],
    )
}

/// RGBA f32 deinterleave, chunk-of-8-pixels (scalar). Loads 32 f32 values
/// from interleaved RGBARGBA... layout and returns four planar `[f32; 8]`
/// arrays.
#[inline]
pub fn rgba_f32_chunk8_to_planes_scalar(
    chunk: &[f32; 32],
) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
    (
        [
            chunk[0], chunk[4], chunk[8], chunk[12], chunk[16], chunk[20], chunk[24], chunk[28],
        ],
        [
            chunk[1], chunk[5], chunk[9], chunk[13], chunk[17], chunk[21], chunk[25], chunk[29],
        ],
        [
            chunk[2], chunk[6], chunk[10], chunk[14], chunk[18], chunk[22], chunk[26], chunk[30],
        ],
        [
            chunk[3], chunk[7], chunk[11], chunk[15], chunk[19], chunk[23], chunk[27], chunk[31],
        ],
    )
}

/// RGB f32 interleave, chunk-of-8-pixels (scalar). Stores three planar
/// `[f32; 8]` arrays into 24 f32 in interleaved RGBRGB... layout.
#[inline]
pub fn planes_to_rgb_f32_chunk8_scalar(r: &[f32; 8], g: &[f32; 8], b: &[f32; 8]) -> [f32; 24] {
    [
        r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2], r[3], g[3], b[3], r[4], g[4], b[4],
        r[5], g[5], b[5], r[6], g[6], b[6], r[7], g[7], b[7],
    ]
}

/// RGBA f32 interleave, chunk-of-8-pixels (scalar). Stores four planar
/// `[f32; 8]` arrays into 32 f32 in interleaved RGBARGBA... layout.
#[inline]
pub fn planes_to_rgba_f32_chunk8_scalar(
    r: &[f32; 8],
    g: &[f32; 8],
    b: &[f32; 8],
    a: &[f32; 8],
) -> [f32; 32] {
    [
        r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], r[3], g[3], b[3],
        a[3], r[4], g[4], b[4], a[4], r[5], g[5], b[5], a[5], r[6], g[6], b[6], a[6], r[7], g[7],
        b[7], a[7],
    ]
}

// --- Chunk-of-16 deinterleave / interleave (scalar) ----------------------

/// RGB f32 deinterleave, chunk-of-16-pixels (scalar). Loads 48 f32 values
/// from interleaved RGBRGB... layout and returns three planar `[f32; 16]`
/// arrays.
#[inline]
pub fn rgb_f32_chunk16_to_planes_scalar(chunk: &[f32; 48]) -> ([f32; 16], [f32; 16], [f32; 16]) {
    let mut r = [0.0f32; 16];
    let mut g = [0.0f32; 16];
    let mut b = [0.0f32; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = chunk[i * 3];
        g[i] = chunk[i * 3 + 1];
        b[i] = chunk[i * 3 + 2];
        i += 1;
    }
    (r, g, b)
}

/// RGBA f32 deinterleave, chunk-of-16-pixels (scalar). Loads 64 f32 values
/// from interleaved RGBARGBA... layout and returns four planar `[f32; 16]`
/// arrays.
#[inline]
pub fn rgba_f32_chunk16_to_planes_scalar(
    chunk: &[f32; 64],
) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
    let mut r = [0.0f32; 16];
    let mut g = [0.0f32; 16];
    let mut b = [0.0f32; 16];
    let mut a = [0.0f32; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = chunk[i * 4];
        g[i] = chunk[i * 4 + 1];
        b[i] = chunk[i * 4 + 2];
        a[i] = chunk[i * 4 + 3];
        i += 1;
    }
    (r, g, b, a)
}

/// RGB f32 interleave, chunk-of-16-pixels (scalar). Stores three planar
/// `[f32; 16]` arrays into 48 f32 in interleaved RGBRGB... layout.
#[inline]
pub fn planes_to_rgb_f32_chunk16_scalar(r: &[f32; 16], g: &[f32; 16], b: &[f32; 16]) -> [f32; 48] {
    let mut out = [0.0f32; 48];
    let mut i = 0;
    while i < 16 {
        out[i * 3] = r[i];
        out[i * 3 + 1] = g[i];
        out[i * 3 + 2] = b[i];
        i += 1;
    }
    out
}

/// RGBA f32 interleave, chunk-of-16-pixels (scalar). Stores four planar
/// `[f32; 16]` arrays into 64 f32 in interleaved RGBARGBA... layout.
#[inline]
pub fn planes_to_rgba_f32_chunk16_scalar(
    r: &[f32; 16],
    g: &[f32; 16],
    b: &[f32; 16],
    a: &[f32; 16],
) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    let mut i = 0;
    while i < 16 {
        out[i * 4] = r[i];
        out[i * 4 + 1] = g[i];
        out[i * 4 + 2] = b[i];
        out[i * 4 + 3] = a[i];
        i += 1;
    }
    out
}

// --- Bare-name aliases (point at `*_scalar` for backward compat) ---------
//
// PR #5 published these without a suffix. We keep the no-suffix names as
// thin wrappers around the scalar variant so existing callers compile
// untouched while the per-arch SIMD specializations live behind `_v3`,
// `_neon`, `_wasm128` suffixes.

/// Backward-compat alias for [`rgb_f32_chunk4_to_planes_scalar`].
#[inline]
pub fn rgb_f32_chunk4_to_planes(chunk: &[f32; 12]) -> ([f32; 4], [f32; 4], [f32; 4]) {
    rgb_f32_chunk4_to_planes_scalar(chunk)
}

/// Backward-compat alias for [`rgba_f32_chunk4_to_planes_scalar`].
#[inline]
pub fn rgba_f32_chunk4_to_planes(chunk: &[f32; 16]) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    rgba_f32_chunk4_to_planes_scalar(chunk)
}

/// Backward-compat alias for [`planes_to_rgb_f32_chunk4_scalar`].
#[inline]
pub fn planes_to_rgb_f32_chunk4(r: &[f32; 4], g: &[f32; 4], b: &[f32; 4]) -> [f32; 12] {
    planes_to_rgb_f32_chunk4_scalar(r, g, b)
}

/// Backward-compat alias for [`planes_to_rgba_f32_chunk4_scalar`].
#[inline]
pub fn planes_to_rgba_f32_chunk4(
    r: &[f32; 4],
    g: &[f32; 4],
    b: &[f32; 4],
    a: &[f32; 4],
) -> [f32; 16] {
    planes_to_rgba_f32_chunk4_scalar(r, g, b, a)
}

/// Backward-compat alias for [`rgb_f32_chunk8_to_planes_scalar`].
#[inline]
pub fn rgb_f32_chunk8_to_planes(chunk: &[f32; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    rgb_f32_chunk8_to_planes_scalar(chunk)
}

/// Backward-compat alias for [`rgba_f32_chunk8_to_planes_scalar`].
#[inline]
pub fn rgba_f32_chunk8_to_planes(chunk: &[f32; 32]) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
    rgba_f32_chunk8_to_planes_scalar(chunk)
}

/// Backward-compat alias for [`planes_to_rgb_f32_chunk8_scalar`].
#[inline]
pub fn planes_to_rgb_f32_chunk8(r: &[f32; 8], g: &[f32; 8], b: &[f32; 8]) -> [f32; 24] {
    planes_to_rgb_f32_chunk8_scalar(r, g, b)
}

/// Backward-compat alias for [`planes_to_rgba_f32_chunk8_scalar`].
#[inline]
pub fn planes_to_rgba_f32_chunk8(
    r: &[f32; 8],
    g: &[f32; 8],
    b: &[f32; 8],
    a: &[f32; 8],
) -> [f32; 32] {
    planes_to_rgba_f32_chunk8_scalar(r, g, b, a)
}

/// Backward-compat alias for [`rgb_f32_chunk16_to_planes_scalar`].
#[inline]
pub fn rgb_f32_chunk16_to_planes(chunk: &[f32; 48]) -> ([f32; 16], [f32; 16], [f32; 16]) {
    rgb_f32_chunk16_to_planes_scalar(chunk)
}

/// Backward-compat alias for [`rgba_f32_chunk16_to_planes_scalar`].
#[inline]
pub fn rgba_f32_chunk16_to_planes(
    chunk: &[f32; 64],
) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
    rgba_f32_chunk16_to_planes_scalar(chunk)
}

/// Backward-compat alias for [`planes_to_rgb_f32_chunk16_scalar`].
#[inline]
pub fn planes_to_rgb_f32_chunk16(r: &[f32; 16], g: &[f32; 16], b: &[f32; 16]) -> [f32; 48] {
    planes_to_rgb_f32_chunk16_scalar(r, g, b)
}

/// Backward-compat alias for [`planes_to_rgba_f32_chunk16_scalar`].
#[inline]
pub fn planes_to_rgba_f32_chunk16(
    r: &[f32; 16],
    g: &[f32; 16],
    b: &[f32; 16],
    a: &[f32; 16],
) -> [f32; 64] {
    planes_to_rgba_f32_chunk16_scalar(r, g, b, a)
}

// ===========================================================================
// Per-arch chunk-level SIMD specializations
// ---------------------------------------------------------------------------
// All `#[rite]` so the body fuses into the caller's `#[arcane]` /
// `#[target_feature]` region — no `call` / `b` instruction at the use site.
// Re-exported from the parent module under `*_v3` / `*_neon` / `*_wasm128`
// names so callers from inside their own SIMD region call the right one
// without any per-chunk dispatch overhead.
// ===========================================================================

#[cfg(target_arch = "aarch64")]
mod arm_f32_chunks {
    use super::*;

    // RGB chunk-4: vld3q_f32 hardware structure-load → 3×float32x4.
    #[rite]
    pub fn rgb_f32_chunk4_to_planes_neon(
        _t: NeonToken,
        chunk: &[f32; 12],
    ) -> ([f32; 4], [f32; 4], [f32; 4]) {
        let float32x4x3_t(r, g, b) = vld3q_f32(chunk);
        let mut r_out = [0.0f32; 4];
        let mut g_out = [0.0f32; 4];
        let mut b_out = [0.0f32; 4];
        vst1q_f32(&mut r_out, r);
        vst1q_f32(&mut g_out, g);
        vst1q_f32(&mut b_out, b);
        (r_out, g_out, b_out)
    }

    // RGBA chunk-4: vld4q_f32 → 4×float32x4.
    #[rite]
    pub fn rgba_f32_chunk4_to_planes_neon(
        _t: NeonToken,
        chunk: &[f32; 16],
    ) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        let float32x4x4_t(r, g, b, a) = vld4q_f32(chunk);
        let mut r_out = [0.0f32; 4];
        let mut g_out = [0.0f32; 4];
        let mut b_out = [0.0f32; 4];
        let mut a_out = [0.0f32; 4];
        vst1q_f32(&mut r_out, r);
        vst1q_f32(&mut g_out, g);
        vst1q_f32(&mut b_out, b);
        vst1q_f32(&mut a_out, a);
        (r_out, g_out, b_out, a_out)
    }

    // RGB chunk-4 interleave: vst3q_f32.
    #[rite]
    pub fn planes_to_rgb_f32_chunk4_neon(
        _t: NeonToken,
        r: &[f32; 4],
        g: &[f32; 4],
        b: &[f32; 4],
    ) -> [f32; 12] {
        let r_v = vld1q_f32(r);
        let g_v = vld1q_f32(g);
        let b_v = vld1q_f32(b);
        let mut out = [0.0f32; 12];
        vst3q_f32(&mut out, float32x4x3_t(r_v, g_v, b_v));
        out
    }

    // RGBA chunk-4 interleave: vst4q_f32.
    #[rite]
    pub fn planes_to_rgba_f32_chunk4_neon(
        _t: NeonToken,
        r: &[f32; 4],
        g: &[f32; 4],
        b: &[f32; 4],
        a: &[f32; 4],
    ) -> [f32; 16] {
        let r_v = vld1q_f32(r);
        let g_v = vld1q_f32(g);
        let b_v = vld1q_f32(b);
        let a_v = vld1q_f32(a);
        let mut out = [0.0f32; 16];
        vst4q_f32(&mut out, float32x4x4_t(r_v, g_v, b_v, a_v));
        out
    }

    // RGB chunk-8: 2 × vld3q_f32 (2× 4-pixel chunks).
    #[rite]
    pub fn rgb_f32_chunk8_to_planes_neon(
        _t: NeonToken,
        chunk: &[f32; 24],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let lo: &[f32; 12] = chunk[0..12].try_into().unwrap();
        let hi: &[f32; 12] = chunk[12..24].try_into().unwrap();
        let float32x4x3_t(r0, g0, b0) = vld3q_f32(lo);
        let float32x4x3_t(r1, g1, b1) = vld3q_f32(hi);
        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        let r_split: &mut [f32; 8] = &mut r_out;
        let g_split: &mut [f32; 8] = &mut g_out;
        let b_split: &mut [f32; 8] = &mut b_out;
        let (r_lo, r_hi) = r_split.split_at_mut(4);
        let (g_lo, g_hi) = g_split.split_at_mut(4);
        let (b_lo, b_hi) = b_split.split_at_mut(4);
        vst1q_f32(<&mut [f32; 4]>::try_from(r_lo).unwrap(), r0);
        vst1q_f32(<&mut [f32; 4]>::try_from(r_hi).unwrap(), r1);
        vst1q_f32(<&mut [f32; 4]>::try_from(g_lo).unwrap(), g0);
        vst1q_f32(<&mut [f32; 4]>::try_from(g_hi).unwrap(), g1);
        vst1q_f32(<&mut [f32; 4]>::try_from(b_lo).unwrap(), b0);
        vst1q_f32(<&mut [f32; 4]>::try_from(b_hi).unwrap(), b1);
        (r_out, g_out, b_out)
    }

    // RGBA chunk-8: 2 × vld4q_f32.
    #[rite]
    pub fn rgba_f32_chunk8_to_planes_neon(
        _t: NeonToken,
        chunk: &[f32; 32],
    ) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
        let lo: &[f32; 16] = chunk[0..16].try_into().unwrap();
        let hi: &[f32; 16] = chunk[16..32].try_into().unwrap();
        let float32x4x4_t(r0, g0, b0, a0) = vld4q_f32(lo);
        let float32x4x4_t(r1, g1, b1, a1) = vld4q_f32(hi);
        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        let mut a_out = [0.0f32; 8];
        let (r_lo, r_hi) = r_out.split_at_mut(4);
        let (g_lo, g_hi) = g_out.split_at_mut(4);
        let (b_lo, b_hi) = b_out.split_at_mut(4);
        let (a_lo, a_hi) = a_out.split_at_mut(4);
        vst1q_f32(<&mut [f32; 4]>::try_from(r_lo).unwrap(), r0);
        vst1q_f32(<&mut [f32; 4]>::try_from(r_hi).unwrap(), r1);
        vst1q_f32(<&mut [f32; 4]>::try_from(g_lo).unwrap(), g0);
        vst1q_f32(<&mut [f32; 4]>::try_from(g_hi).unwrap(), g1);
        vst1q_f32(<&mut [f32; 4]>::try_from(b_lo).unwrap(), b0);
        vst1q_f32(<&mut [f32; 4]>::try_from(b_hi).unwrap(), b1);
        vst1q_f32(<&mut [f32; 4]>::try_from(a_lo).unwrap(), a0);
        vst1q_f32(<&mut [f32; 4]>::try_from(a_hi).unwrap(), a1);
        (r_out, g_out, b_out, a_out)
    }

    // RGB chunk-8 interleave: 2 × vst3q_f32.
    #[rite]
    pub fn planes_to_rgb_f32_chunk8_neon(
        _t: NeonToken,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
    ) -> [f32; 24] {
        let r_lo: &[f32; 4] = r[0..4].try_into().unwrap();
        let r_hi: &[f32; 4] = r[4..8].try_into().unwrap();
        let g_lo: &[f32; 4] = g[0..4].try_into().unwrap();
        let g_hi: &[f32; 4] = g[4..8].try_into().unwrap();
        let b_lo: &[f32; 4] = b[0..4].try_into().unwrap();
        let b_hi: &[f32; 4] = b[4..8].try_into().unwrap();
        let mut out = [0.0f32; 24];
        let (out_lo, out_hi) = out.split_at_mut(12);
        vst3q_f32(
            <&mut [f32; 12]>::try_from(out_lo).unwrap(),
            float32x4x3_t(vld1q_f32(r_lo), vld1q_f32(g_lo), vld1q_f32(b_lo)),
        );
        vst3q_f32(
            <&mut [f32; 12]>::try_from(out_hi).unwrap(),
            float32x4x3_t(vld1q_f32(r_hi), vld1q_f32(g_hi), vld1q_f32(b_hi)),
        );
        out
    }

    // RGBA chunk-8 interleave: 2 × vst4q_f32.
    #[rite]
    pub fn planes_to_rgba_f32_chunk8_neon(
        _t: NeonToken,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
        a: &[f32; 8],
    ) -> [f32; 32] {
        let r_lo: &[f32; 4] = r[0..4].try_into().unwrap();
        let r_hi: &[f32; 4] = r[4..8].try_into().unwrap();
        let g_lo: &[f32; 4] = g[0..4].try_into().unwrap();
        let g_hi: &[f32; 4] = g[4..8].try_into().unwrap();
        let b_lo: &[f32; 4] = b[0..4].try_into().unwrap();
        let b_hi: &[f32; 4] = b[4..8].try_into().unwrap();
        let a_lo: &[f32; 4] = a[0..4].try_into().unwrap();
        let a_hi: &[f32; 4] = a[4..8].try_into().unwrap();
        let mut out = [0.0f32; 32];
        let (out_lo, out_hi) = out.split_at_mut(16);
        vst4q_f32(
            <&mut [f32; 16]>::try_from(out_lo).unwrap(),
            float32x4x4_t(
                vld1q_f32(r_lo),
                vld1q_f32(g_lo),
                vld1q_f32(b_lo),
                vld1q_f32(a_lo),
            ),
        );
        vst4q_f32(
            <&mut [f32; 16]>::try_from(out_hi).unwrap(),
            float32x4x4_t(
                vld1q_f32(r_hi),
                vld1q_f32(g_hi),
                vld1q_f32(b_hi),
                vld1q_f32(a_hi),
            ),
        );
        out
    }

    // RGB chunk-16: 4 × vld3q_f32.
    #[rite]
    pub fn rgb_f32_chunk16_to_planes_neon(
        _t: NeonToken,
        chunk: &[f32; 48],
    ) -> ([f32; 16], [f32; 16], [f32; 16]) {
        let mut r_out = [0.0f32; 16];
        let mut g_out = [0.0f32; 16];
        let mut b_out = [0.0f32; 16];
        let mut k = 0;
        while k < 4 {
            let in_chunk: &[f32; 12] = chunk[k * 12..k * 12 + 12].try_into().unwrap();
            let float32x4x3_t(rv, gv, bv) = vld3q_f32(in_chunk);
            let r_slice: &mut [f32; 4] = (&mut r_out[k * 4..k * 4 + 4]).try_into().unwrap();
            let g_slice: &mut [f32; 4] = (&mut g_out[k * 4..k * 4 + 4]).try_into().unwrap();
            let b_slice: &mut [f32; 4] = (&mut b_out[k * 4..k * 4 + 4]).try_into().unwrap();
            vst1q_f32(r_slice, rv);
            vst1q_f32(g_slice, gv);
            vst1q_f32(b_slice, bv);
            k += 1;
        }
        (r_out, g_out, b_out)
    }

    // RGBA chunk-16: 4 × vld4q_f32.
    #[rite]
    pub fn rgba_f32_chunk16_to_planes_neon(
        _t: NeonToken,
        chunk: &[f32; 64],
    ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
        let mut r_out = [0.0f32; 16];
        let mut g_out = [0.0f32; 16];
        let mut b_out = [0.0f32; 16];
        let mut a_out = [0.0f32; 16];
        let mut k = 0;
        while k < 4 {
            let in_chunk: &[f32; 16] = chunk[k * 16..k * 16 + 16].try_into().unwrap();
            let float32x4x4_t(rv, gv, bv, av) = vld4q_f32(in_chunk);
            let r_slice: &mut [f32; 4] = (&mut r_out[k * 4..k * 4 + 4]).try_into().unwrap();
            let g_slice: &mut [f32; 4] = (&mut g_out[k * 4..k * 4 + 4]).try_into().unwrap();
            let b_slice: &mut [f32; 4] = (&mut b_out[k * 4..k * 4 + 4]).try_into().unwrap();
            let a_slice: &mut [f32; 4] = (&mut a_out[k * 4..k * 4 + 4]).try_into().unwrap();
            vst1q_f32(r_slice, rv);
            vst1q_f32(g_slice, gv);
            vst1q_f32(b_slice, bv);
            vst1q_f32(a_slice, av);
            k += 1;
        }
        (r_out, g_out, b_out, a_out)
    }

    // RGB chunk-16 interleave: 4 × vst3q_f32.
    #[rite]
    pub fn planes_to_rgb_f32_chunk16_neon(
        _t: NeonToken,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
    ) -> [f32; 48] {
        let mut out = [0.0f32; 48];
        let mut k = 0;
        while k < 4 {
            let r_slice: &[f32; 4] = r[k * 4..k * 4 + 4].try_into().unwrap();
            let g_slice: &[f32; 4] = g[k * 4..k * 4 + 4].try_into().unwrap();
            let b_slice: &[f32; 4] = b[k * 4..k * 4 + 4].try_into().unwrap();
            let out_slice: &mut [f32; 12] = (&mut out[k * 12..k * 12 + 12]).try_into().unwrap();
            vst3q_f32(
                out_slice,
                float32x4x3_t(vld1q_f32(r_slice), vld1q_f32(g_slice), vld1q_f32(b_slice)),
            );
            k += 1;
        }
        out
    }

    // RGBA chunk-16 interleave: 4 × vst4q_f32.
    #[rite]
    pub fn planes_to_rgba_f32_chunk16_neon(
        _t: NeonToken,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
        a: &[f32; 16],
    ) -> [f32; 64] {
        let mut out = [0.0f32; 64];
        let mut k = 0;
        while k < 4 {
            let r_slice: &[f32; 4] = r[k * 4..k * 4 + 4].try_into().unwrap();
            let g_slice: &[f32; 4] = g[k * 4..k * 4 + 4].try_into().unwrap();
            let b_slice: &[f32; 4] = b[k * 4..k * 4 + 4].try_into().unwrap();
            let a_slice: &[f32; 4] = a[k * 4..k * 4 + 4].try_into().unwrap();
            let out_slice: &mut [f32; 16] = (&mut out[k * 16..k * 16 + 16]).try_into().unwrap();
            vst4q_f32(
                out_slice,
                float32x4x4_t(
                    vld1q_f32(r_slice),
                    vld1q_f32(g_slice),
                    vld1q_f32(b_slice),
                    vld1q_f32(a_slice),
                ),
            );
            k += 1;
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
pub use arm_f32_chunks::{
    planes_to_rgb_f32_chunk4_neon, planes_to_rgb_f32_chunk8_neon, planes_to_rgb_f32_chunk16_neon,
    planes_to_rgba_f32_chunk4_neon, planes_to_rgba_f32_chunk8_neon,
    planes_to_rgba_f32_chunk16_neon, rgb_f32_chunk4_to_planes_neon, rgb_f32_chunk8_to_planes_neon,
    rgb_f32_chunk16_to_planes_neon, rgba_f32_chunk4_to_planes_neon, rgba_f32_chunk8_to_planes_neon,
    rgba_f32_chunk16_to_planes_neon,
};

// ---------------------------------------------------------------------------
// x86_64 AVX2 (`v3`) chunk-level SIMD specializations
// ---------------------------------------------------------------------------
//
// The deinterleave path uses the canonical 5-shuffle f32 stride-3 recipe on
// 128-bit registers (works on AVX2; encoded as VEX `vshufps` /
// `vmovlhps` / `vmovhlps` / `vshufps`). The 256-bit AVX2 forms of these
// shuffles operate on independent 128-bit lanes, so doubling up two 128-bit
// chunks side-by-side is just two interleaved 128-bit kernels.
//
// For RGBA: `_mm256_unpacklo_ps` / `_mm256_unpackhi_ps` + 256-bit
// `_mm256_shuffle_ps` / `_mm256_permute2f128_ps` form the standard 4-channel
// transpose; this is the `_MM_TRANSPOSE4_PS` block extended to AVX2 lanes.
//
// All shuffles use stable, hand-encoded 8-bit immediates. `_MM_SHUFFLE`
// (which is `(d<<6)|(c<<4)|(b<<2)|a`) is unstable in Rust, so we encode by
// hand with a comment showing the picked lanes.

#[cfg(target_arch = "x86_64")]
mod x86_f32_chunks {
    use super::*;

    // ----- chunk-4 RGB deinterleave (128-bit AVX2) ------------------------
    //
    // Three 128-bit loads cover 12 f32 = 4 RGB pixels.
    //   a = src[0..4]  = [r0, g0, b0, r1]
    //   b = src[4..8]  = [g1, b1, r2, g2]
    //   c = src[8..12] = [b2, r3, g3, b3]
    //
    // Output:
    //   R = [r0, r1, r2, r3] = [a[0], a[3], b[2], c[1]]
    //   G = [g0, g1, g2, g3] = [a[1], b[0], b[3], c[2]]
    //   B = [b0, b1, b2, b3] = [a[2], b[1], c[0], c[3]]
    //
    // Recipe per plane:
    //   shuf(a, b, IMM) lays a-lanes in result[0..2], b-lanes in result[2..4].
    //   That can place 2 of the 4 needed lanes; we pull the remaining 2 from
    //   c via permute + blend. The detailed lane math is in the comments
    //   inside `rgb_f32_chunk4_to_planes_v3` below.

    /// AVX2 chunk-4 RGB deinterleave. 128-bit shuffles. 9 SIMD ops total
    /// (3 loads, 3 shuf_ps on a/b, 3 permute_ps on c, 3 blend_ps, 3 stores).
    #[rite]
    pub fn rgb_f32_chunk4_to_planes_v3(
        _t: X64V3Token,
        chunk: &[f32; 12],
    ) -> ([f32; 4], [f32; 4], [f32; 4]) {
        // a = [r0, g0, b0, r1]
        // b = [g1, b1, r2, g2]
        // c = [b2, r3, g3, b3]
        let a_arr: &[f32; 4] = chunk[0..4].try_into().unwrap();
        let b_arr: &[f32; 4] = chunk[4..8].try_into().unwrap();
        let c_arr: &[f32; 4] = chunk[8..12].try_into().unwrap();
        let a = _mm_loadu_ps(a_arr);
        let b = _mm_loadu_ps(b_arr);
        let c = _mm_loadu_ps(c_arr);

        // _mm_shuffle_ps imm decoding: result[0]=a[imm[1:0]], result[1]=a[imm[3:2]],
        //   result[2]=b[imm[5:4]], result[3]=b[imm[7:6]]
        // _mm_permute_ps imm decoding: result[i] = src[(imm >> 2*i) & 3]
        // _mm_blend_ps mask: bit i = 0 → from first src, bit i = 1 → from second src.

        // R-plane = [r0, r1, r2, r3] = [a[0], a[3], b[2], c[1]]:
        //   step 1: r_ab = shuf(a, b, 0x6C) = [a[0], a[3], b[2], b[1]] = [r0, r1, r2, b1]
        //   step 2: c_r = perm(c, 0x55) broadcasts c[1] = [r3, r3, r3, r3]
        //   step 3: r = blend(r_ab, c_r, 0x8) = [r0, r1, r2, r3]
        let r_ab = _mm_shuffle_ps::<0x6C>(a, b);
        let c_r = _mm_permute_ps::<0x55>(c);
        let r = _mm_blend_ps::<0x8>(r_ab, c_r);

        // G-plane = [g0, g1, g2, g3] = [a[1], b[0], b[3], c[2]]:
        //   step 1: g_ab = shuf(a, b, 0xC1) = [a[1], a[0], b[0], b[3]] = [g0, r0, g1, g2]
        //   step 2: g_perm = perm(g_ab, 0x38) = [g_ab[0], g_ab[2], g_ab[3], g_ab[0]] = [g0, g1, g2, g0]
        //   step 3: c_g = perm(c, 0xAA) broadcasts c[2] = [g3, g3, g3, g3]
        //   step 4: g = blend(g_perm, c_g, 0x8) = [g0, g1, g2, g3]
        let g_ab = _mm_shuffle_ps::<0xC1>(a, b);
        let g_perm = _mm_permute_ps::<0x38>(g_ab);
        let c_g = _mm_permute_ps::<0xAA>(c);
        let g = _mm_blend_ps::<0x8>(g_perm, c_g);

        // B-plane = [b0, b1, b2, b3] = [a[2], b[1], c[0], c[3]]:
        //   step 1: b_ab = shuf(a, b, 0xD2) = [a[2], a[0], b[1], b[3]] = [b0, r0, b1, g2]
        //   step 2: b_perm = perm(b_ab, 0x08) = [b_ab[0], b_ab[2], b_ab[0], b_ab[0]] = [b0, b1, b0, b0]
        //   step 3: c_b = perm(c, 0xC0) = [c[0], c[0], c[0], c[3]] = [b2, b2, b2, b3]
        //     (we only care about lanes 2, 3 = c[0], c[3])
        //   step 4: b = blend(b_perm, c_b, 0xC) = [b_perm[0], b_perm[1], c_b[2], c_b[3]] = [b0, b1, b2, b3]
        let b_ab = _mm_shuffle_ps::<0xD2>(a, b);
        let b_perm = _mm_permute_ps::<0x08>(b_ab);
        let c_b = _mm_permute_ps::<0xC0>(c);
        let b_result = _mm_blend_ps::<0xC>(b_perm, c_b);

        let mut r_out = [0.0f32; 4];
        let mut g_out = [0.0f32; 4];
        let mut b_out = [0.0f32; 4];
        _mm_storeu_ps(&mut r_out, r);
        _mm_storeu_ps(&mut g_out, g);
        _mm_storeu_ps(&mut b_out, b_result);
        (r_out, g_out, b_out)
    }

    /// AVX2 chunk-4 RGBA deinterleave. 4-way f32 transpose via
    /// `_mm_unpacklo_ps` / `_mm_unpackhi_ps` / `_mm_movelh_ps` / `_mm_movehl_ps`
    /// — the standard `_MM_TRANSPOSE4_PS` recipe.
    #[rite]
    pub fn rgba_f32_chunk4_to_planes_v3(
        _t: X64V3Token,
        chunk: &[f32; 16],
    ) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        // 4 rows of 4 floats each:
        //   row0 = [r0, g0, b0, a0]
        //   row1 = [r1, g1, b1, a1]
        //   row2 = [r2, g2, b2, a2]
        //   row3 = [r3, g3, b3, a3]
        // After transpose:
        //   col0 = [r0, r1, r2, r3]   ← R
        //   col1 = [g0, g1, g2, g3]   ← G
        //   col2 = [b0, b1, b2, b3]   ← B
        //   col3 = [a0, a1, a2, a3]   ← A
        let r0_arr: &[f32; 4] = chunk[0..4].try_into().unwrap();
        let r1_arr: &[f32; 4] = chunk[4..8].try_into().unwrap();
        let r2_arr: &[f32; 4] = chunk[8..12].try_into().unwrap();
        let r3_arr: &[f32; 4] = chunk[12..16].try_into().unwrap();
        let row0 = _mm_loadu_ps(r0_arr);
        let row1 = _mm_loadu_ps(r1_arr);
        let row2 = _mm_loadu_ps(r2_arr);
        let row3 = _mm_loadu_ps(r3_arr);
        // _MM_TRANSPOSE4_PS:
        //   tmp0 = unpacklo(row0, row1) = [r0, r1, g0, g1]
        //   tmp1 = unpacklo(row2, row3) = [r2, r3, g2, g3]
        //   tmp2 = unpackhi(row0, row1) = [b0, b1, a0, a1]
        //   tmp3 = unpackhi(row2, row3) = [b2, b3, a2, a3]
        //   row0 = movelh(tmp0, tmp1) = [r0, r1, r2, r3]   ← R
        //   row1 = movehl(tmp1, tmp0) = [g0, g1, g2, g3]   ← G
        //   row2 = movelh(tmp2, tmp3) = [b0, b1, b2, b3]   ← B
        //   row3 = movehl(tmp3, tmp2) = [a0, a1, a2, a3]   ← A
        let tmp0 = _mm_unpacklo_ps(row0, row1);
        let tmp1 = _mm_unpacklo_ps(row2, row3);
        let tmp2 = _mm_unpackhi_ps(row0, row1);
        let tmp3 = _mm_unpackhi_ps(row2, row3);
        let r = _mm_movelh_ps(tmp0, tmp1);
        let g = _mm_movehl_ps(tmp1, tmp0);
        let b = _mm_movelh_ps(tmp2, tmp3);
        let a = _mm_movehl_ps(tmp3, tmp2);

        let mut r_out = [0.0f32; 4];
        let mut g_out = [0.0f32; 4];
        let mut b_out = [0.0f32; 4];
        let mut a_out = [0.0f32; 4];
        _mm_storeu_ps(&mut r_out, r);
        _mm_storeu_ps(&mut g_out, g);
        _mm_storeu_ps(&mut b_out, b);
        _mm_storeu_ps(&mut a_out, a);
        (r_out, g_out, b_out, a_out)
    }

    /// AVX2 chunk-4 RGB interleave. Inverse of `rgb_f32_chunk4_to_planes_v3`
    /// using shufps/blend.
    #[rite]
    pub fn planes_to_rgb_f32_chunk4_v3(
        _t: X64V3Token,
        r: &[f32; 4],
        g: &[f32; 4],
        b: &[f32; 4],
    ) -> [f32; 12] {
        // Need:
        //   out[0..4]   = [r0, g0, b0, r1]
        //   out[4..8]   = [g1, b1, r2, g2]
        //   out[8..12]  = [b2, r3, g3, b3]
        let r_v = _mm_loadu_ps(r);
        let g_v = _mm_loadu_ps(g);
        let b_v = _mm_loadu_ps(b);

        // out0 = [r[0], g[0], b[0], r[1]]:
        //   step 1: rg = shuf(r_v, g_v, imm) = [r[a], r[b], g[c], g[d]]
        //     Want [r0, r1, g0, _]: bits[1:0]=0, bits[3:2]=1, bits[5:4]=0, bits[7:6]=anything
        //     imm = (?<<6)|(0<<4)|(1<<2)|0 = 0x?4. pick imm = 0x04
        //   rg = [r0, r1, g0, g0]
        //   step 2: out0 = blend with b lane 2 at result lane 2, swap lanes 1↔2:
        //     We need [r0, g0, b0, r1]. rg has [r0, r1, g0, g0]. Permute to [r0, g0, _, r1]:
        //       _mm_permute_ps(rg, imm): bits[1:0]=0 → r0, bits[3:2]=2 → g0, bits[5:4]=any, bits[7:6]=1 → r1
        //       imm = (1<<6)|(?<<4)|(2<<2)|0 = 0x?8 | 0x40 | 0x8 = 0x4? + 8. pick imm = 0x48
        //     rg_perm = [r0, g0, g0, r1]
        //   step 3: build b_part = [_, _, b[0], _]: _mm_shuffle_ps(b_v, b_v) broadcast b[0] = imm 0x00
        //     b_b = _mm_permute_ps(b_v, 0x00) = [b0, b0, b0, b0]
        //   step 4: blend(rg_perm, b_b, 0b0100) — keep lane 2 from b_b
        //     out0 = [r0, g0, b0, r1] ✓
        let rg = _mm_shuffle_ps::<0x04>(r_v, g_v); // [r0, r1, g0, g0]
        let rg_perm = _mm_permute_ps::<0x48>(rg); // [r0, g0, g0, r1]
        let b_bcst0 = _mm_permute_ps::<0x00>(b_v); // [b0, b0, b0, b0]
        let out0 = _mm_blend_ps::<0x4>(rg_perm, b_bcst0); // [r0, g0, b0, r1]

        // out1 = [g[1], b[1], r[2], g[2]]:
        //   gb = shuf(g_v, b_v, imm) = [g[a], g[b], b[c], b[d]]
        //   Want [g1, g2, b1, _]: bits[1:0]=1, bits[3:2]=2, bits[5:4]=1
        //   imm = (?<<6)|(1<<4)|(2<<2)|1 = 0x?9 | 0x10 = 0x?9 + 0x10 = 0x19 + ?<<6
        //   imm = (1<<6)|(1<<4)|(2<<2)|1 = 0x59
        let gb = _mm_shuffle_ps::<0x59>(g_v, b_v); // [g1, g2, b1, b1]
        // permute to [g1, b1, _, g2]:
        //   bits[1:0]=0 → g1, bits[3:2]=2 → b1, bits[5:4]=any, bits[7:6]=1 → g2
        //   imm = (1<<6)|(?<<4)|(2<<2)|0 = 0x48 + ?<<4. pick ? = 0: imm = 0x48
        let gb_perm = _mm_permute_ps::<0x48>(gb); // [g1, b1, g1, g2]
        // need lane 2 = r[2]:
        let r_bcst2 = _mm_permute_ps::<0xAA>(r_v); // [r2, r2, r2, r2]
        let out1 = _mm_blend_ps::<0x4>(gb_perm, r_bcst2); // [g1, b1, r2, g2]

        // out2 = [b[2], r[3], g[3], b[3]]:
        //   br = shuf(b_v, r_v, imm) = [b[a], b[b], r[c], r[d]]
        //   Want [b2, b3, r3, _]: bits[1:0]=2, bits[3:2]=3, bits[5:4]=3
        //   imm = (?<<6)|(3<<4)|(3<<2)|2 = 0x3E + ?<<6. pick ? = 3: imm = 0xFE
        let br = _mm_shuffle_ps::<0xFE>(b_v, r_v); // [b2, b3, r3, r3]
        // permute to [b2, r3, _, b3]:
        //   bits[1:0]=0 → b2, bits[3:2]=2 → r3, bits[5:4]=any, bits[7:6]=1 → b3
        //   imm = (1<<6)|(?<<4)|(2<<2)|0 = 0x48
        let br_perm = _mm_permute_ps::<0x48>(br); // [b2, r3, b2, b3]
        let g_bcst3 = _mm_permute_ps::<0xFF>(g_v); // [g3, g3, g3, g3]
        let out2 = _mm_blend_ps::<0x4>(br_perm, g_bcst3); // [b2, r3, g3, b3]

        let mut out = [0.0f32; 12];
        let (lo_slice, rest) = out.split_at_mut(4);
        let (mi_slice, hi_slice) = rest.split_at_mut(4);
        let lo: &mut [f32; 4] = lo_slice.try_into().unwrap();
        let mi: &mut [f32; 4] = mi_slice.try_into().unwrap();
        let hi: &mut [f32; 4] = hi_slice.try_into().unwrap();
        _mm_storeu_ps(lo, out0);
        _mm_storeu_ps(mi, out1);
        _mm_storeu_ps(hi, out2);
        out
    }

    /// AVX2 chunk-4 RGBA interleave. Inverse 4×4 transpose.
    #[rite]
    pub fn planes_to_rgba_f32_chunk4_v3(
        _t: X64V3Token,
        r: &[f32; 4],
        g: &[f32; 4],
        b: &[f32; 4],
        a: &[f32; 4],
    ) -> [f32; 16] {
        // Inverse transpose of the 4×4 layout: rows of the input become
        // columns of the output.
        let r_v = _mm_loadu_ps(r);
        let g_v = _mm_loadu_ps(g);
        let b_v = _mm_loadu_ps(b);
        let a_v = _mm_loadu_ps(a);
        let tmp0 = _mm_unpacklo_ps(r_v, g_v); // [r0, g0, r1, g1]
        let tmp1 = _mm_unpacklo_ps(b_v, a_v); // [b0, a0, b1, a1]
        let tmp2 = _mm_unpackhi_ps(r_v, g_v); // [r2, g2, r3, g3]
        let tmp3 = _mm_unpackhi_ps(b_v, a_v); // [b2, a2, b3, a3]
        let row0 = _mm_movelh_ps(tmp0, tmp1); // [r0, g0, b0, a0]
        let row1 = _mm_movehl_ps(tmp1, tmp0); // [r1, g1, b1, a1]
        let row2 = _mm_movelh_ps(tmp2, tmp3); // [r2, g2, b2, a2]
        let row3 = _mm_movehl_ps(tmp3, tmp2); // [r3, g3, b3, a3]

        let mut out = [0.0f32; 16];
        let (p0_slice, rest) = out.split_at_mut(4);
        let (p1_slice, rest) = rest.split_at_mut(4);
        let (p2_slice, p3_slice) = rest.split_at_mut(4);
        let p0: &mut [f32; 4] = p0_slice.try_into().unwrap();
        let p1: &mut [f32; 4] = p1_slice.try_into().unwrap();
        let p2: &mut [f32; 4] = p2_slice.try_into().unwrap();
        let p3: &mut [f32; 4] = p3_slice.try_into().unwrap();
        _mm_storeu_ps(p0, row0);
        _mm_storeu_ps(p1, row1);
        _mm_storeu_ps(p2, row2);
        _mm_storeu_ps(p3, row3);
        out
    }

    // ----- chunk-8 (256-bit AVX2) ------------------------------------------
    //
    // RGB chunk-8: 24 f32 = 3 × 256-bit. Use `_mm256_permutevar8x32_ps`
    // (`vpermps`) with an 8-i32 index vector — single instruction, one
    // output plane per source register.
    //
    // But that handles within-256-reg only. We have 3 source regs and need
    // to gather across them. Cleaner approach: use the chunk-4 path twice,
    // 256-bit-style, by combining via `_mm256_set_m128`.
    //
    // We instead apply `_mm_unpacklo_ps`-style transpose recursively on
    // 256-bit lanes: AVX2 256-bit `_mm256_unpacklo_ps` operates per
    // 128-bit-lane independently, so we treat the chunk as 6 × 4-pixel
    // sub-chunks worth of work. Cleanest: run the chunk-4 path twice and
    // concatenate.

    /// AVX2 chunk-8 RGB deinterleave: two chunk-4 calls; each emits 4 R,
    /// 4 G, 4 B floats; we concatenate.
    #[rite]
    pub fn rgb_f32_chunk8_to_planes_v3(
        _t: X64V3Token,
        chunk: &[f32; 24],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let lo: &[f32; 12] = chunk[0..12].try_into().unwrap();
        let hi: &[f32; 12] = chunk[12..24].try_into().unwrap();
        let (r0, g0, b0) = rgb_f32_chunk4_to_planes_v3(_t, lo);
        let (r1, g1, b1) = rgb_f32_chunk4_to_planes_v3(_t, hi);
        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        r_out[0..4].copy_from_slice(&r0);
        r_out[4..8].copy_from_slice(&r1);
        g_out[0..4].copy_from_slice(&g0);
        g_out[4..8].copy_from_slice(&g1);
        b_out[0..4].copy_from_slice(&b0);
        b_out[4..8].copy_from_slice(&b1);
        (r_out, g_out, b_out)
    }

    /// AVX2 chunk-8 RGBA deinterleave: two chunk-4 calls; concat.
    #[rite]
    pub fn rgba_f32_chunk8_to_planes_v3(
        _t: X64V3Token,
        chunk: &[f32; 32],
    ) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
        let lo: &[f32; 16] = chunk[0..16].try_into().unwrap();
        let hi: &[f32; 16] = chunk[16..32].try_into().unwrap();
        let (r0, g0, b0, a0) = rgba_f32_chunk4_to_planes_v3(_t, lo);
        let (r1, g1, b1, a1) = rgba_f32_chunk4_to_planes_v3(_t, hi);
        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        let mut a_out = [0.0f32; 8];
        r_out[0..4].copy_from_slice(&r0);
        r_out[4..8].copy_from_slice(&r1);
        g_out[0..4].copy_from_slice(&g0);
        g_out[4..8].copy_from_slice(&g1);
        b_out[0..4].copy_from_slice(&b0);
        b_out[4..8].copy_from_slice(&b1);
        a_out[0..4].copy_from_slice(&a0);
        a_out[4..8].copy_from_slice(&a1);
        (r_out, g_out, b_out, a_out)
    }

    /// AVX2 chunk-8 RGB interleave: two chunk-4 calls; concat.
    #[rite]
    pub fn planes_to_rgb_f32_chunk8_v3(
        _t: X64V3Token,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
    ) -> [f32; 24] {
        let r_lo: &[f32; 4] = r[0..4].try_into().unwrap();
        let r_hi: &[f32; 4] = r[4..8].try_into().unwrap();
        let g_lo: &[f32; 4] = g[0..4].try_into().unwrap();
        let g_hi: &[f32; 4] = g[4..8].try_into().unwrap();
        let b_lo: &[f32; 4] = b[0..4].try_into().unwrap();
        let b_hi: &[f32; 4] = b[4..8].try_into().unwrap();
        let lo = planes_to_rgb_f32_chunk4_v3(_t, r_lo, g_lo, b_lo);
        let hi = planes_to_rgb_f32_chunk4_v3(_t, r_hi, g_hi, b_hi);
        let mut out = [0.0f32; 24];
        out[0..12].copy_from_slice(&lo);
        out[12..24].copy_from_slice(&hi);
        out
    }

    /// AVX2 chunk-8 RGBA interleave: two chunk-4 calls; concat.
    #[rite]
    pub fn planes_to_rgba_f32_chunk8_v3(
        _t: X64V3Token,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
        a: &[f32; 8],
    ) -> [f32; 32] {
        let r_lo: &[f32; 4] = r[0..4].try_into().unwrap();
        let r_hi: &[f32; 4] = r[4..8].try_into().unwrap();
        let g_lo: &[f32; 4] = g[0..4].try_into().unwrap();
        let g_hi: &[f32; 4] = g[4..8].try_into().unwrap();
        let b_lo: &[f32; 4] = b[0..4].try_into().unwrap();
        let b_hi: &[f32; 4] = b[4..8].try_into().unwrap();
        let a_lo: &[f32; 4] = a[0..4].try_into().unwrap();
        let a_hi: &[f32; 4] = a[4..8].try_into().unwrap();
        let lo = planes_to_rgba_f32_chunk4_v3(_t, r_lo, g_lo, b_lo, a_lo);
        let hi = planes_to_rgba_f32_chunk4_v3(_t, r_hi, g_hi, b_hi, a_hi);
        let mut out = [0.0f32; 32];
        out[0..16].copy_from_slice(&lo);
        out[16..32].copy_from_slice(&hi);
        out
    }

    /// AVX2 chunk-16 RGB deinterleave: four chunk-4 calls; concat.
    #[rite]
    pub fn rgb_f32_chunk16_to_planes_v3(
        _t: X64V3Token,
        chunk: &[f32; 48],
    ) -> ([f32; 16], [f32; 16], [f32; 16]) {
        let mut r_out = [0.0f32; 16];
        let mut g_out = [0.0f32; 16];
        let mut b_out = [0.0f32; 16];
        let mut k = 0;
        while k < 4 {
            let in_chunk: &[f32; 12] = chunk[k * 12..k * 12 + 12].try_into().unwrap();
            let (rv, gv, bv) = rgb_f32_chunk4_to_planes_v3(_t, in_chunk);
            r_out[k * 4..k * 4 + 4].copy_from_slice(&rv);
            g_out[k * 4..k * 4 + 4].copy_from_slice(&gv);
            b_out[k * 4..k * 4 + 4].copy_from_slice(&bv);
            k += 1;
        }
        (r_out, g_out, b_out)
    }

    /// AVX2 chunk-16 RGBA deinterleave: four chunk-4 calls; concat.
    #[rite]
    pub fn rgba_f32_chunk16_to_planes_v3(
        _t: X64V3Token,
        chunk: &[f32; 64],
    ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
        let mut r_out = [0.0f32; 16];
        let mut g_out = [0.0f32; 16];
        let mut b_out = [0.0f32; 16];
        let mut a_out = [0.0f32; 16];
        let mut k = 0;
        while k < 4 {
            let in_chunk: &[f32; 16] = chunk[k * 16..k * 16 + 16].try_into().unwrap();
            let (rv, gv, bv, av) = rgba_f32_chunk4_to_planes_v3(_t, in_chunk);
            r_out[k * 4..k * 4 + 4].copy_from_slice(&rv);
            g_out[k * 4..k * 4 + 4].copy_from_slice(&gv);
            b_out[k * 4..k * 4 + 4].copy_from_slice(&bv);
            a_out[k * 4..k * 4 + 4].copy_from_slice(&av);
            k += 1;
        }
        (r_out, g_out, b_out, a_out)
    }

    /// AVX2 chunk-16 RGB interleave: four chunk-4 calls; concat.
    #[rite]
    pub fn planes_to_rgb_f32_chunk16_v3(
        _t: X64V3Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
    ) -> [f32; 48] {
        let mut out = [0.0f32; 48];
        let mut k = 0;
        while k < 4 {
            let r_slice: &[f32; 4] = r[k * 4..k * 4 + 4].try_into().unwrap();
            let g_slice: &[f32; 4] = g[k * 4..k * 4 + 4].try_into().unwrap();
            let b_slice: &[f32; 4] = b[k * 4..k * 4 + 4].try_into().unwrap();
            let part = planes_to_rgb_f32_chunk4_v3(_t, r_slice, g_slice, b_slice);
            out[k * 12..k * 12 + 12].copy_from_slice(&part);
            k += 1;
        }
        out
    }

    /// AVX2 chunk-16 RGBA interleave: four chunk-4 calls; concat.
    #[rite]
    pub fn planes_to_rgba_f32_chunk16_v3(
        _t: X64V3Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
        a: &[f32; 16],
    ) -> [f32; 64] {
        let mut out = [0.0f32; 64];
        let mut k = 0;
        while k < 4 {
            let r_slice: &[f32; 4] = r[k * 4..k * 4 + 4].try_into().unwrap();
            let g_slice: &[f32; 4] = g[k * 4..k * 4 + 4].try_into().unwrap();
            let b_slice: &[f32; 4] = b[k * 4..k * 4 + 4].try_into().unwrap();
            let a_slice: &[f32; 4] = a[k * 4..k * 4 + 4].try_into().unwrap();
            let part = planes_to_rgba_f32_chunk4_v3(_t, r_slice, g_slice, b_slice, a_slice);
            out[k * 16..k * 16 + 16].copy_from_slice(&part);
            k += 1;
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86_f32_chunks::{
    planes_to_rgb_f32_chunk4_v3, planes_to_rgb_f32_chunk8_v3, planes_to_rgb_f32_chunk16_v3,
    planes_to_rgba_f32_chunk4_v3, planes_to_rgba_f32_chunk8_v3, planes_to_rgba_f32_chunk16_v3,
    rgb_f32_chunk4_to_planes_v3, rgb_f32_chunk8_to_planes_v3, rgb_f32_chunk16_to_planes_v3,
    rgba_f32_chunk4_to_planes_v3, rgba_f32_chunk8_to_planes_v3, rgba_f32_chunk16_to_planes_v3,
};

// ---------------------------------------------------------------------------
// wasm32 SIMD128 (`wasm128`) chunk-level SIMD specializations
// ---------------------------------------------------------------------------
//
// SIMD128 has no structure-load instruction; we use `i32x4_shuffle!`
// (lane-permute across two v128 sources, producing one v128) to replicate
// the same 5-shuffle recipe used on x86. The macro takes 8 immediate lane
// indices 0..=7 (0..=3 from first source, 4..=7 from second).

#[cfg(target_arch = "wasm32")]
mod wasm_f32_chunks {
    use super::*;

    /// WASM SIMD128 chunk-4 RGB deinterleave. Uses `i32x4_shuffle` macros
    /// (which compile to `i8x16.shuffle` SIMD instructions).
    #[rite]
    pub fn rgb_f32_chunk4_to_planes_wasm128(
        _t: Wasm128Token,
        chunk: &[f32; 12],
    ) -> ([f32; 4], [f32; 4], [f32; 4]) {
        // Reinterpret f32 lanes as i32 lanes for the shuffle macro
        // (shuffle macros operate on byte-level but the i32x4 form
        // selects whole 4-byte lanes — equivalent to f32x4 shuffle).
        let a_arr: &[u8; 16] =
            bytemuck::cast_ref::<[f32; 4], [u8; 16]>(<&[f32; 4]>::try_from(&chunk[0..4]).unwrap());
        let b_arr: &[u8; 16] =
            bytemuck::cast_ref::<[f32; 4], [u8; 16]>(<&[f32; 4]>::try_from(&chunk[4..8]).unwrap());
        let c_arr: &[u8; 16] =
            bytemuck::cast_ref::<[f32; 4], [u8; 16]>(<&[f32; 4]>::try_from(&chunk[8..12]).unwrap());
        // a = [r0, g0, b0, r1]
        // b = [g1, b1, r2, g2]
        // c = [b2, r3, g3, b3]
        let a = v128_load(a_arr);
        let b = v128_load(b_arr);
        let c = v128_load(c_arr);

        // R = [a[0], a[3], b[2], c[1]]:
        //   ab = i32x4_shuffle::<0, 3, 6, 5>(a, b) → [a[0], a[3], b[2], b[1]]
        //     (lanes 0..=3 = a, lanes 4..=7 = b → 6 = b[2], 5 = b[1])
        //   r  = i32x4_shuffle::<0, 1, 2, 5>(ab, c) → [ab[0], ab[1], ab[2], c[1]]
        //     lanes 0..=3 from ab, lanes 4..=7 from c (so 5 = c[1])
        let r_ab = i32x4_shuffle::<0, 3, 6, 5>(a, b); // [r0, r1, r2, b1]
        let r = i32x4_shuffle::<0, 1, 2, 5>(r_ab, c); // [r0, r1, r2, r3]

        // G = [a[1], b[0], b[3], c[2]]:
        let g_ab = i32x4_shuffle::<1, 1, 4, 7>(a, b); // [a[1], a[1], b[0], b[3]] = [g0, _, g1, g2]
        let g = i32x4_shuffle::<0, 2, 3, 6>(g_ab, c); // [g_ab[0]=g0, g_ab[2]=g1, g_ab[3]=g2, c[2]=g3]

        // B = [a[2], b[1], c[0], c[3]]:
        let b_ab = i32x4_shuffle::<2, 2, 5, 5>(a, b); // [a[2], a[2], b[1], b[1]] = [b0, _, b1, _]
        let b_result = i32x4_shuffle::<0, 2, 4, 7>(b_ab, c); // [b_ab[0]=b0, b_ab[2]=b1, c[0]=b2, c[3]=b3]

        let mut r_bytes = [0u8; 16];
        let mut g_bytes = [0u8; 16];
        let mut b_bytes = [0u8; 16];
        v128_store(&mut r_bytes, r);
        v128_store(&mut g_bytes, g);
        v128_store(&mut b_bytes, b_result);
        let r_out: [f32; 4] = bytemuck::cast(r_bytes);
        let g_out: [f32; 4] = bytemuck::cast(g_bytes);
        let b_out: [f32; 4] = bytemuck::cast(b_bytes);
        (r_out, g_out, b_out)
    }

    /// WASM SIMD128 chunk-4 RGBA deinterleave: 4×4 transpose via shuffle.
    #[rite]
    pub fn rgba_f32_chunk4_to_planes_wasm128(
        _t: Wasm128Token,
        chunk: &[f32; 16],
    ) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        let r0_arr: &[u8; 16] =
            bytemuck::cast_ref::<[f32; 4], [u8; 16]>(<&[f32; 4]>::try_from(&chunk[0..4]).unwrap());
        let r1_arr: &[u8; 16] =
            bytemuck::cast_ref::<[f32; 4], [u8; 16]>(<&[f32; 4]>::try_from(&chunk[4..8]).unwrap());
        let r2_arr: &[u8; 16] =
            bytemuck::cast_ref::<[f32; 4], [u8; 16]>(<&[f32; 4]>::try_from(&chunk[8..12]).unwrap());
        let r3_arr: &[u8; 16] = bytemuck::cast_ref::<[f32; 4], [u8; 16]>(
            <&[f32; 4]>::try_from(&chunk[12..16]).unwrap(),
        );
        // row0 = [r0, g0, b0, a0], row1 = [r1, g1, b1, a1], etc.
        let row0 = v128_load(r0_arr);
        let row1 = v128_load(r1_arr);
        let row2 = v128_load(r2_arr);
        let row3 = v128_load(r3_arr);
        // 4×4 transpose. Two-step: first interleave pairs, then merge.
        // tmp0 = [row0[0], row1[0], row0[1], row1[1]] = [r0, r1, g0, g1]
        // tmp1 = [row2[0], row3[0], row2[1], row3[1]] = [r2, r3, g2, g3]
        // tmp2 = [row0[2], row1[2], row0[3], row1[3]] = [b0, b1, a0, a1]
        // tmp3 = [row2[2], row3[2], row2[3], row3[3]] = [b2, b3, a2, a3]
        let tmp0 = i32x4_shuffle::<0, 4, 1, 5>(row0, row1);
        let tmp1 = i32x4_shuffle::<0, 4, 1, 5>(row2, row3);
        let tmp2 = i32x4_shuffle::<2, 6, 3, 7>(row0, row1);
        let tmp3 = i32x4_shuffle::<2, 6, 3, 7>(row2, row3);
        // R = [r0, r1, r2, r3] = [tmp0[0], tmp0[1], tmp1[0], tmp1[1]]
        let r = i32x4_shuffle::<0, 1, 4, 5>(tmp0, tmp1);
        let g = i32x4_shuffle::<2, 3, 6, 7>(tmp0, tmp1);
        let b_v = i32x4_shuffle::<0, 1, 4, 5>(tmp2, tmp3);
        let a = i32x4_shuffle::<2, 3, 6, 7>(tmp2, tmp3);

        let mut r_bytes = [0u8; 16];
        let mut g_bytes = [0u8; 16];
        let mut b_bytes = [0u8; 16];
        let mut a_bytes = [0u8; 16];
        v128_store(&mut r_bytes, r);
        v128_store(&mut g_bytes, g);
        v128_store(&mut b_bytes, b_v);
        v128_store(&mut a_bytes, a);
        let r_out: [f32; 4] = bytemuck::cast(r_bytes);
        let g_out: [f32; 4] = bytemuck::cast(g_bytes);
        let b_out: [f32; 4] = bytemuck::cast(b_bytes);
        let a_out: [f32; 4] = bytemuck::cast(a_bytes);
        (r_out, g_out, b_out, a_out)
    }

    /// WASM SIMD128 chunk-4 RGB interleave (inverse of deinterleave).
    #[rite]
    pub fn planes_to_rgb_f32_chunk4_wasm128(
        _t: Wasm128Token,
        r: &[f32; 4],
        g: &[f32; 4],
        b: &[f32; 4],
    ) -> [f32; 12] {
        let r_bytes: &[u8; 16] = bytemuck::cast_ref(r);
        let g_bytes: &[u8; 16] = bytemuck::cast_ref(g);
        let b_bytes: &[u8; 16] = bytemuck::cast_ref(b);
        let r_v = v128_load(r_bytes);
        let g_v = v128_load(g_bytes);
        let b_v = v128_load(b_bytes);
        // Need:
        //   out0 = [r[0], g[0], b[0], r[1]]
        //   out1 = [g[1], b[1], r[2], g[2]]
        //   out2 = [b[2], r[3], g[3], b[3]]
        let rg0 = i32x4_shuffle::<0, 4, 1, 5>(r_v, g_v); // [r0, g0, r1, g1]
        let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg0, b_v); // [r0, g0, b0, r1]
        // out1 = [g[1], b[1], r[2], g[2]]:
        //   gb01 = shuffle::<1, 5, 1, 5>(g_v, b_v) = [g[1], b[1], g[1], b[1]]
        //   rg23 = shuffle::<2, 6, 2, 6>(r_v, g_v) = [r[2], g[2], r[2], g[2]]
        //   out1 = shuffle::<0, 1, 4, 5>(gb01, rg23) = [gb01[0], gb01[1], rg23[0], rg23[1]]
        //        = [g[1], b[1], r[2], g[2]]
        let out1 = {
            let gb01 = i32x4_shuffle::<1, 5, 1, 5>(g_v, b_v);
            let rg23 = i32x4_shuffle::<2, 6, 2, 6>(r_v, g_v);
            i32x4_shuffle::<0, 1, 4, 5>(gb01, rg23)
        };
        let out2 = {
            // [b2, r3, g3, b3]
            let br = i32x4_shuffle::<2, 7, 2, 7>(b_v, r_v); // [b2, r3, b2, r3]
            let gb_top = i32x4_shuffle::<3, 7, 3, 7>(g_v, b_v); // [g3, b3, g3, b3]
            i32x4_shuffle::<0, 1, 4, 5>(br, gb_top)
        };

        let mut out = [0.0f32; 12];
        let mut p0 = [0u8; 16];
        let mut p1 = [0u8; 16];
        let mut p2 = [0u8; 16];
        v128_store(&mut p0, out0);
        v128_store(&mut p1, out1);
        v128_store(&mut p2, out2);
        let f0: [f32; 4] = bytemuck::cast(p0);
        let f1: [f32; 4] = bytemuck::cast(p1);
        let f2: [f32; 4] = bytemuck::cast(p2);
        out[0..4].copy_from_slice(&f0);
        out[4..8].copy_from_slice(&f1);
        out[8..12].copy_from_slice(&f2);
        out
    }

    /// WASM SIMD128 chunk-4 RGBA interleave (inverse 4×4 transpose).
    #[rite]
    pub fn planes_to_rgba_f32_chunk4_wasm128(
        _t: Wasm128Token,
        r: &[f32; 4],
        g: &[f32; 4],
        b: &[f32; 4],
        a: &[f32; 4],
    ) -> [f32; 16] {
        let r_bytes: &[u8; 16] = bytemuck::cast_ref(r);
        let g_bytes: &[u8; 16] = bytemuck::cast_ref(g);
        let b_bytes: &[u8; 16] = bytemuck::cast_ref(b);
        let a_bytes: &[u8; 16] = bytemuck::cast_ref(a);
        let r_v = v128_load(r_bytes);
        let g_v = v128_load(g_bytes);
        let b_v = v128_load(b_bytes);
        let a_v = v128_load(a_bytes);
        // Inverse transpose:
        //   tmp0 = [r0, g0, r1, g1]
        //   tmp1 = [b0, a0, b1, a1]
        //   tmp2 = [r2, g2, r3, g3]
        //   tmp3 = [b2, a2, b3, a3]
        //   row0 = [r0, g0, b0, a0] = [tmp0[0], tmp0[1], tmp1[0], tmp1[1]]
        //   row1 = [r1, g1, b1, a1] = [tmp0[2], tmp0[3], tmp1[2], tmp1[3]]
        //   row2 = [r2, g2, b2, a2] = [tmp2[0], tmp2[1], tmp3[0], tmp3[1]]
        //   row3 = [r3, g3, b3, a3] = [tmp2[2], tmp2[3], tmp3[2], tmp3[3]]
        let tmp0 = i32x4_shuffle::<0, 4, 1, 5>(r_v, g_v);
        let tmp1 = i32x4_shuffle::<0, 4, 1, 5>(b_v, a_v);
        let tmp2 = i32x4_shuffle::<2, 6, 3, 7>(r_v, g_v);
        let tmp3 = i32x4_shuffle::<2, 6, 3, 7>(b_v, a_v);
        let row0 = i32x4_shuffle::<0, 1, 4, 5>(tmp0, tmp1);
        let row1 = i32x4_shuffle::<2, 3, 6, 7>(tmp0, tmp1);
        let row2 = i32x4_shuffle::<0, 1, 4, 5>(tmp2, tmp3);
        let row3 = i32x4_shuffle::<2, 3, 6, 7>(tmp2, tmp3);

        let mut out = [0.0f32; 16];
        let mut p0 = [0u8; 16];
        let mut p1 = [0u8; 16];
        let mut p2 = [0u8; 16];
        let mut p3 = [0u8; 16];
        v128_store(&mut p0, row0);
        v128_store(&mut p1, row1);
        v128_store(&mut p2, row2);
        v128_store(&mut p3, row3);
        let f0: [f32; 4] = bytemuck::cast(p0);
        let f1: [f32; 4] = bytemuck::cast(p1);
        let f2: [f32; 4] = bytemuck::cast(p2);
        let f3: [f32; 4] = bytemuck::cast(p3);
        out[0..4].copy_from_slice(&f0);
        out[4..8].copy_from_slice(&f1);
        out[8..12].copy_from_slice(&f2);
        out[12..16].copy_from_slice(&f3);
        out
    }

    /// WASM SIMD128 chunk-8 RGB deinterleave (2 × chunk-4).
    #[rite]
    pub fn rgb_f32_chunk8_to_planes_wasm128(
        _t: Wasm128Token,
        chunk: &[f32; 24],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let lo: &[f32; 12] = chunk[0..12].try_into().unwrap();
        let hi: &[f32; 12] = chunk[12..24].try_into().unwrap();
        let (r0, g0, b0) = rgb_f32_chunk4_to_planes_wasm128(_t, lo);
        let (r1, g1, b1) = rgb_f32_chunk4_to_planes_wasm128(_t, hi);
        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        r_out[0..4].copy_from_slice(&r0);
        r_out[4..8].copy_from_slice(&r1);
        g_out[0..4].copy_from_slice(&g0);
        g_out[4..8].copy_from_slice(&g1);
        b_out[0..4].copy_from_slice(&b0);
        b_out[4..8].copy_from_slice(&b1);
        (r_out, g_out, b_out)
    }

    /// WASM SIMD128 chunk-8 RGBA deinterleave (2 × chunk-4).
    #[rite]
    pub fn rgba_f32_chunk8_to_planes_wasm128(
        _t: Wasm128Token,
        chunk: &[f32; 32],
    ) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
        let lo: &[f32; 16] = chunk[0..16].try_into().unwrap();
        let hi: &[f32; 16] = chunk[16..32].try_into().unwrap();
        let (r0, g0, b0, a0) = rgba_f32_chunk4_to_planes_wasm128(_t, lo);
        let (r1, g1, b1, a1) = rgba_f32_chunk4_to_planes_wasm128(_t, hi);
        let mut r_out = [0.0f32; 8];
        let mut g_out = [0.0f32; 8];
        let mut b_out = [0.0f32; 8];
        let mut a_out = [0.0f32; 8];
        r_out[0..4].copy_from_slice(&r0);
        r_out[4..8].copy_from_slice(&r1);
        g_out[0..4].copy_from_slice(&g0);
        g_out[4..8].copy_from_slice(&g1);
        b_out[0..4].copy_from_slice(&b0);
        b_out[4..8].copy_from_slice(&b1);
        a_out[0..4].copy_from_slice(&a0);
        a_out[4..8].copy_from_slice(&a1);
        (r_out, g_out, b_out, a_out)
    }

    /// WASM SIMD128 chunk-8 RGB interleave (2 × chunk-4).
    #[rite]
    pub fn planes_to_rgb_f32_chunk8_wasm128(
        _t: Wasm128Token,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
    ) -> [f32; 24] {
        let r_lo: &[f32; 4] = r[0..4].try_into().unwrap();
        let r_hi: &[f32; 4] = r[4..8].try_into().unwrap();
        let g_lo: &[f32; 4] = g[0..4].try_into().unwrap();
        let g_hi: &[f32; 4] = g[4..8].try_into().unwrap();
        let b_lo: &[f32; 4] = b[0..4].try_into().unwrap();
        let b_hi: &[f32; 4] = b[4..8].try_into().unwrap();
        let lo = planes_to_rgb_f32_chunk4_wasm128(_t, r_lo, g_lo, b_lo);
        let hi = planes_to_rgb_f32_chunk4_wasm128(_t, r_hi, g_hi, b_hi);
        let mut out = [0.0f32; 24];
        out[0..12].copy_from_slice(&lo);
        out[12..24].copy_from_slice(&hi);
        out
    }

    /// WASM SIMD128 chunk-8 RGBA interleave (2 × chunk-4).
    #[rite]
    pub fn planes_to_rgba_f32_chunk8_wasm128(
        _t: Wasm128Token,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
        a: &[f32; 8],
    ) -> [f32; 32] {
        let r_lo: &[f32; 4] = r[0..4].try_into().unwrap();
        let r_hi: &[f32; 4] = r[4..8].try_into().unwrap();
        let g_lo: &[f32; 4] = g[0..4].try_into().unwrap();
        let g_hi: &[f32; 4] = g[4..8].try_into().unwrap();
        let b_lo: &[f32; 4] = b[0..4].try_into().unwrap();
        let b_hi: &[f32; 4] = b[4..8].try_into().unwrap();
        let a_lo: &[f32; 4] = a[0..4].try_into().unwrap();
        let a_hi: &[f32; 4] = a[4..8].try_into().unwrap();
        let lo = planes_to_rgba_f32_chunk4_wasm128(_t, r_lo, g_lo, b_lo, a_lo);
        let hi = planes_to_rgba_f32_chunk4_wasm128(_t, r_hi, g_hi, b_hi, a_hi);
        let mut out = [0.0f32; 32];
        out[0..16].copy_from_slice(&lo);
        out[16..32].copy_from_slice(&hi);
        out
    }

    /// WASM SIMD128 chunk-16 RGB deinterleave (4 × chunk-4).
    #[rite]
    pub fn rgb_f32_chunk16_to_planes_wasm128(
        _t: Wasm128Token,
        chunk: &[f32; 48],
    ) -> ([f32; 16], [f32; 16], [f32; 16]) {
        let mut r_out = [0.0f32; 16];
        let mut g_out = [0.0f32; 16];
        let mut b_out = [0.0f32; 16];
        let mut k = 0;
        while k < 4 {
            let in_chunk: &[f32; 12] = chunk[k * 12..k * 12 + 12].try_into().unwrap();
            let (rv, gv, bv) = rgb_f32_chunk4_to_planes_wasm128(_t, in_chunk);
            r_out[k * 4..k * 4 + 4].copy_from_slice(&rv);
            g_out[k * 4..k * 4 + 4].copy_from_slice(&gv);
            b_out[k * 4..k * 4 + 4].copy_from_slice(&bv);
            k += 1;
        }
        (r_out, g_out, b_out)
    }

    /// WASM SIMD128 chunk-16 RGBA deinterleave (4 × chunk-4).
    #[rite]
    pub fn rgba_f32_chunk16_to_planes_wasm128(
        _t: Wasm128Token,
        chunk: &[f32; 64],
    ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
        let mut r_out = [0.0f32; 16];
        let mut g_out = [0.0f32; 16];
        let mut b_out = [0.0f32; 16];
        let mut a_out = [0.0f32; 16];
        let mut k = 0;
        while k < 4 {
            let in_chunk: &[f32; 16] = chunk[k * 16..k * 16 + 16].try_into().unwrap();
            let (rv, gv, bv, av) = rgba_f32_chunk4_to_planes_wasm128(_t, in_chunk);
            r_out[k * 4..k * 4 + 4].copy_from_slice(&rv);
            g_out[k * 4..k * 4 + 4].copy_from_slice(&gv);
            b_out[k * 4..k * 4 + 4].copy_from_slice(&bv);
            a_out[k * 4..k * 4 + 4].copy_from_slice(&av);
            k += 1;
        }
        (r_out, g_out, b_out, a_out)
    }

    /// WASM SIMD128 chunk-16 RGB interleave (4 × chunk-4).
    #[rite]
    pub fn planes_to_rgb_f32_chunk16_wasm128(
        _t: Wasm128Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
    ) -> [f32; 48] {
        let mut out = [0.0f32; 48];
        let mut k = 0;
        while k < 4 {
            let r_slice: &[f32; 4] = r[k * 4..k * 4 + 4].try_into().unwrap();
            let g_slice: &[f32; 4] = g[k * 4..k * 4 + 4].try_into().unwrap();
            let b_slice: &[f32; 4] = b[k * 4..k * 4 + 4].try_into().unwrap();
            let part = planes_to_rgb_f32_chunk4_wasm128(_t, r_slice, g_slice, b_slice);
            out[k * 12..k * 12 + 12].copy_from_slice(&part);
            k += 1;
        }
        out
    }

    /// WASM SIMD128 chunk-16 RGBA interleave (4 × chunk-4).
    #[rite]
    pub fn planes_to_rgba_f32_chunk16_wasm128(
        _t: Wasm128Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
        a: &[f32; 16],
    ) -> [f32; 64] {
        let mut out = [0.0f32; 64];
        let mut k = 0;
        while k < 4 {
            let r_slice: &[f32; 4] = r[k * 4..k * 4 + 4].try_into().unwrap();
            let g_slice: &[f32; 4] = g[k * 4..k * 4 + 4].try_into().unwrap();
            let b_slice: &[f32; 4] = b[k * 4..k * 4 + 4].try_into().unwrap();
            let a_slice: &[f32; 4] = a[k * 4..k * 4 + 4].try_into().unwrap();
            let part = planes_to_rgba_f32_chunk4_wasm128(_t, r_slice, g_slice, b_slice, a_slice);
            out[k * 16..k * 16 + 16].copy_from_slice(&part);
            k += 1;
        }
        out
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_f32_chunks::{
    planes_to_rgb_f32_chunk4_wasm128, planes_to_rgb_f32_chunk8_wasm128,
    planes_to_rgb_f32_chunk16_wasm128, planes_to_rgba_f32_chunk4_wasm128,
    planes_to_rgba_f32_chunk8_wasm128, planes_to_rgba_f32_chunk16_wasm128,
    rgb_f32_chunk4_to_planes_wasm128, rgb_f32_chunk8_to_planes_wasm128,
    rgb_f32_chunk16_to_planes_wasm128, rgba_f32_chunk4_to_planes_wasm128,
    rgba_f32_chunk8_to_planes_wasm128, rgba_f32_chunk16_to_planes_wasm128,
};

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

    // ----- Chunk-level f32 deinterleave / interleave round-trips --------

    fn make_rgb_chunk<const N: usize, const M: usize>() -> [f32; M] {
        // M = N * 3
        let mut out = [0.0f32; M];
        let mut i = 0;
        while i < M {
            out[i] = i as f32;
            i += 1;
        }
        out
    }

    fn make_rgba_chunk<const N: usize, const M: usize>() -> [f32; M] {
        // M = N * 4
        let mut out = [0.0f32; M];
        let mut i = 0;
        while i < M {
            out[i] = i as f32 * 0.25 - 7.0;
            i += 1;
        }
        out
    }

    // --- chunk-4 -----------------------------------------------------------

    #[test]
    fn rgb_f32_chunk4_order_preserving() {
        let src: [f32; 12] = make_rgb_chunk::<4, 12>();
        let (r, g, b) = rgb_f32_chunk4_to_planes(&src);
        assert_eq!(r, [0.0, 3.0, 6.0, 9.0]);
        assert_eq!(g, [1.0, 4.0, 7.0, 10.0]);
        assert_eq!(b, [2.0, 5.0, 8.0, 11.0]);
    }

    #[test]
    fn rgb_f32_chunk4_round_trip() {
        let src: [f32; 12] = make_rgb_chunk::<4, 12>();
        let (r, g, b) = rgb_f32_chunk4_to_planes(&src);
        let back = planes_to_rgb_f32_chunk4(&r, &g, &b);
        assert_eq!(back, src);
    }

    #[test]
    fn rgb_f32_chunk4_matches_slice_api() {
        let src: [f32; 12] = make_rgb_chunk::<4, 12>();
        let (r_chunk, g_chunk, b_chunk) = rgb_f32_chunk4_to_planes(&src);
        let mut r = [0.0f32; 4];
        let mut g = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        rgb_f32_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r_chunk, r);
        assert_eq!(g_chunk, g);
        assert_eq!(b_chunk, b);
    }

    #[test]
    fn rgba_f32_chunk4_order_preserving() {
        let src: [f32; 16] = make_rgba_chunk::<4, 16>();
        let (r, g, b, a) = rgba_f32_chunk4_to_planes(&src);
        for i in 0..4 {
            assert_eq!(r[i], src[i * 4]);
            assert_eq!(g[i], src[i * 4 + 1]);
            assert_eq!(b[i], src[i * 4 + 2]);
            assert_eq!(a[i], src[i * 4 + 3]);
        }
    }

    #[test]
    fn rgba_f32_chunk4_round_trip() {
        let src: [f32; 16] = make_rgba_chunk::<4, 16>();
        let (r, g, b, a) = rgba_f32_chunk4_to_planes(&src);
        let back = planes_to_rgba_f32_chunk4(&r, &g, &b, &a);
        assert_eq!(back, src);
    }

    #[test]
    fn rgba_f32_chunk4_matches_slice_api() {
        let src: [f32; 16] = make_rgba_chunk::<4, 16>();
        let (r_chunk, g_chunk, b_chunk, a_chunk) = rgba_f32_chunk4_to_planes(&src);
        let mut r = [0.0f32; 4];
        let mut g = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        let mut a = [0.0f32; 4];
        rgba_f32_to_planes_f32(&src, &mut r, &mut g, &mut b, &mut a).unwrap();
        assert_eq!(r_chunk, r);
        assert_eq!(g_chunk, g);
        assert_eq!(b_chunk, b);
        assert_eq!(a_chunk, a);
    }

    // --- chunk-8 -----------------------------------------------------------

    #[test]
    fn rgb_f32_chunk8_order_preserving() {
        let src: [f32; 24] = make_rgb_chunk::<8, 24>();
        let (r, g, b) = rgb_f32_chunk8_to_planes(&src);
        assert_eq!(r, [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]);
        assert_eq!(g, [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]);
        assert_eq!(b, [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0]);
    }

    #[test]
    fn rgb_f32_chunk8_round_trip() {
        let src: [f32; 24] = make_rgb_chunk::<8, 24>();
        let (r, g, b) = rgb_f32_chunk8_to_planes(&src);
        let back = planes_to_rgb_f32_chunk8(&r, &g, &b);
        assert_eq!(back, src);
    }

    #[test]
    fn rgb_f32_chunk8_matches_slice_api() {
        let src: [f32; 24] = make_rgb_chunk::<8, 24>();
        let (r_chunk, g_chunk, b_chunk) = rgb_f32_chunk8_to_planes(&src);
        let mut r = [0.0f32; 8];
        let mut g = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        rgb_f32_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r_chunk, r);
        assert_eq!(g_chunk, g);
        assert_eq!(b_chunk, b);
    }

    #[test]
    fn rgba_f32_chunk8_order_preserving() {
        let src: [f32; 32] = make_rgba_chunk::<8, 32>();
        let (r, g, b, a) = rgba_f32_chunk8_to_planes(&src);
        for i in 0..8 {
            assert_eq!(r[i], src[i * 4]);
            assert_eq!(g[i], src[i * 4 + 1]);
            assert_eq!(b[i], src[i * 4 + 2]);
            assert_eq!(a[i], src[i * 4 + 3]);
        }
    }

    #[test]
    fn rgba_f32_chunk8_round_trip() {
        let src: [f32; 32] = make_rgba_chunk::<8, 32>();
        let (r, g, b, a) = rgba_f32_chunk8_to_planes(&src);
        let back = planes_to_rgba_f32_chunk8(&r, &g, &b, &a);
        assert_eq!(back, src);
    }

    #[test]
    fn rgba_f32_chunk8_matches_slice_api() {
        let src: [f32; 32] = make_rgba_chunk::<8, 32>();
        let (r_chunk, g_chunk, b_chunk, a_chunk) = rgba_f32_chunk8_to_planes(&src);
        let mut r = [0.0f32; 8];
        let mut g = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        let mut a = [0.0f32; 8];
        rgba_f32_to_planes_f32(&src, &mut r, &mut g, &mut b, &mut a).unwrap();
        assert_eq!(r_chunk, r);
        assert_eq!(g_chunk, g);
        assert_eq!(b_chunk, b);
        assert_eq!(a_chunk, a);
    }

    // --- chunk-16 ----------------------------------------------------------

    #[test]
    fn rgb_f32_chunk16_order_preserving() {
        let src: [f32; 48] = make_rgb_chunk::<16, 48>();
        let (r, g, b) = rgb_f32_chunk16_to_planes(&src);
        for i in 0..16 {
            assert_eq!(r[i], (i * 3) as f32);
            assert_eq!(g[i], (i * 3 + 1) as f32);
            assert_eq!(b[i], (i * 3 + 2) as f32);
        }
    }

    #[test]
    fn rgb_f32_chunk16_round_trip() {
        let src: [f32; 48] = make_rgb_chunk::<16, 48>();
        let (r, g, b) = rgb_f32_chunk16_to_planes(&src);
        let back = planes_to_rgb_f32_chunk16(&r, &g, &b);
        assert_eq!(back, src);
    }

    #[test]
    fn rgb_f32_chunk16_matches_slice_api() {
        let src: [f32; 48] = make_rgb_chunk::<16, 48>();
        let (r_chunk, g_chunk, b_chunk) = rgb_f32_chunk16_to_planes(&src);
        let mut r = [0.0f32; 16];
        let mut g = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        rgb_f32_to_planes_f32(&src, &mut r, &mut g, &mut b).unwrap();
        assert_eq!(r_chunk, r);
        assert_eq!(g_chunk, g);
        assert_eq!(b_chunk, b);
    }

    #[test]
    fn rgba_f32_chunk16_order_preserving() {
        let src: [f32; 64] = make_rgba_chunk::<16, 64>();
        let (r, g, b, a) = rgba_f32_chunk16_to_planes(&src);
        for i in 0..16 {
            assert_eq!(r[i], src[i * 4]);
            assert_eq!(g[i], src[i * 4 + 1]);
            assert_eq!(b[i], src[i * 4 + 2]);
            assert_eq!(a[i], src[i * 4 + 3]);
        }
    }

    #[test]
    fn rgba_f32_chunk16_round_trip() {
        let src: [f32; 64] = make_rgba_chunk::<16, 64>();
        let (r, g, b, a) = rgba_f32_chunk16_to_planes(&src);
        let back = planes_to_rgba_f32_chunk16(&r, &g, &b, &a);
        assert_eq!(back, src);
    }

    #[test]
    fn rgba_f32_chunk16_matches_slice_api() {
        let src: [f32; 64] = make_rgba_chunk::<16, 64>();
        let (r_chunk, g_chunk, b_chunk, a_chunk) = rgba_f32_chunk16_to_planes(&src);
        let mut r = [0.0f32; 16];
        let mut g = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        let mut a = [0.0f32; 16];
        rgba_f32_to_planes_f32(&src, &mut r, &mut g, &mut b, &mut a).unwrap();
        assert_eq!(r_chunk, r);
        assert_eq!(g_chunk, g);
        assert_eq!(b_chunk, b);
        assert_eq!(a_chunk, a);
    }

    // ----- Per-arch SIMD chunk parity tests ------------------------------
    //
    // For each (arch, chunk size, layout, direction), verify the SIMD chunk
    // function produces byte-identical output to the scalar reference.
    // Skips cleanly via `Token::summon()` returning `None` when the host
    // lacks the required ISA — ScalarToken still summons unconditionally.

    // --- x86_64 AVX2 chunk parity ---

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgb_f32_chunk4_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let src: [f32; 12] = core::array::from_fn(|i| (i as f32) * 0.123 - 4.0);
            let s = rgb_f32_chunk4_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: X64V3Token, c: &[f32; 12]) -> ([f32; 4], [f32; 4], [f32; 4]) {
                rgb_f32_chunk4_to_planes_v3(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v, "rgb_f32_chunk4_v3 mismatch with scalar");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgba_f32_chunk4_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.25 - 7.0);
            let s = rgba_f32_chunk4_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: X64V3Token, c: &[f32; 16]) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
                rgba_f32_chunk4_to_planes_v3(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planes_to_rgb_f32_chunk4_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let r: [f32; 4] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 4] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 4] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk4_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: X64V3Token, r: &[f32; 4], g: &[f32; 4], b: &[f32; 4]) -> [f32; 12] {
                planes_to_rgb_f32_chunk4_v3(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planes_to_rgba_f32_chunk4_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let r: [f32; 4] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 4] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 4] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 4] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk4_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: X64V3Token,
                r: &[f32; 4],
                g: &[f32; 4],
                b: &[f32; 4],
                a: &[f32; 4],
            ) -> [f32; 16] {
                planes_to_rgba_f32_chunk4_v3(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgb_f32_chunk8_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let src: [f32; 24] = core::array::from_fn(|i| (i as f32) * 0.31 - 5.0);
            let s = rgb_f32_chunk8_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: X64V3Token, c: &[f32; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
                rgb_f32_chunk8_to_planes_v3(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgba_f32_chunk8_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let src: [f32; 32] = core::array::from_fn(|i| (i as f32) * 0.17 + 2.0);
            let s = rgba_f32_chunk8_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: X64V3Token, c: &[f32; 32]) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
                rgba_f32_chunk8_to_planes_v3(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planes_to_rgb_f32_chunk8_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let r: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 8] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk8_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: X64V3Token, r: &[f32; 8], g: &[f32; 8], b: &[f32; 8]) -> [f32; 24] {
                planes_to_rgb_f32_chunk8_v3(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planes_to_rgba_f32_chunk8_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let r: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 8] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk8_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: X64V3Token,
                r: &[f32; 8],
                g: &[f32; 8],
                b: &[f32; 8],
                a: &[f32; 8],
            ) -> [f32; 32] {
                planes_to_rgba_f32_chunk8_v3(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgb_f32_chunk16_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let src: [f32; 48] = core::array::from_fn(|i| (i as f32) * 0.21 - 1.0);
            let s = rgb_f32_chunk16_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: X64V3Token, c: &[f32; 48]) -> ([f32; 16], [f32; 16], [f32; 16]) {
                rgb_f32_chunk16_to_planes_v3(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgba_f32_chunk16_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let src: [f32; 64] = core::array::from_fn(|i| (i as f32) * 0.11 + 5.0);
            let s = rgba_f32_chunk16_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: X64V3Token, c: &[f32; 64]) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
                rgba_f32_chunk16_to_planes_v3(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planes_to_rgb_f32_chunk16_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk16_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: X64V3Token, r: &[f32; 16], g: &[f32; 16], b: &[f32; 16]) -> [f32; 48] {
                planes_to_rgb_f32_chunk16_v3(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planes_to_rgba_f32_chunk16_v3_matches_scalar() {
        if let Some(t) = X64V3Token::summon() {
            let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk16_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: X64V3Token,
                r: &[f32; 16],
                g: &[f32; 16],
                b: &[f32; 16],
                a: &[f32; 16],
            ) -> [f32; 64] {
                planes_to_rgba_f32_chunk16_v3(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    // --- aarch64 NEON chunk parity ---

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rgb_f32_chunk4_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let src: [f32; 12] = core::array::from_fn(|i| (i as f32) * 0.123 - 4.0);
            let s = rgb_f32_chunk4_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: NeonToken, c: &[f32; 12]) -> ([f32; 4], [f32; 4], [f32; 4]) {
                rgb_f32_chunk4_to_planes_neon(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rgba_f32_chunk4_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.25 - 7.0);
            let s = rgba_f32_chunk4_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: NeonToken, c: &[f32; 16]) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
                rgba_f32_chunk4_to_planes_neon(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn planes_to_rgb_f32_chunk4_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let r: [f32; 4] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 4] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 4] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk4_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: NeonToken, r: &[f32; 4], g: &[f32; 4], b: &[f32; 4]) -> [f32; 12] {
                planes_to_rgb_f32_chunk4_neon(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn planes_to_rgba_f32_chunk4_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let r: [f32; 4] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 4] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 4] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 4] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk4_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: NeonToken,
                r: &[f32; 4],
                g: &[f32; 4],
                b: &[f32; 4],
                a: &[f32; 4],
            ) -> [f32; 16] {
                planes_to_rgba_f32_chunk4_neon(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rgb_f32_chunk8_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let src: [f32; 24] = core::array::from_fn(|i| (i as f32) * 0.31 - 5.0);
            let s = rgb_f32_chunk8_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: NeonToken, c: &[f32; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
                rgb_f32_chunk8_to_planes_neon(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rgba_f32_chunk8_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let src: [f32; 32] = core::array::from_fn(|i| (i as f32) * 0.17 + 2.0);
            let s = rgba_f32_chunk8_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: NeonToken, c: &[f32; 32]) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
                rgba_f32_chunk8_to_planes_neon(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn planes_to_rgb_f32_chunk8_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let r: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 8] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk8_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: NeonToken, r: &[f32; 8], g: &[f32; 8], b: &[f32; 8]) -> [f32; 24] {
                planes_to_rgb_f32_chunk8_neon(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn planes_to_rgba_f32_chunk8_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let r: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 8] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk8_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: NeonToken,
                r: &[f32; 8],
                g: &[f32; 8],
                b: &[f32; 8],
                a: &[f32; 8],
            ) -> [f32; 32] {
                planes_to_rgba_f32_chunk8_neon(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rgb_f32_chunk16_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let src: [f32; 48] = core::array::from_fn(|i| (i as f32) * 0.21 - 1.0);
            let s = rgb_f32_chunk16_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: NeonToken, c: &[f32; 48]) -> ([f32; 16], [f32; 16], [f32; 16]) {
                rgb_f32_chunk16_to_planes_neon(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rgba_f32_chunk16_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let src: [f32; 64] = core::array::from_fn(|i| (i as f32) * 0.11 + 5.0);
            let s = rgba_f32_chunk16_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: NeonToken, c: &[f32; 64]) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
                rgba_f32_chunk16_to_planes_neon(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn planes_to_rgb_f32_chunk16_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk16_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: NeonToken, r: &[f32; 16], g: &[f32; 16], b: &[f32; 16]) -> [f32; 48] {
                planes_to_rgb_f32_chunk16_neon(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn planes_to_rgba_f32_chunk16_neon_matches_scalar() {
        if let Some(t) = NeonToken::summon() {
            let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk16_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: NeonToken,
                r: &[f32; 16],
                g: &[f32; 16],
                b: &[f32; 16],
                a: &[f32; 16],
            ) -> [f32; 64] {
                planes_to_rgba_f32_chunk16_neon(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    // --- wasm32 SIMD128 chunk parity ---
    //
    // wasm32 tests run only when the host can execute the wasm via a runtime
    // configured by the test harness; on native builds these are gated out.

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn rgb_f32_chunk4_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let src: [f32; 12] = core::array::from_fn(|i| (i as f32) * 0.123 - 4.0);
            let s = rgb_f32_chunk4_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: Wasm128Token, c: &[f32; 12]) -> ([f32; 4], [f32; 4], [f32; 4]) {
                rgb_f32_chunk4_to_planes_wasm128(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn rgba_f32_chunk4_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.25 - 7.0);
            let s = rgba_f32_chunk4_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: Wasm128Token, c: &[f32; 16]) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
                rgba_f32_chunk4_to_planes_wasm128(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn planes_to_rgb_f32_chunk4_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let r: [f32; 4] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 4] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 4] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk4_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: Wasm128Token, r: &[f32; 4], g: &[f32; 4], b: &[f32; 4]) -> [f32; 12] {
                planes_to_rgb_f32_chunk4_wasm128(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn planes_to_rgba_f32_chunk4_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let r: [f32; 4] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 4] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 4] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 4] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk4_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: Wasm128Token,
                r: &[f32; 4],
                g: &[f32; 4],
                b: &[f32; 4],
                a: &[f32; 4],
            ) -> [f32; 16] {
                planes_to_rgba_f32_chunk4_wasm128(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn rgb_f32_chunk8_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let src: [f32; 24] = core::array::from_fn(|i| (i as f32) * 0.31 - 5.0);
            let s = rgb_f32_chunk8_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: Wasm128Token, c: &[f32; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
                rgb_f32_chunk8_to_planes_wasm128(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn rgba_f32_chunk8_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let src: [f32; 32] = core::array::from_fn(|i| (i as f32) * 0.17 + 2.0);
            let s = rgba_f32_chunk8_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: Wasm128Token, c: &[f32; 32]) -> ([f32; 8], [f32; 8], [f32; 8], [f32; 8]) {
                rgba_f32_chunk8_to_planes_wasm128(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn planes_to_rgb_f32_chunk8_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let r: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 8] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk8_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: Wasm128Token, r: &[f32; 8], g: &[f32; 8], b: &[f32; 8]) -> [f32; 24] {
                planes_to_rgb_f32_chunk8_wasm128(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn planes_to_rgba_f32_chunk8_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let r: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 8] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk8_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: Wasm128Token,
                r: &[f32; 8],
                g: &[f32; 8],
                b: &[f32; 8],
                a: &[f32; 8],
            ) -> [f32; 32] {
                planes_to_rgba_f32_chunk8_wasm128(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn rgb_f32_chunk16_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let src: [f32; 48] = core::array::from_fn(|i| (i as f32) * 0.21 - 1.0);
            let s = rgb_f32_chunk16_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(t: Wasm128Token, c: &[f32; 48]) -> ([f32; 16], [f32; 16], [f32; 16]) {
                rgb_f32_chunk16_to_planes_wasm128(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn rgba_f32_chunk16_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let src: [f32; 64] = core::array::from_fn(|i| (i as f32) * 0.11 + 5.0);
            let s = rgba_f32_chunk16_to_planes_scalar(&src);
            #[archmage::arcane]
            fn call(
                t: Wasm128Token,
                c: &[f32; 64],
            ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
                rgba_f32_chunk16_to_planes_wasm128(t, c)
            }
            let v = call(t, &src);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn planes_to_rgb_f32_chunk16_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let s = planes_to_rgb_f32_chunk16_scalar(&r, &g, &b);
            #[archmage::arcane]
            fn call(t: Wasm128Token, r: &[f32; 16], g: &[f32; 16], b: &[f32; 16]) -> [f32; 48] {
                planes_to_rgb_f32_chunk16_wasm128(t, r, g, b)
            }
            let v = call(t, &r, &g, &b);
            assert_eq!(s, v);
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn planes_to_rgba_f32_chunk16_wasm128_matches_scalar() {
        if let Some(t) = Wasm128Token::summon() {
            let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
            let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
            let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
            let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
            let s = planes_to_rgba_f32_chunk16_scalar(&r, &g, &b, &a);
            #[archmage::arcane]
            fn call(
                t: Wasm128Token,
                r: &[f32; 16],
                g: &[f32; 16],
                b: &[f32; 16],
                a: &[f32; 16],
            ) -> [f32; 64] {
                planes_to_rgba_f32_chunk16_wasm128(t, r, g, b, a)
            }
            let v = call(t, &r, &g, &b, &a);
            assert_eq!(s, v);
        }
    }
}
