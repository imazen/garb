// ---------------------------------------------------------------------------
// Row-level pixel swizzle operations — B↔R swap, format expansion, alpha fill
// ---------------------------------------------------------------------------
//
// SIMD dispatch: AVX2 (x86-64) → NEON (aarch64) → SIMD128 (wasm32) → scalar.

use archmage::incant;
use archmage::prelude::*;

// ===========================================================================
// Public API — all functions dispatch to the best available SIMD tier
// ===========================================================================

/// Swap bytes 0 and 2 of a u32 — B↔R channel swap for BGRA/RGBA pixels.
#[inline(always)]
fn swap_br_u32(v: u32) -> u32 {
    (v & 0xFF00_FF00) | (v.rotate_left(16) & 0x00FF_00FF)
}

/// Swap B and R channels in-place for a row of 4bpp pixels (BGRA↔RGBA).
///
/// The row length must be a multiple of 4 bytes.
pub fn swap_br_inplace(row: &mut [u8]) {
    debug_assert!(row.len() % 4 == 0, "4bpp row length must be a multiple of 4");
    incant!(swap_br_impl(row), [v3, arm_v2, wasm128, scalar]);
}

/// Copy a pixel row, swapping B↔R channels (BGRA↔RGBA). Symmetric operation.
///
/// Both `src` and `dst` lengths must be multiples of 4. Only
/// `min(src.len(), dst.len())` bytes are processed.
pub fn copy_swap_br(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() % 4 == 0, "4bpp row length must be a multiple of 4");
    incant!(copy_swap_br_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
}

/// Set the alpha channel of every 4bpp pixel to 255 (fully opaque).
///
/// Alpha is byte 3 of each 4-byte pixel. Works for BGRA, RGBA, BGRX, RGBX.
pub fn set_alpha_to_255(row: &mut [u8]) {
    debug_assert!(row.len() % 4 == 0, "4bpp row length must be a multiple of 4");
    incant!(set_alpha_impl(row), [v3, arm_v2, wasm128, scalar]);
}

/// RGB24 → BGRA. 3 src bytes → 4 dst bytes per pixel. Alpha = 255.
///
/// Also usable as BGR → RGBA (same byte shuffle).
pub fn rgb_to_bgra(src: &[u8], dst: &mut [u8]) {
    incant!(rgb_to_bgra_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
}

/// RGB24 → RGBA. 3 src bytes → 4 dst bytes per pixel. Alpha = 255.
///
/// Also usable as BGR → BGRA (same byte shuffle — keeps order, adds alpha).
pub fn rgb_to_rgba(src: &[u8], dst: &mut [u8]) {
    incant!(rgb_to_rgba_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
}

/// L8 → 4bpp. 1 src byte → 4 dst bytes per pixel. R=G=B=gray, A=255.
///
/// Output is identical whether interpreted as BGRA or RGBA (R=G=B).
pub fn gray_to_4bpp(src: &[u8], dst: &mut [u8]) {
    incant!(gray_to_bgra_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
}

/// LA → 4bpp. 2 src bytes → 4 dst bytes per pixel. R=G=B=gray, A=alpha.
///
/// Output is identical whether interpreted as BGRA or RGBA (R=G=B).
pub fn gray_alpha_to_4bpp(src: &[u8], dst: &mut [u8]) {
    incant!(gray_alpha_to_bgra_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
}

/// Swap R and B channels in-place for a row of 3bpp pixels (RGB↔BGR).
///
/// The row length must be a multiple of 3 bytes.
pub fn swap_rb_3bpp_inplace(row: &mut [u8]) {
    debug_assert!(row.len() % 3 == 0, "3bpp row length must be a multiple of 3");
    for px in row.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

/// Copy a row of 3bpp pixels, swapping R↔B (RGB→BGR or BGR→RGB).
///
/// Both `src` and `dst` lengths must be multiples of 3.
pub fn copy_swap_rb_3bpp(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() % 3 == 0, "3bpp row length must be a multiple of 3");
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

/// 4bpp → 3bpp by dropping byte 3 (alpha) from each pixel.
///
/// Keeps bytes 0,1,2. Use after `swap_br_inplace` if you also need a
/// channel swap.
pub fn drop_alpha_4to3(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() % 4 == 0);
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

/// 4bpp → 3bpp, dropping byte 3 and reversing bytes 0↔2.
///
/// BGRA → RGB or RGBA → BGR in one pass.
pub fn drop_alpha_swap_4to3(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() % 4 == 0);
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// ===========================================================================
// Scalar fallback implementations
// ===========================================================================

fn swap_br_impl_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        let s = swap_br_u32(v);
        px.copy_from_slice(&s.to_ne_bytes());
    }
}

fn copy_swap_br_impl_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s_px, d_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let v = u32::from_ne_bytes([s_px[0], s_px[1], s_px[2], s_px[3]]);
        let s = swap_br_u32(v);
        d_px.copy_from_slice(&s.to_ne_bytes());
    }
}

fn set_alpha_impl_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        px[3] = 0xFF;
    }
}

/// Scalar RGB→BGRA: reverse RGB to BGR, set A=0xFF.
fn rgb_to_bgra_impl_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d_px[0] = s[2];
        d_px[1] = s[1];
        d_px[2] = s[0];
        d_px[3] = 0xFF;
    }
}

/// Scalar RGB→RGBA: keep order, set A=0xFF.
fn rgb_to_rgba_impl_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d_px) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d_px[0] = s[0];
        d_px[1] = s[1];
        d_px[2] = s[2];
        d_px[3] = 0xFF;
    }
}

/// Scalar Gray→BGRA: broadcast gray byte to B,G,R, set A=0xFF.
fn gray_to_bgra_impl_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (&v, d_px) in src.iter().zip(dst.chunks_exact_mut(4)) {
        d_px[0] = v;
        d_px[1] = v;
        d_px[2] = v;
        d_px[3] = 0xFF;
    }
}

/// Scalar GrayAlpha→BGRA: broadcast gray to B,G,R, copy alpha.
fn gray_alpha_to_bgra_impl_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (ga, d_px) in src.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
        d_px[0] = ga[0];
        d_px[1] = ga[0];
        d_px[2] = ga[0];
        d_px[3] = ga[1];
    }
}

// ===========================================================================
// x86-64 AVX2 (V3 tier) — 8 pixels / 32 bytes per iteration
// ===========================================================================

/// Byte shuffle mask: swap bytes 0↔2 within each 4-byte pixel.
#[cfg(target_arch = "x86_64")]
const BR_SHUF_MASK_AVX: [i8; 32] = [
    2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
    2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
];

/// Alpha mask: 0xFF at byte 3 of each pixel.
#[cfg(target_arch = "x86_64")]
const ALPHA_FF_MASK_AVX: [i8; 32] = [
    0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1,
    0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1,
];

/// Gray expand: replicate each of 8 bytes to B,G,R positions, zero alpha.
#[cfg(target_arch = "x86_64")]
const GRAY_EXPAND_MASK_AVX: [i8; 32] = [
    0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3, -128,
    4, 4, 4, -128, 5, 5, 5, -128, 6, 6, 6, -128, 7, 7, 7, -128,
];

/// GrayAlpha expand: replicate gray to B,G,R, keep alpha byte.
#[cfg(target_arch = "x86_64")]
const GA_EXPAND_MASK_AVX: [i8; 32] = [
    0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7,
    8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15,
];

/// RGB→BGRA shuffle: reverse RGB to BGR within each 3-byte pixel, zero alpha.
#[cfg(target_arch = "x86_64")]
const RGB_TO_BGRA_SHUF_AVX: [i8; 32] = [
    2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128,
    2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128,
];

/// RGB→RGBA shuffle: keep RGB order, zero alpha position.
#[cfg(target_arch = "x86_64")]
const RGB_TO_RGBA_SHUF_AVX: [i8; 32] = [
    0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128,
    0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128,
];

/// Permute for RGB→4bpp: align 24 bytes so each lane has 12 valid bytes.
#[cfg(target_arch = "x86_64")]
const RGB_ALIGN_PERM_AVX: [i8; 32] = [
    0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
    3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0,
];

#[cfg(target_arch = "x86_64")]
#[arcane]
fn swap_br_impl_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&BR_SHUF_MASK_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_si256(arr);
        let shuffled = _mm256_shuffle_epi8(v, mask);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(out, shuffled);
        i += 32;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn copy_swap_br_impl_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&BR_SHUF_MASK_AVX);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let s: &[u8; 32] = src[i..i + 32].try_into().unwrap();
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_si256(s);
        let shuffled = _mm256_shuffle_epi8(v, mask);
        let d: &mut [u8; 32] = (&mut dst[i..i + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, shuffled);
        i += 32;
    }
    for (s, d) in bytemuck::cast_slice::<u8, u32>(&src[i..])
        .iter()
        .zip(bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i..]))
    {
        *d = swap_br_u32(*s);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn set_alpha_impl_v3(_token: X64V3Token, row: &mut [u8]) {
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = safe_unaligned_simd::x86_64::_mm256_loadu_si256(arr);
        let result = _mm256_or_si256(v, alpha);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(out, result);
        i += 32;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

/// RGB→BGRA AVX2: vpermd align, vpshufb reverse, vpor alpha.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_bgra_impl_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_TO_BGRA_SHUF_AVX);
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 32 <= src_len && i_dst + 32 <= dst_len {
        let s: &[u8; 32] = src[i_src..i_src + 32].try_into().unwrap();
        let rgb = safe_unaligned_simd::x86_64::_mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let bgr0 = _mm256_shuffle_epi8(aligned, shuf);
        let bgra = _mm256_or_si256(bgr0, alpha);
        let d: &mut [u8; 32] = (&mut dst[i_dst..i_dst + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, bgra);
        i_src += 24;
        i_dst += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (s, d) in src[i_src..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

/// RGB→RGBA AVX2: vpermd align, vpshufb keep order, vpor alpha.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_rgba_impl_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_TO_RGBA_SHUF_AVX);
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 32 <= src_len && i_dst + 32 <= dst_len {
        let s: &[u8; 32] = src[i_src..i_src + 32].try_into().unwrap();
        let rgb = safe_unaligned_simd::x86_64::_mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let rgba0 = _mm256_shuffle_epi8(aligned, shuf);
        let rgba = _mm256_or_si256(rgba0, alpha);
        let d: &mut [u8; 32] = (&mut dst[i_dst..i_dst + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, rgba);
        i_src += 24;
        i_dst += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (s, d) in src[i_src..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

/// Gray→BGRA AVX2: broadcast + shuffle expand + alpha OR.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn gray_to_bgra_impl_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&GRAY_EXPAND_MASK_AVX);
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 8 <= src_len && i_dst + 32 <= dst_len {
        let gray8 = u64::from_ne_bytes(src[i_src..i_src + 8].try_into().unwrap());
        let grays = _mm256_set1_epi64x(gray8 as i64);
        let expanded = _mm256_shuffle_epi8(grays, expand);
        let bgra = _mm256_or_si256(expanded, alpha);
        let d: &mut [u8; 32] = (&mut dst[i_dst..i_dst + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, bgra);
        i_src += 8;
        i_dst += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (&v, d) in src[i_src..].iter().zip(dst32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

/// GrayAlpha→BGRA AVX2: load 16 GA bytes into lanes, shuffle expand.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn gray_alpha_to_bgra_impl_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&GA_EXPAND_MASK_AVX);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 32 <= dst_len {
        let lo = u64::from_ne_bytes(src[i_src..i_src + 8].try_into().unwrap());
        let hi = u64::from_ne_bytes(src[i_src + 8..i_src + 16].try_into().unwrap());
        let gas = _mm256_set_epi64x(hi as i64, lo as i64, hi as i64, lo as i64);
        let bgra = _mm256_shuffle_epi8(gas, expand);
        let d: &mut [u8; 32] = (&mut dst[i_dst..i_dst + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, bgra);
        i_src += 16;
        i_dst += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (ga, d) in src[i_src..].chunks_exact(2).zip(dst32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// ===========================================================================
// ARM NEON (arm_v2 tier) — vqtbl1q_u8 + vorrq_u8
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn swap_br_impl_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;

    let mask_bytes: [u8; 16] = [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15];
    let mask = safe_unaligned_simd::aarch64::vld1q_u8(&mask_bytes);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(arr);
        let shuffled = vqtbl1q_u8(v, mask);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(out, shuffled);
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn copy_swap_br_impl_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;

    let mask_bytes: [u8; 16] = [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15];
    let mask = safe_unaligned_simd::aarch64::vld1q_u8(&mask_bytes);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let shuffled = vqtbl1q_u8(v, mask);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d, shuffled);
        i += 16;
    }
    for (s, d) in bytemuck::cast_slice::<u8, u32>(&src[i..])
        .iter()
        .zip(bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i..]))
    {
        *d = swap_br_u32(*s);
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn set_alpha_impl_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
    use core::arch::aarch64::vorrq_u8;

    let alpha_bytes: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&alpha_bytes);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(arr);
        let result = vorrq_u8(v, alpha);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(out, result);
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb_to_bgra_impl_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{vorrq_u8, vqtbl1q_u8};

    let shuf_bytes: [u8; 16] = [2, 1, 0, 0x80, 5, 4, 3, 0x80, 8, 7, 6, 0x80, 11, 10, 9, 0x80];
    let shuf = safe_unaligned_simd::aarch64::vld1q_u8(&shuf_bytes);
    let alpha_bytes: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&alpha_bytes);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 16 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let bgr0 = vqtbl1q_u8(v, shuf);
        let bgra = vorrq_u8(bgr0, alpha);
        let d: &mut [u8; 16] = (&mut dst[i_dst..i_dst + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d, bgra);
        i_src += 12;
        i_dst += 16;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (s, d) in src[i_src..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb_to_rgba_impl_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{vorrq_u8, vqtbl1q_u8};

    let shuf_bytes: [u8; 16] = [0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80];
    let shuf = safe_unaligned_simd::aarch64::vld1q_u8(&shuf_bytes);
    let alpha_bytes: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&alpha_bytes);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 16 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let rgba0 = vqtbl1q_u8(v, shuf);
        let rgba = vorrq_u8(rgba0, alpha);
        let d: &mut [u8; 16] = (&mut dst[i_dst..i_dst + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d, rgba);
        i_src += 12;
        i_dst += 16;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (s, d) in src[i_src..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn gray_to_bgra_impl_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{vorrq_u8, vqtbl1q_u8};

    let masks: [[u8; 16]; 4] = [
        [0, 0, 0, 0x80, 1, 1, 1, 0x80, 2, 2, 2, 0x80, 3, 3, 3, 0x80],
        [4, 4, 4, 0x80, 5, 5, 5, 0x80, 6, 6, 6, 0x80, 7, 7, 7, 0x80],
        [8, 8, 8, 0x80, 9, 9, 9, 0x80, 10, 10, 10, 0x80, 11, 11, 11, 0x80],
        [12, 12, 12, 0x80, 13, 13, 13, 0x80, 14, 14, 14, 0x80, 15, 15, 15, 0x80],
    ];
    let m0 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[0]);
    let m1 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[1]);
    let m2 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[2]);
    let m3 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[3]);
    let alpha_bytes: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&alpha_bytes);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 64 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let grays = safe_unaligned_simd::aarch64::vld1q_u8(s);
        for (j, m) in [m0, m1, m2, m3].iter().enumerate() {
            let expanded = vqtbl1q_u8(grays, *m);
            let bgra = vorrq_u8(expanded, alpha);
            let d: &mut [u8; 16] =
                (&mut dst[i_dst + j * 16..i_dst + (j + 1) * 16]).try_into().unwrap();
            safe_unaligned_simd::aarch64::vst1q_u8(d, bgra);
        }
        i_src += 16;
        i_dst += 64;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (&v, d) in src[i_src..].iter().zip(dst32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn gray_alpha_to_bgra_impl_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;

    let masks: [[u8; 16]; 2] = [
        [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7],
        [8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15],
    ];
    let m0 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[0]);
    let m1 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[1]);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 32 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let gas = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let r0 = vqtbl1q_u8(gas, m0);
        let d0: &mut [u8; 16] = (&mut dst[i_dst..i_dst + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d0, r0);
        let r1 = vqtbl1q_u8(gas, m1);
        let d1: &mut [u8; 16] = (&mut dst[i_dst + 16..i_dst + 32]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d1, r1);
        i_src += 16;
        i_dst += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (ga, d) in src[i_src..].chunks_exact(2).zip(dst32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// ===========================================================================
// WASM SIMD128 — i8x16_swizzle + v128_or
// ===========================================================================

#[cfg(target_arch = "wasm32")]
#[arcane]
fn swap_br_impl_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle};

    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(arr);
        let shuffled = i8x16_swizzle(v, mask);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(out, shuffled);
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn copy_swap_br_impl_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle};

    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(s);
        let shuffled = i8x16_swizzle(v, mask);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d, shuffled);
        i += 16;
    }
    for (s, d) in bytemuck::cast_slice::<u8, u32>(&src[i..])
        .iter()
        .zip(bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i..]))
    {
        *d = swap_br_u32(*s);
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn set_alpha_impl_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    use core::arch::wasm32::{u32x4_splat, v128_or};

    let alpha = u32x4_splat(0xFF000000);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(arr);
        let result = v128_or(v, alpha);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(out, result);
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb_to_bgra_impl_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};

    let shuf = i8x16(2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128);
    let alpha = u32x4_splat(0xFF000000);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 16 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(s);
        let bgr0 = i8x16_swizzle(v, shuf);
        let bgra = v128_or(bgr0, alpha);
        let d: &mut [u8; 16] = (&mut dst[i_dst..i_dst + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d, bgra);
        i_src += 12;
        i_dst += 16;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (s, d) in src[i_src..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb_to_rgba_impl_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};

    let shuf = i8x16(0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128);
    let alpha = u32x4_splat(0xFF000000);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 16 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(s);
        let rgba0 = i8x16_swizzle(v, shuf);
        let rgba = v128_or(rgba0, alpha);
        let d: &mut [u8; 16] = (&mut dst[i_dst..i_dst + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d, rgba);
        i_src += 12;
        i_dst += 16;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (s, d) in src[i_src..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn gray_to_bgra_impl_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};

    let m0 = i8x16(0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3, -128);
    let m1 = i8x16(4, 4, 4, -128, 5, 5, 5, -128, 6, 6, 6, -128, 7, 7, 7, -128);
    let m2 = i8x16(8, 8, 8, -128, 9, 9, 9, -128, 10, 10, 10, -128, 11, 11, 11, -128);
    let m3 = i8x16(12, 12, 12, -128, 13, 13, 13, -128, 14, 14, 14, -128, 15, 15, 15, -128);
    let alpha = u32x4_splat(0xFF000000);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 64 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let grays = safe_unaligned_simd::wasm32::v128_load(s);
        for (j, m) in [m0, m1, m2, m3].iter().enumerate() {
            let expanded = i8x16_swizzle(grays, *m);
            let bgra = v128_or(expanded, alpha);
            let d: &mut [u8; 16] =
                (&mut dst[i_dst + j * 16..i_dst + (j + 1) * 16]).try_into().unwrap();
            safe_unaligned_simd::wasm32::v128_store(d, bgra);
        }
        i_src += 16;
        i_dst += 64;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (&v, d) in src[i_src..].iter().zip(dst32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn gray_alpha_to_bgra_impl_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle};

    let m0 = i8x16(0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7);
    let m1 = i8x16(8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15);
    let src_len = src.len();
    let dst_len = dst.len();
    let mut i_src = 0;
    let mut i_dst = 0;
    while i_src + 16 <= src_len && i_dst + 32 <= dst_len {
        let s: &[u8; 16] = src[i_src..i_src + 16].try_into().unwrap();
        let gas = safe_unaligned_simd::wasm32::v128_load(s);
        let r0 = i8x16_swizzle(gas, m0);
        let d0: &mut [u8; 16] = (&mut dst[i_dst..i_dst + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d0, r0);
        let r1 = i8x16_swizzle(gas, m1);
        let d1: &mut [u8; 16] = (&mut dst[i_dst + 16..i_dst + 32]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d1, r1);
        i_src += 16;
        i_dst += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i_dst..]);
    for (ga, d) in src[i_src..].chunks_exact(2).zip(dst32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::{vec, vec::Vec};
    use super::*;

    #[test]
    fn test_swap_br_inplace() {
        let mut row = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        swap_br_inplace(&mut row);
        assert_eq!(row, [30, 20, 10, 255, 60, 50, 40, 128]);
        // Double swap = identity
        swap_br_inplace(&mut row);
        assert_eq!(row, [10, 20, 30, 255, 40, 50, 60, 128]);
    }

    #[test]
    fn test_copy_swap_br() {
        let src = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut dst = vec![0u8; 8];
        copy_swap_br(&src, &mut dst);
        assert_eq!(dst, [30, 20, 10, 255, 60, 50, 40, 128]);
    }

    #[test]
    fn test_set_alpha_to_255() {
        let mut row = vec![10u8, 20, 30, 0, 40, 50, 60, 100];
        set_alpha_to_255(&mut row);
        assert_eq!(row, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn test_rgb_to_bgra() {
        let src = vec![255u8, 128, 0, 10, 20, 30];
        let mut dst = vec![0u8; 8];
        rgb_to_bgra(&src, &mut dst);
        assert_eq!(dst, [0, 128, 255, 255, 30, 20, 10, 255]);
    }

    #[test]
    fn test_rgb_to_rgba() {
        let src = vec![255u8, 128, 0, 10, 20, 30];
        let mut dst = vec![0u8; 8];
        rgb_to_rgba(&src, &mut dst);
        assert_eq!(dst, [255, 128, 0, 255, 10, 20, 30, 255]);
    }

    #[test]
    fn test_gray_to_4bpp() {
        let src = vec![100u8, 200];
        let mut dst = vec![0u8; 8];
        gray_to_4bpp(&src, &mut dst);
        assert_eq!(dst, [100, 100, 100, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn test_gray_alpha_to_4bpp() {
        let src = vec![100u8, 50, 200, 128];
        let mut dst = vec![0u8; 8];
        gray_alpha_to_4bpp(&src, &mut dst);
        assert_eq!(dst, [100, 100, 100, 50, 200, 200, 200, 128]);
    }

    #[test]
    fn test_swap_rb_3bpp_inplace() {
        let mut row = vec![10u8, 20, 30, 40, 50, 60];
        swap_rb_3bpp_inplace(&mut row);
        assert_eq!(row, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_copy_swap_rb_3bpp() {
        let src = vec![10u8, 20, 30, 40, 50, 60];
        let mut dst = vec![0u8; 6];
        copy_swap_rb_3bpp(&src, &mut dst);
        assert_eq!(dst, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_drop_alpha_4to3() {
        let src = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut dst = vec![0u8; 6];
        drop_alpha_4to3(&src, &mut dst);
        assert_eq!(dst, [10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_drop_alpha_swap_4to3() {
        let src = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut dst = vec![0u8; 6];
        drop_alpha_swap_4to3(&src, &mut dst);
        assert_eq!(dst, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_large_swap_br_hits_simd() {
        // 64 pixels = 256 bytes, enough to exercise SIMD paths
        let mut row: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let original = row.clone();
        swap_br_inplace(&mut row);
        for (i, chunk) in row.chunks(4).enumerate() {
            let o = &original[i * 4..i * 4 + 4];
            assert_eq!(chunk[0], o[2]);
            assert_eq!(chunk[1], o[1]);
            assert_eq!(chunk[2], o[0]);
            assert_eq!(chunk[3], o[3]);
        }
    }

    #[test]
    fn test_large_rgb_to_bgra_hits_simd() {
        // 32 pixels = 96 RGB bytes → 128 BGRA bytes
        let src: Vec<u8> = (0..96).map(|i| i as u8).collect();
        let mut dst = vec![0u8; 128];
        rgb_to_bgra(&src, &mut dst);
        for (i, chunk) in dst.chunks(4).enumerate() {
            let s = &src[i * 3..i * 3 + 3];
            assert_eq!(chunk[0], s[2], "pixel {i} B");
            assert_eq!(chunk[1], s[1], "pixel {i} G");
            assert_eq!(chunk[2], s[0], "pixel {i} R");
            assert_eq!(chunk[3], 255, "pixel {i} A");
        }
    }

    #[test]
    fn test_large_rgb_to_rgba_hits_simd() {
        let src: Vec<u8> = (0..96).map(|i| i as u8).collect();
        let mut dst = vec![0u8; 128];
        rgb_to_rgba(&src, &mut dst);
        for (i, chunk) in dst.chunks(4).enumerate() {
            let s = &src[i * 3..i * 3 + 3];
            assert_eq!(chunk[0], s[0], "pixel {i} R");
            assert_eq!(chunk[1], s[1], "pixel {i} G");
            assert_eq!(chunk[2], s[2], "pixel {i} B");
            assert_eq!(chunk[3], 255, "pixel {i} A");
        }
    }
}
