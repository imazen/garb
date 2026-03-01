use archmage::prelude::*;
use safe_unaligned_simd::x86_64::{
    _mm_loadu_si128, _mm_storeu_si128, _mm256_loadu_si256, _mm256_storeu_si256,
};

use super::swap_br_u32;

// ===========================================================================
// SIMD constants
// ===========================================================================

const BR_SHUF_MASK_AVX: [i8; 32] = [
    2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14,
    13, 12, 15,
];

const ALPHA_FF_MASK_AVX: [i8; 32] = [
    0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0,
    0, 0, -1,
];

const GRAY_EXPAND_MASK_AVX: [i8; 32] = [
    0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3, -128, 4, 4, 4, -128, 5, 5, 5, -128, 6, 6,
    6, -128, 7, 7, 7, -128,
];

const GA_EXPAND_MASK_AVX: [i8; 32] = [
    0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14,
    14, 14, 15,
];

const RGB_TO_BGRA_SHUF_AVX: [i8; 32] = [
    2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128, 2, 1, 0, -128, 5, 4, 3, -128, 8,
    7, 6, -128, 11, 10, 9, -128,
];

const RGB_TO_RGBA_SHUF_AVX: [i8; 32] = [
    0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128, 0, 1, 2, -128, 3, 4, 5, -128, 6,
    7, 8, -128, 9, 10, 11, -128,
];

const RGB_ALIGN_PERM_AVX: [i8; 32] = [
    0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0,
];

// 3bpp swap shuffle: reverse bytes 0↔2 in each 3-byte group (4 pixels per 16 bytes)
// Bytes 12-15 pass through unchanged — safe for in-place 16-byte stores advancing by 12.
const BGR_SWAP_SHUF_SSE: [i8; 16] = [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15];

// RGBA→RGB shuffle: extract bytes 0,1,2 from each 4-byte pixel (4 pixels → 12 bytes)
const RGBA_TO_RGB_SHUF_AVX: [i8; 32] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128, -128, -128, -128, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12,
    13, 14, -128, -128, -128, -128,
];

// BGRA→RGB shuffle: extract bytes 2,1,0 from each 4-byte pixel (swap + strip)
const BGRA_TO_RGB_SHUF_AVX: [i8; 32] = [
    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -128, -128, -128, -128, 2, 1, 0, 6, 5, 4, 10, 9, 8, 14,
    13, 12, -128, -128, -128, -128,
];

// Pack permutation: merge 12 bytes from each 16-byte lane into contiguous 24 bytes
const PACK_3X4_PERM_AVX: [i8; 32] = [
    0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

// ===========================================================================
// x86-64 AVX2 — rite row implementations
// ===========================================================================

#[rite]
pub(super) fn swap_br_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BR_SHUF_MASK_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(arr);
        let shuffled = _mm256_shuffle_epi8(v, mask);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(out, shuffled);
        i += 32;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[rite]
pub(super) fn copy_swap_br_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BR_SHUF_MASK_AVX);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let s: &[u8; 32] = src[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(s);
        let shuffled = _mm256_shuffle_epi8(v, mask);
        let d: &mut [u8; 32] = (&mut dst[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, shuffled);
        i += 32;
    }
    for (s, d) in bytemuck::cast_slice::<u8, u32>(&src[i..])
        .iter()
        .zip(bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i..]))
    {
        *d = swap_br_u32(*s);
    }
}

#[rite]
pub(super) fn fill_alpha_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let alpha = _mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(arr);
        let result = _mm256_or_si256(v, alpha);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(out, result);
        i += 32;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[rite]
pub(super) fn rgb_to_bgra_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = _mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = _mm256_loadu_si256(&RGB_TO_BGRA_SHUF_AVX);
    let alpha = _mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 32 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let rgb = _mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let bgr0 = _mm256_shuffle_epi8(aligned, shuf);
        let bgra = _mm256_or_si256(bgr0, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, bgra);
        is += 24;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[rite]
pub(super) fn rgb_to_rgba_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = _mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = _mm256_loadu_si256(&RGB_TO_RGBA_SHUF_AVX);
    let alpha = _mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 32 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let rgb = _mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let rgba0 = _mm256_shuffle_epi8(aligned, shuf);
        let rgba = _mm256_or_si256(rgba0, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, rgba);
        is += 24;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[rite]
pub(super) fn gray_to_4bpp_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = _mm256_loadu_si256(&GRAY_EXPAND_MASK_AVX);
    let alpha = _mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 8 <= slen && id + 32 <= dlen {
        let gray8 = u64::from_ne_bytes(src[is..is + 8].try_into().unwrap());
        let grays = _mm256_set1_epi64x(gray8 as i64);
        let expanded = _mm256_shuffle_epi8(grays, expand);
        let bgra = _mm256_or_si256(expanded, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, bgra);
        is += 8;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (&v, d) in src[is..].iter().zip(dst32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

#[rite]
pub(super) fn gray_alpha_to_4bpp_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = _mm256_loadu_si256(&GA_EXPAND_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let lo = u64::from_ne_bytes(src[is..is + 8].try_into().unwrap());
        let hi = u64::from_ne_bytes(src[is + 8..is + 16].try_into().unwrap());
        let gas = _mm256_set_epi64x(hi as i64, lo as i64, hi as i64, lo as i64);
        let bgra = _mm256_shuffle_epi8(gas, expand);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, bgra);
        is += 16;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (ga, d) in src[is..].chunks_exact(2).zip(dst32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// 3bpp swap in-place: use 128-bit SSE pshufb with passthrough for safe 16-byte stores
#[rite]
pub(super) fn swap_bgr_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm_loadu_si128(&BGR_SWAP_SHUF_SSE);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = _mm_loadu_si128(arr);
        let shuffled = _mm_shuffle_epi8(v, mask);
        // Store to temp and copy 12 bytes to avoid overlapping 16-byte store-forwarding stalls
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(&mut tmp, shuffled);
        row[i..i + 12].copy_from_slice(&tmp[..12]);
        i += 12;
    }
    for px in row[i..].chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

// 3bpp swap copy: same 128-bit approach, write to separate dst
#[rite]
pub(super) fn copy_swap_bgr_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm_loadu_si128(&BGR_SWAP_SHUF_SSE);
    let (slen, dlen) = (src.len(), dst.len());
    let mut i = 0;
    while i + 16 <= slen && i + 16 <= dlen {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = _mm_loadu_si128(s);
        let shuffled = _mm_shuffle_epi8(v, mask);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        _mm_storeu_si128(d, shuffled);
        i += 12;
    }
    for (s, d) in src[i..].chunks_exact(3).zip(dst[i..].chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// 4→3 strip alpha (keep order): AVX2 pshufb + vpermd pack
#[rite]
pub(super) fn rgba_to_rgb_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let shuf = _mm256_loadu_si256(&RGBA_TO_RGB_SHUF_AVX);
    let pack = _mm256_loadu_si256(&PACK_3X4_PERM_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 24 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(s);
        let stripped = _mm256_shuffle_epi8(v, shuf);
        let packed = _mm256_permutevar8x32_epi32(stripped, pack);
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(&mut tmp, packed);
        dst[id..id + 24].copy_from_slice(&tmp[..24]);
        is += 32;
        id += 24;
    }
    for (s, d) in src[is..].chunks_exact(4).zip(dst[id..].chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

// 4→3 strip alpha + swap: BGRA→RGB
#[rite]
pub(super) fn bgra_to_rgb_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let shuf = _mm256_loadu_si256(&BGRA_TO_RGB_SHUF_AVX);
    let pack = _mm256_loadu_si256(&PACK_3X4_PERM_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 24 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(s);
        let stripped = _mm256_shuffle_epi8(v, shuf);
        let packed = _mm256_permutevar8x32_epi32(stripped, pack);
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(&mut tmp, packed);
        dst[id..id + 24].copy_from_slice(&tmp[..24]);
        is += 32;
        id += 24;
    }
    for (s, d) in src[is..].chunks_exact(4).zip(dst[id..].chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// ===========================================================================
// Depth conversions — AVX2 rite row implementations
// ===========================================================================

#[rite]
pub(super) fn convert_u8_to_u16_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mul257 = _mm256_set1_epi16(257);
    let n = src.len();
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(dst);
    let mut i = 0;
    // Process 16 u8 → 16 u16 per iteration
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = _mm_loadu_si128(s);
        let wide = _mm256_cvtepu8_epi16(v);
        let result = _mm256_mullo_epi16(wide, mul257);
        let d: &mut [u8; 32] = bytemuck::cast_slice_mut(&mut dst16[i..i + 16])
            .try_into()
            .unwrap();
        _mm256_storeu_si256(d, result);
        i += 16;
    }
    for j in i..n {
        dst16[j] = (src[j] as u16) * 257;
    }
}

#[rite]
pub(super) fn convert_u16_to_u8_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let src16: &[u16] = bytemuck::cast_slice(src);
    let mul255 = _mm256_set1_epi32(255);
    let add_half = _mm256_set1_epi32(32768);
    let n = src16.len();
    let mut i = 0;
    // Process 8 u16 → 8 u8 per iteration (need to widen to u32 for multiply)
    while i + 8 <= n {
        let s: &[u8; 16] = bytemuck::cast_slice(&src16[i..i + 8]).try_into().unwrap();
        let v = _mm_loadu_si128(s);
        let wide = _mm256_cvtepu16_epi32(v);
        let prod = _mm256_add_epi32(_mm256_mullo_epi32(wide, mul255), add_half);
        let shifted = _mm256_srli_epi32::<16>(prod);
        // Pack i32 → i16 → u8
        let packed16 = _mm256_packus_epi32(shifted, shifted); // [0..3,0..3,4..7,4..7]
        let packed8 = _mm256_packus_epi16(packed16, packed16); // [0..3,0..3,0..3,0..3,4..7,...]
        // Extract the 8 bytes we need
        let lo = _mm256_extracti128_si256::<0>(packed8);
        let hi = _mm256_extracti128_si256::<1>(packed8);
        let combined = _mm_unpacklo_epi32(lo, hi);
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(&mut tmp, combined);
        dst[i..i + 8].copy_from_slice(&tmp[..8]);
        i += 8;
    }
    for j in i..n {
        dst[j] = ((src16[j] as u32 * 255 + 32768) >> 16) as u8;
    }
}

#[rite]
pub(super) fn convert_u8_to_f32_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let scale = _mm256_set1_ps(1.0 / 255.0);
    let n = src.len();
    let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
    let mut i = 0;
    // Process 8 u8 → 8 f32 per iteration
    while i + 8 <= n {
        let mut tmp = [0u8; 16];
        tmp[..8].copy_from_slice(&src[i..i + 8]);
        let v = _mm_loadu_si128(&tmp);
        let wide32 = _mm256_cvtepu8_epi32(v);
        let floats = _mm256_cvtepi32_ps(wide32);
        let result = _mm256_mul_ps(floats, scale);
        let d: &mut [u8; 32] = bytemuck::cast_slice_mut(&mut dst_f[i..i + 8])
            .try_into()
            .unwrap();
        _mm256_storeu_si256(d, _mm256_castps_si256(result));
        i += 8;
    }
    for j in i..n {
        dst_f[j] = src[j] as f32 / 255.0;
    }
}

#[rite]
pub(super) fn convert_f32_to_u8_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let src_f: &[f32] = bytemuck::cast_slice(src);
    let scale = _mm256_set1_ps(255.0);
    let half = _mm256_set1_ps(0.5);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let n = src_f.len();
    let mut i = 0;
    // Process 8 f32 → 8 u8 per iteration
    while i + 8 <= n {
        let s: &[u8; 32] = bytemuck::cast_slice(&src_f[i..i + 8]).try_into().unwrap();
        let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
        let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
        let scaled = _mm256_add_ps(_mm256_mul_ps(clamped, scale), half);
        let ints = _mm256_cvttps_epi32(scaled);
        // Pack i32 → i16 → u8
        let packed16 = _mm256_packus_epi32(ints, ints);
        let packed8 = _mm256_packus_epi16(packed16, packed16);
        let lo = _mm256_extracti128_si256::<0>(packed8);
        let hi = _mm256_extracti128_si256::<1>(packed8);
        let combined = _mm_unpacklo_epi32(lo, hi);
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(&mut tmp, combined);
        dst[i..i + 8].copy_from_slice(&tmp[..8]);
        i += 8;
    }
    for j in i..n {
        dst[j] = (src_f[j].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

#[rite]
pub(super) fn convert_u16_to_f32_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let src16: &[u16] = bytemuck::cast_slice(src);
    let scale = _mm256_set1_ps(1.0 / 65535.0);
    let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
    let n = src16.len();
    let mut i = 0;
    // Process 8 u16 → 8 f32 per iteration
    while i + 8 <= n {
        let s: &[u8; 16] = bytemuck::cast_slice(&src16[i..i + 8]).try_into().unwrap();
        let v = _mm_loadu_si128(s);
        let wide32 = _mm256_cvtepu16_epi32(v);
        let floats = _mm256_cvtepi32_ps(wide32);
        let result = _mm256_mul_ps(floats, scale);
        let d: &mut [u8; 32] = bytemuck::cast_slice_mut(&mut dst_f[i..i + 8])
            .try_into()
            .unwrap();
        _mm256_storeu_si256(d, _mm256_castps_si256(result));
        i += 8;
    }
    for j in i..n {
        dst_f[j] = src16[j] as f32 / 65535.0;
    }
}

#[rite]
pub(super) fn convert_f32_to_u16_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let src_f: &[f32] = bytemuck::cast_slice(src);
    let scale = _mm256_set1_ps(65535.0);
    let half = _mm256_set1_ps(0.5);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let n = src_f.len();
    let mut i = 0;
    // Process 8 f32 → 8 u16 per iteration
    while i + 8 <= n {
        let s: &[u8; 32] = bytemuck::cast_slice(&src_f[i..i + 8]).try_into().unwrap();
        let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
        let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
        let scaled = _mm256_add_ps(_mm256_mul_ps(clamped, scale), half);
        let ints = _mm256_cvttps_epi32(scaled);
        // Pack i32 → u16
        let packed16 = _mm256_packus_epi32(ints, ints); // [0..3,0..3,4..7,4..7]
        // Permute to get contiguous 8 u16
        let perm = _mm256_permute4x64_epi64::<0b00_00_10_00>(packed16);
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(&mut tmp, perm);
        dst[i * 2..i * 2 + 16].copy_from_slice(&tmp[..16]);
        i += 8;
    }
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(dst);
    for j in i..n {
        dst16[j] = (src_f[j].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

// ===========================================================================
// x86-64 arcane contiguous wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_impl_v3(t: X64V3Token, b: &mut [u8]) {
    swap_br_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_swap_br_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_v3(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_impl_v3(t: X64V3Token, b: &mut [u8]) {
    fill_alpha_row_v3(t, b);
}
#[arcane]
pub(super) fn rgb_to_bgra_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_v3(t, s, d);
}
#[arcane]
pub(super) fn rgb_to_rgba_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_v3(t, s, d);
}
#[arcane]
pub(super) fn gray_to_4bpp_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_v3(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_v3(t, s, d);
}
#[arcane]
pub(super) fn swap_bgr_impl_v3(t: X64V3Token, b: &mut [u8]) {
    swap_bgr_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_swap_bgr_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_swap_bgr_row_v3(t, s, d);
}
#[arcane]
pub(super) fn rgba_to_rgb_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgba_to_rgb_row_v3(t, s, d);
}
#[arcane]
pub(super) fn bgra_to_rgb_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    bgra_to_rgb_row_v3(t, s, d);
}

// Depth conversion arcane contiguous wrappers
#[arcane]
pub(super) fn convert_u8_to_u16_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    convert_u8_to_u16_row_v3(t, s, d);
}
#[arcane]
pub(super) fn convert_u16_to_u8_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    convert_u16_to_u8_row_v3(t, s, d);
}
#[arcane]
pub(super) fn convert_u8_to_f32_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    convert_u8_to_f32_row_v3(t, s, d);
}
#[arcane]
pub(super) fn convert_f32_to_u8_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    convert_f32_to_u8_row_v3(t, s, d);
}
#[arcane]
pub(super) fn convert_u16_to_f32_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    convert_u16_to_f32_row_v3(t, s, d);
}
#[arcane]
pub(super) fn convert_f32_to_u16_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    convert_f32_to_u16_row_v3(t, s, d);
}

// ===========================================================================
// x86-64 arcane strided wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_strided_v3(t: X64V3Token, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        swap_br_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_swap_br_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_br_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_bgra_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_rgba_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_v3(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_v3(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn swap_bgr_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        swap_bgr_row_v3(t, &mut buf[y * stride..][..w * 3]);
    }
}
#[arcane]
pub(super) fn copy_swap_bgr_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_bgr_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn rgba_to_rgb_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgba_to_rgb_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn bgra_to_rgb_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        bgra_to_rgb_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}

// Depth conversion strided wrappers
#[arcane]
pub(super) fn convert_u8_to_u16_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        convert_u8_to_u16_row_v3(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 2]);
    }
}
#[arcane]
pub(super) fn convert_u16_to_u8_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        convert_u16_to_u8_row_v3(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w]);
    }
}
#[arcane]
pub(super) fn convert_u8_to_f32_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        convert_u8_to_f32_row_v3(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn convert_f32_to_u8_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        convert_f32_to_u8_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w]);
    }
}
#[arcane]
pub(super) fn convert_u16_to_f32_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        convert_u16_to_f32_row_v3(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn convert_f32_to_u16_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        convert_f32_to_u16_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 2]);
    }
}
