use archmage::prelude::*;

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

// Rotate left by 1 byte per 4-byte pixel: [a,b,c,d]→[b,c,d,a]
const ROTATE_LEFT_SHUF_AVX: [i8; 32] = [
    1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13,
    14, 15, 12,
];

// Rotate right by 1 byte per 4-byte pixel: [a,b,c,d]→[d,a,b,c]
const ROTATE_RIGHT_SHUF_AVX: [i8; 32] = [
    3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15,
    12, 13, 14,
];

// Reverse each 4-byte pixel: [a,b,c,d]→[d,c,b,a]
const REVERSE_4BPP_SHUF_AVX: [i8; 32] = [
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15,
    14, 13, 12,
];

// Alpha-first mask: byte 0 of each 4-byte pixel = 0xFF
const ALPHA_FIRST_MASK_AVX: [i8; 32] = [
    -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1,
    0, 0, 0,
];

// RGB → ARGB: [0xFF, r, g, b] from packed RGB
const RGB_TO_ARGB_SHUF_AVX: [i8; 32] = [
    -128, 0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128, 0, 1, 2, -128, 3, 4, 5,
    -128, 6, 7, 8, -128, 9, 10, 11,
];

// RGB → ABGR: [0xFF, b, g, r] from packed RGB
const RGB_TO_ABGR_SHUF_AVX: [i8; 32] = [
    -128, 2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128, 2, 1, 0, -128, 5, 4, 3,
    -128, 8, 7, 6, -128, 11, 10, 9,
];

// ARGB → RGB: extract bytes 1,2,3 from each 4-byte pixel
const ARGB_TO_RGB_SHUF_AVX: [i8; 32] = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -128, -128, -128, -128, 1, 2, 3, 5, 6, 7, 9, 10, 11,
    13, 14, 15, -128, -128, -128, -128,
];

// ARGB → BGR: extract bytes 3,2,1 from each 4-byte pixel (reverse of channels)
const ARGB_TO_BGR_SHUF_AVX: [i8; 32] = [
    3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, -128, -128, -128, -128, 3, 2, 1, 7, 6, 5, 11, 10, 9,
    15, 14, 13, -128, -128, -128, -128,
];

// Gray expand for alpha-first: [0xFF, g, g, g] from each gray byte
const GRAY_EXPAND_ALPHA_FIRST_MASK_AVX: [i8; 32] = [
    -128, 0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3, -128, 4, 4, 4, -128, 5, 5, 5, -128,
    6, 6, 6, -128, 7, 7, 7,
];

// GrayAlpha expand for alpha-first: [a, g, g, g] from [g, a] pairs
const GA_EXPAND_ALPHA_FIRST_MASK_AVX: [i8; 32] = [
    1, 0, 0, 0, 3, 2, 2, 2, 5, 4, 4, 4, 7, 6, 6, 6, 9, 8, 8, 8, 11, 10, 10, 10, 13, 12, 12, 12, 15,
    14, 14, 14,
];

// ===========================================================================
// x86-64 AVX2 — core rite row implementations
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
    for px in row[i..].chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        let v = u32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
        d.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
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
    for px in row[i..].chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&(v | 0xFF00_0000).to_ne_bytes());
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
    for (s, d) in src[is..].chunks_exact(3).zip(dst[id..].chunks_exact_mut(4)) {
        d.copy_from_slice(
            &(s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000)
                .to_ne_bytes(),
        );
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
    for (s, d) in src[is..].chunks_exact(3).zip(dst[id..].chunks_exact_mut(4)) {
        d.copy_from_slice(
            &(s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000)
                .to_ne_bytes(),
        );
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
    for (&v, d) in src[is..].iter().zip(dst[id..].chunks_exact_mut(4)) {
        let g = v as u32;
        d.copy_from_slice(&(g | (g << 8) | (g << 16) | 0xFF00_0000).to_ne_bytes());
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
    for (ga, d) in src[is..].chunks_exact(2).zip(dst[id..].chunks_exact_mut(4)) {
        let g = ga[0] as u32;
        d.copy_from_slice(&(g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24)).to_ne_bytes());
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
// x86-64 AVX2 — ARGB/XRGB rite row implementations
// ===========================================================================

#[rite]
pub(super) fn rotate_left_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&ROTATE_LEFT_SHUF_AVX);
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
    for px in row[i..].chunks_exact_mut(4) {
        let a = px[0];
        px[0] = px[1];
        px[1] = px[2];
        px[2] = px[3];
        px[3] = a;
    }
}

#[rite]
pub(super) fn copy_rotate_left_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&ROTATE_LEFT_SHUF_AVX);
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
        d[3] = s[0];
    }
}

#[rite]
pub(super) fn rotate_right_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&ROTATE_RIGHT_SHUF_AVX);
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
    for px in row[i..].chunks_exact_mut(4) {
        let d = px[3];
        px[3] = px[2];
        px[2] = px[1];
        px[1] = px[0];
        px[0] = d;
    }
}

#[rite]
pub(super) fn copy_rotate_right_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&ROTATE_RIGHT_SHUF_AVX);
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[3];
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
}

#[rite]
pub(super) fn reverse_4bpp_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&REVERSE_4BPP_SHUF_AVX);
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
    for px in row[i..].chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&v.swap_bytes().to_ne_bytes());
    }
}

#[rite]
pub(super) fn copy_reverse_4bpp_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&REVERSE_4BPP_SHUF_AVX);
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
}

#[rite]
pub(super) fn fill_alpha_first_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let alpha = _mm256_loadu_si256(&ALPHA_FIRST_MASK_AVX);
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
    for px in row[i..].chunks_exact_mut(4) {
        px[0] = 0xFF;
    }
}

#[rite]
pub(super) fn rgb_to_argb_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = _mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = _mm256_loadu_si256(&RGB_TO_ARGB_SHUF_AVX);
    let alpha = _mm256_loadu_si256(&ALPHA_FIRST_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 32 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let rgb = _mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let argb0 = _mm256_shuffle_epi8(aligned, shuf);
        let argb = _mm256_or_si256(argb0, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, argb);
        is += 24;
        id += 32;
    }
    for (s, d) in src[is..].chunks_exact(3).zip(dst[id..].chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
}

#[rite]
pub(super) fn rgb_to_abgr_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = _mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = _mm256_loadu_si256(&RGB_TO_ABGR_SHUF_AVX);
    let alpha = _mm256_loadu_si256(&ALPHA_FIRST_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 32 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let rgb = _mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let abgr0 = _mm256_shuffle_epi8(aligned, shuf);
        let abgr = _mm256_or_si256(abgr0, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, abgr);
        is += 24;
        id += 32;
    }
    for (s, d) in src[is..].chunks_exact(3).zip(dst[id..].chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
}

#[rite]
pub(super) fn argb_to_rgb_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let shuf = _mm256_loadu_si256(&ARGB_TO_RGB_SHUF_AVX);
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
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
    }
}

#[rite]
pub(super) fn argb_to_bgr_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let shuf = _mm256_loadu_si256(&ARGB_TO_BGR_SHUF_AVX);
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
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
    }
}

#[rite]
pub(super) fn gray_to_4bpp_alpha_first_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = _mm256_loadu_si256(&GRAY_EXPAND_ALPHA_FIRST_MASK_AVX);
    let alpha = _mm256_loadu_si256(&ALPHA_FIRST_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 8 <= slen && id + 32 <= dlen {
        let gray8 = u64::from_ne_bytes(src[is..is + 8].try_into().unwrap());
        let grays = _mm256_set1_epi64x(gray8 as i64);
        let expanded = _mm256_shuffle_epi8(grays, expand);
        let argb = _mm256_or_si256(expanded, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, argb);
        is += 8;
        id += 32;
    }
    for (&v, d) in src[is..].iter().zip(dst[id..].chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = v;
        d[2] = v;
        d[3] = v;
    }
}

#[rite]
pub(super) fn gray_alpha_to_4bpp_alpha_first_row_v3(
    _token: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
) {
    let expand = _mm256_loadu_si256(&GA_EXPAND_ALPHA_FIRST_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let lo = u64::from_ne_bytes(src[is..is + 8].try_into().unwrap());
        let hi = u64::from_ne_bytes(src[is + 8..is + 16].try_into().unwrap());
        let gas = _mm256_set_epi64x(hi as i64, lo as i64, hi as i64, lo as i64);
        let argb = _mm256_shuffle_epi8(gas, expand);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, argb);
        is += 16;
        id += 32;
    }
    for (ga, d) in src[is..].chunks_exact(2).zip(dst[id..].chunks_exact_mut(4)) {
        d[0] = ga[1];
        d[1] = ga[0];
        d[2] = ga[0];
        d[3] = ga[0];
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
#[arcane]
pub(super) fn rotate_left_impl_v3(t: X64V3Token, b: &mut [u8]) {
    rotate_left_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_rotate_left_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_rotate_left_row_v3(t, s, d);
}
#[arcane]
pub(super) fn rotate_right_impl_v3(t: X64V3Token, b: &mut [u8]) {
    rotate_right_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_rotate_right_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_rotate_right_row_v3(t, s, d);
}
#[arcane]
pub(super) fn reverse_4bpp_impl_v3(t: X64V3Token, b: &mut [u8]) {
    reverse_4bpp_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_reverse_4bpp_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_reverse_4bpp_row_v3(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_first_impl_v3(t: X64V3Token, b: &mut [u8]) {
    fill_alpha_first_row_v3(t, b);
}
#[arcane]
pub(super) fn rgb_to_argb_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgb_to_argb_row_v3(t, s, d);
}
#[arcane]
pub(super) fn rgb_to_abgr_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgb_to_abgr_row_v3(t, s, d);
}
#[arcane]
pub(super) fn argb_to_rgb_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    argb_to_rgb_row_v3(t, s, d);
}
#[arcane]
pub(super) fn argb_to_bgr_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    argb_to_bgr_row_v3(t, s, d);
}
#[arcane]
pub(super) fn gray_to_4bpp_alpha_first_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_alpha_first_row_v3(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_alpha_first_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_alpha_first_row_v3(t, s, d);
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

// ===========================================================================
// x86-64 ARGB/XRGB arcane strided wrappers
// ===========================================================================

#[arcane]
pub(super) fn rotate_left_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_left_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rotate_left_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_left_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rotate_right_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_right_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rotate_right_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_right_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn reverse_4bpp_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        reverse_4bpp_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_reverse_4bpp_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_reverse_4bpp_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_first_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_first_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_argb_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_argb_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_abgr_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_abgr_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn argb_to_rgb_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        argb_to_rgb_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn argb_to_bgr_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        argb_to_bgr_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_alpha_first_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_alpha_first_row_v3(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_alpha_first_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_alpha_first_row_v3(
            t,
            &src[y * ss..][..w * 2],
            &mut dst[y * ds..][..w * 4],
        );
    }
}

// ===========================================================================
// BRAG format shuffles
// ===========================================================================

// RGBA→BRAG: [R,G,B,A]→[B,R,A,G] per pixel — indices [2,0,3,1]
const RGBA_TO_BRAG_SHUF_AVX: [i8; 32] = [
    2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14, 12, 15, 13, 2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9, 14,
    12, 15, 13,
];

// BRAG→RGBA: [B,R,A,G]→[R,G,B,A] per pixel — indices [1,3,0,2]
const BRAG_TO_RGBA_SHUF_AVX: [i8; 32] = [
    1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13, 15, 12, 14, 1, 3, 0, 2, 5, 7, 4, 6, 9, 11, 8, 10, 13,
    15, 12, 14,
];

// BGRA→BRAG: [B,G,R,A]→[B,R,A,G] per pixel — indices [0,2,3,1]
const BGRA_TO_BRAG_SHUF_AVX: [i8; 32] = [
    0, 2, 3, 1, 4, 6, 7, 5, 8, 10, 11, 9, 12, 14, 15, 13, 0, 2, 3, 1, 4, 6, 7, 5, 8, 10, 11, 9, 12,
    14, 15, 13,
];

// BRAG→BGRA: [B,R,A,G]→[B,G,R,A] per pixel — indices [0,3,1,2]
const BRAG_TO_BGRA_SHUF_AVX: [i8; 32] = [
    0, 3, 1, 2, 4, 7, 5, 6, 8, 11, 9, 10, 12, 15, 13, 14, 0, 3, 1, 2, 4, 7, 5, 6, 8, 11, 9, 10, 12,
    15, 13, 14,
];

#[rite]
pub(super) fn rgba_to_brag_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&RGBA_TO_BRAG_SHUF_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(arr);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(out, _mm256_shuffle_epi8(v, mask));
        i += 32;
    }
    for px in row[i..].chunks_exact_mut(4) {
        let [r, g, b, a] = [px[0], px[1], px[2], px[3]];
        px[0] = b;
        px[1] = r;
        px[2] = a;
        px[3] = g;
    }
}

#[rite]
pub(super) fn copy_rgba_to_brag_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&RGBA_TO_BRAG_SHUF_AVX);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let s: &[u8; 32] = src[i..i + 32].try_into().unwrap();
        let d: &mut [u8; 32] = (&mut dst[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, _mm256_shuffle_epi8(_mm256_loadu_si256(s), mask));
        i += 32;
    }
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[2];
        d[1] = s[0];
        d[2] = s[3];
        d[3] = s[1];
    }
}

#[rite]
pub(super) fn brag_to_rgba_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BRAG_TO_RGBA_SHUF_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(arr);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(out, _mm256_shuffle_epi8(v, mask));
        i += 32;
    }
    for px in row[i..].chunks_exact_mut(4) {
        let [b, r, a, g] = [px[0], px[1], px[2], px[3]];
        px[0] = r;
        px[1] = g;
        px[2] = b;
        px[3] = a;
    }
}

#[rite]
pub(super) fn copy_brag_to_rgba_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BRAG_TO_RGBA_SHUF_AVX);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let s: &[u8; 32] = src[i..i + 32].try_into().unwrap();
        let d: &mut [u8; 32] = (&mut dst[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, _mm256_shuffle_epi8(_mm256_loadu_si256(s), mask));
        i += 32;
    }
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[1];
        d[1] = s[3];
        d[2] = s[0];
        d[3] = s[2];
    }
}

#[rite]
pub(super) fn bgra_to_brag_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BGRA_TO_BRAG_SHUF_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(arr);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(out, _mm256_shuffle_epi8(v, mask));
        i += 32;
    }
    for px in row[i..].chunks_exact_mut(4) {
        let [b, g, r, a] = [px[0], px[1], px[2], px[3]];
        px[0] = b;
        px[1] = r;
        px[2] = a;
        px[3] = g;
    }
}

#[rite]
pub(super) fn copy_bgra_to_brag_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BGRA_TO_BRAG_SHUF_AVX);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let s: &[u8; 32] = src[i..i + 32].try_into().unwrap();
        let d: &mut [u8; 32] = (&mut dst[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, _mm256_shuffle_epi8(_mm256_loadu_si256(s), mask));
        i += 32;
    }
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[2];
        d[2] = s[3];
        d[3] = s[1];
    }
}

#[rite]
pub(super) fn brag_to_bgra_row_v3(_token: X64V3Token, row: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BRAG_TO_BGRA_SHUF_AVX);
    let n = row.len();
    let mut i = 0;
    while i + 32 <= n {
        let arr: &[u8; 32] = row[i..i + 32].try_into().unwrap();
        let v = _mm256_loadu_si256(arr);
        let out: &mut [u8; 32] = (&mut row[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(out, _mm256_shuffle_epi8(v, mask));
        i += 32;
    }
    for px in row[i..].chunks_exact_mut(4) {
        let [b, r, a, g] = [px[0], px[1], px[2], px[3]];
        px[0] = b;
        px[1] = g;
        px[2] = r;
        px[3] = a;
    }
}

#[rite]
pub(super) fn copy_brag_to_bgra_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let mask = _mm256_loadu_si256(&BRAG_TO_BGRA_SHUF_AVX);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let s: &[u8; 32] = src[i..i + 32].try_into().unwrap();
        let d: &mut [u8; 32] = (&mut dst[i..i + 32]).try_into().unwrap();
        _mm256_storeu_si256(d, _mm256_shuffle_epi8(_mm256_loadu_si256(s), mask));
        i += 32;
    }
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[3];
        d[2] = s[1];
        d[3] = s[2];
    }
}

// BRAG AVX2 wrappers
#[arcane]
pub(super) fn rgba_to_brag_impl_v3(t: X64V3Token, b: &mut [u8]) {
    rgba_to_brag_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_rgba_to_brag_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_rgba_to_brag_row_v3(t, s, d);
}
#[arcane]
pub(super) fn brag_to_rgba_impl_v3(t: X64V3Token, b: &mut [u8]) {
    brag_to_rgba_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_brag_to_rgba_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_brag_to_rgba_row_v3(t, s, d);
}
#[arcane]
pub(super) fn bgra_to_brag_impl_v3(t: X64V3Token, b: &mut [u8]) {
    bgra_to_brag_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_bgra_to_brag_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_bgra_to_brag_row_v3(t, s, d);
}
#[arcane]
pub(super) fn brag_to_bgra_impl_v3(t: X64V3Token, b: &mut [u8]) {
    brag_to_bgra_row_v3(t, b);
}
#[arcane]
pub(super) fn copy_brag_to_bgra_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_brag_to_bgra_row_v3(t, s, d);
}

// BRAG AVX2 strided wrappers
#[arcane]
pub(super) fn rgba_to_brag_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rgba_to_brag_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rgba_to_brag_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rgba_to_brag_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn brag_to_rgba_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        brag_to_rgba_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_brag_to_rgba_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_brag_to_rgba_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn bgra_to_brag_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        bgra_to_brag_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_bgra_to_brag_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_bgra_to_brag_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn brag_to_bgra_strided_v3(
    t: X64V3Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        brag_to_bgra_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_brag_to_bgra_strided_v3(
    t: X64V3Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_brag_to_bgra_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}

// ===========================================================================
// Experimental: depth, luma, premul (feature = "experimental")
// ===========================================================================

#[cfg(feature = "experimental")]
mod experimental {
    use archmage::prelude::*;

    // -----------------------------------------------------------------------
    // Depth conversions — AVX2 rite row implementations
    // -----------------------------------------------------------------------

    #[rite]
    pub(in crate::bytes) fn convert_u8_to_u16_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let mul257 = _mm256_set1_epi16(257);
        let n = src.len();
        let mut i = 0;
        while i + 16 <= n {
            let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let wide = _mm256_cvtepu8_epi16(v);
            let result = _mm256_mullo_epi16(wide, mul257);
            let d: &mut [u8; 32] = (&mut dst[i * 2..i * 2 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, result);
            i += 16;
        }
        for j in i..n {
            dst[j * 2..j * 2 + 2].copy_from_slice(&((src[j] as u16 * 257).to_ne_bytes()));
        }
    }

    #[rite]
    pub(in crate::bytes) fn convert_u16_to_u8_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let mul255 = _mm256_set1_epi32(255);
        let add_half = _mm256_set1_epi32(32768);
        let n = src.len() / 2;
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 16] = src[i * 2..i * 2 + 16].try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let wide = _mm256_cvtepu16_epi32(v);
            let prod = _mm256_add_epi32(_mm256_mullo_epi32(wide, mul255), add_half);
            let shifted = _mm256_srli_epi32::<16>(prod);
            let packed16 = _mm256_packus_epi32(shifted, shifted);
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
            let v = u16::from_ne_bytes([src[j * 2], src[j * 2 + 1]]);
            dst[j] = ((v as u32 * 255 + 32768) >> 16) as u8;
        }
    }

    #[rite]
    pub(in crate::bytes) fn convert_u8_to_f32_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let scale = _mm256_set1_ps(1.0 / 255.0);
        let n = src.len();
        let mut i = 0;
        while i + 8 <= n {
            let mut tmp = [0u8; 16];
            tmp[..8].copy_from_slice(&src[i..i + 8]);
            let v = _mm_loadu_si128(&tmp);
            let wide32 = _mm256_cvtepu8_epi32(v);
            let floats = _mm256_cvtepi32_ps(wide32);
            let result = _mm256_mul_ps(floats, scale);
            let d: &mut [u8; 32] = (&mut dst[i * 4..i * 4 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, _mm256_castps_si256(result));
            i += 8;
        }
        for j in i..n {
            dst[j * 4..j * 4 + 4].copy_from_slice(&(src[j] as f32 / 255.0).to_ne_bytes());
        }
    }

    #[rite]
    pub(in crate::bytes) fn convert_f32_to_u8_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let scale = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let n = src.len() / 4;
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 32] = src[i * 4..i * 4 + 32].try_into().unwrap();
            let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
            let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
            let scaled = _mm256_add_ps(_mm256_mul_ps(clamped, scale), half);
            let ints = _mm256_cvttps_epi32(scaled);
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
            let v =
                f32::from_ne_bytes([src[j * 4], src[j * 4 + 1], src[j * 4 + 2], src[j * 4 + 3]]);
            dst[j] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }

    #[rite]
    pub(in crate::bytes) fn convert_u16_to_f32_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let scale = _mm256_set1_ps(1.0 / 65535.0);
        let n = src.len() / 2;
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 16] = src[i * 2..i * 2 + 16].try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let wide32 = _mm256_cvtepu16_epi32(v);
            let floats = _mm256_cvtepi32_ps(wide32);
            let result = _mm256_mul_ps(floats, scale);
            let d: &mut [u8; 32] = (&mut dst[i * 4..i * 4 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, _mm256_castps_si256(result));
            i += 8;
        }
        for j in i..n {
            let v = u16::from_ne_bytes([src[j * 2], src[j * 2 + 1]]);
            dst[j * 4..j * 4 + 4].copy_from_slice(&(v as f32 / 65535.0).to_ne_bytes());
        }
    }

    #[rite]
    pub(in crate::bytes) fn convert_f32_to_u16_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let scale = _mm256_set1_ps(65535.0);
        let half = _mm256_set1_ps(0.5);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let n = src.len() / 4;
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 32] = src[i * 4..i * 4 + 32].try_into().unwrap();
            let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
            let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
            let scaled = _mm256_add_ps(_mm256_mul_ps(clamped, scale), half);
            let ints = _mm256_cvttps_epi32(scaled);
            let packed16 = _mm256_packus_epi32(ints, ints);
            let perm = _mm256_permute4x64_epi64::<0b00_00_10_00>(packed16);
            let mut tmp = [0u8; 32];
            _mm256_storeu_si256(&mut tmp, perm);
            dst[i * 2..i * 2 + 16].copy_from_slice(&tmp[..16]);
            i += 8;
        }
        for j in i..n {
            let v =
                f32::from_ne_bytes([src[j * 4], src[j * 4 + 1], src[j * 4 + 2], src[j * 4 + 3]]);
            let u16_val = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            dst[j * 2..j * 2 + 2].copy_from_slice(&u16_val.to_ne_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Weighted luma — AVX2 rite row implementations
    // -----------------------------------------------------------------------

    /// 4bpp→gray luma via 16-bit multiply. Processes 8 RGBA pixels per iteration.
    ///
    /// Uses channel extraction + u16 multiply instead of `maddubs` because weights
    /// can exceed 127 (e.g. BT.709 green = 183) and `maddubs` treats the second
    /// operand as signed i8.
    #[rite]
    pub(in crate::bytes) fn luma_4bpp_row_v3(
        _t: X64V3Token,
        src: &[u8],
        dst: &mut [u8],
        w0: u8,
        w1: u8,
        w2: u8,
    ) {
        let r_shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 12, 8, 4, 0,
        );
        let g_shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 9, 5, 1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 13, 9, 5, 1,
        );
        let b_shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 10, 6, 2, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 14, 10, 6, 2,
        );
        let w_r = _mm256_set1_epi16(w0 as i16);
        let w_g = _mm256_set1_epi16(w1 as i16);
        let w_b = _mm256_set1_epi16(w2 as i16);
        let add128 = _mm256_set1_epi16(128);
        let zero = _mm256_setzero_si256();

        let n = src.len() / 4;
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 32] = src[i * 4..i * 4 + 32].try_into().unwrap();
            let pixels = _mm256_loadu_si256(s);

            let r8 = _mm256_shuffle_epi8(pixels, r_shuf);
            let g8 = _mm256_shuffle_epi8(pixels, g_shuf);
            let b8 = _mm256_shuffle_epi8(pixels, b_shuf);

            let r16 = _mm256_unpacklo_epi8(r8, zero);
            let g16 = _mm256_unpacklo_epi8(g8, zero);
            let b16 = _mm256_unpacklo_epi8(b8, zero);

            let sum = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(r16, w_r), _mm256_mullo_epi16(g16, w_g)),
                _mm256_add_epi16(_mm256_mullo_epi16(b16, w_b), add128),
            );

            let shifted = _mm256_srli_epi16::<8>(sum);
            let packed = _mm256_packus_epi16(shifted, zero);

            let lo = _mm256_extracti128_si256::<0>(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let combined = _mm_unpacklo_epi32(lo, hi);
            let mut tmp = [0u8; 16];
            _mm_storeu_si128(&mut tmp, combined);
            dst[i..i + 8].copy_from_slice(&tmp[..8]);
            i += 8;
        }
        for j in i..n {
            let px = &src[j * 4..j * 4 + 4];
            dst[j] = ((px[0] as u16 * w0 as u16
                + px[1] as u16 * w1 as u16
                + px[2] as u16 * w2 as u16
                + 128)
                >> 8) as u8;
        }
    }

    /// 3bpp→gray luma. Scalar fallback (3-byte stride is awkward for AVX2).
    #[rite]
    pub(in crate::bytes) fn luma_3bpp_row_v3(
        _t: X64V3Token,
        src: &[u8],
        dst: &mut [u8],
        w0: u8,
        w1: u8,
        w2: u8,
    ) {
        for (px, d) in src.chunks_exact(3).zip(dst.iter_mut()) {
            *d = ((px[0] as u16 * w0 as u16
                + px[1] as u16 * w1 as u16
                + px[2] as u16 * w2 as u16
                + 128)
                >> 8) as u8;
        }
    }

    macro_rules! luma_v3_wrappers {
        ($matrix:ident, $r:expr, $g:expr, $b:expr) => {
            paste::paste! {
                #[arcane]
                pub(in crate::bytes) fn [<rgb_to_gray_ $matrix _impl_v3>](t: X64V3Token, s: &[u8], d: &mut [u8]) {
                    luma_3bpp_row_v3(t, s, d, $r, $g, $b);
                }
                #[arcane]
                pub(in crate::bytes) fn [<bgr_to_gray_ $matrix _impl_v3>](t: X64V3Token, s: &[u8], d: &mut [u8]) {
                    luma_3bpp_row_v3(t, s, d, $b, $g, $r);
                }
                #[arcane]
                pub(in crate::bytes) fn [<rgba_to_gray_ $matrix _impl_v3>](t: X64V3Token, s: &[u8], d: &mut [u8]) {
                    luma_4bpp_row_v3(t, s, d, $r, $g, $b);
                }
                #[arcane]
                pub(in crate::bytes) fn [<bgra_to_gray_ $matrix _impl_v3>](t: X64V3Token, s: &[u8], d: &mut [u8]) {
                    luma_4bpp_row_v3(t, s, d, $b, $g, $r);
                }
                #[arcane]
                pub(in crate::bytes) fn [<rgb_to_gray_ $matrix _strided_v3>](
                    t: X64V3Token, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize,
                ) {
                    for y in 0..h {
                        luma_3bpp_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w], $r, $g, $b);
                    }
                }
                #[arcane]
                pub(in crate::bytes) fn [<bgr_to_gray_ $matrix _strided_v3>](
                    t: X64V3Token, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize,
                ) {
                    for y in 0..h {
                        luma_3bpp_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w], $b, $g, $r);
                    }
                }
                #[arcane]
                pub(in crate::bytes) fn [<rgba_to_gray_ $matrix _strided_v3>](
                    t: X64V3Token, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize,
                ) {
                    for y in 0..h {
                        luma_4bpp_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w], $r, $g, $b);
                    }
                }
                #[arcane]
                pub(in crate::bytes) fn [<bgra_to_gray_ $matrix _strided_v3>](
                    t: X64V3Token, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize,
                ) {
                    for y in 0..h {
                        luma_4bpp_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w], $b, $g, $r);
                    }
                }
            }
        };
    }

    luma_v3_wrappers!(bt709, 54, 183, 19);
    luma_v3_wrappers!(bt601, 77, 150, 29);
    luma_v3_wrappers!(bt2020, 67, 174, 15);

    // -----------------------------------------------------------------------
    // f32 alpha premultiplication — AVX2 rite row implementations
    // -----------------------------------------------------------------------

    #[rite]
    pub(in crate::bytes) fn premul_f32_row_v3(_t: X64V3Token, buf: &mut [u8]) {
        let n = buf.len() / 16;
        let mut i = 0;
        while i + 2 <= n {
            let s: &[u8; 32] = buf[i * 16..i * 16 + 32].try_into().unwrap();
            let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
            let alpha = _mm256_permute_ps::<0xFF>(v);
            let premul = _mm256_mul_ps(v, alpha);
            let result = _mm256_blend_ps::<0b1000_1000>(premul, v);
            let d: &mut [u8; 32] = (&mut buf[i * 16..i * 16 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, _mm256_castps_si256(result));
            i += 2;
        }
        if i < n {
            for px in buf[i * 16..].chunks_exact_mut(16) {
                let r = f32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
                let g = f32::from_ne_bytes([px[4], px[5], px[6], px[7]]);
                let b = f32::from_ne_bytes([px[8], px[9], px[10], px[11]]);
                let a = f32::from_ne_bytes([px[12], px[13], px[14], px[15]]);
                px[0..4].copy_from_slice(&(r * a).to_ne_bytes());
                px[4..8].copy_from_slice(&(g * a).to_ne_bytes());
                px[8..12].copy_from_slice(&(b * a).to_ne_bytes());
            }
        }
    }

    #[rite]
    pub(in crate::bytes) fn premul_f32_copy_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let n = src.len() / 16;
        let mut i = 0;
        while i + 2 <= n {
            let s: &[u8; 32] = src[i * 16..i * 16 + 32].try_into().unwrap();
            let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
            let alpha = _mm256_permute_ps::<0xFF>(v);
            let premul = _mm256_mul_ps(v, alpha);
            let result = _mm256_blend_ps::<0b1000_1000>(premul, v);
            let d: &mut [u8; 32] = (&mut dst[i * 16..i * 16 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, _mm256_castps_si256(result));
            i += 2;
        }
        if i < n {
            for (s, d) in src[i * 16..]
                .chunks_exact(16)
                .zip(dst[i * 16..].chunks_exact_mut(16))
            {
                let r = f32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
                let g = f32::from_ne_bytes([s[4], s[5], s[6], s[7]]);
                let b = f32::from_ne_bytes([s[8], s[9], s[10], s[11]]);
                let a = f32::from_ne_bytes([s[12], s[13], s[14], s[15]]);
                d[0..4].copy_from_slice(&(r * a).to_ne_bytes());
                d[4..8].copy_from_slice(&(g * a).to_ne_bytes());
                d[8..12].copy_from_slice(&(b * a).to_ne_bytes());
                d[12..16].copy_from_slice(&a.to_ne_bytes());
            }
        }
    }

    #[rite]
    pub(in crate::bytes) fn unpremul_f32_row_v3(_t: X64V3Token, buf: &mut [u8]) {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let n = buf.len() / 16;
        let mut i = 0;
        while i + 2 <= n {
            let s: &[u8; 32] = buf[i * 16..i * 16 + 32].try_into().unwrap();
            let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
            let alpha = _mm256_permute_ps::<0xFF>(v);
            let zero_mask = _mm256_cmp_ps::<0>(alpha, zero); // _CMP_EQ_OQ
            let inv_alpha = _mm256_div_ps(one, alpha);
            let unpremul = _mm256_mul_ps(v, inv_alpha);
            let blended = _mm256_blend_ps::<0b1000_1000>(unpremul, v);
            let result = _mm256_andnot_ps(zero_mask, blended);
            let d: &mut [u8; 32] = (&mut buf[i * 16..i * 16 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, _mm256_castps_si256(result));
            i += 2;
        }
        if i < n {
            for px in buf[i * 16..].chunks_exact_mut(16) {
                let r = f32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
                let g = f32::from_ne_bytes([px[4], px[5], px[6], px[7]]);
                let b = f32::from_ne_bytes([px[8], px[9], px[10], px[11]]);
                let a = f32::from_ne_bytes([px[12], px[13], px[14], px[15]]);
                if a == 0.0 {
                    px[0..12].fill(0);
                } else {
                    let inv_a = 1.0 / a;
                    px[0..4].copy_from_slice(&(r * inv_a).to_ne_bytes());
                    px[4..8].copy_from_slice(&(g * inv_a).to_ne_bytes());
                    px[8..12].copy_from_slice(&(b * inv_a).to_ne_bytes());
                }
            }
        }
    }

    #[rite]
    pub(in crate::bytes) fn unpremul_f32_copy_row_v3(_t: X64V3Token, src: &[u8], dst: &mut [u8]) {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let n = src.len() / 16;
        let mut i = 0;
        while i + 2 <= n {
            let s: &[u8; 32] = src[i * 16..i * 16 + 32].try_into().unwrap();
            let v = _mm256_castsi256_ps(_mm256_loadu_si256(s));
            let alpha = _mm256_permute_ps::<0xFF>(v);
            let zero_mask = _mm256_cmp_ps::<0>(alpha, zero);
            let inv_alpha = _mm256_div_ps(one, alpha);
            let unpremul = _mm256_mul_ps(v, inv_alpha);
            let blended = _mm256_blend_ps::<0b1000_1000>(unpremul, v);
            let result = _mm256_andnot_ps(zero_mask, blended);
            let d: &mut [u8; 32] = (&mut dst[i * 16..i * 16 + 32]).try_into().unwrap();
            _mm256_storeu_si256(d, _mm256_castps_si256(result));
            i += 2;
        }
        if i < n {
            for (s, d) in src[i * 16..]
                .chunks_exact(16)
                .zip(dst[i * 16..].chunks_exact_mut(16))
            {
                let r = f32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
                let g = f32::from_ne_bytes([s[4], s[5], s[6], s[7]]);
                let b = f32::from_ne_bytes([s[8], s[9], s[10], s[11]]);
                let a = f32::from_ne_bytes([s[12], s[13], s[14], s[15]]);
                if a == 0.0 {
                    d.fill(0);
                } else {
                    let inv_a = 1.0 / a;
                    d[0..4].copy_from_slice(&(r * inv_a).to_ne_bytes());
                    d[4..8].copy_from_slice(&(g * inv_a).to_ne_bytes());
                    d[8..12].copy_from_slice(&(b * inv_a).to_ne_bytes());
                    d[12..16].copy_from_slice(&a.to_ne_bytes());
                }
            }
        }
    }

    // Premul arcane contiguous wrappers
    #[arcane]
    pub(in crate::bytes) fn premul_f32_impl_v3(t: X64V3Token, b: &mut [u8]) {
        premul_f32_row_v3(t, b);
    }
    #[arcane]
    pub(in crate::bytes) fn premul_f32_copy_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        premul_f32_copy_row_v3(t, s, d);
    }
    #[arcane]
    pub(in crate::bytes) fn unpremul_f32_impl_v3(t: X64V3Token, b: &mut [u8]) {
        unpremul_f32_row_v3(t, b);
    }
    #[arcane]
    pub(in crate::bytes) fn unpremul_f32_copy_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        unpremul_f32_copy_row_v3(t, s, d);
    }

    // Premul arcane strided wrappers
    #[arcane]
    pub(in crate::bytes) fn premul_f32_strided_v3(
        t: X64V3Token,
        buf: &mut [u8],
        w: usize,
        h: usize,
        stride: usize,
    ) {
        for y in 0..h {
            premul_f32_row_v3(t, &mut buf[y * stride..][..w * 16]);
        }
    }
    #[arcane]
    pub(in crate::bytes) fn premul_f32_copy_strided_v3(
        t: X64V3Token,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            premul_f32_copy_row_v3(t, &src[y * ss..][..w * 16], &mut dst[y * ds..][..w * 16]);
        }
    }
    #[arcane]
    pub(in crate::bytes) fn unpremul_f32_strided_v3(
        t: X64V3Token,
        buf: &mut [u8],
        w: usize,
        h: usize,
        stride: usize,
    ) {
        for y in 0..h {
            unpremul_f32_row_v3(t, &mut buf[y * stride..][..w * 16]);
        }
    }
    #[arcane]
    pub(in crate::bytes) fn unpremul_f32_copy_strided_v3(
        t: X64V3Token,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            unpremul_f32_copy_row_v3(t, &src[y * ss..][..w * 16], &mut dst[y * ds..][..w * 16]);
        }
    }

    // Depth conversion arcane contiguous wrappers
    #[arcane]
    pub(in crate::bytes) fn convert_u8_to_u16_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        convert_u8_to_u16_row_v3(t, s, d);
    }
    #[arcane]
    pub(in crate::bytes) fn convert_u16_to_u8_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        convert_u16_to_u8_row_v3(t, s, d);
    }
    #[arcane]
    pub(in crate::bytes) fn convert_u8_to_f32_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        convert_u8_to_f32_row_v3(t, s, d);
    }
    #[arcane]
    pub(in crate::bytes) fn convert_f32_to_u8_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        convert_f32_to_u8_row_v3(t, s, d);
    }
    #[arcane]
    pub(in crate::bytes) fn convert_u16_to_f32_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        convert_u16_to_f32_row_v3(t, s, d);
    }
    #[arcane]
    pub(in crate::bytes) fn convert_f32_to_u16_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
        convert_f32_to_u16_row_v3(t, s, d);
    }

    // Depth conversion strided wrappers
    #[arcane]
    pub(in crate::bytes) fn convert_u8_to_u16_strided_v3(
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
    pub(in crate::bytes) fn convert_u16_to_u8_strided_v3(
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
    pub(in crate::bytes) fn convert_u8_to_f32_strided_v3(
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
    pub(in crate::bytes) fn convert_f32_to_u8_strided_v3(
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
    pub(in crate::bytes) fn convert_u16_to_f32_strided_v3(
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
    pub(in crate::bytes) fn convert_f32_to_u16_strided_v3(
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
}

#[cfg(feature = "experimental")]
pub(super) use experimental::*;
