// ---------------------------------------------------------------------------
// Row-level pixel swizzle operations with SIMD dispatch.
//
// Architecture: #[rite] row functions contain the SIMD loops.
// #[arcane] wrappers dispatch via incant! — contiguous (single call)
// and strided (loop over rows, single dispatch).
// ---------------------------------------------------------------------------

use crate::SizeError;
use archmage::incant;
use archmage::prelude::*;

// ===========================================================================
// Validation helpers
// ===========================================================================

#[inline]
fn check_inplace(len: usize, bpp: usize) -> Result<(), SizeError> {
    if len % bpp != 0 {
        Err(SizeError)
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
    if src_len % src_bpp != 0 {
        return Err(SizeError);
    }
    if dst_len < (src_len / src_bpp) * dst_bpp {
        return Err(SizeError);
    }
    Ok(())
}

#[inline]
fn check_strided(
    len: usize,
    stride: usize,
    width: usize,
    height: usize,
    bpp: usize,
) -> Result<(), SizeError> {
    let row_bytes = width.checked_mul(bpp).ok_or(SizeError)?;
    if row_bytes > stride {
        return Err(SizeError);
    }
    if height > 0 {
        let total = (height - 1)
            .checked_mul(stride)
            .ok_or(SizeError)?
            .checked_add(row_bytes)
            .ok_or(SizeError)?;
        if len < total {
            return Err(SizeError);
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
// SIMD constants
// ===========================================================================

#[cfg(target_arch = "x86_64")]
const BR_SHUF_MASK_AVX: [i8; 32] = [
    2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14,
    13, 12, 15,
];

#[cfg(target_arch = "x86_64")]
const ALPHA_FF_MASK_AVX: [i8; 32] = [
    0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0,
    0, 0, -1,
];

#[cfg(target_arch = "x86_64")]
const GRAY_EXPAND_MASK_AVX: [i8; 32] = [
    0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3, -128, 4, 4, 4, -128, 5, 5, 5, -128, 6, 6,
    6, -128, 7, 7, 7, -128,
];

#[cfg(target_arch = "x86_64")]
const GA_EXPAND_MASK_AVX: [i8; 32] = [
    0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14,
    14, 14, 15,
];

#[cfg(target_arch = "x86_64")]
const RGB_TO_BGRA_SHUF_AVX: [i8; 32] = [
    2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128, 2, 1, 0, -128, 5, 4, 3, -128, 8,
    7, 6, -128, 11, 10, 9, -128,
];

#[cfg(target_arch = "x86_64")]
const RGB_TO_RGBA_SHUF_AVX: [i8; 32] = [
    0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128, 0, 1, 2, -128, 3, 4, 5, -128, 6,
    7, 8, -128, 9, 10, 11, -128,
];

#[cfg(target_arch = "x86_64")]
const RGB_ALIGN_PERM_AVX: [i8; 32] = [
    0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0,
];

// ===========================================================================
// Scalar row implementations
// ===========================================================================

fn swap_br_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
    }
}

fn copy_swap_br_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let v = u32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
        d.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
    }
}

fn fill_alpha_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        px[3] = 0xFF;
    }
}

fn rgb_to_bgra_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 0xFF;
    }
}

fn rgb_to_rgba_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 0xFF;
    }
}

fn gray_to_4bpp_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (&v, d) in src.iter().zip(dst.chunks_exact_mut(4)) {
        d[0] = v;
        d[1] = v;
        d[2] = v;
        d[3] = 0xFF;
    }
}

fn gray_alpha_to_4bpp_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (ga, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
        d[0] = ga[0];
        d[1] = ga[0];
        d[2] = ga[0];
        d[3] = ga[1];
    }
}

// ===========================================================================
// Scalar contiguous wrappers (dispatch targets for incant!)
// ===========================================================================

fn swap_br_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    swap_br_row_scalar(t, b);
}
fn copy_swap_br_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_scalar(t, s, d);
}
fn fill_alpha_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    fill_alpha_row_scalar(t, b);
}
fn rgb_to_bgra_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_scalar(t, s, d);
}
fn rgb_to_rgba_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_scalar(t, s, d);
}
fn gray_to_4bpp_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_scalar(t, s, d);
}
fn gray_alpha_to_4bpp_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_scalar(t, s, d);
}

// ===========================================================================
// Scalar strided wrappers
// ===========================================================================

fn swap_br_strided_scalar(t: ScalarToken, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        swap_br_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
fn copy_swap_br_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        copy_swap_br_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
fn fill_alpha_strided_scalar(t: ScalarToken, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        fill_alpha_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
fn rgb_to_bgra_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
fn rgb_to_rgba_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
fn gray_to_4bpp_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
fn gray_alpha_to_4bpp_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_scalar(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}

// ===========================================================================
// x86-64 AVX2 — rite row implementations
// ===========================================================================

#[cfg(target_arch = "x86_64")]
#[rite]
fn swap_br_row_v3(_token: X64V3Token, row: &mut [u8]) {
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
#[rite]
fn copy_swap_br_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
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
#[rite]
fn fill_alpha_row_v3(_token: X64V3Token, row: &mut [u8]) {
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

#[cfg(target_arch = "x86_64")]
#[rite]
fn rgb_to_bgra_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_TO_BGRA_SHUF_AVX);
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 32 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let rgb = safe_unaligned_simd::x86_64::_mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let bgr0 = _mm256_shuffle_epi8(aligned, shuf);
        let bgra = _mm256_or_si256(bgr0, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, bgra);
        is += 24;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn rgb_to_rgba_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let perm = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_ALIGN_PERM_AVX);
    let shuf = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&RGB_TO_RGBA_SHUF_AVX);
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 32 <= slen && id + 32 <= dlen {
        let s: &[u8; 32] = src[is..is + 32].try_into().unwrap();
        let rgb = safe_unaligned_simd::x86_64::_mm256_loadu_si256(s);
        let aligned = _mm256_permutevar8x32_epi32(rgb, perm);
        let rgba0 = _mm256_shuffle_epi8(aligned, shuf);
        let rgba = _mm256_or_si256(rgba0, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, rgba);
        is += 24;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(dst32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn gray_to_4bpp_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&GRAY_EXPAND_MASK_AVX);
    let alpha = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&ALPHA_FF_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 8 <= slen && id + 32 <= dlen {
        let gray8 = u64::from_ne_bytes(src[is..is + 8].try_into().unwrap());
        let grays = _mm256_set1_epi64x(gray8 as i64);
        let expanded = _mm256_shuffle_epi8(grays, expand);
        let bgra = _mm256_or_si256(expanded, alpha);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, bgra);
        is += 8;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (&v, d) in src[is..].iter().zip(dst32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn gray_alpha_to_4bpp_row_v3(_token: X64V3Token, src: &[u8], dst: &mut [u8]) {
    let expand = safe_unaligned_simd::x86_64::_mm256_loadu_si256(&GA_EXPAND_MASK_AVX);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let lo = u64::from_ne_bytes(src[is..is + 8].try_into().unwrap());
        let hi = u64::from_ne_bytes(src[is + 8..is + 16].try_into().unwrap());
        let gas = _mm256_set_epi64x(hi as i64, lo as i64, hi as i64, lo as i64);
        let bgra = _mm256_shuffle_epi8(gas, expand);
        let d: &mut [u8; 32] = (&mut dst[id..id + 32]).try_into().unwrap();
        safe_unaligned_simd::x86_64::_mm256_storeu_si256(d, bgra);
        is += 16;
        id += 32;
    }
    let dst32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (ga, d) in src[is..].chunks_exact(2).zip(dst32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// x86-64 arcane contiguous wrappers
#[cfg(target_arch = "x86_64")]
#[arcane]
fn swap_br_impl_v3(t: X64V3Token, b: &mut [u8]) {
    swap_br_row_v3(t, b);
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn copy_swap_br_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_v3(t, s, d);
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fill_alpha_impl_v3(t: X64V3Token, b: &mut [u8]) {
    fill_alpha_row_v3(t, b);
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_bgra_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_v3(t, s, d);
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_rgba_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_v3(t, s, d);
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn gray_to_4bpp_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_v3(t, s, d);
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn gray_alpha_to_4bpp_impl_v3(t: X64V3Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_v3(t, s, d);
}

// x86-64 arcane strided wrappers
#[cfg(target_arch = "x86_64")]
#[arcane]
fn swap_br_strided_v3(t: X64V3Token, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        swap_br_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn copy_swap_br_strided_v3(
    t: X64V3Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        copy_swap_br_row_v3(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fill_alpha_strided_v3(t: X64V3Token, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        fill_alpha_row_v3(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_bgra_strided_v3(
    t: X64V3Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb_to_rgba_strided_v3(
    t: X64V3Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_v3(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn gray_to_4bpp_strided_v3(
    t: X64V3Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_v3(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn gray_alpha_to_4bpp_strided_v3(
    t: X64V3Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_v3(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}

// ===========================================================================
// ARM NEON — rite row implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[rite]
fn swap_br_row_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
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
#[rite]
fn copy_swap_br_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
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
#[rite]
fn fill_alpha_row_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
    use core::arch::aarch64::vorrq_u8;
    let ab: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&ab);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(out, vorrq_u8(v, alpha));
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn rgb_to_bgra_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{vorrq_u8, vqtbl1q_u8};
    let sb: [u8; 16] = [2, 1, 0, 0x80, 5, 4, 3, 0x80, 8, 7, 6, 0x80, 11, 10, 9, 0x80];
    let shuf = safe_unaligned_simd::aarch64::vld1q_u8(&sb);
    let ab: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&ab);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d, vorrq_u8(vqtbl1q_u8(v, shuf), alpha));
        is += 12;
        id += 16;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(d32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn rgb_to_rgba_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{vorrq_u8, vqtbl1q_u8};
    let sb: [u8; 16] = [0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80];
    let shuf = safe_unaligned_simd::aarch64::vld1q_u8(&sb);
    let ab: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&ab);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d, vorrq_u8(vqtbl1q_u8(v, shuf), alpha));
        is += 12;
        id += 16;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(d32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn gray_to_4bpp_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{vorrq_u8, vqtbl1q_u8};
    let masks: [[u8; 16]; 4] = [
        [0, 0, 0, 0x80, 1, 1, 1, 0x80, 2, 2, 2, 0x80, 3, 3, 3, 0x80],
        [4, 4, 4, 0x80, 5, 5, 5, 0x80, 6, 6, 6, 0x80, 7, 7, 7, 0x80],
        [
            8, 8, 8, 0x80, 9, 9, 9, 0x80, 10, 10, 10, 0x80, 11, 11, 11, 0x80,
        ],
        [
            12, 12, 12, 0x80, 13, 13, 13, 0x80, 14, 14, 14, 0x80, 15, 15, 15, 0x80,
        ],
    ];
    let m: [_; 4] = core::array::from_fn(|i| safe_unaligned_simd::aarch64::vld1q_u8(&masks[i]));
    let ab: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = safe_unaligned_simd::aarch64::vld1q_u8(&ab);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 64 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let grays = safe_unaligned_simd::aarch64::vld1q_u8(s);
        for j in 0..4 {
            let d: &mut [u8; 16] = (&mut dst[id + j * 16..id + (j + 1) * 16])
                .try_into()
                .unwrap();
            safe_unaligned_simd::aarch64::vst1q_u8(d, vorrq_u8(vqtbl1q_u8(grays, m[j]), alpha));
        }
        is += 16;
        id += 64;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (&v, d) in src[is..].iter().zip(d32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn gray_alpha_to_4bpp_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;
    let masks: [[u8; 16]; 2] = [
        [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7],
        [8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15],
    ];
    let m0 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[0]);
    let m1 = safe_unaligned_simd::aarch64::vld1q_u8(&masks[1]);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let gas = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let d0: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d0, vqtbl1q_u8(gas, m0));
        let d1: &mut [u8; 16] = (&mut dst[id + 16..id + 32]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d1, vqtbl1q_u8(gas, m1));
        is += 16;
        id += 32;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (ga, d) in src[is..].chunks_exact(2).zip(d32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// ARM arcane contiguous wrappers
#[cfg(target_arch = "aarch64")]
#[arcane]
fn swap_br_impl_arm_v2(t: Arm64V2Token, b: &mut [u8]) {
    swap_br_row_arm_v2(t, b);
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn copy_swap_br_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_arm_v2(t, s, d);
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn fill_alpha_impl_arm_v2(t: Arm64V2Token, b: &mut [u8]) {
    fill_alpha_row_arm_v2(t, b);
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb_to_bgra_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_arm_v2(t, s, d);
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb_to_rgba_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_arm_v2(t, s, d);
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn gray_to_4bpp_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_arm_v2(t, s, d);
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn gray_alpha_to_4bpp_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_arm_v2(t, s, d);
}

// ARM arcane strided wrappers
#[cfg(target_arch = "aarch64")]
#[arcane]
fn swap_br_strided_arm_v2(t: Arm64V2Token, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        swap_br_row_arm_v2(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn copy_swap_br_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        copy_swap_br_row_arm_v2(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn fill_alpha_strided_arm_v2(t: Arm64V2Token, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        fill_alpha_row_arm_v2(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb_to_bgra_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_arm_v2(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb_to_rgba_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_arm_v2(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn gray_to_4bpp_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_arm_v2(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "aarch64")]
#[arcane]
fn gray_alpha_to_4bpp_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_arm_v2(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}

// ===========================================================================
// WASM SIMD128 — rite row implementations
// ===========================================================================

#[cfg(target_arch = "wasm32")]
#[rite]
fn swap_br_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle};
    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(out, i8x16_swizzle(v, mask));
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[cfg(target_arch = "wasm32")]
#[rite]
fn copy_swap_br_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle};
    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d, i8x16_swizzle(v, mask));
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
#[rite]
fn fill_alpha_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    use core::arch::wasm32::{u32x4_splat, v128_or};
    let alpha = u32x4_splat(0xFF000000);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::wasm32::v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(out, v128_or(v, alpha));
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[rite]
fn rgb_to_bgra_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};
    let shuf = i8x16(2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128);
    let alpha = u32x4_splat(0xFF000000);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(
            d,
            v128_or(
                i8x16_swizzle(safe_unaligned_simd::wasm32::v128_load(s), shuf),
                alpha,
            ),
        );
        is += 12;
        id += 16;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(d32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[rite]
fn rgb_to_rgba_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};
    let shuf = i8x16(0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128);
    let alpha = u32x4_splat(0xFF000000);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(
            d,
            v128_or(
                i8x16_swizzle(safe_unaligned_simd::wasm32::v128_load(s), shuf),
                alpha,
            ),
        );
        is += 12;
        id += 16;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(d32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[rite]
fn gray_to_4bpp_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};
    let m0 = i8x16(0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3, -128);
    let m1 = i8x16(4, 4, 4, -128, 5, 5, 5, -128, 6, 6, 6, -128, 7, 7, 7, -128);
    let m2 = i8x16(
        8, 8, 8, -128, 9, 9, 9, -128, 10, 10, 10, -128, 11, 11, 11, -128,
    );
    let m3 = i8x16(
        12, 12, 12, -128, 13, 13, 13, -128, 14, 14, 14, -128, 15, 15, 15, -128,
    );
    let alpha = u32x4_splat(0xFF000000);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 64 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let grays = safe_unaligned_simd::wasm32::v128_load(s);
        for (j, m) in [m0, m1, m2, m3].iter().enumerate() {
            let d: &mut [u8; 16] = (&mut dst[id + j * 16..id + (j + 1) * 16])
                .try_into()
                .unwrap();
            safe_unaligned_simd::wasm32::v128_store(d, v128_or(i8x16_swizzle(grays, *m), alpha));
        }
        is += 16;
        id += 64;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (&v, d) in src[is..].iter().zip(d32.iter_mut()) {
        let g = v as u32;
        *d = g | (g << 8) | (g << 16) | 0xFF00_0000;
    }
}

#[cfg(target_arch = "wasm32")]
#[rite]
fn gray_alpha_to_4bpp_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::wasm32::{i8x16, i8x16_swizzle};
    let m0 = i8x16(0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7);
    let m1 = i8x16(8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let gas = safe_unaligned_simd::wasm32::v128_load(s);
        let d0: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d0, i8x16_swizzle(gas, m0));
        let d1: &mut [u8; 16] = (&mut dst[id + 16..id + 32]).try_into().unwrap();
        safe_unaligned_simd::wasm32::v128_store(d1, i8x16_swizzle(gas, m1));
        is += 16;
        id += 32;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (ga, d) in src[is..].chunks_exact(2).zip(d32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// WASM arcane contiguous wrappers
#[cfg(target_arch = "wasm32")]
#[arcane]
fn swap_br_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    swap_br_row_wasm128(t, b);
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn copy_swap_br_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_wasm128(t, s, d);
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn fill_alpha_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    fill_alpha_row_wasm128(t, b);
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb_to_bgra_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_wasm128(t, s, d);
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb_to_rgba_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_wasm128(t, s, d);
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn gray_to_4bpp_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_wasm128(t, s, d);
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn gray_alpha_to_4bpp_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_wasm128(t, s, d);
}

// WASM arcane strided wrappers
#[cfg(target_arch = "wasm32")]
#[arcane]
fn swap_br_strided_wasm128(t: Wasm128Token, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        swap_br_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn copy_swap_br_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        copy_swap_br_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn fill_alpha_strided_wasm128(t: Wasm128Token, buf: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        fill_alpha_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb_to_bgra_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb_to_rgba_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn gray_to_4bpp_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_wasm128(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn gray_alpha_to_4bpp_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_wasm128(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}

// ===========================================================================
// Public API — contiguous
// ===========================================================================

/// Swap B↔R channels in-place for 4bpp pixels (RGBA↔BGRA).
pub fn rgba_to_bgra_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(swap_br_impl(buf), [v3, arm_v2, wasm128, scalar]);
    Ok(())
}

/// Copy 4bpp pixels, swapping B↔R (RGBA→BGRA or BGRA→RGBA).
pub fn rgba_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 4)?;
    incant!(copy_swap_br_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
    Ok(())
}

/// Set the alpha channel (byte 3) to 255 for every 4bpp pixel.
pub fn fill_alpha(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 4)?;
    incant!(fill_alpha_impl(buf), [v3, arm_v2, wasm128, scalar]);
    Ok(())
}

/// RGB (3 bytes/px) → BGRA (4 bytes/px). Reverses channel order, alpha=255.
pub fn rgb_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 4)?;
    incant!(rgb_to_bgra_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
    Ok(())
}

/// RGB (3 bytes/px) → RGBA (4 bytes/px). Keeps channel order, alpha=255.
pub fn rgb_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 4)?;
    incant!(rgb_to_rgba_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
    Ok(())
}

/// Gray (1 byte/px) → RGBA/BGRA (4 bytes/px). R=G=B=gray, alpha=255.
pub fn gray_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 1, dst.len(), 4)?;
    incant!(gray_to_4bpp_impl(src, dst), [v3, arm_v2, wasm128, scalar]);
    Ok(())
}

/// GrayAlpha (2 bytes/px) → RGBA/BGRA (4 bytes/px). R=G=B=gray.
pub fn gray_alpha_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 2, dst.len(), 4)?;
    incant!(
        gray_alpha_to_4bpp_impl(src, dst),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

// ===========================================================================
// Public API — strided
// ===========================================================================

/// Swap B↔R in-place for a strided 4bpp image. Single SIMD dispatch.
pub fn rgba_to_bgra_inplace_strided(
    buf: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), stride, width, height, 4)?;
    incant!(
        swap_br_strided(buf, stride, width, height),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

/// Copy 4bpp pixels with stride, swapping B↔R. Single SIMD dispatch.
pub fn rgba_to_bgra_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 4)?;
    check_strided(dst.len(), ds, w, h, 4)?;
    incant!(
        copy_swap_br_strided(src, ss, dst, ds, w, h),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

/// Fill alpha with stride. Single SIMD dispatch.
pub fn fill_alpha_strided(
    buf: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), stride, width, height, 4)?;
    incant!(
        fill_alpha_strided(buf, stride, width, height),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

/// RGB→BGRA with stride. Single SIMD dispatch.
pub fn rgb_to_bgra_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 3)?;
    check_strided(dst.len(), ds, w, h, 4)?;
    incant!(
        rgb_to_bgra_strided(src, ss, dst, ds, w, h),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

/// RGB→RGBA with stride. Single SIMD dispatch.
pub fn rgb_to_rgba_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 3)?;
    check_strided(dst.len(), ds, w, h, 4)?;
    incant!(
        rgb_to_rgba_strided(src, ss, dst, ds, w, h),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

/// Gray→RGBA with stride. Single SIMD dispatch.
pub fn gray_to_rgba_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 1)?;
    check_strided(dst.len(), ds, w, h, 4)?;
    incant!(
        gray_to_4bpp_strided(src, ss, dst, ds, w, h),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

/// GrayAlpha→RGBA with stride. Single SIMD dispatch.
pub fn gray_alpha_to_rgba_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 2)?;
    check_strided(dst.len(), ds, w, h, 4)?;
    incant!(
        gray_alpha_to_4bpp_strided(src, ss, dst, ds, w, h),
        [v3, arm_v2, wasm128, scalar]
    );
    Ok(())
}

// ===========================================================================
// Scalar-only operations
// ===========================================================================

/// Swap R↔B in-place for 3bpp pixels (RGB↔BGR).
pub fn rgb_to_bgr_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    check_inplace(buf.len(), 3)?;
    for px in buf.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    Ok(())
}

/// Copy 3bpp pixels, swapping R↔B (RGB→BGR or BGR→RGB).
pub fn rgb_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 3, dst.len(), 3)?;
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
    Ok(())
}

/// 4bpp → 3bpp by dropping byte 3 (alpha). Keeps byte order.
pub fn rgba_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 3)?;
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
    Ok(())
}

/// 4bpp → 3bpp, dropping alpha and reversing bytes 0↔2 (BGRA→RGB, RGBA→BGR).
pub fn bgra_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    check_copy(src.len(), 4, dst.len(), 3)?;
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
    Ok(())
}

/// Strided 3bpp swap R↔B in-place.
pub fn rgb_to_bgr_inplace_strided(
    buf: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) -> Result<(), SizeError> {
    check_strided(buf.len(), stride, width, height, 3)?;
    for y in 0..height {
        for px in buf[y * stride..][..width * 3].chunks_exact_mut(3) {
            px.swap(0, 2);
        }
    }
    Ok(())
}

/// Strided 4bpp → 3bpp, dropping alpha.
pub fn rgba_to_rgb_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 4)?;
    check_strided(dst.len(), ds, w, h, 3)?;
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 3].chunks_exact_mut(3))
        {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
    Ok(())
}

/// Strided BGRA→RGB (4→3, drop alpha + swap).
pub fn bgra_to_rgb_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    check_strided(src.len(), ss, w, h, 4)?;
    check_strided(dst.len(), ds, w, h, 3)?;
    for y in 0..h {
        for (s, d) in src[y * ss..][..w * 4]
            .chunks_exact(4)
            .zip(dst[y * ds..][..w * 3].chunks_exact_mut(3))
        {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
        }
    }
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
#[inline(always)]
pub fn bgra_to_rgba_inplace_strided(
    buf: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    rgba_to_bgra_inplace_strided(buf, stride, w, h)
}
#[inline(always)]
pub fn bgra_to_rgba_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    rgba_to_bgra_strided(src, ss, dst, ds, w, h)
}
#[inline(always)]
pub fn bgr_to_rgba_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    rgb_to_bgra_strided(src, ss, dst, ds, w, h)
}
#[inline(always)]
pub fn bgr_to_bgra_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    rgb_to_rgba_strided(src, ss, dst, ds, w, h)
}
#[inline(always)]
pub fn gray_to_bgra_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    gray_to_rgba_strided(src, ss, dst, ds, w, h)
}
#[inline(always)]
pub fn gray_alpha_to_bgra_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    gray_alpha_to_rgba_strided(src, ss, dst, ds, w, h)
}
#[inline(always)]
pub fn bgra_to_bgr_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    rgba_to_rgb_strided(src, ss, dst, ds, w, h)
}
#[inline(always)]
pub fn rgba_to_bgr_strided(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) -> Result<(), SizeError> {
    bgra_to_rgb_strided(src, ss, dst, ds, w, h)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_rgba_to_bgra_inplace() {
        let mut row = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        rgba_to_bgra_inplace(&mut row).unwrap();
        assert_eq!(row, [30, 20, 10, 255, 60, 50, 40, 128]);
        rgba_to_bgra_inplace(&mut row).unwrap();
        assert_eq!(row, [10, 20, 30, 255, 40, 50, 60, 128]);
    }

    #[test]
    fn test_rgba_to_bgra_copy() {
        let src = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut dst = vec![0u8; 8];
        rgba_to_bgra(&src, &mut dst).unwrap();
        assert_eq!(dst, [30, 20, 10, 255, 60, 50, 40, 128]);
    }

    #[test]
    fn test_fill_alpha() {
        let mut row = vec![10u8, 20, 30, 0, 40, 50, 60, 100];
        fill_alpha(&mut row).unwrap();
        assert_eq!(row, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn test_rgb_to_bgra() {
        let src = vec![255u8, 128, 0, 10, 20, 30];
        let mut dst = vec![0u8; 8];
        rgb_to_bgra(&src, &mut dst).unwrap();
        assert_eq!(dst, [0, 128, 255, 255, 30, 20, 10, 255]);
    }

    #[test]
    fn test_rgb_to_rgba() {
        let src = vec![255u8, 128, 0, 10, 20, 30];
        let mut dst = vec![0u8; 8];
        rgb_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [255, 128, 0, 255, 10, 20, 30, 255]);
    }

    #[test]
    fn test_gray_to_rgba() {
        let src = vec![100u8, 200];
        let mut dst = vec![0u8; 8];
        gray_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [100, 100, 100, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn test_gray_alpha_to_rgba() {
        let src = vec![100u8, 50, 200, 128];
        let mut dst = vec![0u8; 8];
        gray_alpha_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [100, 100, 100, 50, 200, 200, 200, 128]);
    }

    #[test]
    fn test_rgb_to_bgr_inplace() {
        let mut row = vec![10u8, 20, 30, 40, 50, 60];
        rgb_to_bgr_inplace(&mut row).unwrap();
        assert_eq!(row, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_rgba_to_rgb() {
        let src = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut dst = vec![0u8; 6];
        rgba_to_rgb(&src, &mut dst).unwrap();
        assert_eq!(dst, [10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_bgra_to_rgb() {
        let src = vec![10u8, 20, 30, 255, 40, 50, 60, 128];
        let mut dst = vec![0u8; 6];
        bgra_to_rgb(&src, &mut dst).unwrap();
        assert_eq!(dst, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_size_error() {
        assert_eq!(rgba_to_bgra_inplace(&mut [0; 5]), Err(SizeError));
        assert_eq!(rgb_to_bgra(&[0; 6], &mut [0; 4]), Err(SizeError));
        assert_eq!(rgb_to_bgr_inplace(&mut [0; 5]), Err(SizeError));
    }

    #[test]
    fn test_large_simd() {
        let mut row: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let orig = row.clone();
        rgba_to_bgra_inplace(&mut row).unwrap();
        for (i, chunk) in row.chunks(4).enumerate() {
            let o = &orig[i * 4..i * 4 + 4];
            assert_eq!(
                [chunk[0], chunk[1], chunk[2], chunk[3]],
                [o[2], o[1], o[0], o[3]]
            );
        }
    }

    #[test]
    fn test_large_rgb_to_bgra_simd() {
        let src: Vec<u8> = (0..96).map(|i| i as u8).collect();
        let mut dst = vec![0u8; 128];
        rgb_to_bgra(&src, &mut dst).unwrap();
        for (i, chunk) in dst.chunks(4).enumerate() {
            let s = &src[i * 3..i * 3 + 3];
            assert_eq!(
                [chunk[0], chunk[1], chunk[2], chunk[3]],
                [s[2], s[1], s[0], 255],
                "pixel {i}"
            );
        }
    }

    #[test]
    fn test_strided_inplace() {
        // 2 pixels wide, stride 12 bytes (3 pixels), 2 rows
        let mut buf = vec![
            10u8, 20, 30, 255, 40, 50, 60, 128, 0, 0, 0, 0, 70, 80, 90, 200, 100, 110, 120, 150, 0,
            0, 0, 0,
        ];
        rgba_to_bgra_inplace_strided(&mut buf, 12, 2, 2).unwrap();
        assert_eq!(&buf[0..4], &[30, 20, 10, 255]);
        assert_eq!(&buf[4..8], &[60, 50, 40, 128]);
        assert_eq!(&buf[8..12], &[0, 0, 0, 0]); // padding untouched
        assert_eq!(&buf[12..16], &[90, 80, 70, 200]);
    }

    #[test]
    fn test_strided_copy() {
        let src = vec![1u8, 2, 3, 4, 5, 6, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0];
        let mut dst = vec![0u8; 16]; // 2 px × 2 rows × 4bpp, no padding
        rgb_to_bgra_strided(&src, 9, &mut dst, 8, 2, 2).unwrap();
        assert_eq!(&dst[0..4], &[3, 2, 1, 255]);
        assert_eq!(&dst[4..8], &[6, 5, 4, 255]);
        assert_eq!(&dst[8..12], &[9, 8, 7, 255]);
        assert_eq!(&dst[12..16], &[12, 11, 10, 255]);
    }
}
