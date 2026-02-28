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
    if len == 0 || len % bpp != 0 {
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
    if src_len == 0 || src_len % src_bpp != 0 {
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
    if width == 0 || height == 0 {
        return Err(SizeError);
    }
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
    extern crate std;
    use super::*;
    use alloc::{vec, vec::Vec};
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    fn policy() -> CompileTimePolicy {
        if std::env::var_os("CI").is_some() {
            CompileTimePolicy::Fail
        } else {
            CompileTimePolicy::WarnStderr
        }
    }

    // --- Helpers to generate test data ---

    fn make_4bpp(n_pixels: usize) -> Vec<u8> {
        (0..n_pixels * 4).map(|i| (i % 251) as u8).collect()
    }

    fn make_3bpp(n_pixels: usize) -> Vec<u8> {
        (0..n_pixels * 3).map(|i| (i % 251) as u8).collect()
    }

    fn make_1bpp(n_pixels: usize) -> Vec<u8> {
        (0..n_pixels).map(|i| (i % 251) as u8).collect()
    }

    fn make_2bpp(n_pixels: usize) -> Vec<u8> {
        (0..n_pixels * 2).map(|i| (i % 251) as u8).collect()
    }

    // --- Reference (scalar-only) implementations for comparison ---

    fn ref_swap_br(data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        for px in out.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        out
    }

    fn ref_copy_swap_br(src: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = s[3];
        }
        out
    }

    fn ref_fill_alpha(data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        for px in out.chunks_exact_mut(4) {
            px[3] = 255;
        }
        out
    }

    fn ref_rgb_to_bgra(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 3;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(3).zip(out.chunks_exact_mut(4)) {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 255;
        }
        out
    }

    fn ref_rgb_to_rgba(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 3;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(3).zip(out.chunks_exact_mut(4)) {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = 255;
        }
        out
    }

    fn ref_gray_to_4bpp(src: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; src.len() * 4];
        for (s, d) in src.iter().zip(out.chunks_exact_mut(4)) {
            d[0] = *s;
            d[1] = *s;
            d[2] = *s;
            d[3] = 255;
        }
        out
    }

    fn ref_gray_alpha_to_4bpp(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 2;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
            d[0] = s[0];
            d[1] = s[0];
            d[2] = s[0];
            d[3] = s[1];
        }
        out
    }

    // Test sizes: small (remainder only), medium (SIMD + remainder), large (multiple SIMD chunks)
    const TEST_PIXEL_COUNTS: &[usize] = &[1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100];

    // -----------------------------------------------------------------------
    // SIMD-dispatched operations — tested at every capability tier
    // -----------------------------------------------------------------------

    #[test]
    fn permutation_swap_br_inplace() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let mut data = make_4bpp(n);
                let expected = ref_swap_br(&data);
                rgba_to_bgra_inplace(&mut data).unwrap();
                assert_eq!(data, expected, "swap_br_inplace n={n} tier={perm}");
            }
        });
        std::eprintln!("swap_br_inplace: {report}");
    }

    #[test]
    fn permutation_copy_swap_br() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_4bpp(n);
                let expected = ref_copy_swap_br(&src);
                let mut dst = vec![0u8; n * 4];
                rgba_to_bgra(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "copy_swap_br n={n} tier={perm}");
            }
        });
        std::eprintln!("copy_swap_br: {report}");
    }

    #[test]
    fn permutation_fill_alpha() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let mut data = make_4bpp(n);
                let expected = ref_fill_alpha(&data);
                fill_alpha(&mut data).unwrap();
                assert_eq!(data, expected, "fill_alpha n={n} tier={perm}");
            }
        });
        std::eprintln!("fill_alpha: {report}");
    }

    #[test]
    fn permutation_rgb_to_bgra() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_3bpp(n);
                let expected = ref_rgb_to_bgra(&src);
                let mut dst = vec![0u8; n * 4];
                rgb_to_bgra(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgb_to_bgra n={n} tier={perm}");
            }
        });
        std::eprintln!("rgb_to_bgra: {report}");
    }

    #[test]
    fn permutation_rgb_to_rgba() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_3bpp(n);
                let expected = ref_rgb_to_rgba(&src);
                let mut dst = vec![0u8; n * 4];
                rgb_to_rgba(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgb_to_rgba n={n} tier={perm}");
            }
        });
        std::eprintln!("rgb_to_rgba: {report}");
    }

    #[test]
    fn permutation_gray_to_rgba() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_1bpp(n);
                let expected = ref_gray_to_4bpp(&src);
                let mut dst = vec![0u8; n * 4];
                gray_to_rgba(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "gray_to_rgba n={n} tier={perm}");
            }
        });
        std::eprintln!("gray_to_rgba: {report}");
    }

    #[test]
    fn permutation_gray_alpha_to_rgba() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_2bpp(n);
                let expected = ref_gray_alpha_to_4bpp(&src);
                let mut dst = vec![0u8; n * 4];
                gray_alpha_to_rgba(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "gray_alpha_to_rgba n={n} tier={perm}");
            }
        });
        std::eprintln!("gray_alpha_to_rgba: {report}");
    }

    // -----------------------------------------------------------------------
    // Strided variants — also tested at every tier
    // -----------------------------------------------------------------------

    #[test]
    fn permutation_strided_swap_br() {
        let report = for_each_token_permutation(policy(), |perm| {
            // 10 pixels wide, stride 48 bytes (12 pixels × 4bpp), 4 rows
            let w = 10;
            let h = 4;
            let stride = 48;
            let mut buf = vec![0xCCu8; stride * h];
            // Fill active area with known data
            for y in 0..h {
                for x in 0..w {
                    let i = y * stride + x * 4;
                    buf[i] = (y * w + x) as u8;
                    buf[i + 1] = 100;
                    buf[i + 2] = 200;
                    buf[i + 3] = 255;
                }
            }
            let orig = buf.clone();
            rgba_to_bgra_inplace_strided(&mut buf, stride, w, h).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let i = y * stride + x * 4;
                    let o = &orig[i..i + 4];
                    assert_eq!(
                        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]],
                        [o[2], o[1], o[0], o[3]],
                        "strided swap y={y} x={x} tier={perm}"
                    );
                }
                // Padding untouched
                for i in (w * 4)..(stride / 1) {
                    assert_eq!(
                        buf[y * stride + i],
                        0xCC,
                        "padding corrupted y={y} i={i} tier={perm}"
                    );
                }
            }
        });
        std::eprintln!("strided_swap_br: {report}");
    }

    #[test]
    fn permutation_strided_copy() {
        let report = for_each_token_permutation(policy(), |perm| {
            let w = 10;
            let h = 3;
            let src_stride = w * 3 + 6; // 6 bytes padding per row
            let dst_stride = w * 4 + 8; // 8 bytes padding per row
            let src = make_3bpp(src_stride / 3 * h); // oversized, use stride
            let mut dst = vec![0xCCu8; dst_stride * h];
            rgb_to_bgra_strided(&src, src_stride, &mut dst, dst_stride, w, h).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let si = y * src_stride + x * 3;
                    let di = y * dst_stride + x * 4;
                    assert_eq!(
                        [dst[di], dst[di + 1], dst[di + 2], dst[di + 3]],
                        [src[si + 2], src[si + 1], src[si], 255],
                        "strided_copy y={y} x={x} tier={perm}"
                    );
                }
            }
        });
        std::eprintln!("strided_copy: {report}");
    }

    #[test]
    fn permutation_strided_fill_alpha() {
        let report = for_each_token_permutation(policy(), |perm| {
            let w = 8;
            let h = 3;
            let stride = w * 4 + 12;
            let mut buf = vec![0u8; stride * h];
            // Set some non-255 alpha values
            for y in 0..h {
                for x in 0..w {
                    let i = y * stride + x * 4;
                    buf[i] = 10;
                    buf[i + 1] = 20;
                    buf[i + 2] = 30;
                    buf[i + 3] = (x * 10) as u8; // varying alpha
                }
            }
            fill_alpha_strided(&mut buf, stride, w, h).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let i = y * stride + x * 4;
                    assert_eq!(
                        buf[i + 3],
                        255,
                        "strided fill_alpha y={y} x={x} tier={perm}"
                    );
                    // RGB untouched
                    assert_eq!(buf[i], 10);
                    assert_eq!(buf[i + 1], 20);
                    assert_eq!(buf[i + 2], 30);
                }
            }
        });
        std::eprintln!("strided_fill_alpha: {report}");
    }

    #[test]
    fn permutation_strided_gray_and_gray_alpha() {
        let report = for_each_token_permutation(policy(), |perm| {
            let w = 12;
            let h = 3;

            // Gray → RGBA strided
            let src_stride = w + 4;
            let dst_stride = w * 4 + 8;
            let src: Vec<u8> = (0..src_stride * h).map(|i| (i % 251) as u8).collect();
            let mut dst = vec![0u8; dst_stride * h];
            gray_to_rgba_strided(&src, src_stride, &mut dst, dst_stride, w, h).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let g = src[y * src_stride + x];
                    let di = y * dst_stride + x * 4;
                    assert_eq!(
                        [dst[di], dst[di + 1], dst[di + 2], dst[di + 3]],
                        [g, g, g, 255],
                        "strided gray y={y} x={x} tier={perm}"
                    );
                }
            }

            // GrayAlpha → RGBA strided
            let src_stride2 = w * 2 + 6;
            let src2: Vec<u8> = (0..src_stride2 * h).map(|i| (i % 251) as u8).collect();
            let mut dst2 = vec![0u8; dst_stride * h];
            gray_alpha_to_rgba_strided(&src2, src_stride2, &mut dst2, dst_stride, w, h).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let si = y * src_stride2 + x * 2;
                    let g = src2[si];
                    let a = src2[si + 1];
                    let di = y * dst_stride + x * 4;
                    assert_eq!(
                        [dst2[di], dst2[di + 1], dst2[di + 2], dst2[di + 3]],
                        [g, g, g, a],
                        "strided gray_alpha y={y} x={x} tier={perm}"
                    );
                }
            }
        });
        std::eprintln!("strided_gray: {report}");
    }

    // -----------------------------------------------------------------------
    // Scalar-only operations (no SIMD dispatch, basic correctness)
    // -----------------------------------------------------------------------

    #[test]
    fn test_rgb_to_bgr_inplace() {
        let mut row = vec![10u8, 20, 30, 40, 50, 60];
        rgb_to_bgr_inplace(&mut row).unwrap();
        assert_eq!(row, [30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn test_rgb_to_bgr_copy() {
        let src = vec![10u8, 20, 30, 40, 50, 60];
        let mut dst = vec![0u8; 6];
        rgb_to_bgr(&src, &mut dst).unwrap();
        assert_eq!(dst, [30, 20, 10, 60, 50, 40]);
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
    fn test_strided_scalar_ops() {
        // rgb_to_bgr_inplace_strided
        let mut buf = vec![
            10u8, 20, 30, 40, 50, 60, 0, 0, 0, 70, 80, 90, 100, 110, 120, 0, 0, 0,
        ];
        rgb_to_bgr_inplace_strided(&mut buf, 9, 2, 2).unwrap();
        assert_eq!(&buf[0..6], &[30, 20, 10, 60, 50, 40]);
        assert_eq!(&buf[9..15], &[90, 80, 70, 120, 110, 100]);

        // rgba_to_rgb_strided
        let src = make_4bpp(3 * 2); // 3 wide, but arranged with stride
        let stride_src = 3 * 4 + 4; // 16 bytes stride
        let mut padded_src = vec![0u8; stride_src * 2];
        padded_src[..12].copy_from_slice(&src[..12]);
        padded_src[stride_src..stride_src + 12].copy_from_slice(&src[12..24]);
        let stride_dst = 3 * 3 + 3;
        let mut dst = vec![0u8; stride_dst * 2];
        rgba_to_rgb_strided(&padded_src, stride_src, &mut dst, stride_dst, 3, 2).unwrap();
        for y in 0..2 {
            for x in 0..3 {
                let si = y * stride_src + x * 4;
                let di = y * stride_dst + x * 3;
                assert_eq!(
                    [dst[di], dst[di + 1], dst[di + 2]],
                    [padded_src[si], padded_src[si + 1], padded_src[si + 2]]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Size validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_errors() {
        assert_eq!(rgba_to_bgra_inplace(&mut [0; 5]), Err(SizeError));
        assert_eq!(rgba_to_bgra_inplace(&mut [0; 0]), Err(SizeError));
        assert_eq!(rgb_to_bgra(&[0; 6], &mut [0; 4]), Err(SizeError));
        assert_eq!(rgb_to_bgr_inplace(&mut [0; 5]), Err(SizeError));
        assert_eq!(gray_to_rgba(&[0; 3], &mut [0; 8]), Err(SizeError));
        assert_eq!(gray_alpha_to_rgba(&[0; 3], &mut [0; 8]), Err(SizeError));
        assert_eq!(fill_alpha(&mut [0; 5]), Err(SizeError));
        assert_eq!(rgba_to_rgb(&[0; 8], &mut [0; 3]), Err(SizeError));
    }

    #[test]
    fn test_strided_size_errors() {
        // stride < width * bpp
        assert_eq!(
            rgba_to_bgra_inplace_strided(&mut [0; 32], 4, 2, 2),
            Err(SizeError)
        );
        // buffer too small
        assert_eq!(
            rgba_to_bgra_inplace_strided(&mut [0; 10], 8, 2, 2),
            Err(SizeError)
        );
        // zero width
        assert_eq!(
            rgba_to_bgra_inplace_strided(&mut [0; 8], 8, 0, 1),
            Err(SizeError)
        );
        // zero height
        assert_eq!(
            rgba_to_bgra_inplace_strided(&mut [0; 8], 8, 2, 0),
            Err(SizeError)
        );
    }

    // -----------------------------------------------------------------------
    // Alias correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_aliases_match() {
        let data = make_4bpp(16);

        // bgra_to_rgba = rgba_to_bgra (symmetric)
        let mut a = data.clone();
        let mut b = data.clone();
        rgba_to_bgra_inplace(&mut a).unwrap();
        bgra_to_rgba_inplace(&mut b).unwrap();
        assert_eq!(a, b);

        // bgr_to_rgba = rgb_to_bgra
        let src3 = make_3bpp(16);
        let mut dst_a = vec![0u8; 64];
        let mut dst_b = vec![0u8; 64];
        bgr_to_rgba(&src3, &mut dst_a).unwrap();
        rgb_to_bgra(&src3, &mut dst_b).unwrap();
        assert_eq!(dst_a, dst_b);

        // bgr_to_bgra = rgb_to_rgba
        let mut dst_c = vec![0u8; 64];
        let mut dst_d = vec![0u8; 64];
        bgr_to_bgra(&src3, &mut dst_c).unwrap();
        rgb_to_rgba(&src3, &mut dst_d).unwrap();
        assert_eq!(dst_c, dst_d);

        // gray_to_bgra = gray_to_rgba
        let gray = make_1bpp(16);
        let mut dst_e = vec![0u8; 64];
        let mut dst_f = vec![0u8; 64];
        gray_to_bgra(&gray, &mut dst_e).unwrap();
        gray_to_rgba(&gray, &mut dst_f).unwrap();
        assert_eq!(dst_e, dst_f);
    }
}
