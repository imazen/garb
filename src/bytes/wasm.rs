use core::arch::wasm32::{i8x16, i8x16_swizzle, u32x4_splat, v128_or};

use archmage::prelude::*;
use safe_unaligned_simd::wasm32::{v128_load, v128_store};

use super::swap_br_u32;

// ===========================================================================
// WASM SIMD128 — rite row implementations
// ===========================================================================

#[rite]
pub(super) fn swap_br_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        v128_store(out, i8x16_swizzle(v, mask));
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[rite]
pub(super) fn copy_swap_br_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = v128_load(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        v128_store(d, i8x16_swizzle(v, mask));
        i += 16;
    }
    for (s, d) in bytemuck::cast_slice::<u8, u32>(&src[i..])
        .iter()
        .zip(bytemuck::cast_slice_mut::<u8, u32>(&mut dst[i..]))
    {
        *d = swap_br_u32(*s);
    }
}

#[rite]
pub(super) fn fill_alpha_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let alpha = u32x4_splat(0xFF000000);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        v128_store(out, v128_or(v, alpha));
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[rite]
pub(super) fn rgb_to_bgra_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let shuf = i8x16(2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128);
    let alpha = u32x4_splat(0xFF000000);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        v128_store(d, v128_or(i8x16_swizzle(v128_load(s), shuf), alpha));
        is += 12;
        id += 16;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(d32.iter_mut()) {
        *d = s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000;
    }
}

#[rite]
pub(super) fn rgb_to_rgba_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let shuf = i8x16(0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128);
    let alpha = u32x4_splat(0xFF000000);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        v128_store(d, v128_or(i8x16_swizzle(v128_load(s), shuf), alpha));
        is += 12;
        id += 16;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (s, d) in src[is..].chunks_exact(3).zip(d32.iter_mut()) {
        *d = s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000;
    }
}

#[rite]
pub(super) fn gray_to_4bpp_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
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
        let grays = v128_load(s);
        for (j, m) in [m0, m1, m2, m3].iter().enumerate() {
            let d: &mut [u8; 16] = (&mut dst[id + j * 16..id + (j + 1) * 16])
                .try_into()
                .unwrap();
            v128_store(d, v128_or(i8x16_swizzle(grays, *m), alpha));
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

#[rite]
pub(super) fn gray_alpha_to_4bpp_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let m0 = i8x16(0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7);
    let m1 = i8x16(8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let gas = v128_load(s);
        let d0: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        v128_store(d0, i8x16_swizzle(gas, m0));
        let d1: &mut [u8; 16] = (&mut dst[id + 16..id + 32]).try_into().unwrap();
        v128_store(d1, i8x16_swizzle(gas, m1));
        is += 16;
        id += 32;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (ga, d) in src[is..].chunks_exact(2).zip(d32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// 3bpp swap in-place on WASM: 4 pixels per iter with passthrough
#[rite]
pub(super) fn swap_bgr_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let mask = i8x16(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let mut tmp = [0u8; 16];
        v128_store(&mut tmp, i8x16_swizzle(v, mask));
        row[i..i + 12].copy_from_slice(&tmp[..12]);
        i += 12;
    }
    for px in row[i..].chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

// 3bpp swap copy on WASM
#[rite]
pub(super) fn copy_swap_bgr_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15);
    let (slen, dlen) = (src.len(), dst.len());
    let mut i = 0;
    while i + 16 <= slen && i + 16 <= dlen {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = v128_load(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        v128_store(d, i8x16_swizzle(v, mask));
        i += 12;
    }
    for (s, d) in src[i..].chunks_exact(3).zip(dst[i..].chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// 4→3 strip alpha (keep order) on WASM
#[rite]
pub(super) fn rgba_to_rgb_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 12 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let v = v128_load(s);
        let mut tmp = [0u8; 16];
        v128_store(&mut tmp, i8x16_swizzle(v, mask));
        dst[id..id + 12].copy_from_slice(&tmp[..12]);
        is += 16;
        id += 12;
    }
    for (s, d) in src[is..].chunks_exact(4).zip(dst[id..].chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

// 4→3 strip alpha + swap on WASM (BGRA→RGB)
#[rite]
pub(super) fn bgra_to_rgb_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 12 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let v = v128_load(s);
        let mut tmp = [0u8; 16];
        v128_store(&mut tmp, i8x16_swizzle(v, mask));
        dst[id..id + 12].copy_from_slice(&tmp[..12]);
        is += 16;
        id += 12;
    }
    for (s, d) in src[is..].chunks_exact(4).zip(dst[id..].chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// ===========================================================================
// WASM arcane contiguous wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    swap_br_row_wasm128(t, b);
}
#[arcane]
pub(super) fn copy_swap_br_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    fill_alpha_row_wasm128(t, b);
}
#[arcane]
pub(super) fn rgb_to_bgra_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn rgb_to_rgba_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn gray_to_4bpp_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn swap_bgr_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    swap_bgr_row_wasm128(t, b);
}
#[arcane]
pub(super) fn copy_swap_bgr_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    copy_swap_bgr_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn rgba_to_rgb_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgba_to_rgb_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn bgra_to_rgb_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    bgra_to_rgb_row_wasm128(t, s, d);
}

// ===========================================================================
// WASM arcane strided wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        swap_br_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_swap_br_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_br_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_bgra_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_rgba_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_wasm128(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_wasm128(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn swap_bgr_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        swap_bgr_row_wasm128(t, &mut buf[y * stride..][..w * 3]);
    }
}
#[arcane]
pub(super) fn copy_swap_bgr_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_bgr_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn rgba_to_rgb_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgba_to_rgb_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn bgra_to_rgb_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        bgra_to_rgb_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
