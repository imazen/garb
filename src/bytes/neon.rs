use core::arch::aarch64::{uint8x16x3_t, vorrq_u8, vqtbl1q_u8};

use archmage::prelude::*;
use safe_unaligned_simd::aarch64::{vld1q_u8, vld3q_u8, vst1q_u8, vst3q_u8};

use super::swap_br_u32;

// ===========================================================================
// ARM NEON — rite row implementations
//
// Cross-bpp operations (3↔4 channel, 3bpp copy+swap) are intentionally
// omitted: LLVM's autovectorizer generates faster code than explicit
// vld3q/vld4q/vst3q/vst4q structure loads on all tested aarch64 platforms
// (Ampere, Apple Silicon, Snapdragon). Those ops dispatch directly to scalar.
// ===========================================================================

#[rite]
pub(super) fn swap_br_row_neon(_token: NeonToken, row: &mut [u8]) {
    let mask_bytes: [u8; 16] = [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15];
    let mask = vld1q_u8(&mask_bytes);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = vld1q_u8(arr);
        let shuffled = vqtbl1q_u8(v, mask);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        vst1q_u8(out, shuffled);
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v = swap_br_u32(*v);
    }
}

#[rite]
pub(super) fn copy_swap_br_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
    let mask_bytes: [u8; 16] = [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15];
    let mask = vld1q_u8(&mask_bytes);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = vld1q_u8(s);
        let shuffled = vqtbl1q_u8(v, mask);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        vst1q_u8(d, shuffled);
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
pub(super) fn fill_alpha_row_neon(_token: NeonToken, row: &mut [u8]) {
    let ab: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = vld1q_u8(&ab);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = vld1q_u8(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        vst1q_u8(out, vorrq_u8(v, alpha));
        i += 16;
    }
    for v in bytemuck::cast_slice_mut::<u8, u32>(&mut row[i..]) {
        *v |= 0xFF00_0000;
    }
}

#[rite]
pub(super) fn gray_to_4bpp_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
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
    let m: [_; 4] = core::array::from_fn(|i| vld1q_u8(&masks[i]));
    let ab: [u8; 16] = [0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF];
    let alpha = vld1q_u8(&ab);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 64 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let grays = vld1q_u8(s);
        for j in 0..4 {
            let d: &mut [u8; 16] = (&mut dst[id + j * 16..id + (j + 1) * 16])
                .try_into()
                .unwrap();
            vst1q_u8(d, vorrq_u8(vqtbl1q_u8(grays, m[j]), alpha));
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
pub(super) fn gray_alpha_to_4bpp_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
    let masks: [[u8; 16]; 2] = [
        [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7],
        [8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15],
    ];
    let m0 = vld1q_u8(&masks[0]);
    let m1 = vld1q_u8(&masks[1]);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 32 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let gas = vld1q_u8(s);
        let d0: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        vst1q_u8(d0, vqtbl1q_u8(gas, m0));
        let d1: &mut [u8; 16] = (&mut dst[id + 16..id + 32]).try_into().unwrap();
        vst1q_u8(d1, vqtbl1q_u8(gas, m1));
        is += 16;
        id += 32;
    }
    let d32 = bytemuck::cast_slice_mut::<u8, u32>(&mut dst[id..]);
    for (ga, d) in src[is..].chunks_exact(2).zip(d32.iter_mut()) {
        let g = ga[0] as u32;
        *d = g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24);
    }
}

// 3bpp swap in-place: vld3q deinterleaves channels, swap, vst3q reinterleaves.
// This is 2.3x faster than scalar because LLVM can't autovectorize an inplace
// 3-byte element swap (overlapping reads/writes with non-power-of-2 stride).
#[rite]
pub(super) fn swap_bgr_row_neon(_token: NeonToken, row: &mut [u8]) {
    let n = row.len();
    let mut i = 0;
    while i + 48 <= n {
        let s: &[u8; 48] = row[i..i + 48].try_into().unwrap();
        let uint8x16x3_t(c0, c1, c2) = vld3q_u8(s);
        let d: &mut [u8; 48] = (&mut row[i..i + 48]).try_into().unwrap();
        vst3q_u8(d, uint8x16x3_t(c2, c1, c0));
        i += 48;
    }
    for px in row[i..].chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

// ===========================================================================
// ARM arcane contiguous wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_impl_neon(t: NeonToken, b: &mut [u8]) {
    swap_br_row_neon(t, b);
}
#[arcane]
pub(super) fn copy_swap_br_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_neon(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_impl_neon(t: NeonToken, b: &mut [u8]) {
    fill_alpha_row_neon(t, b);
}
#[arcane]
pub(super) fn gray_to_4bpp_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_neon(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_neon(t, s, d);
}
#[arcane]
pub(super) fn swap_bgr_impl_neon(t: NeonToken, b: &mut [u8]) {
    swap_bgr_row_neon(t, b);
}

// ===========================================================================
// ARM arcane strided wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        swap_br_row_neon(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_swap_br_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_br_row_neon(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_row_neon(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_neon(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_neon(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn swap_bgr_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        swap_bgr_row_neon(t, &mut buf[y * stride..][..w * 3]);
    }
}
