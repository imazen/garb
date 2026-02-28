use archmage::prelude::*;

use super::swap_br_u32;

// ===========================================================================
// ARM NEON — rite row implementations
// ===========================================================================

#[rite]
pub(super) fn swap_br_row_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
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

#[rite]
pub(super) fn copy_swap_br_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
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

#[rite]
pub(super) fn fill_alpha_row_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
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

#[rite]
pub(super) fn rgb_to_bgra_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
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

#[rite]
pub(super) fn rgb_to_rgba_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
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

#[rite]
pub(super) fn gray_to_4bpp_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
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

#[rite]
pub(super) fn gray_alpha_to_4bpp_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
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

// 3bpp swap in-place on NEON: 4 pixels per iter with passthrough
#[rite]
pub(super) fn swap_bgr_row_arm_v2(_token: Arm64V2Token, row: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;
    let mb: [u8; 16] = [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15];
    let mask = safe_unaligned_simd::aarch64::vld1q_u8(&mb);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(arr);
        let mut tmp = [0u8; 16];
        safe_unaligned_simd::aarch64::vst1q_u8(&mut tmp, vqtbl1q_u8(v, mask));
        row[i..i + 12].copy_from_slice(&tmp[..12]);
        i += 12;
    }
    for px in row[i..].chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

// 3bpp swap copy on NEON
#[rite]
pub(super) fn copy_swap_bgr_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;
    let mb: [u8; 16] = [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15];
    let mask = safe_unaligned_simd::aarch64::vld1q_u8(&mb);
    let (slen, dlen) = (src.len(), dst.len());
    let mut i = 0;
    while i + 16 <= slen && i + 16 <= dlen {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        safe_unaligned_simd::aarch64::vst1q_u8(d, vqtbl1q_u8(v, mask));
        i += 12;
    }
    for (s, d) in src[i..].chunks_exact(3).zip(dst[i..].chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// 4→3 strip alpha (keep order) on NEON
#[rite]
pub(super) fn rgba_to_rgb_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;
    let sb: [u8; 16] = [
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80,
    ];
    let shuf = safe_unaligned_simd::aarch64::vld1q_u8(&sb);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 12 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let mut tmp = [0u8; 16];
        safe_unaligned_simd::aarch64::vst1q_u8(&mut tmp, vqtbl1q_u8(v, shuf));
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

// 4→3 strip alpha + swap on NEON (BGRA→RGB)
#[rite]
pub(super) fn bgra_to_rgb_row_arm_v2(_token: Arm64V2Token, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::vqtbl1q_u8;
    let sb: [u8; 16] = [
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80,
    ];
    let shuf = safe_unaligned_simd::aarch64::vld1q_u8(&sb);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 12 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let v = safe_unaligned_simd::aarch64::vld1q_u8(s);
        let mut tmp = [0u8; 16];
        safe_unaligned_simd::aarch64::vst1q_u8(&mut tmp, vqtbl1q_u8(v, shuf));
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
// ARM arcane contiguous wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_impl_arm_v2(t: Arm64V2Token, b: &mut [u8]) {
    swap_br_row_arm_v2(t, b);
}
#[arcane]
pub(super) fn copy_swap_br_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_impl_arm_v2(t: Arm64V2Token, b: &mut [u8]) {
    fill_alpha_row_arm_v2(t, b);
}
#[arcane]
pub(super) fn rgb_to_bgra_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn rgb_to_rgba_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn gray_to_4bpp_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn swap_bgr_impl_arm_v2(t: Arm64V2Token, b: &mut [u8]) {
    swap_bgr_row_arm_v2(t, b);
}
#[arcane]
pub(super) fn copy_swap_bgr_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    copy_swap_bgr_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn rgba_to_rgb_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    rgba_to_rgb_row_arm_v2(t, s, d);
}
#[arcane]
pub(super) fn bgra_to_rgb_impl_arm_v2(t: Arm64V2Token, s: &[u8], d: &mut [u8]) {
    bgra_to_rgb_row_arm_v2(t, s, d);
}

// ===========================================================================
// ARM arcane strided wrappers
// ===========================================================================

#[arcane]
pub(super) fn swap_br_strided_arm_v2(t: Arm64V2Token, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        swap_br_row_arm_v2(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_swap_br_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_br_row_arm_v2(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_strided_arm_v2(t: Arm64V2Token, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        fill_alpha_row_arm_v2(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_bgra_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_arm_v2(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_rgba_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_arm_v2(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_arm_v2(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_arm_v2(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn swap_bgr_strided_arm_v2(t: Arm64V2Token, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        swap_bgr_row_arm_v2(t, &mut buf[y * stride..][..w * 3]);
    }
}
#[arcane]
pub(super) fn copy_swap_bgr_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_bgr_row_arm_v2(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn rgba_to_rgb_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgba_to_rgb_row_arm_v2(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn bgra_to_rgb_strided_arm_v2(
    t: Arm64V2Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        bgra_to_rgb_row_arm_v2(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
