use archmage::prelude::*;

use super::swap_br_u32;

// ===========================================================================
// ARM NEON — rite row implementations
//
// Cross-bpp operations (3↔4 channel, 3bpp copy+swap) are intentionally
// omitted: LLVM's autovectorizer generates faster code than explicit
// vld3q/vld4q/vst3q/vst4q structure loads on all tested aarch64 platforms
// (Ampere, Apple Silicon, Snapdragon). Those ops dispatch directly to scalar.
//
// RE-VERIFIED 2026-07-28 on Apple M4 Pro (macOS, rustc release, no
// target-cpu=native). Explicit vld4q_u8→vst3q_u8 / vld3q_u8→vst4q_u8 kernels
// were implemented for all nine cross-bpp ops and measured against the
// autovectorized scalar path via `bench_crossbpp_sweep` (SIMD-vs-scalar A/B at
// 64x64 / 256x256 / 1024x1024 / 4096x4096). The structure-load kernels lost at
// every size:
//
//   rgba_to_rgb (4→3):  +2.9% / +1.5% / +1.9% / +1.7% slower
//   rgb_to_rgba (3→4): +27%  / +32%  / +17%  / +7%   slower
//   rgb_to_bgr  (3bpp):  -9% /  tie  / +17%  /  tie
//
// Apple's cores crack vld4q/vst3q into several µops with limited structure
// load/store throughput, while the autovectorized form uses plain vld1q + tbl
// shuffles that dual-issue better. At 1920x1080 these ops are additionally
// single-core memory-bandwidth-bound (~67 GB/s), so all variants converge.
// Do not re-add explicit structure-load kernels here without re-running that
// sweep on the target hardware first.
//
// This does NOT apply to elementwise ops (depth conversion, premul/unpremul),
// which use vmovl/vcvtq/vmulq rather than structure loads — see below.
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
    for px in row[i..].chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        let v = u32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
        d.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
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
    for px in row[i..].chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&(v | 0xFF00_0000).to_ne_bytes());
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
    for (&v, d) in src[is..].iter().zip(dst[id..].chunks_exact_mut(4)) {
        let g = v as u32;
        d.copy_from_slice(&(g | (g << 8) | (g << 16) | 0xFF00_0000).to_ne_bytes());
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
    for (ga, d) in src[is..].chunks_exact(2).zip(dst[id..].chunks_exact_mut(4)) {
        let g = ga[0] as u32;
        d.copy_from_slice(&(g | (g << 8) | (g << 16) | ((ga[1] as u32) << 24)).to_ne_bytes());
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
// ARM NEON — ARGB/XRGB rite row implementations
// ===========================================================================

#[rite]
pub(super) fn rotate_left_row_neon(_token: NeonToken, row: &mut [u8]) {
    let mask_bytes: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
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
    for px in row[i..].chunks_exact_mut(4) {
        let a = px[0];
        px[0] = px[1];
        px[1] = px[2];
        px[2] = px[3];
        px[3] = a;
    }
}

#[rite]
pub(super) fn copy_rotate_left_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
    let mask_bytes: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
        d[3] = s[0];
    }
}

#[rite]
pub(super) fn rotate_right_row_neon(_token: NeonToken, row: &mut [u8]) {
    let mask_bytes: [u8; 16] = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14];
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
    for px in row[i..].chunks_exact_mut(4) {
        let d = px[3];
        px[3] = px[2];
        px[2] = px[1];
        px[1] = px[0];
        px[0] = d;
    }
}

#[rite]
pub(super) fn copy_rotate_right_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
    let mask_bytes: [u8; 16] = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14];
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[3];
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
}

#[rite]
pub(super) fn reverse_4bpp_row_neon(_token: NeonToken, row: &mut [u8]) {
    let mask_bytes: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];
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
    for px in row[i..].chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&v.swap_bytes().to_ne_bytes());
    }
}

#[rite]
pub(super) fn copy_reverse_4bpp_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
    let mask_bytes: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];
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
    for (s, d) in src[i..].chunks_exact(4).zip(dst[i..].chunks_exact_mut(4)) {
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
}

#[rite]
pub(super) fn fill_alpha_first_row_neon(_token: NeonToken, row: &mut [u8]) {
    let ab: [u8; 16] = [0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0];
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
    for px in row[i..].chunks_exact_mut(4) {
        px[0] = 0xFF;
    }
}

#[rite]
pub(super) fn gray_to_4bpp_alpha_first_row_neon(_token: NeonToken, src: &[u8], dst: &mut [u8]) {
    let masks: [[u8; 16]; 4] = [
        [0x80, 0, 0, 0, 0x80, 1, 1, 1, 0x80, 2, 2, 2, 0x80, 3, 3, 3],
        [0x80, 4, 4, 4, 0x80, 5, 5, 5, 0x80, 6, 6, 6, 0x80, 7, 7, 7],
        [
            0x80, 8, 8, 8, 0x80, 9, 9, 9, 0x80, 10, 10, 10, 0x80, 11, 11, 11,
        ],
        [
            0x80, 12, 12, 12, 0x80, 13, 13, 13, 0x80, 14, 14, 14, 0x80, 15, 15, 15,
        ],
    ];
    let m: [_; 4] = core::array::from_fn(|i| vld1q_u8(&masks[i]));
    let ab: [u8; 16] = [0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF, 0, 0, 0];
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
    for (&v, d) in src[is..].iter().zip(dst[id..].chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = v;
        d[2] = v;
        d[3] = v;
    }
}

#[rite]
pub(super) fn gray_alpha_to_4bpp_alpha_first_row_neon(
    _token: NeonToken,
    src: &[u8],
    dst: &mut [u8],
) {
    let masks: [[u8; 16]; 2] = [
        [1, 0, 0, 0, 3, 2, 2, 2, 5, 4, 4, 4, 7, 6, 6, 6],
        [9, 8, 8, 8, 11, 10, 10, 10, 13, 12, 12, 12, 15, 14, 14, 14],
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
    for (ga, d) in src[is..].chunks_exact(2).zip(dst[id..].chunks_exact_mut(4)) {
        d[0] = ga[1];
        d[1] = ga[0];
        d[2] = ga[0];
        d[3] = ga[0];
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
#[arcane]
pub(super) fn rotate_left_impl_neon(t: NeonToken, b: &mut [u8]) {
    rotate_left_row_neon(t, b);
}
#[arcane]
pub(super) fn copy_rotate_left_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    copy_rotate_left_row_neon(t, s, d);
}
#[arcane]
pub(super) fn rotate_right_impl_neon(t: NeonToken, b: &mut [u8]) {
    rotate_right_row_neon(t, b);
}
#[arcane]
pub(super) fn copy_rotate_right_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    copy_rotate_right_row_neon(t, s, d);
}
#[arcane]
pub(super) fn reverse_4bpp_impl_neon(t: NeonToken, b: &mut [u8]) {
    reverse_4bpp_row_neon(t, b);
}
#[arcane]
pub(super) fn copy_reverse_4bpp_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    copy_reverse_4bpp_row_neon(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_first_impl_neon(t: NeonToken, b: &mut [u8]) {
    fill_alpha_first_row_neon(t, b);
}
#[arcane]
pub(super) fn gray_to_4bpp_alpha_first_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_alpha_first_row_neon(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_alpha_first_impl_neon(t: NeonToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_alpha_first_row_neon(t, s, d);
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
#[arcane]
pub(super) fn rotate_left_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_left_row_neon(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rotate_left_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_left_row_neon(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rotate_right_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_right_row_neon(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rotate_right_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_right_row_neon(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn reverse_4bpp_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        reverse_4bpp_row_neon(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_reverse_4bpp_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_reverse_4bpp_row_neon(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_first_strided_neon(
    t: NeonToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_first_row_neon(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_alpha_first_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_alpha_first_row_neon(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_alpha_first_strided_neon(
    t: NeonToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_alpha_first_row_neon(
            t,
            &src[y * ss..][..w * 2],
            &mut dst[y * ds..][..w * 4],
        );
    }
}

// ===========================================================================
// Experimental: depth conversion + premultiply (feature = "experimental")
//
// These are elementwise (vmovl / vcvtq / vmulq / vdivq), not structure loads,
// so the cross-bpp finding documented at the top of this file does not apply.
// Every kernel below is bit-exact with its scalar counterpart: true IEEE
// division (vdivq_f32) rather than a reciprocal approximation, and
// truncating vcvtq_u32_f32 to match Rust's saturating `as` cast.
// ===========================================================================

#[cfg(feature = "experimental")]
mod experimental {
    use archmage::prelude::*;

    /// Reinterpret 16 bytes as 4 × f32 lanes.
    #[rite]
    fn load_f32x4(_t: NeonToken, chunk: &[u8; 16]) -> float32x4_t {
        vreinterpretq_f32_u8(vld1q_u8(chunk))
    }

    /// Store 4 × f32 lanes as 16 bytes.
    #[rite]
    fn store_f32x4(_t: NeonToken, chunk: &mut [u8; 16], v: float32x4_t) {
        vst1q_u8(chunk, vreinterpretq_u8_f32(v));
    }

    // -----------------------------------------------------------------------
    // Depth conversions
    // -----------------------------------------------------------------------

    /// u8 → u16 via `v * 257`. 16 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_u8_to_u16_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let n = src.len().min(dst.len() / 2);
        let mut i = 0;
        while i + 16 <= n {
            let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
            let v = vld1q_u8(s);
            let lo = vmulq_n_u16(vmovl_u8(vget_low_u8(v)), 257);
            let hi = vmulq_n_u16(vmovl_high_u8(v), 257);
            let d0: &mut [u8; 16] = (&mut dst[i * 2..i * 2 + 16]).try_into().unwrap();
            vst1q_u8(d0, vreinterpretq_u8_u16(lo));
            let d1: &mut [u8; 16] = (&mut dst[i * 2 + 16..i * 2 + 32]).try_into().unwrap();
            vst1q_u8(d1, vreinterpretq_u8_u16(hi));
            i += 16;
        }
        for j in i..n {
            dst[j * 2..j * 2 + 2].copy_from_slice(&((src[j] as u16) * 257).to_ne_bytes());
        }
    }

    /// u16 → u8 via `(v * 255 + 32768) >> 16`. 8 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_u16_to_u8_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let n = (src.len() / 2).min(dst.len());
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 16] = src[i * 2..i * 2 + 16].try_into().unwrap();
            let v = vreinterpretq_u16_u8(vld1q_u8(s));
            let lo = vshrq_n_u32::<16>(vaddq_u32(
                vmulq_n_u32(vmovl_u16(vget_low_u16(v)), 255),
                vdupq_n_u32(32768),
            ));
            let hi = vshrq_n_u32::<16>(vaddq_u32(
                vmulq_n_u32(vmovl_high_u16(v), 255),
                vdupq_n_u32(32768),
            ));
            let packed16 = vcombine_u16(vmovn_u32(lo), vmovn_u32(hi));
            let packed8 = vmovn_u16(packed16);
            let out: &mut [u8; 8] = (&mut dst[i..i + 8]).try_into().unwrap();
            vst1_u8(out, packed8);
            i += 8;
        }
        for j in i..n {
            let v = u16::from_ne_bytes([src[j * 2], src[j * 2 + 1]]);
            dst[j] = ((v as u32 * 255 + 32768) >> 16) as u8;
        }
    }

    /// u8 → f32 via `v / 255.0`. 16 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_u8_to_f32_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        // Reciprocal-multiply, matching the AVX2 tier (`_mm256_set1_ps(1.0/255.0)`).
        // vdivq_f32 would be bit-exact with the scalar path but measured ~53%
        // SLOWER than scalar here: FDIV throughput on Apple cores is low enough
        // that the kernel stops being memory-bound. See CHANGELOG for the
        // pre-existing scalar-vs-SIMD f32 discrepancy this inherits.
        let scale = vdupq_n_f32(1.0 / 255.0);
        let n = src.len().min(dst.len() / 4);
        let mut i = 0;
        while i + 16 <= n {
            let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
            let v = vld1q_u8(s);
            let w16lo = vmovl_u8(vget_low_u8(v));
            let w16hi = vmovl_high_u8(v);
            // Straight-line: binding these in an array forces a stack spill of
            // every vector, which measured 52% slower than the scalar path.
            let f0 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16lo))), scale);
            let f1 = vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(w16lo)), scale);
            let f2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16hi))), scale);
            let f3 = vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(w16hi)), scale);
            let base = i * 4;
            let d0: &mut [u8; 16] = (&mut dst[base..base + 16]).try_into().unwrap();
            store_f32x4(_t, d0, f0);
            let d1: &mut [u8; 16] = (&mut dst[base + 16..base + 32]).try_into().unwrap();
            store_f32x4(_t, d1, f1);
            let d2: &mut [u8; 16] = (&mut dst[base + 32..base + 48]).try_into().unwrap();
            store_f32x4(_t, d2, f2);
            let d3: &mut [u8; 16] = (&mut dst[base + 48..base + 64]).try_into().unwrap();
            store_f32x4(_t, d3, f3);
            i += 16;
        }
        for j in i..n {
            dst[j * 4..j * 4 + 4].copy_from_slice(&(src[j] as f32 / 255.0).to_ne_bytes());
        }
    }

    /// u16 → f32 via `v / 65535.0`. 8 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_u16_to_f32_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        // Reciprocal-multiply, matching the AVX2 tier. See convert_u8_to_f32.
        let scale = vdupq_n_f32(1.0 / 65535.0);
        let n = (src.len() / 2).min(dst.len() / 4);
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 16] = src[i * 2..i * 2 + 16].try_into().unwrap();
            let v = vreinterpretq_u16_u8(vld1q_u8(s));
            let f0 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v))), scale);
            let f1 = vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(v)), scale);
            let base = i * 4;
            let d0: &mut [u8; 16] = (&mut dst[base..base + 16]).try_into().unwrap();
            store_f32x4(_t, d0, f0);
            let d1: &mut [u8; 16] = (&mut dst[base + 16..base + 32]).try_into().unwrap();
            store_f32x4(_t, d1, f1);
            i += 8;
        }
        for j in i..n {
            let v = u16::from_ne_bytes([src[j * 2], src[j * 2 + 1]]);
            dst[j * 4..j * 4 + 4].copy_from_slice(&(v as f32 / 65535.0).to_ne_bytes());
        }
    }

    /// Clamp to [0,1], scale, add 0.5, truncate — shared by the f32→int kernels.
    #[rite]
    fn quantize(_t: NeonToken, v: float32x4_t, scale: float32x4_t) -> uint32x4_t {
        let clamped = vminq_f32(vmaxq_f32(v, vdupq_n_f32(0.0)), vdupq_n_f32(1.0));
        vcvtq_u32_f32(vaddq_f32(vmulq_f32(clamped, scale), vdupq_n_f32(0.5)))
    }

    /// f32 → u8 via `clamp(v,0,1) * 255 + 0.5`, truncated. 16 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_f32_to_u8_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let scale = vdupq_n_f32(255.0);
        let n = (src.len() / 4).min(dst.len());
        let mut i = 0;
        while i + 16 <= n {
            let mut q = [vdupq_n_u32(0); 4];
            for (k, qk) in q.iter_mut().enumerate() {
                let s: &[u8; 16] = src[(i + k * 4) * 4..(i + k * 4) * 4 + 16]
                    .try_into()
                    .unwrap();
                *qk = quantize(_t, load_f32x4(_t, s), scale);
            }
            let lo = vcombine_u16(vmovn_u32(q[0]), vmovn_u32(q[1]));
            let hi = vcombine_u16(vmovn_u32(q[2]), vmovn_u32(q[3]));
            let packed = vcombine_u8(vmovn_u16(lo), vmovn_u16(hi));
            let out: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
            vst1q_u8(out, packed);
            i += 16;
        }
        for j in i..n {
            let s: &[u8; 4] = src[j * 4..j * 4 + 4].try_into().unwrap();
            let v = f32::from_ne_bytes(*s);
            dst[j] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }

    /// f32 → u16 via `clamp(v,0,1) * 65535 + 0.5`, truncated. 8 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_f32_to_u16_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let scale = vdupq_n_f32(65535.0);
        let n = (src.len() / 4).min(dst.len() / 2);
        let mut i = 0;
        while i + 8 <= n {
            let mut q = [vdupq_n_u32(0); 2];
            for (k, qk) in q.iter_mut().enumerate() {
                let s: &[u8; 16] = src[(i + k * 4) * 4..(i + k * 4) * 4 + 16]
                    .try_into()
                    .unwrap();
                *qk = quantize(_t, load_f32x4(_t, s), scale);
            }
            let packed = vcombine_u16(vmovn_u32(q[0]), vmovn_u32(q[1]));
            let out: &mut [u8; 16] = (&mut dst[i * 2..i * 2 + 16]).try_into().unwrap();
            vst1q_u8(out, vreinterpretq_u8_u16(packed));
            i += 8;
        }
        for j in i..n {
            let s: &[u8; 4] = src[j * 4..j * 4 + 4].try_into().unwrap();
            let v = f32::from_ne_bytes(*s);
            dst[j * 2..j * 2 + 2]
                .copy_from_slice(&((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_ne_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Premultiply / unpremultiply (f32 RGBA, 16 bytes per pixel)
    // -----------------------------------------------------------------------

    /// Lane mask selecting only the alpha lane (lane 3).
    #[rite]
    fn alpha_lane_mask(_t: NeonToken) -> uint32x4_t {
        let m: [u32; 4] = [0, 0, 0, u32::MAX];
        vld1q_u32(&m)
    }

    /// `C' = C * A`, alpha preserved.
    #[rite]
    fn premul_px(_t: NeonToken, v: float32x4_t, keep_a: uint32x4_t) -> float32x4_t {
        vbslq_f32(keep_a, v, vmulq_laneq_f32::<3>(v, v))
    }

    /// `C' = C / A`, alpha preserved; all-zero RGB where A == 0.
    ///
    /// Branchless: the scalar path's `if a == 0.0` becomes a vceqzq_f32 mask
    /// plus a select, which is what makes this worth vectorizing at all.
    #[rite]
    fn unpremul_px(_t: NeonToken, v: float32x4_t, keep_a: uint32x4_t) -> float32x4_t {
        let a = vdupq_laneq_f32::<3>(v);
        let scaled = vmulq_f32(v, vdivq_f32(vdupq_n_f32(1.0), a));
        let unpremulled = vbslq_f32(vceqzq_f32(a), vdupq_n_f32(0.0), scaled);
        vbslq_f32(keep_a, v, unpremulled)
    }

    #[rite]
    pub(in crate::bytes) fn premul_f32_row_neon(_t: NeonToken, buf: &mut [u8]) {
        let keep_a = alpha_lane_mask(_t);
        for px in buf.chunks_exact_mut(16) {
            let px: &mut [u8; 16] = px.try_into().unwrap();
            store_f32x4(_t, px, premul_px(_t, load_f32x4(_t, px), keep_a));
        }
    }

    #[rite]
    pub(in crate::bytes) fn premul_f32_copy_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let keep_a = alpha_lane_mask(_t);
        for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
            let s: &[u8; 16] = s.try_into().unwrap();
            let d: &mut [u8; 16] = d.try_into().unwrap();
            store_f32x4(_t, d, premul_px(_t, load_f32x4(_t, s), keep_a));
        }
    }

    /// Transpose 4 interleaved RGBA pixels into (r, g, b, a) planes.
    ///
    /// Avoids vld4q_f32 (which needs an f32-aligned `&[f32; 16]`; our buffers
    /// are `&[u8]`) and lets one vdivq_f32 serve four pixels instead of one.
    #[rite]
    fn transpose4(
        _t: NeonToken,
        p0: float32x4_t,
        p1: float32x4_t,
        p2: float32x4_t,
        p3: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let t0 = vtrn1q_f32(p0, p1);
        let t1 = vtrn2q_f32(p0, p1);
        let t2 = vtrn1q_f32(p2, p3);
        let t3 = vtrn2q_f32(p2, p3);
        (
            vcombine_f32(vget_low_f32(t0), vget_low_f32(t2)),
            vcombine_f32(vget_low_f32(t1), vget_low_f32(t3)),
            vcombine_f32(vget_high_f32(t0), vget_high_f32(t2)),
            vcombine_f32(vget_high_f32(t1), vget_high_f32(t3)),
        )
    }

    /// Unpremultiply 4 pixels held as planes. One divide serves all four.
    #[rite]
    fn unpremul_planes(
        _t: NeonToken,
        r: float32x4_t,
        g: float32x4_t,
        b: float32x4_t,
        a: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        // Bit-exact with the scalar path: reciprocal first, then multiply.
        let inv = vdivq_f32(vdupq_n_f32(1.0), a);
        let zero_a = vceqzq_f32(a);
        let zero = vdupq_n_f32(0.0);
        (
            vbslq_f32(zero_a, zero, vmulq_f32(r, inv)),
            vbslq_f32(zero_a, zero, vmulq_f32(g, inv)),
            vbslq_f32(zero_a, zero, vmulq_f32(b, inv)),
        )
    }

    #[rite]
    pub(in crate::bytes) fn unpremul_f32_row_neon(_t: NeonToken, buf: &mut [u8]) {
        let keep_a = alpha_lane_mask(_t);
        let mut i = 0;
        let n = buf.len();
        // 4 pixels (64 bytes) per iteration: one vdivq_f32 instead of four.
        while i + 64 <= n {
            let p0 = load_f32x4(_t, buf[i..i + 16].try_into().unwrap());
            let p1 = load_f32x4(_t, buf[i + 16..i + 32].try_into().unwrap());
            let p2 = load_f32x4(_t, buf[i + 32..i + 48].try_into().unwrap());
            let p3 = load_f32x4(_t, buf[i + 48..i + 64].try_into().unwrap());
            let (r, g, b, a) = transpose4(_t, p0, p1, p2, p3);
            let (r, g, b) = unpremul_planes(_t, r, g, b, a);
            let (q0, q1, q2, q3) = transpose4(_t, r, g, b, a);
            store_f32x4(_t, (&mut buf[i..i + 16]).try_into().unwrap(), q0);
            store_f32x4(_t, (&mut buf[i + 16..i + 32]).try_into().unwrap(), q1);
            store_f32x4(_t, (&mut buf[i + 32..i + 48]).try_into().unwrap(), q2);
            store_f32x4(_t, (&mut buf[i + 48..i + 64]).try_into().unwrap(), q3);
            i += 64;
        }
        for px in buf[i..].chunks_exact_mut(16) {
            let px: &mut [u8; 16] = px.try_into().unwrap();
            store_f32x4(_t, px, unpremul_px(_t, load_f32x4(_t, px), keep_a));
        }
    }

    #[rite]
    pub(in crate::bytes) fn unpremul_f32_copy_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let keep_a = alpha_lane_mask(_t);
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + 64 <= n {
            let p0 = load_f32x4(_t, src[i..i + 16].try_into().unwrap());
            let p1 = load_f32x4(_t, src[i + 16..i + 32].try_into().unwrap());
            let p2 = load_f32x4(_t, src[i + 32..i + 48].try_into().unwrap());
            let p3 = load_f32x4(_t, src[i + 48..i + 64].try_into().unwrap());
            let (r, g, b, a) = transpose4(_t, p0, p1, p2, p3);
            let (r, g, b) = unpremul_planes(_t, r, g, b, a);
            let (q0, q1, q2, q3) = transpose4(_t, r, g, b, a);
            store_f32x4(_t, (&mut dst[i..i + 16]).try_into().unwrap(), q0);
            store_f32x4(_t, (&mut dst[i + 16..i + 32]).try_into().unwrap(), q1);
            store_f32x4(_t, (&mut dst[i + 32..i + 48]).try_into().unwrap(), q2);
            store_f32x4(_t, (&mut dst[i + 48..i + 64]).try_into().unwrap(), q3);
            i += 64;
        }
        for (s, d) in src[i..].chunks_exact(16).zip(dst[i..].chunks_exact_mut(16)) {
            let s: &[u8; 16] = s.try_into().unwrap();
            let d: &mut [u8; 16] = d.try_into().unwrap();
            store_f32x4(_t, d, unpremul_px(_t, load_f32x4(_t, s), keep_a));
        }
    }

    // -----------------------------------------------------------------------
    // Contiguous arcane wrappers
    // -----------------------------------------------------------------------

    macro_rules! neon_copy_impl {
        ($name:ident, $row:ident) => {
            #[arcane]
            pub(in crate::bytes) fn $name(t: NeonToken, s: &[u8], d: &mut [u8]) {
                $row(t, s, d);
            }
        };
    }
    macro_rules! neon_inplace_impl {
        ($name:ident, $row:ident) => {
            #[arcane]
            pub(in crate::bytes) fn $name(t: NeonToken, b: &mut [u8]) {
                $row(t, b);
            }
        };
    }

    neon_copy_impl!(convert_u8_to_u16_impl_neon, convert_u8_to_u16_row_neon);
    neon_copy_impl!(convert_u16_to_u8_impl_neon, convert_u16_to_u8_row_neon);
    neon_copy_impl!(convert_u8_to_f32_impl_neon, convert_u8_to_f32_row_neon);
    neon_copy_impl!(convert_f32_to_u8_impl_neon, convert_f32_to_u8_row_neon);
    neon_copy_impl!(convert_u16_to_f32_impl_neon, convert_u16_to_f32_row_neon);
    neon_copy_impl!(convert_f32_to_u16_impl_neon, convert_f32_to_u16_row_neon);
    neon_copy_impl!(premul_f32_copy_impl_neon, premul_f32_copy_row_neon);
    neon_copy_impl!(unpremul_f32_copy_impl_neon, unpremul_f32_copy_row_neon);
    neon_inplace_impl!(premul_f32_impl_neon, premul_f32_row_neon);
    neon_inplace_impl!(unpremul_f32_impl_neon, unpremul_f32_row_neon);

    // -----------------------------------------------------------------------
    // Strided arcane wrappers
    // -----------------------------------------------------------------------

    /// Strided copy wrapper. `$sb`/`$db` are source/dest bytes per element.
    macro_rules! neon_copy_strided {
        ($name:ident, $row:ident, $sb:expr, $db:expr) => {
            #[arcane]
            pub(in crate::bytes) fn $name(
                t: NeonToken,
                src: &[u8],
                dst: &mut [u8],
                w: usize,
                h: usize,
                ss: usize,
                ds: usize,
            ) {
                for y in 0..h {
                    $row(t, &src[y * ss..][..w * $sb], &mut dst[y * ds..][..w * $db]);
                }
            }
        };
    }

    neon_copy_strided!(
        convert_u8_to_u16_strided_neon,
        convert_u8_to_u16_row_neon,
        1,
        2
    );
    neon_copy_strided!(
        convert_u16_to_u8_strided_neon,
        convert_u16_to_u8_row_neon,
        2,
        1
    );
    neon_copy_strided!(
        convert_u8_to_f32_strided_neon,
        convert_u8_to_f32_row_neon,
        1,
        4
    );
    neon_copy_strided!(
        convert_f32_to_u8_strided_neon,
        convert_f32_to_u8_row_neon,
        4,
        1
    );
    neon_copy_strided!(
        convert_u16_to_f32_strided_neon,
        convert_u16_to_f32_row_neon,
        2,
        4
    );
    neon_copy_strided!(
        convert_f32_to_u16_strided_neon,
        convert_f32_to_u16_row_neon,
        4,
        2
    );
    neon_copy_strided!(
        premul_f32_copy_strided_neon,
        premul_f32_copy_row_neon,
        16,
        16
    );
    neon_copy_strided!(
        unpremul_f32_copy_strided_neon,
        unpremul_f32_copy_row_neon,
        16,
        16
    );

    #[arcane]
    pub(in crate::bytes) fn premul_f32_strided_neon(
        t: NeonToken,
        buf: &mut [u8],
        w: usize,
        h: usize,
        stride: usize,
    ) {
        for y in 0..h {
            premul_f32_row_neon(t, &mut buf[y * stride..][..w * 16]);
        }
    }

    #[arcane]
    pub(in crate::bytes) fn unpremul_f32_strided_neon(
        t: NeonToken,
        buf: &mut [u8],
        w: usize,
        h: usize,
        stride: usize,
    ) {
        for y in 0..h {
            unpremul_f32_row_neon(t, &mut buf[y * stride..][..w * 16]);
        }
    }
}

#[cfg(feature = "experimental")]
pub(super) use experimental::*;
