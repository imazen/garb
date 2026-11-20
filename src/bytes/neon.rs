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
// RE-VERIFIED 2026-07-29 on Apple Silicon, and the REASON is memory bandwidth,
// not instruction throughput — which is worth stating, because the in-place
// kernels below use exactly these structure ops and win big, so the rule looks
// self-contradictory without it. Measured (benches/kernel_tiers.rs, 1 MP,
// neon vs forced scalar):
//
//   rgb_to_bgr_inplace   47.2 -> 302.1 us   6.40x   (in-place, half the traffic)
//   rgba_to_bgra         70.6 -> 267.1 us   3.78x   (u32 swizzle, no structure ops)
//   rgb_to_bgr  (copy)   52.9 -> 52.9 us    1.00x
//   rgb_to_rgba          60.9 -> 60.9 us    1.00x
//   rgba_to_rgb         115.4 -> 115.1 us   1.00x
//
// A real vld3q/vst3q arm was written for the copy variant `copy_swap_bgr` and
// measured 0.99x — live and correct (poisoning it fails
// `permutation_copy_swap_bgr`), just not faster. At ~55 GB/s combined read+write
// the copy kernels are bandwidth-bound, so no instruction-level change can help;
// the in-place twin wins because it moves half the bytes. The arm was reverted
// rather than shipped for zero gain. Do not re-attempt without first showing the
// op is not bandwidth-bound.
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
// EXTENDED 2026-08-01 to the gray family. The six gray ops (gray_to_rgb,
// gray_alpha_to_rgb, gray_to_gray_alpha, gray_alpha_to_gray,
// rgb_to_gray_identity, rgba_to_gray_identity) dispatch [scalar] only and were
// in no benchmark, so they looked like the same "no arm exists" gap that turned
// out to be real in zenblend. They are not. `rgb_to_gray_identity` is the
// strongest candidate in the family — a PURE deinterleave (take every 3rd byte)
// where vld3q_u8 does the whole job in hardware and 2/3 of the loaded data is
// discarded, so the write side is a third of the read side. That is a genuinely
// different shape from the 3->4 / 4->3 / 3bpp ops measured above, which is why
// it was worth testing rather than assuming.
//
// Measured anyway: neon 89.0us vs scalar 87.9us on 1 MP = 0.99x. No gain. The
// rule above therefore covers reductions and pure-deinterleave shapes too, not
// only the same-or-growing-width copies it was originally derived from. The
// kernel was reverted; the bench entry stays (benches/kernel_tiers.rs) so the
// family is no longer unmeasured.
//
// Note these six are behind `feature = "experimental"` and are not in a default
// build, which is a second reason not to carry a kernel for them.
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
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
    for (s, d) in src[i..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[i..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
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
    for (&v, d) in src[is..]
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for (ga, d) in src[is..]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for px in row[i..].as_chunks_mut::<3>().0.iter_mut() {
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
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
    for (s, d) in src[i..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[i..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
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
    for (s, d) in src[i..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[i..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
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
    for (s, d) in src[i..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[i..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
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
    for (&v, d) in src[is..]
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
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
    for (ga, d) in src[is..]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
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
// Integer depth conversions (u8 <-> u16)
// ===========================================================================

mod integer_depth {
    use archmage::prelude::*;

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

    /// u16 → u8 rounded to nearest, equivalent to `(v + 128) / 257`. 8 elements/iter.
    #[rite]
    pub(in crate::bytes) fn convert_u16_to_u8_row_neon(_t: NeonToken, src: &[u8], dst: &mut [u8]) {
        let n = (src.len() / 2).min(dst.len());
        let mut i = 0;
        while i + 8 <= n {
            let s: &[u8; 16] = src[i * 2..i * 2 + 16].try_into().unwrap();
            let v = vreinterpretq_u16_u8(vld1q_u8(s));
            let lo = vshrq_n_u32::<16>(vmulq_n_u32(
                vaddq_u32(vmovl_u16(vget_low_u16(v)), vdupq_n_u32(129)),
                255,
            ));
            let hi = vshrq_n_u32::<16>(vmulq_n_u32(
                vaddq_u32(vmovl_high_u16(v), vdupq_n_u32(129)),
                255,
            ));
            let packed16 = vcombine_u16(vmovn_u32(lo), vmovn_u32(hi));
            let packed8 = vmovn_u16(packed16);
            let out: &mut [u8; 8] = (&mut dst[i..i + 8]).try_into().unwrap();
            vst1_u8(out, packed8);
            i += 8;
        }
        for j in i..n {
            let v = u16::from_ne_bytes([src[j * 2], src[j * 2 + 1]]);
            dst[j] = ((v as u32 + 128) / 257) as u8;
        }
    }

    macro_rules! neon_copy_impl {
        ($name:ident, $row:ident) => {
            #[arcane]
            pub(in crate::bytes) fn $name(t: NeonToken, s: &[u8], d: &mut [u8]) {
                $row(t, s, d);
            }
        };
    }

    neon_copy_impl!(convert_u8_to_u16_impl_neon, convert_u8_to_u16_row_neon);
    neon_copy_impl!(convert_u16_to_u8_impl_neon, convert_u16_to_u8_row_neon);

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
}

pub(super) use integer_depth::*;
