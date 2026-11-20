use archmage::prelude::*;

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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
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
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&(v | 0xFF00_0000).to_ne_bytes());
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
    for (s, d) in src[is..]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
        d.copy_from_slice(
            &(s[2] as u32 | ((s[1] as u32) << 8) | ((s[0] as u32) << 16) | 0xFF00_0000)
                .to_ne_bytes(),
        );
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
    for (s, d) in src[is..]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
        d.copy_from_slice(
            &(s[0] as u32 | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | 0xFF00_0000)
                .to_ne_bytes(),
        );
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
    for (&v, d) in src[is..]
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
        let g = v as u32;
        d.copy_from_slice(&(g | (g << 8) | (g << 16) | 0xFF00_0000).to_ne_bytes());
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
    for px in row[i..].as_chunks_mut::<3>().0.iter_mut() {
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
    for (s, d) in src[i..]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[i..].as_chunks_mut::<3>().0.iter_mut())
    {
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
    for (s, d) in src[is..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<3>().0.iter_mut())
    {
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
    for (s, d) in src[is..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<3>().0.iter_mut())
    {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

// ===========================================================================
// WASM SIMD128 — ARGB/XRGB rite row implementations
// ===========================================================================

#[rite]
pub(super) fn rotate_left_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let mask = i8x16(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        v128_store(out, i8x16_swizzle(v, mask));
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
pub(super) fn copy_rotate_left_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = v128_load(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        v128_store(d, i8x16_swizzle(v, mask));
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
pub(super) fn rotate_right_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let mask = i8x16(3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        v128_store(out, i8x16_swizzle(v, mask));
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
pub(super) fn copy_rotate_right_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = v128_load(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        v128_store(d, i8x16_swizzle(v, mask));
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
pub(super) fn reverse_4bpp_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let mask = i8x16(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        v128_store(out, i8x16_swizzle(v, mask));
        i += 16;
    }
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&v.swap_bytes().to_ne_bytes());
    }
}

#[rite]
pub(super) fn copy_reverse_4bpp_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        let s: &[u8; 16] = src[i..i + 16].try_into().unwrap();
        let v = v128_load(s);
        let d: &mut [u8; 16] = (&mut dst[i..i + 16]).try_into().unwrap();
        v128_store(d, i8x16_swizzle(v, mask));
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
pub(super) fn fill_alpha_first_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    let alpha = u32x4_splat(0x000000FF);
    let n = row.len();
    let mut i = 0;
    while i + 16 <= n {
        let arr: &[u8; 16] = row[i..i + 16].try_into().unwrap();
        let v = v128_load(arr);
        let out: &mut [u8; 16] = (&mut row[i..i + 16]).try_into().unwrap();
        v128_store(out, v128_or(v, alpha));
        i += 16;
    }
    for px in row[i..].as_chunks_mut::<4>().0.iter_mut() {
        px[0] = 0xFF;
    }
}

#[rite]
pub(super) fn rgb_to_argb_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let shuf = i8x16(-128, 0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11);
    let alpha = u32x4_splat(0x000000FF);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        v128_store(d, v128_or(i8x16_swizzle(v128_load(s), shuf), alpha));
        is += 12;
        id += 16;
    }
    for (s, d) in src[is..]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
        d[0] = 0xFF;
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
}

#[rite]
pub(super) fn rgb_to_abgr_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let shuf = i8x16(-128, 2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9);
    let alpha = u32x4_splat(0x000000FF);
    let (slen, dlen) = (src.len(), dst.len());
    let (mut is, mut id) = (0, 0);
    while is + 16 <= slen && id + 16 <= dlen {
        let s: &[u8; 16] = src[is..is + 16].try_into().unwrap();
        let d: &mut [u8; 16] = (&mut dst[id..id + 16]).try_into().unwrap();
        v128_store(d, v128_or(i8x16_swizzle(v128_load(s), shuf), alpha));
        is += 12;
        id += 16;
    }
    for (s, d) in src[is..]
        .as_chunks::<3>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<4>().0.iter_mut())
    {
        d[0] = 0xFF;
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
}

#[rite]
pub(super) fn argb_to_rgb_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -1, -1, -1, -1);
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
    for (s, d) in src[is..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<3>().0.iter_mut())
    {
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
    }
}

#[rite]
pub(super) fn argb_to_bgr_row_wasm128(_token: Wasm128Token, src: &[u8], dst: &mut [u8]) {
    let mask = i8x16(3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, -1, -1, -1, -1);
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
    for (s, d) in src[is..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[id..].as_chunks_mut::<3>().0.iter_mut())
    {
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
    }
}

#[rite]
pub(super) fn gray_to_4bpp_alpha_first_row_wasm128(
    _token: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
) {
    let m0 = i8x16(-128, 0, 0, 0, -128, 1, 1, 1, -128, 2, 2, 2, -128, 3, 3, 3);
    let m1 = i8x16(-128, 4, 4, 4, -128, 5, 5, 5, -128, 6, 6, 6, -128, 7, 7, 7);
    let m2 = i8x16(
        -128, 8, 8, 8, -128, 9, 9, 9, -128, 10, 10, 10, -128, 11, 11, 11,
    );
    let m3 = i8x16(
        -128, 12, 12, 12, -128, 13, 13, 13, -128, 14, 14, 14, -128, 15, 15, 15,
    );
    let alpha = u32x4_splat(0x000000FF);
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
pub(super) fn gray_alpha_to_4bpp_alpha_first_row_wasm128(
    _token: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
) {
    let m0 = i8x16(1, 0, 0, 0, 3, 2, 2, 2, 5, 4, 4, 4, 7, 6, 6, 6);
    let m1 = i8x16(9, 8, 8, 8, 11, 10, 10, 10, 13, 12, 12, 12, 15, 14, 14, 14);
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
#[arcane]
pub(super) fn rotate_left_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    rotate_left_row_wasm128(t, b);
}
#[arcane]
pub(super) fn copy_rotate_left_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    copy_rotate_left_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn rotate_right_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    rotate_right_row_wasm128(t, b);
}
#[arcane]
pub(super) fn copy_rotate_right_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    copy_rotate_right_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn reverse_4bpp_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    reverse_4bpp_row_wasm128(t, b);
}
#[arcane]
pub(super) fn copy_reverse_4bpp_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    copy_reverse_4bpp_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn fill_alpha_first_impl_wasm128(t: Wasm128Token, b: &mut [u8]) {
    fill_alpha_first_row_wasm128(t, b);
}
#[arcane]
pub(super) fn rgb_to_argb_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgb_to_argb_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn rgb_to_abgr_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    rgb_to_abgr_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn argb_to_rgb_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    argb_to_rgb_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn argb_to_bgr_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    argb_to_bgr_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn gray_to_4bpp_alpha_first_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_alpha_first_row_wasm128(t, s, d);
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_alpha_first_impl_wasm128(t: Wasm128Token, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_alpha_first_row_wasm128(t, s, d);
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
#[arcane]
pub(super) fn rotate_left_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_left_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rotate_left_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_left_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rotate_right_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_right_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_rotate_right_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_right_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn reverse_4bpp_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        reverse_4bpp_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn copy_reverse_4bpp_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_reverse_4bpp_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn fill_alpha_first_strided_wasm128(
    t: Wasm128Token,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_first_row_wasm128(t, &mut buf[y * stride..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_argb_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_argb_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn rgb_to_abgr_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_abgr_row_wasm128(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn argb_to_rgb_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        argb_to_rgb_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn argb_to_bgr_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        argb_to_bgr_row_wasm128(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
#[arcane]
pub(super) fn gray_to_4bpp_alpha_first_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_alpha_first_row_wasm128(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
#[arcane]
pub(super) fn gray_alpha_to_4bpp_alpha_first_strided_wasm128(
    t: Wasm128Token,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_alpha_first_row_wasm128(
            t,
            &src[y * ss..][..w * 2],
            &mut dst[y * ds..][..w * 4],
        );
    }
}

#[cfg(feature = "experimental")]
mod float_pixels {
    use super::super::scalar::*;
    use archmage::prelude::*;
    use magetypes::simd::generic::f32x4 as F32x4;

    #[rite]
    fn u8_to_f32(t: Wasm128Token, src: &[u8], dst: &mut [u8]) {
        let (blocks, tail) = src.as_chunks::<4>();
        let scale = F32x4::splat_t(t, 1.0 / 255.0);
        for (src, dst) in blocks.iter().zip(dst.as_chunks_mut::<16>().0) {
            dst.copy_from_slice((F32x4::from_u8_t(t, src) * scale).as_bytes());
        }
        convert_u8_to_f32_row_scalar(ScalarToken, tail, &mut dst[blocks.len() * 16..]);
    }

    #[rite]
    fn u16_to_f32(t: Wasm128Token, src: &[u8], dst: &mut [u8]) {
        let (blocks, tail) = src.as_chunks::<8>();
        let scale = F32x4::splat_t(t, 1.0 / 65535.0);
        for (src, dst) in blocks.iter().zip(dst.as_chunks_mut::<16>().0) {
            let words = core::array::from_fn(|i| u16::from_ne_bytes([src[i * 2], src[i * 2 + 1]]));
            dst.copy_from_slice((F32x4::from_u16_t(t, &words) * scale).as_bytes());
        }
        convert_u16_to_f32_row_scalar(ScalarToken, tail, &mut dst[blocks.len() * 16..]);
    }

    const IN_RANGE: u32 = magetypes::numerics::IntConvert::new().in_range().bits();

    #[rite]
    fn quantize_input(t: Wasm128Token, bytes: &[u8; 16], max: f32) -> F32x4<Wasm128Token> {
        let v = v128_load(bytes);
        // Pseudo-max retains the first operand on unordered lanes.
        let clamped = f32x4_pmin(f32x4_pmax(f32x4_splat(0.0), v), f32x4_splat(1.0));
        F32x4::from_raw_t(t, f32x4_mul(clamped, f32x4_splat(max)))
    }

    #[rite]
    fn f32_to_u8(t: Wasm128Token, src: &[u8], dst: &mut [u8]) {
        let (blocks, tail) = src.as_chunks::<16>();
        for (src, dst) in blocks.iter().zip(dst.as_chunks_mut::<4>().0) {
            *dst = quantize_input(t, src, 255.0).to_u8_with::<IN_RANGE>();
        }
        convert_f32_to_u8_row_scalar(ScalarToken, tail, &mut dst[blocks.len() * 4..]);
    }

    #[rite]
    fn f32_to_u16(t: Wasm128Token, src: &[u8], dst: &mut [u8]) {
        let (blocks, tail) = src.as_chunks::<16>();
        for (src, dst) in blocks.iter().zip(dst.as_chunks_mut::<8>().0) {
            let words = quantize_input(t, src, 65535.0).to_u16_with::<IN_RANGE>();
            dst.copy_from_slice(bytemuck::bytes_of(&words));
        }
        convert_f32_to_u16_row_scalar(ScalarToken, tail, &mut dst[blocks.len() * 8..]);
    }

    #[rite]
    fn alpha_pixel<const UNPREMULTIPLY: bool>(t: Wasm128Token, pixel: &[u8; 16]) -> [u8; 16] {
        let pixel = F32x4::from_bytes_t(t, pixel);
        let alpha = pixel.broadcast_rgba_alpha();
        let result = if UNPREMULTIPLY {
            let zero = F32x4::zero_t(t);
            let one = F32x4::splat_t(t, 1.0);
            let zero_mask = alpha.simd_eq(zero);
            let scaled = (pixel * (one / alpha)).with_rgba_alpha(pixel);
            F32x4::blend(zero_mask, zero, scaled)
        } else {
            (pixel * alpha).with_rgba_alpha(pixel)
        };
        *result.as_bytes()
    }

    macro_rules! copy_workers {
        ($row:ident $(::<$param:literal>)?, $whole:ident, $strided:ident, $src_bpp:literal, $dst_bpp:literal) => {
            #[arcane]
            pub(in crate::bytes) fn $whole(t: Wasm128Token, src: &[u8], dst: &mut [u8]) {
                $row $(::<$param>)?(t, src, dst);
            }

            #[arcane]
            pub(in crate::bytes) fn $strided(
                t: Wasm128Token, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize,
            ) {
                for y in 0..h {
                    $row $(::<$param>)?(t, &src[y * ss..][..w * $src_bpp], &mut dst[y * ds..][..w * $dst_bpp]);
                }
            }
        };
    }

    #[rite]
    fn unpremultiply_four(t: Wasm128Token, src: &[u8; 64]) -> [u8; 64] {
        let packed = src.as_chunks::<16>().0;
        let [r, g, b, a] =
            F32x4::deinterleave_4ch(core::array::from_fn(|i| F32x4::from_bytes_t(t, &packed[i])));
        let zero = F32x4::zero_t(t);
        let inv = F32x4::splat_t(t, 1.0) / a;
        let mask = a.simd_ne(zero);
        let out = F32x4::interleave_4ch([
            (r * inv) & mask,
            (g * inv) & mask,
            (b * inv) & mask,
            a & mask,
        ]);
        bytemuck::cast(out.map(|v| *v.as_bytes()))
    }

    #[rite]
    fn alpha_copy<const UNPREMULTIPLY: bool>(t: Wasm128Token, src: &[u8], dst: &mut [u8]) {
        let (src, dst) = if UNPREMULTIPLY {
            let (src_blocks, src_tail) = src.as_chunks::<64>();
            let (dst_blocks, dst_tail) = dst.as_chunks_mut::<64>();
            for (src, dst) in src_blocks.iter().zip(dst_blocks) {
                *dst = unpremultiply_four(t, src);
            }
            (src_tail, dst_tail)
        } else {
            (src, dst)
        };
        for (src, dst) in src
            .as_chunks::<16>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<16>().0)
        {
            *dst = alpha_pixel::<UNPREMULTIPLY>(t, src);
        }
    }

    #[rite]
    fn alpha_in_place<const UNPREMULTIPLY: bool>(t: Wasm128Token, src: &mut [u8]) {
        let src = if UNPREMULTIPLY {
            let (blocks, tail) = src.as_chunks_mut::<64>();
            for block in blocks {
                *block = unpremultiply_four(t, block);
            }
            tail
        } else {
            src
        };
        for pixel in src.as_chunks_mut::<16>().0 {
            *pixel = alpha_pixel::<UNPREMULTIPLY>(t, pixel);
        }
    }

    macro_rules! alpha_workers {
        ($unpremul:literal, $whole:ident, $strided:ident, $copy:ident, $copy_strided:ident) => {
            copy_workers!(alpha_copy::<$unpremul>, $copy, $copy_strided, 16, 16);

            #[arcane]
            pub(in crate::bytes) fn $whole(t: Wasm128Token, buf: &mut [u8]) {
                alpha_in_place::<$unpremul>(t, buf);
            }

            #[arcane]
            pub(in crate::bytes) fn $strided(
                t: Wasm128Token,
                buf: &mut [u8],
                w: usize,
                h: usize,
                stride: usize,
            ) {
                for y in 0..h {
                    alpha_in_place::<$unpremul>(t, &mut buf[y * stride..][..w * 16]);
                }
            }
        };
    }

    copy_workers!(
        u8_to_f32,
        convert_u8_to_f32_impl_wasm128,
        convert_u8_to_f32_strided_wasm128,
        1,
        4
    );
    copy_workers!(
        u16_to_f32,
        convert_u16_to_f32_impl_wasm128,
        convert_u16_to_f32_strided_wasm128,
        2,
        4
    );
    copy_workers!(
        f32_to_u8,
        convert_f32_to_u8_impl_wasm128,
        convert_f32_to_u8_strided_wasm128,
        4,
        1
    );
    copy_workers!(
        f32_to_u16,
        convert_f32_to_u16_impl_wasm128,
        convert_f32_to_u16_strided_wasm128,
        4,
        2
    );
    alpha_workers!(
        false,
        premul_f32_impl_wasm128,
        premul_f32_strided_wasm128,
        premul_f32_copy_impl_wasm128,
        premul_f32_copy_strided_wasm128
    );
    alpha_workers!(
        true,
        unpremul_f32_impl_wasm128,
        unpremul_f32_strided_wasm128,
        unpremul_f32_copy_impl_wasm128,
        unpremul_f32_copy_strided_wasm128
    );
}

#[cfg(feature = "experimental")]
pub(super) use float_pixels::*;
