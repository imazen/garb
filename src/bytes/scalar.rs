use archmage::prelude::*;

use super::swap_br_u32;

// ===========================================================================
// Scalar row implementations
// ===========================================================================

pub(super) fn swap_bgr_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

pub(super) fn copy_swap_bgr_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

pub(super) fn rgba_to_rgb_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

pub(super) fn bgra_to_rgb_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

pub(super) fn swap_br_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
    }
}

pub(super) fn copy_swap_br_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let v = u32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
        d.copy_from_slice(&swap_br_u32(v).to_ne_bytes());
    }
}

pub(super) fn fill_alpha_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        px[3] = 0xFF;
    }
}

pub(super) fn rgb_to_bgra_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 0xFF;
    }
}

pub(super) fn rgb_to_rgba_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 0xFF;
    }
}

pub(super) fn gray_to_4bpp_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (&v, d) in src.iter().zip(dst.chunks_exact_mut(4)) {
        d[0] = v;
        d[1] = v;
        d[2] = v;
        d[3] = 0xFF;
    }
}

pub(super) fn gray_alpha_to_4bpp_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
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

pub(super) fn swap_br_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    swap_br_row_scalar(t, b);
}
pub(super) fn copy_swap_br_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    copy_swap_br_row_scalar(t, s, d);
}
pub(super) fn fill_alpha_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    fill_alpha_row_scalar(t, b);
}
pub(super) fn rgb_to_bgra_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_bgra_row_scalar(t, s, d);
}
pub(super) fn rgb_to_rgba_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_rgba_row_scalar(t, s, d);
}
pub(super) fn gray_to_4bpp_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_row_scalar(t, s, d);
}
pub(super) fn gray_alpha_to_4bpp_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_row_scalar(t, s, d);
}
pub(super) fn swap_bgr_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    swap_bgr_row_scalar(t, b);
}
pub(super) fn copy_swap_bgr_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    copy_swap_bgr_row_scalar(t, s, d);
}
pub(super) fn rgba_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgba_to_rgb_row_scalar(t, s, d);
}
pub(super) fn bgra_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    bgra_to_rgb_row_scalar(t, s, d);
}

// ===========================================================================
// Scalar strided wrappers
// ===========================================================================

pub(super) fn swap_br_strided_scalar(t: ScalarToken, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        swap_br_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
pub(super) fn copy_swap_br_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_br_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn fill_alpha_strided_scalar(t: ScalarToken, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        fill_alpha_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
pub(super) fn rgb_to_bgra_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_bgra_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn rgb_to_rgba_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_rgba_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn gray_to_4bpp_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn gray_alpha_to_4bpp_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_row_scalar(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn swap_bgr_strided_scalar(t: ScalarToken, buf: &mut [u8], w: usize, h: usize, stride: usize) {
    for y in 0..h {
        swap_bgr_row_scalar(t, &mut buf[y * stride..][..w * 3]);
    }
}
pub(super) fn copy_swap_bgr_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_swap_bgr_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 3]);
    }
}
pub(super) fn rgba_to_rgb_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgba_to_rgb_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
pub(super) fn bgra_to_rgb_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        bgra_to_rgb_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
