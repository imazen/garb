use archmage::prelude::*;

use super::swap_br_u32;

// ===========================================================================
// Core scalar row implementations
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
// ARGB/XRGB row implementations
// ===========================================================================

/// Rotate each 4-byte pixel left by 1: [a,b,c,d]→[b,c,d,a]. ARGB→RGBA, ABGR→BGRA.
pub(super) fn rotate_left_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        let a = px[0];
        px[0] = px[1];
        px[1] = px[2];
        px[2] = px[3];
        px[3] = a;
    }
}

/// Rotate each 4-byte pixel right by 1: [a,b,c,d]→[d,a,b,c]. RGBA→ARGB, BGRA→ABGR.
pub(super) fn rotate_right_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        let d = px[3];
        px[3] = px[2];
        px[2] = px[1];
        px[1] = px[0];
        px[0] = d;
    }
}

/// Reverse each 4-byte pixel: [a,b,c,d]→[d,c,b,a]. ARGB↔BGRA, ABGR↔RGBA.
pub(super) fn reverse_4bpp_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        let v = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
        px.copy_from_slice(&v.swap_bytes().to_ne_bytes());
    }
}

pub(super) fn copy_rotate_left_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
        d[3] = s[0];
    }
}

pub(super) fn copy_rotate_right_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[3];
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
}

pub(super) fn copy_reverse_4bpp_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
}

/// Set byte 0 to 0xFF in each 4-byte pixel (XRGB/XBGR → ARGB/ABGR).
pub(super) fn fill_alpha_first_row_scalar(_token: ScalarToken, row: &mut [u8]) {
    for px in row.chunks_exact_mut(4) {
        px[0] = 0xFF;
    }
}

/// RGB → ARGB: [R,G,B] → [0xFF,R,G,B].
pub(super) fn rgb_to_argb_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
}

/// RGB → ABGR: [R,G,B] → [0xFF,B,G,R]. Same as BGR → ARGB.
pub(super) fn rgb_to_abgr_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
}

/// ARGB → RGB: [A,R,G,B] → [R,G,B]. Same as ABGR → BGR.
pub(super) fn argb_to_rgb_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
    }
}

/// ARGB → BGR: [A,R,G,B] → [B,G,R]. Same as ABGR → RGB.
pub(super) fn argb_to_bgr_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
    }
}

/// Gray → ARGB: G → [0xFF,G,G,G].
pub(super) fn gray_to_4bpp_alpha_first_row_scalar(_token: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (&v, d) in src.iter().zip(dst.chunks_exact_mut(4)) {
        d[0] = 0xFF;
        d[1] = v;
        d[2] = v;
        d[3] = v;
    }
}

/// GrayAlpha → ARGB: [G,A] → [A,G,G,G].
pub(super) fn gray_alpha_to_4bpp_alpha_first_row_scalar(
    _token: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
) {
    for (ga, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
        d[0] = ga[1];
        d[1] = ga[0];
        d[2] = ga[0];
        d[3] = ga[0];
    }
}

// ===========================================================================
// Core scalar contiguous wrappers
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
pub(super) fn rotate_left_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    rotate_left_row_scalar(t, b);
}
pub(super) fn copy_rotate_left_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    copy_rotate_left_row_scalar(t, s, d);
}
pub(super) fn rotate_right_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    rotate_right_row_scalar(t, b);
}
pub(super) fn copy_rotate_right_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    copy_rotate_right_row_scalar(t, s, d);
}
pub(super) fn reverse_4bpp_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    reverse_4bpp_row_scalar(t, b);
}
pub(super) fn copy_reverse_4bpp_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    copy_reverse_4bpp_row_scalar(t, s, d);
}
pub(super) fn fill_alpha_first_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    fill_alpha_first_row_scalar(t, b);
}
pub(super) fn rgb_to_argb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_argb_row_scalar(t, s, d);
}
pub(super) fn rgb_to_abgr_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_abgr_row_scalar(t, s, d);
}
pub(super) fn argb_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    argb_to_rgb_row_scalar(t, s, d);
}
pub(super) fn argb_to_bgr_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    argb_to_bgr_row_scalar(t, s, d);
}
pub(super) fn gray_to_4bpp_alpha_first_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_to_4bpp_alpha_first_row_scalar(t, s, d);
}
pub(super) fn gray_alpha_to_4bpp_alpha_first_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_4bpp_alpha_first_row_scalar(t, s, d);
}

// ===========================================================================
// Core scalar strided wrappers
// ===========================================================================

pub(super) fn swap_br_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
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
pub(super) fn fill_alpha_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
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
pub(super) fn swap_bgr_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
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

// ===========================================================================
// ARGB/XRGB scalar strided wrappers
// ===========================================================================

pub(super) fn rotate_left_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_left_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
pub(super) fn copy_rotate_left_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_left_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn rotate_right_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        rotate_right_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
pub(super) fn copy_rotate_right_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_rotate_right_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn reverse_4bpp_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        reverse_4bpp_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
pub(super) fn copy_reverse_4bpp_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        copy_reverse_4bpp_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn fill_alpha_first_strided_scalar(
    t: ScalarToken,
    buf: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
) {
    for y in 0..h {
        fill_alpha_first_row_scalar(t, &mut buf[y * stride..][..w * 4]);
    }
}
pub(super) fn rgb_to_argb_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_argb_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn rgb_to_abgr_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        rgb_to_abgr_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn argb_to_rgb_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        argb_to_rgb_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
pub(super) fn argb_to_bgr_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        argb_to_bgr_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 3]);
    }
}
pub(super) fn gray_to_4bpp_alpha_first_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_to_4bpp_alpha_first_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
    }
}
pub(super) fn gray_alpha_to_4bpp_alpha_first_strided_scalar(
    t: ScalarToken,
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ss: usize,
    ds: usize,
) {
    for y in 0..h {
        gray_alpha_to_4bpp_alpha_first_row_scalar(
            t,
            &src[y * ss..][..w * 2],
            &mut dst[y * ds..][..w * 4],
        );
    }
}

// ===========================================================================
// Experimental: depth, gray layout, luma, premul (feature = "experimental")
// ===========================================================================

#[cfg(feature = "experimental")]
mod experimental {
    use archmage::prelude::*;

    // -----------------------------------------------------------------------
    // Depth conversion row implementations
    // -----------------------------------------------------------------------

    pub(in crate::bytes) fn convert_u8_to_u16_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let dst16: &mut [u16] = bytemuck::cast_slice_mut(dst);
        for (s, d) in src.iter().zip(dst16.iter_mut()) {
            *d = (*s as u16) * 257;
        }
    }
    pub(in crate::bytes) fn convert_u16_to_u8_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let src16: &[u16] = bytemuck::cast_slice(src);
        for (s, d) in src16.iter().zip(dst.iter_mut()) {
            *d = ((*s as u32 * 255 + 32768) >> 16) as u8;
        }
    }
    pub(in crate::bytes) fn convert_u8_to_f32_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
        for (s, d) in src.iter().zip(dst_f.iter_mut()) {
            *d = *s as f32 / 255.0;
        }
    }
    pub(in crate::bytes) fn convert_f32_to_u8_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let src_f: &[f32] = bytemuck::cast_slice(src);
        for (s, d) in src_f.iter().zip(dst.iter_mut()) {
            *d = (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    pub(in crate::bytes) fn convert_u16_to_f32_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let src16: &[u16] = bytemuck::cast_slice(src);
        let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
        for (s, d) in src16.iter().zip(dst_f.iter_mut()) {
            *d = *s as f32 / 65535.0;
        }
    }
    pub(in crate::bytes) fn convert_f32_to_u16_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let src_f: &[f32] = bytemuck::cast_slice(src);
        let dst16: &mut [u16] = bytemuck::cast_slice_mut(dst);
        for (s, d) in src_f.iter().zip(dst16.iter_mut()) {
            *d = (s.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }

    // Depth contiguous wrappers
    pub(in crate::bytes) fn convert_u8_to_u16_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        convert_u8_to_u16_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn convert_u16_to_u8_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        convert_u16_to_u8_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn convert_u8_to_f32_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        convert_u8_to_f32_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn convert_f32_to_u8_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        convert_f32_to_u8_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn convert_u16_to_f32_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        convert_u16_to_f32_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn convert_f32_to_u16_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        convert_f32_to_u16_row_scalar(t, s, d);
    }

    // Depth strided wrappers
    pub(in crate::bytes) fn convert_u8_to_u16_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            convert_u8_to_u16_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 2]);
        }
    }
    pub(in crate::bytes) fn convert_u16_to_u8_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            convert_u16_to_u8_row_scalar(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w]);
        }
    }
    pub(in crate::bytes) fn convert_u8_to_f32_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            convert_u8_to_f32_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 4]);
        }
    }
    pub(in crate::bytes) fn convert_f32_to_u8_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            convert_f32_to_u8_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w]);
        }
    }
    pub(in crate::bytes) fn convert_u16_to_f32_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            convert_u16_to_f32_row_scalar(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 4]);
        }
    }
    pub(in crate::bytes) fn convert_f32_to_u16_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            convert_f32_to_u16_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w * 2]);
        }
    }

    // -----------------------------------------------------------------------
    // Gray layout conversion row implementations
    // -----------------------------------------------------------------------

    pub(in crate::bytes) fn gray_to_rgb_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
        for (&g, d) in src.iter().zip(dst.chunks_exact_mut(3)) {
            d[0] = g;
            d[1] = g;
            d[2] = g;
        }
    }
    pub(in crate::bytes) fn gray_alpha_to_rgb_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        for (ga, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(3)) {
            d[0] = ga[0];
            d[1] = ga[0];
            d[2] = ga[0];
        }
    }
    pub(in crate::bytes) fn gray_to_gray_alpha_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        for (&g, d) in src.iter().zip(dst.chunks_exact_mut(2)) {
            d[0] = g;
            d[1] = 0xFF;
        }
    }
    pub(in crate::bytes) fn gray_alpha_to_gray_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        for (ga, d) in src.chunks_exact(2).zip(dst.iter_mut()) {
            *d = ga[0];
        }
    }
    pub(in crate::bytes) fn rgb_to_gray_identity_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        for (px, d) in src.chunks_exact(3).zip(dst.iter_mut()) {
            *d = px[0];
        }
    }
    pub(in crate::bytes) fn rgba_to_gray_identity_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        for (px, d) in src.chunks_exact(4).zip(dst.iter_mut()) {
            *d = px[0];
        }
    }

    // Gray layout contiguous wrappers
    pub(in crate::bytes) fn gray_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        gray_to_rgb_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn gray_alpha_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        gray_alpha_to_rgb_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn gray_to_gray_alpha_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        gray_to_gray_alpha_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn gray_alpha_to_gray_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        gray_alpha_to_gray_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn rgb_to_gray_identity_impl_scalar(
        t: ScalarToken,
        s: &[u8],
        d: &mut [u8],
    ) {
        rgb_to_gray_identity_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn rgba_to_gray_identity_impl_scalar(
        t: ScalarToken,
        s: &[u8],
        d: &mut [u8],
    ) {
        rgba_to_gray_identity_row_scalar(t, s, d);
    }

    // Gray layout strided wrappers
    pub(in crate::bytes) fn gray_to_rgb_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            gray_to_rgb_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 3]);
        }
    }
    pub(in crate::bytes) fn gray_alpha_to_rgb_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            gray_alpha_to_rgb_row_scalar(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w * 3]);
        }
    }
    pub(in crate::bytes) fn gray_to_gray_alpha_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            gray_to_gray_alpha_row_scalar(t, &src[y * ss..][..w], &mut dst[y * ds..][..w * 2]);
        }
    }
    pub(in crate::bytes) fn gray_alpha_to_gray_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            gray_alpha_to_gray_row_scalar(t, &src[y * ss..][..w * 2], &mut dst[y * ds..][..w]);
        }
    }
    pub(in crate::bytes) fn rgb_to_gray_identity_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            rgb_to_gray_identity_row_scalar(t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w]);
        }
    }
    pub(in crate::bytes) fn rgba_to_gray_identity_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            rgba_to_gray_identity_row_scalar(t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w]);
        }
    }

    // -----------------------------------------------------------------------
    // Weighted luma conversion row implementations
    // -----------------------------------------------------------------------

    #[inline(always)]
    fn luma_3bpp_row(src: &[u8], dst: &mut [u8], w_r: u16, w_g: u16, w_b: u16) {
        for (px, d) in src.chunks_exact(3).zip(dst.iter_mut()) {
            *d = ((px[0] as u16 * w_r + px[1] as u16 * w_g + px[2] as u16 * w_b + 128) >> 8) as u8;
        }
    }

    #[inline(always)]
    fn luma_4bpp_row(src: &[u8], dst: &mut [u8], w_r: u16, w_g: u16, w_b: u16) {
        for (px, d) in src.chunks_exact(4).zip(dst.iter_mut()) {
            *d = ((px[0] as u16 * w_r + px[1] as u16 * w_g + px[2] as u16 * w_b + 128) >> 8) as u8;
        }
    }

    macro_rules! luma_scalar_all {
        ($matrix:ident, $r:expr, $g:expr, $b:expr) => {
            paste::paste! {
                // Row impls
                pub(in crate::bytes) fn [<rgb_to_gray_ $matrix _row_scalar>](_t: ScalarToken, s: &[u8], d: &mut [u8]) { luma_3bpp_row(s, d, $r, $g, $b); }
                pub(in crate::bytes) fn [<bgr_to_gray_ $matrix _row_scalar>](_t: ScalarToken, s: &[u8], d: &mut [u8]) { luma_3bpp_row(s, d, $b, $g, $r); }
                pub(in crate::bytes) fn [<rgba_to_gray_ $matrix _row_scalar>](_t: ScalarToken, s: &[u8], d: &mut [u8]) { luma_4bpp_row(s, d, $r, $g, $b); }
                pub(in crate::bytes) fn [<bgra_to_gray_ $matrix _row_scalar>](_t: ScalarToken, s: &[u8], d: &mut [u8]) { luma_4bpp_row(s, d, $b, $g, $r); }
                // Contiguous wrappers
                pub(in crate::bytes) fn [<rgb_to_gray_ $matrix _impl_scalar>](t: ScalarToken, s: &[u8], d: &mut [u8]) { [<rgb_to_gray_ $matrix _row_scalar>](t, s, d); }
                pub(in crate::bytes) fn [<bgr_to_gray_ $matrix _impl_scalar>](t: ScalarToken, s: &[u8], d: &mut [u8]) { [<bgr_to_gray_ $matrix _row_scalar>](t, s, d); }
                pub(in crate::bytes) fn [<rgba_to_gray_ $matrix _impl_scalar>](t: ScalarToken, s: &[u8], d: &mut [u8]) { [<rgba_to_gray_ $matrix _row_scalar>](t, s, d); }
                pub(in crate::bytes) fn [<bgra_to_gray_ $matrix _impl_scalar>](t: ScalarToken, s: &[u8], d: &mut [u8]) { [<bgra_to_gray_ $matrix _row_scalar>](t, s, d); }
                // Strided wrappers
                pub(in crate::bytes) fn [<rgb_to_gray_ $matrix _strided_scalar>](t: ScalarToken, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize) {
                    for y in 0..h { [<rgb_to_gray_ $matrix _row_scalar>](t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w]); }
                }
                pub(in crate::bytes) fn [<bgr_to_gray_ $matrix _strided_scalar>](t: ScalarToken, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize) {
                    for y in 0..h { [<bgr_to_gray_ $matrix _row_scalar>](t, &src[y * ss..][..w * 3], &mut dst[y * ds..][..w]); }
                }
                pub(in crate::bytes) fn [<rgba_to_gray_ $matrix _strided_scalar>](t: ScalarToken, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize) {
                    for y in 0..h { [<rgba_to_gray_ $matrix _row_scalar>](t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w]); }
                }
                pub(in crate::bytes) fn [<bgra_to_gray_ $matrix _strided_scalar>](t: ScalarToken, src: &[u8], dst: &mut [u8], w: usize, h: usize, ss: usize, ds: usize) {
                    for y in 0..h { [<bgra_to_gray_ $matrix _row_scalar>](t, &src[y * ss..][..w * 4], &mut dst[y * ds..][..w]); }
                }
            }
        };
    }

    luma_scalar_all!(bt709, 54, 183, 19);
    luma_scalar_all!(bt601, 77, 150, 29);
    luma_scalar_all!(bt2020, 67, 174, 15);

    // -----------------------------------------------------------------------
    // f32 alpha premultiplication
    // -----------------------------------------------------------------------

    pub(in crate::bytes) fn premul_f32_row_scalar(_t: ScalarToken, buf: &mut [u8]) {
        let floats: &mut [f32] = bytemuck::cast_slice_mut(buf);
        for px in floats.chunks_exact_mut(4) {
            let a = px[3];
            px[0] *= a;
            px[1] *= a;
            px[2] *= a;
        }
    }
    pub(in crate::bytes) fn premul_f32_copy_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let src_f: &[f32] = bytemuck::cast_slice(src);
        let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
        for (s, d) in src_f.chunks_exact(4).zip(dst_f.chunks_exact_mut(4)) {
            let a = s[3];
            d[0] = s[0] * a;
            d[1] = s[1] * a;
            d[2] = s[2] * a;
            d[3] = a;
        }
    }
    pub(in crate::bytes) fn unpremul_f32_row_scalar(_t: ScalarToken, buf: &mut [u8]) {
        let floats: &mut [f32] = bytemuck::cast_slice_mut(buf);
        for px in floats.chunks_exact_mut(4) {
            let a = px[3];
            if a == 0.0 {
                px[0] = 0.0;
                px[1] = 0.0;
                px[2] = 0.0;
            } else {
                let inv_a = 1.0 / a;
                px[0] *= inv_a;
                px[1] *= inv_a;
                px[2] *= inv_a;
            }
        }
    }
    pub(in crate::bytes) fn unpremul_f32_copy_row_scalar(
        _t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
    ) {
        let src_f: &[f32] = bytemuck::cast_slice(src);
        let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
        for (s, d) in src_f.chunks_exact(4).zip(dst_f.chunks_exact_mut(4)) {
            let a = s[3];
            if a == 0.0 {
                d[0] = 0.0;
                d[1] = 0.0;
                d[2] = 0.0;
                d[3] = 0.0;
            } else {
                let inv_a = 1.0 / a;
                d[0] = s[0] * inv_a;
                d[1] = s[1] * inv_a;
                d[2] = s[2] * inv_a;
                d[3] = a;
            }
        }
    }

    // Premul contiguous wrappers
    pub(in crate::bytes) fn premul_f32_impl_scalar(t: ScalarToken, b: &mut [u8]) {
        premul_f32_row_scalar(t, b);
    }
    pub(in crate::bytes) fn premul_f32_copy_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        premul_f32_copy_row_scalar(t, s, d);
    }
    pub(in crate::bytes) fn unpremul_f32_impl_scalar(t: ScalarToken, b: &mut [u8]) {
        unpremul_f32_row_scalar(t, b);
    }
    pub(in crate::bytes) fn unpremul_f32_copy_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
        unpremul_f32_copy_row_scalar(t, s, d);
    }

    // Premul strided wrappers
    pub(in crate::bytes) fn premul_f32_strided_scalar(
        t: ScalarToken,
        buf: &mut [u8],
        w: usize,
        h: usize,
        stride: usize,
    ) {
        for y in 0..h {
            premul_f32_row_scalar(t, &mut buf[y * stride..][..w * 16]);
        }
    }
    pub(in crate::bytes) fn premul_f32_copy_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            premul_f32_copy_row_scalar(t, &src[y * ss..][..w * 16], &mut dst[y * ds..][..w * 16]);
        }
    }
    pub(in crate::bytes) fn unpremul_f32_strided_scalar(
        t: ScalarToken,
        buf: &mut [u8],
        w: usize,
        h: usize,
        stride: usize,
    ) {
        for y in 0..h {
            unpremul_f32_row_scalar(t, &mut buf[y * stride..][..w * 16]);
        }
    }
    pub(in crate::bytes) fn unpremul_f32_copy_strided_scalar(
        t: ScalarToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            unpremul_f32_copy_row_scalar(t, &src[y * ss..][..w * 16], &mut dst[y * ds..][..w * 16]);
        }
    }
    // -----------------------------------------------------------------------
    // Packed pixel format expansion (2bpp → 4bpp)
    // -----------------------------------------------------------------------

    // ## Bit layout (all formats are little-endian u16)
    //
    // **RGB565**: bits `[15:11]=R(5) [10:5]=G(6) [4:0]=B(5)`
    //   Byte 0 = low byte, Byte 1 = high byte (standard little-endian).
    //   This matches OpenGL `GL_UNSIGNED_SHORT_5_6_5`, Vulkan `R5G6B5_UNORM`,
    //   and Android `ENCODING_RGB_565`.
    //
    // **RGBA4444**: bits `[15:12]=R(4) [11:8]=G(4) [7:4]=B(4) [3:0]=A(4)`
    //   Byte 0 = low byte, Byte 1 = high byte (standard little-endian).
    //   This matches OpenGL `GL_UNSIGNED_SHORT_4_4_4_4`.
    //
    // ## Big-endian data
    //
    // These functions expect **little-endian** u16 values in the byte slice.
    // If your data is big-endian (e.g. from a network protocol or a big-endian
    // file format), swap each pair of bytes before calling these functions.
    // A simple pre-pass: `for pair in src.chunks_exact_mut(2) { pair.swap(0, 1); }`
    //
    // ## Expansion rounding
    //
    // Channels are expanded to 8 bits by replicating the top bits into the
    // vacated low bits, which maps the full source range [0, max] to [0, 255]
    // exactly (e.g. 5-bit 31 → `31<<3 | 31>>2` = 255, 6-bit 63 → `63<<2 | 63>>4` = 255).

    // Packed format contiguous dispatchers (autoversion generates per-tier variants)
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgb565_to_rgba_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r5 = (v >> 11) & 0x1F;
            let g6 = (v >> 5) & 0x3F;
            let b5 = v & 0x1F;
            d[0] = (r5 << 3 | r5 >> 2) as u8;
            d[1] = (g6 << 2 | g6 >> 4) as u8;
            d[2] = (b5 << 3 | b5 >> 2) as u8;
            d[3] = 0xFF;
        }
    }
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgb565_to_bgra_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r5 = (v >> 11) & 0x1F;
            let g6 = (v >> 5) & 0x3F;
            let b5 = v & 0x1F;
            d[0] = (b5 << 3 | b5 >> 2) as u8;
            d[1] = (g6 << 2 | g6 >> 4) as u8;
            d[2] = (r5 << 3 | r5 >> 2) as u8;
            d[3] = 0xFF;
        }
    }
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba4444_to_rgba_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r4 = (v >> 12) & 0xF;
            let g4 = (v >> 8) & 0xF;
            let b4 = (v >> 4) & 0xF;
            let a4 = v & 0xF;
            d[0] = (r4 << 4 | r4) as u8;
            d[1] = (g4 << 4 | g4) as u8;
            d[2] = (b4 << 4 | b4) as u8;
            d[3] = (a4 << 4 | a4) as u8;
        }
    }
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba4444_to_bgra_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(2).zip(d.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r4 = (v >> 12) & 0xF;
            let g4 = (v >> 8) & 0xF;
            let b4 = (v >> 4) & 0xF;
            let a4 = v & 0xF;
            d[0] = (b4 << 4 | b4) as u8;
            d[1] = (g4 << 4 | g4) as u8;
            d[2] = (r4 << 4 | r4) as u8;
            d[3] = (a4 << 4 | a4) as u8;
        }
    }

    // Packed format strided dispatchers
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgb565_to_rgba_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 2]
                .chunks_exact(2)
                .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
            {
                let v = u16::from_le_bytes([s[0], s[1]]);
                let r5 = (v >> 11) & 0x1F;
                let g6 = (v >> 5) & 0x3F;
                let b5 = v & 0x1F;
                d[0] = (r5 << 3 | r5 >> 2) as u8;
                d[1] = (g6 << 2 | g6 >> 4) as u8;
                d[2] = (b5 << 3 | b5 >> 2) as u8;
                d[3] = 0xFF;
            }
        }
    }
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgb565_to_bgra_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 2]
                .chunks_exact(2)
                .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
            {
                let v = u16::from_le_bytes([s[0], s[1]]);
                let r5 = (v >> 11) & 0x1F;
                let g6 = (v >> 5) & 0x3F;
                let b5 = v & 0x1F;
                d[0] = (b5 << 3 | b5 >> 2) as u8;
                d[1] = (g6 << 2 | g6 >> 4) as u8;
                d[2] = (r5 << 3 | r5 >> 2) as u8;
                d[3] = 0xFF;
            }
        }
    }
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba4444_to_rgba_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 2]
                .chunks_exact(2)
                .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
            {
                let v = u16::from_le_bytes([s[0], s[1]]);
                let r4 = (v >> 12) & 0xF;
                let g4 = (v >> 8) & 0xF;
                let b4 = (v >> 4) & 0xF;
                let a4 = v & 0xF;
                d[0] = (r4 << 4 | r4) as u8;
                d[1] = (g4 << 4 | g4) as u8;
                d[2] = (b4 << 4 | b4) as u8;
                d[3] = (a4 << 4 | a4) as u8;
            }
        }
    }
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba4444_to_bgra_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 2]
                .chunks_exact(2)
                .zip(dst[y * ds..][..w * 4].chunks_exact_mut(4))
            {
                let v = u16::from_le_bytes([s[0], s[1]]);
                let r4 = (v >> 12) & 0xF;
                let g4 = (v >> 8) & 0xF;
                let b4 = (v >> 4) & 0xF;
                let a4 = v & 0xF;
                d[0] = (b4 << 4 | b4) as u8;
                d[1] = (g4 << 4 | g4) as u8;
                d[2] = (r4 << 4 | r4) as u8;
                d[3] = (a4 << 4 | a4) as u8;
            }
        }
    }
    // -----------------------------------------------------------------------
    // Packed pixel format compression (4bpp → 2bpp)
    // -----------------------------------------------------------------------
    //
    // ## Rounding
    //
    // Channels are compressed from 8 bits using round-to-nearest:
    //   compress_N(v) = (v * max_N + 128) >> 8
    // where max_N is the maximum value for N bits (31, 63, or 15).
    //
    // This guarantees:
    // - Perfect roundtrip: expand(compress(expand(x))) == expand(x) for all x
    // - Nearest match for values not from N bits
    // - Endpoints preserved: 0→0, 255→max
    //
    // Alpha is ignored (dropped) for RGB565, preserved for RGBA4444.
    // Output is little-endian u16, matching the expand functions.

    // Contiguous compress dispatchers

    /// RGBA (4bpp) → RGB565 (LE u16, 2bpp). Alpha dropped.
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba_to_rgb565_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
            let r5 = (s[0] as u16 * 31 + 128) >> 8;
            let g6 = (s[1] as u16 * 63 + 128) >> 8;
            let b5 = (s[2] as u16 * 31 + 128) >> 8;
            d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
        }
    }

    /// BGRA (4bpp) → RGB565 (LE u16, 2bpp). Alpha dropped.
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn bgra_to_rgb565_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
            let r5 = (s[2] as u16 * 31 + 128) >> 8;
            let g6 = (s[1] as u16 * 63 + 128) >> 8;
            let b5 = (s[0] as u16 * 31 + 128) >> 8;
            d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
        }
    }

    /// RGBA (4bpp) → RGBA4444 (LE u16, 2bpp).
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba_to_rgba4444_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
            let r4 = (s[0] as u16 * 15 + 128) >> 8;
            let g4 = (s[1] as u16 * 15 + 128) >> 8;
            let b4 = (s[2] as u16 * 15 + 128) >> 8;
            let a4 = (s[3] as u16 * 15 + 128) >> 8;
            d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
        }
    }

    /// BGRA (4bpp) → RGBA4444 (LE u16, 2bpp).
    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn bgra_to_rgba4444_impl(_: SimdToken, s: &[u8], d: &mut [u8]) {
        for (s, d) in s.chunks_exact(4).zip(d.chunks_exact_mut(2)) {
            let r4 = (s[2] as u16 * 15 + 128) >> 8;
            let g4 = (s[1] as u16 * 15 + 128) >> 8;
            let b4 = (s[0] as u16 * 15 + 128) >> 8;
            let a4 = (s[3] as u16 * 15 + 128) >> 8;
            d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
        }
    }

    // Strided compress dispatchers

    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba_to_rgb565_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 4]
                .chunks_exact(4)
                .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
            {
                let r5 = (s[0] as u16 * 31 + 128) >> 8;
                let g6 = (s[1] as u16 * 63 + 128) >> 8;
                let b5 = (s[2] as u16 * 31 + 128) >> 8;
                d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
            }
        }
    }

    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn bgra_to_rgb565_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 4]
                .chunks_exact(4)
                .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
            {
                let r5 = (s[2] as u16 * 31 + 128) >> 8;
                let g6 = (s[1] as u16 * 63 + 128) >> 8;
                let b5 = (s[0] as u16 * 31 + 128) >> 8;
                d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
            }
        }
    }

    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn rgba_to_rgba4444_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 4]
                .chunks_exact(4)
                .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
            {
                let r4 = (s[0] as u16 * 15 + 128) >> 8;
                let g4 = (s[1] as u16 * 15 + 128) >> 8;
                let b4 = (s[2] as u16 * 15 + 128) >> 8;
                let a4 = (s[3] as u16 * 15 + 128) >> 8;
                d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
            }
        }
    }

    #[autoversion(v3, neon, wasm128)]
    pub(in crate::bytes) fn bgra_to_rgba4444_strided_impl(
        _: SimdToken,
        src: &[u8],
        dst: &mut [u8],
        w: usize,
        h: usize,
        ss: usize,
        ds: usize,
    ) {
        for y in 0..h {
            for (s, d) in src[y * ss..][..w * 4]
                .chunks_exact(4)
                .zip(dst[y * ds..][..w * 2].chunks_exact_mut(2))
            {
                let r4 = (s[2] as u16 * 15 + 128) >> 8;
                let g4 = (s[1] as u16 * 15 + 128) >> 8;
                let b4 = (s[0] as u16 * 15 + 128) >> 8;
                let a4 = (s[3] as u16 * 15 + 128) >> 8;
                d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
            }
        }
    }
}

#[cfg(feature = "experimental")]
pub(super) use experimental::*;
