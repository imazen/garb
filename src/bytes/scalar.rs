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
// Depth conversion row implementations
// ===========================================================================

pub(super) fn convert_u8_to_u16_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(dst);
    for (s, d) in src.iter().zip(dst16.iter_mut()) {
        *d = (*s as u16) * 257;
    }
}

pub(super) fn convert_u16_to_u8_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    let src16: &[u16] = bytemuck::cast_slice(src);
    for (s, d) in src16.iter().zip(dst.iter_mut()) {
        *d = ((*s as u32 * 255 + 32768) >> 16) as u8;
    }
}

pub(super) fn convert_u8_to_f32_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
    for (s, d) in src.iter().zip(dst_f.iter_mut()) {
        *d = *s as f32 / 255.0;
    }
}

pub(super) fn convert_f32_to_u8_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    let src_f: &[f32] = bytemuck::cast_slice(src);
    for (s, d) in src_f.iter().zip(dst.iter_mut()) {
        *d = (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

pub(super) fn convert_u16_to_f32_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    let src16: &[u16] = bytemuck::cast_slice(src);
    let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst);
    for (s, d) in src16.iter().zip(dst_f.iter_mut()) {
        *d = *s as f32 / 65535.0;
    }
}

pub(super) fn convert_f32_to_u16_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    let src_f: &[f32] = bytemuck::cast_slice(src);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(dst);
    for (s, d) in src_f.iter().zip(dst16.iter_mut()) {
        *d = (s.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

// Depth conversion contiguous wrappers
pub(super) fn convert_u8_to_u16_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    convert_u8_to_u16_row_scalar(t, s, d);
}
pub(super) fn convert_u16_to_u8_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    convert_u16_to_u8_row_scalar(t, s, d);
}
pub(super) fn convert_u8_to_f32_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    convert_u8_to_f32_row_scalar(t, s, d);
}
pub(super) fn convert_f32_to_u8_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    convert_f32_to_u8_row_scalar(t, s, d);
}
pub(super) fn convert_u16_to_f32_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    convert_u16_to_f32_row_scalar(t, s, d);
}
pub(super) fn convert_f32_to_u16_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    convert_f32_to_u16_row_scalar(t, s, d);
}

// ===========================================================================
// Gray layout conversion row implementations
// ===========================================================================

pub(super) fn gray_to_rgb_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (&g, d) in src.iter().zip(dst.chunks_exact_mut(3)) {
        d[0] = g;
        d[1] = g;
        d[2] = g;
    }
}

pub(super) fn gray_alpha_to_rgb_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (ga, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(3)) {
        d[0] = ga[0];
        d[1] = ga[0];
        d[2] = ga[0];
    }
}

pub(super) fn gray_to_gray_alpha_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (&g, d) in src.iter().zip(dst.chunks_exact_mut(2)) {
        d[0] = g;
        d[1] = 0xFF;
    }
}

pub(super) fn gray_alpha_to_gray_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (ga, d) in src.chunks_exact(2).zip(dst.iter_mut()) {
        *d = ga[0];
    }
}

/// Identity gray extraction from 3bpp — takes byte 0 from each pixel.
pub(super) fn rgb_to_gray_identity_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (px, d) in src.chunks_exact(3).zip(dst.iter_mut()) {
        *d = px[0];
    }
}

/// Identity gray extraction from 4bpp — takes byte 0 from each pixel.
pub(super) fn rgba_to_gray_identity_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
    for (px, d) in src.chunks_exact(4).zip(dst.iter_mut()) {
        *d = px[0];
    }
}

// Gray layout contiguous wrappers
pub(super) fn gray_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_to_rgb_row_scalar(t, s, d);
}
pub(super) fn gray_alpha_to_rgb_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_rgb_row_scalar(t, s, d);
}
pub(super) fn gray_to_gray_alpha_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_to_gray_alpha_row_scalar(t, s, d);
}
pub(super) fn gray_alpha_to_gray_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    gray_alpha_to_gray_row_scalar(t, s, d);
}
pub(super) fn rgb_to_gray_identity_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgb_to_gray_identity_row_scalar(t, s, d);
}
pub(super) fn rgba_to_gray_identity_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    rgba_to_gray_identity_row_scalar(t, s, d);
}

// ===========================================================================
// Weighted luma conversion row implementations
// ===========================================================================

/// Parametric 3bpp→gray luma: gray = (R*w_r + G*w_g + B*w_b + 128) >> 8
#[inline(always)]
fn luma_3bpp_row(src: &[u8], dst: &mut [u8], w_r: u16, w_g: u16, w_b: u16) {
    for (px, d) in src.chunks_exact(3).zip(dst.iter_mut()) {
        *d = ((px[0] as u16 * w_r + px[1] as u16 * w_g + px[2] as u16 * w_b + 128) >> 8) as u8;
    }
}

/// Parametric 4bpp→gray luma: gray = (R*w_r + G*w_g + B*w_b + 128) >> 8, alpha ignored
#[inline(always)]
fn luma_4bpp_row(src: &[u8], dst: &mut [u8], w_r: u16, w_g: u16, w_b: u16) {
    for (px, d) in src.chunks_exact(4).zip(dst.iter_mut()) {
        *d = ((px[0] as u16 * w_r + px[1] as u16 * w_g + px[2] as u16 * w_b + 128) >> 8) as u8;
    }
}

// BT.709: [54, 183, 19]
pub(super) fn rgb_to_gray_bt709_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_3bpp_row(s, d, 54, 183, 19);
}
pub(super) fn bgr_to_gray_bt709_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_3bpp_row(s, d, 19, 183, 54);
}
pub(super) fn rgba_to_gray_bt709_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_4bpp_row(s, d, 54, 183, 19);
}
pub(super) fn bgra_to_gray_bt709_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_4bpp_row(s, d, 19, 183, 54);
}

// BT.601: [77, 150, 29]
pub(super) fn rgb_to_gray_bt601_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_3bpp_row(s, d, 77, 150, 29);
}
pub(super) fn bgr_to_gray_bt601_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_3bpp_row(s, d, 29, 150, 77);
}
pub(super) fn rgba_to_gray_bt601_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_4bpp_row(s, d, 77, 150, 29);
}
pub(super) fn bgra_to_gray_bt601_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_4bpp_row(s, d, 29, 150, 77);
}

// BT.2020: [67, 174, 15]
pub(super) fn rgb_to_gray_bt2020_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_3bpp_row(s, d, 67, 174, 15);
}
pub(super) fn bgr_to_gray_bt2020_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_3bpp_row(s, d, 15, 174, 67);
}
pub(super) fn rgba_to_gray_bt2020_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_4bpp_row(s, d, 67, 174, 15);
}
pub(super) fn bgra_to_gray_bt2020_row_scalar(_t: ScalarToken, s: &[u8], d: &mut [u8]) {
    luma_4bpp_row(s, d, 15, 174, 67);
}

// Luma contiguous wrappers — macro to reduce repetition
macro_rules! luma_impl_scalar {
    ($name:ident, $row:ident) => {
        pub(super) fn $name(t: ScalarToken, s: &[u8], d: &mut [u8]) {
            $row(t, s, d);
        }
    };
}
luma_impl_scalar!(rgb_to_gray_bt709_impl_scalar, rgb_to_gray_bt709_row_scalar);
luma_impl_scalar!(bgr_to_gray_bt709_impl_scalar, bgr_to_gray_bt709_row_scalar);
luma_impl_scalar!(
    rgba_to_gray_bt709_impl_scalar,
    rgba_to_gray_bt709_row_scalar
);
luma_impl_scalar!(
    bgra_to_gray_bt709_impl_scalar,
    bgra_to_gray_bt709_row_scalar
);
luma_impl_scalar!(rgb_to_gray_bt601_impl_scalar, rgb_to_gray_bt601_row_scalar);
luma_impl_scalar!(bgr_to_gray_bt601_impl_scalar, bgr_to_gray_bt601_row_scalar);
luma_impl_scalar!(
    rgba_to_gray_bt601_impl_scalar,
    rgba_to_gray_bt601_row_scalar
);
luma_impl_scalar!(
    bgra_to_gray_bt601_impl_scalar,
    bgra_to_gray_bt601_row_scalar
);
luma_impl_scalar!(
    rgb_to_gray_bt2020_impl_scalar,
    rgb_to_gray_bt2020_row_scalar
);
luma_impl_scalar!(
    bgr_to_gray_bt2020_impl_scalar,
    bgr_to_gray_bt2020_row_scalar
);
luma_impl_scalar!(
    rgba_to_gray_bt2020_impl_scalar,
    rgba_to_gray_bt2020_row_scalar
);
luma_impl_scalar!(
    bgra_to_gray_bt2020_impl_scalar,
    bgra_to_gray_bt2020_row_scalar
);

// Luma strided wrappers
macro_rules! luma_strided_scalar {
    ($name:ident, $row:ident, $bpp:expr) => {
        pub(super) fn $name(
            t: ScalarToken,
            src: &[u8],
            dst: &mut [u8],
            w: usize,
            h: usize,
            ss: usize,
            ds: usize,
        ) {
            for y in 0..h {
                $row(t, &src[y * ss..][..w * $bpp], &mut dst[y * ds..][..w]);
            }
        }
    };
}
luma_strided_scalar!(
    rgb_to_gray_bt709_strided_scalar,
    rgb_to_gray_bt709_row_scalar,
    3
);
luma_strided_scalar!(
    bgr_to_gray_bt709_strided_scalar,
    bgr_to_gray_bt709_row_scalar,
    3
);
luma_strided_scalar!(
    rgba_to_gray_bt709_strided_scalar,
    rgba_to_gray_bt709_row_scalar,
    4
);
luma_strided_scalar!(
    bgra_to_gray_bt709_strided_scalar,
    bgra_to_gray_bt709_row_scalar,
    4
);
luma_strided_scalar!(
    rgb_to_gray_bt601_strided_scalar,
    rgb_to_gray_bt601_row_scalar,
    3
);
luma_strided_scalar!(
    bgr_to_gray_bt601_strided_scalar,
    bgr_to_gray_bt601_row_scalar,
    3
);
luma_strided_scalar!(
    rgba_to_gray_bt601_strided_scalar,
    rgba_to_gray_bt601_row_scalar,
    4
);
luma_strided_scalar!(
    bgra_to_gray_bt601_strided_scalar,
    bgra_to_gray_bt601_row_scalar,
    4
);
luma_strided_scalar!(
    rgb_to_gray_bt2020_strided_scalar,
    rgb_to_gray_bt2020_row_scalar,
    3
);
luma_strided_scalar!(
    bgr_to_gray_bt2020_strided_scalar,
    bgr_to_gray_bt2020_row_scalar,
    3
);
luma_strided_scalar!(
    rgba_to_gray_bt2020_strided_scalar,
    rgba_to_gray_bt2020_row_scalar,
    4
);
luma_strided_scalar!(
    bgra_to_gray_bt2020_strided_scalar,
    bgra_to_gray_bt2020_row_scalar,
    4
);

// Gray layout strided wrappers
pub(super) fn gray_to_rgb_strided_scalar(
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
pub(super) fn gray_alpha_to_rgb_strided_scalar(
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
pub(super) fn gray_to_gray_alpha_strided_scalar(
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
pub(super) fn gray_alpha_to_gray_strided_scalar(
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
pub(super) fn rgb_to_gray_identity_strided_scalar(
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
pub(super) fn rgba_to_gray_identity_strided_scalar(
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

// Depth conversion strided wrappers
pub(super) fn convert_u8_to_u16_strided_scalar(
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
pub(super) fn convert_u16_to_u8_strided_scalar(
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
pub(super) fn convert_u8_to_f32_strided_scalar(
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
pub(super) fn convert_f32_to_u8_strided_scalar(
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
pub(super) fn convert_u16_to_f32_strided_scalar(
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
pub(super) fn convert_f32_to_u16_strided_scalar(
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

// ===========================================================================
// f32 alpha premultiplication
// ===========================================================================

pub(super) fn premul_f32_row_scalar(_t: ScalarToken, buf: &mut [u8]) {
    let floats: &mut [f32] = bytemuck::cast_slice_mut(buf);
    for px in floats.chunks_exact_mut(4) {
        let a = px[3];
        px[0] *= a;
        px[1] *= a;
        px[2] *= a;
    }
}

pub(super) fn premul_f32_copy_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
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

pub(super) fn unpremul_f32_row_scalar(_t: ScalarToken, buf: &mut [u8]) {
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

pub(super) fn unpremul_f32_copy_row_scalar(_t: ScalarToken, src: &[u8], dst: &mut [u8]) {
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
pub(super) fn premul_f32_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    premul_f32_row_scalar(t, b);
}
pub(super) fn premul_f32_copy_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    premul_f32_copy_row_scalar(t, s, d);
}
pub(super) fn unpremul_f32_impl_scalar(t: ScalarToken, b: &mut [u8]) {
    unpremul_f32_row_scalar(t, b);
}
pub(super) fn unpremul_f32_copy_impl_scalar(t: ScalarToken, s: &[u8], d: &mut [u8]) {
    unpremul_f32_copy_row_scalar(t, s, d);
}

// Premul strided wrappers
pub(super) fn premul_f32_strided_scalar(
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
pub(super) fn premul_f32_copy_strided_scalar(
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
pub(super) fn unpremul_f32_strided_scalar(
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
pub(super) fn unpremul_f32_copy_strided_scalar(
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
