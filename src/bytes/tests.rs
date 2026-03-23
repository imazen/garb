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
            fill_alpha_rgba(&mut data).unwrap();
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
        rgba_to_bgra_inplace_strided(&mut buf, w, h, stride).unwrap();
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
        rgb_to_bgra_strided(&src, &mut dst, w, h, src_stride, dst_stride).unwrap();
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
        fill_alpha_rgba_strided(&mut buf, w, h, stride).unwrap();
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
        gray_to_rgba_strided(&src, &mut dst, w, h, src_stride, dst_stride).unwrap();
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
        gray_alpha_to_rgba_strided(&src2, &mut dst2, w, h, src_stride2, dst_stride).unwrap();
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
// 3bpp swap and 4→3 strip — SIMD-dispatched, tested at every tier
// -----------------------------------------------------------------------

fn ref_swap_bgr(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for px in out.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    out
}

fn ref_rgba_to_rgb(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 4;
    let mut out = vec![0u8; n * 3];
    for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
    out
}

fn ref_bgra_to_rgb(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 4;
    let mut out = vec![0u8; n * 3];
    for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
    out
}

#[test]
fn permutation_swap_bgr_inplace() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let mut data = make_3bpp(n);
            let expected = ref_swap_bgr(&data);
            rgb_to_bgr_inplace(&mut data).unwrap();
            assert_eq!(data, expected, "swap_bgr_inplace n={n} tier={perm}");
        }
    });
    std::eprintln!("swap_bgr_inplace: {report}");
}

#[test]
fn permutation_copy_swap_bgr() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_3bpp(n);
            let expected = ref_swap_bgr(&src);
            let mut dst = vec![0u8; n * 3];
            rgb_to_bgr(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "copy_swap_bgr n={n} tier={perm}");
        }
    });
    std::eprintln!("copy_swap_bgr: {report}");
}

#[test]
fn permutation_rgba_to_rgb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_rgba_to_rgb(&src);
            let mut dst = vec![0u8; n * 3];
            rgba_to_rgb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "rgba_to_rgb n={n} tier={perm}");
        }
    });
    std::eprintln!("rgba_to_rgb: {report}");
}

#[test]
fn permutation_bgra_to_rgb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_bgra_to_rgb(&src);
            let mut dst = vec![0u8; n * 3];
            bgra_to_rgb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "bgra_to_rgb n={n} tier={perm}");
        }
    });
    std::eprintln!("bgra_to_rgb: {report}");
}

#[test]
fn permutation_strided_3bpp_and_strip() {
    let report = for_each_token_permutation(policy(), |perm| {
        let w = 10;
        let h = 3;

        // 3bpp swap strided
        let stride_3 = w * 3 + 6;
        let mut buf = vec![0xCCu8; stride_3 * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * stride_3 + x * 3;
                buf[i] = (y * w + x) as u8;
                buf[i + 1] = 100;
                buf[i + 2] = 200;
            }
        }
        let orig = buf.clone();
        rgb_to_bgr_inplace_strided(&mut buf, w, h, stride_3).unwrap();
        for y in 0..h {
            for x in 0..w {
                let i = y * stride_3 + x * 3;
                assert_eq!(
                    [buf[i], buf[i + 1], buf[i + 2]],
                    [orig[i + 2], orig[i + 1], orig[i]],
                    "strided 3bpp swap y={y} x={x} tier={perm}"
                );
            }
        }

        // 4→3 strided (RGBA→RGB)
        let src_stride = w * 4 + 8;
        let dst_stride = w * 3 + 6;
        let src4: Vec<u8> = (0..src_stride * h).map(|i| (i % 251) as u8).collect();
        let mut dst3 = vec![0u8; dst_stride * h];
        rgba_to_rgb_strided(&src4, &mut dst3, w, h, src_stride, dst_stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x * 4;
                let di = y * dst_stride + x * 3;
                assert_eq!(
                    [dst3[di], dst3[di + 1], dst3[di + 2]],
                    [src4[si], src4[si + 1], src4[si + 2]],
                    "strided rgba_to_rgb y={y} x={x} tier={perm}"
                );
            }
        }

        // 4→3 strided (BGRA→RGB)
        let mut dst3b = vec![0u8; dst_stride * h];
        bgra_to_rgb_strided(&src4, &mut dst3b, w, h, src_stride, dst_stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x * 4;
                let di = y * dst_stride + x * 3;
                assert_eq!(
                    [dst3b[di], dst3b[di + 1], dst3b[di + 2]],
                    [src4[si + 2], src4[si + 1], src4[si]],
                    "strided bgra_to_rgb y={y} x={x} tier={perm}"
                );
            }
        }
    });
    std::eprintln!("strided_3bpp_and_strip: {report}");
}

// -----------------------------------------------------------------------
// ARGB/XRGB operations
// -----------------------------------------------------------------------

fn ref_rotate_left(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for px in out.chunks_exact_mut(4) {
        let a = px[0];
        px[0] = px[1];
        px[1] = px[2];
        px[2] = px[3];
        px[3] = a;
    }
    out
}

fn ref_rotate_right(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for px in out.chunks_exact_mut(4) {
        let d = px[3];
        px[3] = px[2];
        px[2] = px[1];
        px[1] = px[0];
        px[0] = d;
    }
    out
}

fn ref_reverse_4bpp(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for px in out.chunks_exact_mut(4) {
        px.reverse();
    }
    out
}

fn ref_fill_alpha_first(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[0] = 255;
    }
    out
}

fn ref_rgb_to_argb(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 3;
    let mut out = vec![0u8; n * 4];
    for (s, d) in src.chunks_exact(3).zip(out.chunks_exact_mut(4)) {
        d[0] = 255;
        d[1] = s[0];
        d[2] = s[1];
        d[3] = s[2];
    }
    out
}

fn ref_rgb_to_abgr(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 3;
    let mut out = vec![0u8; n * 4];
    for (s, d) in src.chunks_exact(3).zip(out.chunks_exact_mut(4)) {
        d[0] = 255;
        d[1] = s[2];
        d[2] = s[1];
        d[3] = s[0];
    }
    out
}

fn ref_argb_to_rgb(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 4;
    let mut out = vec![0u8; n * 3];
    for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(3)) {
        d[0] = s[1];
        d[1] = s[2];
        d[2] = s[3];
    }
    out
}

fn ref_argb_to_bgr(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 4;
    let mut out = vec![0u8; n * 3];
    for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(3)) {
        d[0] = s[3];
        d[1] = s[2];
        d[2] = s[1];
    }
    out
}

fn ref_gray_to_argb(src: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; src.len() * 4];
    for (s, d) in src.iter().zip(out.chunks_exact_mut(4)) {
        d[0] = 255;
        d[1] = *s;
        d[2] = *s;
        d[3] = *s;
    }
    out
}

fn ref_gray_alpha_to_argb(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 2;
    let mut out = vec![0u8; n * 4];
    for (s, d) in src.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
        d[0] = s[1];
        d[1] = s[0];
        d[2] = s[0];
        d[3] = s[0];
    }
    out
}

#[test]
fn permutation_argb_to_rgba_inplace() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let mut data = make_4bpp(n);
            let expected = ref_rotate_left(&data);
            argb_to_rgba_inplace(&mut data).unwrap();
            assert_eq!(data, expected, "argb_to_rgba_inplace n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_to_rgba_inplace: {report}");
}

#[test]
fn permutation_argb_to_rgba_copy() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_rotate_left(&src);
            let mut dst = vec![0u8; n * 4];
            argb_to_rgba(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "argb_to_rgba n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_to_rgba: {report}");
}

#[test]
fn permutation_rgba_to_argb_inplace() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let mut data = make_4bpp(n);
            let expected = ref_rotate_right(&data);
            rgba_to_argb_inplace(&mut data).unwrap();
            assert_eq!(data, expected, "rgba_to_argb_inplace n={n} tier={perm}");
        }
    });
    std::eprintln!("rgba_to_argb_inplace: {report}");
}

#[test]
fn permutation_rgba_to_argb_copy() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_rotate_right(&src);
            let mut dst = vec![0u8; n * 4];
            rgba_to_argb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "rgba_to_argb n={n} tier={perm}");
        }
    });
    std::eprintln!("rgba_to_argb: {report}");
}

#[test]
fn permutation_argb_to_bgra_inplace() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let mut data = make_4bpp(n);
            let expected = ref_reverse_4bpp(&data);
            argb_to_bgra_inplace(&mut data).unwrap();
            assert_eq!(data, expected, "argb_to_bgra_inplace n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_to_bgra_inplace: {report}");
}

#[test]
fn permutation_argb_to_bgra_copy() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_reverse_4bpp(&src);
            let mut dst = vec![0u8; n * 4];
            argb_to_bgra(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "argb_to_bgra n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_to_bgra: {report}");
}

#[test]
fn permutation_fill_alpha_argb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let mut data = make_4bpp(n);
            let expected = ref_fill_alpha_first(&data);
            fill_alpha_argb(&mut data).unwrap();
            assert_eq!(data, expected, "fill_alpha_argb n={n} tier={perm}");
        }
    });
    std::eprintln!("fill_alpha_argb: {report}");
}

#[test]
fn permutation_rgb_to_argb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_3bpp(n);
            let expected = ref_rgb_to_argb(&src);
            let mut dst = vec![0u8; n * 4];
            rgb_to_argb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "rgb_to_argb n={n} tier={perm}");
        }
    });
    std::eprintln!("rgb_to_argb: {report}");
}

#[test]
fn permutation_rgb_to_abgr() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_3bpp(n);
            let expected = ref_rgb_to_abgr(&src);
            let mut dst = vec![0u8; n * 4];
            rgb_to_abgr(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "rgb_to_abgr n={n} tier={perm}");
        }
    });
    std::eprintln!("rgb_to_abgr: {report}");
}

#[test]
fn permutation_argb_to_rgb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_argb_to_rgb(&src);
            let mut dst = vec![0u8; n * 3];
            argb_to_rgb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "argb_to_rgb n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_to_rgb: {report}");
}

#[test]
fn permutation_argb_to_bgr() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_4bpp(n);
            let expected = ref_argb_to_bgr(&src);
            let mut dst = vec![0u8; n * 3];
            argb_to_bgr(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "argb_to_bgr n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_to_bgr: {report}");
}

#[test]
fn permutation_gray_to_argb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_1bpp(n);
            let expected = ref_gray_to_argb(&src);
            let mut dst = vec![0u8; n * 4];
            gray_to_argb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "gray_to_argb n={n} tier={perm}");
        }
    });
    std::eprintln!("gray_to_argb: {report}");
}

#[test]
fn permutation_gray_alpha_to_argb() {
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let src = make_2bpp(n);
            let expected = ref_gray_alpha_to_argb(&src);
            let mut dst = vec![0u8; n * 4];
            gray_alpha_to_argb(&src, &mut dst).unwrap();
            assert_eq!(dst, expected, "gray_alpha_to_argb n={n} tier={perm}");
        }
    });
    std::eprintln!("gray_alpha_to_argb: {report}");
}

#[test]
fn permutation_argb_roundtrip() {
    // ARGB→RGBA→ARGB roundtrip
    let report = for_each_token_permutation(policy(), |perm| {
        for &n in TEST_PIXEL_COUNTS {
            let original = make_4bpp(n);
            let mut data = original.clone();
            argb_to_rgba_inplace(&mut data).unwrap();
            rgba_to_argb_inplace(&mut data).unwrap();
            assert_eq!(data, original, "rotate roundtrip n={n} tier={perm}");
        }
    });
    std::eprintln!("argb_roundtrip: {report}");
}

#[test]
fn permutation_strided_argb() {
    let report = for_each_token_permutation(policy(), |perm| {
        let w = 10;
        let h = 4;
        let stride = 48;

        // ARGB→RGBA strided inplace
        let mut buf = vec![0xCCu8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + x * 4;
                buf[i] = (y * w + x) as u8; // A
                buf[i + 1] = 100; // R
                buf[i + 2] = 150; // G
                buf[i + 3] = 200; // B
            }
        }
        let orig = buf.clone();
        argb_to_rgba_inplace_strided(&mut buf, w, h, stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + x * 4;
                let o = &orig[i..i + 4];
                assert_eq!(
                    [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]],
                    [o[1], o[2], o[3], o[0]],
                    "strided argb_to_rgba y={y} x={x} tier={perm}"
                );
            }
            // Padding untouched
            for i in (w * 4)..stride {
                assert_eq!(
                    buf[y * stride + i],
                    0xCC,
                    "padding corrupted y={y} i={i} tier={perm}"
                );
            }
        }

        // ARGB→BGRA strided (reverse)
        let mut buf2 = orig.clone();
        argb_to_bgra_inplace_strided(&mut buf2, w, h, stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + x * 4;
                let o = &orig[i..i + 4];
                assert_eq!(
                    [buf2[i], buf2[i + 1], buf2[i + 2], buf2[i + 3]],
                    [o[3], o[2], o[1], o[0]],
                    "strided argb_to_bgra y={y} x={x} tier={perm}"
                );
            }
        }

        // fill_alpha_argb strided
        let mut buf3 = orig.clone();
        fill_alpha_argb_strided(&mut buf3, w, h, stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + x * 4;
                assert_eq!(
                    buf3[i], 255,
                    "strided fill_alpha_argb y={y} x={x} tier={perm}"
                );
                assert_eq!(buf3[i + 1], orig[i + 1]);
                assert_eq!(buf3[i + 2], orig[i + 2]);
                assert_eq!(buf3[i + 3], orig[i + 3]);
            }
        }
    });
    std::eprintln!("strided_argb: {report}");
}

#[test]
fn permutation_strided_argb_cross_bpp() {
    let report = for_each_token_permutation(policy(), |perm| {
        let w = 10;
        let h = 3;

        // RGB→ARGB strided
        let src_stride = w * 3 + 6;
        let dst_stride = w * 4 + 8;
        let src = make_3bpp(src_stride / 3 * h);
        let mut dst = vec![0xCCu8; dst_stride * h];
        rgb_to_argb_strided(&src, &mut dst, w, h, src_stride, dst_stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x * 3;
                let di = y * dst_stride + x * 4;
                assert_eq!(
                    [dst[di], dst[di + 1], dst[di + 2], dst[di + 3]],
                    [255, src[si], src[si + 1], src[si + 2]],
                    "strided rgb_to_argb y={y} x={x} tier={perm}"
                );
            }
        }

        // ARGB→RGB strided
        let src_stride4 = w * 4 + 8;
        let dst_stride3 = w * 3 + 6;
        let src4: Vec<u8> = (0..src_stride4 * h).map(|i| (i % 251) as u8).collect();
        let mut dst3 = vec![0u8; dst_stride3 * h];
        argb_to_rgb_strided(&src4, &mut dst3, w, h, src_stride4, dst_stride3).unwrap();
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride4 + x * 4;
                let di = y * dst_stride3 + x * 3;
                assert_eq!(
                    [dst3[di], dst3[di + 1], dst3[di + 2]],
                    [src4[si + 1], src4[si + 2], src4[si + 3]],
                    "strided argb_to_rgb y={y} x={x} tier={perm}"
                );
            }
        }

        // Gray→ARGB strided
        let src_stride1 = w + 4;
        let gray: Vec<u8> = (0..src_stride1 * h).map(|i| (i % 251) as u8).collect();
        let mut dst_ga = vec![0u8; dst_stride * h];
        gray_to_argb_strided(&gray, &mut dst_ga, w, h, src_stride1, dst_stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let g = gray[y * src_stride1 + x];
                let di = y * dst_stride + x * 4;
                assert_eq!(
                    [dst_ga[di], dst_ga[di + 1], dst_ga[di + 2], dst_ga[di + 3]],
                    [255, g, g, g],
                    "strided gray_to_argb y={y} x={x} tier={perm}"
                );
            }
        }

        // GrayAlpha→ARGB strided
        let src_stride2 = w * 2 + 6;
        let ga: Vec<u8> = (0..src_stride2 * h).map(|i| (i % 251) as u8).collect();
        let mut dst_gaa = vec![0u8; dst_stride * h];
        gray_alpha_to_argb_strided(&ga, &mut dst_gaa, w, h, src_stride2, dst_stride).unwrap();
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride2 + x * 2;
                let g = ga[si];
                let a = ga[si + 1];
                let di = y * dst_stride + x * 4;
                assert_eq!(
                    [
                        dst_gaa[di],
                        dst_gaa[di + 1],
                        dst_gaa[di + 2],
                        dst_gaa[di + 3]
                    ],
                    [a, g, g, g],
                    "strided gray_alpha_to_argb y={y} x={x} tier={perm}"
                );
            }
        }
    });
    std::eprintln!("strided_argb_cross_bpp: {report}");
}

// -----------------------------------------------------------------------
// Gray layout conversions (no luma weights)
// -----------------------------------------------------------------------

#[cfg(feature = "experimental")]
mod experimental_tests {
    use super::*;

    #[test]
    fn permutation_gray_to_rgb() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_1bpp(n);
                let mut dst = vec![0u8; n * 3];
                gray_to_rgb(&src, &mut dst).unwrap();
                for (i, &g) in src.iter().enumerate() {
                    let d = &dst[i * 3..i * 3 + 3];
                    assert_eq!(d, &[g, g, g], "gray_to_rgb n={n} i={i} tier={perm}");
                }
            }
        });
        std::eprintln!("gray_to_rgb: {report}");
    }

    #[test]
    fn permutation_gray_alpha_to_rgb() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_2bpp(n);
                let mut dst = vec![0u8; n * 3];
                gray_alpha_to_rgb(&src, &mut dst).unwrap();
                for i in 0..n {
                    let g = src[i * 2];
                    let d = &dst[i * 3..i * 3 + 3];
                    assert_eq!(d, &[g, g, g], "gray_alpha_to_rgb n={n} i={i} tier={perm}");
                }
            }
        });
        std::eprintln!("gray_alpha_to_rgb: {report}");
    }

    #[test]
    fn permutation_gray_to_gray_alpha() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_1bpp(n);
                let mut dst = vec![0u8; n * 2];
                gray_to_gray_alpha(&src, &mut dst).unwrap();
                for (i, &g) in src.iter().enumerate() {
                    assert_eq!(dst[i * 2], g, "gray_to_ga gray n={n} i={i} tier={perm}");
                    assert_eq!(
                        dst[i * 2 + 1],
                        255,
                        "gray_to_ga alpha n={n} i={i} tier={perm}"
                    );
                }
            }
        });
        std::eprintln!("gray_to_gray_alpha: {report}");
    }

    #[test]
    fn permutation_gray_alpha_to_gray() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_2bpp(n);
                let mut dst = vec![0u8; n];
                gray_alpha_to_gray(&src, &mut dst).unwrap();
                for i in 0..n {
                    assert_eq!(dst[i], src[i * 2], "ga_to_gray n={n} i={i} tier={perm}");
                }
            }
        });
        std::eprintln!("gray_alpha_to_gray: {report}");
    }

    #[test]
    fn identity_gray_roundtrip_rgb() {
        let src: Vec<u8> = (0..=255).collect();
        let mut rgb = vec![0u8; 256 * 3];
        let mut dst = vec![0u8; 256];
        gray_to_rgb(&src, &mut rgb).unwrap();
        rgb_to_gray_identity(&rgb, &mut dst).unwrap();
        assert_eq!(src, dst, "gray→rgb→gray_identity roundtrip failed");
    }

    #[test]
    fn identity_gray_roundtrip_rgba() {
        let src: Vec<u8> = (0..=255).collect();
        let mut rgba = vec![0u8; 256 * 4];
        let mut dst = vec![0u8; 256];
        gray_to_rgba(&src, &mut rgba).unwrap();
        rgba_to_gray_identity(&rgba, &mut dst).unwrap();
        assert_eq!(src, dst, "gray→rgba→gray_identity roundtrip failed");
    }

    #[test]
    fn permutation_rgba_to_gray_identity() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_4bpp(n);
                let mut dst = vec![0u8; n];
                rgba_to_gray_identity(&src, &mut dst).unwrap();
                for i in 0..n {
                    assert_eq!(dst[i], src[i * 4], "rgba_to_gray n={n} i={i} tier={perm}");
                }
            }
        });
        std::eprintln!("rgba_to_gray_identity: {report}");
    }

    #[test]
    fn permutation_rgb_to_gray_identity() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_3bpp(n);
                let mut dst = vec![0u8; n];
                rgb_to_gray_identity(&src, &mut dst).unwrap();
                for i in 0..n {
                    assert_eq!(dst[i], src[i * 3], "rgb_to_gray n={n} i={i} tier={perm}");
                }
            }
        });
        std::eprintln!("rgb_to_gray_identity: {report}");
    }

    // -----------------------------------------------------------------------
    // Weighted luma conversions
    // -----------------------------------------------------------------------

    fn ref_luma_3bpp(src: &[u8], w_r: u16, w_g: u16, w_b: u16) -> Vec<u8> {
        src.chunks_exact(3)
            .map(|px| {
                ((px[0] as u16 * w_r + px[1] as u16 * w_g + px[2] as u16 * w_b + 128) >> 8) as u8
            })
            .collect()
    }

    fn ref_luma_4bpp(src: &[u8], w_r: u16, w_g: u16, w_b: u16) -> Vec<u8> {
        src.chunks_exact(4)
            .map(|px| {
                ((px[0] as u16 * w_r + px[1] as u16 * w_g + px[2] as u16 * w_b + 128) >> 8) as u8
            })
            .collect()
    }

    /// Verify that for each matrix, gray→rgb→gray is identity (weights sum to 256).
    #[test]
    fn luma_roundtrip_identity() {
        let matrices: &[(
            fn(&[u8], &mut [u8]) -> Result<(), SizeError>,
            &str,
            u16,
            u16,
            u16,
        )] = &[
            (rgb_to_gray_bt709, "BT.709", 54, 183, 19),
            (rgb_to_gray_bt601, "BT.601", 77, 150, 29),
            (rgb_to_gray_bt2020, "BT.2020", 67, 174, 15),
        ];
        for &(luma_fn, name, wr, wg, wb) in matrices {
            assert_eq!(wr + wg + wb, 256, "weights for {name} don't sum to 256");
            // gray → rgb (R=G=B=v) → luma should give back v
            let src: Vec<u8> = (0..=255).collect();
            let mut rgb = vec![0u8; 256 * 3];
            gray_to_rgb(&src, &mut rgb).unwrap();
            let mut dst = vec![0u8; 256];
            luma_fn(&rgb, &mut dst).unwrap();
            assert_eq!(src, dst, "gray→rgb→gray roundtrip failed for {name}");
        }
    }

    #[test]
    fn permutation_rgba_to_gray_bt709() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_4bpp(n);
                let expected = ref_luma_4bpp(&src, 54, 183, 19);
                let mut dst = vec![0u8; n];
                rgba_to_gray_bt709(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba_to_gray_bt709 n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba_to_gray_bt709: {report}");
    }

    #[test]
    fn permutation_bgra_to_gray_bt709() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_4bpp(n);
                let expected = ref_luma_4bpp(&src, 19, 183, 54);
                let mut dst = vec![0u8; n];
                bgra_to_gray_bt709(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "bgra_to_gray_bt709 n={n} tier={perm}");
            }
        });
        std::eprintln!("bgra_to_gray_bt709: {report}");
    }

    #[test]
    fn permutation_rgb_to_gray_bt709() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_3bpp(n);
                let expected = ref_luma_3bpp(&src, 54, 183, 19);
                let mut dst = vec![0u8; n];
                rgb_to_gray_bt709(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgb_to_gray_bt709 n={n} tier={perm}");
            }
        });
        std::eprintln!("rgb_to_gray_bt709: {report}");
    }

    #[test]
    fn permutation_rgba_to_gray_bt601() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_4bpp(n);
                let expected = ref_luma_4bpp(&src, 77, 150, 29);
                let mut dst = vec![0u8; n];
                rgba_to_gray_bt601(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba_to_gray_bt601 n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba_to_gray_bt601: {report}");
    }

    #[test]
    fn permutation_rgba_to_gray_bt2020() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_PIXEL_COUNTS {
                let src = make_4bpp(n);
                let expected = ref_luma_4bpp(&src, 67, 174, 15);
                let mut dst = vec![0u8; n];
                rgba_to_gray_bt2020(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba_to_gray_bt2020 n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba_to_gray_bt2020: {report}");
    }

    #[test]
    fn luma_default_aliases() {
        let src = make_4bpp(16);
        let mut a = vec![0u8; 16];
        let mut b = vec![0u8; 16];
        rgba_to_gray(&src, &mut a).unwrap();
        rgba_to_gray_bt709(&src, &mut b).unwrap();
        assert_eq!(a, b, "rgba_to_gray should default to bt709");
    }

    // -----------------------------------------------------------------------
    // Depth conversions
    // -----------------------------------------------------------------------

    const TEST_ELEMENT_COUNTS: &[usize] =
        &[1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 255, 256];

    #[test]
    fn depth_u8_to_u16_roundtrip_exhaustive() {
        // u8 → u16 → u8 must be exact identity for all 256 values
        let src: Vec<u8> = (0..=255).collect();
        let mut mid = vec![0u8; 256 * 2];
        let mut dst = vec![0u8; 256];
        convert_u8_to_u16(&src, &mut mid).unwrap();
        convert_u16_to_u8(&mid, &mut dst).unwrap();
        assert_eq!(src, dst, "u8→u16→u8 roundtrip failed");

        // Check boundary values
        let mid16: &[u16] = bytemuck::cast_slice(&mid);
        assert_eq!(mid16[0], 0);
        assert_eq!(mid16[255], 65535);
    }

    #[test]
    fn depth_u8_to_f32_roundtrip_exhaustive() {
        // u8 → f32 → u8 must be exact identity for all 256 values
        let src: Vec<u8> = (0..=255).collect();
        let mut mid = vec![0u8; 256 * 4];
        let mut dst = vec![0u8; 256];
        convert_u8_to_f32(&src, &mut mid).unwrap();
        convert_f32_to_u8(&mid, &mut dst).unwrap();
        assert_eq!(src, dst, "u8→f32→u8 roundtrip failed");

        // Check boundary values
        let mid_f: &[f32] = bytemuck::cast_slice(&mid);
        assert_eq!(mid_f[0], 0.0);
        assert_eq!(mid_f[255], 1.0);
    }

    #[test]
    fn depth_u16_to_f32_roundtrip_exhaustive() {
        // u16 → f32 → u16 must be exact identity for all 65536 values
        let src16: Vec<u16> = (0..=65535u16).collect();
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mut mid = vec![0u8; 65536 * 4];
        let mut dst_bytes = vec![0u8; 65536 * 2];
        convert_u16_to_f32(src, &mut mid).unwrap();
        convert_f32_to_u16(&mid, &mut dst_bytes).unwrap();
        let dst16: &[u16] = bytemuck::cast_slice(&dst_bytes);
        assert_eq!(src16.as_slice(), dst16, "u16→f32→u16 roundtrip failed");

        // Check boundary values
        let mid_f: &[f32] = bytemuck::cast_slice(&mid);
        assert_eq!(mid_f[0], 0.0);
        assert_eq!(mid_f[65535], 1.0);
    }

    #[test]
    fn depth_f32_clamping() {
        // Out-of-range f32 → u8 should clamp
        let src_vals: Vec<f32> = vec![-0.5, -0.001, 0.0, 0.5, 1.0, 1.001, 1.5, 100.0];
        let src: &[u8] = bytemuck::cast_slice(&src_vals);
        let mut dst = vec![0u8; 8];
        convert_f32_to_u8(src, &mut dst).unwrap();
        assert_eq!(dst[0], 0, "negative should clamp to 0");
        assert_eq!(dst[1], 0, "small negative should clamp to 0");
        assert_eq!(dst[2], 0, "zero should stay 0");
        assert_eq!(dst[3], 128, "0.5 should be 128");
        assert_eq!(dst[4], 255, "1.0 should be 255");
        assert_eq!(dst[5], 255, "slightly over 1.0 should clamp to 255");
        assert_eq!(dst[6], 255, "1.5 should clamp to 255");
        assert_eq!(dst[7], 255, "100.0 should clamp to 255");

        // Out-of-range f32 → u16 should clamp
        let mut dst16_bytes = vec![0u8; 16];
        convert_f32_to_u16(src, &mut dst16_bytes).unwrap();
        let dst16: &[u16] = bytemuck::cast_slice(&dst16_bytes);
        assert_eq!(dst16[0], 0);
        assert_eq!(dst16[4], 65535);
        assert_eq!(dst16[7], 65535);
    }

    #[test]
    fn permutation_depth_u8_u16() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_ELEMENT_COUNTS {
                let src: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
                let mut mid = vec![0u8; n * 2];
                convert_u8_to_u16(&src, &mut mid).unwrap();
                // Check each value
                let mid16: &[u16] = bytemuck::cast_slice(&mid);
                for (j, &s) in src.iter().enumerate() {
                    assert_eq!(mid16[j], s as u16 * 257, "u8→u16 n={n} i={j} tier={perm}");
                }
                // Roundtrip
                let mut dst = vec![0u8; n];
                convert_u16_to_u8(&mid, &mut dst).unwrap();
                assert_eq!(src, dst, "u8→u16→u8 roundtrip n={n} tier={perm}");
            }
        });
        std::eprintln!("depth_u8_u16: {report}");
    }

    #[test]
    fn permutation_depth_u8_f32() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_ELEMENT_COUNTS {
                let src: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
                let mut mid = vec![0u8; n * 4];
                convert_u8_to_f32(&src, &mut mid).unwrap();
                let mut dst = vec![0u8; n];
                convert_f32_to_u8(&mid, &mut dst).unwrap();
                assert_eq!(src, dst, "u8→f32→u8 roundtrip n={n} tier={perm}");
            }
        });
        std::eprintln!("depth_u8_f32: {report}");
    }

    #[test]
    fn permutation_depth_u16_f32() {
        let report = for_each_token_permutation(policy(), |perm| {
            for &n in TEST_ELEMENT_COUNTS {
                let src16: Vec<u16> = (0..n).map(|i| (i * 257) as u16).collect();
                let src: Vec<u8> = bytemuck::cast_slice(&src16).to_vec();
                let mut mid = vec![0u8; n * 4];
                convert_u16_to_f32(&src, &mut mid).unwrap();
                let mut dst = vec![0u8; n * 2];
                convert_f32_to_u16(&mid, &mut dst).unwrap();
                let dst16: &[u16] = bytemuck::cast_slice(&dst);
                assert_eq!(
                    src16.as_slice(),
                    dst16,
                    "u16→f32→u16 roundtrip n={n} tier={perm}"
                );
            }
        });
        std::eprintln!("depth_u16_f32: {report}");
    }

    #[test]
    fn depth_size_errors() {
        // Empty source
        assert_eq!(
            convert_u8_to_u16(&[], &mut [0; 2]),
            Err(SizeError::NotPixelAligned)
        );
        // Source not aligned (u16 needs even bytes)
        assert_eq!(
            convert_u16_to_u8(&[0; 3], &mut [0; 2]),
            Err(SizeError::NotPixelAligned)
        );
        // Source not aligned (f32 needs 4-byte multiple)
        assert_eq!(
            convert_f32_to_u8(&[0; 5], &mut [0; 2]),
            Err(SizeError::NotPixelAligned)
        );
        // Dest too small
        assert_eq!(
            convert_u8_to_u16(&[0; 4], &mut [0; 6]),
            Err(SizeError::PixelCountMismatch)
        );
        assert_eq!(
            convert_u8_to_f32(&[0; 4], &mut [0; 12]),
            Err(SizeError::PixelCountMismatch)
        );
    }
} // mod experimental_tests

// -----------------------------------------------------------------------
// Size validation
// -----------------------------------------------------------------------

#[test]
fn test_size_errors() {
    // Not pixel-aligned
    assert_eq!(
        rgba_to_bgra_inplace(&mut [0; 5]),
        Err(SizeError::NotPixelAligned)
    );
    assert_eq!(
        rgba_to_bgra_inplace(&mut [0; 0]),
        Err(SizeError::NotPixelAligned)
    );
    assert_eq!(
        rgb_to_bgr_inplace(&mut [0; 5]),
        Err(SizeError::NotPixelAligned)
    );
    assert_eq!(
        gray_alpha_to_rgba(&[0; 3], &mut [0; 8]),
        Err(SizeError::NotPixelAligned)
    );
    assert_eq!(
        fill_alpha_rgba(&mut [0; 5]),
        Err(SizeError::NotPixelAligned)
    );

    // Pixel count mismatch (src aligned, dst too small)
    assert_eq!(
        rgb_to_bgra(&[0; 6], &mut [0; 4]),
        Err(SizeError::PixelCountMismatch)
    );
    assert_eq!(
        gray_to_rgba(&[0; 3], &mut [0; 8]),
        Err(SizeError::PixelCountMismatch)
    );
    assert_eq!(
        rgba_to_rgb(&[0; 8], &mut [0; 3]),
        Err(SizeError::PixelCountMismatch)
    );
}

#[test]
fn test_strided_size_errors() {
    // stride < width * bpp
    assert_eq!(
        rgba_to_bgra_inplace_strided(&mut [0; 32], 2, 2, 4),
        Err(SizeError::InvalidStride)
    );
    // buffer too small
    assert_eq!(
        rgba_to_bgra_inplace_strided(&mut [0; 10], 2, 2, 8),
        Err(SizeError::InvalidStride)
    );
    // zero width
    assert_eq!(
        rgba_to_bgra_inplace_strided(&mut [0; 8], 0, 1, 8),
        Err(SizeError::InvalidStride)
    );
    // zero height
    assert_eq!(
        rgba_to_bgra_inplace_strided(&mut [0; 8], 2, 0, 8),
        Err(SizeError::InvalidStride)
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

    // ARGB aliases
    // bgra_to_argb = argb_to_bgra (reverse is symmetric)
    let mut a2 = data.clone();
    let mut b2 = data.clone();
    argb_to_bgra_inplace(&mut a2).unwrap();
    bgra_to_argb_inplace(&mut b2).unwrap();
    assert_eq!(a2, b2, "argb_to_bgra = bgra_to_argb");

    // abgr_to_bgra = argb_to_rgba (same rotate left)
    let mut c2 = data.clone();
    let mut d2 = data.clone();
    argb_to_rgba_inplace(&mut c2).unwrap();
    abgr_to_bgra_inplace(&mut d2).unwrap();
    assert_eq!(c2, d2, "argb_to_rgba = abgr_to_bgra");

    // abgr_to_rgba = argb_to_bgra (same reverse)
    let mut e2 = data.clone();
    let mut f2 = data.clone();
    argb_to_bgra_inplace(&mut e2).unwrap();
    abgr_to_rgba_inplace(&mut f2).unwrap();
    assert_eq!(e2, f2, "argb_to_bgra = abgr_to_rgba");

    // rgba_to_abgr = argb_to_bgra (same reverse)
    let mut g2 = data.clone();
    let mut h2 = data.clone();
    argb_to_bgra_inplace(&mut g2).unwrap();
    rgba_to_abgr_inplace(&mut h2).unwrap();
    assert_eq!(g2, h2, "argb_to_bgra = rgba_to_abgr");

    // bgr_to_argb = rgb_to_abgr
    let mut dst_g = vec![0u8; 64];
    let mut dst_h = vec![0u8; 64];
    bgr_to_argb(&src3, &mut dst_g).unwrap();
    rgb_to_abgr(&src3, &mut dst_h).unwrap();
    assert_eq!(dst_g, dst_h, "bgr_to_argb = rgb_to_abgr");

    // bgr_to_abgr = rgb_to_argb
    let mut dst_i = vec![0u8; 64];
    let mut dst_j = vec![0u8; 64];
    bgr_to_abgr(&src3, &mut dst_i).unwrap();
    rgb_to_argb(&src3, &mut dst_j).unwrap();
    assert_eq!(dst_i, dst_j, "bgr_to_abgr = rgb_to_argb");

    // abgr_to_bgr = argb_to_rgb
    let mut dst_k = vec![0u8; 48];
    let mut dst_l = vec![0u8; 48];
    abgr_to_bgr(&data, &mut dst_k).unwrap();
    argb_to_rgb(&data, &mut dst_l).unwrap();
    assert_eq!(dst_k, dst_l, "abgr_to_bgr = argb_to_rgb");

    // abgr_to_rgb = argb_to_bgr
    let mut dst_m = vec![0u8; 48];
    let mut dst_n = vec![0u8; 48];
    abgr_to_rgb(&data, &mut dst_m).unwrap();
    argb_to_bgr(&data, &mut dst_n).unwrap();
    assert_eq!(dst_m, dst_n, "abgr_to_rgb = argb_to_bgr");

    // gray_to_abgr = gray_to_argb
    let mut dst_o = vec![0u8; 64];
    let mut dst_p = vec![0u8; 64];
    gray_to_abgr(&gray, &mut dst_o).unwrap();
    gray_to_argb(&gray, &mut dst_p).unwrap();
    assert_eq!(dst_o, dst_p, "gray_to_abgr = gray_to_argb");

    // fill_alpha_xrgb = fill_alpha_argb = fill_alpha_abgr = fill_alpha_xbgr
    let mut fa1 = data.clone();
    let mut fa2 = data.clone();
    let mut fa3 = data.clone();
    let mut fa4 = data.clone();
    fill_alpha_argb(&mut fa1).unwrap();
    fill_alpha_xrgb(&mut fa2).unwrap();
    fill_alpha_abgr(&mut fa3).unwrap();
    fill_alpha_xbgr(&mut fa4).unwrap();
    assert_eq!(fa1, fa2, "fill_alpha_argb = fill_alpha_xrgb");
    assert_eq!(fa1, fa3, "fill_alpha_argb = fill_alpha_abgr");
    assert_eq!(fa1, fa4, "fill_alpha_argb = fill_alpha_xbgr");
}

// -----------------------------------------------------------------------
// f32 alpha premultiplication
// -----------------------------------------------------------------------

#[cfg(feature = "experimental")]
mod experimental_premul_tests {
    use super::*;

    /// Build f32 RGBA test pixels: R=r, G=g, B=b, A=a packed as bytes.
    fn make_f32_rgba(pixels: &[[f32; 4]]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(pixels.len() * 16);
        for px in pixels {
            for &v in px {
                buf.extend_from_slice(&v.to_ne_bytes());
            }
        }
        buf
    }

    /// Read f32 RGBA pixels back from byte buffer.
    fn read_f32_rgba(buf: &[u8]) -> Vec<[f32; 4]> {
        let floats: &[f32] = bytemuck::cast_slice(buf);
        floats
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }

    #[test]
    fn premul_f32_known_values() {
        // [0.5, 0.3, 0.1, 0.5] → premul → [0.25, 0.15, 0.05, 0.5]
        let mut buf = make_f32_rgba(&[[0.5, 0.3, 0.1, 0.5]]);
        premultiply_alpha_f32(&mut buf).unwrap();
        let result = read_f32_rgba(&buf);
        assert!((result[0][0] - 0.25).abs() < 1e-6, "R premul");
        assert!((result[0][1] - 0.15).abs() < 1e-6, "G premul");
        assert!((result[0][2] - 0.05).abs() < 1e-6, "B premul");
        assert_eq!(result[0][3], 0.5, "A preserved");

        // Full alpha: no change
        let mut buf2 = make_f32_rgba(&[[0.8, 0.6, 0.4, 1.0]]);
        premultiply_alpha_f32(&mut buf2).unwrap();
        let r2 = read_f32_rgba(&buf2);
        assert!((r2[0][0] - 0.8).abs() < 1e-6);
        assert!((r2[0][1] - 0.6).abs() < 1e-6);
        assert!((r2[0][2] - 0.4).abs() < 1e-6);

        // Zero alpha: all zero
        let mut buf3 = make_f32_rgba(&[[0.5, 0.3, 0.1, 0.0]]);
        premultiply_alpha_f32(&mut buf3).unwrap();
        let r3 = read_f32_rgba(&buf3);
        assert_eq!(r3[0], [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn premul_unpremul_f32_roundtrip() {
        let original = vec![
            [0.8, 0.6, 0.2, 0.7],
            [1.0, 0.0, 0.5, 0.3],
            [0.0, 1.0, 1.0, 1.0],
            [0.1, 0.2, 0.3, 0.0], // zero alpha
        ];
        let mut buf = make_f32_rgba(&original);
        premultiply_alpha_f32(&mut buf).unwrap();
        unpremultiply_alpha_f32(&mut buf).unwrap();
        let result = read_f32_rgba(&buf);
        for (i, (r, o)) in result.iter().zip(original.iter()).enumerate() {
            if o[3] == 0.0 {
                // Zero alpha: all channels become 0
                assert_eq!(*r, [0.0, 0.0, 0.0, 0.0], "zero-alpha pixel {i}");
            } else {
                for c in 0..4 {
                    assert!(
                        (r[c] - o[c]).abs() < 1e-6,
                        "pixel {i} channel {c}: {:.8} != {:.8}",
                        r[c],
                        o[c]
                    );
                }
            }
        }
    }

    #[test]
    fn premul_unpremul_f32_copy_roundtrip() {
        let original = vec![
            [0.5, 0.3, 0.1, 0.5],
            [1.0, 1.0, 1.0, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let src = make_f32_rgba(&original);
        let mut mid = vec![0u8; src.len()];
        let mut dst = vec![0u8; src.len()];
        premultiply_alpha_f32_copy(&src, &mut mid).unwrap();
        unpremultiply_alpha_f32_copy(&mid, &mut dst).unwrap();
        let result = read_f32_rgba(&dst);
        for (i, (r, o)) in result.iter().zip(original.iter()).enumerate() {
            if o[3] == 0.0 {
                assert_eq!(*r, [0.0, 0.0, 0.0, 0.0], "zero-alpha pixel {i}");
            } else {
                for c in 0..4 {
                    assert!(
                        (r[c] - o[c]).abs() < 1e-6,
                        "copy pixel {i} channel {c}: {:.8} != {:.8}",
                        r[c],
                        o[c]
                    );
                }
            }
        }
    }

    #[test]
    fn premul_f32_size_errors() {
        // Not 16-byte aligned
        assert_eq!(
            premultiply_alpha_f32(&mut [0; 15]),
            Err(SizeError::NotPixelAligned)
        );
        assert_eq!(
            premultiply_alpha_f32(&mut [0; 0]),
            Err(SizeError::NotPixelAligned)
        );
        // Copy: dst too small
        assert_eq!(
            premultiply_alpha_f32_copy(&[0; 32], &mut [0; 16]),
            Err(SizeError::PixelCountMismatch)
        );
    }

    #[test]
    fn permutation_premul_f32() {
        let report = for_each_token_permutation(policy(), |perm| {
            // Test with various pixel counts to exercise SIMD and scalar paths
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
                let pixels: Vec<[f32; 4]> = (0..n)
                    .map(|i| {
                        let t = i as f32 / n.max(1) as f32;
                        [t, 1.0 - t, t * 0.5, (i % 3) as f32 * 0.5]
                    })
                    .collect();

                // In-place premul
                let mut buf = make_f32_rgba(&pixels);
                premultiply_alpha_f32(&mut buf).unwrap();
                let premul_result = read_f32_rgba(&buf);
                for (i, (r, p)) in premul_result.iter().zip(pixels.iter()).enumerate() {
                    let expected_r = p[0] * p[3];
                    let expected_g = p[1] * p[3];
                    let expected_b = p[2] * p[3];
                    assert!(
                        (r[0] - expected_r).abs() < 1e-6
                            && (r[1] - expected_g).abs() < 1e-6
                            && (r[2] - expected_b).abs() < 1e-6
                            && r[3] == p[3],
                        "premul n={n} i={i} tier={perm}: got {:?}, expected [{expected_r}, {expected_g}, {expected_b}, {}]",
                        r,
                        p[3]
                    );
                }

                // In-place unpremul (roundtrip)
                unpremultiply_alpha_f32(&mut buf).unwrap();
                let result = read_f32_rgba(&buf);
                for (i, (r, p)) in result.iter().zip(pixels.iter()).enumerate() {
                    if p[3] == 0.0 {
                        assert_eq!(
                            *r,
                            [0.0, 0.0, 0.0, 0.0],
                            "zero-alpha n={n} i={i} tier={perm}"
                        );
                    } else {
                        for c in 0..4 {
                            assert!(
                                (r[c] - p[c]).abs() < 1e-6,
                                "roundtrip n={n} i={i} c={c} tier={perm}: {:.8} != {:.8}",
                                r[c],
                                p[c]
                            );
                        }
                    }
                }

                // Copy premul
                let src = make_f32_rgba(&pixels);
                let mut dst = vec![0u8; src.len()];
                premultiply_alpha_f32_copy(&src, &mut dst).unwrap();
                let copy_result = read_f32_rgba(&dst);
                assert_eq!(
                    copy_result, premul_result,
                    "copy vs inplace n={n} tier={perm}"
                );

                // Copy unpremul roundtrip
                let mut dst2 = vec![0u8; src.len()];
                unpremultiply_alpha_f32_copy(&dst, &mut dst2).unwrap();
                let copy_rt = read_f32_rgba(&dst2);
                assert_eq!(copy_rt, result, "copy roundtrip n={n} tier={perm}");
            }
        });
        std::eprintln!("premul_f32: {report}");
    }

    #[test]
    fn premul_f32_aliases() {
        let mut a = make_f32_rgba(&[[0.5, 0.3, 0.1, 0.5]]);
        let mut b = a.clone();
        premultiply_alpha_f32(&mut a).unwrap();
        premultiply_alpha_rgba_f32(&mut b).unwrap();
        assert_eq!(a, b, "rgba alias should match");

        unpremultiply_alpha_f32(&mut a).unwrap();
        unpremultiply_alpha_rgba_f32(&mut b).unwrap();
        assert_eq!(a, b, "rgba unpremul alias should match");
    }
} // mod experimental_premul_tests

#[cfg(feature = "experimental")]
mod premul_u8_tests {
    use super::*;

    /// Reference: exact div-by-255 via integer math.
    fn ref_premul(c: u8, a: u8) -> u8 {
        ((c as u32 * a as u32 + 127) / 255) as u8
    }

    #[test]
    fn premul_u8_known_values() {
        // Full alpha: no change
        let mut buf = vec![128, 64, 32, 255];
        premultiply_alpha_rgba_u8(&mut buf).unwrap();
        assert_eq!(buf, [128, 64, 32, 255]);

        // Zero alpha: all channels become 0
        let mut buf = vec![255, 128, 64, 0];
        premultiply_alpha_rgba_u8(&mut buf).unwrap();
        assert_eq!(buf, [0, 0, 0, 0]);

        // Half alpha
        let mut buf = vec![200, 100, 50, 128];
        premultiply_alpha_rgba_u8(&mut buf).unwrap();
        // 200*128/255 ≈ 100.4 → 100, 100*128/255 ≈ 50.2 → 50, 50*128/255 ≈ 25.1 → 25
        assert_eq!(buf[0], ref_premul(200, 128));
        assert_eq!(buf[1], ref_premul(100, 128));
        assert_eq!(buf[2], ref_premul(50, 128));
        assert_eq!(buf[3], 128);
    }

    #[test]
    fn premul_u8_copy_matches_inplace() {
        let src: Vec<u8> = (0..256).map(|i| (i % 251) as u8).collect();
        let mut inplace = src.clone();
        premultiply_alpha_rgba_u8(&mut inplace).unwrap();
        let mut copy_dst = vec![0u8; src.len()];
        premultiply_alpha_rgba_u8_copy(&src, &mut copy_dst).unwrap();
        assert_eq!(inplace, copy_dst);
    }

    #[test]
    fn premul_u8_exhaustive_channel() {
        // Test all (channel, alpha) pairs for a single channel
        for a in 0..=255u8 {
            for c in 0..=255u8 {
                let mut buf = [c, 0, 0, a];
                premultiply_alpha_rgba_u8(&mut buf).unwrap();
                let expected = ref_premul(c, a);
                assert_eq!(
                    buf[0], expected,
                    "premul({c}, {a}): got {}, expected {expected}",
                    buf[0]
                );
                // Alpha preserved
                assert_eq!(buf[3], a);
                // Result must be ≤ alpha (premul invariant)
                assert!(buf[0] <= a, "premul({c}, {a}) = {} > alpha {a}", buf[0]);
            }
        }
    }

    #[test]
    fn premul_u8_idempotent_at_extremes() {
        // Already premultiplied: premul again should not change if c <= a
        for a in 0..=255u8 {
            let mut buf = [a, a, a, a]; // max premul value
            premultiply_alpha_rgba_u8(&mut buf).unwrap();
            // premul(a, a) = a*a/255, which is ≤ a
            assert!(buf[0] <= a);
            assert_eq!(buf[3], a);
        }
    }

    #[test]
    fn premul_u8_aliases() {
        let mut a = vec![200u8, 100, 50, 128, 255, 0, 128, 64];
        let mut b = a.clone();
        let mut c = a.clone();
        premultiply_alpha_rgba_u8(&mut a).unwrap();
        premultiply_alpha_rgba_u8(&mut b).unwrap();
        premultiply_alpha_bgra_u8(&mut c).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn permutation_premul_u8() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
                let src = make_4bpp(n);

                // In-place
                let mut buf = src.clone();
                premultiply_alpha_rgba_u8(&mut buf).unwrap();
                for (i, (s, d)) in src.chunks_exact(4).zip(buf.chunks_exact(4)).enumerate() {
                    let a = s[3];
                    assert_eq!(d[0], ref_premul(s[0], a), "n={n} i={i} R tier={perm}");
                    assert_eq!(d[1], ref_premul(s[1], a), "n={n} i={i} G tier={perm}");
                    assert_eq!(d[2], ref_premul(s[2], a), "n={n} i={i} B tier={perm}");
                    assert_eq!(d[3], a, "n={n} i={i} A tier={perm}");
                }

                // Copy
                let mut dst = vec![0u8; src.len()];
                premultiply_alpha_rgba_u8_copy(&src, &mut dst).unwrap();
                assert_eq!(dst, buf, "copy vs inplace n={n} tier={perm}");
            }
        });
        std::eprintln!("premul_u8: {report}");
    }

    #[test]
    fn premul_u8_size_errors() {
        assert_eq!(
            premultiply_alpha_rgba_u8(&mut [0; 3]),
            Err(SizeError::NotPixelAligned)
        );
        assert_eq!(
            premultiply_alpha_rgba_u8(&mut []),
            Err(SizeError::NotPixelAligned)
        );
        assert_eq!(
            premultiply_alpha_rgba_u8_copy(&[0; 8], &mut [0; 4]),
            Err(SizeError::PixelCountMismatch)
        );
    }
}

#[cfg(feature = "experimental")]
mod packed_format_tests {
    use super::*;

    /// Encode a single RGB565 pixel as little-endian bytes.
    fn encode_rgb565(r5: u16, g6: u16, b5: u16) -> [u8; 2] {
        let v = (r5 << 11) | (g6 << 5) | b5;
        v.to_le_bytes()
    }

    /// Encode a single RGBA4444 pixel as little-endian bytes.
    fn encode_rgba4444(r4: u16, g4: u16, b4: u16, a4: u16) -> [u8; 2] {
        let v = (r4 << 12) | (g4 << 8) | (b4 << 4) | a4;
        v.to_le_bytes()
    }

    /// Expand 5-bit value to 8-bit using MSB replication.
    fn expand5(v: u8) -> u8 {
        (v << 3) | (v >> 2)
    }
    /// Expand 6-bit value to 8-bit using MSB replication.
    fn expand6(v: u8) -> u8 {
        (v << 2) | (v >> 4)
    }
    /// Expand 4-bit value to 8-bit using nibble duplication.
    fn expand4(v: u8) -> u8 {
        (v << 4) | v
    }

    /// Reference: RGB565 (LE) → RGBA.
    fn ref_rgb565_to_rgba(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 2;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r5 = ((v >> 11) & 0x1F) as u8;
            let g6 = ((v >> 5) & 0x3F) as u8;
            let b5 = (v & 0x1F) as u8;
            d[0] = expand5(r5);
            d[1] = expand6(g6);
            d[2] = expand5(b5);
            d[3] = 0xFF;
        }
        out
    }

    /// Reference: RGB565 (LE) → BGRA.
    fn ref_rgb565_to_bgra(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 2;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let r5 = ((v >> 11) & 0x1F) as u8;
            let g6 = ((v >> 5) & 0x3F) as u8;
            let b5 = (v & 0x1F) as u8;
            d[0] = expand5(b5);
            d[1] = expand6(g6);
            d[2] = expand5(r5);
            d[3] = 0xFF;
        }
        out
    }

    /// Reference: RGBA4444 (LE) → RGBA.
    fn ref_rgba4444_to_rgba(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 2;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            d[0] = expand4(((v >> 12) & 0xF) as u8);
            d[1] = expand4(((v >> 8) & 0xF) as u8);
            d[2] = expand4(((v >> 4) & 0xF) as u8);
            d[3] = expand4((v & 0xF) as u8);
        }
        out
    }

    /// Reference: RGBA4444 (LE) → BGRA.
    fn ref_rgba4444_to_bgra(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 2;
        let mut out = vec![0u8; n * 4];
        for (s, d) in src.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            d[0] = expand4(((v >> 4) & 0xF) as u8); // B
            d[1] = expand4(((v >> 8) & 0xF) as u8); // G
            d[2] = expand4(((v >> 12) & 0xF) as u8); // R
            d[3] = expand4((v & 0xF) as u8); // A
        }
        out
    }

    #[test]
    fn rgb565_known_values() {
        // All zeros → black, alpha 255
        let src = encode_rgb565(0, 0, 0);
        let mut dst = [0u8; 4];
        rgb565_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 0, 255]);

        // All ones → white, alpha 255
        let src = encode_rgb565(31, 63, 31);
        rgb565_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [255, 255, 255, 255]);

        // Pure red (5-bit max)
        let src = encode_rgb565(31, 0, 0);
        rgb565_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [255, 0, 0, 255]);

        // Pure green (6-bit max)
        let src = encode_rgb565(0, 63, 0);
        rgb565_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [0, 255, 0, 255]);

        // Pure blue (5-bit max)
        let src = encode_rgb565(0, 0, 31);
        rgb565_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 255, 255]);

        // Mid-range: R=16, G=32, B=16
        let src = encode_rgb565(16, 32, 16);
        rgb565_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [expand5(16), expand6(32), expand5(16), 255]);
    }

    #[test]
    fn rgba4444_known_values() {
        // All zeros
        let src = encode_rgba4444(0, 0, 0, 0);
        let mut dst = [0u8; 4];
        rgba4444_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 0, 0]);

        // All max
        let src = encode_rgba4444(15, 15, 15, 15);
        rgba4444_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [255, 255, 255, 255]);

        // Pure red, full alpha
        let src = encode_rgba4444(15, 0, 0, 15);
        rgba4444_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [255, 0, 0, 255]);

        // Half alpha
        let src = encode_rgba4444(8, 4, 2, 7);
        rgba4444_to_rgba(&src, &mut dst).unwrap();
        assert_eq!(dst, [expand4(8), expand4(4), expand4(2), expand4(7)]);
    }

    #[test]
    fn rgb565_bgra_known_values() {
        // White → BGRA [255, 255, 255, 255]
        let src = encode_rgb565(31, 63, 31);
        let mut dst = [0u8; 4];
        rgb565_to_bgra(&src, &mut dst).unwrap();
        assert_eq!(dst, [255, 255, 255, 255]);

        // Pure red → BGRA [0, 0, 255, 255]
        let src = encode_rgb565(31, 0, 0);
        rgb565_to_bgra(&src, &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 255, 255]);
    }

    #[test]
    fn permutation_rgb565_to_rgba() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_2bpp(n);
                let expected = ref_rgb565_to_rgba(&src);
                let mut dst = vec![0u8; n * 4];
                rgb565_to_rgba(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgb565_to_rgba n={n} tier={perm}");
            }
        });
        std::eprintln!("rgb565_to_rgba: {report}");
    }

    #[test]
    fn permutation_rgb565_to_bgra() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_2bpp(n);
                let expected = ref_rgb565_to_bgra(&src);
                let mut dst = vec![0u8; n * 4];
                rgb565_to_bgra(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgb565_to_bgra n={n} tier={perm}");
            }
        });
        std::eprintln!("rgb565_to_bgra: {report}");
    }

    #[test]
    fn permutation_rgba4444_to_rgba() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_2bpp(n);
                let expected = ref_rgba4444_to_rgba(&src);
                let mut dst = vec![0u8; n * 4];
                rgba4444_to_rgba(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba4444_to_rgba n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba4444_to_rgba: {report}");
    }

    #[test]
    fn permutation_rgba4444_to_bgra() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_2bpp(n);
                let expected = ref_rgba4444_to_bgra(&src);
                let mut dst = vec![0u8; n * 4];
                rgba4444_to_bgra(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba4444_to_bgra n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba4444_to_bgra: {report}");
    }

    #[test]
    fn packed_size_errors() {
        // Empty source
        assert_eq!(
            rgb565_to_rgba(&[], &mut [0; 4]),
            Err(SizeError::NotPixelAligned)
        );
        // Odd byte count (not pixel-aligned)
        assert_eq!(
            rgb565_to_rgba(&[0; 3], &mut [0; 8]),
            Err(SizeError::NotPixelAligned)
        );
        // Dst too small
        assert_eq!(
            rgb565_to_rgba(&[0; 4], &mut [0; 4]),
            Err(SizeError::PixelCountMismatch)
        );
        // Same for RGBA4444
        assert_eq!(
            rgba4444_to_rgba(&[0; 1], &mut [0; 4]),
            Err(SizeError::NotPixelAligned)
        );
        assert_eq!(
            rgba4444_to_rgba(&[0; 4], &mut [0; 4]),
            Err(SizeError::PixelCountMismatch)
        );
    }

    #[test]
    fn packed_strided_basic() {
        // 2×2 image with stride > width
        let w = 2;
        let h = 2;
        let src_stride = w * 2 + 4; // 4 bytes padding per row
        let dst_stride = w * 4 + 8; // 8 bytes padding per row

        // Build source: 2 pixels per row + padding
        let mut src = vec![0u8; (h - 1) * src_stride + w * 2];
        // Row 0: two known pixels
        src[0..2].copy_from_slice(&encode_rgb565(31, 0, 0)); // red
        src[2..4].copy_from_slice(&encode_rgb565(0, 63, 0)); // green
        // Row 1: two known pixels
        src[src_stride..src_stride + 2].copy_from_slice(&encode_rgb565(0, 0, 31)); // blue
        src[src_stride + 2..src_stride + 4].copy_from_slice(&encode_rgb565(31, 63, 31)); // white

        let mut dst = vec![0u8; (h - 1) * dst_stride + w * 4];
        rgb565_to_rgba_strided(&src, &mut dst, w, h, src_stride, dst_stride).unwrap();

        // Check row 0
        assert_eq!(&dst[0..4], &[255, 0, 0, 255]); // red
        assert_eq!(&dst[4..8], &[0, 255, 0, 255]); // green
        // Check row 1
        assert_eq!(&dst[dst_stride..dst_stride + 4], &[0, 0, 255, 255]); // blue
        assert_eq!(&dst[dst_stride + 4..dst_stride + 8], &[255, 255, 255, 255]); // white
    }

    #[test]
    fn rgb565_expansion_boundaries() {
        // Verify every 5-bit value expands correctly (0→0, 31→255)
        for v in 0..32u16 {
            let src = encode_rgb565(v, 0, 0);
            let mut dst = [0u8; 4];
            rgb565_to_rgba(&src, &mut dst).unwrap();
            let expected = ((v << 3) | (v >> 2)) as u8;
            assert_eq!(dst[0], expected, "5-bit expand({v})");
        }
        // Verify every 6-bit value expands correctly (0→0, 63→255)
        for v in 0..64u16 {
            let src = encode_rgb565(0, v, 0);
            let mut dst = [0u8; 4];
            rgb565_to_rgba(&src, &mut dst).unwrap();
            let expected = ((v << 2) | (v >> 4)) as u8;
            assert_eq!(dst[1], expected, "6-bit expand({v})");
        }
    }

    #[test]
    fn rgba4444_expansion_boundaries() {
        // Verify every 4-bit value expands correctly (0→0, 15→255)
        for v in 0..16u16 {
            let src = encode_rgba4444(v, 0, 0, 0);
            let mut dst = [0u8; 4];
            rgba4444_to_rgba(&src, &mut dst).unwrap();
            let expected = ((v << 4) | v) as u8;
            assert_eq!(dst[0], expected, "4-bit expand({v})");
        }
    }

    #[test]
    fn rgb565_to_rgba_exhaustive() {
        // Test ALL 65536 possible RGB565 values
        let mut src = vec![0u8; 65536 * 2];
        for i in 0u32..65536 {
            let le = (i as u16).to_le_bytes();
            src[i as usize * 2] = le[0];
            src[i as usize * 2 + 1] = le[1];
        }
        let mut dst = vec![0u8; 65536 * 4];
        rgb565_to_rgba(&src, &mut dst).unwrap();
        for i in 0u32..65536 {
            let v = i as u16;
            let r5 = ((v >> 11) & 0x1F) as u8;
            let g6 = ((v >> 5) & 0x3F) as u8;
            let b5 = (v & 0x1F) as u8;
            let expected = [expand5(r5), expand6(g6), expand5(b5), 255];
            let got = &dst[i as usize * 4..i as usize * 4 + 4];
            assert_eq!(got, &expected, "RGB565→RGBA mismatch at 0x{i:04X}");
        }
    }

    #[test]
    fn rgb565_to_bgra_exhaustive() {
        let mut src = vec![0u8; 65536 * 2];
        for i in 0u32..65536 {
            let le = (i as u16).to_le_bytes();
            src[i as usize * 2] = le[0];
            src[i as usize * 2 + 1] = le[1];
        }
        let mut dst = vec![0u8; 65536 * 4];
        rgb565_to_bgra(&src, &mut dst).unwrap();
        for i in 0u32..65536 {
            let v = i as u16;
            let r5 = ((v >> 11) & 0x1F) as u8;
            let g6 = ((v >> 5) & 0x3F) as u8;
            let b5 = (v & 0x1F) as u8;
            let expected = [expand5(b5), expand6(g6), expand5(r5), 255];
            let got = &dst[i as usize * 4..i as usize * 4 + 4];
            assert_eq!(got, &expected, "RGB565→BGRA mismatch at 0x{i:04X}");
        }
    }

    #[test]
    fn rgba4444_to_rgba_exhaustive() {
        let mut src = vec![0u8; 65536 * 2];
        for i in 0u32..65536 {
            let le = (i as u16).to_le_bytes();
            src[i as usize * 2] = le[0];
            src[i as usize * 2 + 1] = le[1];
        }
        let mut dst = vec![0u8; 65536 * 4];
        rgba4444_to_rgba(&src, &mut dst).unwrap();
        for i in 0u32..65536 {
            let v = i as u16;
            let r4 = ((v >> 12) & 0xF) as u8;
            let g4 = ((v >> 8) & 0xF) as u8;
            let b4 = ((v >> 4) & 0xF) as u8;
            let a4 = (v & 0xF) as u8;
            let expected = [expand4(r4), expand4(g4), expand4(b4), expand4(a4)];
            let got = &dst[i as usize * 4..i as usize * 4 + 4];
            assert_eq!(got, &expected, "RGBA4444→RGBA mismatch at 0x{i:04X}");
        }
    }

    #[test]
    fn rgba4444_to_bgra_exhaustive() {
        let mut src = vec![0u8; 65536 * 2];
        for i in 0u32..65536 {
            let le = (i as u16).to_le_bytes();
            src[i as usize * 2] = le[0];
            src[i as usize * 2 + 1] = le[1];
        }
        let mut dst = vec![0u8; 65536 * 4];
        rgba4444_to_bgra(&src, &mut dst).unwrap();
        for i in 0u32..65536 {
            let v = i as u16;
            let r4 = ((v >> 12) & 0xF) as u8;
            let g4 = ((v >> 8) & 0xF) as u8;
            let b4 = ((v >> 4) & 0xF) as u8;
            let a4 = (v & 0xF) as u8;
            let expected = [expand4(b4), expand4(g4), expand4(r4), expand4(a4)];
            let got = &dst[i as usize * 4..i as usize * 4 + 4];
            assert_eq!(got, &expected, "RGBA4444→BGRA mismatch at 0x{i:04X}");
        }
    }

    #[test]
    fn strided_padding_untouched() {
        for w in [1, 3, 7, 16, 31, 64] {
            let h = 4;
            let src_stride = w * 2 + 6;
            let dst_stride = w * 4 + 12;
            let src_total = (h - 1) * src_stride + w * 2;
            let dst_total = (h - 1) * dst_stride + w * 4;
            let mut src = vec![0xDDu8; src_total];
            for y in 0..h {
                for x in 0..w * 2 {
                    src[y * src_stride + x] = ((y * w * 2 + x) % 251) as u8;
                }
            }
            let mut dst = vec![0xEEu8; dst_total];
            rgb565_to_rgba_strided(&src, &mut dst, w, h, src_stride, dst_stride).unwrap();

            // Per-row output must match contiguous
            for y in 0..h {
                let row_src = &src[y * src_stride..y * src_stride + w * 2];
                let mut expected = vec![0u8; w * 4];
                rgb565_to_rgba(row_src, &mut expected).unwrap();
                let got = &dst[y * dst_stride..y * dst_stride + w * 4];
                assert_eq!(got, &expected[..], "strided mismatch w={w} row={y}");
            }

            // Padding bytes must be untouched
            for y in 0..h - 1 {
                let pad = &dst[y * dst_stride + w * 4..(y + 1) * dst_stride];
                for (j, &b) in pad.iter().enumerate() {
                    assert_eq!(b, 0xEE, "padding overwritten w={w} y={y} offset={j}");
                }
            }
        }
    }

    // === Compress direction tests ===

    /// Reference: compress 8-bit to 5-bit with round-to-nearest.
    fn compress5(v: u8) -> u8 {
        ((v as u16 * 31 + 128) >> 8) as u8
    }
    fn compress6(v: u8) -> u8 {
        ((v as u16 * 63 + 128) >> 8) as u8
    }
    fn compress4(v: u8) -> u8 {
        ((v as u16 * 15 + 128) >> 8) as u8
    }

    /// Reference: RGBA → RGB565 (LE).
    fn ref_rgba_to_rgb565(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 4;
        let mut out = vec![0u8; n * 2];
        for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
            let r5 = compress5(s[0]) as u16;
            let g6 = compress6(s[1]) as u16;
            let b5 = compress5(s[2]) as u16;
            d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
        }
        out
    }

    /// Reference: BGRA → RGB565 (LE).
    fn ref_bgra_to_rgb565(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 4;
        let mut out = vec![0u8; n * 2];
        for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
            let r5 = compress5(s[2]) as u16;
            let g6 = compress6(s[1]) as u16;
            let b5 = compress5(s[0]) as u16;
            d.copy_from_slice(&((r5 << 11) | (g6 << 5) | b5).to_le_bytes());
        }
        out
    }

    /// Reference: RGBA → RGBA4444 (LE).
    fn ref_rgba_to_rgba4444(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 4;
        let mut out = vec![0u8; n * 2];
        for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
            let r4 = compress4(s[0]) as u16;
            let g4 = compress4(s[1]) as u16;
            let b4 = compress4(s[2]) as u16;
            let a4 = compress4(s[3]) as u16;
            d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
        }
        out
    }

    /// Reference: BGRA → RGBA4444 (LE).
    fn ref_bgra_to_rgba4444(src: &[u8]) -> Vec<u8> {
        let n = src.len() / 4;
        let mut out = vec![0u8; n * 2];
        for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
            let r4 = compress4(s[2]) as u16;
            let g4 = compress4(s[1]) as u16;
            let b4 = compress4(s[0]) as u16;
            let a4 = compress4(s[3]) as u16;
            d.copy_from_slice(&((r4 << 12) | (g4 << 8) | (b4 << 4) | a4).to_le_bytes());
        }
        out
    }

    #[test]
    fn rgba_to_rgb565_known_values() {
        let mut dst = [0u8; 2];
        // Black
        rgba_to_rgb565(&[0, 0, 0, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0);
        // White
        rgba_to_rgb565(&[255, 255, 255, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0xFFFF);
        // Pure red
        rgba_to_rgb565(&[255, 0, 0, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0xF800);
        // Pure green
        rgba_to_rgb565(&[0, 255, 0, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0x07E0);
        // Pure blue
        rgba_to_rgb565(&[0, 0, 255, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0x001F);
        // Alpha is ignored
        rgba_to_rgb565(&[255, 255, 255, 0], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0xFFFF);
    }

    #[test]
    fn rgba_to_rgba4444_known_values() {
        let mut dst = [0u8; 2];
        // Black transparent
        rgba_to_rgba4444(&[0, 0, 0, 0], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0);
        // White opaque
        rgba_to_rgba4444(&[255, 255, 255, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0xFFFF);
        // Red opaque
        rgba_to_rgba4444(&[255, 0, 0, 255], &mut dst).unwrap();
        assert_eq!(u16::from_le_bytes(dst), 0xF00F);
    }

    #[test]
    fn rgb565_roundtrip_exhaustive() {
        // expand(compress(expand(x))) == expand(x) for all x in [0..32/64/32]
        // Build all 65536 packed values, expand, compress, expand again
        let mut packed = vec![0u8; 65536 * 2];
        for i in 0u32..65536 {
            packed[i as usize * 2..][..2].copy_from_slice(&(i as u16).to_le_bytes());
        }
        let mut expanded = vec![0u8; 65536 * 4];
        rgb565_to_rgba(&packed, &mut expanded).unwrap();

        let mut recompressed = vec![0u8; 65536 * 2];
        rgba_to_rgb565(&expanded, &mut recompressed).unwrap();

        let mut re_expanded = vec![0u8; 65536 * 4];
        rgb565_to_rgba(&recompressed, &mut re_expanded).unwrap();

        assert_eq!(expanded, re_expanded, "RGB565 roundtrip failed");
    }

    #[test]
    fn rgba4444_roundtrip_exhaustive() {
        let mut packed = vec![0u8; 65536 * 2];
        for i in 0u32..65536 {
            packed[i as usize * 2..][..2].copy_from_slice(&(i as u16).to_le_bytes());
        }
        let mut expanded = vec![0u8; 65536 * 4];
        rgba4444_to_rgba(&packed, &mut expanded).unwrap();

        let mut recompressed = vec![0u8; 65536 * 2];
        rgba_to_rgba4444(&expanded, &mut recompressed).unwrap();

        let mut re_expanded = vec![0u8; 65536 * 4];
        rgba4444_to_rgba(&recompressed, &mut re_expanded).unwrap();

        assert_eq!(expanded, re_expanded, "RGBA4444 roundtrip failed");
    }

    #[test]
    fn compress_endpoints() {
        // 0→0, 255→max for all bit depths
        assert_eq!(compress5(0), 0);
        assert_eq!(compress5(255), 31);
        assert_eq!(compress6(0), 0);
        assert_eq!(compress6(255), 63);
        assert_eq!(compress4(0), 0);
        assert_eq!(compress4(255), 15);
    }

    #[test]
    fn permutation_rgba_to_rgb565() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_4bpp(n);
                let expected = ref_rgba_to_rgb565(&src);
                let mut dst = vec![0u8; n * 2];
                rgba_to_rgb565(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba_to_rgb565 n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba_to_rgb565: {report}");
    }

    #[test]
    fn permutation_bgra_to_rgb565() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_4bpp(n);
                let expected = ref_bgra_to_rgb565(&src);
                let mut dst = vec![0u8; n * 2];
                bgra_to_rgb565(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "bgra_to_rgb565 n={n} tier={perm}");
            }
        });
        std::eprintln!("bgra_to_rgb565: {report}");
    }

    #[test]
    fn permutation_rgba_to_rgba4444() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_4bpp(n);
                let expected = ref_rgba_to_rgba4444(&src);
                let mut dst = vec![0u8; n * 2];
                rgba_to_rgba4444(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "rgba_to_rgba4444 n={n} tier={perm}");
            }
        });
        std::eprintln!("rgba_to_rgba4444: {report}");
    }

    #[test]
    fn permutation_bgra_to_rgba4444() {
        let report = for_each_token_permutation(policy(), |perm| {
            for n in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
                let src = make_4bpp(n);
                let expected = ref_bgra_to_rgba4444(&src);
                let mut dst = vec![0u8; n * 2];
                bgra_to_rgba4444(&src, &mut dst).unwrap();
                assert_eq!(dst, expected, "bgra_to_rgba4444 n={n} tier={perm}");
            }
        });
        std::eprintln!("bgra_to_rgba4444: {report}");
    }

    #[test]
    fn compress_size_errors() {
        // Empty
        assert_eq!(
            rgba_to_rgb565(&[], &mut [0; 2]),
            Err(SizeError::NotPixelAligned)
        );
        // Not 4-byte aligned
        assert_eq!(
            rgba_to_rgb565(&[0; 3], &mut [0; 2]),
            Err(SizeError::NotPixelAligned)
        );
        // Dst too small
        assert_eq!(
            rgba_to_rgb565(&[0; 8], &mut [0; 2]),
            Err(SizeError::PixelCountMismatch)
        );
        // Same for RGBA4444
        assert_eq!(
            rgba_to_rgba4444(&[0; 5], &mut [0; 2]),
            Err(SizeError::NotPixelAligned)
        );
        assert_eq!(
            rgba_to_rgba4444(&[0; 8], &mut [0; 2]),
            Err(SizeError::PixelCountMismatch)
        );
    }

    #[test]
    fn compress_bgra_matches_swapped_rgba() {
        // BGRA→RGB565 should give same result as swapping R/B then RGBA→RGB565
        let bgra = make_4bpp(64);
        let mut rgba = bgra.clone();
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let mut dst_from_bgra = vec![0u8; 64 * 2];
        let mut dst_from_rgba = vec![0u8; 64 * 2];
        bgra_to_rgb565(&bgra, &mut dst_from_bgra).unwrap();
        rgba_to_rgb565(&rgba, &mut dst_from_rgba).unwrap();
        assert_eq!(dst_from_bgra, dst_from_rgba);
    }
}
