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
}
