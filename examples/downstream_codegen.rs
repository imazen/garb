//! Versioned codegen & latency probe exercising every `garb::bytes` function
//! used by downstream crates across the zen workspace (`zenpixels-convert`,
//! `zenpng`, `zenjpeg`, `zenwebp`, `zengif`, `zenavif`, `imageflow`).
//!
//! Inspect per-tier assembly with:
//! - `just asm-v3 gb_rgba_to_rgb`
//! - `just asm-v4 gb_rgba_to_rgb`
//! - `just asm-neon gb_rgba_to_rgb`
//! - `just asm-wasm128 gb_rgba_to_rgb`
//! - `just asm-wasm128-relaxed gb_rgba_to_rgb`
//! - `just asm-all gb_rgba_to_rgb`

use core::hint::black_box;
use garb::{SizeError, bytes};

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgba_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgba_to_rgb(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgb_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgb_to_rgba(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_bgra_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::bgra_to_rgb(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgb_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgb_to_bgra(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgba_to_bgra(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgba_to_bgra(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgba_to_bgra_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgba_to_bgra_inplace(buf)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgb_to_bgr(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgb_to_bgr(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgb_to_bgr_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgb_to_bgr_inplace(buf)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_argb_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::argb_to_rgba(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgba_to_argb_inplace(buf: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgba_to_argb_inplace(buf)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_argb_to_rgb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::argb_to_rgb(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_rgb_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::rgb_to_argb(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_gray_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::gray_to_rgba(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_gray_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::gray_to_argb(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_gray_alpha_to_rgba(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::gray_alpha_to_rgba(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_gray_alpha_to_argb(src: &[u8], dst: &mut [u8]) -> Result<(), SizeError> {
    bytes::gray_alpha_to_argb(src, dst)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_fill_alpha_rgba(buf: &mut [u8]) -> Result<(), SizeError> {
    bytes::fill_alpha_rgba(buf)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_strided_rgba_to_rgb(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SizeError> {
    bytes::rgba_to_rgb_strided(src, dst, width, height, src_stride, dst_stride)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn gb_strided_rgba_to_bgra_inplace(
    buf: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), SizeError> {
    bytes::rgba_to_bgra_inplace_strided(buf, width, height, stride)
}

fn main() {
    let mut b4 = [128u8; 256];
    let mut d4 = [0u8; 256];
    let mut b3 = [128u8; 192];
    let mut d3 = [0u8; 192];
    let b2 = [128u8; 128];
    let b1 = [128u8; 64];
    let _ = black_box(gb_rgba_to_rgb(black_box(&b4), black_box(&mut d3)));
    let _ = black_box(gb_rgb_to_rgba(black_box(&b3), black_box(&mut d4)));
    let _ = black_box(gb_bgra_to_rgb(black_box(&b4), black_box(&mut d3)));
    let _ = black_box(gb_rgb_to_bgra(black_box(&b3), black_box(&mut d4)));
    let _ = black_box(gb_rgba_to_bgra(black_box(&b4), black_box(&mut d4)));
    let _ = black_box(gb_rgba_to_bgra_inplace(black_box(&mut b4)));
    let _ = black_box(gb_rgb_to_bgr(black_box(&b3), black_box(&mut d3)));
    let _ = black_box(gb_rgb_to_bgr_inplace(black_box(&mut b3)));
    let _ = black_box(gb_argb_to_rgba(black_box(&b4), black_box(&mut d4)));
    let _ = black_box(gb_rgba_to_argb_inplace(black_box(&mut b4)));
    let _ = black_box(gb_argb_to_rgb(black_box(&b4), black_box(&mut d3)));
    let _ = black_box(gb_rgb_to_argb(black_box(&b3), black_box(&mut d4)));
    let _ = black_box(gb_gray_to_rgba(black_box(&b1), black_box(&mut d4)));
    let _ = black_box(gb_gray_to_argb(black_box(&b1), black_box(&mut d4)));
    let _ = black_box(gb_gray_alpha_to_rgba(black_box(&b2), black_box(&mut d4)));
    let _ = black_box(gb_gray_alpha_to_argb(black_box(&b2), black_box(&mut d4)));
    let _ = black_box(gb_fill_alpha_rgba(black_box(&mut b4)));
    let _ = black_box(gb_strided_rgba_to_rgb(
        black_box(&b4),
        black_box(&mut d3),
        16,
        4,
        64,
        48,
    ));
    let _ = black_box(gb_strided_rgba_to_bgra_inplace(
        black_box(&mut b4),
        16,
        4,
        64,
    ));
}
