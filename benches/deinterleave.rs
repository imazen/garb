//! Interleaved-RGB → planar-f32 deinterleave benchmark.
//!
//! Compares the explicit-SIMD path against the scalar (LLVM-vectorize-or-not)
//! baseline across `(u8, u16) × (tiny, small, medium, large)`. The grid covers
//! 64-pixel inputs (per-call fixed overhead dominates) through 4096×4096
//! (per-pixel cost dominates), so we can fit `α + β · pixels` and report
//! both the intercept and the slope.

use archmage::SimdToken;
use garb::deinterleave::{
    planes_f32_to_rgb_f32, planes_f32_to_rgba_f32, rgb_f32_to_planes_f32, rgb24_to_planes_f32,
    rgb48_to_planes_f32, rgba_f32_to_planes_f32,
};
use zenbench::prelude::*;

#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;

fn make_u8(pixels: usize) -> Vec<u8> {
    (0..pixels * 3)
        .map(|i| (i.wrapping_mul(31) & 0xFF) as u8)
        .collect()
}

fn make_u16(pixels: usize) -> Vec<u16> {
    (0..pixels * 3)
        .map(|i| (i.wrapping_mul(8191) & 0xFFFF) as u16)
        .collect()
}

fn make_f32(pixels: usize, channels: usize) -> Vec<f32> {
    (0..pixels * channels)
        .map(|i| (i as f32) * 0.0125 - 8.0)
        .collect()
}

// Sizes chosen to span cache tiers on a typical desktop CPU. Total working set
// per call (RGB24): 15 B per pixel = 3 B input + 12 B output (3 f32 planes).
// RGB48 uses 18 B per pixel. The points are deliberately dense at L1/L2 where
// the SIMD compute speedup is observable, and sparser at DRAM where the path
// is purely memory-bound.
const SIZES: &[(&str, usize)] = &[
    ("256px (L1)", 256),
    ("4096px (L1)", 4096),
    ("65536px (L2)", 65_536),    // 256x256
    ("262144px (L3)", 262_144),  // 512x512
    ("1MP (L3)", 1_048_576),     // 1024x1024
    ("4MP (DRAM)", 4_194_304),   // 2048x2048
    ("16MP (DRAM)", 16_777_216), // 4096x4096
];

fn print_simd_info() {
    eprintln!("=== SIMD Tier Detection ===");
    #[cfg(target_arch = "x86_64")]
    {
        eprintln!(
            "  AVX2+FMA (x86-64-v3):    {}",
            if X64V3Token::summon().is_some() {
                "available (will dispatch)"
            } else {
                "not available (scalar only)"
            }
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        eprintln!("  NEON: available (will dispatch)");
    }
}

fn bench_rgb24(suite: &mut Suite) {
    print_simd_info();
    suite.group("rgb24_to_planes_f32 (u8)", |g| {
        for &(label, pixels) in SIZES {
            g.subgroup(label);

            // Throughput in input bytes (3 bytes per pixel).
            let bytes = pixels * 3;
            g.throughput(Throughput::Bytes(bytes as u64));

            g.bench(&format!("{label} :: scalar"), move |b| {
                b.with_input(move || {
                    let src = make_u8(pixels);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    garb::deinterleave::scalar_only_rgb24(&src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: scalar+autovec(avx2)"), move |b| {
                b.with_input(move || {
                    let src = make_u8(pixels);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    garb::deinterleave::autovec_avx2_rgb24(&src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: simd-dispatch"), move |b| {
                b.with_input(move || {
                    let src = make_u8(pixels);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    rgb24_to_planes_f32(&src, &mut r, &mut gp, &mut bp).unwrap();
                    (src, r, gp, bp)
                })
            });
        }
    });
}

fn bench_rgb48(suite: &mut Suite) {
    suite.group("rgb48_to_planes_f32 (u16)", |g| {
        for &(label, pixels) in SIZES {
            g.subgroup(label);

            let bytes = pixels * 6;
            g.throughput(Throughput::Bytes(bytes as u64));

            g.bench(&format!("{label} :: scalar"), move |b| {
                b.with_input(move || {
                    let src = make_u16(pixels);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    garb::deinterleave::scalar_only_rgb48(&src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: scalar+autovec(avx2)"), move |b| {
                b.with_input(move || {
                    let src = make_u16(pixels);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    garb::deinterleave::autovec_avx2_rgb48(&src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: simd-dispatch"), move |b| {
                b.with_input(move || {
                    let src = make_u16(pixels);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    rgb48_to_planes_f32(&src, &mut r, &mut gp, &mut bp).unwrap();
                    (src, r, gp, bp)
                })
            });
        }
    });
}

// ===========================================================================
// f32 RGB / RGBA  ⇄  planes (identity)
// ===========================================================================
//
// 1:1 memory ratio (12 B in, 12 B out for RGB; 16/16 for RGBA), so DRAM
// bandwidth dominates much sooner than the integer→f32 paths above.
// Compares scalar (cargo default features, basically SSE2-era codegen) vs
// dispatched (which today routes to an #[arcane] AVX2 wrapper that just
// calls the scalar source — i.e. autovec). If hand-tuned permutevar8x32
// shuffles are added later, this is where we'll see them light up.

fn bench_rgb_f32(suite: &mut Suite) {
    suite.group("rgb_f32 ⇄ planes_f32 (identity)", |g| {
        for &(label, pixels) in SIZES {
            g.subgroup(label);
            let bytes = pixels * 3 * 4;
            g.throughput(Throughput::Bytes(bytes as u64));

            g.bench(&format!("{label} :: scatter scalar"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    garb::deinterleave::scalar_only_rgb_f32_to_planes(
                        &src, &mut r, &mut gp, &mut bp,
                    );
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: scatter dispatch(avx2)"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(|(src, mut r, mut gp, mut bp)| {
                    rgb_f32_to_planes_f32(&src, &mut r, &mut gp, &mut bp).unwrap();
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: gather scalar"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 3];
                    (r, gp, bp, dst)
                })
                .run(|(r, gp, bp, mut dst)| {
                    garb::deinterleave::scalar_only_planes_f32_to_rgb(&r, &gp, &bp, &mut dst);
                    (r, gp, bp, dst)
                })
            });

            g.bench(&format!("{label} :: gather dispatch(avx2)"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 3];
                    (r, gp, bp, dst)
                })
                .run(|(r, gp, bp, mut dst)| {
                    planes_f32_to_rgb_f32(&r, &gp, &bp, &mut dst).unwrap();
                    (r, gp, bp, dst)
                })
            });
        }
    });
}

fn bench_rgba_f32(suite: &mut Suite) {
    suite.group("rgba_f32 ⇄ planes_f32 (identity)", |g| {
        for &(label, pixels) in SIZES {
            g.subgroup(label);
            let bytes = pixels * 4 * 4;
            g.throughput(Throughput::Bytes(bytes as u64));

            g.bench(&format!("{label} :: scatter scalar"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 4);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    let ap = vec![0.0f32; pixels];
                    (src, r, gp, bp, ap)
                })
                .run(|(src, mut r, mut gp, mut bp, mut ap)| {
                    garb::deinterleave::scalar_only_rgba_f32_to_planes(
                        &src, &mut r, &mut gp, &mut bp, &mut ap,
                    );
                    (src, r, gp, bp, ap)
                })
            });

            g.bench(&format!("{label} :: scatter dispatch(avx2)"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 4);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    let ap = vec![0.0f32; pixels];
                    (src, r, gp, bp, ap)
                })
                .run(|(src, mut r, mut gp, mut bp, mut ap)| {
                    rgba_f32_to_planes_f32(&src, &mut r, &mut gp, &mut bp, &mut ap).unwrap();
                    (src, r, gp, bp, ap)
                })
            });

            g.bench(&format!("{label} :: gather scalar"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let ap = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 4];
                    (r, gp, bp, ap, dst)
                })
                .run(|(r, gp, bp, ap, mut dst)| {
                    garb::deinterleave::scalar_only_planes_f32_to_rgba(&r, &gp, &bp, &ap, &mut dst);
                    (r, gp, bp, ap, dst)
                })
            });

            g.bench(&format!("{label} :: gather dispatch(avx2)"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let ap = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 4];
                    (r, gp, bp, ap, dst)
                })
                .run(|(r, gp, bp, ap, mut dst)| {
                    planes_f32_to_rgba_f32(&r, &gp, &bp, &ap, &mut dst).unwrap();
                    (r, gp, bp, ap, dst)
                })
            });
        }
    });
}

// ===========================================================================
// Dispatch cadence: per-chunk dispatch vs whole-loop autovec
// ===========================================================================
//
// Same total work (64K f32 pixels, L2-resident), split into a sweep of chunk
// sizes. We compare:
//
//   A. dispatch-per-chunk: outer scalar loop, each iteration calls the
//      `rgb_f32_to_planes_f32` public API which goes through #[arcane]
//      AVX2 dispatch (cached `is_x86_feature_detected!` branch + a
//      target_feature trampoline).
//
//   B. autovec-whole-loop: outer loop is itself wrapped in #[arcane] so the
//      whole loop body runs inside a single AVX2 target_feature region,
//      and the inner work is the plain scalar inline loop (which gets
//      autovectorized inside that region).
//
// Both produce the same SIMD code in the inner body. Only the placement of
// the dispatch boundary differs. This tells us how cheap the per-call
// dispatch really is when callers split work into many small pieces.

#[archmage::arcane]
fn rgb_f32_outer_avx2(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    chunk_pixels: usize,
) {
    let total_pixels = src.len() / 3;
    let n_chunks = total_pixels / chunk_pixels;
    for ci in 0..n_chunks {
        let bs = ci * chunk_pixels * 3;
        let ps = ci * chunk_pixels;
        garb::deinterleave::scalar_only_rgb_f32_to_planes(
            &src[bs..bs + chunk_pixels * 3],
            &mut r[ps..ps + chunk_pixels],
            &mut g[ps..ps + chunk_pixels],
            &mut b[ps..ps + chunk_pixels],
        );
    }
    // tail: any pixels not covered by the chunked loop
    let tail_start = n_chunks * chunk_pixels;
    if tail_start < total_pixels {
        let tail_len = total_pixels - tail_start;
        garb::deinterleave::scalar_only_rgb_f32_to_planes(
            &src[tail_start * 3..(tail_start + tail_len) * 3],
            &mut r[tail_start..tail_start + tail_len],
            &mut g[tail_start..tail_start + tail_len],
            &mut b[tail_start..tail_start + tail_len],
        );
    }
}

fn rgb_f32_outer_dispatched(
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    chunk_pixels: usize,
) {
    let total_pixels = src.len() / 3;
    let n_chunks = total_pixels / chunk_pixels;
    for ci in 0..n_chunks {
        let bs = ci * chunk_pixels * 3;
        let ps = ci * chunk_pixels;
        rgb_f32_to_planes_f32(
            &src[bs..bs + chunk_pixels * 3],
            &mut r[ps..ps + chunk_pixels],
            &mut g[ps..ps + chunk_pixels],
            &mut b[ps..ps + chunk_pixels],
        )
        .unwrap();
    }
    let tail_start = n_chunks * chunk_pixels;
    if tail_start < total_pixels {
        let tail_len = total_pixels - tail_start;
        rgb_f32_to_planes_f32(
            &src[tail_start * 3..(tail_start + tail_len) * 3],
            &mut r[tail_start..tail_start + tail_len],
            &mut g[tail_start..tail_start + tail_len],
            &mut b[tail_start..tail_start + tail_len],
        )
        .unwrap();
    }
}

fn bench_dispatch_cadence(suite: &mut Suite) {
    let token = X64V3Token::summon();
    if token.is_none() {
        eprintln!("[dispatch_cadence] AVX2 unavailable — skipping group");
        return;
    }
    let token = token.unwrap();

    suite.group("rgb_f32 dispatch cadence (64K total)", |g| {
        let total_pixels = 65_536_usize;
        g.throughput(Throughput::Bytes((total_pixels * 3 * 4) as u64));

        for &chunk in &[8_usize, 32, 128, 512, 2048, 8192] {
            g.subgroup(&format!("chunk={chunk}"));

            g.bench(&format!("chunk={chunk} :: dispatch-per-chunk"), move |b| {
                b.with_input(move || {
                    let src = make_f32(total_pixels, 3);
                    let r = vec![0.0f32; total_pixels];
                    let gp = vec![0.0f32; total_pixels];
                    let bp = vec![0.0f32; total_pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_outer_dispatched(&src, &mut r, &mut gp, &mut bp, chunk);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("chunk={chunk} :: autovec-whole-loop"), move |b| {
                b.with_input(move || {
                    let src = make_f32(total_pixels, 3);
                    let r = vec![0.0f32; total_pixels];
                    let gp = vec![0.0f32; total_pixels];
                    let bp = vec![0.0f32; total_pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_outer_avx2(token, &src, &mut r, &mut gp, &mut bp, chunk);
                    (src, r, gp, bp)
                })
            });
        }
    });
}

// ===========================================================================
// Chunk-size choice for the slice dispatcher pipeline
// ===========================================================================
//
// The current `rgb_f32_to_planes_impl_v3` cascades chunk16 → chunk8 → chunk4 →
// scalar. Question: how much does each tier earn its keep, and is the
// chunk16 path alone (with a flat scalar tail of up to 15 pixels) good
// enough? We A/B four dispatcher variants over realistic sizes (a mix of
// clean multiples and ones with awkward tails) so the cost of the tail
// path is part of what we measure.

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn rgb_f32_dispatch_chunk16_only(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let pixels = src.len() / 3;
    let n_chunks = pixels / 16;
    for ci in 0..n_chunks {
        let bs = ci * 48;
        let ps = ci * 16;
        let chunk: &[f32; 48] = src[bs..bs + 48].try_into().unwrap();
        let (rc, gc, bc) = garb::deinterleave::rgb_f32_chunk16_to_planes_tokenless_v3(chunk);
        r[ps..ps + 16].copy_from_slice(&rc);
        g[ps..ps + 16].copy_from_slice(&gc);
        b[ps..ps + 16].copy_from_slice(&bc);
    }
    for p in (n_chunks * 16)..pixels {
        r[p] = src[p * 3];
        g[p] = src[p * 3 + 1];
        b[p] = src[p * 3 + 2];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn rgb_f32_dispatch_chunk8_only(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let pixels = src.len() / 3;
    let n_chunks = pixels / 8;
    for ci in 0..n_chunks {
        let bs = ci * 24;
        let ps = ci * 8;
        let chunk: &[f32; 24] = src[bs..bs + 24].try_into().unwrap();
        let (rc, gc, bc) = garb::deinterleave::rgb_f32_chunk8_to_planes_tokenless_v3(chunk);
        r[ps..ps + 8].copy_from_slice(&rc);
        g[ps..ps + 8].copy_from_slice(&gc);
        b[ps..ps + 8].copy_from_slice(&bc);
    }
    for p in (n_chunks * 8)..pixels {
        r[p] = src[p * 3];
        g[p] = src[p * 3 + 1];
        b[p] = src[p * 3 + 2];
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn rgb_f32_dispatch_chunk4_only(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let pixels = src.len() / 3;
    let n_chunks = pixels / 4;
    for ci in 0..n_chunks {
        let bs = ci * 12;
        let ps = ci * 4;
        let chunk: &[f32; 12] = src[bs..bs + 12].try_into().unwrap();
        let (rc, gc, bc) = garb::deinterleave::rgb_f32_chunk4_to_planes_tokenless_v3(chunk);
        r[ps..ps + 4].copy_from_slice(&rc);
        g[ps..ps + 4].copy_from_slice(&gc);
        b[ps..ps + 4].copy_from_slice(&bc);
    }
    for p in (n_chunks * 4)..pixels {
        r[p] = src[p * 3];
        g[p] = src[p * 3 + 1];
        b[p] = src[p * 3 + 2];
    }
}

// Cascade replicates the current `rgb_f32_to_planes_impl_v3` body shape
// (chunk16 → chunk8 → chunk4 → scalar) inline so we're comparing the same
// per-tier kernels, not the public dispatcher with its `incant!` overhead.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn rgb_f32_dispatch_cascade(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let pixels = src.len() / 3;
    let mut p = 0;
    while p + 16 <= pixels {
        let chunk: &[f32; 48] = src[p * 3..p * 3 + 48].try_into().unwrap();
        let (rc, gc, bc) = garb::deinterleave::rgb_f32_chunk16_to_planes_tokenless_v3(chunk);
        r[p..p + 16].copy_from_slice(&rc);
        g[p..p + 16].copy_from_slice(&gc);
        b[p..p + 16].copy_from_slice(&bc);
        p += 16;
    }
    while p + 8 <= pixels {
        let chunk: &[f32; 24] = src[p * 3..p * 3 + 24].try_into().unwrap();
        let (rc, gc, bc) = garb::deinterleave::rgb_f32_chunk8_to_planes_tokenless_v3(chunk);
        r[p..p + 8].copy_from_slice(&rc);
        g[p..p + 8].copy_from_slice(&gc);
        b[p..p + 8].copy_from_slice(&bc);
        p += 8;
    }
    while p + 4 <= pixels {
        let chunk: &[f32; 12] = src[p * 3..p * 3 + 12].try_into().unwrap();
        let (rc, gc, bc) = garb::deinterleave::rgb_f32_chunk4_to_planes_tokenless_v3(chunk);
        r[p..p + 4].copy_from_slice(&rc);
        g[p..p + 4].copy_from_slice(&gc);
        b[p..p + 4].copy_from_slice(&bc);
        p += 4;
    }
    while p < pixels {
        r[p] = src[p * 3];
        g[p] = src[p * 3 + 1];
        b[p] = src[p * 3 + 2];
        p += 1;
    }
}

// Sizes chosen to exercise the trade-off:
//
//   16    → 1 chunk16 / 2 chunk8 / 4 chunk4 / 0 tail (clean for all)
//   17    → 1 chunk16 + 1 tail / 2 chunk8 + 1 tail / 4 chunk4 + 1 tail (chunk16 has the longest tail)
//   31    → 1 chunk16 + 15 tail / 3 chunk8 + 7 tail / 7 chunk4 + 3 tail (chunk16 worst-case tail)
//   1024  → mid-L1, dense — pure per-pixel rate test
//   4099  → 256 chunk16 + 3 tail / 512 chunk8 + 3 tail / 1024 chunk4 + 3 tail (3-pixel tail across all)
//   65536 → L2-resident, big enough that fixed-overhead is negligible
const CHUNK_CHOICE_SIZES: &[(&str, usize)] = &[
    ("16px (clean)", 16),
    ("17px (chunk16 +1 tail)", 17),
    ("31px (chunk16 +15 tail)", 31),
    ("1024px (L1, clean)", 1024),
    ("4099px (L1, +3 tail)", 4099),
    ("65536px (L2)", 65_536),
];

#[cfg(target_arch = "x86_64")]
fn bench_chunk_size_choice(suite: &mut Suite) {
    let token = X64V3Token::summon();
    if token.is_none() {
        eprintln!("[chunk_size_choice] AVX2 unavailable — skipping group");
        return;
    }
    let token = token.unwrap();

    suite.group("rgb_f32 chunk size choice", |g| {
        for &(label, pixels) in CHUNK_CHOICE_SIZES {
            g.subgroup(label);
            g.throughput(Throughput::Bytes((pixels * 3 * 4) as u64));

            g.bench(&format!("{label} :: chunk16+tail"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_dispatch_chunk16_only(token, &src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: chunk8+tail"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_dispatch_chunk8_only(token, &src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: chunk4+tail"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_dispatch_chunk4_only(token, &src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: cascade 16-8-4-tail"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_dispatch_cascade(token, &src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });
        }
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_chunk_size_choice(_suite: &mut Suite) {}

// ===========================================================================
// Hand-written chunk SIMD vs LLVM autovec under #[arcane(v3)]
// ===========================================================================
//
// The f32 chunk SIMD bodies (in `mod x86_f32_chunks`) use 128-bit `_mm_*`
// intrinsics exclusively (75 ops vs 0 `_mm256_*` ops in the f32 chunk
// module — only the u8/u16 chunks use 256-bit AVX2 for the widening +
// store steps). That means chunk size only changes loop overhead — not
// SIMD throughput.
//
// The honest comparison is hand-written chunk SIMD vs LLVM autovec on
// the plain scalar loop wrapped in `#[arcane]` AVX2 target_feature.
// LLVM autovec under target_feature avx2,fma should be free to emit
// 256-bit YMM ops if the loop shape allows. If autovec ties or beats
// the hand-written chunks, the hand-written work is net-zero or
// net-negative.

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn autovec_avx2_rgb_f32_slice(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    garb::deinterleave::scalar_only_rgb_f32_to_planes(src, r, g, b);
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn autovec_avx2_rgba_f32_slice(
    _t: X64V3Token,
    src: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    a: &mut [f32],
) {
    garb::deinterleave::scalar_only_rgba_f32_to_planes(src, r, g, b, a);
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn autovec_avx2_planes_to_rgb_f32_slice(
    _t: X64V3Token,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    dst: &mut [f32],
) {
    garb::deinterleave::scalar_only_planes_f32_to_rgb(r, g, b, dst);
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn autovec_avx2_planes_to_rgba_f32_slice(
    _t: X64V3Token,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    a: &[f32],
    dst: &mut [f32],
) {
    garb::deinterleave::scalar_only_planes_f32_to_rgba(r, g, b, a, dst);
}

const AUTOVEC_VS_CHUNK_SIZES: &[(&str, usize)] = &[
    ("64px (L1)", 64),
    ("256px (L1)", 256),
    ("1024px (L1)", 1024),
    ("4096px (L1)", 4096),
    ("16384px (L1)", 16_384),
    ("65536px (L2)", 65_536),
    ("262144px (L3)", 262_144),
    ("1MP (L3)", 1_048_576),
];

#[cfg(target_arch = "x86_64")]
fn bench_autovec_vs_chunk(suite: &mut Suite) {
    let token = X64V3Token::summon();
    if token.is_none() {
        eprintln!("[autovec_vs_chunk] AVX2 unavailable — skipping group");
        return;
    }
    let token = token.unwrap();

    suite.group("rgb_f32 autovec vs chunk SIMD", |g| {
        for &(label, pixels) in AUTOVEC_VS_CHUNK_SIZES {
            g.subgroup(label);
            g.throughput(Throughput::Bytes((pixels * 3 * 4) as u64));

            g.bench(&format!("{label} :: scatter chunk-SIMD"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    rgb_f32_to_planes_f32(&src, &mut r, &mut gp, &mut bp).unwrap();
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: scatter autovec(avx2)"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 3);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    (src, r, gp, bp)
                })
                .run(move |(src, mut r, mut gp, mut bp)| {
                    autovec_avx2_rgb_f32_slice(token, &src, &mut r, &mut gp, &mut bp);
                    (src, r, gp, bp)
                })
            });

            g.bench(&format!("{label} :: gather chunk-SIMD"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 3];
                    (r, gp, bp, dst)
                })
                .run(move |(r, gp, bp, mut dst)| {
                    planes_f32_to_rgb_f32(&r, &gp, &bp, &mut dst).unwrap();
                    (r, gp, bp, dst)
                })
            });

            g.bench(&format!("{label} :: gather autovec(avx2)"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 3];
                    (r, gp, bp, dst)
                })
                .run(move |(r, gp, bp, mut dst)| {
                    autovec_avx2_planes_to_rgb_f32_slice(token, &r, &gp, &bp, &mut dst);
                    (r, gp, bp, dst)
                })
            });
        }
    });

    suite.group("rgba_f32 autovec vs chunk SIMD", |g| {
        for &(label, pixels) in AUTOVEC_VS_CHUNK_SIZES {
            g.subgroup(label);
            g.throughput(Throughput::Bytes((pixels * 4 * 4) as u64));

            g.bench(&format!("{label} :: scatter chunk-SIMD"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 4);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    let ap = vec![0.0f32; pixels];
                    (src, r, gp, bp, ap)
                })
                .run(move |(src, mut r, mut gp, mut bp, mut ap)| {
                    rgba_f32_to_planes_f32(&src, &mut r, &mut gp, &mut bp, &mut ap).unwrap();
                    (src, r, gp, bp, ap)
                })
            });

            g.bench(&format!("{label} :: scatter autovec(avx2)"), move |b| {
                b.with_input(move || {
                    let src = make_f32(pixels, 4);
                    let r = vec![0.0f32; pixels];
                    let gp = vec![0.0f32; pixels];
                    let bp = vec![0.0f32; pixels];
                    let ap = vec![0.0f32; pixels];
                    (src, r, gp, bp, ap)
                })
                .run(move |(src, mut r, mut gp, mut bp, mut ap)| {
                    autovec_avx2_rgba_f32_slice(token, &src, &mut r, &mut gp, &mut bp, &mut ap);
                    (src, r, gp, bp, ap)
                })
            });

            g.bench(&format!("{label} :: gather chunk-SIMD"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let ap = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 4];
                    (r, gp, bp, ap, dst)
                })
                .run(move |(r, gp, bp, ap, mut dst)| {
                    planes_f32_to_rgba_f32(&r, &gp, &bp, &ap, &mut dst).unwrap();
                    (r, gp, bp, ap, dst)
                })
            });

            g.bench(&format!("{label} :: gather autovec(avx2)"), move |b| {
                b.with_input(move || {
                    let r = make_f32(pixels, 1);
                    let gp = make_f32(pixels, 1);
                    let bp = make_f32(pixels, 1);
                    let ap = make_f32(pixels, 1);
                    let dst = vec![0.0f32; pixels * 4];
                    (r, gp, bp, ap, dst)
                })
                .run(move |(r, gp, bp, ap, mut dst)| {
                    autovec_avx2_planes_to_rgba_f32_slice(token, &r, &gp, &bp, &ap, &mut dst);
                    (r, gp, bp, ap, dst)
                })
            });
        }
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_autovec_vs_chunk(_suite: &mut Suite) {}

zenbench::main!(
    bench_rgb24,
    bench_rgb48,
    bench_rgb_f32,
    bench_rgba_f32,
    bench_dispatch_cadence,
    bench_chunk_size_choice,
    bench_autovec_vs_chunk
);
