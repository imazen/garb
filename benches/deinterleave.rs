//! Interleaved-RGB → planar-f32 deinterleave benchmark.
//!
//! Compares the explicit-SIMD path against the scalar (LLVM-vectorize-or-not)
//! baseline across `(u8, u16) × (tiny, small, medium, large)`. The grid covers
//! 64-pixel inputs (per-call fixed overhead dominates) through 4096×4096
//! (per-pixel cost dominates), so we can fit `α + β · pixels` and report
//! both the intercept and the slope.

use archmage::SimdToken;
use garb::deinterleave::{
    planes_f32_to_rgb_f32, planes_f32_to_rgba_f32, rgb24_to_planes_f32, rgb48_to_planes_f32,
    rgb_f32_to_planes_f32, rgba_f32_to_planes_f32,
};
use zenbench::prelude::*;

#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;

fn make_u8(pixels: usize) -> Vec<u8> {
    (0..pixels * 3).map(|i| (i.wrapping_mul(31) & 0xFF) as u8).collect()
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
                    garb::deinterleave::scalar_only_planes_f32_to_rgba(
                        &r, &gp, &bp, &ap, &mut dst,
                    );
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

zenbench::main!(
    bench_rgb24,
    bench_rgb48,
    bench_rgb_f32,
    bench_rgba_f32,
    bench_dispatch_cadence
);
