use archmage::SimdToken;
use criterion::{BenchmarkGroup, Criterion, Throughput, measurement::WallTime};

// === SIMD tier detection ===

fn probe<T: SimdToken>() -> &'static str {
    if T::summon().is_some() {
        "available"
    } else {
        "not available"
    }
}

fn print_simd_info() {
    eprintln!("=== SIMD Tier Detection ===");
    #[cfg(target_arch = "x86_64")]
    {
        eprintln!(
            "  AVX-512 (x86-64-v4):     {}",
            probe::<archmage::X64V4Token>()
        );
        eprintln!(
            "  AVX2+FMA (x86-64-v3):    {}",
            probe::<archmage::X64V3Token>()
        );
        eprintln!(
            "  SSE4.2 (x86-64-v2):      {}",
            probe::<archmage::X64V2Token>()
        );
        eprintln!(
            "  SSE2 (x86-64-v1):        {}",
            probe::<archmage::X64V1Token>()
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        eprintln!(
            "  Arm64-v3:                {}",
            probe::<archmage::Arm64V3Token>()
        );
        eprintln!(
            "  Arm64-v2:                {}",
            probe::<archmage::Arm64V2Token>()
        );
        eprintln!(
            "  NEON:                    {}",
            probe::<archmage::NeonToken>()
        );
    }
    #[cfg(target_arch = "wasm32")]
    {
        eprintln!(
            "  WASM SIMD128:            {}",
            probe::<archmage::Wasm128Token>()
        );
    }
    eprintln!("  Scalar:                  always available");
    eprintln!("===========================");
}

// === Scalar disable/enable via archmage ===

fn disable_all_simd() {
    let _ = archmage::dangerously_disable_tokens_except_wasm(true);
}

fn enable_all_simd() {
    let _ = archmage::dangerously_disable_tokens_except_wasm(false);
}

// === Naive scalar baselines ===

fn naive_rgba_bgra_inplace(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
}

fn naive_rgb_bgr_inplace(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
}

fn naive_rgb_bgr_copy(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

fn naive_rgba_to_rgb(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

fn naive_bgra_to_rgb(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
    }
}

fn naive_rgb_to_rgba(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = 255;
    }
}

fn naive_fill_alpha(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(4) {
        px[3] = 255;
    }
}

// === Benchmark helpers ===

const W: usize = 1920;
const H: usize = 1080;

/// Benchmark an in-place operation with 3 variants: garb (best SIMD), garb_scalar, naive.
fn bench_inplace(
    group: &mut BenchmarkGroup<WallTime>,
    garb_fn: fn(&mut [u8]) -> Result<(), garb::SizeError>,
    naive_fn: fn(&mut [u8]),
    buf: &[u8],
) {
    group.bench_function("garb", |b| {
        let mut v = buf.to_vec();
        b.iter(|| garb_fn(&mut v).unwrap());
    });

    disable_all_simd();
    group.bench_function("garb_scalar", |b| {
        let mut v = buf.to_vec();
        b.iter(|| garb_fn(&mut v).unwrap());
    });
    enable_all_simd();

    group.bench_function("naive", |b| {
        let mut v = buf.to_vec();
        b.iter(|| naive_fn(&mut v));
    });
}

/// Benchmark a copy operation with 3 variants: garb (best SIMD), garb_scalar, naive.
fn bench_copy(
    group: &mut BenchmarkGroup<WallTime>,
    garb_fn: fn(&[u8], &mut [u8]) -> Result<(), garb::SizeError>,
    naive_fn: fn(&[u8], &mut [u8]),
    src: &[u8],
    dst_len: usize,
) {
    group.bench_function("garb", |b| {
        let mut dst = vec![0u8; dst_len];
        b.iter(|| garb_fn(src, &mut dst).unwrap());
    });

    disable_all_simd();
    group.bench_function("garb_scalar", |b| {
        let mut dst = vec![0u8; dst_len];
        b.iter(|| garb_fn(src, &mut dst).unwrap());
    });
    enable_all_simd();

    group.bench_function("naive", |b| {
        let mut dst = vec![0u8; dst_len];
        b.iter(|| naive_fn(src, &mut dst));
    });
}

// === Benchmark groups ===

fn bench_4bpp_inplace_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("4bpp_inplace_swap");
    let n = W * H * 4;
    group.throughput(Throughput::Bytes(n as u64));
    let buf: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    bench_inplace(
        &mut group,
        garb::bytes::rgba_to_bgra_inplace,
        naive_rgba_bgra_inplace,
        &buf,
    );
    group.finish();
}

fn bench_3bpp_inplace_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("3bpp_inplace_swap");
    let n = W * H * 3;
    group.throughput(Throughput::Bytes(n as u64));
    let buf: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    bench_inplace(
        &mut group,
        garb::bytes::rgb_to_bgr_inplace,
        naive_rgb_bgr_inplace,
        &buf,
    );
    group.finish();
}

fn bench_3bpp_copy_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("3bpp_copy_swap");
    let n = W * H * 3;
    group.throughput(Throughput::Bytes(n as u64));
    let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    bench_copy(
        &mut group,
        garb::bytes::rgb_to_bgr,
        naive_rgb_bgr_copy,
        &src,
        n,
    );
    group.finish();
}

fn bench_4to3_strip(c: &mut Criterion) {
    let mut group = c.benchmark_group("4to3_strip_rgba_to_rgb");
    let src_n = W * H * 4;
    let dst_n = W * H * 3;
    group.throughput(Throughput::Bytes(src_n as u64));
    let src: Vec<u8> = (0..src_n).map(|i| (i % 251) as u8).collect();
    bench_copy(
        &mut group,
        garb::bytes::rgba_to_rgb,
        naive_rgba_to_rgb,
        &src,
        dst_n,
    );
    group.finish();
}

fn bench_4to3_strip_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("4to3_strip_swap_bgra_to_rgb");
    let src_n = W * H * 4;
    let dst_n = W * H * 3;
    group.throughput(Throughput::Bytes(src_n as u64));
    let src: Vec<u8> = (0..src_n).map(|i| (i % 251) as u8).collect();
    bench_copy(
        &mut group,
        garb::bytes::bgra_to_rgb,
        naive_bgra_to_rgb,
        &src,
        dst_n,
    );
    group.finish();
}

fn bench_3to4_expand(c: &mut Criterion) {
    let mut group = c.benchmark_group("3to4_expand_rgb_to_rgba");
    let src_n = W * H * 3;
    let dst_n = W * H * 4;
    group.throughput(Throughput::Bytes(dst_n as u64));
    let src: Vec<u8> = (0..src_n).map(|i| (i % 251) as u8).collect();
    bench_copy(
        &mut group,
        garb::bytes::rgb_to_rgba,
        naive_rgb_to_rgba,
        &src,
        dst_n,
    );
    group.finish();
}

fn bench_fill_alpha(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_alpha");
    let n = W * H * 4;
    group.throughput(Throughput::Bytes(n as u64));
    let buf: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    bench_inplace(
        &mut group,
        garb::bytes::fill_alpha_rgba,
        naive_fill_alpha,
        &buf,
    );
    group.finish();
}

// === Custom main for tier detection before criterion runs ===

fn main() {
    print_simd_info();

    let mut criterion = Criterion::default().configure_from_args();
    bench_4bpp_inplace_swap(&mut criterion);
    bench_3bpp_inplace_swap(&mut criterion);
    bench_3bpp_copy_swap(&mut criterion);
    bench_4to3_strip(&mut criterion);
    bench_4to3_strip_swap(&mut criterion);
    bench_3to4_expand(&mut criterion);
    bench_fill_alpha(&mut criterion);
    criterion.final_summary();
}
