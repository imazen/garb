use criterion::{Criterion, Throughput, criterion_group, criterion_main};

// Simple scalar loops that a compiler might autovectorize.
// These are the baseline — garb's SIMD should beat them.

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

const W: usize = 1920;
const H: usize = 1080;

fn bench_4bpp_inplace_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("4bpp_inplace_swap");
    let n = W * H * 4;
    group.throughput(Throughput::Bytes(n as u64));

    let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();

    group.bench_function("garb", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            garb::rgba_to_bgra_inplace(&mut buf).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            naive_rgba_bgra_inplace(&mut buf);
        });
    });

    group.finish();
}

fn bench_3bpp_inplace_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("3bpp_inplace_swap");
    let n = W * H * 3;
    group.throughput(Throughput::Bytes(n as u64));

    let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();

    group.bench_function("garb", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            garb::rgb_to_bgr_inplace(&mut buf).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            naive_rgb_bgr_inplace(&mut buf);
        });
    });

    group.finish();
}

fn bench_3bpp_copy_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("3bpp_copy_swap");
    let n = W * H * 3;
    group.throughput(Throughput::Bytes(n as u64));

    let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    let mut dst = vec![0u8; n];

    group.bench_function("garb", |b| {
        b.iter(|| {
            garb::rgb_to_bgr(&src, &mut dst).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        b.iter(|| {
            naive_rgb_bgr_copy(&src, &mut dst);
        });
    });

    group.finish();
}

fn bench_4to3_strip(c: &mut Criterion) {
    let mut group = c.benchmark_group("4to3_strip_rgba_to_rgb");
    let src_n = W * H * 4;
    let dst_n = W * H * 3;
    group.throughput(Throughput::Bytes(src_n as u64));

    let src: Vec<u8> = (0..src_n).map(|i| (i % 251) as u8).collect();
    let mut dst = vec![0u8; dst_n];

    group.bench_function("garb", |b| {
        b.iter(|| {
            garb::rgba_to_rgb(&src, &mut dst).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        b.iter(|| {
            naive_rgba_to_rgb(&src, &mut dst);
        });
    });

    group.finish();
}

fn bench_4to3_strip_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("4to3_strip_swap_bgra_to_rgb");
    let src_n = W * H * 4;
    let dst_n = W * H * 3;
    group.throughput(Throughput::Bytes(src_n as u64));

    let src: Vec<u8> = (0..src_n).map(|i| (i % 251) as u8).collect();
    let mut dst = vec![0u8; dst_n];

    group.bench_function("garb", |b| {
        b.iter(|| {
            garb::bgra_to_rgb(&src, &mut dst).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        b.iter(|| {
            naive_bgra_to_rgb(&src, &mut dst);
        });
    });

    group.finish();
}

fn bench_3to4_expand(c: &mut Criterion) {
    let mut group = c.benchmark_group("3to4_expand_rgb_to_rgba");
    let src_n = W * H * 3;
    let dst_n = W * H * 4;
    group.throughput(Throughput::Bytes(dst_n as u64));

    let src: Vec<u8> = (0..src_n).map(|i| (i % 251) as u8).collect();
    let mut dst = vec![0u8; dst_n];

    group.bench_function("garb", |b| {
        b.iter(|| {
            garb::rgb_to_rgba(&src, &mut dst).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        b.iter(|| {
            naive_rgb_to_rgba(&src, &mut dst);
        });
    });

    group.finish();
}

fn bench_fill_alpha(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_alpha");
    let n = W * H * 4;
    group.throughput(Throughput::Bytes(n as u64));

    let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();

    group.bench_function("garb", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            garb::fill_alpha_rgba(&mut buf).unwrap();
        });
    });

    group.bench_function("naive", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            naive_fill_alpha(&mut buf);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_4bpp_inplace_swap,
    bench_3bpp_inplace_swap,
    bench_3bpp_copy_swap,
    bench_4to3_strip,
    bench_4to3_strip_swap,
    bench_3to4_expand,
    bench_fill_alpha,
);
criterion_main!(benches);
