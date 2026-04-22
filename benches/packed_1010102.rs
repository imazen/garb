//! Throughput bench for the scalar RGBA1010102 pack/unpack functions added
//! in PR #3. The question this answers: is the per-pixel bit math the
//! bottleneck, or is byte movement (memory bandwidth) the ceiling?
//!
//! Two sizes are exercised:
//!  - 1920x1080 (≈ 2.07 Mpx) — too large for L2/L3, exposes DRAM bandwidth
//!  - 256x256 (≈ 64 Kpx)    — fits comfortably in L2, exposes compute
//!
//! Run: `cargo bench --bench packed_1010102 --features experimental`
//! No `RUSTFLAGS=-C target-cpu=native` — runtime dispatch is what users get.
//!
//! `Throughput::Bytes` is set to (src_bytes + dst_bytes) so the GB/s column
//! reflects total bytes touched, which is the right comparison against a
//! memory-bandwidth ceiling.
//!
//! Requires `--features experimental` (the 1010102 module is gated on it).

#[cfg(not(feature = "experimental"))]
fn main() {
    eprintln!(
        "packed_1010102 bench requires --features experimental \
         (the rgba1010102 module is gated on it)"
    );
}

#[cfg(feature = "experimental")]
mod experimental {
    use garb::bytes::{rgba16_to_rgba1010102, rgba1010102_to_rgba16};
    use zenbench::prelude::*;

    const W_LARGE: usize = 1920;
    const H_LARGE: usize = 1080;
    const W_SMALL: usize = 256;
    const H_SMALL: usize = 256;

    /// Tiny LCG so we get non-trivial inputs without a dev-dep.
    fn fill_packed(buf: &mut [u8], seed: u64) {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_F00D;
        for chunk in buf.chunks_exact_mut(4) {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (s >> 32) as u32;
            chunk.copy_from_slice(&v.to_le_bytes());
        }
    }

    fn fill_chans(buf: &mut [u16], seed: u64) {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1010_1010_2020_2020;
        for slot in buf.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Keep values inside [0, 1023] so the packer's mask is a no-op
            // (representative of well-behaved upstream data).
            *slot = ((s >> 32) as u16) & 0x3FF;
        }
    }

    pub fn bench_unpack(suite: &mut Suite) {
        suite.group("rgba1010102_to_rgba16", |g| {
            for &(w, h, label) in &[
                (W_LARGE, H_LARGE, "1920x1080"),
                (W_SMALL, H_SMALL, "256x256"),
            ] {
                let pixels = w * h;
                let src_bytes = pixels * 4; // 4 packed bytes/px
                let dst_bytes = pixels * 8; // 4 u16/px = 8 bytes/px
                let total_bytes = src_bytes + dst_bytes;

                // Throughput::Bytes(total_bytes) so the report shows GB/s of
                // bytes touched (input read + output written).
                g.throughput(Throughput::Bytes(total_bytes as u64));

                g.bench(format!("scalar/{label}"), move |b| {
                    let mut src = vec![0u8; src_bytes];
                    fill_packed(&mut src, 0xA110_1010_0001_u64);
                    let mut dst = vec![0u16; pixels * 4];
                    b.iter(move || {
                        rgba1010102_to_rgba16(black_box(&src), black_box(&mut dst)).unwrap();
                        black_box(dst.as_ptr());
                    })
                });
            }
        });
    }

    pub fn bench_pack(suite: &mut Suite) {
        suite.group("rgba16_to_rgba1010102", |g| {
            for &(w, h, label) in &[
                (W_LARGE, H_LARGE, "1920x1080"),
                (W_SMALL, H_SMALL, "256x256"),
            ] {
                let pixels = w * h;
                let src_bytes = pixels * 8;
                let dst_bytes = pixels * 4;
                let total_bytes = src_bytes + dst_bytes;

                g.throughput(Throughput::Bytes(total_bytes as u64));

                g.bench(format!("scalar/{label}"), move |b| {
                    let mut src = vec![0u16; pixels * 4];
                    fill_chans(&mut src, 0x1010_1010_0002_u64);
                    let mut dst = vec![0u8; dst_bytes];
                    b.iter(move || {
                        rgba16_to_rgba1010102(black_box(&src), black_box(&mut dst)).unwrap();
                        black_box(dst.as_ptr());
                    })
                });
            }
        });
    }
}

#[cfg(feature = "experimental")]
zenbench::main!(experimental::bench_unpack, experimental::bench_pack);
