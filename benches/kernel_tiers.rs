//! Per-kernel NEON-vs-forced-scalar for garb's conversion kernels.
//!
//! garb had 194 public kernels, 271 dispatch sites and NO tier benchmark, so a
//! kernel slower than (or missing) its SIMD tier was invisible. The existing
//! benches measure absolute throughput, which cannot show either.
//!
//! Run: `cargo bench --bench kernel_tiers`
//! Do NOT build with `-C target-cpu=native` (the tier then cannot be disabled).

use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool { false }

const PX: usize = 1 << 20; // 1 MP

fn bench(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let src3: &'static [u8] =
        Box::leak((0..PX * 3).map(|i| (i % 251) as u8).collect::<Vec<_>>().into_boxed_slice());
    let src4: &'static [u8] =
        Box::leak((0..PX * 4).map(|i| (i % 251) as u8).collect::<Vec<_>>().into_boxed_slice());
    let src1: &'static [u8] =
        Box::leak((0..PX).map(|i| (i % 251) as u8).collect::<Vec<_>>().into_boxed_slice());

    macro_rules! ab {
        ($name:expr, $bytes:expr, $out:expr, $call:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes($bytes as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || { set_simd(simd); vec![0u8; $out] })
                            .run(move |mut d| { let _ = $call(&mut d); d })
                    });
                }
            });
        };
    }

    // Shape-distinct families. 3->4 expand and 4->3 contract are the two most
    // common conversions in an image pipeline and are exactly the shapes
    // vld3/vst4 exist for.
    ab!("rgb_to_rgba", PX * 3, PX * 4, |d: &mut Vec<u8>| garb::bytes::rgb_to_rgba(src3, d));
    ab!("rgba_to_rgb", PX * 4, PX * 3, |d: &mut Vec<u8>| garb::bytes::rgba_to_rgb(src4, d));
    ab!("rgb_to_bgr", PX * 3, PX * 3, |d: &mut Vec<u8>| garb::bytes::rgb_to_bgr(src3, d));
    ab!("rgba_to_bgra", PX * 4, PX * 4, |d: &mut Vec<u8>| garb::bytes::rgba_to_bgra(src4, d));
    ab!("gray_to_rgba", PX, PX * 4, |d: &mut Vec<u8>| garb::bytes::gray_to_rgba(src1, d));

    // In-place swizzle (already has a neon arm) as a control.
    suite.compare("rgb_to_bgr_inplace", |g| {
        g.throughput(Throughput::Bytes((PX * 3) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || { set_simd(simd); src3.to_vec() })
                    .run(move |mut d| { let _ = garb::bytes::rgb_to_bgr_inplace(&mut d); d })
            });
        }
    });

    set_simd(true);
}

zenbench::main!(bench);
