//! Inline-check harness for the f32 RGB/RGBA chunk-level SIMD primitives.
//!
//! Each `sample_caller_*` function uses `#[archmage::arcane]` to set up the
//! per-arch `target_feature` region and then calls a `*_chunkN_*_<arch>`
//! function defined with `#[rite]`. Because `#[rite]` injects `#[inline]` +
//! the matching `target_feature` attrs, the chunk function should fuse into
//! the caller's body — the asm dump for `sample_caller_*` should contain
//! the chunk function's shuffle instructions inline, with no `call` /
//! `b` instruction targeting the chunk function's symbol.
//!
//! How to verify the inline:
//!
//! ```text
//! cargo asm --release --lib --features experimental \
//!     "asm_inline_check::sample_caller_v3_rgb_chunk16"
//! ```
//!
//! On x86_64, expect to see vmovups / vshufps / vinsertps / vblendps in the
//! body and **no** `call garb::deinterleave::*chunk16*` instruction.
//! On aarch64, expect `ld3 { v0.4s, v1.4s, v2.4s }` × 4 (one per chunk-4
//! sub-chunk) and no `b garb::deinterleave::*chunk16*` instruction.
//! On wasm32 with `+simd128`, expect multiple `v128.load` / `i8x16.shuffle`
//! and no `call $garb::deinterleave::*chunk16*` instruction.
//!
//! Each sample is also compiled into a runnable test that exercises the
//! function with a fixed input and asserts the output matches the scalar
//! reference — so a regression in the inline (e.g. a future macro change
//! that breaks the target_feature region) still surfaces as a test failure
//! when the host has the required ISA.

#![cfg(feature = "experimental")]

#[cfg(target_arch = "x86_64")]
mod x86_inline {
    use archmage::prelude::*;
    use garb::deinterleave::{
        planes_to_rgb_f32_chunk16_v3, planes_to_rgba_f32_chunk16_v3, rgb_f32_chunk16_to_planes_v3,
        rgba_f32_chunk16_to_planes_v3,
    };

    #[archmage::arcane]
    pub fn sample_caller_v3_rgb_chunk16(
        t: X64V3Token,
        src: &[f32; 48],
    ) -> ([f32; 16], [f32; 16], [f32; 16]) {
        rgb_f32_chunk16_to_planes_v3(t, src)
    }

    #[archmage::arcane]
    pub fn sample_caller_v3_rgba_chunk16(
        t: X64V3Token,
        src: &[f32; 64],
    ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
        rgba_f32_chunk16_to_planes_v3(t, src)
    }

    #[archmage::arcane]
    pub fn sample_caller_v3_planes_to_rgb_chunk16(
        t: X64V3Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
    ) -> [f32; 48] {
        planes_to_rgb_f32_chunk16_v3(t, r, g, b)
    }

    #[archmage::arcane]
    pub fn sample_caller_v3_planes_to_rgba_chunk16(
        t: X64V3Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
        a: &[f32; 16],
    ) -> [f32; 64] {
        planes_to_rgba_f32_chunk16_v3(t, r, g, b, a)
    }
}

#[cfg(target_arch = "aarch64")]
mod neon_inline {
    use archmage::prelude::*;
    use garb::deinterleave::{
        planes_to_rgb_f32_chunk16_neon, planes_to_rgba_f32_chunk16_neon,
        rgb_f32_chunk16_to_planes_neon, rgba_f32_chunk16_to_planes_neon,
    };

    #[archmage::arcane]
    pub fn sample_caller_neon_rgb_chunk16(
        t: NeonToken,
        src: &[f32; 48],
    ) -> ([f32; 16], [f32; 16], [f32; 16]) {
        rgb_f32_chunk16_to_planes_neon(t, src)
    }

    #[archmage::arcane]
    pub fn sample_caller_neon_rgba_chunk16(
        t: NeonToken,
        src: &[f32; 64],
    ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
        rgba_f32_chunk16_to_planes_neon(t, src)
    }

    #[archmage::arcane]
    pub fn sample_caller_neon_planes_to_rgb_chunk16(
        t: NeonToken,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
    ) -> [f32; 48] {
        planes_to_rgb_f32_chunk16_neon(t, r, g, b)
    }

    #[archmage::arcane]
    pub fn sample_caller_neon_planes_to_rgba_chunk16(
        t: NeonToken,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
        a: &[f32; 16],
    ) -> [f32; 64] {
        planes_to_rgba_f32_chunk16_neon(t, r, g, b, a)
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_inline {
    use archmage::prelude::*;
    use garb::deinterleave::{
        planes_to_rgb_f32_chunk16_wasm128, planes_to_rgba_f32_chunk16_wasm128,
        rgb_f32_chunk16_to_planes_wasm128, rgba_f32_chunk16_to_planes_wasm128,
    };

    #[archmage::arcane]
    pub fn sample_caller_wasm128_rgb_chunk16(
        t: Wasm128Token,
        src: &[f32; 48],
    ) -> ([f32; 16], [f32; 16], [f32; 16]) {
        rgb_f32_chunk16_to_planes_wasm128(t, src)
    }

    #[archmage::arcane]
    pub fn sample_caller_wasm128_rgba_chunk16(
        t: Wasm128Token,
        src: &[f32; 64],
    ) -> ([f32; 16], [f32; 16], [f32; 16], [f32; 16]) {
        rgba_f32_chunk16_to_planes_wasm128(t, src)
    }

    #[archmage::arcane]
    pub fn sample_caller_wasm128_planes_to_rgb_chunk16(
        t: Wasm128Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
    ) -> [f32; 48] {
        planes_to_rgb_f32_chunk16_wasm128(t, r, g, b)
    }

    #[archmage::arcane]
    pub fn sample_caller_wasm128_planes_to_rgba_chunk16(
        t: Wasm128Token,
        r: &[f32; 16],
        g: &[f32; 16],
        b: &[f32; 16],
        a: &[f32; 16],
    ) -> [f32; 64] {
        planes_to_rgba_f32_chunk16_wasm128(t, r, g, b, a)
    }
}

// --------------------------------------------------------------------------
// Runtime-callable smoke tests — assert correctness when the host can run
// the SIMD path. Skipped cleanly via `Token::summon()` returning `None`.
// --------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_inline_callers_are_correct() {
    use archmage::prelude::*;
    if let Some(t) = X64V3Token::summon() {
        let src48: [f32; 48] = core::array::from_fn(|i| i as f32 * 0.21 - 1.0);
        let (r_v, g_v, b_v) = x86_inline::sample_caller_v3_rgb_chunk16(t, &src48);
        let (r_s, g_s, b_s) = garb::deinterleave::rgb_f32_chunk16_to_planes_scalar(&src48);
        assert_eq!(r_v, r_s);
        assert_eq!(g_v, g_s);
        assert_eq!(b_v, b_s);

        let src64: [f32; 64] = core::array::from_fn(|i| i as f32 * 0.11 + 5.0);
        let (r_v, g_v, b_v, a_v) = x86_inline::sample_caller_v3_rgba_chunk16(t, &src64);
        let (r_s, g_s, b_s, a_s) = garb::deinterleave::rgba_f32_chunk16_to_planes_scalar(&src64);
        assert_eq!(r_v, r_s);
        assert_eq!(g_v, g_s);
        assert_eq!(b_v, b_s);
        assert_eq!(a_v, a_s);

        let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
        let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
        let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
        let v = x86_inline::sample_caller_v3_planes_to_rgb_chunk16(t, &r, &g, &b);
        let s = garb::deinterleave::planes_to_rgb_f32_chunk16_scalar(&r, &g, &b);
        assert_eq!(v, s);

        let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
        let v = x86_inline::sample_caller_v3_planes_to_rgba_chunk16(t, &r, &g, &b, &a);
        let s = garb::deinterleave::planes_to_rgba_f32_chunk16_scalar(&r, &g, &b, &a);
        assert_eq!(v, s);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_inline_callers_are_correct() {
    use archmage::prelude::*;
    if let Some(t) = NeonToken::summon() {
        let src48: [f32; 48] = core::array::from_fn(|i| i as f32 * 0.21 - 1.0);
        let (r_v, g_v, b_v) = neon_inline::sample_caller_neon_rgb_chunk16(t, &src48);
        let (r_s, g_s, b_s) = garb::deinterleave::rgb_f32_chunk16_to_planes_scalar(&src48);
        assert_eq!(r_v, r_s);
        assert_eq!(g_v, g_s);
        assert_eq!(b_v, b_s);

        let src64: [f32; 64] = core::array::from_fn(|i| i as f32 * 0.11 + 5.0);
        let (r_v, g_v, b_v, a_v) = neon_inline::sample_caller_neon_rgba_chunk16(t, &src64);
        let (r_s, g_s, b_s, a_s) = garb::deinterleave::rgba_f32_chunk16_to_planes_scalar(&src64);
        assert_eq!(r_v, r_s);
        assert_eq!(g_v, g_s);
        assert_eq!(b_v, b_s);
        assert_eq!(a_v, a_s);

        let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
        let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
        let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
        let v = neon_inline::sample_caller_neon_planes_to_rgb_chunk16(t, &r, &g, &b);
        let s = garb::deinterleave::planes_to_rgb_f32_chunk16_scalar(&r, &g, &b);
        assert_eq!(v, s);

        let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
        let v = neon_inline::sample_caller_neon_planes_to_rgba_chunk16(t, &r, &g, &b, &a);
        let s = garb::deinterleave::planes_to_rgba_f32_chunk16_scalar(&r, &g, &b, &a);
        assert_eq!(v, s);
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_inline_callers_are_correct() {
    use archmage::prelude::*;
    if let Some(t) = Wasm128Token::summon() {
        let src48: [f32; 48] = core::array::from_fn(|i| i as f32 * 0.21 - 1.0);
        let (r_v, g_v, b_v) = wasm_inline::sample_caller_wasm128_rgb_chunk16(t, &src48);
        let (r_s, g_s, b_s) = garb::deinterleave::rgb_f32_chunk16_to_planes_scalar(&src48);
        assert_eq!(r_v, r_s);
        assert_eq!(g_v, g_s);
        assert_eq!(b_v, b_s);

        let src64: [f32; 64] = core::array::from_fn(|i| i as f32 * 0.11 + 5.0);
        let (r_v, g_v, b_v, a_v) = wasm_inline::sample_caller_wasm128_rgba_chunk16(t, &src64);
        let (r_s, g_s, b_s, a_s) = garb::deinterleave::rgba_f32_chunk16_to_planes_scalar(&src64);
        assert_eq!(r_v, r_s);
        assert_eq!(g_v, g_s);
        assert_eq!(b_v, b_s);
        assert_eq!(a_v, a_s);

        let r: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
        let g: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.5 + 1.0);
        let b: [f32; 16] = core::array::from_fn(|i| i as f32 * -0.75 + 3.0);
        let v = wasm_inline::sample_caller_wasm128_planes_to_rgb_chunk16(t, &r, &g, &b);
        let s = garb::deinterleave::planes_to_rgb_f32_chunk16_scalar(&r, &g, &b);
        assert_eq!(v, s);

        let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
        let v = wasm_inline::sample_caller_wasm128_planes_to_rgba_chunk16(t, &r, &g, &b, &a);
        let s = garb::deinterleave::planes_to_rgba_f32_chunk16_scalar(&r, &g, &b, &a);
        assert_eq!(v, s);
    }
}
