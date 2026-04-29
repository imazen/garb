# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

## [0.2.7] - 2026-04-29

### Added (experimental)

- **`deinterleave` module** under `experimental` — pure identity (no
  transfer-function, no color matrix) interleave/deinterleave between
  packed and planar pixel layouts.
  - `rgb24_to_planes_f32(&[u8], &mut [f32]; 3)` — packed RGB24 → 3×f32
    planes. AVX2 path uses 6×vpshufb + 3×vpor + 3×vpmovzxbd + 3×vcvtdq2ps
    per 8-pixel chunk in place of the 21-vpinsrb scatter LLVM produces
    for the naïve loop. NEON path uses `vld3q_u8` hardware structure-load
    (16-pixel chunks). (`a4fd62d` — feat, `f12c51c` — aarch64 fix)
  - `rgb48_to_planes_f32(&[u16], &mut [f32]; 3)` — same shape for u16
    sources. `vld3q_u16` on NEON.
  - `rgb_f32_to_planes_f32` / `rgba_f32_to_planes_f32` — f32 RGB(A)
    interleaved → planes. AVX2 routing via `#[arcane]` autovec wrapper;
    explicit `permutevar8x32` not landed yet (autovec captures most of
    the available win at 1:1 memory ratio).
  - `planes_f32_to_rgb_f32` / `planes_f32_to_rgba_f32` — inverse
    (planes → interleaved). Same autovec routing.
  - `#[doc(hidden)]` benchmark handles: `scalar_only_*`,
    `autovec_avx2_rgb24/48`. The `scalar_only_*` set is `#[inline(always)]`
    so callers can hoist dispatch outside hot loops by wrapping their
    own `#[arcane]` boundary and calling these as inline scalar inner
    kernels.
- **Chunk-level primitives** (8 pixels per call) for callers already
  inside a `#[target_feature]` region:
  - `rgb24_chunk8_to_planes_v3(X64V3Token, &[u8; 24]) -> ([f32; 8]; 3)`
  - `rgb48_chunk8_to_planes_v3(X64V3Token, &[u16; 24]) -> ([f32; 8]; 3)`
  - `rgb24_chunk8_to_planes_scalar(&[u8; 24]) -> ([f32; 8]; 3)`
  - `rgb48_chunk8_to_planes_scalar(&[u16; 24]) -> ([f32; 8]; 3)`
  These are the hooks zenanalyze's `#[magetypes]`-decorated tier1 kernels
  use to replace the inline scatter without adding raw intrinsics in
  zenanalyze (which forbids `unsafe_code`). (`58c1cd7`)
- `benches/deinterleave.rs` (zenbench harness) sweeping 256 px → 16 MP
  across cache tiers, plus a dispatch-cadence group that quantifies
  per-call `#[arcane]` overhead at ~9 ns/call (breakeven at ~128 px).

### Bench results (7950X with AVX2)

  RGB24 → planes (hand-SIMD vs naive scalar): peak 4.9× at L3-resident
  sizes (~262K-1MP); 1.05× at 4 MP+ where DRAM write bandwidth dominates.

  RGB48 → planes: peak 4.0×; same DRAM cliff.

  f32 RGB(A) ⇄ planes: 1.4-2.1× scatter at L1/L2; gather already tight
  (autovec captures the win); flat at L3+.

  Caller benefit (zenanalyze `tier1_bench`): Variance feature 5-7%
  faster at 1-16 MP; mixed feature sets 2-4% faster (deinterleave is
  proportionally less of total tier1 work).

### Changed

- `archmage` 0.9.21 → 0.9.21 (pin unchanged; CI now hits 0.9.23 via
  range resolution). Pulls in upstream dispatch fixes.
- `Cargo.lock` refreshed for the `deinterleave` additions.
- `cargo fmt` ran on the whole tree.

### Internal

- The `deinterleave` module is gated behind `experimental` so the API
  shape can iterate without 0.2.x→0.3.0 churn while concrete callers
  (zenanalyze, zenfilters) settle on what they want centralised here.

## [0.2.6] - 2026-04-22

### Added

- **RGBA1010102 packed-format pack/unpack** under `experimental` feature
  (`bytes::rgba1010102_to_rgba16`, `bytes::rgba16_to_rgba1010102`, plus
  `_strided` variants). Layout matches DXGI `R10G10B10A2_UNORM` /
  Vulkan `A2B10G10R10_UNORM_PACK32` / WGPU `Rgb10a2Unorm` (R in low bits,
  A in MSBs). Unpacks to interleaved `u16` channels with values in
  `[0, 1023]`; 2-bit alpha is expanded by bit replication per the
  graphics-API convention. Transfer functions are not applied — chain with
  `linear-srgb` for PQ/HLG. (PR [#3](https://github.com/imazen/garb/pull/3),
  squashed as `18b9f18`)
- `#[autoversion]` wrappers on the new RGBA1010102 pack/unpack hot loops so
  they participate in archmage's runtime SIMD dispatch alongside the rest of
  the experimental surface. (`18b9f18`)
- README and `bytes::packed_1010102` module docs document the new functions,
  the graphics-API layout match, and the alpha bit-replication convention.
  (`18b9f18`)

### Changed

- **archmage** 0.9.14 → 0.9.21. Pulls in upstream dispatch/codegen fixes.
- All dependency versions written in full per workspace policy
  (no truncated `"1"` / `"0.8"` strings): `bytemuck = "1.25.0"`,
  `rgb = "0.8.53"`, `paste = "1.0.15"`, `imgref = "1.12.0"`,
  `criterion = "0.8.2"`.
- README badges switched to `?style=flat-square` and inline with the
  `# garb` header per the imazen badge convention; added the `lib.rs`
  badge so the required-five (CI / crates.io / lib.rs / docs.rs /
  license) is complete.

## [0.2.5]

### Fixed

- **Fixed alignment panics on unaligned `&[u8]` buffers.** `bytemuck::cast_slice`
  replaced with `try_cast_slice` + scalar fallback, so byte-level functions no
  longer panic when the input isn't naturally aligned (e.g. `Vec<u8>` on Windows).
  Closes [#1](https://github.com/imazen/garb/issues/1).

### Added

- README badges (crates.io, docs.rs, CI, license, MSRV).
- Alignment benchmarks comparing aligned vs unaligned buffers across SIMD tiers.

### Changed

- CI now tests on Windows x86_64, Windows aarch64, Windows i686, macOS Intel,
  macOS aarch64, Linux x86_64, Linux aarch64, and wasm32.

## 0.2.4

Archmage migration + dep bumps. No public API changes.

### Changed

- **Migrated 20 `#[autoversion]` functions from `SimdToken` to tokenless.**
  Removes all archmage deprecation warnings. These scalar fallback functions
  are called directly (not via `incant!`), so `#[autoversion]` injects the
  token internally in tokenless mode.
- **archmage** 0.9.5 → 0.9.12.
- **criterion** 0.5 → 0.8 (dev-dependency).
- Fixed `stride / 1` identity op and complex type in test code (clippy).
- Fixed `rgb565_to_rgba` doc link to private module (rustdoc).

### Notes

- 125 tests passing, clippy clean (lib + tests), docs clean.
- No public API changes. No MSRV change (1.89).

## 0.2.3

Previous release.
