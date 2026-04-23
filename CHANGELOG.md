# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

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
