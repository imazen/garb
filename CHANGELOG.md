# Changelog

## 0.2.4

Dependency-only release. No source changes to pixel conversion logic.

### Changed

- **archmage** 0.9.5 → 0.9.12. Archmage 0.9.12 deprecates `SimdToken` in
  favor of `ScalarToken` / tokenless `#[autoversion]`. Garb still uses
  `SimdToken` throughout (20 call sites); the crate carries
  `#![allow(deprecated)]` until migration in 0.3.0.
- **criterion** 0.5 → 0.8 (dev-dependency).

### Notes

- 125 tests passing (115 lib + 4 doc + 6 integration), all features enabled.
- No public API changes. No MSRV change (1.89).

## 0.2.3

Previous release.
