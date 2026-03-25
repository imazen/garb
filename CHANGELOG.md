# Changelog

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
