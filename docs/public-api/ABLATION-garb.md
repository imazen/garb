# garb public-API ablation report

**Date:** 2026-06-11
**Snapshot commit:** 0618744
**Crate analyzed:** `garb` (133 default / 457 all-features items)
**Grep template:** `ugrep -r --include="*.rs" --include="*.toml" "<symbol>" /home/lilith/work/ --exclude-dir=target --exclude-dir=.jj`

## Consumer context

No org-internal consumers found in the current scan (garb is a published standalone crate). The intended consumers per the Crate Index are `zenpixels-convert` and codec swizzle paths, but those are not currently taking a live dep.

## Summary

**0 items flagged for action.**

### Structure

garb's surface is almost entirely free functions:
- `garb::bytes` — 120+ `&[u8]` pixel-format conversion functions (contiguous + `_strided` variants for each)
- `garb::deinterleave` — planar ↔ interleaved f32 conversions, including chunk-level building blocks
- `garb::imgref` — whole-image typed conversions on `ImgVec`/`ImgRef` (feature-gated)
- `garb::typed_rgb` — type-safe conversions using `rgb` crate pixel types (feature-gated)
- `garb::SizeError` — single error type, `#[non_exhaustive]`, correct

### Observations (informational, no action needed)

1. **`garb::deinterleave::*_chunk{4,8,16}_scalar` functions** — e.g. `rgb_f32_chunk4_to_planes_scalar`, `planes_to_rgba_f32_chunk8_scalar`, etc. These are low-level building blocks documented as "scalar fallback chunk-level" routines for SIMD callers. They have full Rust docstrings and are NOT `#[doc(hidden)]`. They are intentionally pub — the doc explains "feeds directly into `vld3q_f32` hardware structure-load" and similar SIMD use cases. KEEP.

2. **`garb::deinterleave::rgb24_chunk8_to_planes_scalar` / `rgb48_chunk8_to_planes_scalar`** — These two ARE marked `#[doc(hidden)]` in the source (confirmed at line ~534), indicating they are internal/transient dispatch helpers. They would not appear in rendered docs. No action needed on our side (they already have `#[doc(hidden)]`).

3. **No leaked FFI, no sys bindings** — garb is pure safe Rust with no `unsafe` FFI shims in the public surface.

4. **No zencodec adapter** — garb is a low-level conversion library, not a codec; no streaming-decoder pattern to check.

## Flagged items

| # | Item | Category | Proposal | Confidence |
|---|------|----------|----------|------------|
| — | (none) | — | — | — |

**0 flagged. 0 % of surface.**

## Digest

garb's surface is a flat collection of conversion functions organized by calling convention (raw bytes, typed rgb, imgref). No struct fields, no trait leaks, no leaked internals. The deinterleave chunk helpers are documented building blocks, not accidental leaks. The two `rgb{24,48}_chunk8_to_planes_scalar` functions are already `#[doc(hidden)]`. Surface is intentional throughout.
