# garb [![CI](https://img.shields.io/github/actions/workflow/status/imazen/garb/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/garb/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/garb?style=flat-square)](https://crates.io/crates/garb) [![lib.rs](https://img.shields.io/crates/v/garb?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/garb) [![docs.rs](https://img.shields.io/docsrs/garb?style=flat-square)](https://docs.rs/garb) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/garb?style=flat-square)](#license) [![codecov](https://img.shields.io/codecov/c/github/imazen/garb?style=flat-square)](https://codecov.io/gh/imazen/garb)

*Dress your pixels for the occasion.*

You can't show up to a function in the wrong style. Swap your BGR for your RGB, your ARGB for your RGBA, and tie up loose
ends like that unreliable alpha BGRX.

SIMD-accelerated pixel format conversions: x86-64 AVX2, ARM NEON, WASM SIMD128,
with automatic scalar fallback. `no_std` compatible, `#![forbid(unsafe_code)]`.

## Quick start

```toml
[dependencies]
garb = "0.3.0"
```

```rust
use garb::bytes::{rgba_to_bgra_inplace, rgb_to_bgra};

// In place: swap R↔B across a 4-bytes-per-pixel buffer (RGBA → BGRA).
let mut pixels = vec![255u8, 0, 128, 255];
rgba_to_bgra_inplace(&mut pixels)?;
assert_eq!(pixels, [128, 0, 255, 255]);

// Copy + expand: RGB (3 bpp) → BGRA (4 bpp), alpha filled to 255.
let rgb = [255u8, 0, 128];
let mut bgra = [0u8; 4];
rgb_to_bgra(&rgb, &mut bgra)?;
assert_eq!(bgra, [128, 0, 255, 255]);
# Ok::<(), garb::SizeError>(())
```

Every function returns `Result<(), SizeError>` — no panics, no silent truncation —
and ships a `_strided` companion for padded row layouts. Enable the `rgb` /
`imgref` features to let the compiler pick the conversion from typed pixel slices
(see [Usage](#usage)). For the full operation list, see the
[Function reference](#function-reference).

## What it does

Converts between pixel layouts at the byte-slice level. Every image decoder
and renderer has an opinion about channel order and pixel width, and none of
them agree. garb handles the mechanical part — swapping, expanding, and
stripping channels — so you can get back to the interesting work.

**SIMD-optimized (contiguous and strided)**
- RGBA ↔ BGRA (in-place and copy)
- RGB ↔ BGR (in-place and copy)
- RGB → RGBA / BGRA
- BGR → BGRA / RGBA
- RGBA / BGRA → RGB / BGR (drop alpha)
- Gray → RGBA / BGRA
- GrayAlpha → RGBA / BGRA
- Fill alpha (set byte 3 = 255 in each 4-byte pixel, for RGBA/BGRA layouts)
- ARGB ↔ RGBA / BGRA / ABGR (in-place and copy)
- RGB → ARGB / ABGR
- BGR → ARGB / ABGR
- ARGB / ABGR → RGB / BGR (drop alpha)
- Gray → ARGB / ABGR
- GrayAlpha → ARGB / ABGR
- Fill alpha (set byte 0 = 255 in each 4-byte pixel, for ARGB/ABGR/XRGB/XBGR layouts)
- Gray → RGB / BGR, GrayAlpha → RGB / BGR, Gray ↔ GrayAlpha
- RGB / RGBA / BGR / BGRA → Gray (weighted integer luma: BT.709, BT.601, BT.2020; or identity)
- Integer depth conversion: u8 ↔ u16 (exact `v * 257` and `(v + 128) / 257`)
- u8 alpha premultiply for RGBA / BGRA (exact integer `(c * a + 127) / 255`, in-place and copy)

**Experimental** (feature `experimental` — API may change)
- RGB565 → RGBA / BGRA (little-endian packed 16-bit, auto-vectorized)
- RGBA / BGRA → RGB565 (lossy compress, round-to-nearest, auto-vectorized)
- RGBA4444 → RGBA / BGRA (little-endian packed 16-bit, auto-vectorized)
- RGBA / BGRA → RGBA4444 (lossy compress, round-to-nearest, auto-vectorized)
- RGBA1010102 ↔ interleaved RGBA u16 (LE packed `r | g<<10 | b<<20 | a<<30`,
  matches DXGI `R10G10B10A2_UNORM` / Vulkan `A2B10G10R10_UNORM_PACK32` /
  WGPU `Rgb10a2Unorm`; 2-bit alpha bit-replicated to 10 bits per the graphics-API
  convention; transfer functions are NOT applied — chain with `linear-srgb` for PQ/HLG)

All operations have `_strided` variants for images with padding between rows
(common in video frames and GPU textures).

## Performance

Hand-written SIMD beats the naive autovectorized `chunks_exact` loop by up to
**7×** on x86-64 (AVX2), **3.6×** on aarch64 (NEON), and **8×** on WASM
(SIMD128), measured on 1920×1080 buffers. Run `cargo bench` for hardware-specific
numbers; full per-platform tables, methodology, and reproduction commands are in
[`benchmarks/`](https://github.com/imazen/garb/tree/main/benchmarks).

<!-- crates.io:skip-start -->
All benchmarks on 1920×1080 buffers. "Naive" is the obvious `chunks_exact`
loop — what the compiler autovectorizes on its own. Numbers from GitHub
Actions CI (runtime SIMD dispatch, built without `-C target-cpu=native`); run
`cargo bench` locally for hardware-specific results.

### x86-64 (AVX2) — Linux, Zen 4

| Operation | garb | naive | speedup |
|---|---|---|---|
| RGBA ↔ BGRA (in-place) | 150 µs | 1,078 µs | **7.2x** |
| RGB ↔ BGR (in-place) | 329 µs | 1,038 µs | **3.2x** |
| RGB ↔ BGR (copy) | 209 µs | 1,509 µs | **7.2x** |
| RGBA → RGB (strip alpha) | 255 µs | 1,556 µs | **6.1x** |
| BGRA → RGB (strip + swap) | 260 µs | 1,556 µs | **6.0x** |
| RGB → RGBA (expand) | 328 µs | 1,764 µs | **5.4x** |
| Fill alpha | 138 µs | 329 µs | **2.4x** |

### aarch64 (NEON) — Linux, Ampere Altra

| Operation | garb | naive | speedup |
|---|---|---|---|
| RGBA ↔ BGRA (in-place) | 243 µs | 865 µs | **3.6x** |
| RGB ↔ BGR (in-place) | 369 µs | 857 µs | **2.3x** |
| Fill alpha | 242 µs | 495 µs | **2.0x** |
| RGB ↔ BGR (copy) | 221 µs | 219 µs | ~1x |
| RGBA → RGB (strip alpha) | 279 µs | 278 µs | ~1x |
| BGRA → RGB (strip + swap) | 277 µs | 278 µs | ~1x |
| RGB → RGBA (expand) | 316 µs | 313 µs | ~1x |

In-place swaps and fill use hand-written NEON and are 2–3.6x faster on all
ARM hardware tested (Ampere Altra, Apple Silicon, Snapdragon X). Cross-bpp
operations (3↔4 channel, 3bpp copy) use LLVM's autovectorizer, which
generates optimal code for these patterns on AArch64.

### WASM (SIMD128) — wasmtime

| Operation | garb | naive | speedup |
|---|---|---|---|
| RGBA ↔ BGRA (in-place) | 230 µs | 1,041 µs | **4.5x** |
| RGB ↔ BGR (in-place) | 494 µs | 1,027 µs | **2.1x** |
| RGB ↔ BGR (copy) | 333 µs | 2,753 µs | **8.3x** |
| RGBA → RGB (strip alpha) | 506 µs | 1,623 µs | **3.2x** |
| BGRA → RGB (strip + swap) | 659 µs | 2,309 µs | **3.5x** |
| RGB → RGBA (expand) | 998 µs | 2,271 µs | **2.3x** |
| Fill alpha | 193 µs | 650 µs | **3.4x** |

Full benchmark results for all six native platforms plus WASM are available
in the [CI artifacts](https://github.com/imazen/garb/actions/workflows/bench.yml).
Run `cargo bench` to reproduce locally.
<!-- crates.io:skip-end -->

## Usage

The core `&[u8]` / `&mut [u8]` API lives in [`garb::bytes`](https://docs.rs/garb/latest/garb/bytes/).
Every function returns `Result<(), SizeError>` — no panics, no silent truncation.

```rust
use garb::bytes::{rgba_to_bgra_inplace, rgb_to_bgra};
use garb::SizeError;

// In-place: swap R↔B in a 4bpp buffer
let mut pixels = vec![255u8, 0, 128, 255,  0, 200, 100, 255];
rgba_to_bgra_inplace(&mut pixels)?;
assert_eq!(pixels, [128, 0, 255, 255,  100, 200, 0, 255]);

// Copy: RGB (3 bpp) → BGRA (4 bpp), alpha filled to 255
let rgb = vec![255u8, 0, 128];
let mut bgra = vec![0u8; 4];
rgb_to_bgra(&rgb, &mut bgra)?;
assert_eq!(bgra, [128, 0, 255, 255]);

// ARGB → RGBA: rotate bytes left [A,R,G,B] → [R,G,B,A]
let mut argb = vec![255u8, 128, 0, 64];
garb::bytes::argb_to_rgba_inplace(&mut argb)?;
assert_eq!(argb, [128, 0, 64, 255]);
# Ok::<(), SizeError>(())
```

### Strided images

A **stride** is the distance between the start of one row and the start of
the next, measured in units of the slice's element type. For the core `&[u8]`
API that means bytes; for the typed `imgref` API it means elements of the
slice's item type (e.g. pixel count for `ImgRef<Rgba<u8>>`). When
`stride > width` the gap is padding — garb never reads or writes it.

All `_strided` functions take dimensions before strides:
- In-place: `(buf, width, height, stride)`
- Copy: `(src, dst, width, height, src_stride, dst_stride)`

```rust
use garb::bytes::{rgba_to_bgra_inplace_strided, rgb_to_bgra_strided};

// In-place: 60 pixels wide, stride=256 bytes, 100 rows
let mut buf = vec![0u8; 256 * 100];
rgba_to_bgra_inplace_strided(&mut buf, 60, 100, 256)?;

// Copy with different strides: RGB (stride=192) → BGRA (stride=256)
let rgb_buf = vec![0u8; 192 * 100];
let mut bgra_buf = vec![0u8; 256 * 100];
rgb_to_bgra_strided(&rgb_buf, &mut bgra_buf, 60, 100, 192, 256)?;
# Ok::<(), garb::SizeError>(())
```

### Type-safe conversions (feature `rgb`)

With the [`rgb`](https://crates.io/crates/rgb) crate, use `garb::convert`
and `garb::convert_inplace` with typed pixel slices. The right conversion is
selected at compile time from the src/dst types — no need to remember
function names. In-place swaps return reinterpreted references (zero-copy).

```rust
use rgb::{Rgba, Bgra, Rgb};
use garb::{convert, convert_inplace};

// In-place: type-inferred from the return binding
let mut pixels: Vec<Rgba<u8>> = vec![Rgba::new(255, 0, 128, 255); 100];
let bgra: &mut [Bgra<u8>] = convert_inplace(&mut pixels);

// Copy: type-inferred from src and dst
let rgb = vec![Rgb::new(255u8, 0, 128); 100];
let mut bgra = vec![Bgra::default(); 100];
convert(&rgb, &mut bgra).unwrap();
```

### Whole-image conversions (feature `imgref`)

`garb::convert_imgref` and `garb::convert_imgref_inplace` handle strided
`ImgVec` / `ImgRef` / `ImgRefMut` types from the
[`imgref`](https://crates.io/crates/imgref) crate. In-place conversions
consume and return the `ImgVec` with the buffer reinterpreted. Copy
conversions take `ImgRef` + `ImgRefMut` — you own the destination buffer.

```rust
use rgb::{Rgba, Bgra};
use imgref::ImgVec;
use garb::convert_imgref_inplace;

let rgba_img = ImgVec::new(vec![Rgba::new(255, 0, 128, 200); 640 * 480], 640, 480);
let bgra_img: ImgVec<Bgra<u8>> = convert_imgref_inplace(rgba_img);
```

## Feature flags

| Feature  | Default | What it adds |
|----------|---------|--------------|
| `std`    | yes     | Enables `std` on dependencies (e.g. `archmage`) |
| `experimental` | no | Packed integer formats (`RGB565`, `RGBA4444`, `RGBA1010102` — API may change) |
| `rgb`    | no      | `garb::typed_rgb` — conversions on `Rgba<u8>`, `Bgra<u8>`, etc. (enables `bytemuck`) |
| `imgref` | no      | `garb::imgref` — whole-image conversions on `ImgVec` / `ImgRef` (implies `rgb` + `alloc`) |

The crate is `no_std` by default — the core byte-slice API, plus the
`experimental` and `rgb` surfaces, need no allocator (they operate on
caller-owned slices). Only the `imgref` feature pulls in `alloc`.

## Precision

`garb` 0.3.0 is strictly integer-only (floating-point depth and alpha conversions
live in `zenpixels-convert` on `magetypes`). Integer narrowing rounds to nearest:
`u16 → u8` uses exact `(v + 128) / 257`, `u8 → u16` uses exact `v * 257`, and
`u8` alpha premultiply uses exact `(c * a + 127) / 255` across scalar, AVX2, and
NEON. See the [precision review](https://github.com/imazen/garb/blob/main/docs/precision.md).

## SIMD dispatch

garb uses [archmage](https://crates.io/crates/archmage) for runtime SIMD
detection with compile-time acceleration. On each platform:

- **x86-64**: AVX2 (checked at runtime via `cpuid`)
- **aarch64**: NEON (compile-time guaranteed on AArch64)
- **wasm32**: SIMD128 (compile-time via `target-feature=+simd128`)
- **Fallback**: Scalar code on all platforms, always available

The first call to each function detects and caches the best available tier.
There's no setup, no feature flags to configure, and no `unsafe` — archmage
handles it all behind safe token types.

## Function reference

Functions follow `{src}_to_{dst}` for copies, `{src}_to_{dst}_inplace` for
mutations. Symmetric swaps (like RGBA↔BGRA) provide both names as aliases.
Append `_strided` for padded row layouts.

### Core API — `garb::bytes` (`&[u8]`)

Every function returns `Result<(), SizeError>`. All have `_strided` variants.

| Function | Operation |
|----------|-----------|
| `rgba_to_bgra_inplace` | Swap R↔B in 4bpp buffer (RGBA↔BGRA) |
| `rgba_to_bgra` | Copy 4bpp, swapping R↔B |
| `rgb_to_bgr_inplace` | Swap R↔B in 3bpp buffer (RGB↔BGR) |
| `rgb_to_bgr` | Copy 3bpp, swapping R↔B |
| `rgb_to_rgba` | 3bpp → 4bpp, alpha = 255 |
| `rgb_to_bgra` | 3bpp → 4bpp, swap R↔B, alpha = 255 |
| `bgr_to_rgba` | 3bpp → 4bpp, swap R↔B, alpha = 255 |
| `bgr_to_bgra` | 3bpp → 4bpp, alpha = 255 |
| `rgba_to_rgb` | 4bpp → 3bpp, drop alpha |
| `bgra_to_rgb` | 4bpp → 3bpp, swap R↔B, drop alpha |
| `bgra_to_bgr` | 4bpp → 3bpp, drop alpha |
| `rgba_to_bgr` | 4bpp → 3bpp, swap R↔B, drop alpha |
| `gray_to_rgba` | 1bpp → 4bpp (R=G=B=gray, A=255) |
| `gray_alpha_to_rgba` | 2bpp → 4bpp (R=G=B=gray, A=alpha) |
| `fill_alpha_rgba` | Set byte 3 to 255 in each 4-byte pixel (alpha-last: RGBA/BGRA) |
| `argb_to_rgba_inplace` | Rotate bytes left in 4bpp buffer: \[A,R,G,B\]→\[R,G,B,A\] |
| `argb_to_rgba` | Copy 4bpp, rotating bytes left by 1 (ARGB→RGBA) |
| `rgba_to_argb_inplace` | Rotate bytes right in 4bpp buffer: \[R,G,B,A\]→\[A,R,G,B\] |
| `rgba_to_argb` | Copy 4bpp, rotating bytes right by 1 (RGBA→ARGB) |
| `argb_to_bgra_inplace` | Reverse each pixel's 4 bytes: \[A,R,G,B\]→\[B,G,R,A\] |
| `argb_to_bgra` | Copy 4bpp, reversing byte order (ARGB→BGRA) |
| `fill_alpha_argb` | Set byte 0 to 255 in each 4-byte pixel (alpha-first: ARGB/ABGR) |
| `rgb_to_argb` | 3bpp → 4bpp, alpha=255 prepended |
| `rgb_to_abgr` | 3bpp → 4bpp, channels reversed, alpha=255 prepended |
| `argb_to_rgb` | 4bpp → 3bpp, drop leading alpha |
| `argb_to_bgr` | 4bpp → 3bpp, drop alpha + reverse channels |
| `gray_to_argb` | 1bpp → 4bpp (A=255, R=G=B=gray) |
| `gray_alpha_to_argb` | 2bpp → 4bpp (alpha first, R=G=B=gray) |
| `gray_to_rgb` | 1bpp → 3bpp (R=G=B=gray) |
| `gray_alpha_to_rgb` | 2bpp → 3bpp (R=G=B=gray, drop alpha) |
| `gray_to_gray_alpha` | 1bpp → 2bpp (A=255) |
| `gray_alpha_to_gray` | 2bpp → 1bpp (drop alpha) |
| `rgb_to_gray` | 3bpp → 1bpp weighted luma (BT.709 default) |
| `rgba_to_gray` | 4bpp → 1bpp weighted luma (BT.709 default) |
| `rgb_to_gray_bt709` | 3bpp → 1bpp BT.709 luma (also `_bt601`, `_bt2020`) |
| `rgba_to_gray_bt601` | 4bpp → 1bpp BT.601 luma (also `_bt709`, `_bt2020`) |
| `rgb_to_gray_identity` | 3bpp → 1bpp, take first channel (for R=G=B data) |
| `rgba_to_gray_identity` | 4bpp → 1bpp, take first channel (for R=G=B data) |
| `convert_u8_to_u16` | Depth: u8 → u16 (0–255 → 0–65535, LE u16 bytes) |
| `convert_u16_to_u8` | Depth: u16 → u8 (exact `(v + 128) / 257`, LE u16 bytes) |
| `premultiply_alpha_rgba_u8` | Premultiply alpha in `[R,G,B,A]` u8 buffer (in-place) |
| `premultiply_alpha_rgba_u8_copy` | Premultiply alpha u8, copy variant |

Aliases: `bgra_to_rgba_inplace`, `bgra_to_rgba`, `bgr_to_rgb_inplace`,
`bgr_to_rgb`, `gray_to_bgra`, `gray_alpha_to_bgra`,
`abgr_to_bgra_inplace`, `abgr_to_bgra`, `bgra_to_abgr_inplace`, `bgra_to_abgr`,
`bgra_to_argb_inplace`, `bgra_to_argb`, `abgr_to_rgba_inplace`, `abgr_to_rgba`,
`rgba_to_abgr_inplace`, `rgba_to_abgr`,
`fill_alpha_abgr`, `fill_alpha_xrgb`, `fill_alpha_xbgr`,
`bgr_to_argb`, `bgr_to_abgr`, `abgr_to_bgr`, `abgr_to_rgb`,
`gray_to_abgr`, `gray_alpha_to_abgr`,
`gray_to_bgr`, `gray_alpha_to_bgr`,
`bgr_to_gray`, `bgra_to_gray`, plus `_bt601`, `_bt709`, `_bt2020`, and `_identity` variants,
`premultiply_alpha_bgra_u8`, `premultiply_alpha_bgra_u8_copy`.

#### Experimental (`feature = "experimental"`)

| Function | Operation |
|----------|-----------|
| `rgb565_to_rgba` | RGB565 (LE u16, 2bpp) → RGBA (4bpp), A=255 |
| `rgb565_to_bgra` | RGB565 (LE u16, 2bpp) → BGRA (4bpp), A=255 |
| `rgba_to_rgb565` | RGBA (4bpp) → RGB565 (LE u16, 2bpp), lossy, alpha dropped |
| `bgra_to_rgb565` | BGRA (4bpp) → RGB565 (LE u16, 2bpp), lossy, alpha dropped |
| `rgba4444_to_rgba` | RGBA4444 (LE u16, 2bpp) → RGBA (4bpp) |
| `rgba4444_to_bgra` | RGBA4444 (LE u16, 2bpp) → BGRA (4bpp) |
| `rgba_to_rgba4444` | RGBA (4bpp) → RGBA4444 (LE u16, 2bpp), lossy |
| `bgra_to_rgba4444` | BGRA (4bpp) → RGBA4444 (LE u16, 2bpp), lossy |
| `rgba1010102_to_rgba16` | RGBA1010102 (LE u32, 4bpp) → interleaved RGBA u16 (4 channels in `[0, 1023]`); 2-bit alpha bit-replicated to 10 bits |
| `rgba16_to_rgba1010102` | Interleaved RGBA u16 → RGBA1010102 (LE u32, 4bpp); alpha rounded to 2 bits via `(a*3+511)/1023` |

All experimental functions also have `_strided` variants.

### Generic API — `convert` / `convert_inplace` (feature `rgb`)

Type-inferred conversions on `rgb` crate pixel slices. The compiler selects
the right SIMD-optimized conversion from the source and destination types.

| Function | Description |
|----------|-------------|
| `convert(&[S], &mut [D])` | Copy-convert between any supported pixel types |
| `convert_inplace(&mut [S]) -> &mut [D]` | In-place swap for same-size types (zero-copy) |
| `typed_rgb::fill_alpha_rgba` | Set A=255 in `&mut [Rgba<u8>]` |
| `typed_rgb::fill_alpha_bgra` | Set A=255 in `&mut [Bgra<u8>]` |

**`convert_inplace` pairs** (same-size, zero-copy):

| From | To |
|------|----|
| `Rgba<u8>` | `Bgra<u8>` |
| `Bgra<u8>` | `Rgba<u8>` |
| `Rgb<u8>` | `Bgr<u8>` |
| `Bgr<u8>` | `Rgb<u8>` |

**`convert` pairs** (copy):

| From | To |
|------|----|
| `Rgba<u8>` | `Bgra<u8>`, `Rgb<u8>`, `Bgr<u8>`, `Gray<u8>` (identity) |
| `Bgra<u8>` | `Rgba<u8>`, `Bgr<u8>`, `Rgb<u8>`, `Gray<u8>` (identity) |
| `Rgb<u8>` | `Rgba<u8>`, `Bgra<u8>`, `Gray<u8>` (identity) |
| `Bgr<u8>` | `Rgba<u8>`, `Bgra<u8>`, `Gray<u8>` (identity) |
| `Gray<u8>` | `Rgba<u8>`, `Bgra<u8>`, `Rgb<u8>`, `Bgr<u8>`, `GrayAlpha<u8>` |
| `GrayAlpha<u8>` | `Rgba<u8>`, `Bgra<u8>`, `Rgb<u8>`, `Bgr<u8>`, `Gray<u8>` |

Weighted luma conversions (`rgb_to_gray_bt709_buf`, etc.) and `u8` alpha
premultiply (`premultiply_rgba_u8`, `premultiply_rgba_u8_buf`, etc.) remain as
named functions.

ARGB/ABGR types are not in the `rgb` crate, so those conversions are only
available through `garb::bytes`.

The previous named functions (`rgba_to_bgra_mut`, `rgb_to_bgra_buf`, etc.)
are deprecated but still available.

### Generic API — `convert_imgref` / `convert_imgref_inplace` (feature `imgref`)

Type-inferred conversions on `ImgVec` / `ImgRef` / `ImgRefMut` from the
[`imgref`](https://crates.io/crates/imgref) crate. Same type pairs as above,
plus named weighted-luma and `u8` alpha-premultiply helpers.

| Function | Description |
|----------|-------------|
| `convert_imgref(ImgRef<S>, ImgRefMut<D>)` | Copy-convert between images |
| `convert_imgref_inplace(ImgVec<S>) -> ImgVec<D>` | In-place, returns reinterpreted image |

The previous named functions (`swap_rgba_to_bgra`, `convert_rgb_to_bgra`, etc.)
are deprecated but still available.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/imazen/garb/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/imazen/garb/blob/main/LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · **garb** · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
