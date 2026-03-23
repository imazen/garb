# garb

*Dress your pixels for the occasion.*

You can't show up to a function in the wrong style. Swap your BGR for your RGB, your ARGB for your RGBA, and tie up loose
ends like that unreliable alpha BGRX.

SIMD-accelerated pixel format conversions: x86-64 AVX2, ARM NEON, WASM SIMD128,
with automatic scalar fallback. `no_std` compatible, `#![forbid(unsafe_code)]`.

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

**Experimental** (feature `experimental` — API may change)
- RGB565 → RGBA / BGRA (little-endian packed 16-bit, auto-vectorized)
- RGBA4444 → RGBA / BGRA (little-endian packed 16-bit, auto-vectorized)
- Gray → RGB, GrayAlpha → RGB, Gray ↔ GrayAlpha
- RGB / RGBA / BGR / BGRA → Gray (weighted luma: BT.709, BT.601, BT.2020; or identity)
- Depth conversion: u8 ↔ u16, u8 ↔ f32, u16 ↔ f32
- F32 alpha premultiply / unpremultiply (in-place and copy, AVX2 SIMD)

All operations have `_strided` variants for images with padding between rows
(common in video frames and GPU textures).

## Performance

All benchmarks on 1920×1080 buffers. "Naive" is the obvious `chunks_exact`
loop — what the compiler autovectorizes on its own. Numbers from GitHub
Actions CI; run `cargo bench` locally for hardware-specific results.

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
| `experimental` | no | Packed formats (RGB565, RGBA4444), gray layout, weighted luma, depth conversion, f32 premul (API may change) |
| `rgb`    | no      | `garb::typed_rgb` — conversions on `Rgba<u8>`, `Bgra<u8>`, etc. |
| `imgref` | no      | `garb::imgref` — whole-image conversions on `ImgVec` / `ImgRef` (implies `rgb`) |

The crate is `no_std + alloc` by default. The `imgref` feature requires `alloc`.
The core byte-slice API has no allocator requirement.

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

Aliases: `bgra_to_rgba_inplace`, `bgra_to_rgba`, `bgr_to_rgb_inplace`,
`bgr_to_rgb`, `gray_to_bgra`, `gray_alpha_to_bgra`,
`abgr_to_bgra_inplace`, `abgr_to_bgra`, `bgra_to_abgr_inplace`, `bgra_to_abgr`,
`bgra_to_argb_inplace`, `bgra_to_argb`, `abgr_to_rgba_inplace`, `abgr_to_rgba`,
`rgba_to_abgr_inplace`, `rgba_to_abgr`,
`fill_alpha_abgr`, `fill_alpha_xrgb`, `fill_alpha_xbgr`,
`bgr_to_argb`, `bgr_to_abgr`, `abgr_to_bgr`, `abgr_to_rgb`,
`gray_to_abgr`, `gray_alpha_to_abgr`.

#### Experimental (`feature = "experimental"`)

| Function | Operation |
|----------|-----------|
| `rgb565_to_rgba` | RGB565 (LE u16, 2bpp) → RGBA (4bpp), A=255 |
| `rgb565_to_bgra` | RGB565 (LE u16, 2bpp) → BGRA (4bpp), A=255 |
| `rgba4444_to_rgba` | RGBA4444 (LE u16, 2bpp) → RGBA (4bpp) |
| `rgba4444_to_bgra` | RGBA4444 (LE u16, 2bpp) → BGRA (4bpp) |
| `gray_to_rgb` | 1bpp → 3bpp (R=G=B=gray) |
| `gray_alpha_to_rgb` | 2bpp → 3bpp (R=G=B=gray, drop alpha) |
| `gray_to_gray_alpha` | 1bpp → 2bpp (A=255) |
| `gray_alpha_to_gray` | 2bpp → 1bpp (drop alpha) |
| `rgb_to_gray` | 3bpp → 1bpp weighted luma (BT.709 default) |
| `rgba_to_gray` | 4bpp → 1bpp weighted luma (BT.709 default) |
| `rgb_to_gray_bt709` | 3bpp → 1bpp BT.709 luma |
| `rgba_to_gray_bt601` | 4bpp → 1bpp BT.601 luma (also `_bt709`, `_bt2020`) |
| `rgb_to_gray_identity` | 3bpp → 1bpp, take first channel (for R=G=B data) |
| `rgba_to_gray_identity` | 4bpp → 1bpp, take first channel (for R=G=B data) |
| `convert_u8_to_u16` | Depth: u8 → u16 (0–255 → 0–65535) |
| `convert_u16_to_u8` | Depth: u16 → u8 |
| `convert_u8_to_f32` | Depth: u8 → f32 (0–255 → 0.0–1.0) |
| `convert_f32_to_u8` | Depth: f32 → u8 (clamped) |
| `convert_u16_to_f32` | Depth: u16 → f32 (0–65535 → 0.0–1.0) |
| `convert_f32_to_u16` | Depth: f32 → u16 (clamped) |
| `premultiply_alpha_f32` | Premultiply alpha in `[R,G,B,A]` f32 buffer (in-place) |
| `unpremultiply_alpha_f32` | Unpremultiply alpha in `[R,G,B,A]` f32 buffer (in-place) |
| `premultiply_alpha_f32_copy` | Premultiply alpha, copy variant |
| `unpremultiply_alpha_f32_copy` | Unpremultiply alpha, copy variant |

Aliases: `bgr_to_gray`, `bgra_to_gray`, plus `_identity` and BGR variants
for all gray conversions.

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
| `Rgba<u8>` | `Bgra<u8>`, `Rgb<u8>`, `Bgr<u8>` |
| `Bgra<u8>` | `Rgba<u8>`, `Bgr<u8>`, `Rgb<u8>` |
| `Rgb<u8>` | `Rgba<u8>`, `Bgra<u8>` |
| `Bgr<u8>` | `Rgba<u8>`, `Bgra<u8>` |
| `Gray<u8>` | `Rgba<u8>`, `Bgra<u8>` |
| `GrayAlpha<u8>` | `Rgba<u8>`, `Bgra<u8>` |

**Additional pairs with `experimental`:**

| From | To |
|------|----|
| `Gray<u8>` | `Rgb<u8>`, `Bgr<u8>`, `GrayAlpha<u8>` |
| `GrayAlpha<u8>` | `Rgb<u8>`, `Bgr<u8>`, `Gray<u8>` |
| `Rgb<u8>` | `Gray<u8>` (identity) |
| `Rgba<u8>` | `Gray<u8>` (identity) |
| `Bgr<u8>` | `Gray<u8>` (identity) |
| `Bgra<u8>` | `Gray<u8>` (identity) |

Weighted luma conversions (`rgb_to_gray_bt709_buf`, etc.) and
`premultiply_rgba_f32` / `unpremultiply_rgba_f32` remain as named functions.

ARGB/ABGR types are not in the `rgb` crate, so those conversions are only
available through `garb::bytes`.

The previous named functions (`rgba_to_bgra_mut`, `rgb_to_bgra_buf`, etc.)
are deprecated but still available.

### Generic API — `convert_imgref` / `convert_imgref_inplace` (feature `imgref`)

Type-inferred conversions on `ImgVec` / `ImgRef` / `ImgRefMut` from the
[`imgref`](https://crates.io/crates/imgref) crate. Same type pairs as above.

| Function | Description |
|----------|-------------|
| `convert_imgref(ImgRef<S>, ImgRefMut<D>)` | Copy-convert between images |
| `convert_imgref_inplace(ImgVec<S>) -> ImgVec<D>` | In-place, returns reinterpreted image |

With `experimental`: additional pairs plus weighted luma and premultiply.

The previous named functions (`swap_rgba_to_bgra`, `convert_rgb_to_bgra`, etc.)
are deprecated but still available.

## License

MIT OR Apache-2.0
