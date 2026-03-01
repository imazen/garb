# garb

*Dress your pixels for the occasion.*

You can't show up to a function in the wrong style. Get with the times and
swap your BGR for your RGB, your ARGB for your RGBA, your BGRX for your
BGRA, and tie up loose ends like that unreliable alpha.

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
- Gray → RGBA
- GrayAlpha → RGBA
- Fill alpha (set byte 3 = 255 in each 4-byte pixel, for RGBA/BGRA layouts)

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
| RGB ↔ BGR (copy) | 228 µs | 219 µs | ~1x |
| RGBA → RGB (strip alpha) | 435 µs | 278 µs | 0.6x |
| BGRA → RGB (strip + swap) | 435 µs | 278 µs | 0.6x |
| RGB → RGBA (expand) | 375 µs | 313 µs | 0.8x |
| Fill alpha | 242 µs | 495 µs | **2.0x** |

In-place swaps and fill are 2–3.6x faster on all ARM hardware tested
(Ampere Altra, Apple Silicon, Snapdragon X). Cross-bpp copies (3↔4 channel)
don't yet benefit from the hand-written NEON shuffles — the compiler's
autovectorizer matches or beats them. These will be improved in a future
release.

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

If you're already using the [`rgb`](https://crates.io/crates/rgb) crate,
`garb::typed_rgb` gives you typed wrappers. In-place swaps return
reinterpreted references — zero-copy, zero-allocation.

```rust
use rgb::{Rgba, Bgra};
use garb::typed_rgb;

let mut pixels: Vec<Rgba<u8>> = vec![Rgba::new(255, 0, 128, 255); 100];
let bgra: &mut [Bgra<u8>] = typed_rgb::rgba_to_bgra_mut(&mut pixels);
```

### Whole-image conversions (feature `imgref`)

`garb::imgref` handles strided `ImgVec` / `ImgRef` / `ImgRefMut` types
from the [`imgref`](https://crates.io/crates/imgref) crate. In-place swaps
consume and return the `ImgVec` with the buffer reinterpreted. Copy
conversions take `ImgRef` + `ImgRefMut` — you own the destination buffer.

```rust
use rgb::{Rgba, Bgra};
use imgref::ImgVec;
use garb::imgref;

let rgba_img = ImgVec::new(vec![Rgba::new(255, 0, 128, 200); 640 * 480], 640, 480);
let bgra_img: ImgVec<Bgra<u8>> = imgref::swap_rgba_to_bgra(rgba_img);
```

## Feature flags

| Feature  | Default | What it adds |
|----------|---------|--------------|
| `std`    | yes     | Enables `std` on dependencies (e.g. `archmage`) |
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

Aliases: `bgra_to_rgba_inplace`, `bgra_to_rgba`, `bgr_to_rgb_inplace`,
`bgr_to_rgb`, `gray_to_bgra`, `gray_alpha_to_bgra` (same underlying
operations).

### `typed_rgb` (feature `rgb`)

Typed wrappers using `rgb` crate pixel types. In-place swaps return
reinterpreted `&mut` references (zero-copy).

| Function | Signature |
|----------|-----------|
| `rgba_to_bgra_mut` | `&mut [Rgba<u8>]` → `&mut [Bgra<u8>]` |
| `bgra_to_rgba_mut` | `&mut [Bgra<u8>]` → `&mut [Rgba<u8>]` |
| `rgb_to_bgr_mut` | `&mut [Rgb<u8>]` → `&mut [Bgr<u8>]` |
| `bgr_to_rgb_mut` | `&mut [Bgr<u8>]` → `&mut [Rgb<u8>]` |
| `rgba_to_bgra_buf` | `&[Rgba<u8>]` + `&mut [Bgra<u8>]` |
| `bgra_to_rgba_buf` | `&[Bgra<u8>]` + `&mut [Rgba<u8>]` |
| `rgb_to_bgra_buf` | `&[Rgb<u8>]` + `&mut [Bgra<u8>]` |
| `rgb_to_rgba_buf` | `&[Rgb<u8>]` + `&mut [Rgba<u8>]` |
| `bgr_to_rgba_buf` | `&[Bgr<u8>]` + `&mut [Rgba<u8>]` |
| `bgr_to_bgra_buf` | `&[Bgr<u8>]` + `&mut [Bgra<u8>]` |
| `gray_to_rgba_buf` | `&[Gray<u8>]` + `&mut [Rgba<u8>]` |
| `gray_to_bgra_buf` | `&[Gray<u8>]` + `&mut [Bgra<u8>]` |
| `gray_alpha_to_rgba_buf` | `&[GrayAlpha<u8>]` + `&mut [Rgba<u8>]` |
| `gray_alpha_to_bgra_buf` | `&[GrayAlpha<u8>]` + `&mut [Bgra<u8>]` |
| `rgba_to_rgb_buf` | `&[Rgba<u8>]` + `&mut [Rgb<u8>]` |
| `bgra_to_bgr_buf` | `&[Bgra<u8>]` + `&mut [Bgr<u8>]` |
| `bgra_to_rgb_buf` | `&[Bgra<u8>]` + `&mut [Rgb<u8>]` |
| `rgba_to_bgr_buf` | `&[Rgba<u8>]` + `&mut [Bgr<u8>]` |
| `fill_alpha_rgba` | Set A=255 in `&mut [Rgba<u8>]` |
| `fill_alpha_bgra` | Set A=255 in `&mut [Bgra<u8>]` |

### `imgref` (feature `imgref`)

Whole-image conversions on `ImgVec` / `ImgRef` / `ImgRefMut`. In-place
swaps consume and return an `ImgVec` with the buffer reinterpreted.

| Function | Operation |
|----------|-----------|
| `swap_rgba_to_bgra` | `ImgVec<Rgba<u8>>` → `ImgVec<Bgra<u8>>` |
| `swap_bgra_to_rgba` | `ImgVec<Bgra<u8>>` → `ImgVec<Rgba<u8>>` |
| `convert_rgba_to_bgra` | `ImgRef<Rgba<u8>>` + `ImgRefMut<Bgra<u8>>` |
| `convert_bgra_to_rgba` | `ImgRef<Bgra<u8>>` + `ImgRefMut<Rgba<u8>>` |
| `convert_rgb_to_bgra` | `ImgRef<Rgb<u8>>` + `ImgRefMut<Bgra<u8>>` |
| `convert_rgb_to_rgba` | `ImgRef<Rgb<u8>>` + `ImgRefMut<Rgba<u8>>` |
| `convert_bgr_to_rgba` | `ImgRef<Bgr<u8>>` + `ImgRefMut<Rgba<u8>>` |
| `convert_bgr_to_bgra` | `ImgRef<Bgr<u8>>` + `ImgRefMut<Bgra<u8>>` |
| `convert_gray_to_rgba` | `ImgRef<Gray<u8>>` + `ImgRefMut<Rgba<u8>>` |
| `convert_gray_to_bgra` | `ImgRef<Gray<u8>>` + `ImgRefMut<Bgra<u8>>` |
| `convert_gray_alpha_to_rgba` | `ImgRef<GrayAlpha<u8>>` + `ImgRefMut<Rgba<u8>>` |
| `convert_gray_alpha_to_bgra` | `ImgRef<GrayAlpha<u8>>` + `ImgRefMut<Bgra<u8>>` |
| `convert_rgba_to_rgb` | `ImgRef<Rgba<u8>>` + `ImgRefMut<Rgb<u8>>` |
| `convert_bgra_to_rgb` | `ImgRef<Bgra<u8>>` + `ImgRefMut<Rgb<u8>>` |
| `convert_bgra_to_bgr` | `ImgRef<Bgra<u8>>` + `ImgRefMut<Bgr<u8>>` |
| `convert_rgba_to_bgr` | `ImgRef<Rgba<u8>>` + `ImgRefMut<Bgr<u8>>` |

## License

MIT OR Apache-2.0
