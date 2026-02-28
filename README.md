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
- Fill alpha (set alpha=255 across a 4bpp buffer)

All operations have `_strided` variants for images with padding between rows
(common in video frames and GPU textures).

## Performance

Benchmarked on 1920×1080 buffers (x86-64 AVX2, Zen 4). "Naive" is the
obvious `chunks_exact` + swap/copy loop — what the compiler autovectorizes
on its own.

| Operation | garb | naive | speedup | throughput |
|---|---|---|---|---|
| RGBA ↔ BGRA (in-place) | 81 µs | 644 µs | **8.0x** | 96 GiB/s |
| RGB ↔ BGR (in-place) | 208 µs | 823 µs | **4.0x** | 28 GiB/s |
| RGB ↔ BGR (copy) | 128 µs | 1,059 µs | **8.3x** | 45 GiB/s |
| RGBA → RGB (strip alpha) | 147 µs | 857 µs | **5.8x** | 52 GiB/s |
| BGRA → RGB (strip + swap) | 147 µs | 859 µs | **5.8x** | 52 GiB/s |
| RGB → RGBA (expand) | 146 µs | 1,056 µs | **7.2x** | 53 GiB/s |
| Fill alpha | 84 µs | 206 µs | **2.5x** | 92 GiB/s |

Run `cargo bench` to reproduce.

## Usage

The core API operates on `&[u8]` / `&mut [u8]` slices. Every function
returns `Result<(), SizeError>` — no panics, no silent truncation.

```rust
use garb::{rgba_to_bgra_inplace, rgb_to_bgra, SizeError};

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

For buffers where rows have padding (stride > width × bpp):

```rust
use garb::rgba_to_bgra_inplace_strided;

let mut buf = vec![0u8; 256 * 100]; // 60 pixels wide, stride=256, 100 rows
rgba_to_bgra_inplace_strided(&mut buf, 256, 60, 100)?;
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
| `std`    | yes     | Currently unused; reserved for future `std::error::Error` impl |
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

## Naming convention

Functions follow `{src}_to_{dst}` for copies, `{src}_to_{dst}_inplace` for
mutations. Symmetric swaps (like RGBA↔BGRA) provide both names as aliases.
Append `_strided` for padded row layouts.

## License

MIT OR Apache-2.0
