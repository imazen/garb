# garb benchmarks

This directory holds committed benchmark runs for garb's conversion kernels.
Every result is reproducible from this file alone: the environment, the exact
command, and the contenders are recorded below. No number here is hand-edited —
the `.log` files are raw harness output and the `.meta` files carry the commit,
host, date, and command for each run.

This directory is excluded from the published crate (`exclude` in `Cargo.toml`),
so it lives on GitHub only.

## What is measured, and against what

garb's SIMD path is always compared against the **naive** baseline — the obvious
`chunks_exact` loop that LLVM autovectorizes on its own. Both contenders run in
the same harness, on the same input, in the same process, so the delta isolates
the hand-written kernel from the autovectorizer. There is no external crate to
pin: the comparison is garb-SIMD vs compiler-autovec of the same operation.

| Bench | Harness | Covers |
|-------|---------|--------|
| `benches/swizzle.rs` | criterion 0.8 | RGBA↔BGRA, RGB↔BGR, channel expand/strip, fill-alpha — the core `garb::bytes` API. Source of the per-platform tables in the top-level README. |
| `benches/deinterleave.rs` | zenbench 0.1.7 | `garb::deinterleave` (`experimental`): packed RGB(A) ⇄ f32 planes, plus dispatch-cadence and chunk-size studies. |

## Environment

- **Local host** (the committed `.meta` files): AMD Ryzen 9 7950X (Zen 4,
  water-cooled, 128 GB DDR5), Linux.
- **Cross-platform** (`bench.yml` matrix): Linux x86_64 / aarch64, macOS
  aarch64 / x86_64, Windows x86_64 / aarch64, plus WASM SIMD128 under wasmtime
  and aarch64 under QEMU.
- **Build:** release/bench profile, **without** `-C target-cpu=native`. Runtime
  SIMD dispatch (archmage `cpuid` on x86-64, compile-time NEON/SIMD128 elsewhere)
  is what ships, so that is what is timed. The `bench.yml` workflow also runs a
  separate, clearly-labelled `target-cpu=native` job for reference; the headline
  numbers are from the non-native matrix.
- **Threading:** single-threaded. garb's kernels carry no internal threading;
  each call processes the buffer on the calling thread.
- **IO:** excluded. Both harnesses allocate and fill input buffers before the
  timed region and convert into a pre-allocated output, so no allocation or IO
  is inside the measured loop.

## Reproduce

```sh
git clone https://github.com/imazen/garb && cd garb
git checkout <commit>      # see the .meta file for the commit each run used

# Core swizzle/expand/strip benchmarks (the README per-platform tables):
cargo bench --bench swizzle -- --noplot

# Deinterleave (experimental) benchmarks:
cargo bench --bench deinterleave --features experimental
```

For a specific zenbench group, append `-- --group="<name>"` (e.g.
`--group="rgb24_to_planes_f32"`), as recorded in the relevant `.meta`.

## Committed runs

| Files | Question it answers |
|-------|---------------------|
| `deinterleave_2026-04-29.*` | Baseline deinterleave sweep, 256 px → 16 MP across cache tiers. |
| `deinterleave_f32_2026-04-29.*`, `deinterleave_rgba_f32_2026-04-29.log` | f32 RGB/RGBA plane (de)interleave. |
| `deinterleave_dispatch_cadence_2026-04-29.*` | Per-call `#[arcane]` dispatch overhead vs buffer size (breakeven point). |
| `deinterleave_2026-04-29_with_autovec.*` | Adds the autovec baseline column to the deinterleave sweep. |
| `rgb24_chunk_vs_autovec_2026-05-07.*`, `rgb48_chunk_vs_autovec_2026-05-07.*` | Does the u8/u16 hand-written chunk SIMD beat autovec? (Yes, at L1–L3 sizes.) |
| `deinterleave_autovec_vs_chunk_2026-05-07.*` | Does the f32 128-bit-XMM chunk SIMD beat `#[arcane]` autovec? (No — autovec wins; f32 hand-chunks were dropped.) |
| `deinterleave_chunk_size_choice_2026-05-07.*` | chunk4 vs chunk8 vs chunk16 sizing. |

## Reading caveats

- The zenbench `iB/s` throughput column in these runs is miscalibrated (orders
  of magnitude too large). Trust the `ns` timings and compute bytes/sec by hand.
- At ~4 MP and above, RGB24/RGB48 deinterleave is DRAM-write-bandwidth-bound;
  the SIMD compute advantage is only visible while the working set fits in L3
  (roughly ≤ 1 MP). The per-size `.log` rows show exactly where the cliff lands.
- Deinterleave runs converged at zenbench's "noisy" stop (27–32% CV on some
  sizes); treat small-size deltas that cross zero as ties.
