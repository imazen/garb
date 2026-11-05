# SIMD boundaries and outstanding work

Observed 2026-11-05 against main `7ab73292`. Re-check branch and issue state before acting.

## API guidance

Ordinary callers use the dispatched functions in `garb::bytes` or the bulk `garb::deinterleave` API. The experimental chunk functions already use arrays, not public archmage/magetypes types. No additive renaming or deprecation was needed here.

`rgb24_chunk8_to_planes_tokenless_v3` accepts `&[u8; 24]`; the RGB48 counterpart accepts `&[u16; 24]`. Both return three `[f32; 8]` arrays. **Neither accepts a token.** On native x86, call inside a matching V3 feature function. A runtime CPU check alone does not establish the compiler context; put the call inside a feature function rather than adding an unsafe block. V3 requires `sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe` in the current archmage tier. Scalar chunk functions require no CPU feature context. Do not infer shipping NEON/WASM chunk APIs from historical naming comments.

Returned RGB24 values are widened `0..255`; RGB48 values are `0..65535`. Normalize before feeding a transfer function that expects normalized sRGB samples. With linear-srgb's proposed [array API](https://github.com/imazen/linear-srgb/pull/30), keep tokens and vector types internal to each dependency and exchange arrays in the surrounding feature function. That API is not in linear-srgb's crates.io 0.6.12 release. Never relabel integer code values as normalized or linear light.

The chunk functions are `#[doc(hidden)]` experimental hooks. Preserve their current signatures; this documentation change does not promote them into a new stability promise. Inlining and register passing depend on the compiled caller: inspect assembly and benchmark before making a new cost claim.

## Branch and PR inventory

| Ref / tip | State and next action |
| --- | --- |
| `main` / `7ab73292` | Current baseline, including rounding corrections and published archmage beta.2 compatibility. |
| Remote `feat/brag-format` / `c437c7ae` | Two commits not in main, touching all four byte-conversion backends and tests. No open PR. Preserve; review the BRAG formats, stride/error behavior and independent per-backend byte parity before proposing its new public functions. Do not merge it as part of the token transition. |
| Local `feat/tokenless-rite-deinterleave` / `3942728d` | Divergent history, but [PR #8](https://github.com/imazen/garb/pull/8) merged on 2026-05-08 (`b12de840`). Current source already contains the tokenless hooks. The uncontained tip is not evidence that this implementation never landed. Compare residual changes before any cleanup. |
| Separate ablation history / `f0bd1d33` | Two-commit experimental baseline/removal study, not a production replacement. Preserve independently; no cleanup was authorized. |

All two live remote branches and the additional local feature ref are accounted for. PRs #3, #4, #6 and #8 merged; #5 closed and was superseded by #8. No open PR was present at observation time. [Issue #7](https://github.com/imazen/garb/issues/7) records the tokenless migration decision.

## Issues and TODOs

- [#9](https://github.com/imazen/garb/issues/9) remains open, but the observed current [CI run](https://github.com/imazen/garb/actions/runs/41985965661) passes `clippy + fmt` and all platform test jobs. Its old failing-job counts are historical evidence, not the current lint census. Review/resolve the issue using that evidence; this inventory does not weaken the lint or change its expected output.
- The other three issues (#1, #2 and #7) are closed historical alignment/transition work. Closed state alone is not a new re-test of every historical report.
- No TODO/FIXME/HACK/XXX markers were found in the inspected source, tests, examples, docs, manifest and justfile. Open work is recorded in branches/issues instead; absence of markers is not a claim that the crate needs no further work.
- The June [ablation report](public-api/ABLATION-garb.md) is a dated consumer scan. Re-check current source and consumers before using it to remove or promote an API.
- No branch was deleted, feature merged, API changed, test relaxed or release published by this inventory.
