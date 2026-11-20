# SIMD boundaries and outstanding work

Updated 2026-11-19 for `0.3.0` (`feat/0.3-integer-only`) from earlier 2026-11-05 inventory (`7ab73292`).

## API guidance

`garb` 0.3.0 is strictly integer-only. All public entry points in `garb::bytes`, `garb::typed_rgb`, and `garb::imgref` take byte or typed integer slices — no `archmage` tokens or SIMD vector types appear in any public signature.

In `0.3.0`, floating-point depth conversions (`u8`/`u16` ↔ `f32`), `f32` alpha premultiply/unpremultiply, and `garb::deinterleave` were removed so floating-point policies (`DepthPolicy`, `FloatFacts`, transfer functions) live in `zenpixels-convert` on `magetypes`. Integer grayscale layout, weighted integer luma (`bt601`, `bt709`, `bt2020`), `u8 ↔ u16` depth conversion, and `u8` RGBA/BGRA premultiply were promoted out of `experimental` into core `garb::bytes`.

With floating-point kernels removed, soft-float targets such as `x86_64-unknown-none` compile cleanly without extra feature guards.

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
- A separate later CI regression ([run 41990539720](https://github.com/imazen/garb/actions/runs/41990539720)) came from the newly added precision example calling experimental grayscale APIs in no-default builds. Its Cargo feature requirement is fixed in `9ec9fa71`; both feature combinations and the no-default library/test suite pass locally. This is separate from #9's lint issue. See [project notes](../CLAUDE.md).
- The other three issues (#1, #2 and #7) are closed historical alignment/transition work. Closed state alone is not a new re-test of every historical report.
- No TODO/FIXME/HACK/XXX markers were found in the inspected source, tests, examples, docs, manifest and justfile. Open work is recorded in branches/issues instead; absence of markers is not a claim that the crate needs no further work.
- The June [ablation report](public-api/ABLATION-garb.md) is a dated consumer scan. Re-check current source and consumers before using it to remove or promote an API.
- No branch was deleted, feature merged, API changed, test relaxed or release published by this inventory.
