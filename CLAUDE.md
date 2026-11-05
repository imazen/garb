# garb project guidance

See [docs/SIMD_TRANSITION.md](docs/SIMD_TRANSITION.md) for branch/issue dispositions and the existing tokenless array hooks. Ordinary callers use dispatched functions. Experimental native chunk hooks require matching CPU-feature context; no token crosses their boundary.

## Known Bugs

- Resolved 2026-11-05: `precision_policies` used experimental grayscale APIs without declaring its required Cargo feature, breaking no-default-feature CI on native and WASM targets ([run 41990539720](https://github.com/imazen/garb/actions/runs/41990539720)). The example now requires `experimental`; pixel tests and example arithmetic remain intact. Keep its feature requirement if the example is extended.
