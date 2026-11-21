# garb dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the standalone apidoc/ package, so it is
# never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Validate the owner-requested archmage beta constraint update.
beta-check:
    cargo fmt --check
    cargo clippy --locked --lib -- -D warnings
    cargo test --locked --lib

beta-library-check:
    cargo test --locked --all-features --lib

# Cover optional packed/float formats in the current strict lint gate.
beta-all-formats-check:
    cargo fmt --check
    cargo check --locked --benches
    cargo check --locked --benches --features experimental
    cargo clippy --locked --all-features --lib -- -D warnings -A deprecated
    cargo test --locked --all-features --lib
    cargo clippy --locked --no-default-features --lib -- -D warnings

# Exhaustive nearest-value depth regression across dispatch tiers and row splits.
precision-depth-check:
    cargo test --all-features --lib depth_u16_to_u8_nearest_exhaustive

precision-float-check:
    cargo test --all-features --lib depth_f32_nearest_at_every_quantization_boundary

# Required release validation, without publishing.
release-check:
    cargo test --all-targets --all-features
    cargo test --doc --all-features
    cargo +1.98.1 semver-checks --baseline-version 0.2.8

precision-policies:
    cargo run --release --all-features --example precision_policies

# The precision example calls experimental grayscale conversions explicitly.
check-precision-example-features:
    cargo check --no-default-features --examples
    cargo check --all-features --example precision_policies
    cargo test --no-default-features -- --test-threads=1

# Inspect downstream-exercised x86-64-v3 (AVX2) full kernel assembly via cargo-show-asm (omit symbol to list all).
asm-v3 symbol="":
    CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 RUSTFLAGS="-C target-cpu=x86-64-v3" cargo asm --example downstream_codegen --target x86_64-unknown-linux-gnu {{symbol}}

# Inspect downstream-exercised x86-64-v4 (AVX-512) full kernel assembly via cargo-show-asm (omit symbol to list all).
asm-v4 symbol="":
    CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 RUSTFLAGS="-C target-cpu=x86-64-v4" cargo asm --example downstream_codegen --target x86_64-unknown-linux-gnu {{symbol}}

# Inspect downstream-exercised AArch64 NEON full kernel assembly via cargo-show-asm (omit symbol to list all).
asm-neon symbol="":
    CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo asm --example downstream_codegen --target aarch64-unknown-linux-gnu {{symbol}}

# Inspect downstream-exercised WebAssembly SIMD128 full kernel assembly via cargo-show-asm (omit symbol to list all).
asm-wasm128 symbol="":
    CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 RUSTFLAGS="-C target-feature=+simd128" cargo asm --example downstream_codegen --target wasm32-unknown-unknown --wasm {{symbol}}

# Inspect downstream-exercised WebAssembly Relaxed-SIMD full kernel assembly via cargo-show-asm (omit symbol to list all).
asm-wasm128-relaxed symbol="":
    CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo asm --example downstream_codegen --target wasm32-unknown-unknown --wasm {{symbol}}

# Dump a downstream-exercised symbol across all 5 SIMD tiers (v3, v4, neon, wasm128, wasm128-relaxed).
asm-all symbol:
    @echo "=== x86-64-v3 ===" && just asm-v3 {{symbol}}
    @echo "=== x86-64-v4 ===" && just asm-v4 {{symbol}}
    @echo "=== aarch64-neon ===" && just asm-neon {{symbol}}
    @echo "=== wasm128 ===" && just asm-wasm128 {{symbol}}
    @echo "=== wasm128-relaxed ===" && just asm-wasm128-relaxed {{symbol}}
