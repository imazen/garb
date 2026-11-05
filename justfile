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
    cargo clippy --locked --all-features --lib -- -D warnings
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
    cargo semver-checks --baseline-version 0.2.8
