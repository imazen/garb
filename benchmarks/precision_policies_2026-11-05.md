# Retained format policies, 2026-11-05

Source base: `edf8c269d0af32a2a2cb767f9ca5f95206c3b7c7` plus the example
committed with this record. Command: `cargo run --release --all-features
--example precision_policies`, through run-heavy 16G/10 jobs. x86-64 Rust
1.99.1. The wrapper for the measurement and strict lint recorded rc=101
because the newly added example required a mechanical chunk-iterator lint
fix; the measurement completed. Its peak-RSS was 0.53 GiB. The corrected
example and all-target/doc/semver checks then passed (rc=0, 207 s,
peak-RSS 0.61 GiB, min-available 17,047 MiB, peak-load 1.74).

| Existing policy | Domain | Values differing from exact decimal ratio, nearest | Max code delta |
|---|---:|---:|---:|
| BT.601 integer luma | 16,777,216 RGB triples | 2,243,315 | 1 |
| BT.709 integer luma | 16,777,216 RGB triples | 2,679,642 | 1 |
| BT.2020 integer luma | 16,777,216 RGB triples | 2,103,840 | 1 |
| RGB 5→8 bit replication | 32 | 4 | 1 |
| RGB 6→8 bit replication | 64 | 10 | 1 |
| 10→16 bit replication | 1024 | 234 | 1 |
| 12→16 bit replication | 4096 | 952 | 1 |
| 14→16 bit replication | 16384 | 2730 | 1 |

The last three are inventory comparisons, not garb entry points. Decimal
luma references use exact integer ratios (299/587/114 over 1000 and
2126/7152/722, 2627/6780/593 over 10000) and half-up rounding. They compare
signal-domain coefficients, not linear-light luminance conversion.

Keep existing luma and packed expansion policies in this patch. Offer an
explicit exact-coefficient/full-range policy through separate API review
before changing their bytes. No throughput comparison of those alternatives
was made. Raw validation logs: `~/tmp/precision-logs/garb-release-*.log` on
r7900x.
