# Precision review, November 2026

`garb` 0.3.0 is strictly integer-only. Integer depth conversion normalizes the
full integer range (`255` for `u8`, `65,535` for `u16`) and rounds narrowing to
nearest. `garb` does not apply transfer functions; floating-point depth and
alpha conversions live in `zenpixels-convert` on `magetypes`.

| Function | 0.2.8 finding | Current behavior (`0.3.0`) |
|---|---|---|
| `bytes::convert_u16_to_u8` | 127 / 65,536 inputs one code too low | Exact `(v + 128) / 257` across scalar, AVX2, and NEON (`ecf6c6b3`) |
| `bytes::convert_u8_to_u16` | Exact `v * 257` | Unchanged (promoted from `experimental` to core `garb::bytes`) |
| `bytes::premultiply_alpha_{rgba,bgra}_u8` | Exact `(c * a + 127) / 255` | Promoted from `experimental` to core `garb::bytes` across scalar, AVX2, and NEON |

Regression tests exhaustively check all 65,536 `u16` inputs, all `(c, a)` `u8`
pairs, row splits, and dispatch tiers.

Packed compression (`experimental`) also uses nearest division by 255: this corrects 15/31/7
of 256 channel inputs at 5/6/4 bits (`7b0272f5`). Bit-replicating RGB565
expansion remains: 4 of 32 five-bit codes and 10 of 64 six-bit codes differ
from nearest normalized u8 scaling by one output code.

Weighted grayscale uses integer coefficient sets rather than exact decimal
BT.601/709/2020 weights. Exhaustive RGB triples (16,777,216 per curve) differ
from decimal-weight nearest rounding at 2,243,315 / 2,679,642 / 2,103,840
triples respectively, maximum one u8 code. `just precision-policies`
reproduces this. Keep this existing fixed-point contract; an exact-coefficient
variant needs separate API review. RGB565/RGBA4444 intentionally discard
depth. RGBA1010102 stores 10-bit RGB in the low bits of u16; it does not
scale to full-range u16. Its alpha expands to 0/341/682/1023 and narrows to
nearest. These are format policies, distinct from generic depth conversion.
