# Precision review, November 2026

Depth conversion normalizes the full integer range: 255 for u8 and 65,535
for u16. Narrowing rounds to nearest; f32 half-code ties round upward after
clamping. NaN narrows to zero. garb does not apply transfer functions.

| Function | 0.2.8 finding | Current behavior |
|---|---|---|
| `bytes::convert_u16_to_u8` | 127 / 65,536 inputs one code too low | Exact `(v + 128) / 257` (`ecf6c6b3`) |
| `bytes::convert_f32_to_u8` | 32 / 4,194,304 uniform samples one code too high | Widen before scaling; zero differences (`4310dda3`) |
| `bytes::convert_f32_to_u16` | 5,460 / 4,194,304 samples one code too high | Same policy; zero differences (`4310dda3`) |
| `bytes::convert_u8_to_f32` | 126 / 256 values one ULP off in full vector bodies | Corrected reciprocal, matching division (`605276d7`) |
| `bytes::convert_u16_to_f32` | 512 / 65,536 values one ULP off in vector bodies | Corrected reciprocal, matching division (`605276d7`) |
| `bytes::convert_u8_to_u16` | Exact `v * 257` | Unchanged |

The clamped f32 product with a 16-bit integer scale and the 0.5 bias are
represented exactly in f64. Regression tests independently check adjacent
f32 values around every u8/u16 half-code boundary, every u16 input, row splits
and dispatch tiers. AVX2 and NEON use the same policy. This fixes double
rounding without changing the tie policy.

Packed compression also uses nearest division by 255: this corrects 15/31/7
of 256 channel inputs at 5/6/4 bits (`7b0272f5`). Bit-replicating RGB565
expansion remains: 4 of 32 five-bit codes and 10 of 64 six-bit codes differ
from nearest normalized u8 scaling by one output code.

Weighted grayscale uses integer coefficient sets rather than exact
floating-point BT.601/709/2020 weights. RGB565/RGBA4444 intentionally discard
depth. RGBA1010102 stores 10-bit RGB in the low bits of u16; it does not
scale to full-range u16. Its alpha expands to 0/341/682/1023 and narrows to
nearest. These are format policies, distinct from generic depth conversion.

Consumers can see changed normalized f32 bits and corrected narrowing bytes.
Review output digests when upgrading. Depth APIs retain their existing
`experimental` gate; this review adds no public API. An inventory finding no
workspace callers does not authorize removal of an existing public function.
