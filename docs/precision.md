# Precision review, November 2026

Depth conversion normalizes the full integer range. u8 and u16 widening
multiply by the rounded f32 reciprocal of 255 and 65,535, respectively.
The result is within one ULP of division, with identical bits across
backends, row splits and scalar tails.

f32 narrowing clamps to [0, 1], scales in f32, and rounds the resulting
value to nearest with ties to even. NaN narrows to zero. The f32 product
can round onto a half-code before the integer rounding step. The tests
check the three adjacent f32 inputs at every mathematical half-code
threshold, all u8 and u16 widening codes, odd rows and stride padding.

Integer depth conversion uses `v * 257` for u8 to u16 and
`(v + 128) / 257` for nearest u16 to u8. Garb applies no transfer function.

Packed compression also uses nearest division by 255: this corrects 15/31/7
of 256 channel inputs at 5/6/4 bits (`7b0272f5`). Bit-replicating RGB565
expansion remains: 4 of 32 five-bit codes and 10 of 64 six-bit codes differ
from nearest normalized u8 scaling by one output code.

Weighted grayscale uses integer coefficient sets rather than exact decimal
BT.601/709/2020 weights. Exhaustive RGB triples (16,777,216 per curve) differ
from decimal-weight nearest rounding at 2,243,315 / 2,679,642 / 2,103,840
triples respectively, maximum one u8 code. `just precision-policies`
reproduces this. Keep this existing fixed-point contract in the patch;
an exact-coefficient variant needs separate API review. RGB565/RGBA4444 intentionally discard
depth. RGBA1010102 stores 10-bit RGB in the low bits of u16; it does not
scale to full-range u16. Its alpha expands to 0/341/682/1023 and narrows to
nearest. These are format policies, distinct from generic depth conversion.

Consumers can see changed normalized f32 bits and corrected narrowing bytes.
Review output digests when upgrading. Depth APIs retain their existing
`experimental` gate; this review adds no public API. An inventory finding no
workspace callers does not authorize removal of an existing public function.
