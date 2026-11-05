#![cfg(feature = "experimental")]
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use garb::bytes as b;

#[test]
fn packed_channels_round_to_nearest_full_range_code() {
    let source: Vec<u8> = (0..=255u8)
        .flat_map(|v| {
            [
                v,
                v.wrapping_add(81),
                v.wrapping_add(159),
                v.wrapping_add(47),
            ]
        })
        .collect();
    type Convert = fn(&[u8], &mut [u8]) -> Result<(), garb::SizeError>;
    let cases: [(Convert, bool, bool); 4] = [
        (b::rgba_to_rgb565, false, false),
        (b::bgra_to_rgb565, true, false),
        (b::rgba_to_rgba4444, false, true),
        (b::bgra_to_rgba4444, true, true),
    ];
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for (f, bgra, rgba4) in cases {
            for width in [1, 7, 8, 31, 256] {
                let mut output = vec![0u8; 512];
                for (s, d) in source.chunks(width * 4).zip(output.chunks_mut(width * 2)) {
                    f(s, d).unwrap();
                }
                for (i, (s, d)) in source
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(output.as_chunks::<2>().0)
                    .enumerate()
                {
                    // Independent floating reference; no 4/5/6-bit half-code
                    // lies exactly on an integer input in the 255-based scale.
                    let q = |v: u8, max: f64| (f64::from(v) * max / 255.0).round() as u16;
                    let r = s[if bgra { 2 } else { 0 }];
                    let g = s[1];
                    let blue = s[if bgra { 0 } else { 2 }];
                    let want = if rgba4 {
                        (q(r, 15.0) << 12)
                            | (q(g, 15.0) << 8)
                            | (q(blue, 15.0) << 4)
                            | q(s[3], 15.0)
                    } else {
                        (q(r, 31.0) << 11) | (q(g, 63.0) << 5) | q(blue, 31.0)
                    };
                    assert_eq!(
                        u16::from_le_bytes(*d),
                        want,
                        "pixel={i} width={width} tier={perm}"
                    );
                }
            }
        }
    });
}
