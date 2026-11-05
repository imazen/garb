//! Compare retained format policies with exact decimal ratios.
fn main() {
    for (name, reference, denominator) in [
        ("BT601", [299u32, 587, 114], 1000),
        ("BT709", [2126, 7152, 722], 10000),
        ("BT2020", [2627, 6780, 593], 10000),
    ] {
        let (mut wrong, mut max_error) = (0u64, 0);
        for r in 0..256 {
            let input: Vec<u8> = (0..65536)
                .flat_map(|gb| [r as u8, (gb >> 8) as u8, gb as u8])
                .collect();
            let mut output = vec![0u8; 65536];
            match name {
                "BT601" => garb::bytes::rgb_to_gray_bt601(&input, &mut output),
                "BT709" => garb::bytes::rgb_to_gray_bt709(&input, &mut output),
                _ => garb::bytes::rgb_to_gray_bt2020(&input, &mut output),
            }
            .unwrap();
            for (rgb, actual) in input.as_chunks::<3>().0.iter().zip(output) {
                let sum: u32 = rgb.iter().zip(reference).map(|(&v, w)| v as u32 * w).sum();
                let expected = (sum + denominator / 2) / denominator;
                let error = (actual as u32).abs_diff(expected);
                wrong += u64::from(error != 0);
                max_error = max_error.max(error);
            }
        }
        println!("{name}: {wrong}/16777216 differ; max={max_error} u8 code");
    }
    for depth in [5, 6, 10, 12, 14] {
        let max = (1u32 << depth) - 1;
        let output_depth = if depth < 8 { 8 } else { 16 };
        let output_max = (1u32 << output_depth) - 1;
        let (mut wrong, mut max_error) = (0u64, 0);
        for value in 0..=max {
            let repeated =
                (value << (output_depth - depth)) | (value >> (2 * depth - output_depth));
            let nearest = (value * output_max + max / 2) / max;
            wrong += u64::from(repeated != nearest);
            max_error = max_error.max(repeated.abs_diff(nearest));
        }
        println!(
            "bit replication {depth}->{output_depth}: {wrong}/{} differ; max={max_error}",
            max + 1
        );
    }
}
