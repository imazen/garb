//! Validates the code examples from README.md compile and behave correctly.

#[test]
fn readme_core_api() {
    use garb::{rgb_to_bgra, rgba_to_bgra_inplace};

    let mut pixels = vec![255u8, 0, 128, 255, 0, 200, 100, 255];
    rgba_to_bgra_inplace(&mut pixels).unwrap();
    assert_eq!(pixels, [128, 0, 255, 255, 100, 200, 0, 255]);

    let rgb = vec![255u8, 0, 128];
    let mut bgra = vec![0u8; 4];
    rgb_to_bgra(&rgb, &mut bgra).unwrap();
    assert_eq!(bgra, [128, 0, 255, 255]);
}

#[test]
fn readme_strided() {
    use garb::rgba_to_bgra_inplace_strided;

    let mut buf = vec![0u8; 256 * 100];
    rgba_to_bgra_inplace_strided(&mut buf, 256, 60, 100).unwrap();
}

#[test]
fn readme_typed_rgb() {
    use garb::typed_rgb;
    use rgb::{Bgra, Rgba};

    let mut pixels: Vec<Rgba<u8>> = vec![Rgba::new(255, 0, 128, 255); 100];
    let bgra: &mut [Bgra<u8>] = typed_rgb::rgba_to_bgra_mut(&mut pixels);
    assert_eq!(bgra[0], Bgra { b: 128, g: 0, r: 255, a: 255 });
}

#[test]
fn readme_imgref() {
    use garb::imgref;
    use ::imgref::ImgVec;
    use rgb::{Bgra, Rgba};

    let rgba_img = ImgVec::new(vec![Rgba::new(255, 0, 128, 200); 640 * 480], 640, 480);
    let bgra_img: ImgVec<Bgra<u8>> = imgref::swap_rgba_to_bgra(rgba_img);
    assert_eq!(bgra_img.width(), 640);
    assert_eq!(bgra_img.height(), 480);
}
