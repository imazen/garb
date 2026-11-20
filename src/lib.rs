//! # garb
//!
//! *Dress your pixels for the occasion.*
//!
//! You can't show up to a function in the wrong style. Swap your BGR for your RGB, your ARGB for your RGBA, and tie up loose
//! ends like that unreliable alpha BGRX.
//!
//! SIMD-optimized pixel format conversions for row-level and whole-image
//! operations. Supports x86-64 AVX2, ARM NEON, and WASM SIMD128 with
//! automatic fallback to scalar code.
//!
//! ## Modules
//!
//! - [`bytes`] — Core `&[u8]` conversions (contiguous and strided).
//! - `typed_rgb` — Type-safe wrappers using `rgb` crate pixel types (feature `rgb`).
//! - `imgref` — Whole-image conversions on `ImgVec` / `ImgRef` (feature `imgref`).
//!
//! ## Naming convention
//!
//! All functions follow the pattern `{src}_to_{dst}` for copy operations and
//! `{src}_to_{dst}_inplace` for in-place mutations. Append `_strided` for
//! multi-row buffers with stride.
//!
//! ## Strides
//!
//! A **stride** (also called "pitch" or "row pitch") is the distance between
//! the start of one row and the start of the next, measured in units of the
//! slice's element type:
//!
//! - For `&[u8]` functions: stride is in **bytes**.
//! - For typed APIs (`imgref`, `typed_rgb`): stride is in **elements** of the
//!   slice's item type — e.g. pixel count for `ImgRef<Rgba<u8>>`, but byte
//!   count for `ImgRef<u8>`.
//!
//! When `stride == width` (in the appropriate unit) the image is contiguous.
//! When `stride > width` the gap at the end of each row is padding — garb
//! never reads or writes it.
//!
//! The required buffer length (in elements) for a strided image is:
//! `(height - 1) * stride + width`
//!
//! All `_strided` functions take dimensions *before* strides:
//! - In-place: `(buf, width, height, stride)`
//! - Copy: `(src, dst, width, height, src_stride, dst_stride)`
//!
//! ## Feature flags
//!
//! - **`std`** — Enables `std` on dependencies (e.g. `archmage`).
//! - **`experimental`** *(deprecated in `0.2.9`, queued for removal in `0.3.0`)* —
//!   Gray layout, weighted luma, depth conversion, packed formats, planar
//!   deinterleave, and alpha premultiply/unpremultiply. Use `zenpixels-convert` instead.
//! - **`rgb`** *(deprecated in `0.2.9`, queued for removal in `0.3.0`)* —
//!   Type-safe conversions using [`rgb`](https://docs.rs/rgb) crate pixel types
//!   via bytemuck. Cast slices to `&[u8]` / `&mut [u8]` and call [`bytes`] directly.
//! - **`imgref`** *(deprecated in `0.2.9`, queued for removal in `0.3.0`)* —
//!   Multi-row conversions using `ImgRef` / `ImgRefMut` from the
//!   [`imgref`](https://docs.rs/imgref) crate. Call `garb::bytes::*_strided` directly.

#![no_std]
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(deprecated))]
#![cfg_attr(
    all(feature = "experimental", not(feature = "rgb")),
    deprecated(
        since = "0.2.9",
        note = "garb Cargo feature `experimental` is deprecated and queued for removal in 0.3.0; remove `experimental` from Cargo features (use `zenpixels-convert` for depth/luma/alpha)"
    )
)]
#![cfg_attr(
    all(
        feature = "rgb",
        not(feature = "imgref"),
        not(feature = "experimental")
    ),
    deprecated(
        since = "0.2.9",
        note = "garb Cargo feature `rgb` is deprecated and queued for removal in 0.3.0; cast typed slices to `&[u8]` and use `garb::bytes` instead"
    )
)]
#![cfg_attr(
    all(feature = "imgref", not(feature = "experimental")),
    deprecated(
        since = "0.2.9",
        note = "garb Cargo features `imgref` and `rgb` are deprecated and queued for removal in 0.3.0; use `garb::bytes::*_strided` instead"
    )
)]
#![cfg_attr(
    all(feature = "experimental", feature = "rgb"),
    deprecated(
        since = "0.2.9",
        note = "garb Cargo features `experimental`, `rgb`, and `imgref` are deprecated and queued for removal in 0.3.0; depend on `garb` without those features"
    )
)]

#[cfg(all(feature = "experimental", not(test)))]
mod _deprecated_feature_experimental {
    #[deprecated(
        since = "0.2.9",
        note = "garb Cargo feature `experimental` is deprecated and queued for removal in 0.3.0; remove `experimental` from Cargo features"
    )]
    const EXPERIMENTAL_FEATURE_ENABLED: () = ();
    const _: () = EXPERIMENTAL_FEATURE_ENABLED;
}

#[cfg(all(feature = "rgb", not(test)))]
mod _deprecated_feature_rgb {
    #[deprecated(
        since = "0.2.9",
        note = "garb Cargo feature `rgb` is deprecated and queued for removal in 0.3.0; remove `rgb` from Cargo features and use `garb::bytes`"
    )]
    const RGB_FEATURE_ENABLED: () = ();
    const _: () = RGB_FEATURE_ENABLED;
}

#[cfg(all(feature = "imgref", not(test)))]
mod _deprecated_feature_imgref {
    #[deprecated(
        since = "0.2.9",
        note = "garb Cargo feature `imgref` is deprecated and queued for removal in 0.3.0; remove `imgref` from Cargo features and use `garb::bytes::*_strided`"
    )]
    const IMGREF_FEATURE_ENABLED: () = ();
    const _: () = IMGREF_FEATURE_ENABLED;
}

#[cfg(feature = "imgref")]
extern crate alloc;

pub mod bytes;

#[cfg(feature = "experimental")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `experimental` and `deinterleave` will be removed in 0.3.0; vendor chunk helpers or use zenpixels-convert"
)]
pub mod deinterleave;

#[cfg(feature = "rgb")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `rgb` (`typed_rgb`) will be removed in 0.3.0; cast typed slices to `&[u8]` and use `garb::bytes`"
)]
pub mod typed_rgb;

#[cfg(feature = "imgref")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `imgref` will be removed in 0.3.0; use `garb::bytes::*_strided`"
)]
pub mod imgref;

/// Pixel buffer size or alignment error.
///
/// Returned when a buffer's length is not a multiple of the pixel size,
/// a destination buffer is too small, or stride/dimensions are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SizeError {
    /// Buffer length is not a multiple of the expected bytes per pixel (or is empty).
    NotPixelAligned,
    /// Destination buffer has fewer pixels than the source.
    PixelCountMismatch,
    /// Stride, dimensions, or total buffer size are inconsistent.
    InvalidStride,
}

impl core::fmt::Display for SizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPixelAligned => f.write_str("buffer length is not pixel-aligned"),
            Self::PixelCountMismatch => f.write_str("destination has fewer pixels than source"),
            Self::InvalidStride => {
                f.write_str("stride, dimensions, or buffer size are inconsistent")
            }
        }
    }
}

impl core::error::Error for SizeError {}

// ===========================================================================
// Generic conversion traits
// ===========================================================================

/// Copy-convert a pixel slice: `&[Src]` → `&mut [Dst]`.
///
/// Implemented for every valid `(Src, Dst)` pair when `feature = "rgb"`.
/// Use [`convert`] for the free-function form.
#[cfg(feature = "rgb")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `rgb` (`ConvertTo`) will be removed in 0.3.0; cast typed slices to `&[u8]` and use `garb::bytes`"
)]
#[allow(deprecated)]
pub trait ConvertTo<Dst>: Sized {
    /// Convert pixels from `src` into `dst`.
    fn convert_to(src: &[Self], dst: &mut [Dst]) -> Result<(), SizeError>;
}

/// In-place pixel conversion: `&mut [Src]` → `&mut [Dst]` (same pixel size).
///
/// Only implemented for same-size pixel pairs (e.g. 4bpp↔4bpp, 3bpp↔3bpp).
/// Use [`convert_inplace`] for the free-function form.
#[cfg(feature = "rgb")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `rgb` (`ConvertInplace`) will be removed in 0.3.0; cast typed slices to `&mut [u8]` and use `garb::bytes`"
)]
pub trait ConvertInplace<Dst>: Sized {
    /// Convert pixels in-place and return the buffer reinterpreted as `Dst`.
    fn convert_inplace(buf: &mut [Self]) -> &mut [Dst];
}

/// Copy-convert pixel slices. Type inference selects the right conversion.
///
/// ```rust
/// # #[allow(deprecated)]
/// # fn main() {
/// use rgb::{Rgb, Bgra};
/// use garb::convert;
///
/// let rgb = vec![Rgb::new(255u8, 0, 128); 2];
/// let mut bgra = vec![Bgra::default(); 2];
/// convert(&rgb, &mut bgra).unwrap();
/// assert_eq!(bgra[0], Bgra { b: 128, g: 0, r: 255, a: 255 });
/// # }
/// ```
#[cfg(feature = "rgb")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `rgb` (`convert`) will be removed in 0.3.0; cast typed slices to `&[u8]` and use `garb::bytes`"
)]
#[allow(deprecated)]
#[inline(always)]
pub fn convert<S: ConvertTo<D>, D>(src: &[S], dst: &mut [D]) -> Result<(), SizeError> {
    S::convert_to(src, dst)
}

/// In-place pixel conversion. Returns the buffer reinterpreted as the target type.
///
/// ```rust
/// # #[allow(deprecated)]
/// # fn main() {
/// use rgb::{Rgba, Bgra};
/// use garb::convert_inplace;
///
/// let mut pixels = vec![Rgba::new(255u8, 0, 128, 255); 2];
/// let bgra: &mut [Bgra<u8>] = convert_inplace(&mut pixels);
/// assert_eq!(bgra[0], Bgra { b: 128, g: 0, r: 255, a: 255 });
/// # }
/// ```
#[cfg(feature = "rgb")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `rgb` (`convert_inplace`) will be removed in 0.3.0; cast typed slices to `&mut [u8]` and use `garb::bytes`"
)]
#[allow(deprecated)]
#[inline(always)]
pub fn convert_inplace<S: ConvertInplace<D>, D>(buf: &mut [S]) -> &mut [D] {
    S::convert_inplace(buf)
}

/// Copy-convert an image: `ImgRef<Src>` → `ImgRefMut<Dst>`.
///
/// Implemented for every valid `(Src, Dst)` pair when `feature = "imgref"`.
/// Use [`convert_imgref`] for the free-function form.
#[cfg(feature = "imgref")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `imgref` (`ConvertImage`) will be removed in 0.3.0; use `garb::bytes::*_strided`"
)]
#[allow(deprecated)]
pub trait ConvertImage<Dst>: Sized {
    /// Convert pixels from `src` image into `dst` image.
    fn convert_image(
        src: ::imgref::ImgRef<'_, Self>,
        dst: ::imgref::ImgRefMut<'_, Dst>,
    ) -> Result<(), SizeError>;
}

/// In-place image conversion: consumes `ImgVec<Src>`, returns `ImgVec<Dst>`.
///
/// Only implemented for same-size pixel pairs.
/// Use [`convert_imgref_inplace`] for the free-function form.
#[cfg(feature = "imgref")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `imgref` (`ConvertImageInplace`) will be removed in 0.3.0; use `garb::bytes::*_strided`"
)]
pub trait ConvertImageInplace<Dst>: Sized {
    /// Convert an image in-place and return it reinterpreted as `Dst`.
    fn convert_image_inplace(img: ::imgref::ImgVec<Self>) -> ::imgref::ImgVec<Dst>;
}

/// Copy-convert an image. Type inference selects the right conversion.
///
/// ```rust
/// # #[allow(deprecated)]
/// # fn main() {
/// use rgb::{Rgb, Bgra};
/// use imgref::{ImgVec, ImgRefMut};
/// use garb::convert_imgref;
///
/// let src = ImgVec::new(vec![Rgb::new(255u8, 0, 128); 4], 2, 2);
/// let mut dst_buf = vec![Bgra::default(); 4];
/// let dst = ImgRefMut::new(&mut dst_buf, 2, 2);
/// convert_imgref(src.as_ref(), dst).unwrap();
/// # }
/// ```
#[cfg(feature = "imgref")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `imgref` (`convert_imgref`) will be removed in 0.3.0; use `garb::bytes::*_strided`"
)]
#[allow(deprecated)]
#[inline(always)]
pub fn convert_imgref<S: ConvertImage<D>, D>(
    src: ::imgref::ImgRef<'_, S>,
    dst: ::imgref::ImgRefMut<'_, D>,
) -> Result<(), SizeError> {
    S::convert_image(src, dst)
}

/// In-place image conversion. Consumes and returns the image with reinterpreted pixels.
///
/// ```rust
/// # #[allow(deprecated)]
/// # fn main() {
/// use rgb::{Rgba, Bgra};
/// use imgref::ImgVec;
/// use garb::convert_imgref_inplace;
///
/// let img = ImgVec::new(vec![Rgba::new(255u8, 0, 128, 200); 4], 2, 2);
/// let bgra_img: ImgVec<Bgra<u8>> = convert_imgref_inplace(img);
/// # }
/// ```
#[cfg(feature = "imgref")]
#[deprecated(
    since = "0.2.9",
    note = "garb Cargo feature `imgref` (`convert_imgref_inplace`) will be removed in 0.3.0; use `garb::bytes::*_strided`"
)]
#[allow(deprecated)]
#[inline(always)]
pub fn convert_imgref_inplace<S: ConvertImageInplace<D>, D>(
    img: ::imgref::ImgVec<S>,
) -> ::imgref::ImgVec<D> {
    S::convert_image_inplace(img)
}
