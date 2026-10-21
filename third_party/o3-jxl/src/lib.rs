//! o3-jxl — high-performance JPEG XL codec targeting native and WebAssembly.
//!
//! Decoding currently uses [jxl-oxide] for bitstream parsing, entropy coding,
//! and render. Numerically heavy stages that this crate owns (2/4/8 DCT, XYB,
//! YCbCr, Gaborish, RGBA packing) run through [`std::simd`] portable SIMD,
//! which maps to WASM SIMD128 as well as NEON/SSE/AVX.
//!
//! Encoding is a from-scratch modular path: SIMD YCoCg-R + Gradient / Zero
//! residuals, rANS entropy, optional Squeeze and palette, 8- and 16-bit
//! integer samples, and multi-frame animation. Lossy modular uses Squeeze with
//! per-band quant (fine LF, Zero-coded HF). The codestream is readable by this
//! decoder and by libjxl / djxl.
//!
//! Nightly Rust is required (`#![feature(portable_simd)]`).

#![feature(portable_simd)]

mod decoder;
mod encoder;
mod error;
mod image;
mod progressive;
pub mod simd;

#[cfg(feature = "wasm")]
mod wasm;

pub use decoder::{
    Decoder, extract_exif, extract_xmp, reconstruct_jpeg, write_apng, write_pam, write_pam_depth,
    write_pfm, write_pgm, write_pgx, write_png, write_png_depth, write_png_writer,
    write_png_writer_depth, write_ppm,
};
pub use encoder::{Encoder, PixelFormat, quality_to_distance};
pub use error::{Error, Result};
pub use image::{DecodedFrame, ImageInfo};
pub use progressive::{ProgressiveDecoder, psnr_rgb};
pub use simd::{
    dct2_2d, dct4_2d, dct8_2d, dct16_2d, floats_to_rgba, floats_to_rgba16, gaborish_plane,
    idct2_2d, idct4_2d, idct8_2d, idct8_2d_blocks, linear_rgb_to_xyb, rgb_to_ycbcr, upsample2_h,
    xyb_to_linear_rgb, ycbcr_to_rgb,
};
