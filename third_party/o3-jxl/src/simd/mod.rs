//! Portable SIMD kernels (`std::simd`) shared by decode (and later encode).
//!
//! Lane widths are chosen so they map cleanly to WASM SIMD128 (`f32x4` /
//! `u8x16`) as well as native NEON/SSE/AVX.

pub mod color;
pub mod dct;
pub mod gabor;
pub mod idct;
pub mod quantize;
mod util;

pub use color::{
    INTENSITY_TARGET, OPSIN_BIAS, OPSIN_INV_MAT, OpsinParams, linear_rgb_to_xyb, rgb_to_ycbcr,
    srgb_eotf, srgb_eotf_lut, srgb8_rgb_to_xyb_padded, xyb_to_linear_rgb, ycbcr_to_rgb,
};
pub use dct::{
    dct_rect, dct_rect_from_plane, dct2_2d, dct2_2d_blocks, dct4_2d, dct4_2d_blocks, dct8_2d,
    dct8_2d_blocks, dct8_fwd_from_plane, dct8_fwd_from_plane_2, dct16_2d, dct16_fwd_from_plane,
    dct32_2d, dct32_fwd_from_plane, idct_rect, idct2_2d, idct4_2d, idct8_factored_2d, idct16_2d,
    idct32_2d,
};
pub use gabor::{GABORISH_INV_MUL_VARDCT, gaborish_inverse_plane, gaborish_plane, upsample2_h};
pub use idct::{idct8_2d, idct8_2d_blocks};
pub use quantize::{
    floats_to_rgba, floats_to_rgba16, floats_unit_to_rgba, floats_unit_to_rgba16,
    planar_encoded_to_rgba, planar_linear_to_rgba,
};
