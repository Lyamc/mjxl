//! One JPEG XL codestream: full decode, or a progressive prefix.

use o3_jxl::{Decoder, ProgressiveDecoder};

use crate::error::{Error, Result};

pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let n = rgba.len() / 4;
    let mut rgb = vec![0u8; n * 3];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        rgb[i * 3] = px[0];
        rgb[i * 3 + 1] = px[1];
        rgb[i * 3 + 2] = px[2];
    }
    rgb
}

pub fn decode_jxl_rgb(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let frames = Decoder::new()
        .request_srgb(true)
        .keep_linear(false)
        .decode_bytes(bytes)?;
    let fr = frames
        .into_iter()
        .next()
        .ok_or_else(|| Error::from("empty jxl sample"))?;
    Ok((fr.width, fr.height, rgba_to_rgb(&fr.rgba)))
}

/// `fraction` is in `0.0..=1.0` of `bytes`. `Ok(None)` means the prefix cannot
/// paint a canvas yet (header only, or no groups).
pub fn decode_jxl_prefix(bytes: &[u8], fraction: f32) -> Result<Option<(u32, u32, Vec<u8>)>> {
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction >= 1.0 {
        return Ok(Some(decode_jxl_rgb(bytes)?));
    }
    let n = ((bytes.len() as f32) * fraction).round() as usize;
    let n = n.min(bytes.len());
    let mut dec = ProgressiveDecoder::new().request_srgb(true);
    dec.push(&bytes[..n])?;
    match dec.render()? {
        Some(fr) => Ok(Some((fr.width, fr.height, rgba_to_rgb(&fr.rgba)))),
        None => Ok(None),
    }
}
