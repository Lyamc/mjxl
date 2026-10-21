//! Progressive rendering.
//!
//! Two mechanisms:
//!
//! * **Codestream.** Modular frames are written with smaller groups (and Squeeze
//!   when the distance is positive) so a truncated sample still paints. VarDCT
//!   samples already store LF groups before HF groups; a short prefix is a
//!   blurry picture. [`crate::still::decode_jxl_prefix`] renders either.
//! * **Frequency passes.** Before entropy coding, an 8×8 DCT (GPU when a
//!   Vulkan/DX12/Metal adapter exists, otherwise the `o3-jxl` SIMD factorization)
//!   can show DC, the top-left 2×2, and the full block. That is the VarDCT
//!   progressive preview, independent of ANS.

use crate::error::Result;
use crate::gpu::{self, Backend, Gpu};

#[derive(Clone, Debug)]
pub struct FrequencyPasses {
    pub dc: Vec<u8>,
    pub low: Vec<u8>,
    pub full: Vec<u8>,
}

pub fn frequency_passes(
    rgb: &[u8],
    width: u32,
    height: u32,
    backend: Backend,
) -> Result<FrequencyPasses> {
    let (blocks, bw, bh) = rgb_to_blocks(rgb, width, height);
    let gpu = gpu::select(backend)?;
    let coeffs = xform(&blocks, false, gpu.as_deref())?;
    let dc = blocks_to_rgb(
        &xform(&mask(&coeffs, 0), true, gpu.as_deref())?,
        bw,
        bh,
        width,
        height,
    );
    let low = blocks_to_rgb(
        &xform(&mask(&coeffs, 1), true, gpu.as_deref())?,
        bw,
        bh,
        width,
        height,
    );
    let full = blocks_to_rgb(
        &xform(&coeffs, true, gpu.as_deref())?,
        bw,
        bh,
        width,
        height,
    );
    Ok(FrequencyPasses { dc, low, full })
}

fn xform(blocks: &[f32], inverse: bool, gpu: Option<&Gpu>) -> Result<Vec<f32>> {
    if let Some(gpu) = gpu {
        match gpu.dct_blocks(blocks, inverse) {
            Ok(v) => return Ok(v),
            Err(e) => eprintln!("mjxl: GPU DCT failed ({e}); using CPU"),
        }
    }
    let mut out = blocks.to_vec();
    if inverse {
        for chunk in out.chunks_mut(64) {
            let mut block = [0f32; 64];
            block.copy_from_slice(chunk);
            o3_jxl::simd::idct8_factored_2d(&mut block);
            chunk.copy_from_slice(&block);
        }
    } else {
        o3_jxl::simd::dct8_2d_blocks(&mut out);
    }
    Ok(out)
}

fn mask(src: &[f32], mode: u8) -> Vec<f32> {
    let mut out = src.to_vec();
    for block in out.chunks_mut(64) {
        for (i, c) in block.iter_mut().enumerate() {
            let row = i / 8;
            let col = i % 8;
            let keep = match mode {
                0 => i == 0,
                1 => row < 2 && col < 2,
                _ => true,
            };
            if !keep {
                *c = 0.0;
            }
        }
    }
    out
}

fn rgb_to_blocks(rgb: &[u8], width: u32, height: u32) -> (Vec<f32>, u32, u32) {
    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);
    let plane = (bw * bh) as usize;
    let mut blocks = vec![0f32; plane * 3 * 64];
    for by in 0..bh {
        for bx in 0..bw {
            for ch in 0..3u32 {
                let bi = ((ch as usize * plane) + (by * bw + bx) as usize) * 64;
                for r in 0..8u32 {
                    for c in 0..8u32 {
                        let x = (bx * 8 + c).min(width - 1);
                        let y = (by * 8 + r).min(height - 1);
                        let p = ((y * width + x) * 3 + ch) as usize;
                        blocks[bi + (r * 8 + c) as usize] = rgb[p] as f32;
                    }
                }
            }
        }
    }
    (blocks, bw, bh)
}

fn blocks_to_rgb(blocks: &[f32], bw: u32, bh: u32, width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; width as usize * height as usize * 3];
    let plane = (bw * bh) as usize;
    for by in 0..bh {
        for bx in 0..bw {
            let bi = (by * bw + bx) as usize;
            for r in 0..8u32 {
                let y = by * 8 + r;
                if y >= height {
                    continue;
                }
                for c in 0..8u32 {
                    let x = bx * 8 + c;
                    if x >= width {
                        continue;
                    }
                    let li = (r * 8 + c) as usize;
                    let o = ((y * width + x) * 3) as usize;
                    for ch in 0..3 {
                        let v = blocks[(ch * plane + bi) * 64 + li];
                        rgb[o + ch] = v.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_survives_every_pass() {
        let mut rgb = Vec::new();
        for _ in 0..16 * 16 {
            rgb.extend_from_slice(&[40, 80, 120]);
        }
        let passes = frequency_passes(&rgb, 16, 16, Backend::Cpu).unwrap();
        assert_eq!(passes.dc, rgb);
        assert_eq!(passes.low, rgb);
        assert_eq!(passes.full, rgb);
    }

    #[test]
    fn full_pass_roundtrips_a_ramp() {
        let w = 16u32;
        let h = 16u32;
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.push((x * 15) as u8);
                rgb.push((y * 12) as u8);
                rgb.push(30);
            }
        }
        let passes = frequency_passes(&rgb, w, h, Backend::Cpu).unwrap();
        let mut max = 0i32;
        for (a, b) in rgb.iter().zip(&passes.full) {
            max = max.max((*a as i32 - *b as i32).abs());
        }
        assert!(max <= 1, "full DCT roundtrip drifted by {max}");
        // DC is flatter than the source: at least one pixel moves.
        assert_ne!(passes.dc, rgb);
    }
}
