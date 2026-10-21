use bytemuck::{Pod, Zeroable};

use super::{Gpu, dispatch, readback, storage_buf, uniform_buf};
use crate::error::{Error, Result};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    num_blocks: u32,
    base: u32,
    inverse: u32,
    pad: u32,
}

impl Gpu {
    /// Forward (`inverse == false`) or factored inverse 8×8 DCT on contiguous blocks.
    pub fn dct_blocks(&self, blocks: &[f32], inverse: bool) -> Result<Vec<f32>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        if blocks.len() % 64 != 0 {
            return Err(Error::Gpu("DCT buffer is not a multiple of 64".into()));
        }
        let n = (blocks.len() / 64) as u32;
        let bytes = bytemuck::cast_slice::<f32, u8>(blocks);
        let input = storage_buf(&self.device, &self.queue, "dct-in", bytes, true);
        let output = storage_buf(
            &self.device,
            &self.queue,
            "dct-out",
            &vec![0u8; bytes.len()],
            false,
        );
        // Workgroup x is the block index and must stay within the device limit.
        const CHUNK: u32 = 4096;
        let mut done = 0u32;
        while done < n {
            let count = (n - done).min(CHUNK);
            let params = Params {
                num_blocks: count,
                base: done,
                inverse: u32::from(inverse),
                pad: 0,
            };
            let uni = uniform_buf(&self.device, &self.queue, bytemuck::bytes_of(&params));
            dispatch(
                self,
                &self.dct,
                &self.dct_layout,
                &[&input, &output, &uni],
                count,
                1,
            );
            done += count;
        }
        let raw = readback(&self.device, &self.queue, &output, bytes.len() as u64)?;
        let out: Vec<f32> = bytemuck::cast_slice(&raw[..bytes.len()]).to_vec();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_gpu() -> Option<Gpu> {
        match Gpu::new() {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("Shader")
                    || msg.contains("parsing error")
                    || msg.contains("Validation")
                {
                    panic!("gpu shader failed to compile: {e}");
                }
                eprintln!("skip gpu: {e}");
                None
            }
        }
    }

    #[test]
    fn gpu_dct_matches_o3_jxl() {
        let Some(gpu) = open_gpu() else {
            return;
        };
        let mut block = [0f32; 64];
        for (i, s) in block.iter_mut().enumerate() {
            *s = (i as f32 * 3.0) - 20.0;
        }
        let cpu = {
            let mut b = block;
            o3_jxl::dct8_2d(&mut b);
            b
        };
        let gpu_out = gpu.dct_blocks(&block, false).unwrap();
        let mut max = 0f32;
        for (a, b) in cpu.iter().zip(&gpu_out) {
            max = max.max((a - b).abs());
        }
        assert!(max < 1e-2, "forward DCT drifted by {max}");
        let back = gpu.dct_blocks(&gpu_out, true).unwrap();
        let mut max = 0f32;
        for (a, b) in block.iter().zip(&back) {
            max = max.max((a - b).abs());
        }
        assert!(max < 1e-2, "DCT roundtrip drifted by {max}");
    }

    #[test]
    fn gpu_dirty_matches_cpu() {
        let Some(gpu) = open_gpu() else {
            return;
        };
        let w = 32u32;
        let h = 32u32;
        let prev = vec![10u8; (w * h * 3) as usize];
        let mut curr = prev.clone();
        curr[0] = 200;
        curr[1] = 200;
        curr[2] = 200;
        for i in (16 * w as usize * 3)..curr.len() {
            curr[i] = 80;
        }
        let cpu = crate::temporal::tile_mask(&prev, &curr, w, h);
        let gpu_mask = gpu.dirty_tiles(&prev, &curr, w, h).unwrap();
        assert_eq!(gpu_mask, cpu);
        let _ = prev;
    }

    #[test]
    fn gpu_paste_matches_cpu() {
        let Some(gpu) = open_gpu() else {
            return;
        };
        let mut canvas = vec![1u8; 16 * 16 * 3];
        let src: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 255) as u8).collect();
        let mut expect = canvas.clone();
        crate::temporal::paste_rgb(&mut expect, 16, 4, 4, 8, 8, &src);
        gpu.paste_rects(
            &mut canvas,
            16,
            16,
            &[crate::gpu::RgbRect {
                x: 4,
                y: 4,
                w: 8,
                h: 8,
                rgb: &src,
            }],
        )
        .unwrap();
        assert_eq!(canvas, expect);
    }
}
