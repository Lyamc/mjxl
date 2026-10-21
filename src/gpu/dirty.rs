use bytemuck::{Pod, Zeroable};

use super::{Gpu, dispatch, readback, storage_buf, uniform_buf};
use crate::error::Result;
use crate::temporal::{SKIP_MAE, TILE};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    tile: u32,
    threshold: u32,
    tw: u32,
    th: u32,
    pad0: u32,
    pad1: u32,
}

impl Gpu {
    /// `0`/`1` per tile, row-major, same rule as [`crate::temporal::tile_mask`].
    pub fn dirty_tiles(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<u8>, usize, usize)> {
        let tile = TILE as usize;
        let tw = (width as usize).div_ceil(tile);
        let th = (height as usize).div_ceil(tile);
        if tw == 0 || th == 0 {
            return Ok((Vec::new(), tw, th));
        }
        let params = Params {
            width,
            height,
            tile: TILE,
            threshold: SKIP_MAE,
            tw: tw as u32,
            th: th as u32,
            pad0: 0,
            pad1: 0,
        };
        let prev_b = storage_buf(&self.device, &self.queue, "prev", &pad4(prev), true);
        let curr_b = storage_buf(&self.device, &self.queue, "curr", &pad4(curr), true);
        let mask_bytes = (tw * th * 4) as u64;
        let mask_b = storage_buf(
            &self.device,
            &self.queue,
            "mask",
            &vec![0u8; mask_bytes as usize],
            false,
        );
        let uni = uniform_buf(&self.device, &self.queue, bytemuck::bytes_of(&params));
        let groups = (tw * th).div_ceil(64) as u32;
        dispatch(
            self,
            &self.dirty,
            &self.dirty_layout,
            &[&prev_b, &curr_b, &mask_b, &uni],
            groups,
            1,
        );
        let raw = readback(&self.device, &self.queue, &mask_b, mask_bytes)?;
        let words: &[u32] = bytemuck::cast_slice(&raw[..tw * th * 4]);
        let mask = words.iter().map(|&w| u8::from(w != 0)).collect();
        Ok((mask, tw, th))
    }
}

fn pad4(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    while v.len() % 4 != 0 {
        v.push(0);
    }
    v
}
