use bytemuck::{Pod, Zeroable};

use super::{Gpu, dispatch, readback, storage_buf, uniform_buf};
use crate::error::{Error, Result};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One pasted RGB rect.
pub struct RgbRect<'a> {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub rgb: &'a [u8],
}

impl Gpu {
    /// Paste RGB rects into `canvas` (packed RGB, row-major). One upload, one download.
    pub fn paste_rects(
        &self,
        canvas: &mut [u8],
        canvas_w: u32,
        canvas_h: u32,
        rects: &[RgbRect<'_>],
    ) -> Result<()> {
        let expect = canvas_w as usize * canvas_h as usize * 3;
        if canvas.len() < expect || canvas_w == 0 {
            return Err(Error::Gpu("canvas buffer is short".into()));
        }
        let n = canvas_w as usize * canvas_h as usize;
        let mut px = vec![0u32; n];
        for i in 0..n {
            let o = i * 3;
            px[i] = canvas[o] as u32
                | ((canvas[o + 1] as u32) << 8)
                | ((canvas[o + 2] as u32) << 16)
                | (255 << 24);
        }
        let buf = storage_buf(
            &self.device,
            &self.queue,
            "canvas",
            bytemuck::cast_slice(&px),
            false,
        );
        for rect in rects {
            if rect.w == 0 || rect.h == 0 {
                continue;
            }
            let x1 = rect
                .x
                .checked_add(rect.w)
                .ok_or_else(|| Error::Gpu("paste rect overflows".into()))?;
            let y1 = rect
                .y
                .checked_add(rect.h)
                .ok_or_else(|| Error::Gpu("paste rect overflows".into()))?;
            if x1 > canvas_w || y1 > canvas_h {
                return Err(Error::Gpu("paste rect outside canvas".into()));
            }
            let need = rect.w as usize * rect.h as usize * 3;
            if rect.rgb.len() < need {
                return Err(Error::Gpu("paste source is short".into()));
            }
            let src = storage_buf(
                &self.device,
                &self.queue,
                "paste-src",
                &pad4(&rect.rgb[..need]),
                true,
            );
            let params = Params {
                canvas_w,
                x: rect.x,
                y: rect.y,
                w: rect.w,
                h: rect.h,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let uni = uniform_buf(&self.device, &self.queue, bytemuck::bytes_of(&params));
            let groups = (rect.w as usize * rect.h as usize).div_ceil(64) as u32;
            dispatch(
                self,
                &self.paste,
                &self.paste_layout,
                &[&buf, &src, &uni],
                groups,
                1,
            );
        }
        let raw = readback(&self.device, &self.queue, &buf, (n * 4) as u64)?;
        let words: &[u32] = bytemuck::cast_slice(&raw[..n * 4]);
        for (i, &w) in words.iter().enumerate() {
            let o = i * 3;
            canvas[o] = (w & 0xff) as u8;
            canvas[o + 1] = ((w >> 8) & 0xff) as u8;
            canvas[o + 2] = ((w >> 16) & 0xff) as u8;
        }
        Ok(())
    }
}

fn pad4(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    while v.len() % 4 != 0 {
        v.push(0);
    }
    v
}
