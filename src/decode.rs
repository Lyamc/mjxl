//! Decode an `.mjxl` sample stream back to packed RGB frames.

use crate::container::{self, Header, Sample};
use crate::error::{Error, Result};
use crate::gpu::RgbRect;
use crate::gpu::{self, Backend};
use crate::still;
use crate::temporal;

#[derive(Clone, Copy, Debug)]
pub struct DecodeOpts {
    /// `None` or `>= 1` decodes each sample fully. Smaller values render a prefix.
    pub prefix: Option<f32>,
    pub backend: Backend,
}

impl Default for DecodeOpts {
    fn default() -> Self {
        Self {
            prefix: None,
            backend: Backend::Auto,
        }
    }
}

pub struct Decoded {
    pub header: Header,
    pub frames: Vec<Vec<u8>>,
}

pub fn decode_rgb(file: &[u8], opts: &DecodeOpts) -> Result<Decoded> {
    let header = container::read_header(file)?;
    let groups = container::display_frames(&header);
    // Full-frame samples do not share a canvas. Decode them together; a single
    // 960×540 VarDCT frame does not fill the machine.
    let progressive = opts.prefix.is_some_and(|f| f < 1.0);
    if !progressive && groups_are_full_frames(&header, &groups) {
        let frames = decode_full_frames(file, &groups, opts)?;
        return Ok(Decoded { header, frames });
    }
    let gpu = gpu::select(opts.backend)?;
    let mut frames = Vec::new();
    let mut canvas = vec![0u8; header.width as usize * header.height as usize * 3];
    for group in groups {
        let mut pending: Vec<(u32, u32, u32, u32, Vec<u8>)> = Vec::new();
        for sample in group {
            if sample.size == 0 {
                continue;
            }
            let bytes = container::sample_bytes(file, sample)?;
            let rgb = render_sample(bytes, sample, opts.prefix)?;
            if let Some(rgb) = rgb {
                if sample.w == header.width
                    && sample.h == header.height
                    && sample.x == 0
                    && sample.y == 0
                {
                    canvas = rgb;
                    pending.clear();
                } else {
                    pending.push((sample.x, sample.y, sample.w, sample.h, rgb));
                }
            }
        }
        if !pending.is_empty() {
            let use_gpu = gpu.as_ref().filter(|_| {
                header.width as u64 * header.height as u64 >= 160 * 160
                    || opts.backend == Backend::Gpu
            });
            if let Some(gpu) = use_gpu {
                let rects: Vec<RgbRect<'_>> = pending
                    .iter()
                    .map(|(x, y, w, h, rgb)| RgbRect {
                        x: *x,
                        y: *y,
                        w: *w,
                        h: *h,
                        rgb,
                    })
                    .collect();
                match gpu.paste_rects(&mut canvas, header.width, header.height, &rects) {
                    Ok(()) => {}
                    Err(e) if opts.backend == Backend::Gpu => return Err(e),
                    Err(e) => {
                        eprintln!("mjxl: GPU paste failed ({e}); using CPU");
                        for (x, y, w, h, rgb) in &pending {
                            temporal::paste_rgb(&mut canvas, header.width, *x, *y, *w, *h, rgb);
                        }
                    }
                }
            } else {
                for (x, y, w, h, rgb) in &pending {
                    temporal::paste_rgb(&mut canvas, header.width, *x, *y, *w, *h, rgb);
                }
            }
        }
        frames.push(canvas.clone());
    }
    Ok(Decoded { header, frames })
}

fn groups_are_full_frames(header: &Header, groups: &[Vec<&Sample>]) -> bool {
    !groups.is_empty()
        && groups.iter().all(|g| {
            g.len() == 1
                && g[0].size > 0
                && g[0].x == 0
                && g[0].y == 0
                && g[0].w == header.width
                && g[0].h == header.height
        })
}

fn decode_full_frames(
    file: &[u8],
    groups: &[Vec<&Sample>],
    opts: &DecodeOpts,
) -> Result<Vec<Vec<u8>>> {
    use rayon::prelude::*;
    groups
        .par_iter()
        .map(|g| {
            let sample = g[0];
            let bytes = container::sample_bytes(file, sample)?;
            render_sample(bytes, sample, opts.prefix)?
                .ok_or_else(|| Error::from("full-frame sample decoded empty"))
        })
        .collect()
}

fn render_sample(bytes: &[u8], sample: &Sample, prefix: Option<f32>) -> Result<Option<Vec<u8>>> {
    let (w, h, rgb) = if let Some(fraction) = prefix.filter(|f| *f < 1.0) {
        match still::decode_jxl_prefix(bytes, fraction)? {
            Some(v) => v,
            None => return Ok(None),
        }
    } else {
        still::decode_jxl_rgb(bytes)?
    };
    if w != sample.w || h != sample.h {
        return Err(Error::from(format!(
            "jxl sample is {w}×{h}, index says {}×{}",
            sample.w, sample.h
        )));
    }
    Ok(Some(rgb))
}
