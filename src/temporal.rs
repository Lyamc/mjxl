//! Open-loop (or optional closed-loop) skip and dirty rectangles.
//!
//! A frame is either a full still, a list of changed rectangles, or a skip.
//! Rectangles are independent JPEG XL codestreams pasted onto the previous
//! canvas. There is no motion-vector copy: that needs JXL patches.

use o3_jxl::{Encoder, PixelFormat};

use crate::error::{Error, Result};
use crate::gpu::{self, Backend, Gpu};
use crate::still;

/// Tile edge for the dirty search. Rectangles are aligned to this grid.
pub const TILE: u32 = 16;
/// Mean absolute error per RGB sample at or below this is "unchanged".
pub const SKIP_MAE: u32 = 4;
/// More dirty area than this (8/10 of the frame) is sent as a full frame.
const FULL_NUM: u64 = 8;
const FULL_DEN: u64 = 10;
/// Above this many rectangles, collapse to one bounding box.
const MAX_RECTS: usize = 8;

/// Where one coded rectangle lands on the canvas.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Patch {
    pub jxl: Vec<u8>,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// One displayed frame. Empty `patches` repeats the previous canvas.
#[derive(Clone, Debug)]
pub struct Coded {
    pub duration: u32,
    pub patches: Vec<Patch>,
}

impl Coded {
    pub fn skip(duration: u32, canvas_w: u32, canvas_h: u32) -> Self {
        let _ = (canvas_w, canvas_h);
        Self {
            duration,
            patches: Vec::new(),
        }
    }

    pub fn is_skip(&self) -> bool {
        self.patches.is_empty()
    }

    pub fn is_full(&self, canvas_w: u32, canvas_h: u32) -> bool {
        matches!(
            self.patches.as_slice(),
            [p] if p.x == 0 && p.y == 0 && p.w == canvas_w && p.h == canvas_h
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Dirty {
    Skip,
    Full,
    Rects(Vec<Rect>),
}

#[derive(Clone, Copy, Debug)]
pub struct SequenceOpts {
    pub intra: bool,
    pub closed_loop: bool,
    pub backend: Backend,
}

impl Default for SequenceOpts {
    fn default() -> Self {
        Self {
            intra: false,
            closed_loop: false,
            backend: Backend::Auto,
        }
    }
}

pub fn tile_mask(prev: &[u8], curr: &[u8], width: u32, height: u32) -> (Vec<u8>, usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let tile = TILE as usize;
    let tw = w.div_ceil(tile);
    let th = h.div_ceil(tile);
    let mut mask = vec![0u8; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(w);
            let y1 = (y0 + tile).min(h);
            let mut sad = 0u32;
            let mut n = 0u32;
            for y in y0..y1 {
                let row = y * w;
                for x in x0..x1 {
                    let i = (row + x) * 3;
                    sad += prev[i].abs_diff(curr[i]) as u32;
                    sad += prev[i + 1].abs_diff(curr[i + 1]) as u32;
                    sad += prev[i + 2].abs_diff(curr[i + 2]) as u32;
                    n += 3;
                }
            }
            if n > 0 && sad / n > SKIP_MAE {
                mask[ty * tw + tx] = 1;
            }
        }
    }
    (mask, tw, th)
}

pub fn plan_from_mask(mask: &[u8], tw: usize, th: usize, width: u32, height: u32) -> Dirty {
    if tw == 0 || th == 0 || mask.iter().all(|&m| m == 0) {
        return Dirty::Skip;
    }
    let mut rects: Vec<TileRun> = Vec::new();
    for ty in 0..th {
        let mut tx = 0;
        while tx < tw {
            if mask[ty * tw + tx] == 0 {
                tx += 1;
                continue;
            }
            let tx0 = tx;
            while tx < tw && mask[ty * tw + tx] != 0 {
                tx += 1;
            }
            if let Some(r) = rects
                .iter_mut()
                .find(|r| r.tx0 == tx0 && r.tx1 == tx && r.ty1 == ty)
            {
                r.ty1 = ty + 1;
            } else {
                rects.push(TileRun {
                    tx0,
                    tx1: tx,
                    ty0: ty,
                    ty1: ty + 1,
                });
            }
        }
    }
    let mut px: Vec<Rect> = rects
        .into_iter()
        .map(|r| tile_rect(r, width, height))
        .filter(|r| r.w > 0 && r.h > 0)
        .collect();
    px = merge_close(px);
    if px.is_empty() {
        return Dirty::Skip;
    }
    let full = width as u64 * height as u64;
    let sum: u64 = px.iter().map(|r| r.w as u64 * r.h as u64).sum();
    if full == 0 || sum.saturating_mul(FULL_DEN) > full.saturating_mul(FULL_NUM) {
        return Dirty::Full;
    }
    if px.len() > MAX_RECTS {
        let bbox = px.iter().copied().reduce(union_rect).unwrap();
        let area = bbox.w as u64 * bbox.h as u64;
        if area.saturating_mul(FULL_DEN) > full.saturating_mul(FULL_NUM) {
            return Dirty::Full;
        }
        return Dirty::Rects(vec![bbox]);
    }
    Dirty::Rects(px)
}

struct TileRun {
    tx0: usize,
    tx1: usize,
    ty0: usize,
    ty1: usize,
}

fn tile_rect(r: TileRun, width: u32, height: u32) -> Rect {
    let x = (r.tx0 as u32).saturating_mul(TILE).min(width);
    let y = (r.ty0 as u32).saturating_mul(TILE).min(height);
    let x1 = (r.tx1 as u32).saturating_mul(TILE).min(width);
    let y1 = (r.ty1 as u32).saturating_mul(TILE).min(height);
    Rect {
        x,
        y,
        w: x1.saturating_sub(x),
        h: y1.saturating_sub(y),
    }
}

fn union_rect(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

fn merge_close(mut rects: Vec<Rect>) -> Vec<Rect> {
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < rects.len() {
            let mut j = i + 1;
            while j < rects.len() {
                if gap_small(rects[i], rects[j]) {
                    rects[i] = union_rect(rects[i], rects[j]);
                    rects.swap_remove(j);
                    changed = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    rects
}

fn gap_small(a: Rect, b: Rect) -> bool {
    let gap_x = axis_gap(a.x, a.w, b.x, b.w);
    let gap_y = axis_gap(a.y, a.h, b.y, b.h);
    if gap_x > TILE || gap_y > TILE {
        return false;
    }
    let u = union_rect(a, b);
    let extra = u.w as u64 * u.h as u64 - a.w as u64 * a.h as u64 - b.w as u64 * b.h as u64;
    let sum = a.w as u64 * a.h as u64 + b.w as u64 * b.h as u64;
    // A corner-only touch (the two diagonal tiles) leaves as much empty as it fills.
    extra * 2 <= sum
}

fn axis_gap(a0: u32, a_len: u32, b0: u32, b_len: u32) -> u32 {
    let a1 = a0 + a_len;
    let b1 = b0 + b_len;
    if a1 <= b0 {
        b0 - a1
    } else if b1 <= a0 {
        a0 - b1
    } else {
        0
    }
}

pub fn analyze(prev: &[u8], curr: &[u8], width: u32, height: u32) -> Dirty {
    if prev == curr {
        return Dirty::Skip;
    }
    let (mask, tw, th) = tile_mask(prev, curr, width, height);
    plan_from_mask(&mask, tw, th, width, height)
}

fn analyze_with(
    gpu: Option<&Gpu>,
    backend: Backend,
    prev: &[u8],
    curr: &[u8],
    width: u32,
    height: u32,
) -> Result<Dirty> {
    if prev == curr {
        return Ok(Dirty::Skip);
    }
    if let Some(gpu) = gpu {
        match gpu.dirty_tiles(prev, curr, width, height) {
            Ok((mask, tw, th)) => return Ok(plan_from_mask(&mask, tw, th, width, height)),
            Err(e) if backend == Backend::Gpu => return Err(e),
            Err(e) => eprintln!("mjxl: GPU dirty search failed ({e}); using CPU"),
        }
    }
    Ok(analyze(prev, curr, width, height))
}

pub fn crop_rgb(rgb: &[u8], stride_w: u32, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let sw = stride_w as usize;
    let mut out = vec![0u8; w as usize * h as usize * 3];
    for row in 0..h as usize {
        let src = ((y as usize + row) * sw + x as usize) * 3;
        let dst = row * w as usize * 3;
        out[dst..dst + w as usize * 3].copy_from_slice(&rgb[src..src + w as usize * 3]);
    }
    out
}

pub fn paste_rgb(dst: &mut [u8], stride_w: u32, x: u32, y: u32, w: u32, h: u32, src: &[u8]) {
    let sw = stride_w as usize;
    for row in 0..h as usize {
        let d = ((y as usize + row) * sw + x as usize) * 3;
        let s = row * w as usize * 3;
        dst[d..d + w as usize * 3].copy_from_slice(&src[s..s + w as usize * 3]);
    }
}

pub fn encode_sequence(
    enc: &Encoder,
    width: u32,
    height: u32,
    frames: &[Vec<u8>],
    opts: &SequenceOpts,
) -> Result<Vec<Coded>> {
    let gpu = gpu::select(opts.backend)?;
    let mut out = Vec::with_capacity(frames.len());
    let mut prev: Option<Vec<u8>> = None;
    for rgb in frames {
        let coded = if opts.intra || prev.is_none() {
            encode_full(enc, width, height, rgb)?
        } else {
            let prev_rgb = prev.as_deref().unwrap();
            match analyze_with(gpu.as_deref(), opts.backend, prev_rgb, rgb, width, height)? {
                Dirty::Skip => Coded::skip(1, width, height),
                Dirty::Full => encode_full(enc, width, height, rgb)?,
                Dirty::Rects(rects) => encode_rects(enc, width, rgb, &rects)?,
            }
        };
        prev = Some(if opts.closed_loop {
            reconstruct(width, height, prev.as_deref(), &coded)?
        } else {
            rgb.clone()
        });
        out.push(coded);
    }
    Ok(out)
}

fn encode_full(enc: &Encoder, width: u32, height: u32, rgb: &[u8]) -> Result<Coded> {
    let jxl = enc.encode(width, height, PixelFormat::Rgb, rgb)?;
    Ok(Coded {
        duration: 1,
        patches: vec![Patch {
            jxl,
            x: 0,
            y: 0,
            w: width,
            h: height,
        }],
    })
}

fn encode_rects(enc: &Encoder, stride: u32, rgb: &[u8], rects: &[Rect]) -> Result<Coded> {
    let mut patches = Vec::with_capacity(rects.len());
    for r in rects {
        let crop = crop_rgb(rgb, stride, r.x, r.y, r.w, r.h);
        let jxl = enc.encode(r.w, r.h, PixelFormat::Rgb, &crop)?;
        patches.push(Patch {
            jxl,
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        });
    }
    if patches.is_empty() {
        return Ok(Coded::skip(1, stride, 0));
    }
    Ok(Coded {
        duration: 1,
        patches,
    })
}

fn reconstruct(width: u32, height: u32, prev: Option<&[u8]>, coded: &Coded) -> Result<Vec<u8>> {
    let mut canvas = if coded.is_full(width, height) {
        vec![0u8; width as usize * height as usize * 3]
    } else {
        prev.map(|p| p.to_vec())
            .unwrap_or_else(|| vec![0u8; width as usize * height as usize * 3])
    };
    if coded.is_skip() {
        return Ok(canvas);
    }
    for p in &coded.patches {
        let (_w, _h, rgb) = still::decode_jxl_rgb(&p.jxl)?;
        if _w != p.w || _h != p.h {
            return Err(Error::from(format!(
                "reconstructed patch {}×{} != {}×{}",
                _w, _h, p.w, p.h
            )));
        }
        paste_rgb(&mut canvas, width, p.x, p.y, p.w, p.h, &rgb);
    }
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&rgb);
        }
        v
    }

    #[test]
    fn identical_frames_skip() {
        let a = solid(32, 32, [8, 9, 10]);
        assert_eq!(analyze(&a, &a, 32, 32), Dirty::Skip);
    }

    #[test]
    fn diagonal_tiles_stay_two_rects() {
        let a = solid(32, 32, [0, 0, 0]);
        let mut b = a.clone();
        // Top-left tile and bottom-right tile.
        for y in 0..16 {
            for x in 0..16 {
                let i = ((y * 32 + x) * 3) as usize;
                b[i] = 255;
            }
        }
        for y in 16..32 {
            for x in 16..32 {
                let i = ((y * 32 + x) * 3) as usize;
                b[i] = 255;
            }
        }
        match analyze(&a, &b, 32, 32) {
            Dirty::Rects(r) => {
                assert_eq!(r.len(), 2, "{r:?}");
                assert_eq!(
                    r[0],
                    Rect {
                        x: 0,
                        y: 0,
                        w: 16,
                        h: 16
                    }
                );
                assert_eq!(
                    r[1],
                    Rect {
                        x: 16,
                        y: 16,
                        w: 16,
                        h: 16
                    }
                );
            }
            other => panic!("expected two rects, got {other:?}"),
        }
    }

    #[test]
    fn almost_full_frame_collapses() {
        let a = solid(32, 32, [0, 0, 0]);
        let mut b = a.clone();
        for px in b.iter_mut().take(32 * 28 * 3) {
            *px = 40;
        }
        assert_eq!(analyze(&a, &b, 32, 32), Dirty::Full);
    }

    #[test]
    fn lossless_modular_roundtrip_with_two_rects() {
        let enc = Encoder::new().distance(0.0).effort(3).modular(Some(true));
        let f0 = solid(32, 32, [10, 20, 30]);
        let mut f1 = f0.clone();
        for y in 0..16 {
            for x in 0..16 {
                let i = ((y * 32 + x) * 3) as usize;
                f1[i] = 200;
                f1[i + 1] = 10;
                f1[i + 2] = 10;
            }
        }
        for y in 16..32 {
            for x in 16..32 {
                let i = ((y * 32 + x) * 3) as usize;
                f1[i] = 10;
                f1[i + 1] = 10;
                f1[i + 2] = 200;
            }
        }
        let f2 = f1.clone();
        let coded = encode_sequence(
            &enc,
            32,
            32,
            &[f0.clone(), f1.clone(), f2],
            &SequenceOpts {
                intra: false,
                closed_loop: true,
                backend: Backend::Cpu,
            },
        )
        .unwrap();
        assert!(coded[0].is_full(32, 32));
        assert_eq!(coded[1].patches.len(), 2, "{:?}", coded[1].patches);
        assert!(coded[2].is_skip());
        let mut canvas = vec![0u8; 32 * 32 * 3];
        for (frame, expect) in coded.iter().zip([&f0, &f1, &f1]) {
            for p in &frame.patches {
                let (w, h, rgb) = still::decode_jxl_rgb(&p.jxl).unwrap();
                assert_eq!((w, h), (p.w, p.h));
                paste_rgb(&mut canvas, 32, p.x, p.y, p.w, p.h, &rgb);
            }
            assert_eq!(&canvas, expect);
        }
    }
}
