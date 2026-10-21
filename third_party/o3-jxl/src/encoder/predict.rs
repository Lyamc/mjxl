//! SIMD reversible colour transform and gradient residuals.

use std::simd::prelude::*;

use super::entropy::{ResNode, ResTree, hybrid_uint_400, pack_signed, unpack_signed};

type I32x4 = Simd<i32, 4>;
type U32x4 = Simd<u32, 4>;
type I32x8 = Simd<i32, 8>;

/// YCoCg-R (JPEG XL RCT type 6), in-place on planar i32.
///
/// Forward:
/// `Co = R - B`, `tmp = B + (Co >> 1)`, `Cg = G - tmp`, `Y = tmp + (Cg >> 1)`.
pub fn ycocg_fwd_planes(y: &mut [i32], co: &mut [i32], cg: &mut [i32]) {
    let n = y.len();
    assert_eq!(co.len(), n);
    assert_eq!(cg.len(), n);

    let mut i = 0;
    while i + 8 <= n {
        let r = I32x8::from_slice(&y[i..]);
        let g = I32x8::from_slice(&co[i..]);
        let b = I32x8::from_slice(&cg[i..]);
        let (yy, cco, ccg) = ycocg_fwd_lanes(r, g, b);
        yy.copy_to_slice(&mut y[i..]);
        cco.copy_to_slice(&mut co[i..]);
        ccg.copy_to_slice(&mut cg[i..]);
        i += 8;
    }
    while i + 4 <= n {
        let r = I32x4::from_slice(&y[i..]);
        let g = I32x4::from_slice(&co[i..]);
        let b = I32x4::from_slice(&cg[i..]);
        let (yy, cco, ccg) = ycocg_fwd_lanes(r, g, b);
        yy.copy_to_slice(&mut y[i..]);
        cco.copy_to_slice(&mut co[i..]);
        ccg.copy_to_slice(&mut cg[i..]);
        i += 4;
    }
    for k in i..n {
        let (yy, cco, ccg) = ycocg_fwd_scalar(y[k], co[k], cg[k]);
        y[k] = yy;
        co[k] = cco;
        cg[k] = ccg;
    }
}

#[inline]
pub fn ycocg_fwd_scalar(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r.wrapping_sub(b);
    let tmp = b.wrapping_add(co >> 1);
    let cg = g.wrapping_sub(tmp);
    let y = tmp.wrapping_add(cg >> 1);
    (y, co, cg)
}

#[inline]
#[allow(dead_code)]
pub fn ycocg_inv_scalar(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
    let tmp = y.wrapping_sub(cg >> 1);
    let g = cg.wrapping_add(tmp);
    let b = tmp.wrapping_sub(co >> 1);
    let r = b.wrapping_add(co);
    (r, g, b)
}

#[inline]
fn ycocg_fwd_lanes<const N: usize>(
    r: Simd<i32, N>,
    g: Simd<i32, N>,
    b: Simd<i32, N>,
) -> (Simd<i32, N>, Simd<i32, N>, Simd<i32, N>) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}

/// Deinterleave 8-bit RGB(A) into planar i32, then apply YCoCg-R when `rgb`.
///
/// Channel count 1/2: gray / gray+alpha (no RCT).
/// Channel count 3/4: RGB(A) with YCoCg-R on the first three planes.
pub fn planarize_and_transform(
    width: usize,
    height: usize,
    channels: usize,
    pixels: &[u8],
) -> Vec<Vec<i32>> {
    assert!(matches!(channels, 1..=4));
    let npx = width * height;
    assert_eq!(pixels.len(), npx * channels);

    let mut planes = vec![vec![0i32; npx]; channels];
    match channels {
        1 => {
            for (i, &p) in pixels.iter().enumerate() {
                planes[0][i] = p as i32;
            }
        }
        2 => deinterleave_n::<2>(pixels, npx, &mut planes),
        3 => {
            deinterleave_rgb8_simd(pixels, npx, &mut planes);
            let (y, rest) = planes.split_at_mut(1);
            let (co, rest) = rest.split_at_mut(1);
            ycocg_fwd_planes(&mut y[0], &mut co[0], &mut rest[0]);
        }
        4 => {
            deinterleave_rgba8_simd(pixels, npx, &mut planes);
            let (y, rest) = planes.split_at_mut(1);
            let (co, rest) = rest.split_at_mut(1);
            ycocg_fwd_planes(&mut y[0], &mut co[0], &mut rest[0]);
        }
        _ => unreachable!(),
    }
    planes
}

/// Deinterleave 16-bit packed samples (native endian) into planar i32 + YCoCg-R.
pub fn planarize_and_transform_u16(
    width: usize,
    height: usize,
    channels: usize,
    pixels: &[u16],
) -> Vec<Vec<i32>> {
    assert!(matches!(channels, 1..=4));
    let npx = width * height;
    assert_eq!(pixels.len(), npx * channels);
    let mut planes = vec![vec![0i32; npx]; channels];
    let mut i = 0;
    while i + 4 <= npx {
        for k in 0..4 {
            let o = (i + k) * channels;
            for c in 0..channels {
                planes[c][i + k] = pixels[o + c] as i32;
            }
        }
        i += 4;
    }
    while i < npx {
        let o = i * channels;
        for c in 0..channels {
            planes[c][i] = pixels[o + c] as i32;
        }
        i += 1;
    }
    if channels >= 3 {
        let (y, rest) = planes.split_at_mut(1);
        let (co, rest) = rest.split_at_mut(1);
        ycocg_fwd_planes(&mut y[0], &mut co[0], &mut rest[0]);
    }
    planes
}

fn deinterleave_n<const N: usize>(pixels: &[u8], npx: usize, planes: &mut [Vec<i32>]) {
    let mut i = 0;
    while i + 4 <= npx {
        for k in 0..4 {
            let o = (i + k) * N;
            for c in 0..N {
                planes[c][i + k] = pixels[o + c] as i32;
            }
        }
        i += 4;
    }
    while i < npx {
        let o = i * N;
        for c in 0..N {
            planes[c][i] = pixels[o + c] as i32;
        }
        i += 1;
    }
}

fn deinterleave_rgb8_simd(pixels: &[u8], npx: usize, planes: &mut [Vec<i32>]) {
    let mut i = 0;
    while i + 4 <= npx {
        let o = i * 3;
        let r = I32x4::from_array([
            pixels[o] as i32,
            pixels[o + 3] as i32,
            pixels[o + 6] as i32,
            pixels[o + 9] as i32,
        ]);
        let g = I32x4::from_array([
            pixels[o + 1] as i32,
            pixels[o + 4] as i32,
            pixels[o + 7] as i32,
            pixels[o + 10] as i32,
        ]);
        let b = I32x4::from_array([
            pixels[o + 2] as i32,
            pixels[o + 5] as i32,
            pixels[o + 8] as i32,
            pixels[o + 11] as i32,
        ]);
        r.copy_to_slice(&mut planes[0][i..]);
        g.copy_to_slice(&mut planes[1][i..]);
        b.copy_to_slice(&mut planes[2][i..]);
        i += 4;
    }
    while i < npx {
        let o = i * 3;
        planes[0][i] = pixels[o] as i32;
        planes[1][i] = pixels[o + 1] as i32;
        planes[2][i] = pixels[o + 2] as i32;
        i += 1;
    }
}

fn deinterleave_rgba8_simd(pixels: &[u8], npx: usize, planes: &mut [Vec<i32>]) {
    let mut i = 0;
    while i + 4 <= npx {
        let o = i * 4;
        let px = U32x4::from_array([
            u32::from_le_bytes(pixels[o..o + 4].try_into().unwrap()),
            u32::from_le_bytes(pixels[o + 4..o + 8].try_into().unwrap()),
            u32::from_le_bytes(pixels[o + 8..o + 12].try_into().unwrap()),
            u32::from_le_bytes(pixels[o + 12..o + 16].try_into().unwrap()),
        ]);
        let r = (px & U32x4::splat(0xFF)).cast::<i32>();
        let g = ((px >> 8) & U32x4::splat(0xFF)).cast::<i32>();
        let b = ((px >> 16) & U32x4::splat(0xFF)).cast::<i32>();
        let a = ((px >> 24) & U32x4::splat(0xFF)).cast::<i32>();
        let r: [i32; 4] = r.to_array();
        let g: [i32; 4] = g.to_array();
        let b: [i32; 4] = b.to_array();
        let a: [i32; 4] = a.to_array();
        planes[0][i..i + 4].copy_from_slice(&r);
        planes[1][i..i + 4].copy_from_slice(&g);
        planes[2][i..i + 4].copy_from_slice(&b);
        planes[3][i..i + 4].copy_from_slice(&a);
        i += 4;
    }
    for (j, px) in pixels.as_chunks::<4>().0.iter().enumerate().skip(i) {
        planes[0][j] = px[0] as i32;
        planes[1][j] = px[1] as i32;
        planes[2][j] = px[2] as i32;
        planes[3][j] = px[3] as i32;
    }
}

/// JPEG XL Gradient: `clamp(N + W - NW, min(N, W), max(N, W))`.
#[inline]
pub fn gradient(n: i32, w: i32, nw: i32) -> i32 {
    (n + w - nw).clamp(w.min(n), w.max(n))
}

#[inline]
fn div_round(n: i32, d: i32) -> i32 {
    if d <= 1 {
        n
    } else if n >= 0 {
        (n + d / 2) / d
    } else {
        -((n.wrapping_neg() + d / 2) / d)
    }
}

/// Packed-signed Gradient residuals for one plane (row-major).
///
/// `quant == 1` is lossless. `quant > 1` predicts from reconstructed samples
/// so the decoder's Gradient + MA multiplier stay in lockstep.
pub fn gradient_residuals(width: usize, height: usize, plane: &[i32], quant: u32, out: &mut [u32]) {
    assert_eq!(plane.len(), width * height);
    assert_eq!(out.len(), width * height);
    if width == 0 || height == 0 {
        return;
    }
    if quant > 1 {
        gradient_residuals_quant(width, height, plane, quant as i32, out);
        return;
    }

    // First row: predictor is W (west), NW=N=0.
    {
        let row = &plane[0..width];
        let dst = &mut out[0..width];
        let mut w = 0i32;
        let mut x = 0;
        while x + 8 <= width {
            let px = I32x8::from_slice(&row[x..]);
            // Sequential west: pred[k] = pixel[k-1] (or 0). First row pred = W.
            let west = I32x8::from_array([
                w,
                row[x],
                row[x + 1],
                row[x + 2],
                row[x + 3],
                row[x + 4],
                row[x + 5],
                row[x + 6],
            ]);
            pack_signed_lanes(px - west).copy_to_slice(&mut dst[x..]);
            w = row[x + 7];
            x += 8;
        }
        while x + 4 <= width {
            let px = I32x4::from_slice(&row[x..]);
            let west = I32x4::from_array([w, row[x], row[x + 1], row[x + 2]]);
            pack_signed_lanes(px - west).copy_to_slice(&mut dst[x..]);
            w = row[x + 3];
            x += 4;
        }
        for k in x..width {
            let px = row[k];
            dst[k] = pack_signed(px - w);
            w = px;
        }
    }

    for y in 1..height {
        let prev = &plane[(y - 1) * width..y * width];
        let row = &plane[y * width..(y + 1) * width];
        let dst = &mut out[y * width..(y + 1) * width];

        // x = 0: w = 0, nw = 0, pred = n.
        dst[0] = pack_signed(row[0] - prev[0]);

        let mut x = 1;
        while x + 8 <= width {
            let px = I32x8::from_slice(&row[x..]);
            let n = I32x8::from_slice(&prev[x..]);
            let w = I32x8::from_slice(&row[x - 1..]);
            let nw = I32x8::from_slice(&prev[x - 1..]);
            let pred = gradient_lanes(n, w, nw);
            pack_signed_lanes(px - pred).copy_to_slice(&mut dst[x..]);
            x += 8;
        }
        while x + 4 <= width {
            let px = I32x4::from_slice(&row[x..]);
            let n = I32x4::from_slice(&prev[x..]);
            let w = I32x4::from_slice(&row[x - 1..]);
            let nw = I32x4::from_slice(&prev[x - 1..]);
            let pred = gradient_lanes(n, w, nw);
            pack_signed_lanes(px - pred).copy_to_slice(&mut dst[x..]);
            x += 4;
        }
        for k in x..width {
            dst[k] = pack_signed(row[k] - gradient(prev[k], row[k - 1], prev[k - 1]));
        }
    }
}

fn gradient_residuals_quant(
    width: usize,
    height: usize,
    plane: &[i32],
    quant: i32,
    out: &mut [u32],
) {
    let mut prev = vec![0i32; width];
    let mut curr = vec![0i32; width];
    for y in 0..height {
        let row = &plane[y * width..(y + 1) * width];
        let dst = &mut out[y * width..(y + 1) * width];
        for x in 0..width {
            let n = if y == 0 { 0 } else { prev[x] };
            let w = if x == 0 { 0 } else { curr[x - 1] };
            let nw = if x == 0 || y == 0 { 0 } else { prev[x - 1] };
            let pred = gradient(n, w, nw);
            let q = div_round(row[x] - pred, quant);
            dst[x] = pack_signed(q);
            curr[x] = pred + q * quant;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
}

/// Round samples to a multiple of `quant` (libjxl `QuantizeChannel`).
pub fn quantize_to_grid(plane: &mut [i32], quant: u32) {
    if quant <= 1 {
        return;
    }
    let q = quant as i32;
    let n = plane.len();
    let mut i = 0;
    while i + 8 <= n {
        let px = I32x8::from_slice(&plane[i..]);
        (div_round_lanes(px, q) * I32x8::splat(q)).copy_to_slice(&mut plane[i..]);
        i += 8;
    }
    while i + 4 <= n {
        let px = I32x4::from_slice(&plane[i..]);
        (div_round_lanes(px, q) * I32x4::splat(q)).copy_to_slice(&mut plane[i..]);
        i += 4;
    }
    for s in &mut plane[i..] {
        *s = div_round(*s, q) * q;
    }
}

/// Zero-predictor residuals: `pack_signed(div_round(sample, quant))`.
///
/// Squeeze HF is already decorrelated; Gradient on those planes wastes time
/// and bits. `quant == 1` is a SIMD pack of the samples themselves.
pub fn zero_residuals(plane: &[i32], quant: u32, out: &mut [u32]) {
    assert_eq!(plane.len(), out.len());
    if quant <= 1 {
        pack_plane(plane, out);
        return;
    }
    let q = quant as i32;
    let n = plane.len();
    let mut i = 0;
    while i + 8 <= n {
        pack_signed_lanes(div_round_lanes(I32x8::from_slice(&plane[i..]), q))
            .copy_to_slice(&mut out[i..]);
        i += 8;
    }
    while i + 4 <= n {
        pack_signed_lanes(div_round_lanes(I32x4::from_slice(&plane[i..]), q))
            .copy_to_slice(&mut out[i..]);
        i += 4;
    }
    for k in i..n {
        out[k] = pack_signed(div_round(plane[k], q));
    }
}

fn pack_plane(plane: &[i32], out: &mut [u32]) {
    let n = plane.len();
    let mut i = 0;
    while i + 8 <= n {
        pack_signed_lanes(I32x8::from_slice(&plane[i..])).copy_to_slice(&mut out[i..]);
        i += 8;
    }
    while i + 4 <= n {
        pack_signed_lanes(I32x4::from_slice(&plane[i..])).copy_to_slice(&mut out[i..]);
        i += 4;
    }
    for k in i..n {
        out[k] = pack_signed(plane[k]);
    }
}

#[inline]
fn div_round_lanes<const N: usize>(n: Simd<i32, N>, d: i32) -> Simd<i32, N> {
    let zero = Simd::splat(0);
    let dd = Simd::splat(d);
    let half = Simd::splat(d / 2);
    let neg = n.simd_lt(zero);
    let abs = neg.select(-n, n);
    let q = (abs + half) / dd;
    neg.select(-q, q)
}

/// Zero-predictor reconstructed sample as jxl-oxide stores it.
///
/// 8-bit frames set `modular_16bit_buffers`, so jxl-oxide uses `i16` samples
/// and wrapping multiply. `|N|` / `|W|` must see that value, not i32
/// `unpack * quant` — a product that wraps to 0 is still `|W| == 0`.
pub fn recon_zero_sample(residual: u32, quant: u32, narrow: bool) -> i32 {
    let q = quant.max(1);
    if narrow {
        let bit = (residual & 1) as u16;
        let base = (residual >> 1) as u16;
        let flip = 0u16.wrapping_sub(bit);
        let unpacked = (base ^ flip) as i16;
        i32::from(unpacked.wrapping_mul(q as i16))
    } else {
        unpack_signed(residual).wrapping_mul(q as i32)
    }
}

/// `N`/`W` at each pixel as jxl-oxide `PredictorState` sees them.
///
/// First row: `N` starts at 0 and then copies `W`. New row: `N = W =`
/// previous-row-first (not 0).
pub fn decoder_nw(width: usize, samples: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let (n, w, _) = decoder_nwn(width, samples);
    (n, w)
}

/// `N`, `W`, `NW` as jxl-oxide `PredictorState` sees them.
pub fn decoder_nwn(width: usize, samples: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = samples.len();
    let mut ns = vec![0i32; n];
    let mut ws = vec![0i32; n];
    let mut nws = vec![0i32; n];
    if width == 0 || n == 0 {
        return (ns, ws, nws);
    }
    let mut prev_row: Vec<i32> = Vec::new();
    let mut curr_row: Vec<i32> = Vec::new();
    let mut x = 0usize;
    let mut north = 0i32;
    let mut west = 0i32;
    let mut nw = 0i32;
    for (i, &sample) in samples.iter().enumerate() {
        ns[i] = north;
        ws[i] = west;
        nws[i] = nw;
        if x < curr_row.len() {
            curr_row[x] = sample;
        } else {
            curr_row.push(sample);
        }
        x += 1;
        if x >= width {
            x = 0;
            std::mem::swap(&mut prev_row, &mut curr_row);
            let first = prev_row[0];
            north = first;
            west = first;
            nw = first;
        } else if prev_row.is_empty() {
            west = sample;
            north = sample;
            nw = sample;
        } else {
            nw = north;
            west = sample;
            north = prev_row[x];
        }
    }
    (ns, ws, nws)
}

/// `N` at each pixel as jxl-oxide `PredictorState` sees it.
#[allow(dead_code)]
pub fn decoder_n_samples(width: usize, samples: &[i32]) -> Vec<i32> {
    decoder_nw(width, samples).0
}

/// Left-child (`prop > 0`) mask for a Zero-predictor MA split.
///
/// `prop` is the JPEG XL property index: `3` = x, `4` = `|N|`, `5` = `|W|`.
#[allow(dead_code)]
pub fn split_hi(prop: u8, width: usize, residuals: &[u32], quant: u32, narrow: bool) -> Vec<bool> {
    let n = residuals.len();
    if width == 0 {
        return vec![false; n];
    }
    match prop {
        3 => (0..n).map(|i| i % width > 0).collect(),
        4 | 5 => {
            let samples: Vec<i32> = residuals
                .iter()
                .map(|&r| recon_zero_sample(r, quant, narrow))
                .collect();
            let (ns, ws) = decoder_nw(width, &samples);
            let v = if prop == 4 { ns } else { ws };
            v.into_iter().map(|s| s.unsigned_abs() > 0).collect()
        }
        _ => vec![false; n],
    }
}

/// Two-level residual MA split. Quadrant `(!p0)*2 + (!p1)` like `|W|` then `|N|`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HfSplit {
    pub p0: u8,
    pub v0: i32,
    pub p1: u8,
    pub v1: i32,
}

impl Default for HfSplit {
    fn default() -> Self {
        Self {
            p0: 5,
            v0: 0,
            p1: 4,
            v1: 0,
        }
    }
}

/// Quadrant `(!|W|>0)*2 + (!|N|>0)` for a `|W|` then `|N|` MA split.
#[allow(dead_code)]
pub fn split_wn_quad(width: usize, residuals: &[u32], quant: u32, narrow: bool) -> Vec<u8> {
    split_hf_quad(width, residuals, quant, narrow, HfSplit::default())
}

pub fn split_hf_quad(
    width: usize,
    residuals: &[u32],
    quant: u32,
    narrow: bool,
    split: HfSplit,
) -> Vec<u8> {
    let samples: Vec<i32> = residuals
        .iter()
        .map(|&r| recon_zero_sample(r, quant, narrow))
        .collect();
    let (ns, ws) = decoder_nw(width, &samples);
    ns.into_iter()
        .zip(ws)
        .enumerate()
        .map(|(i, (n, w))| {
            let x = if width == 0 { 0 } else { i % width };
            let h0 = prop_at(split.p0, n, w, x) > split.v0;
            let h1 = prop_at(split.p1, n, w, x) > split.v1;
            u8::from(!h0) * 2 + u8::from(!h1)
        })
        .collect()
}

#[inline]
fn prop_at(prop: u8, n: i32, w: i32, x: usize) -> i32 {
    match prop {
        3 => x as i32,
        4 => n.unsigned_abs() as i32,
        5 => w.unsigned_abs() as i32,
        6 => n,
        7 => w,
        _ => 0,
    }
}

/// Greedy 2-level compact residual tree (properties 4/5/6, split −3..=3).
///
/// Compact MA uints only hold `pack_signed(val) ≤ 7`, so thresholds stay in
/// that range. Default `|W|>0` then `|N|>0` wins unless another pair cuts
/// token entropy enough to cover a few extra tree bits.
#[allow(dead_code)]
pub fn learn_hf_split(planes: &[(usize, &[u32], u32)], narrow: bool) -> HfSplit {
    const CAP: usize = 48_000;
    let mut abs_n = Vec::new();
    let mut abs_w = Vec::new();
    let mut sgn_n = Vec::new();
    let mut tok = Vec::new();
    let mut n_pix = 0usize;
    for &(width, res, quant) in planes {
        if width == 0 || res.is_empty() {
            continue;
        }
        n_pix += res.len();
        let stride = ((n_pix / CAP).max(1)).max(res.len() / (CAP / planes.len().max(1)).max(1));
        let samples: Vec<i32> = res
            .iter()
            .map(|&r| recon_zero_sample(r, quant, narrow))
            .collect();
        let (ns, ws) = decoder_nw(width, &samples);
        for (i, (&n, &w)) in ns.iter().zip(&ws).enumerate().step_by(stride.max(1)) {
            if tok.len() >= CAP {
                break;
            }
            abs_n.push(n.unsigned_abs() as i32);
            abs_w.push(w.unsigned_abs() as i32);
            sgn_n.push(n);
            tok.push(hybrid_uint_400(res[i]).0 as u8);
        }
        if tok.len() >= CAP {
            break;
        }
    }
    let def = HfSplit::default();
    if tok.len() < 512 {
        return def;
    }
    let pval = |prop: u8, i: usize| -> i32 {
        match prop {
            4 => abs_n[i],
            5 => abs_w[i],
            6 => sgn_n[i],
            _ => 0,
        }
    };
    let cost = |p0: u8, v0: i32, p1: u8, v1: i32| -> f64 {
        let mut h = [[0u32; 24]; 4];
        for (i, &t) in tok.iter().enumerate() {
            let q = u8::from(pval(p0, i) <= v0) * 2 + u8::from(pval(p1, i) <= v1);
            let t = t as usize;
            if t < 24 {
                h[q as usize][t] += 1;
            }
        }
        h.iter().map(|b| hist_bits(b)).sum()
    };
    let mut best = def;
    let mut best_c = cost(best.p0, best.v0, best.p1, best.v1);
    let def_c = best_c;
    const ABS: [i32; 7] = [0, 1, 2, 3, 4, 8, 16];
    const SGN: [i32; 7] = [-3, -2, -1, 0, 1, 2, 3];
    let props: [(u8, &[i32]); 3] = [(5, &ABS), (4, &ABS), (6, &SGN)];
    for &(p0, v0s) in &props {
        for &v0 in v0s {
            for &(p1, v1s) in &props {
                if p1 == p0 {
                    continue;
                }
                for &v1 in v1s {
                    let c = cost(p0, v0, p1, v1);
                    // Wide-tree split values (pack_signed > 7) must beat the
                    // compact default by enough to cover the extra MA header.
                    let wide = pack_signed(v0) > 7 || pack_signed(v1) > 7;
                    let margin = if wide { 8000.0 } else { 24.0 };
                    if c + margin < best_c {
                        best_c = c;
                        best = HfSplit { p0, v0, p1, v1 };
                    }
                }
            }
        }
    }
    if std::env::var_os("O3_JXL_DEBUG_SPLIT").is_some() {
        eprintln!(
            "hf_split p{}>{} then p{}>{}  {best_c:.0} bits (default {def_c:.0}, n={})",
            best.p0,
            best.v0,
            best.p1,
            best.v1,
            tok.len()
        );
    }
    best
}

/// Walk a residual MA tree; returns the leaf `id`.
pub fn res_leaf_id(tree: &ResTree, n: i32, w: i32, nw: i32, x: usize) -> u8 {
    let mut i = 0usize;
    for _ in 0..tree.nodes.len() {
        match tree.nodes[i] {
            ResNode::Leaf { id } => return id,
            ResNode::Split {
                prop,
                val,
                left,
                right,
            } => {
                i = if prop_full(prop, n, w, nw, x) > val {
                    left as usize
                } else {
                    right as usize
                };
            }
        }
    }
    0
}

pub fn split_res_tree(
    width: usize,
    residuals: &[u32],
    quant: u32,
    narrow: bool,
    tree: &ResTree,
) -> Vec<u8> {
    let samples: Vec<i32> = residuals
        .iter()
        .map(|&r| recon_zero_sample(r, quant, narrow))
        .collect();
    let (ns, ws, nws) = decoder_nwn(width, &samples);
    ns.into_iter()
        .zip(ws)
        .zip(nws)
        .enumerate()
        .map(|(i, ((n, w), nw))| {
            let x = if width == 0 { 0 } else { i % width };
            res_leaf_id(tree, n, w, nw, x)
        })
        .collect()
}

/// Greedy residual tree, at most 8 leaves. Squeeze properties 4, 5, 6, 9–11.
#[allow(dead_code)]
pub fn learn_res_tree(planes: &[(usize, &[u32], u32)], narrow: bool) -> ResTree {
    const CAP: usize = 32_000;
    const MAX_LEAVES: usize = 8;
    let mut absn = Vec::new();
    let mut absw = Vec::new();
    let mut sn = Vec::new();
    let mut sw = Vec::new();
    let mut grad = Vec::new();
    let mut w_nw = Vec::new();
    let mut nw_n = Vec::new();
    let mut tok = Vec::new();
    for &(width, res, quant) in planes {
        if width == 0 || res.is_empty() {
            continue;
        }
        let stride = (res.len() / 4096).max(1);
        let samples: Vec<i32> = res
            .iter()
            .map(|&r| recon_zero_sample(r, quant, narrow))
            .collect();
        let (ns, ws, nws) = decoder_nwn(width, &samples);
        for (i, ((&n, &w), &nwv)) in ns.iter().zip(&ws).zip(&nws).enumerate().step_by(stride) {
            if tok.len() >= CAP {
                break;
            }
            absn.push(n.unsigned_abs() as i32);
            absw.push(w.unsigned_abs() as i32);
            sn.push(n);
            sw.push(w);
            grad.push(w.wrapping_add(n).wrapping_sub(nwv));
            w_nw.push(w.wrapping_sub(nwv));
            nw_n.push(nwv.wrapping_sub(n));
            tok.push(hybrid_uint_400(res[i]).0.min(23) as u8);
        }
        if tok.len() >= CAP {
            break;
        }
    }
    if tok.len() < 1024 {
        return ResTree::default_wn();
    }
    let pval = |prop: u8, i: usize| -> i32 {
        match prop {
            4 => absn[i],
            5 => absw[i],
            6 => sn[i],
            7 => sw[i],
            9 => grad[i],
            10 => w_nw[i],
            11 => nw_n[i],
            _ => 0,
        }
    };
    let n = tok.len();
    let mut assign = vec![0u8; n];
    let mut nodes = vec![ResNode::Leaf { id: 0 }];
    let mut leaf_of = vec![0u8]; // node index of each leaf id
    let mut n_leaves = 1u8;
    let props: [u8; 3] = [5, 4, 6];
    while (n_leaves as usize) < MAX_LEAVES {
        let mut best: Option<(f64, u8, u8, i32, u8)> = None; // gain, leaf_id, prop, val, node
        for lid in 0..n_leaves {
            let mut idx = Vec::new();
            for (i, &a) in assign.iter().enumerate() {
                if a == lid {
                    idx.push(i);
                }
            }
            if idx.len() < 512 {
                continue;
            }
            let mut parent_h = [0u32; 24];
            for &i in &idx {
                parent_h[tok[i] as usize] += 1;
            }
            let parent_b = hist_bits(&parent_h);
            for &prop in &props {
                let mut vals: Vec<i32> = idx.iter().map(|&i| pval(prop, i)).collect();
                vals.sort_unstable();
                let cand = [
                    vals[0].clamp(-3, 3),
                    vals[vals.len() / 4].clamp(-3, 3),
                    vals[vals.len() / 2].clamp(-3, 3),
                    vals[vals.len() * 3 / 4].clamp(-3, 3),
                    0,
                ];
                for &val in &cand {
                    let mut lh = [0u32; 24];
                    let mut rh = [0u32; 24];
                    let mut ln = 0u32;
                    let mut rn = 0u32;
                    for &i in &idx {
                        if pval(prop, i) > val {
                            lh[tok[i] as usize] += 1;
                            ln += 1;
                        } else {
                            rh[tok[i] as usize] += 1;
                            rn += 1;
                        }
                    }
                    if ln < 256 || rn < 256 {
                        continue;
                    }
                    let gain = parent_b - hist_bits(&lh) - hist_bits(&rh);
                    if best.is_none_or(|(g, _, _, _, _)| gain > g) {
                        best = Some((gain, lid, prop, val, leaf_of[lid as usize]));
                    }
                }
            }
        }
        let Some((gain, lid, prop, val, node)) = best else {
            break;
        };
        if gain < 128.0 {
            break;
        }
        let left_id = n_leaves;
        let right_id = lid;
        n_leaves += 1;
        let left_node = nodes.len() as u8;
        let right_node = (nodes.len() + 1) as u8;
        nodes.push(ResNode::Leaf { id: left_id });
        nodes.push(ResNode::Leaf { id: right_id });
        nodes[node as usize] = ResNode::Split {
            prop,
            val,
            left: left_node,
            right: right_node,
        };
        leaf_of.push(left_node);
        leaf_of[right_id as usize] = right_node;
        for (i, a) in assign.iter_mut().enumerate() {
            if *a == lid && pval(prop, i) > val {
                *a = left_id;
            }
        }
    }
    if n_leaves < 2 {
        return ResTree::default_wn();
    }
    let mut tree = ResTree { nodes };
    tree.compact_ids();
    tree
}

#[inline]
fn prop_full(prop: u8, n: i32, w: i32, nw: i32, x: usize) -> i32 {
    match prop {
        3 => x as i32,
        4 => n.unsigned_abs() as i32,
        5 => w.unsigned_abs() as i32,
        6 => n,
        7 => w,
        9 => w.wrapping_add(n).wrapping_sub(nw),
        10 => w.wrapping_sub(nw),
        11 => nw.wrapping_sub(n),
        _ => 0,
    }
}

fn hist_bits(h: &[u32]) -> f64 {
    let n: u32 = h.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = f64::from(n);
    h.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = f64::from(c) / nf;
            -f64::from(c) * p.log2()
        })
        .sum()
}

#[inline]
fn pack_signed_lanes<const N: usize>(v: Simd<i32, N>) -> Simd<u32, N> {
    ((v << 1) ^ (v >> 31)).cast()
}

#[inline]
fn gradient_lanes<const N: usize>(
    n: Simd<i32, N>,
    w: Simd<i32, N>,
    nw: Simd<i32, N>,
) -> Simd<i32, N> {
    let sum = n + w - nw;
    let lo = n.simd_min(w);
    let hi = n.simd_max(w);
    sum.simd_clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_n_first_row_copies_w() {
        let samples = [0i32, 5, 0, 7, 0, 0];
        let n = decoder_n_samples(3, &samples);
        // y=0: N=0, then W. y=1: N is the row above.
        assert_eq!(n, vec![0, 0, 5, 0, 5, 0]);
    }

    #[test]
    fn abs_w_row_start_is_prev_row_first() {
        // residuals pack_signed(0)=0, pack_signed(1)=2, pack_signed(2)=4
        let res = [0u32, 2, 0, 4, 0, 0];
        let hi = split_hi(5, 3, &res, 1, true);
        // samples [0,1,0, 2,0,0]
        // W: 0, 0, 1,  0(row0 first), 2, 0
        assert_eq!(hi, vec![false, false, true, false, true, false]);
    }

    #[test]
    fn recon_narrow_wraps_like_i16() {
        // unpack(pack_signed(256))=256; 256*256 wraps to 0 in i16.
        let r = pack_signed(256);
        assert_eq!(recon_zero_sample(r, 256, true), 0);
        assert_eq!(recon_zero_sample(r, 256, false), 65536);
    }

    #[test]
    fn split_x_hi_is_first_column() {
        let res = [0u32; 6];
        let hi = split_hi(3, 3, &res, 1, true);
        assert_eq!(hi, vec![false, true, true, false, true, true]);
    }

    #[test]
    fn wn_quad_four_way() {
        // samples [0,1,0, 2,0,0]; W: 0,0,1, 0,2,0; N: 0,0,1, 0,1,0
        let res = [0u32, 2, 0, 4, 0, 0];
        let q = split_wn_quad(3, &res, 1, true);
        assert_eq!(q, vec![3, 3, 0, 3, 0, 3]);
        assert_eq!(q, split_hf_quad(3, &res, 1, true, HfSplit::default()));
    }

    #[test]
    fn learn_hf_split_keeps_default_on_tiny() {
        assert_eq!(learn_hf_split(&[], true), HfSplit::default());
        let res = [0u32; 8];
        assert_eq!(
            learn_hf_split(&[(4, res.as_slice(), 1)], true),
            HfSplit::default()
        );
    }

    #[test]
    fn ycocg_roundtrip() {
        for r in 0..16i32 {
            for g in 0..16i32 {
                for b in 0..16i32 {
                    let (y, co, cg) = ycocg_fwd_scalar(r * 16, g * 16, b * 16);
                    let (r2, g2, b2) = ycocg_inv_scalar(y, co, cg);
                    assert_eq!((r2, g2, b2), (r * 16, g * 16, b * 16));
                }
            }
        }
    }

    #[test]
    fn gradient_matches_scalar_8x8() {
        let w = 8usize;
        let h = 8usize;
        let mut plane = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = (x * 13 + y * 7) as i32;
            }
        }
        let mut out = vec![0u32; w * h];
        gradient_residuals(w, h, &plane, 1, &mut out);

        let mut expect = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let n = if y == 0 { 0 } else { plane[(y - 1) * w + x] };
                let ww = if x == 0 { 0 } else { plane[y * w + x - 1] };
                let nw = if x == 0 || y == 0 {
                    0
                } else {
                    plane[(y - 1) * w + x - 1]
                };
                let pred = gradient(n, ww, nw);
                expect[y * w + x] = pack_signed(plane[y * w + x] - pred);
            }
        }
        assert_eq!(out, expect);
    }

    #[test]
    fn zero_residuals_matches_scalar() {
        let plane: Vec<i32> = (-20..20).collect();
        for q in [1u32, 3, 7, 16] {
            let mut out = vec![0u32; plane.len()];
            zero_residuals(&plane, q, &mut out);
            for (i, &px) in plane.iter().enumerate() {
                assert_eq!(
                    out[i],
                    pack_signed(div_round(px, q as i32)),
                    "q={q} px={px}"
                );
            }
        }
    }
}
