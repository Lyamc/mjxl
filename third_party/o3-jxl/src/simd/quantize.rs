//! Linear float → sRGB u8 conversion and RGBA packing with `std::simd`.

use std::simd::StdFloat;
use std::simd::prelude::*;

type F32x4 = Simd<f32, 4>;
type F32x8 = Simd<f32, 8>;
type F32x16 = Simd<f32, 16>;
type U8x16 = Simd<u8, 16>;

const LINEAR_THRESHOLD: f32 = 0.003_130_8;
const LINEAR_SLOPE: f32 = 12.92;
const GAMMA_EXP: f32 = 1.0 / 2.4;
const SRGB_A: f32 = 1.055;
const SRGB_B: f32 = 0.055;

/// Planar 0..=1 RGB(A) already in the output transfer → packed RGBA8 (no extra OETF).
///
/// jxl-oxide applies the requested transfer before `color_channels()`, so decode
/// packing is a scale-and-saturate, not a second sRGB curve.
#[allow(clippy::too_many_arguments)]
pub fn planar_encoded_to_rgba(
    width: u32,
    height: u32,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    a: Option<&[f32]>,
    stride: usize,
    emit_linear: bool,
) -> (Vec<u8>, Vec<f32>, u32) {
    planar_to_rgba::<false>(width, height, r, g, b, a, stride, emit_linear)
}

/// Planar linear RGB(A) → packed RGBA8 (sRGB OETF) plus interleaved linear.
///
/// `stride` is samples per row in each plane (`>= width`). `a` is optional
/// linear alpha (opaque if `None`). `emit_linear` skips the interleaved buffer.
#[allow(clippy::too_many_arguments)]
pub fn planar_linear_to_rgba(
    width: u32,
    height: u32,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    a: Option<&[f32]>,
    stride: usize,
    emit_linear: bool,
) -> (Vec<u8>, Vec<f32>, u32) {
    planar_to_rgba::<true>(width, height, r, g, b, a, stride, emit_linear)
}

#[allow(clippy::too_many_arguments)]
fn planar_to_rgba<const OETF: bool>(
    width: u32,
    height: u32,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    a: Option<&[f32]>,
    stride: usize,
    emit_linear: bool,
) -> (Vec<u8>, Vec<f32>, u32) {
    let w = width as usize;
    let h = height as usize;
    debug_assert!(stride >= w);
    let npx = w * h;
    let ch = if a.is_some() { 4 } else { 3 };
    let mut rgba = vec![255u8; npx * 4];
    let mut linear = if emit_linear {
        vec![0f32; npx * ch]
    } else {
        Vec::new()
    };
    type U32x8 = Simd<u32, 8>;
    type U32x16 = Simd<u32, 16>;
    let pack_rgb_row = |y: usize, rgba_row: &mut [u8], mut lin_row: Option<&mut [f32]>| {
        let ps = y * stride;
        let mut x = 0;
        while a.is_none() && x + 16 <= w {
            let rv = F32x16::from_slice(&r[ps + x..]);
            let gv = F32x16::from_slice(&g[ps + x..]);
            let bv = F32x16::from_slice(&b[ps + x..]);
            let ru = sample_to_u8::<OETF, 16>(rv).cast::<u32>();
            let gu = sample_to_u8::<OETF, 16>(gv).cast::<u32>();
            let bu = sample_to_u8::<OETF, 16>(bv).cast::<u32>();
            let packed = ru | (gu << 8) | (bu << 16) | U32x16::splat(0xFF00_0000);
            store_u32x16_le(&mut rgba_row[x * 4..], packed);
            if let Some(lin) = lin_row.as_deref_mut() {
                let ra = rv.to_array();
                let ga = gv.to_array();
                let ba = bv.to_array();
                for k in 0..16 {
                    let lo = (x + k) * 3;
                    lin[lo] = ra[k];
                    lin[lo + 1] = ga[k];
                    lin[lo + 2] = ba[k];
                }
            }
            x += 16;
        }
        while x + 8 <= w {
            let rv = F32x8::from_slice(&r[ps + x..]);
            let gv = F32x8::from_slice(&g[ps + x..]);
            let bv = F32x8::from_slice(&b[ps + x..]);
            let ru = sample_to_u8::<OETF, 8>(rv).cast::<u32>();
            let gu = sample_to_u8::<OETF, 8>(gv).cast::<u32>();
            let bu = sample_to_u8::<OETF, 8>(bv).cast::<u32>();
            let packed = if let Some(ap) = a {
                let au = sample_to_u8::<OETF, 8>(F32x8::from_slice(&ap[ps + x..])).cast::<u32>();
                ru | (gu << 8) | (bu << 16) | (au << 24)
            } else {
                ru | (gu << 8) | (bu << 16) | U32x8::splat(0xFF00_0000)
            };
            store_u32x8_le(&mut rgba_row[x * 4..], packed);
            if let Some(lin) = lin_row.as_deref_mut() {
                let ra = rv.to_array();
                let ga = gv.to_array();
                let ba = bv.to_array();
                if let Some(ap) = a {
                    let aa = F32x8::from_slice(&ap[ps + x..]).to_array();
                    for k in 0..8 {
                        let lo = (x + k) * 4;
                        lin[lo] = ra[k];
                        lin[lo + 1] = ga[k];
                        lin[lo + 2] = ba[k];
                        lin[lo + 3] = aa[k];
                    }
                } else {
                    for k in 0..8 {
                        let lo = (x + k) * 3;
                        lin[lo] = ra[k];
                        lin[lo + 1] = ga[k];
                        lin[lo + 2] = ba[k];
                    }
                }
            }
            x += 8;
        }
        while x < w {
            let ri = r[ps + x];
            let gi = g[ps + x];
            let bi = b[ps + x];
            let o = x * 4;
            rgba_row[o] = sample_to_u8_scalar::<OETF>(ri);
            rgba_row[o + 1] = sample_to_u8_scalar::<OETF>(gi);
            rgba_row[o + 2] = sample_to_u8_scalar::<OETF>(bi);
            if let Some(ap) = a {
                let ai = ap[ps + x];
                rgba_row[o + 3] = sample_to_u8_scalar::<OETF>(ai);
                if let Some(lin) = lin_row.as_deref_mut() {
                    let lo = x * 4;
                    lin[lo] = ri;
                    lin[lo + 1] = gi;
                    lin[lo + 2] = bi;
                    lin[lo + 3] = ai;
                }
            } else if let Some(lin) = lin_row.as_deref_mut() {
                let lo = x * 3;
                lin[lo] = ri;
                lin[lo + 1] = gi;
                lin[lo + 2] = bi;
            }
            x += 1;
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if !emit_linear {
            rgba.par_chunks_mut(w * 4)
                .take(h)
                .enumerate()
                .for_each(|(y, row)| pack_rgb_row(y, row, None));
        } else {
            let lin_ch = ch;
            rgba.par_chunks_mut(w * 4)
                .take(h)
                .zip(linear.par_chunks_mut(w * lin_ch).take(h))
                .enumerate()
                .for_each(|(y, (row, lin))| pack_rgb_row(y, row, Some(lin)));
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..h {
            let lin = if emit_linear {
                Some(&mut linear[y * w * ch..(y + 1) * w * ch])
            } else {
                None
            };
            pack_rgb_row(y, &mut rgba[y * w * 4..(y + 1) * w * 4], lin);
        }
    }
    (rgba, linear, ch as u32)
}

#[inline]
fn store_u32x8_le(dst: &mut [u8], v: Simd<u32, 8>) {
    store_u32_le::<8, 32>(dst, v.to_array());
}

#[inline]
fn store_u32x16_le(dst: &mut [u8], v: Simd<u32, 16>) {
    store_u32_le::<16, 64>(dst, v.to_array());
}

#[inline]
fn store_u32_le<const N: usize, const B: usize>(dst: &mut [u8], a: [u32; N]) {
    if cfg!(target_endian = "little") {
        let bytes: [u8; B] = unsafe { std::mem::transmute_copy(&a) };
        dst[..B].copy_from_slice(&bytes);
    } else {
        for (k, p) in a.iter().enumerate() {
            dst[k * 4..k * 4 + 4].copy_from_slice(&p.to_le_bytes());
        }
    }
}

/// Convert interleaved linear float samples to packed RGBA8.
///
/// * 4 channels: identity layout (SIMD over the whole buffer).
/// * 3 channels: RGB → RGBAs with opaque alpha.
/// * 1 channel: gray → RGBAs with opaque alpha.
/// * other channel counts: first sample replicated to RGB, opaque alpha.
pub fn floats_to_rgba(width: u32, height: u32, channels: u32, floats: &[f32]) -> Vec<u8> {
    let npx = width as usize * height as usize;
    let ch = channels as usize;
    debug_assert_eq!(floats.len(), npx.saturating_mul(ch));

    match ch {
        4 => linear_to_u8_buf(floats),
        3 => {
            let encoded = linear_to_u8_buf(floats);
            pack_rgb_to_rgba(&encoded, npx)
        }
        1 => {
            let encoded = linear_to_u8_buf(floats);
            pack_gray_to_rgba(&encoded, npx)
        }
        _ => {
            let mut rgba = vec![255u8; npx * 4];
            for i in 0..npx {
                let s = linear_to_u8_scalar(floats[i * ch]);
                let o = i * 4;
                rgba[o] = s;
                rgba[o + 1] = s;
                rgba[o + 2] = s;
            }
            rgba
        }
    }
}

/// Convert interleaved 0..=1 samples (already in output transfer) to RGBA16.
pub fn floats_unit_to_rgba16(width: u32, height: u32, channels: u32, floats: &[f32]) -> Vec<u16> {
    let npx = width as usize * height as usize;
    let ch = channels as usize;
    debug_assert_eq!(floats.len(), npx.saturating_mul(ch));
    let encoded = unit_to_u16_buf(floats);
    match ch {
        4 => encoded,
        3 => {
            let mut rgba = vec![65535u16; npx * 4];
            for (i, px) in encoded.as_chunks::<3>().0.iter().enumerate() {
                let d = i * 4;
                rgba[d] = px[0];
                rgba[d + 1] = px[1];
                rgba[d + 2] = px[2];
            }
            rgba
        }
        1 => {
            let mut rgba = vec![65535u16; npx * 4];
            for (i, &v) in encoded.iter().enumerate() {
                let d = i * 4;
                rgba[d] = v;
                rgba[d + 1] = v;
                rgba[d + 2] = v;
            }
            rgba
        }
        _ => {
            let mut rgba = vec![65535u16; npx * 4];
            for i in 0..npx {
                let v = unit_to_u16_simd::<4>(F32x4::splat(floats[i * ch]))[0];
                let d = i * 4;
                rgba[d] = v;
                rgba[d + 1] = v;
                rgba[d + 2] = v;
            }
            rgba
        }
    }
}

fn unit_to_u16_simd<const N: usize>(v: Simd<f32, N>) -> [u16; N] {
    let scaled = (v * Simd::splat(65535.0)).round();
    scaled
        .simd_clamp(Simd::splat(0.0), Simd::splat(65535.0))
        .cast::<u16>()
        .to_array()
}

fn unit_to_u16_buf(src: &[f32]) -> Vec<u16> {
    let mut dst = vec![0u16; src.len()];
    let mut i = 0;
    while i + 16 <= src.len() {
        let a = unit_to_u16_simd(F32x16::from_slice(&src[i..]));
        dst[i..i + 16].copy_from_slice(&a);
        i += 16;
    }
    while i < src.len() {
        dst[i] = unit_to_u16_simd::<4>(F32x4::splat(src[i]))[0];
        i += 1;
    }
    dst
}

/// Convert interleaved linear float samples to packed RGBA16 (sRGB OETF).
pub fn floats_to_rgba16(width: u32, height: u32, channels: u32, floats: &[f32]) -> Vec<u16> {
    let npx = width as usize * height as usize;
    let ch = channels as usize;
    debug_assert_eq!(floats.len(), npx.saturating_mul(ch));
    let encoded = linear_to_u16_buf(floats);
    match ch {
        4 => encoded,
        3 => {
            let mut rgba = vec![65535u16; npx * 4];
            for (i, px) in encoded.as_chunks::<3>().0.iter().enumerate() {
                let d = i * 4;
                rgba[d] = px[0];
                rgba[d + 1] = px[1];
                rgba[d + 2] = px[2];
            }
            rgba
        }
        1 => {
            let mut rgba = vec![65535u16; npx * 4];
            for (i, &v) in encoded.iter().enumerate() {
                let d = i * 4;
                rgba[d] = v;
                rgba[d + 1] = v;
                rgba[d + 2] = v;
            }
            rgba
        }
        _ => {
            let mut rgba = vec![65535u16; npx * 4];
            for i in 0..npx {
                let v = linear_to_u16_scalar(floats[i * ch]);
                let d = i * 4;
                rgba[d] = v;
                rgba[d + 1] = v;
                rgba[d + 2] = v;
            }
            rgba
        }
    }
}

fn linear_to_srgb<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N> {
    let c = v.simd_clamp(Simd::splat(0.0), Simd::splat(1.0));
    let linear = c * Simd::splat(LINEAR_SLOPE);
    let gamma =
        Simd::splat(SRGB_A) * (c.log2() * Simd::splat(GAMMA_EXP)).exp2() - Simd::splat(SRGB_B);
    c.simd_le(Simd::splat(LINEAR_THRESHOLD))
        .select(linear, gamma)
}

fn linear_to_u8_simd<const N: usize>(v: Simd<f32, N>) -> Simd<u8, N> {
    let s = linear_to_srgb(v);
    let scaled = (s * Simd::splat(255.0)).round();
    let clamped = scaled.simd_clamp(Simd::splat(0.0), Simd::splat(255.0));
    clamped.cast()
}

fn unit_to_u8_simd<const N: usize>(v: Simd<f32, N>) -> Simd<u8, N> {
    let scaled = (v * Simd::splat(255.0)).round();
    scaled
        .simd_clamp(Simd::splat(0.0), Simd::splat(255.0))
        .cast()
}

#[inline]
fn sample_to_u8<const OETF: bool, const N: usize>(v: Simd<f32, N>) -> Simd<u8, N> {
    if OETF {
        linear_to_u8_simd(v)
    } else {
        unit_to_u8_simd(v)
    }
}

fn linear_to_u8_scalar(v: f32) -> u8 {
    linear_to_u8_simd(F32x4::splat(v)).to_array()[0]
}

#[inline]
fn sample_to_u8_scalar<const OETF: bool>(v: f32) -> u8 {
    sample_to_u8::<OETF, 4>(F32x4::splat(v)).to_array()[0]
}

/// Interleaved 0..=1 samples (already in output transfer) → packed RGBA8.
pub fn floats_unit_to_rgba(width: u32, height: u32, channels: u32, floats: &[f32]) -> Vec<u8> {
    let npx = width as usize * height as usize;
    let ch = channels as usize;
    debug_assert_eq!(floats.len(), npx.saturating_mul(ch));
    match ch {
        4 => unit_to_u8_buf(floats),
        3 => {
            let encoded = unit_to_u8_buf(floats);
            pack_rgb_to_rgba(&encoded, npx)
        }
        1 => {
            let encoded = unit_to_u8_buf(floats);
            pack_gray_to_rgba(&encoded, npx)
        }
        _ => {
            let mut rgba = vec![255u8; npx * 4];
            for i in 0..npx {
                let s = unit_to_u8_simd::<4>(F32x4::splat(floats[i * ch])).to_array()[0];
                let o = i * 4;
                rgba[o] = s;
                rgba[o + 1] = s;
                rgba[o + 2] = s;
            }
            rgba
        }
    }
}

fn unit_to_u8_buf(src: &[f32]) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    let (c16, rem) = src.as_chunks::<16>();
    let (d16, drem) = dst.as_chunks_mut::<16>();
    for (s, d) in c16.iter().zip(d16.iter_mut()) {
        unit_to_u8_simd(F32x16::from_array(*s)).copy_to_slice(d);
    }
    let (c8, rem) = rem.as_chunks::<8>();
    let (d8, drem) = drem.as_chunks_mut::<8>();
    for (s, d) in c8.iter().zip(d8.iter_mut()) {
        unit_to_u8_simd(F32x8::from_array(*s)).copy_to_slice(d);
    }
    let (c4, rem) = rem.as_chunks::<4>();
    let (d4, drem) = drem.as_chunks_mut::<4>();
    for (s, d) in c4.iter().zip(d4.iter_mut()) {
        unit_to_u8_simd(F32x4::from_array(*s)).copy_to_slice(d);
    }
    for (s, d) in rem.iter().zip(drem.iter_mut()) {
        *d = unit_to_u8_simd::<4>(F32x4::splat(*s)).to_array()[0];
    }
    dst
}

fn linear_to_u16_simd<const N: usize>(v: Simd<f32, N>) -> [u16; N] {
    let s = linear_to_srgb(v);
    let scaled = (s * Simd::splat(65535.0)).round();
    let clamped = scaled.simd_clamp(Simd::splat(0.0), Simd::splat(65535.0));
    clamped.cast::<u16>().to_array()
}

fn linear_to_u16_scalar(v: f32) -> u16 {
    linear_to_u16_simd(F32x4::splat(v))[0]
}

fn linear_to_u16_buf(src: &[f32]) -> Vec<u16> {
    let mut dst = vec![0u16; src.len()];
    let mut i = 0;
    while i + 16 <= src.len() {
        let a = linear_to_u16_simd(F32x16::from_slice(&src[i..]));
        dst[i..i + 16].copy_from_slice(&a);
        i += 16;
    }
    while i < src.len() {
        dst[i] = linear_to_u16_scalar(src[i]);
        i += 1;
    }
    dst
}

fn linear_to_u8_buf(src: &[f32]) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    let (c16, rem) = src.as_chunks::<16>();
    let (d16, drem) = dst.as_chunks_mut::<16>();
    for (s, d) in c16.iter().zip(d16.iter_mut()) {
        linear_to_u8_simd(F32x16::from_array(*s)).copy_to_slice(d);
    }
    let (c8, rem) = rem.as_chunks::<8>();
    let (d8, drem) = drem.as_chunks_mut::<8>();
    for (s, d) in c8.iter().zip(d8.iter_mut()) {
        linear_to_u8_simd(F32x8::from_array(*s)).copy_to_slice(d);
    }
    let (c4, rem) = rem.as_chunks::<4>();
    let (d4, drem) = drem.as_chunks_mut::<4>();
    for (s, d) in c4.iter().zip(d4.iter_mut()) {
        linear_to_u8_simd(F32x4::from_array(*s)).copy_to_slice(d);
    }
    for (s, d) in rem.iter().zip(drem.iter_mut()) {
        *d = linear_to_u8_scalar(*s);
    }
    dst
}

fn pack_rgb_to_rgba(rgb: &[u8], npx: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; npx * 4];
    let mut i = 0;
    // 4 RGB pixels (12 bytes) → 4 RGBA pixels (16 bytes).
    while i + 4 <= npx {
        let src = i * 3;
        let dst = i * 4;
        let mut tmp = [255u8; 16];
        tmp[..12].copy_from_slice(&rgb[src..src + 12]);
        let v = U8x16::from_array(tmp);
        let packed = simd_swizzle!(v, [0, 1, 2, 15, 3, 4, 5, 15, 6, 7, 8, 15, 9, 10, 11, 15]);
        packed.copy_to_slice(&mut rgba[dst..]);
        i += 4;
    }
    while i < npx {
        let src = i * 3;
        let dst = i * 4;
        rgba[dst] = rgb[src];
        rgba[dst + 1] = rgb[src + 1];
        rgba[dst + 2] = rgb[src + 2];
        i += 1;
    }
    rgba
}

fn pack_gray_to_rgba(gray: &[u8], npx: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; npx * 4];
    let mut i = 0;
    while i + 4 <= npx {
        let g = Simd::<u8, 4>::from_slice(&gray[i..]);
        let expanded: U8x16 = simd_swizzle!(g, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
        let is_alpha = Mask::<i8, 16>::from_array([
            false, false, false, true, false, false, false, true, false, false, false, true, false,
            false, false, true,
        ]);
        let packed = is_alpha.select(U8x16::splat(255), expanded);
        packed.copy_to_slice(&mut rgba[i * 4..]);
        i += 4;
    }
    while i < npx {
        let g = gray[i];
        let dst = i * 4;
        rgba[dst] = g;
        rgba[dst + 1] = g;
        rgba[dst + 2] = g;
        i += 1;
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_linear_to_u8(v: f32) -> u8 {
        let c = v.clamp(0.0, 1.0);
        let s = if c <= LINEAR_THRESHOLD {
            c * LINEAR_SLOPE
        } else {
            SRGB_A * c.powf(GAMMA_EXP) - SRGB_B
        };
        (s * 255.0).round().clamp(0.0, 255.0) as u8
    }

    #[test]
    fn simd_srgb_matches_scalar_u8() {
        let samples: Vec<f32> = (0..64)
            .map(|i| (i as f32 / 63.0) * 1.2 - 0.1)
            .chain([0.0, 0.0031308, 0.0031309, 1.0, -1.0, 2.0])
            .collect();
        let simd = linear_to_u8_buf(&samples);
        for (v, s) in samples.iter().zip(simd.iter()) {
            let sc = scalar_linear_to_u8(*v);
            let diff = (*s as i16 - sc as i16).abs();
            assert!(diff <= 1, "v={v}: simd={s} scalar={sc} (off by {diff})");
        }
    }

    #[test]
    fn encoded_unit_pack_is_scale() {
        let r = [0.0f32, 0.5, 1.0, 1.2];
        let g = [0.0f32, 0.25, 0.75, -0.1];
        let b = [1.0f32, 0.0, 0.2, 0.8];
        let (rgba, _, ch) = planar_encoded_to_rgba(4, 1, &r, &g, &b, None, 4, false);
        assert_eq!(ch, 3);
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[1], 0);
        assert_eq!(rgba[2], 255);
        assert_eq!(rgba[4], 128);
        assert_eq!(rgba[8], 255);
        assert_eq!(rgba[12], 255);
        assert_eq!(rgba[13], 0);
        assert_eq!(rgba[15], 255);
    }

    #[test]
    fn rgba_passthrough_layout() {
        let floats = [0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let rgba = floats_to_rgba(2, 1, 4, &floats);
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[3], 255);
        assert_eq!(rgba[4], 255);
        assert_eq!(rgba[7], 255);
    }

    #[test]
    fn rgb_gets_opaque_alpha() {
        let floats = [
            1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let rgba = floats_to_rgba(4, 1, 3, &floats);
        assert_eq!(rgba.len(), 16);
        assert_eq!(rgba[3], 255);
        assert_eq!(rgba[7], 255);
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[5], 255);
        assert_eq!(rgba[10], 255);
    }

    #[test]
    fn gray_replicates() {
        let floats = [0.0f32, 1.0];
        let rgba = floats_to_rgba(2, 1, 1, &floats);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[255, 255, 255, 255]);
    }
}
