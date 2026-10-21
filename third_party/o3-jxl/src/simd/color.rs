//! SIMD color transforms used by JPEG XL: XYB ↔ linear sRGB and YCbCr ↔ RGB.

use std::simd::StdFloat;
use std::simd::prelude::*;
use std::sync::OnceLock;

type F32x8 = Simd<f32, 8>;

/// Default opsin inverse matrix (LMS-like → linear sRGB, D65).
#[allow(clippy::excessive_precision)]
pub const OPSIN_INV_MAT: [f32; 9] = [
    11.031_566_901_960_783,
    -9.866_943_921_568_629,
    -0.164_622_996_470_588_26,
    -3.254_147_380_392_157,
    4.418_770_392_156_863,
    -0.164_622_996_470_588_26,
    -3.658_851_286_274_509_7,
    2.712_923_047_058_823_5,
    1.945_928_239_215_686_3,
];

/// Default opsin bias (same value in L, M, S).
#[allow(clippy::excessive_precision)]
pub const OPSIN_BIAS: [f32; 3] = [-0.003_793_073_255_275_449_3; 3];

/// Default intensity target (nits). `itscale = 255 / intensity_target`.
pub const INTENSITY_TARGET: f32 = 255.0;

/// Opsin inverse parameters (bias, intensity, LMS→RGB matrix).
#[derive(Debug, Clone, Copy)]
pub struct OpsinParams {
    pub bias: [f32; 3],
    pub intensity_target: f32,
    pub inv_mat: [f32; 9],
}

impl Default for OpsinParams {
    fn default() -> Self {
        Self {
            bias: OPSIN_BIAS,
            intensity_target: INTENSITY_TARGET,
            inv_mat: OPSIN_INV_MAT,
        }
    }
}

const Y_OFFSET: f32 = 128.0 / 255.0;
const CR_R: f32 = 1.402;
const CB_B: f32 = 1.772;
#[allow(clippy::excessive_precision)]
const CR_G: f32 = -0.299 * 1.402 / 0.587;
#[allow(clippy::excessive_precision)]
const CB_G: f32 = -0.114 * 1.772 / 0.587;
/// sRGB EOTF (encoded 0..=1 → linear).
#[inline]
pub fn srgb_eotf(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// 8-bit sRGB → linear LUT (index = encoded byte).
pub fn srgb_eotf_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = srgb_eotf(i as f32 / 255.0);
        }
        t
    })
}

fn default_fwd_mat() -> [f32; 9] {
    static M: OnceLock<[f32; 9]> = OnceLock::new();
    *M.get_or_init(|| invert3(OPSIN_INV_MAT))
}

/// Packed sRGB8 RGB → planar XYB, edge-padded to `pad_w` × `pad_h`.
pub fn srgb8_rgb_to_xyb_padded(
    width: u32,
    height: u32,
    pixels: &[u8],
    pad_w: usize,
    pad_h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w = width as usize;
    let h = height as usize;
    debug_assert_eq!(pixels.len(), w * h * 3);
    debug_assert!(pad_w >= w && pad_h >= h);

    let lut = srgb_eotf_lut();
    let fwd = default_fwd_mat();
    let opsin_bias = OPSIN_BIAS;
    let cbrt_ob = opsin_bias.map(cbrt_scalar);
    let nout = pad_w * pad_h;
    let mut x_out = vec![0f32; nout];
    let mut y_out = vec![0f32; nout];
    let mut b_out = vec![0f32; nout];

    let fill = |row: usize, x_row: &mut [f32], y_row: &mut [f32], b_row: &mut [f32]| {
        fill_xyb_row(
            pixels, lut, fwd, opsin_bias, cbrt_ob, w, row, x_row, y_row, b_row,
        );
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let (x_body, _) = x_out.split_at_mut(h * pad_w);
        let (y_body, _) = y_out.split_at_mut(h * pad_w);
        let (b_body, _) = b_out.split_at_mut(h * pad_w);
        x_body
            .par_chunks_mut(pad_w)
            .zip(y_body.par_chunks_mut(pad_w))
            .zip(b_body.par_chunks_mut(pad_w))
            .enumerate()
            .for_each(|(row, ((xr, yr), br))| fill(row, xr, yr, br));
    }
    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..h {
            let d = row * pad_w;
            fill(
                row,
                &mut x_out[d..d + pad_w],
                &mut y_out[d..d + pad_w],
                &mut b_out[d..d + pad_w],
            );
        }
    }
    if pad_h > h && h > 0 {
        let last_row = (h - 1) * pad_w;
        let x_last = x_out[last_row..last_row + pad_w].to_vec();
        let y_last = y_out[last_row..last_row + pad_w].to_vec();
        let b_last = b_out[last_row..last_row + pad_w].to_vec();
        for row in h..pad_h {
            let d = row * pad_w;
            x_out[d..d + pad_w].copy_from_slice(&x_last);
            y_out[d..d + pad_w].copy_from_slice(&y_last);
            b_out[d..d + pad_w].copy_from_slice(&b_last);
        }
    }
    (x_out, y_out, b_out)
}

#[allow(clippy::too_many_arguments)]
fn fill_xyb_row(
    pixels: &[u8],
    lut: &[f32; 256],
    fwd: [f32; 9],
    opsin_bias: [f32; 3],
    cbrt_ob: [f32; 3],
    w: usize,
    row: usize,
    x_row: &mut [f32],
    y_row: &mut [f32],
    b_row: &mut [f32],
) {
    let src_row = row * w * 3;
    let mut x = 0;
    while x + 8 <= w {
        let mut rv = [0f32; 8];
        let mut gv = [0f32; 8];
        let mut bv = [0f32; 8];
        let p = src_row + x * 3;
        for k in 0..8 {
            let o = p + k * 3;
            rv[k] = lut[pixels[o] as usize];
            gv[k] = lut[pixels[o + 1] as usize];
            bv[k] = lut[pixels[o + 2] as usize];
        }
        let (xx, yy, bb) = linear_to_xyb_simd(
            F32x8::from_array(rv),
            F32x8::from_array(gv),
            F32x8::from_array(bv),
            fwd,
            opsin_bias,
            cbrt_ob,
        );
        xx.copy_to_slice(&mut x_row[x..]);
        yy.copy_to_slice(&mut y_row[x..]);
        bb.copy_to_slice(&mut b_row[x..]);
        x += 8;
    }
    while x < w {
        let o = src_row + x * 3;
        let (xx, yy, bb) = linear_to_xyb_one(
            lut[pixels[o] as usize],
            lut[pixels[o + 1] as usize],
            lut[pixels[o + 2] as usize],
            fwd,
            opsin_bias,
            cbrt_ob,
        );
        x_row[x] = xx;
        y_row[x] = yy;
        b_row[x] = bb;
        x += 1;
    }
    if x_row.len() > w && w > 0 {
        let last_x = x_row[w - 1];
        let last_y = y_row[w - 1];
        let last_b = b_row[w - 1];
        for xx in w..x_row.len() {
            x_row[xx] = last_x;
            y_row[xx] = last_y;
            b_row[xx] = last_b;
        }
    }
}

#[inline]
fn linear_to_xyb_simd<const N: usize>(
    r: Simd<f32, N>,
    g: Simd<f32, N>,
    b: Simd<f32, N>,
    fwd: [f32; 9],
    opsin_bias: [f32; 3],
    cbrt_ob: [f32; 3],
) -> (Simd<f32, N>, Simd<f32, N>, Simd<f32, N>) {
    type S<const M: usize> = Simd<f32, M>;
    let (mut l, mut m, mut s) = mul3(fwd, r, g, b);
    l = cbrt_signed(l - S::<N>::splat(opsin_bias[0])) + S::<N>::splat(cbrt_ob[0]);
    m = cbrt_signed(m - S::<N>::splat(opsin_bias[1])) + S::<N>::splat(cbrt_ob[1]);
    s = cbrt_signed(s - S::<N>::splat(opsin_bias[2])) + S::<N>::splat(cbrt_ob[2]);
    (
        (l - m) * S::<N>::splat(0.5),
        (l + m) * S::<N>::splat(0.5),
        s,
    )
}

#[inline]
fn linear_to_xyb_one(
    r: f32,
    g: f32,
    b: f32,
    fwd: [f32; 9],
    opsin_bias: [f32; 3],
    cbrt_ob: [f32; 3],
) -> (f32, f32, f32) {
    let l0 = fwd[0] * r + fwd[1] * g + fwd[2] * b;
    let m0 = fwd[3] * r + fwd[4] * g + fwd[5] * b;
    let s0 = fwd[6] * r + fwd[7] * g + fwd[8] * b;
    let l = cbrt_scalar(l0 - opsin_bias[0]) + cbrt_ob[0];
    let m = cbrt_scalar(m0 - opsin_bias[1]) + cbrt_ob[1];
    let s = cbrt_scalar(s0 - opsin_bias[2]) + cbrt_ob[2];
    ((l - m) * 0.5, (l + m) * 0.5, s)
}

const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;

/// Planar XYB → planar linear sRGB (D65 primaries), using default opsin params.
pub fn xyb_to_linear_rgb(xyb: [&[f32]; 3], rgb: [&mut [f32]; 3]) {
    xyb_to_linear_rgb_params(xyb, rgb, &OpsinParams::default());
}

/// Planar linear sRGB → planar XYB using default opsin params.
pub fn linear_rgb_to_xyb(rgb: [&[f32]; 3], xyb: [&mut [f32]; 3]) {
    linear_rgb_to_xyb_params(rgb, xyb, &OpsinParams::default());
}

pub fn xyb_to_linear_rgb_params(xyb: [&[f32]; 3], rgb: [&mut [f32]; 3], params: &OpsinParams) {
    let [x, y, b] = xyb;
    let [r_out, g_out, b_out] = rgb;
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(r_out.len(), n);
    assert_eq!(g_out.len(), n);
    assert_eq!(b_out.len(), n);

    let itscale = 255.0 / params.intensity_target;
    let opsin_bias = params.bias;
    let inv_mat = params.inv_mat;
    let cbrt_ob = opsin_bias.map(cbrt_scalar);
    let mut i = 0;
    while i + 8 <= n {
        let xv = F32x8::from_slice(&x[i..]);
        let yv = F32x8::from_slice(&y[i..]);
        let bv = F32x8::from_slice(&b[i..]);

        let mut g_l = yv + xv - F32x8::splat(cbrt_ob[0]);
        let mut g_m = yv - xv - F32x8::splat(cbrt_ob[1]);
        let mut g_s = bv - F32x8::splat(cbrt_ob[2]);

        g_l = cube(g_l) + F32x8::splat(opsin_bias[0]);
        g_m = cube(g_m) + F32x8::splat(opsin_bias[1]);
        g_s = cube(g_s) + F32x8::splat(opsin_bias[2]);
        let scale = F32x8::splat(itscale);
        g_l *= scale;
        g_m *= scale;
        g_s *= scale;

        let (r, g, bl) = mul3(inv_mat, g_l, g_m, g_s);
        r.copy_to_slice(&mut r_out[i..]);
        g.copy_to_slice(&mut g_out[i..]);
        bl.copy_to_slice(&mut b_out[i..]);
        i += 8;
    }
    while i < n {
        let g_l = y[i] + x[i] - cbrt_ob[0];
        let g_m = y[i] - x[i] - cbrt_ob[1];
        let g_s = b[i] - cbrt_ob[2];
        let l = (cube_s(g_l) + opsin_bias[0]) * itscale;
        let m = (cube_s(g_m) + opsin_bias[1]) * itscale;
        let s = (cube_s(g_s) + opsin_bias[2]) * itscale;
        r_out[i] = inv_mat[0] * l + inv_mat[1] * m + inv_mat[2] * s;
        g_out[i] = inv_mat[3] * l + inv_mat[4] * m + inv_mat[5] * s;
        b_out[i] = inv_mat[6] * l + inv_mat[7] * m + inv_mat[8] * s;
        i += 1;
    }
}

pub fn linear_rgb_to_xyb_params(rgb: [&[f32]; 3], xyb: [&mut [f32]; 3], params: &OpsinParams) {
    let [r, g, b] = rgb;
    let [x_out, y_out, b_out] = xyb;
    let n = r.len();
    assert_eq!(g.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(x_out.len(), n);
    assert_eq!(y_out.len(), n);
    assert_eq!(b_out.len(), n);

    let opsin_bias = params.bias;
    let fwd = if params.inv_mat == OPSIN_INV_MAT {
        default_fwd_mat()
    } else {
        invert3(params.inv_mat)
    };
    let itscale = 255.0 / params.intensity_target;
    let inv_scale = 1.0 / itscale;
    let cbrt_ob = opsin_bias.map(cbrt_scalar);

    let mut i = 0;
    while i + 8 <= n {
        let rv = F32x8::from_slice(&r[i..]);
        let gv = F32x8::from_slice(&g[i..]);
        let bv = F32x8::from_slice(&b[i..]);
        let (mut l, mut m, mut s) = mul3(fwd, rv, gv, bv);
        let invs = F32x8::splat(inv_scale);
        l = cbrt_signed(l * invs - F32x8::splat(opsin_bias[0])) + F32x8::splat(cbrt_ob[0]);
        m = cbrt_signed(m * invs - F32x8::splat(opsin_bias[1])) + F32x8::splat(cbrt_ob[1]);
        s = cbrt_signed(s * invs - F32x8::splat(opsin_bias[2])) + F32x8::splat(cbrt_ob[2]);
        let x = (l - m) * F32x8::splat(0.5);
        let y = (l + m) * F32x8::splat(0.5);
        x.copy_to_slice(&mut x_out[i..]);
        y.copy_to_slice(&mut y_out[i..]);
        s.copy_to_slice(&mut b_out[i..]);
        i += 8;
    }
    while i < n {
        let l0 = fwd[0] * r[i] + fwd[1] * g[i] + fwd[2] * b[i];
        let m0 = fwd[3] * r[i] + fwd[4] * g[i] + fwd[5] * b[i];
        let s0 = fwd[6] * r[i] + fwd[7] * g[i] + fwd[8] * b[i];
        let l = cbrt_scalar(l0 * inv_scale - opsin_bias[0]) + cbrt_ob[0];
        let m = cbrt_scalar(m0 * inv_scale - opsin_bias[1]) + cbrt_ob[1];
        let s = cbrt_scalar(s0 * inv_scale - opsin_bias[2]) + cbrt_ob[2];
        x_out[i] = (l - m) * 0.5;
        y_out[i] = (l + m) * 0.5;
        b_out[i] = s;
        i += 1;
    }
}

/// JPEG XL YCbCr → RGB. Channels are planar and in **Cb, Y, Cr** order (libjxl).
///
/// `y` is stored with a 128/255 offset already subtracted.
pub fn ycbcr_to_rgb(cb: &mut [f32], y: &mut [f32], cr: &mut [f32]) {
    let n = y.len();
    assert_eq!(cb.len(), n);
    assert_eq!(cr.len(), n);
    let mut i = 0;
    let yoff = F32x8::splat(Y_OFFSET);
    let cr_r = F32x8::splat(CR_R);
    let cr_g = F32x8::splat(CR_G);
    let cb_g = F32x8::splat(CB_G);
    let cb_b = F32x8::splat(CB_B);
    while i + 8 <= n {
        let cbv = F32x8::from_slice(&cb[i..]);
        let yv = F32x8::from_slice(&y[i..]) + yoff;
        let crv = F32x8::from_slice(&cr[i..]);
        let r = crv.mul_add(cr_r, yv);
        let g = cbv.mul_add(cb_g, crv.mul_add(cr_g, yv));
        let b = cbv.mul_add(cb_b, yv);
        r.copy_to_slice(&mut cb[i..]);
        g.copy_to_slice(&mut y[i..]);
        b.copy_to_slice(&mut cr[i..]);
        i += 8;
    }
    while i < n {
        let cbi = cb[i];
        let yi = y[i] + Y_OFFSET;
        let cri = cr[i];
        cb[i] = cri.mul_add(CR_R, yi);
        y[i] = cbi.mul_add(CB_G, cri.mul_add(CR_G, yi));
        cr[i] = cbi.mul_add(CB_B, yi);
        i += 1;
    }
}

/// RGB → JPEG XL YCbCr (planar, **Cb, Y, Cr** order, Y offset subtracted).
pub fn rgb_to_ycbcr(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let n = r.len();
    assert_eq!(g.len(), n);
    assert_eq!(b.len(), n);
    let mut i = 0;
    let kr = F32x8::splat(KR);
    let kg = F32x8::splat(KG);
    let kb = F32x8::splat(KB);
    let yoff = F32x8::splat(Y_OFFSET);
    let inv_cb = F32x8::splat(1.0 / CB_B);
    let inv_cr = F32x8::splat(1.0 / CR_R);
    while i + 8 <= n {
        let rv = F32x8::from_slice(&r[i..]);
        let gv = F32x8::from_slice(&g[i..]);
        let bv = F32x8::from_slice(&b[i..]);
        let yv = rv.mul_add(kr, gv.mul_add(kg, bv * kb));
        let cb = (bv - yv) * inv_cb;
        let cr = (rv - yv) * inv_cr;
        cb.copy_to_slice(&mut r[i..]);
        (yv - yoff).copy_to_slice(&mut g[i..]);
        cr.copy_to_slice(&mut b[i..]);
        i += 8;
    }
    while i < n {
        let yv = KR * r[i] + KG * g[i] + KB * b[i];
        let cb = (b[i] - yv) / CB_B;
        let cr = (r[i] - yv) / CR_R;
        r[i] = cb;
        g[i] = yv - Y_OFFSET;
        b[i] = cr;
        i += 1;
    }
}

fn cube<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N> {
    v * v * v
}

fn cube_s(v: f32) -> f32 {
    v * v * v
}

fn cbrt_signed<const N: usize>(v: Simd<f32, N>) -> Simd<f32, N> {
    // Integer bit-hack + two Newton steps (faster than log2/exp2).
    let a = v.abs();
    let magic = Simd::<u32, N>::splat(0x2a51_1cd0);
    let i = a.to_bits() / Simd::splat(3) + magic;
    let mut g = Simd::<f32, N>::from_bits(i);
    let third = Simd::<f32, N>::splat(1.0 / 3.0);
    for _ in 0..2 {
        let g2 = g * g;
        g = (g + g + a / g2) * third;
    }
    let zero = a.simd_eq(Simd::splat(0.0));
    g = zero.select(Simd::splat(0.0), g);
    v.is_sign_negative().select(-g, g)
}

fn cbrt_scalar(v: f32) -> f32 {
    v.abs().cbrt().copysign(v)
}

fn mul3<const N: usize>(
    m: [f32; 9],
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> (Simd<f32, N>, Simd<f32, N>, Simd<f32, N>) {
    type S<const M: usize> = Simd<f32, M>;
    (
        S::<N>::splat(m[0]).mul_add(x, S::<N>::splat(m[1]).mul_add(y, S::<N>::splat(m[2]) * z)),
        S::<N>::splat(m[3]).mul_add(x, S::<N>::splat(m[4]).mul_add(y, S::<N>::splat(m[5]) * z)),
        S::<N>::splat(m[6]).mul_add(x, S::<N>::splat(m[7]).mul_add(y, S::<N>::splat(m[8]) * z)),
    )
}

fn invert3(m: [f32; 9]) -> [f32; 9] {
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[3], m[4], m[5]);
    let (g, h, i) = (m[6], m[7], m[8]);
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv = det.recip();
    [
        (e * i - f * h) * inv,
        (c * h - b * i) * inv,
        (b * f - c * e) * inv,
        (f * g - d * i) * inv,
        (a * i - c * g) * inv,
        (c * d - a * f) * inv,
        (d * h - e * g) * inv,
        (b * g - a * h) * inv,
        (a * e - b * d) * inv,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_cbrt_close_to_libm() {
        let samples: [f32; 8] = [1e-6, 0.001, 0.05, 0.2, 0.5, 1.0, 2.5, 8.0];
        let got = cbrt_signed(F32x8::from_array(samples)).to_array();
        for (x, y) in samples.iter().zip(got) {
            let w = x.cbrt();
            let e = (y - w).abs() / w.max(1e-12);
            assert!(e < 2e-5, "cbrt({x}) simd={y} libm={w} rel={e}");
        }
    }

    #[test]
    fn eotf_lut_matches_scalar() {
        let lut = srgb_eotf_lut();
        for (i, &a) in lut.iter().enumerate() {
            let b = srgb_eotf(i as f32 / 255.0);
            assert!((a - b).abs() < 1e-7, "i={i} lut={a} fn={b}");
        }
    }

    #[test]
    fn xyb_roundtrip_neutral_gray() {
        let n = 17; // hits SIMD + scalar tail
        let r = vec![0.2f32; n];
        let g = vec![0.2f32; n];
        let b = vec![0.2f32; n];
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut z = vec![0.0; n];
        linear_rgb_to_xyb([&r, &g, &b], [&mut x, &mut y, &mut z]);
        let mut ro = vec![0.0; n];
        let mut go = vec![0.0; n];
        let mut bo = vec![0.0; n];
        xyb_to_linear_rgb([&x, &y, &z], [&mut ro, &mut go, &mut bo]);
        for i in 0..n {
            assert!((ro[i] - 0.2).abs() < 2e-3, "r[{i}]={}", ro[i]);
            assert!((go[i] - 0.2).abs() < 2e-3, "g[{i}]={}", go[i]);
            assert!((bo[i] - 0.2).abs() < 2e-3, "b[{i}]={}", bo[i]);
        }
    }

    #[test]
    fn ycbcr_neutral_is_mid_gray() {
        let n = 13;
        let mut cb = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut cr = vec![0.0f32; n];
        ycbcr_to_rgb(&mut cb, &mut y, &mut cr);
        let mid = Y_OFFSET;
        for i in 0..n {
            assert!((cb[i] - mid).abs() < 1e-5);
            assert!((y[i] - mid).abs() < 1e-5);
            assert!((cr[i] - mid).abs() < 1e-5);
        }
    }

    #[test]
    fn ycbcr_roundtrip() {
        let n = 19;
        let mut r: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let mut g: Vec<f32> = (0..n).map(|i| (n - 1 - i) as f32 / n as f32).collect();
        let mut b = vec![0.4f32; n];
        let orig = (r.clone(), g.clone(), b.clone());
        rgb_to_ycbcr(&mut r, &mut g, &mut b);
        ycbcr_to_rgb(&mut r, &mut g, &mut b);
        for i in 0..n {
            assert!((r[i] - orig.0[i]).abs() < 2e-5);
            assert!((g[i] - orig.1[i]).abs() < 2e-5);
            assert!((b[i] - orig.2[i]).abs() < 2e-5);
        }
    }
}
