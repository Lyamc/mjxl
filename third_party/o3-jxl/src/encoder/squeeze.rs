//! Forward Squeeze (Haar-like) matching libjxl / jxl-oxide.
//!
//! Default parameters (`num_sq = 0` in the bitstream) first squeeze chroma
//! to 4:2:0 when three same-size colour channels exist, then repeatedly
//! squeeze every original channel in-place until both axes are ≤ 8.

use std::simd::prelude::*;

type I32x4 = Simd<i32, 4>;

/// One Squeeze step as in `SqueezeParams`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SqueezeStep {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_c: u32,
    pub num_c: u32,
}

/// One modular plane after RCT / Squeeze, with shift metadata for grouping.
#[derive(Clone, Debug)]
pub struct EncChannel {
    pub width: u32,
    pub height: u32,
    pub hshift: i32,
    pub vshift: i32,
    pub samples: Vec<i32>,
}

impl EncChannel {
    pub fn from_plane(width: u32, height: u32, samples: Vec<i32>) -> Self {
        debug_assert_eq!(samples.len(), width as usize * height as usize);
        Self {
            width,
            height,
            hshift: 0,
            vshift: 0,
            samples,
        }
    }

    /// Palette meta-channel: `hshift = vshift = -1` (not grouped).
    pub fn meta(width: u32, height: u32, samples: Vec<i32>) -> Self {
        let mut ch = Self::from_plane(width, height, samples);
        ch.hshift = -1;
        ch.vshift = -1;
        ch
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// libjxl `kMaxFirstPreviewSize`.
const MAX_PREVIEW: u32 = 8;

/// Default squeeze parameter list (same as jxl-oxide `set_default_params`).
///
/// `nb_meta` is the number of leading meta channels (palette); they are not squeezed.
pub fn default_squeeze_params(channels: &[EncChannel], nb_meta: u32) -> Vec<SqueezeStep> {
    let first = nb_meta as usize;
    if first >= channels.len() {
        return Vec::new();
    }
    let mut sp = Vec::new();
    let mut w = channels[first].width;
    let mut h = channels[first].height;
    let n = (channels.len() - first) as u32;

    if n >= 3 && channels[first + 1].width == w && channels[first + 1].height == h {
        let chroma = SqueezeStep {
            begin_c: nb_meta + 1,
            num_c: 2,
            in_place: false,
            horizontal: false,
        };
        sp.push(SqueezeStep {
            horizontal: true,
            ..chroma
        });
        sp.push(chroma);
    }

    let all = SqueezeStep {
        begin_c: nb_meta,
        num_c: n,
        in_place: true,
        horizontal: false,
    };
    if h >= w && h > MAX_PREVIEW {
        sp.push(SqueezeStep {
            horizontal: false,
            ..all
        });
        h = h.div_ceil(2);
    }
    while w > MAX_PREVIEW || h > MAX_PREVIEW {
        if w > MAX_PREVIEW {
            sp.push(SqueezeStep {
                horizontal: true,
                ..all
            });
            w = w.div_ceil(2);
        }
        if h > MAX_PREVIEW {
            sp.push(SqueezeStep {
                horizontal: false,
                ..all
            });
            h = h.div_ceil(2);
        }
    }
    sp
}

pub fn apply_squeeze(channels: &mut Vec<EncChannel>, steps: &[SqueezeStep]) {
    for step in steps {
        apply_one(channels, *step);
    }
}

fn apply_one(channels: &mut Vec<EncChannel>, step: SqueezeStep) {
    let begin = step.begin_c as usize;
    let end = begin + step.num_c as usize;
    debug_assert!(end <= channels.len());

    let mut residuals = Vec::with_capacity(step.num_c as usize);
    for ch in &mut channels[begin..end] {
        residuals.push(if step.horizontal {
            fwd_h(ch)
        } else {
            fwd_v(ch)
        });
    }

    if step.in_place {
        residuals.extend(channels.drain(end..));
    }
    channels.extend(residuals);
}

/// Horizontal squeeze of `ch` in-place; returns the residual channel.
fn fwd_h(ch: &mut EncChannel) -> EncChannel {
    let w = ch.width as usize;
    let h = ch.height as usize;
    let avg_w = w.div_ceil(2);
    let res_w = w / 2;
    let mut avg = vec![0i32; avg_w * h];
    let mut res = vec![0i32; res_w * h];

    for y in 0..h {
        let src = &ch.samples[y * w..y * w + w];
        let dst_a = &mut avg[y * avg_w..y * avg_w + avg_w];
        if res_w == 0 {
            dst_a[0] = src[0];
            continue;
        }
        let dst_r = &mut res[y * res_w..y * res_w + res_w];
        squeeze_row_h(src, dst_a, dst_r, w);
    }

    let mut residual = ch.clone();
    residual.width = res_w as u32;
    residual.samples = res;
    if residual.hshift >= 0 {
        residual.hshift += 1;
    }

    ch.width = avg_w as u32;
    ch.samples = avg;
    if ch.hshift >= 0 {
        ch.hshift += 1;
    }
    residual
}

fn fwd_v(ch: &mut EncChannel) -> EncChannel {
    let w = ch.width as usize;
    let h = ch.height as usize;
    let avg_h = h.div_ceil(2);
    let res_h = h / 2;
    let mut avg = vec![0i32; w * avg_h];
    let mut res = vec![0i32; w * res_h];

    if res_h == 0 {
        avg[..w].copy_from_slice(&ch.samples[..w]);
    } else {
        for y in 0..res_h {
            let a = &ch.samples[y * 2 * w..y * 2 * w + w];
            let b = &ch.samples[(y * 2 + 1) * w..(y * 2 + 1) * w + w];
            let dst = &mut avg[y * w..y * w + w];
            let mut x = 0;
            while x + 4 <= w {
                average_i32x4(I32x4::from_slice(&a[x..]), I32x4::from_slice(&b[x..]))
                    .copy_to_slice(&mut dst[x..]);
                x += 4;
            }
            for k in x..w {
                dst[k] = average_i32(a[k], b[k]);
            }
        }
        if h % 2 == 1 {
            avg[(avg_h - 1) * w..avg_h * w].copy_from_slice(&ch.samples[(h - 1) * w..]);
        }

        for y in 0..res_h {
            let a_off = y * 2 * w;
            let b_off = a_off + w;
            let avg_off = y * w;
            for x in 0..w {
                let left = if y > 0 {
                    ch.samples[(y * 2 - 1) * w + x]
                } else {
                    avg[avg_off + x]
                };
                let avg_s = avg[avg_off + x];
                let next_avg = if y + 1 < res_h {
                    avg[(y + 1) * w + x]
                } else if h % 2 == 1 {
                    ch.samples[(h - 1) * w + x]
                } else {
                    avg_s
                };
                let t = tendency_i32(left, avg_s, next_avg);
                res[avg_off + x] = ch.samples[a_off + x]
                    .wrapping_sub(ch.samples[b_off + x])
                    .wrapping_sub(t);
            }
        }
    }

    let mut residual = ch.clone();
    residual.height = res_h as u32;
    residual.samples = res;
    if residual.vshift >= 0 {
        residual.vshift += 1;
    }

    ch.height = avg_h as u32;
    ch.samples = avg;
    if ch.vshift >= 0 {
        ch.vshift += 1;
    }
    residual
}

/// Squeeze a 1-D pair sequence (a row for H, a column for V).
fn squeeze_row_h(src: &[i32], dst_avg: &mut [i32], dst_res: &mut [i32], src_len: usize) {
    let res_w = dst_res.len();
    let avg_w = dst_avg.len();
    debug_assert_eq!(avg_w, src_len.div_ceil(2));
    debug_assert_eq!(res_w, src_len / 2);

    // Averages first so `next_avg` is available.
    let mut i = 0;
    while i + 4 <= res_w {
        let even = I32x4::from_array([src[i * 2], src[i * 2 + 2], src[i * 2 + 4], src[i * 2 + 6]]);
        let odd = I32x4::from_array([
            src[i * 2 + 1],
            src[i * 2 + 3],
            src[i * 2 + 5],
            src[i * 2 + 7],
        ]);
        let avg = average_i32x4(even, odd);
        avg.copy_to_slice(&mut dst_avg[i..i + 4]);
        i += 4;
    }
    for k in i..res_w {
        dst_avg[k] = average_i32(src[k * 2], src[k * 2 + 1]);
    }
    if src_len % 2 == 1 {
        dst_avg[avg_w - 1] = src[src_len - 1];
    }

    for x in 0..res_w {
        let a = src[x * 2];
        let b = src[x * 2 + 1];
        let avg = dst_avg[x];
        let next_avg = if x + 1 < res_w {
            dst_avg[x + 1]
        } else if src_len % 2 == 1 {
            src[x * 2 + 2]
        } else {
            avg
        };
        let left = if x > 0 { src[x * 2 - 1] } else { avg };
        let t = tendency_i32(left, avg, next_avg);
        dst_res[x] = a.wrapping_sub(b).wrapping_sub(t);
    }
}

#[inline]
pub fn average_i32(a: i32, b: i32) -> i32 {
    // libjxl AVERAGE: ((A + B + (A > B)) >> 1) on 32-bit pixels.
    (a.wrapping_add(b).wrapping_add(i32::from(a > b))) >> 1
}

#[inline]
fn average_i32x4(a: I32x4, b: I32x4) -> I32x4 {
    let one = a.simd_gt(b).select(I32x4::splat(1), I32x4::splat(0));
    (a + b + one) >> 1
}

/// jxl-oxide `tendency_i16` / libjxl `SmoothTendency` (32-bit).
pub fn tendency_i32(a: i32, b: i32, c: i32) -> i32 {
    if a >= b && b >= c {
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        let mut x = (4 * a - 3 * c - b - 6) / 12;
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    }
}

/// Inverse horizontal squeeze (for tests). `out_w = avg_w + res_w`.
#[cfg(test)]
fn inverse_h(avg: &[i32], res: &[i32], avg_w: usize, res_w: usize, h: usize) -> Vec<i32> {
    let out_w = avg_w + res_w;
    let mut out = vec![0i32; out_w * h];
    for y in 0..h {
        let avg_row = &avg[y * avg_w..y * avg_w + avg_w];
        let res_row = if res_w == 0 {
            &[][..]
        } else {
            &res[y * res_w..y * res_w + res_w]
        };
        let dst = &mut out[y * out_w..y * out_w + out_w];
        let mut avg_s = avg_row[0];
        let mut left = avg_s;
        for x in 0..res_w {
            let residu = res_row[x];
            let next_avg = avg_row.get(x + 1).copied().unwrap_or(avg_s);
            let diff = residu.wrapping_add(tendency_i32(left, avg_s, next_avg));
            let first = avg_s.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            dst[x * 2] = first;
            dst[x * 2 + 1] = second;
            avg_s = next_avg;
            left = second;
        }
        if out_w % 2 == 1 {
            dst[out_w - 1] = avg_row[avg_w - 1];
        }
    }
    out
}

#[cfg(test)]
fn inverse_v(avg: &[i32], res: &[i32], w: usize, avg_h: usize, res_h: usize) -> Vec<i32> {
    let out_h = avg_h + res_h;
    let mut out = vec![0i32; w * out_h];
    for x in 0..w {
        let mut col_a = vec![0i32; avg_h];
        let mut col_r = vec![0i32; res_h];
        for y in 0..avg_h {
            col_a[y] = avg[y * w + x];
        }
        for y in 0..res_h {
            col_r[y] = res[y * w + x];
        }
        let col = inverse_h(&col_a, &col_r, avg_h, res_h, 1);
        for y in 0..out_h {
            out[y * w + x] = col[y];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_matches_libjxl() {
        assert_eq!(average_i32(5, 3), 4);
        assert_eq!(average_i32(3, 5), 4);
        assert_eq!(average_i32(4, 3), 4);
        assert_eq!(average_i32(3, 4), 3);
    }

    #[test]
    fn h_squeeze_roundtrip_even() {
        let w = 8usize;
        let h = 4usize;
        let mut src = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = (x * 13 + y * 7) as i32;
            }
        }
        let mut ch = EncChannel::from_plane(w as u32, h as u32, src.clone());
        let res = fwd_h(&mut ch);
        let back = inverse_h(
            &ch.samples,
            &res.samples,
            ch.width as usize,
            res.width as usize,
            h,
        );
        assert_eq!(back, src);
    }

    #[test]
    fn h_squeeze_roundtrip_odd() {
        let w = 9usize;
        let h = 3usize;
        let mut src = vec![0i32; w * h];
        for (i, s) in src.iter_mut().enumerate() {
            *s = (i * 9 % 251) as i32;
        }
        let mut ch = EncChannel::from_plane(w as u32, h as u32, src.clone());
        let res = fwd_h(&mut ch);
        let back = inverse_h(
            &ch.samples,
            &res.samples,
            ch.width as usize,
            res.width as usize,
            h,
        );
        assert_eq!(back, src);
    }

    #[test]
    fn v_squeeze_roundtrip() {
        let w = 5usize;
        let h = 7usize;
        let mut src = vec![0i32; w * h];
        for (i, s) in src.iter_mut().enumerate() {
            *s = (i as i32) * 3 - 10;
        }
        let mut ch = EncChannel::from_plane(w as u32, h as u32, src.clone());
        let res = fwd_v(&mut ch);
        let back = inverse_v(
            &ch.samples,
            &res.samples,
            w,
            ch.height as usize,
            res.height as usize,
        );
        assert_eq!(back, src);
    }

    #[test]
    fn default_params_16x16_rgb() {
        let ch: Vec<EncChannel> = (0..3)
            .map(|_| EncChannel::from_plane(16, 16, vec![0; 256]))
            .collect();
        let sp = default_squeeze_params(&ch, 0);
        // chroma H, chroma V, then vertical (h >= w && h > 8), then horizontal.
        assert_eq!(sp.len(), 4);
        assert!(sp[0].horizontal && !sp[0].in_place && sp[0].begin_c == 1);
        assert!(!sp[1].horizontal && !sp[1].in_place);
        assert!(!sp[2].horizontal && sp[2].in_place && sp[2].num_c == 3);
        assert!(sp[3].horizontal && sp[3].in_place);
    }

    #[test]
    fn default_params_tiny_noop() {
        let ch = vec![EncChannel::from_plane(4, 4, vec![0; 16])];
        assert!(default_squeeze_params(&ch, 0).is_empty());
    }
}
