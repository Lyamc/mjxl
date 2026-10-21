//! Modular JPEG XL encoder (`std::simd` residuals + YCoCg-R, Squeeze, palette).

mod ans;
mod bits;
mod entropy;
mod header;
mod palette;
mod predict;
mod squeeze;
mod vardct;

use crate::error::{Error, Result};

use ans::{
    AnsHist, AnsToken, ans_rle_distance, collect_ans_tokens_ctx_len, collect_ans_tokens_len,
    write_ans_histograms_len, write_ans_tokens_ex,
};
use bits::BitWriter;
use entropy::{
    LZ77_MIN_LENGTH, MaLeaf, MaShape, ResTree, for_each_token_ctx_len, for_each_token_len,
    ma_bfs_leaf_ctxs_hf, ma_stream_ctxs, ma_stream_leaf_count, write_ma_tree,
};
use header::{
    AnimHeader, ImageParams, ModularTransform, write_frame_header, write_image_header,
    write_local_modular_prefix, write_modular_header, write_toc,
};
use palette::try_palette;
use predict::{
    HfSplit, gradient_residuals, planarize_and_transform, split_res_tree, zero_residuals,
};
use squeeze::{EncChannel, apply_squeeze, default_squeeze_params};

use crate::simd::srgb8_rgb_to_xyb_padded;

/// Pixel layout of the 8-bit input buffer (packed, row-major).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    Gray,
    GrayA,
    Rgb,
    Rgba,
}

impl PixelFormat {
    pub fn channels(self) -> usize {
        match self {
            PixelFormat::Gray => 1,
            PixelFormat::GrayA => 2,
            PixelFormat::Rgb => 3,
            PixelFormat::Rgba => 4,
        }
    }
}

/// JPEG XL encoder: **modular** (lossless and lossy residuals) and **VarDCT**.
///
/// Both modes stay. Distance 0 is lossless modular. Lossy RGB defaults to 8×8
/// VarDCT in XYB (`modular(Some(false))`); `modular(Some(true))` keeps
/// YCoCg-R + Squeeze at a positive distance (Gradient on LF, Zero on HF,
/// libjxl per-band quant). Modular is the path we expect to grow. Produces a
/// codestream our [`crate::Decoder`] (and libjxl / djxl) can read. Heavy inner
/// loops use [`std::simd`].
///
/// Flags mirror cjxl where we implement the same mode: `--distance` / `--quality`,
/// `--effort`, `--progressive`, `--responsive` (Squeeze), palette, `-m`.
/// 16-bit integer samples use [`Encoder::encode_u16`]. Animation uses
/// [`Encoder::encode_animation`]. JPEG lossless transcoding (`-j`) is not
/// implemented.
#[derive(Clone, Debug)]
pub struct Encoder {
    distance: f32,
    effort: u8,
    progressive: bool,
    /// `None` = auto (lossy modular or effort ≥ 8). `Some(true)` = force squeeze.
    responsive: Option<bool>,
    /// Max palette colours (`0` = disable). Default 256, like cjxl.
    palette_colors: u32,
    /// `None` = auto (VarDCT when lossy RGB). `Some(true)` forces modular.
    force_modular: Option<bool>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            distance: 0.0,
            effort: 7,
            progressive: false,
            responsive: None,
            palette_colors: 256,
            force_modular: None,
        }
    }

    /// `Some(true)` = modular, `Some(false)` = VarDCT, `None` = auto.
    pub fn modular(mut self, force: Option<bool>) -> Self {
        self.force_modular = force;
        self
    }

    /// Butteraugli-style distance. `0` is mathematically lossless.
    pub fn distance(mut self, d: f32) -> Self {
        self.distance = d.clamp(0.0, 25.0);
        self
    }

    /// cjxl-compatible quality 0..=100 (`100` = lossless). Mutually exclusive
    /// with [`distance`](Self::distance) at the CLI; last setter wins here.
    pub fn quality(self, q: f32) -> Self {
        self.distance(quality_to_distance(q))
    }

    /// Encoder effort 1..=10 (cjxl range). Higher = more groups / LZ77.
    pub fn effort(mut self, e: u8) -> Self {
        self.effort = e.clamp(1, 10);
        self
    }

    /// Smaller groups so a truncated prefix still paints more of the canvas.
    pub fn progressive(mut self, enable: bool) -> Self {
        self.progressive = enable;
        self
    }

    /// Squeeze transform (`cjxl -R` / `--responsive`). `None` = auto.
    pub fn responsive(mut self, enable: Option<bool>) -> Self {
        self.responsive = enable;
        self
    }

    /// Maximum palette size. `0` disables palette. Default `256`.
    pub fn palette_colors(mut self, n: u32) -> Self {
        self.palette_colors = n;
        self
    }

    /// cjxl lightning–cheetah (effort ≤ 4) skip palette search.
    /// Lossy modular also skips it (libjxl does; palette + residual quant fights).
    fn effective_palette_colors(&self) -> u32 {
        if self.effort <= 4 || self.distance > 0.0 {
            0
        } else {
            self.palette_colors
        }
    }

    fn use_vardct(&self, format: PixelFormat) -> bool {
        if format != PixelFormat::Rgb || self.distance <= 0.0 {
            return false;
        }
        match self.force_modular {
            Some(true) => false,
            Some(false) | None => true,
        }
    }

    fn use_squeeze(&self) -> bool {
        match self.responsive {
            Some(v) => v,
            None => self.distance > 0.0 || self.effort >= 8,
        }
    }

    fn tuned(&self) -> EncodeTune {
        let mut shift = match self.effort {
            1..=4 => 3,
            5..=7 => 2,
            _ => 1,
        };
        if self.progressive {
            shift = shift.min(1);
        }
        let (mul_log, mul_bits, quant) = multiplier_parts(distance_to_quant(self.distance));
        EncodeTune {
            group_size_shift: shift,
            // Lossy pass groups + LZ77 can emit a repeat as the first symbol
            // of a context (UnexpectedLz77Repeat). Keep RLE for lossless.
            lz77: self.effort >= 3 && self.distance <= 0.0,
            lz77_min_len: LZ77_MIN_LENGTH,
            quant,
            mul_log,
            mul_bits,
        }
    }

    pub fn encode_gray8(&self, width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>> {
        self.encode(width, height, PixelFormat::Gray, pixels)
    }

    pub fn encode_rgb8(&self, width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>> {
        self.encode(width, height, PixelFormat::Rgb, pixels)
    }

    pub fn encode_rgba8(&self, width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>> {
        self.encode(width, height, PixelFormat::Rgba, pixels)
    }

    pub fn encode(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u8],
    ) -> Result<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(Error::Encode("width and height must be > 0".into()));
        }
        if width > (1 << 30) || height > (1 << 30) {
            return Err(Error::Encode("image dimension exceeds 2^30".into()));
        }
        let ch = format.channels();
        let expected = width as usize * height as usize * ch;
        if pixels.len() != expected {
            return Err(Error::Encode(format!(
                "pixel buffer length {} != {width}×{height}×{ch} = {expected}",
                pixels.len()
            )));
        }

        if self.use_vardct(format) {
            return vardct::encode_vardct_rgb(width, height, pixels, self.distance, self.effort);
        }

        let pal = try_palette(
            width as usize,
            height as usize,
            ch,
            pixels,
            self.effective_palette_colors(),
        );
        let use_xyb = self.distance > 0.0 && format == PixelFormat::Rgb && pal.is_none();
        let planes = if pal.is_some() {
            Vec::new()
        } else if use_xyb {
            xyb_modular_planes(width, height, pixels)
        } else {
            planarize_and_transform(width as usize, height as usize, ch, pixels)
        };
        self.encode_inner_xyb(width, height, format, planes, pal, 8, use_xyb)
    }

    /// 16-bit packed samples (native endian), same channel layouts as [`encode`].
    pub fn encode_u16(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u16],
    ) -> Result<Vec<u8>> {
        let ch = format.channels();
        let expected = width as usize * height as usize * ch;
        if pixels.len() != expected {
            return Err(Error::Encode(format!(
                "pixel buffer length {} != {width}×{height}×{ch} = {expected}",
                pixels.len()
            )));
        }
        if width == 0 || height == 0 {
            return Err(Error::Encode("width and height must be > 0".into()));
        }
        let planes =
            predict::planarize_and_transform_u16(width as usize, height as usize, ch, pixels);
        self.encode_inner(width, height, format, planes, None, 16)
    }

    fn encode_inner(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
        planes: Vec<Vec<i32>>,
        pal: Option<palette::PalettePlanes>,
        bit_depth: u32,
    ) -> Result<Vec<u8>> {
        self.encode_inner_xyb(width, height, format, planes, pal, bit_depth, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_inner_xyb(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
        planes: Vec<Vec<i32>>,
        pal: Option<palette::PalettePlanes>,
        bit_depth: u32,
        xyb_encoded: bool,
    ) -> Result<Vec<u8>> {
        let mut tune = self.tuned();
        if width > (1 << 30) || height > (1 << 30) {
            return Err(Error::Encode("image dimension exceeds 2^30".into()));
        }
        // libjxl: modular_group_size_shift == -1 → 256, or 512 if both axes ≤ 400.
        if width <= 400 && height <= 400 {
            tune.group_size_shift = 2;
        } else {
            tune.group_size_shift = 1;
        }
        if self.progressive {
            tune.group_size_shift = tune.group_size_shift.min(1);
        }
        let ch = format.channels();
        let params = ImageParams {
            width,
            height,
            channels: ch as u32,
            group_size_shift: tune.group_size_shift,
            bit_depth,
            animation: None,
            duration: 0,
            is_last: true,
            xyb_encoded,
            vardct: false,
            gab: false,
            epf_iters: 0,
        };
        self.encode_with_params(params, format, planes, pal, &tune)
    }

    /// Encode an animation. `duration` is in ticks (`tps_num / tps_den` ticks per second).
    /// `num_loops == 0` means forever. Each frame is packed 8-bit samples in `format`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_animation(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
        tps_num: u32,
        tps_den: u32,
        num_loops: u32,
        frames: &[(u32, &[u8])],
    ) -> Result<Vec<u8>> {
        if frames.is_empty() {
            return Err(Error::Encode("animation needs at least one frame".into()));
        }
        if width == 0 || height == 0 {
            return Err(Error::Encode("width and height must be > 0".into()));
        }
        let ch = format.channels();
        let expected = width as usize * height as usize * ch;
        let tune = self.tuned();
        let n = frames.len();
        let mut out = BitWriter::with_capacity(64 + expected * n);
        let header_params = ImageParams {
            width,
            height,
            channels: ch as u32,
            group_size_shift: tune.group_size_shift,
            bit_depth: 8,
            animation: Some(AnimHeader {
                tps_num: tps_num.max(1),
                tps_den: tps_den.max(1),
                num_loops,
            }),
            duration: 0,
            is_last: false,
            xyb_encoded: false,
            vardct: false,
            gab: false,
            epf_iters: 0,
        };
        write_image_header(&mut out, header_params);
        for (i, &(duration, pixels)) in frames.iter().enumerate() {
            if pixels.len() != expected {
                return Err(Error::Encode(format!(
                    "frame {i} pixel buffer length {} != {expected}",
                    pixels.len()
                )));
            }
            let pal = try_palette(
                width as usize,
                height as usize,
                ch,
                pixels,
                self.effective_palette_colors(),
            );
            let planes = if pal.is_some() {
                Vec::new()
            } else {
                planarize_and_transform(width as usize, height as usize, ch, pixels)
            };
            let mut fp = header_params;
            fp.duration = duration.max(1);
            fp.is_last = i + 1 == n;
            let sections = self.build_sections(fp, format, planes, pal, &tune)?;
            write_frame_and_toc(&mut out, fp, &sections);
        }
        Ok(out.finish())
    }

    fn encode_with_params(
        &self,
        params: ImageParams,
        format: PixelFormat,
        planes: Vec<Vec<i32>>,
        pal: Option<palette::PalettePlanes>,
        tune: &EncodeTune,
    ) -> Result<Vec<u8>> {
        let sections = self.build_sections(params, format, planes, pal, tune)?;
        let mut out =
            BitWriter::with_capacity(64 + sections.iter().map(|s| s.len()).sum::<usize>());
        write_image_header(&mut out, params);
        write_frame_and_toc(&mut out, params, &sections);
        Ok(out.finish())
    }

    fn build_sections(
        &self,
        params: ImageParams,
        format: PixelFormat,
        planes: Vec<Vec<i32>>,
        pal: Option<palette::PalettePlanes>,
        tune: &EncodeTune,
    ) -> Result<Vec<Vec<u8>>> {
        let ch = format.channels();
        let width = params.width;
        let height = params.height;

        let used_palette = pal.is_some();
        let (mut channels, nb_meta, mut transforms) = if let Some(pal) = pal {
            let t = ModularTransform::Palette {
                begin_c: 0,
                num_c: pal.num_c,
                nb_colours: pal.nb_colours,
            };
            (vec![pal.meta, pal.index], 1u32, vec![t])
        } else {
            let chs: Vec<EncChannel> = planes
                .into_iter()
                .map(|p| EncChannel::from_plane(width, height, p))
                .collect();
            let mut tr = Vec::new();
            if ch >= 3 && !params.xyb_encoded {
                tr.push(ModularTransform::RctYCoCg { begin_c: 0 });
            }
            (chs, 0u32, tr)
        };

        let mut tune = *tune;
        let squeeze = self.use_squeeze();
        if squeeze {
            let steps = default_squeeze_params(&channels, nb_meta);
            if !steps.is_empty() {
                apply_squeeze(&mut channels, &steps);
                transforms.push(ModularTransform::SqueezeDefault);
            }
        }

        let dests = classify_channels(&channels, nb_meta, params.group_dim());
        if self.distance > 0.0 && self.effort >= 3 && dests.iter().all(|d| *d == Dest::Gmodular) {
            tune.lz77 = true;
        }
        let n_enc = channels.len();
        let n_orig = ch;
        let lossy = self.distance > 0.0 && !used_palette;
        let plan = if lossy {
            Some(lossy_plan(
                &mut channels,
                &dests,
                n_orig,
                squeeze,
                self.distance,
                params.bit_depth,
                params.num_lf_groups(),
                params.num_groups(),
                params.xyb_encoded,
            ))
        } else {
            None
        };
        if plan.as_ref().is_some_and(|p| p.global_groups) {
            tune.lz77 = true;
        }
        let freqs = collect_freqs_enc(
            &params,
            &channels,
            &dests,
            tune.lz77,
            tune.lz77_min_len,
            tune.quant,
            plan.as_ref(),
        );
        // Lossless: squeezed / paletted / RGBA share one Gradient leaf; 1..=3
        // colour channels without squeeze keep per-channel leaves.
        // Lossy: `lossy_plan` already grouped LF/HF (and Y vs chroma).
        let (leaf_codes, codes_by_channel, ma_shape, cluster_of) = if let Some(ref p) = plan {
            let gmod_split = p.gmod_ctx.iter().any(|c| ctx_is_split(c, p.hf_leaves));
            if p.global_groups || gmod_split {
                let (leaf_codes, cluster_of) = lossy_global_hists(
                    &params,
                    &channels,
                    &dests,
                    p,
                    tune.lz77,
                    tune.lz77_min_len,
                    params.bit_depth < 16 && !params.xyb_encoded,
                );
                let codes_by_channel: Vec<AnsHist> = (0..n_enc)
                    .map(|c| {
                        let leaf = p.leaf_of.get(c).copied().unwrap_or(0);
                        let cl = cluster_of.get(leaf).copied().unwrap_or(0) as usize;
                        leaf_codes
                            .get(cl)
                            .cloned()
                            .unwrap_or_else(|| leaf_codes[0].clone())
                    })
                    .collect();
                (
                    leaf_codes,
                    codes_by_channel,
                    p.shape.clone(),
                    Some(cluster_of),
                )
            } else {
                let n_leaves = p.leaf_of.iter().copied().max().unwrap_or(0) + 1;
                let n_hist = n_leaves + usize::from(tune.lz77);
                let gmod_only = |c: usize| dests[c] == Dest::Gmodular;
                let mut leaf_freqs = vec![vec![0u64; 1]; n_leaves];
                for (c, f) in freqs.iter().enumerate() {
                    if !gmod_only(c) {
                        continue;
                    }
                    let leaf = p.leaf_of[c];
                    if leaf_freqs[leaf].len() < f.len() {
                        leaf_freqs[leaf].resize(f.len(), 0);
                    }
                    for (i, &v) in f.iter().enumerate() {
                        leaf_freqs[leaf][i] = leaf_freqs[leaf][i].saturating_add(v);
                    }
                }
                for f in &mut leaf_freqs {
                    if f.iter().all(|&x| x == 0) {
                        f[0] = 1;
                    }
                }
                let (leaf_codes, cluster_of) = if tune.lz77 && n_leaves > 1 {
                    // Tiny all-gmodular frames: histogram headers dominate. Pass
                    // groups keep per-leaf tables (woman).
                    let merged = AnsHist::from_freqs(&merge_freqs(&leaf_freqs));
                    (vec![merged], Some(vec![0u8; n_leaves]))
                } else if n_hist <= 8 {
                    let leaf_codes: Vec<AnsHist> =
                        leaf_freqs.iter().map(|f| AnsHist::from_freqs(f)).collect();
                    (leaf_codes, None::<Vec<u8>>)
                } else {
                    // Gmodular Squeeze makes ~30 tiny planes. Seven q-bucket tables
                    // beat 30 histogram headers; pass groups keep per-channel tables.
                    let mut raw = vec![0u8; n_leaves];
                    for (c, &leaf) in p.leaf_of.iter().enumerate() {
                        if gmod_only(c) {
                            raw[leaf] = quant_cluster(p.quant[c]);
                        }
                    }
                    let cluster_of = densify_clusters(&raw);
                    let n_cl = 1 + *cluster_of.iter().max().unwrap_or(&0) as usize;
                    let mut cf = vec![vec![0u64; 1]; n_cl];
                    for (c, f) in freqs.iter().enumerate() {
                        if !gmod_only(c) {
                            continue;
                        }
                        let cl = cluster_of[p.leaf_of[c]] as usize;
                        if cf[cl].len() < f.len() {
                            cf[cl].resize(f.len(), 0);
                        }
                        for (i, &v) in f.iter().enumerate() {
                            cf[cl][i] = cf[cl][i].saturating_add(v);
                        }
                    }
                    for f in &mut cf {
                        if f.iter().all(|&x| x == 0) {
                            f[0] = 1;
                        }
                    }
                    (
                        cf.iter().map(|f| AnsHist::from_freqs(f)).collect(),
                        Some(cluster_of),
                    )
                };
                let codes_by_channel: Vec<AnsHist> = (0..n_enc)
                    .map(|c| {
                        if gmod_only(c) {
                            if let Some(ref co) = cluster_of {
                                leaf_codes[co[p.leaf_of[c]] as usize].clone()
                            } else {
                                leaf_codes[p.leaf_of[c]].clone()
                            }
                        } else {
                            AnsHist::from_freqs(&freqs[c])
                        }
                    })
                    .collect();
                (leaf_codes, codes_by_channel, p.shape.clone(), cluster_of)
            }
        } else {
            let tree_leaves = if squeeze || used_palette || ch == 4 || n_enc > 3 {
                1
            } else {
                n_enc as u32
            };
            let (leaf_codes, codes_by_channel) = if tree_leaves == 1 {
                let shared = AnsHist::from_freqs(&merge_freqs(&freqs));
                (vec![shared.clone()], vec![shared; n_enc])
            } else {
                let codes_by_channel: Vec<AnsHist> =
                    freqs.iter().map(|f| AnsHist::from_freqs(f)).collect();
                let leaf_codes = leaf_channel_order(tree_leaves)
                    .into_iter()
                    .map(|c| codes_by_channel[c as usize].clone())
                    .collect();
                (leaf_codes, codes_by_channel)
            };
            (
                leaf_codes,
                codes_by_channel,
                MaShape::PerChannel {
                    n: tree_leaves,
                    mul_log: tune.mul_log,
                    mul_bits: tune.mul_bits,
                },
                None::<Vec<u8>>,
            )
        };

        let lf_global = encode_lf_global(
            params,
            &channels,
            &dests,
            &leaf_codes,
            &codes_by_channel,
            &transforms,
            &tune,
            &ma_shape,
            plan.as_ref(),
            cluster_of.as_deref(),
        );
        let mut sections: Vec<Vec<u8>> = Vec::new();

        if params.is_single_group() {
            sections.push(lf_global);
        } else {
            sections.push(lf_global);
            for g in 0..params.num_lf_groups() {
                sections.push(encode_lf_group(
                    params,
                    &channels,
                    &dests,
                    &leaf_codes,
                    &codes_by_channel,
                    g,
                    &tune,
                    plan.as_ref(),
                    cluster_of.as_deref(),
                ));
            }
            sections.push(Vec::new()); // HF global
            for g in 0..params.num_groups() {
                sections.push(encode_pass_group(
                    params,
                    &channels,
                    &dests,
                    &leaf_codes,
                    &codes_by_channel,
                    g,
                    &tune,
                    plan.as_ref(),
                    cluster_of.as_deref(),
                ));
            }
        }
        Ok(sections)
    }
}

fn write_frame_and_toc(out: &mut BitWriter, params: ImageParams, sections: &[Vec<u8>]) {
    let sizes: Vec<u32> = sections.iter().map(|s| s.len() as u32).collect();
    write_frame_header(out, params);
    write_toc(out, &sizes);
    for s in sections {
        out.append_bytes(s);
    }
}

#[derive(Clone, Copy, Debug)]
struct EncodeTune {
    group_size_shift: u32,
    lz77: bool,
    lz77_min_len: u32,
    quant: u32,
    mul_log: u32,
    mul_bits: u32,
}

/// libjxl `QualityToDistance` mapping used by cjxl `-q`.
pub fn quality_to_distance(quality: f32) -> f32 {
    let q = quality.clamp(0.0, 100.0);
    if q >= 100.0 {
        0.0
    } else if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        (53.0 / 3000.0) * q * q - (23.0 / 20.0) * q + 23.0
    }
}

fn distance_to_quant(d: f32) -> u32 {
    if d <= 0.0 {
        1
    } else {
        (d * 4.0).round().clamp(1.0, 64.0) as u32
    }
}

/// `multiplier = (mul_bits + 1) << mul_log` with both fields in 0..=7.
fn multiplier_parts(want: u32) -> (u32, u32, u32) {
    let want = want.max(1);
    let mut best = (0u32, 0u32, 1u32);
    let mut best_err = want.abs_diff(1);
    for log in 0..=7 {
        for bits in 0..=7 {
            let v: u32 = (bits + 1) << log;
            let err = v.abs_diff(want);
            if err < best_err || (err == best_err && v < best.2) {
                best = (log, bits, v);
                best_err = err;
            }
        }
    }
    best
}

/// Exact residual multiplier. Compact 0..=7 when that hits `want`, else
/// `mul_log=0, mul_bits=want-1` (wide MA-tree uint, matching libjxl `q`).
fn multiplier_parts_exact(want: u32) -> (u32, u32, u32) {
    let want = want.max(1);
    for log in 0..=7 {
        for bits in 0..=7 {
            if (bits + 1) << log == want {
                return (log, bits, want);
            }
        }
    }
    (0, want - 1, want)
}

/// libjxl default Squeeze luma table (quality 50); other distances scale it.
const LUMA_QTABLE: [f32; 16] = [
    163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01,
    0.005,
];
/// Finest two chroma layers (~4:2:0) at d=1 with the 0.35 quality factor.
const CHROMA_QTABLE: [f32; 16] = [
    1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5,
];
/// libjxl `squeeze_xyb_qtable` (Y, X, B−Y).
const XYB_QTABLE: [[f32; 16]; 3] = [
    [
        163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02,
        0.01, 0.005,
    ],
    [
        1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5,
    ],
    [
        2048.0, 1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5,
        0.5,
    ],
];

fn squeeze_quant(
    distance: f32,
    maxval: f32,
    shift: i32,
    component: usize,
    squeezed: bool,
    xyb: bool,
) -> u32 {
    let mut s = (shift.max(0) as usize).min(15);
    s = s.saturating_sub(1);
    // libjxl v0.11.2: 0.25 * dist * bitdepth_correction * table (linear dist).
    let mut q = if xyb && component < 3 {
        0.25 * distance * 4.8 * XYB_QTABLE[component][s]
    } else {
        let chroma = component == 1 || component == 2;
        let table = if chroma {
            CHROMA_QTABLE[s]
        } else {
            1.1 * LUMA_QTABLE[s]
        };
        0.25 * distance * (maxval / 255.0) * 0.35 * table
    };
    if !squeezed {
        q *= 0.1;
    }
    q.round().clamp(1.0, 1_048_576.0) as u32
}

fn squeeze_component(i: usize, n_orig: usize, n_total: usize, squeezed: bool) -> usize {
    if n_orig < 3 {
        return 0;
    }
    // libjxl: last 4 squeeze channels are final chroma residuals.
    if squeezed && n_orig > 2 && i + 4 >= n_total {
        return 1;
    }
    i % n_orig
}

/// cjxl modular lossy: XYB as Y, X, B−Y integers (`enc_factors` 4096 / 65536 / 4096).
fn xyb_modular_planes(width: u32, height: u32, pixels: &[u8]) -> Vec<Vec<i32>> {
    let w = width as usize;
    let h = height as usize;
    let (xs, ys, bs) = srgb8_rgb_to_xyb_padded(width, height, pixels, w, h);
    let n = w * h;
    let mut y = vec![0i32; n];
    let mut x = vec![0i32; n];
    let mut bmy = vec![0i32; n];
    for i in 0..n {
        y[i] = round_toward_inf(ys[i] * 4096.0);
        x[i] = round_toward_inf(xs[i] * 65536.0);
        bmy[i] = round_toward_inf(bs[i] * 4096.0) - y[i];
    }
    vec![y, x, bmy]
}

fn round_toward_inf(v: f32) -> i32 {
    if v < 0.0 {
        (v - 0.5).ceil() as i32
    } else {
        (v + 0.5).floor() as i32
    }
}

struct LossyPlan {
    shape: MaShape,
    leaf_of: Vec<usize>,
    quant: Vec<u32>,
    zero: Vec<bool>,
    gmod_ctx: Vec<[u8; 8]>,
    lf_ctx: Vec<[u8; 8]>,
    pass_ctx: Vec<[u8; 8]>,
    /// Pass/LF groups reuse the global tree (no per-group MA + histograms).
    global_groups: bool,
    hf_split: HfSplit,
    hf_tree: ResTree,
    hf_leaves: usize,
    tree_leaves: usize,
}

/// Per-band quant + one MA leaf per Squeeze plane (libjxl tables).
///
/// Residual divisor matches that leaf's multiplier. Last-4 chroma extras with
/// q ≥ 32 are zeroed (4:2:0 at typical d=1). More than 8 entropy contexts share
/// one prefix table (simple clustering only has a 3-bit cluster id).
///
/// Global tree covers gmod plus pass/LF when those dests exist: property 1
/// (`stream_idx`) then a per-kind channel spine. Groups set `use_global_tree`
/// so we do not rewrite the MA tree in every pass/LF group. Stream splits are
/// BFS-serialized (DFS left-chains are not valid once the left child is more
/// than a leaf).
#[allow(clippy::too_many_arguments)]
fn lossy_plan(
    channels: &mut [EncChannel],
    dests: &[Dest],
    n_orig: usize,
    squeezed: bool,
    distance: f32,
    bit_depth: u32,
    num_lf_groups: u32,
    num_groups: u32,
    xyb: bool,
) -> LossyPlan {
    let n = channels.len();
    let n_orig = n_orig.min(n).max(1);
    let squeezed = squeezed && n > n_orig;
    let maxval = if bit_depth >= 32 {
        1.0
    } else {
        ((1u32 << bit_depth.min(31)) - 1) as f32
    };
    let want: Vec<u32> = channels
        .iter()
        .enumerate()
        .map(|(i, ch)| {
            let component = squeeze_component(i, n_orig, n, squeezed);
            squeeze_quant(
                distance,
                maxval,
                ch.hshift + ch.vshift,
                component,
                squeezed,
                xyb,
            )
        })
        .collect();

    if squeezed && n_orig >= 3 && n >= n_orig + 4 && !xyb {
        for i in n - 4..n {
            if want[i] >= 32 {
                channels[i].samples.fill(0);
            }
        }
    }
    if squeezed {
        for (ch, &q) in channels.iter_mut().zip(&want) {
            predict::quantize_to_grid(&mut ch.samples, q);
        }
    }

    // libjxl uses Zero for every Squeeze plane, including the coarsest LF.
    let zero: Vec<bool> = vec![squeezed; n];
    let make_leaf = |i: usize| {
        let (log, bits, _) = multiplier_parts_exact(want[i]);
        if zero[i] {
            MaLeaf::zero(log, bits)
        } else {
            MaLeaf::gradient(log, bits)
        }
    };

    let mut g_idx = Vec::new();
    let mut lf_idx = Vec::new();
    let mut pass_idx = Vec::new();
    for (i, dest) in dests.iter().enumerate() {
        match *dest {
            Dest::Gmodular => g_idx.push(i),
            Dest::LfGroup => lf_idx.push(i),
            Dest::PassGroup => pass_idx.push(i),
        }
    }
    // Global tree+hists (and LZ77) beat per-group copies from 2 groups up.
    // Very sparse pass planes (parrot) still prefer local RLE tables when
    // there are fewer than 6 groups.
    let sparse_pass = pass_zero_frac(channels, dests) >= 0.80;
    let global_groups = (!lf_idx.is_empty() || !pass_idx.is_empty())
        && num_groups >= 2
        && (num_groups >= 6 || !sparse_pass);
    let mixed = |i: usize| mixed_plane(&channels[i]);
    // Squeeze HF that still fits in one group (kodim at group_dim 512) lives
    // in gmodular. Only split those when there are no pass/LF groups —
    // cloning the 6-leaf tree onto every gmod plane plus pass exceeds 256
    // MA leaves (woman).
    let n_mixed_gmod = g_idx.iter().filter(|&&i| zero[i] && mixed(i)).count();
    let gmod_split = squeezed
        && lf_idx.is_empty()
        && pass_idx.is_empty()
        && n_mixed_gmod > 0
        && n_mixed_gmod.saturating_mul(5) + g_idx.len() < 250;
    let hf_split = HfSplit::default();
    let hf_tree = if global_groups || gmod_split {
        ResTree::wn_then_n_sign()
    } else {
        ResTree::default_wn()
    };
    let hf_leaves = hf_tree.n_leaves().clamp(1, 8);
    let make_hf = |i: usize| {
        let (log, bits, _) = multiplier_parts_exact(want[i]);
        if zero[i] {
            MaLeaf::zero_split_2(
                log,
                bits,
                hf_split.p0,
                hf_split.v0,
                hf_split.p1,
                hf_split.v1,
            )
        } else {
            MaLeaf::gradient(log, bits)
        }
    };
    let mut gmod: Vec<MaLeaf> = g_idx
        .iter()
        .map(|&i| {
            if gmod_split && zero[i] && mixed(i) {
                make_hf(i)
            } else {
                make_leaf(i)
            }
        })
        .collect();
    let lf: Vec<MaLeaf> = if global_groups {
        lf_idx.iter().map(|&i| make_hf(i)).collect()
    } else {
        Vec::new()
    };
    let pass: Vec<MaLeaf> = if global_groups {
        pass_idx.iter().map(|&i| make_hf(i)).collect()
    } else {
        Vec::new()
    };
    if gmod.is_empty() && !global_groups {
        gmod = vec![MaLeaf::zero(0, 0)];
    }
    let stream_cut = if lf.is_empty() { 0 } else { 2 * num_lf_groups };
    let [gmod_ctx, lf_ctx, pass_ctx] =
        ma_stream_ctxs(&gmod, &lf, &pass, stream_cut, Some(&hf_tree));
    let tree_leaves = ma_stream_leaf_count(&gmod, &lf, &pass, stream_cut, Some(&hf_tree));
    let mut leaf_of = vec![0usize; n];
    for (local, &g) in g_idx.iter().enumerate() {
        leaf_of[g] = gmod_ctx[local][0] as usize;
    }
    LossyPlan {
        shape: MaShape::Leaves {
            gmod,
            lf,
            pass,
            stream_cut,
            prop_split: 0,
            hf: Some(hf_tree.clone()),
        },
        leaf_of,
        quant: want,
        zero,
        gmod_ctx,
        lf_ctx,
        pass_ctx,
        global_groups,
        hf_split,
        hf_tree,
        hf_leaves,
        tree_leaves,
    }
}

fn mixed_plane(ch: &EncChannel) -> bool {
    !ch.samples.is_empty() && ch.samples.contains(&0) && ch.samples.iter().any(|&s| s != 0)
}

fn pass_zero_frac(channels: &[EncChannel], dests: &[Dest]) -> f32 {
    let mut n = 0u64;
    let mut z = 0u64;
    for (ch, dest) in channels.iter().zip(dests) {
        if *dest != Dest::PassGroup {
            continue;
        }
        n += ch.samples.len() as u64;
        z += ch.samples.iter().filter(|&&s| s == 0).count() as u64;
    }
    if n == 0 { 0.0 } else { z as f32 / n as f32 }
}

fn ctx_is_split(ctx: &[u8], n: usize) -> bool {
    let n = n.max(1).min(ctx.len());
    (1..n).any(|i| ctx[i] != ctx[0])
}

/// Cluster map for one MA tree: residual slots `0..n_hf` then unsplit.
///
/// Length equals the number of MA leaves. Ids are dense `0..n_cl` (libjxl
/// `ClusterHole` is `max(id)+1 != unique`).
fn hf_cluster_map(
    gmod: &[[u8; 8]],
    lf: &[[u8; 8]],
    pass: &[[u8; 8]],
    n_hf: usize,
    n_ctx: usize,
) -> Vec<u8> {
    let n_hf = n_hf.clamp(1, 8);
    let mut raw = vec![0xffu8; n_ctx.max(1)];
    for ctx in gmod.iter().chain(lf).chain(pass) {
        if ctx_is_split(ctx, n_hf) {
            for (q, &c) in ctx.iter().take(n_hf).enumerate() {
                let leaf = c as usize;
                if leaf < raw.len() {
                    raw[leaf] = q as u8;
                }
            }
        }
    }
    let mut next = n_hf as u8;
    for ctx in gmod.iter().chain(lf).chain(pass) {
        if !ctx_is_split(ctx, n_hf) {
            let leaf = ctx[0] as usize;
            if leaf < raw.len() && raw[leaf] == 0xff {
                raw[leaf] = next;
                next = next.saturating_add(1);
            }
        }
    }
    for x in &mut raw {
        if *x == 0xff {
            *x = 0;
        }
    }
    densify_clusters(&raw)
}

fn quant_cluster(q: u32) -> u8 {
    match q {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        _ => 6,
    }
}

fn densify_clusters(raw: &[u8]) -> Vec<u8> {
    let mut map = [0xffu8; 256];
    let mut next = 0u8;
    let mut out = vec![0u8; raw.len()];
    for (i, &r) in raw.iter().enumerate() {
        if map[r as usize] == 0xff {
            map[r as usize] = next;
            next += 1;
        }
        out[i] = map[r as usize];
    }
    out
}

fn leaf_channel_order(n: u32) -> Vec<u32> {
    match n {
        1 => vec![0],
        2 => vec![1, 0],
        3 => vec![2, 1, 0],
        4 => vec![3, 2, 1, 0],
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dest {
    Gmodular,
    LfGroup,
    PassGroup,
}

fn classify_channels(channels: &[EncChannel], nb_meta: u32, group_dim: u32) -> Vec<Dest> {
    let mut still_gmod = true;
    let mut dests = Vec::with_capacity(channels.len());
    for (i, ch) in channels.iter().enumerate() {
        if still_gmod && (i < nb_meta as usize || (ch.width <= group_dim && ch.height <= group_dim))
        {
            dests.push(Dest::Gmodular);
        } else {
            still_gmod = false;
            if ch.hshift >= 3 && ch.vshift >= 3 {
                dests.push(Dest::LfGroup);
            } else {
                dests.push(Dest::PassGroup);
            }
        }
    }
    dests
}

fn crop_plane(plane: &[i32], width: usize, x: usize, y: usize, gw: usize, gh: usize) -> Vec<i32> {
    let mut out = vec![0i32; gw * gh];
    for row in 0..gh {
        let src = (y + row) * width + x;
        out[row * gw..(row + 1) * gw].copy_from_slice(&plane[src..src + gw]);
    }
    out
}

/// Crop of a squeezed plane for pass group `g` (`lf = false`) or LF group `g`.
fn group_crop(
    ch: &EncChannel,
    params: &ImageParams,
    g: u32,
    lf: bool,
) -> Option<(usize, usize, usize, usize)> {
    if ch.is_empty() {
        return None;
    }
    let hs = ch.hshift.max(0) as u32;
    let vs = ch.vshift.max(0) as u32;
    let (gw, gh, stride, gy) = if lf {
        let dim = params.group_dim() << 3;
        let gw = dim >> hs;
        let gh = dim >> vs;
        if gw == 0 || gh == 0 {
            return None;
        }
        let lf_per_row = params.width.div_ceil(dim).max(1);
        (gw, gh, lf_per_row, g / lf_per_row)
    } else {
        let dim = params.group_dim();
        let gw = dim >> hs;
        let gh = dim >> vs;
        if gw == 0 || gh == 0 {
            return None;
        }
        let stride = params.groups_per_row();
        (gw, gh, stride, g / stride)
    };
    let gx = g % stride;
    let x = ((gx * gw) as usize).min(ch.width as usize);
    let y = ((gy * gh) as usize).min(ch.height as usize);
    let w = (ch.width as usize - x).min(gw as usize);
    let h = (ch.height as usize - y).min(gh as usize);
    if w == 0 || h == 0 {
        None
    } else {
        Some((x, y, w, h))
    }
}

fn plane_residuals(ch: &EncChannel, quant: u32, zero: bool) -> Vec<u32> {
    let w = ch.width as usize;
    let h = ch.height as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut res = vec![0u32; w * h];
    if zero {
        zero_residuals(&ch.samples, quant, &mut res);
    } else {
        gradient_residuals(w, h, &ch.samples, quant, &mut res);
    }
    res
}

fn plane_residuals_c(
    ch: &EncChannel,
    c: usize,
    fallback_q: u32,
    plan: Option<&LossyPlan>,
) -> Vec<u32> {
    match plan {
        Some(p) => plane_residuals(ch, p.quant[c], p.zero[c]),
        None => plane_residuals(ch, fallback_q, false),
    }
}

fn crop_residuals_pred(
    ch: &EncChannel,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    quant: u32,
    zero: bool,
) -> Vec<u32> {
    let crop = crop_plane(&ch.samples, ch.width as usize, x, y, w, h);
    let mut res = vec![0u32; w * h];
    if zero {
        zero_residuals(&crop, quant, &mut res);
    } else {
        gradient_residuals(w, h, &crop, quant, &mut res);
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn crop_residuals_c(
    ch: &EncChannel,
    c: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    fallback_q: u32,
    plan: Option<&LossyPlan>,
) -> Vec<u32> {
    match plan {
        Some(p) => crop_residuals_pred(ch, x, y, w, h, p.quant[c], p.zero[c]),
        None => crop_residuals_pred(ch, x, y, w, h, fallback_q, false),
    }
}

fn acc_tokens(freqs: &mut Vec<u64>, residuals: &[u32], lz77: bool, min_len: u32) {
    for_each_token_len(residuals, lz77, min_len, |token, _, _| {
        let t = token as usize;
        if t >= freqs.len() {
            freqs.resize(t + 1, 0);
        }
        freqs[t] += 1;
    });
}

fn collect_freqs_enc(
    params: &ImageParams,
    channels: &[EncChannel],
    dests: &[Dest],
    lz77: bool,
    min_len: u32,
    quant: u32,
    plan: Option<&LossyPlan>,
) -> Vec<Vec<u64>> {
    let cap = if lz77 { 257 } else { 32 };
    let mut freqs = vec![vec![0u64; cap]; channels.len()];
    for (c, (ch, dest)) in channels.iter().zip(dests).enumerate() {
        match dest {
            Dest::Gmodular => acc_tokens(
                &mut freqs[c],
                &plane_residuals_c(ch, c, quant, plan),
                lz77,
                min_len,
            ),
            Dest::LfGroup => {
                for g in 0..params.num_lf_groups() {
                    if let Some((x, y, w, h)) = group_crop(ch, params, g, true) {
                        acc_tokens(
                            &mut freqs[c],
                            &crop_residuals_c(ch, c, x, y, w, h, quant, plan),
                            lz77,
                            min_len,
                        );
                    }
                }
            }
            Dest::PassGroup => {
                for g in 0..params.num_groups() {
                    if let Some((x, y, w, h)) = group_crop(ch, params, g, false) {
                        acc_tokens(
                            &mut freqs[c],
                            &crop_residuals_c(ch, c, x, y, w, h, quant, plan),
                            lz77,
                            min_len,
                        );
                    }
                }
            }
        }
    }
    for f in &mut freqs {
        if f.iter().all(|&x| x == 0) {
            f[0] = 1;
        }
    }
    freqs
}

fn merge_freqs(freqs: &[Vec<u64>]) -> Vec<u64> {
    let cap = freqs.iter().map(Vec::len).max().unwrap_or(1);
    let mut out = vec![0u64; cap];
    for f in freqs {
        for (i, &v) in f.iter().enumerate() {
            out[i] = out[i].saturating_add(v);
        }
    }
    out
}

fn gmod_cluster(
    c: usize,
    plan: Option<&LossyPlan>,
    cluster_of: Option<&[u8]>,
    n_enc: usize,
    n_leaf_codes: usize,
) -> u8 {
    if let Some(p) = plan {
        let leaf = p.leaf_of[c];
        cluster_of.map(|co| co[leaf]).unwrap_or(leaf as u8)
    } else if n_leaf_codes <= 1 {
        0
    } else {
        // PerChannel tree: ctx 0 is the last colour channel.
        (n_enc - 1 - c) as u8
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_lf_global(
    params: ImageParams,
    channels: &[EncChannel],
    dests: &[Dest],
    leaf_codes: &[AnsHist],
    _codes_by_channel: &[AnsHist],
    transforms: &[ModularTransform],
    tune: &EncodeTune,
    ma_shape: &MaShape,
    plan: Option<&LossyPlan>,
    cluster_of: Option<&[u8]>,
) -> Vec<u8> {
    let mut w = BitWriter::with_capacity((params.width * params.height) as usize + 1024);
    if params.xyb_encoded {
        // Custom DC: InvDCQuant = (65536, 4096, 4096); stored as m_*_lf = 128 / inv.
        w.put_bool(false);
        w.put_f16(1.0 / 512.0);
        w.put_f16(1.0 / 32.0);
        w.put_f16(1.0 / 32.0);
    } else {
        w.put_bool(true); // LfChannelDequantization all_default
    }
    w.put_bool(true); // global MA tree present
    write_ma_tree(&mut w, ma_shape);
    let dist = tune.lz77.then(ans_rle_distance);
    write_ans_histograms_len(
        &mut w,
        leaf_codes,
        cluster_of,
        dist.as_ref(),
        tune.lz77_min_len,
    );
    write_modular_header(&mut w, transforms);

    let dist_cluster = dist.as_ref().map(|_| leaf_codes.len() as u8);
    let mut tokens: Vec<AnsToken> = Vec::new();
    let n_enc = channels.len();
    let remap = |leaf: u8| -> u8 {
        cluster_of
            .and_then(|co| co.get(leaf as usize).copied())
            .unwrap_or(leaf)
    };
    let mut gi = 0usize;
    let narrow = params.bit_depth < 16 && !params.xyb_encoded;
    for (c, (ch, dest)) in channels.iter().zip(dests).enumerate() {
        if *dest != Dest::Gmodular {
            continue;
        }
        if let Some(p) = plan
            && gi < p.gmod_ctx.len()
        {
            let ctx8 = p.gmod_ctx[gi];
            gi += 1;
            if ch.is_empty() {
                continue;
            }
            let res = plane_residuals_c(ch, c, tune.quant, plan);
            if ctx_is_split(&ctx8, p.hf_leaves) {
                let ctxs: Vec<u8> =
                    split_res_tree(ch.width as usize, &res, p.quant[c], narrow, &p.hf_tree)
                        .into_iter()
                        .map(|q| remap(ctx8[q.min(7) as usize]))
                        .collect();
                collect_ans_tokens_ctx_len(
                    &res,
                    &ctxs,
                    tune.lz77,
                    dist_cluster,
                    tune.lz77_min_len,
                    &mut tokens,
                );
                continue;
            }
            collect_ans_tokens_len(
                &res,
                tune.lz77,
                remap(ctx8[0]),
                dist_cluster,
                tune.lz77_min_len,
                &mut tokens,
            );
            continue;
        }
        if ch.is_empty() {
            continue;
        }
        let res = plane_residuals_c(ch, c, tune.quant, plan);
        let cluster = gmod_cluster(c, plan, cluster_of, n_enc, leaf_codes.len());
        collect_ans_tokens_len(
            &res,
            tune.lz77,
            cluster,
            dist_cluster,
            tune.lz77_min_len,
            &mut tokens,
        );
    }
    write_ans_tokens_ex(&mut w, leaf_codes, dist.as_ref(), &tokens);
    w.finish()
}

#[allow(clippy::too_many_arguments)]
fn encode_grouped(
    params: ImageParams,
    channels: &[EncChannel],
    dests: &[Dest],
    leaf_codes: &[AnsHist],
    _codes_by_channel: &[AnsHist],
    group_idx: u32,
    lf: bool,
    want: Dest,
    tune: &EncodeTune,
    plan: Option<&LossyPlan>,
    cluster_of: Option<&[u8]>,
) -> Vec<u8> {
    let mut pieces: Vec<GroupPiece> = Vec::new();
    let mut local = 0usize;
    for (i, (ch, dest)) in channels.iter().zip(dests).enumerate() {
        if *dest != want {
            continue;
        }
        if let Some((x, y, w, h)) = group_crop(ch, &params, group_idx, lf) {
            pieces.push(GroupPiece {
                ch: i,
                local,
                res: crop_residuals_c(ch, i, x, y, w, h, tune.quant, plan),
                width: w,
            });
        }
        local += 1;
    }
    if pieces.is_empty() {
        return Vec::new();
    }
    let mut w = BitWriter::with_capacity(4096);
    let nres: usize = pieces.iter().map(|p| p.res.len()).sum();
    let nzero: usize = pieces
        .iter()
        .map(|p| p.res.iter().filter(|&&x| x == 0).count())
        .sum();
    let global_groups = plan.is_some_and(|p| p.global_groups);
    let local_tree = plan.is_some() && want != Dest::Gmodular && !global_groups;
    let local_lz77 =
        local_tree && pieces.len() <= 16 && nres > 0 && nzero.saturating_mul(2) >= nres;
    let use_lz77 = local_lz77 || (!local_tree && tune.lz77);
    let narrow = params.bit_depth < 16 && !params.xyb_encoded;
    let (stream_codes_owned, dist, local_ma, leaf_ctx) =
        if let Some(p) = plan.filter(|_| local_tree) {
            let (hist, dist, leaf_ctx) =
                write_local_ma(&mut w, &pieces, p, local_lz77, tune.lz77_min_len, narrow);
            (Some(hist), dist, true, leaf_ctx)
        } else {
            write_modular_header(&mut w, &[]);
            let dist = use_lz77.then(ans_rle_distance);
            let n = pieces.len() as u8;
            let leaf_ctx: Vec<[u8; 8]> = (0..n).rev().map(|c| [c; 8]).collect();
            (None, dist, false, leaf_ctx)
        };
    let stream_codes: &[AnsHist] = stream_codes_owned.as_deref().unwrap_or(leaf_codes);
    let dist_cluster = dist.as_ref().map(|_| stream_codes.len() as u8);
    let min_len = tune.lz77_min_len;
    let mut tokens: Vec<AnsToken> = Vec::new();
    let n_enc = channels.len();
    let remap = |leaf: u8| -> u8 {
        cluster_of
            .and_then(|co| co.get(leaf as usize).copied())
            .unwrap_or(leaf)
    };
    for (pi, piece) in pieces.iter().enumerate() {
        let ch_i = piece.ch;
        let res = &piece.res;
        let width = piece.width;
        if local_ma {
            let ctx8 = leaf_ctx[pi];
            let p = plan.unwrap();
            if ctx_is_split(&ctx8, p.hf_leaves) {
                let q = p.quant[ch_i];
                let ctxs: Vec<u8> = split_res_tree(width, res, q, narrow, &p.hf_tree)
                    .into_iter()
                    .map(|quad| ctx8[quad.min(7) as usize])
                    .collect();
                collect_ans_tokens_ctx_len(
                    res,
                    &ctxs,
                    use_lz77,
                    dist_cluster,
                    min_len,
                    &mut tokens,
                );
            } else {
                collect_ans_tokens_len(res, use_lz77, ctx8[0], dist_cluster, min_len, &mut tokens);
            }
        } else if global_groups {
            let p = plan.unwrap();
            let ctxs_map = if lf { &p.lf_ctx } else { &p.pass_ctx };
            let ctx8 = ctxs_map.get(piece.local).copied().unwrap_or([0; 8]);
            if ctx_is_split(&ctx8, p.hf_leaves) {
                let q = p.quant[ch_i];
                let ctxs: Vec<u8> = split_res_tree(width, res, q, narrow, &p.hf_tree)
                    .into_iter()
                    .map(|quad| remap(ctx8[quad.min(7) as usize]))
                    .collect();
                collect_ans_tokens_ctx_len(
                    res,
                    &ctxs,
                    use_lz77,
                    dist_cluster,
                    min_len,
                    &mut tokens,
                );
            } else {
                collect_ans_tokens_len(
                    res,
                    use_lz77,
                    remap(ctx8[0]),
                    dist_cluster,
                    min_len,
                    &mut tokens,
                );
            }
        } else {
            let cluster = gmod_cluster(ch_i, plan, cluster_of, n_enc, leaf_codes.len());
            collect_ans_tokens_len(res, use_lz77, cluster, dist_cluster, min_len, &mut tokens);
        }
    }
    write_ans_tokens_ex(&mut w, stream_codes, dist.as_ref(), &tokens);
    w.finish()
}

/// Squeeze-HF extra MA split. `5` = `|W|` then `|N|` (four quadrants).
///
/// Tree nodes are written in BFS (libjxl `to_decode` order).
struct GroupPiece {
    ch: usize,
    local: usize,
    res: Vec<u32>,
    width: usize,
}

fn write_local_ma(
    w: &mut BitWriter,
    pieces: &[GroupPiece],
    plan: &LossyPlan,
    lz77: bool,
    min_len: u32,
    narrow: bool,
) -> (Vec<AnsHist>, Option<AnsHist>, Vec<[u8; 8]>) {
    write_local_modular_prefix(w);
    let leaves: Vec<MaLeaf> = pieces
        .iter()
        .map(|p| {
            let (log, bits, _) = multiplier_parts(plan.quant[p.ch]);
            if plan.zero[p.ch] && p.res.contains(&0) && p.res.iter().any(|&r| r != 0) {
                MaLeaf::zero_split_2(
                    log,
                    bits,
                    plan.hf_split.p0,
                    plan.hf_split.v0,
                    plan.hf_split.p1,
                    plan.hf_split.v1,
                )
            } else if plan.zero[p.ch] {
                MaLeaf::zero(log, bits)
            } else {
                MaLeaf::gradient(log, bits)
            }
        })
        .collect();
    let split_at: Vec<bool> = leaves.iter().map(|l| l.split_prop > 0).collect();
    let leaf_ctx = ma_bfs_leaf_ctxs_hf(&leaves, Some(&plan.hf_tree));
    write_ma_tree(
        w,
        &MaShape::Leaves {
            gmod: leaves,
            lf: Vec::new(),
            pass: Vec::new(),
            stream_cut: 0,
            prop_split: 0,
            hf: Some(plan.hf_tree.clone()),
        },
    );
    let dist = lz77.then(ans_rle_distance);
    let cap = if lz77 { 257 } else { 32 };
    let n_hf = plan.hf_leaves.clamp(1, 8);
    let n_leaves = 1 + leaf_ctx
        .iter()
        .enumerate()
        .map(|(pi, ctx)| {
            if split_at[pi] {
                ctx[..n_hf].iter().copied().max().unwrap_or(0)
            } else {
                ctx[0]
            }
        })
        .max()
        .unwrap_or(0) as usize;
    let mut freqs = vec![vec![0u64; cap]; n_leaves];
    for (pi, p) in pieces.iter().enumerate() {
        let ctx8 = leaf_ctx[pi];
        if split_at[pi] {
            let ctxs: Vec<u8> =
                split_res_tree(p.width, &p.res, plan.quant[p.ch], narrow, &plan.hf_tree)
                    .into_iter()
                    .map(|quad| ctx8[quad.min(7) as usize])
                    .collect();
            for_each_token_ctx_len(&p.res, &ctxs, lz77, min_len, |c, token, _, _| {
                let f = &mut freqs[c as usize];
                let t = token as usize;
                if t >= f.len() {
                    f.resize(t + 1, 0);
                }
                f[t] += 1;
            });
        } else {
            acc_tokens(&mut freqs[ctx8[0] as usize], &p.res, lz77, min_len);
        }
    }
    let n_split = split_at.iter().filter(|&&s| s).count();
    let (hist, quad_cl) = if n_split == 0 {
        let hist: Vec<AnsHist> = freqs
            .into_iter()
            .map(|mut f| {
                if f.iter().all(|&x| x == 0) {
                    f[0] = 1;
                }
                AnsHist::from_freqs(&f)
            })
            .collect();
        (hist, leaf_ctx)
    } else {
        let mut unsplit_cl = vec![0u8; split_at.len()];
        let mut next = n_hf as u8;
        for (i, &s) in split_at.iter().enumerate() {
            if !s {
                unsplit_cl[i] = next;
                next = next.saturating_add(1);
            }
        }
        let n_cl = next as usize;
        let mut cf = vec![vec![0u64; cap]; n_cl];
        let mut cluster_of = vec![0u8; n_leaves];
        let mut quad_cl = vec![[0u8; 8]; split_at.len()];
        for (pi, &s) in split_at.iter().enumerate() {
            if s {
                for q in 0..n_hf {
                    let leaf = leaf_ctx[pi][q] as usize;
                    let cl = q as u8;
                    cluster_of[leaf] = cl;
                    quad_cl[pi][q] = cl;
                    add_freq(&mut cf[cl as usize], &freqs[leaf]);
                }
            } else {
                let c = unsplit_cl[pi];
                let leaf = leaf_ctx[pi][0] as usize;
                cluster_of[leaf] = c;
                quad_cl[pi] = [c; 8];
                add_freq(&mut cf[c as usize], &freqs[leaf]);
            }
        }
        for f in &mut cf {
            if f.iter().all(|&x| x == 0) {
                f[0] = 1;
            }
        }
        let hist: Vec<AnsHist> = cf.into_iter().map(|f| AnsHist::from_freqs(&f)).collect();
        write_ans_histograms_len(w, &hist, Some(&cluster_of), dist.as_ref(), min_len);
        return (hist, dist, quad_cl);
    };
    write_ans_histograms_len(
        w,
        &hist,
        Some(&(0..hist.len() as u8).collect::<Vec<_>>()),
        dist.as_ref(),
        min_len,
    );
    (hist, dist, quad_cl)
}

fn lossy_global_hists(
    params: &ImageParams,
    channels: &[EncChannel],
    dests: &[Dest],
    plan: &LossyPlan,
    lz77: bool,
    min_len: u32,
    narrow: bool,
) -> (Vec<AnsHist>, Vec<u8>) {
    let cap = if lz77 { 257 } else { 32 };
    let n_hf = plan.hf_leaves.clamp(1, 8);
    let n_leaves = plan.tree_leaves.max(1);
    let mut freqs = vec![vec![0u64; cap]; n_leaves];

    let mut gi = 0usize;
    for (c, (ch, dest)) in channels.iter().zip(dests).enumerate() {
        if *dest != Dest::Gmodular {
            continue;
        }
        if gi >= plan.gmod_ctx.len() {
            continue;
        }
        let ctx8 = plan.gmod_ctx[gi];
        gi += 1;
        if ch.is_empty() {
            continue;
        }
        let res = plane_residuals_c(ch, c, 1, Some(plan));
        if ctx_is_split(&ctx8, plan.hf_leaves) {
            let mapped: Vec<u8> = split_res_tree(
                ch.width as usize,
                &res,
                plan.quant[c],
                narrow,
                &plan.hf_tree,
            )
            .into_iter()
            .map(|q| ctx8[q.min(7) as usize])
            .collect();
            for_each_token_ctx_len(&res, &mapped, lz77, min_len, |cl, token, _, _| {
                let f = &mut freqs[cl as usize];
                let t = token as usize;
                if t >= f.len() {
                    f.resize(t + 1, 0);
                }
                f[t] += 1;
            });
        } else {
            acc_tokens(&mut freqs[ctx8[0] as usize], &res, lz77, min_len);
        }
    }

    for (is_lf, want, ctxs) in [
        (true, Dest::LfGroup, plan.lf_ctx.as_slice()),
        (false, Dest::PassGroup, plan.pass_ctx.as_slice()),
    ] {
        let ng = if is_lf {
            params.num_lf_groups()
        } else {
            params.num_groups()
        };
        for g in 0..ng {
            let mut li = 0usize;
            for (c, (ch, dest)) in channels.iter().zip(dests).enumerate() {
                if *dest != want {
                    continue;
                }
                if let Some((x, y, w, h)) = group_crop(ch, params, g, is_lf)
                    && li < ctxs.len()
                {
                    let res = crop_residuals_c(ch, c, x, y, w, h, 1, Some(plan));
                    let ctx8 = ctxs[li];
                    if ctx_is_split(&ctx8, plan.hf_leaves) {
                        let mapped: Vec<u8> =
                            split_res_tree(w, &res, plan.quant[c], narrow, &plan.hf_tree)
                                .into_iter()
                                .map(|q| ctx8[q.min(7) as usize])
                                .collect();
                        for_each_token_ctx_len(&res, &mapped, lz77, min_len, |cl, token, _, _| {
                            let f = &mut freqs[cl as usize];
                            let t = token as usize;
                            if t >= f.len() {
                                f.resize(t + 1, 0);
                            }
                            f[t] += 1;
                        });
                    } else {
                        acc_tokens(&mut freqs[ctx8[0] as usize], &res, lz77, min_len);
                    }
                }
                li += 1;
            }
        }
    }

    // Large frames: one table per MA leaf. Compact frames with many leaves
    // (kodim all-gmod 6-way) share residual slots so histogram headers do
    // not dominate.
    let n_px = u64::from(params.width) * u64::from(params.height);
    let share_hf = n_leaves > 48 && n_px < 1_000_000;
    let cluster_of: Vec<u8> = if n_leaves > 256 || share_hf {
        hf_cluster_map(&plan.gmod_ctx, &plan.lf_ctx, &plan.pass_ctx, n_hf, n_leaves)
    } else {
        (0..n_leaves).map(|i| i as u8).collect()
    };
    let n_cl = 1 + usize::from(*cluster_of.iter().max().unwrap_or(&0));
    let mut cf = vec![vec![0u64; cap]; n_cl];
    for (leaf, f) in freqs.iter().enumerate() {
        if leaf < cluster_of.len() {
            add_freq(&mut cf[cluster_of[leaf] as usize], f);
        }
    }
    for f in &mut cf {
        if f.iter().all(|&x| x == 0) {
            f[0] = 1;
        }
    }
    let hist = cf.into_iter().map(|f| AnsHist::from_freqs(&f)).collect();
    (hist, cluster_of)
}

fn add_freq(dst: &mut Vec<u64>, src: &[u64]) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = d.saturating_add(s);
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_lf_group(
    params: ImageParams,
    channels: &[EncChannel],
    dests: &[Dest],
    leaf_codes: &[AnsHist],
    codes_by_channel: &[AnsHist],
    group_idx: u32,
    tune: &EncodeTune,
    plan: Option<&LossyPlan>,
    cluster_of: Option<&[u8]>,
) -> Vec<u8> {
    encode_grouped(
        params,
        channels,
        dests,
        leaf_codes,
        codes_by_channel,
        group_idx,
        true,
        Dest::LfGroup,
        tune,
        plan,
        cluster_of,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_pass_group(
    params: ImageParams,
    channels: &[EncChannel],
    dests: &[Dest],
    leaf_codes: &[AnsHist],
    codes_by_channel: &[AnsHist],
    group_idx: u32,
    tune: &EncodeTune,
    plan: Option<&LossyPlan>,
    cluster_of: Option<&[u8]>,
) -> Vec<u8> {
    encode_grouped(
        params,
        channels,
        dests,
        leaf_codes,
        codes_by_channel,
        group_idx,
        false,
        Dest::PassGroup,
        tune,
        plan,
        cluster_of,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decoder;

    fn roundtrip(width: u32, height: u32, format: PixelFormat, pixels: &[u8]) {
        let jxl = Encoder::new()
            .encode(width, height, format, pixels)
            .expect("encode");
        assert!(jxl.len() >= 2);
        assert_eq!(&jxl[..2], [0xFF, 0x0A]);

        let info = Decoder::probe_bytes(&jxl).expect("probe");
        assert_eq!(info.width, width);
        assert_eq!(info.height, height);

        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode encoded bitstream");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].width, width);
        assert_eq!(frames[0].height, height);

        let got = &frames[0].rgba;
        for i in 0..(width * height) as usize {
            let gi = i * 4;
            // jxl-oxide already applied the output transfer; packing is 0..=1 → u8.
            let rec = [got[gi], got[gi + 1], got[gi + 2], got[gi + 3]];
            match format {
                PixelFormat::Gray => {
                    near(rec[0], pixels[i], i, "gray");
                    near(rec[1], pixels[i], i, "gray");
                    near(rec[2], pixels[i], i, "gray");
                    near(rec[3], 255, i, "gray-a");
                }
                PixelFormat::GrayA => {
                    near(rec[0], pixels[i * 2], i, "ga y");
                    near(rec[3], pixels[i * 2 + 1], i, "ga a");
                }
                PixelFormat::Rgb => {
                    near(rec[0], pixels[i * 3], i, "r");
                    near(rec[1], pixels[i * 3 + 1], i, "g");
                    near(rec[2], pixels[i * 3 + 2], i, "b");
                    near(rec[3], 255, i, "rgb-a");
                }
                PixelFormat::Rgba => {
                    near(rec[0], pixels[i * 4], i, "r");
                    near(rec[1], pixels[i * 4 + 1], i, "g");
                    near(rec[2], pixels[i * 4 + 2], i, "b");
                    near(rec[3], pixels[i * 4 + 3], i, "a");
                }
            }
        }
    }

    fn near(got: u8, want: u8, i: usize, ch: &str) {
        let d = (got as i16 - want as i16).unsigned_abs();
        assert!(
            d <= 1,
            "{ch} pixel {i}: recovered {got} want {want} (Δ {d})"
        );
    }

    #[test]
    fn encode_1x1_gray() {
        roundtrip(1, 1, PixelFormat::Gray, &[128]);
    }

    #[test]
    fn encode_2x2_rgb() {
        #[rustfmt::skip]
        let px = [
            255, 0, 0,
            0, 255, 0,
            0, 0, 255,
            10, 20, 30,
        ];
        roundtrip(2, 2, PixelFormat::Rgb, &px);
    }

    #[test]
    fn encode_2x2_rgba() {
        #[rustfmt::skip]
        let px = [
            255, 0, 0, 255,
            0, 255, 0, 128,
            0, 0, 255, 64,
            10, 20, 30, 255,
        ];
        roundtrip(2, 2, PixelFormat::Rgba, &px);
    }

    #[test]
    fn encode_multi_group_rgb() {
        // group_dim is 1024; 1025 px wide forces two groups.
        let w = 1025u32;
        let h = 3u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x % 251) as u8;
                px[i + 1] = (y * 80) as u8;
                px[i + 2] = 40;
            }
        }
        roundtrip(w, h, PixelFormat::Rgb, &px);
    }

    #[test]
    fn encode_8x8_gradient_rgba() {
        let w = 8u32;
        let h = 8u32;
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                px[i] = (x * 30) as u8;
                px[i + 1] = (y * 30) as u8;
                px[i + 2] = 80;
                px[i + 3] = 255;
            }
        }
        roundtrip(w, h, PixelFormat::Rgba, &px);
    }

    #[test]
    fn quality_100_is_lossless_distance() {
        assert_eq!(quality_to_distance(100.0), 0.0);
        assert!((quality_to_distance(90.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn encode_lossy_still_decodes() {
        let mut px = vec![0u8; 16 * 16 * 3];
        for (i, p) in px.iter_mut().enumerate() {
            *p = (i * 17) as u8;
        }
        let jxl = Encoder::new()
            .quality(90.0)
            .modular(Some(true))
            .encode_rgb8(16, 16, &px)
            .expect("lossy encode");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode lossy");
        assert_eq!(frames[0].width, 16);
        assert_eq!(frames[0].rgba.len(), 16 * 16 * 4);
        assert!(frames[0].rgba.iter().any(|&b| b != 0));
    }

    #[test]
    fn encode_lossy_modular_d1_beats_lossless_size() {
        let w = 64u32;
        let h = 64u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x.wrapping_mul(3).wrapping_add(y / 2)) as u8;
                px[i + 1] = (80 + ((x as i32 - 32) / 2).clamp(0, 80)) as u8;
                px[i + 2] = (y.wrapping_mul(2)) as u8;
            }
        }
        let lossless = Encoder::new()
            .modular(Some(true))
            .palette_colors(0)
            .responsive(Some(false))
            .encode_rgb8(w, h, &px)
            .expect("lossless");
        let lossy = Encoder::new()
            .distance(1.0)
            .modular(Some(true))
            .encode_rgb8(w, h, &px)
            .expect("lossy modular");
        assert!(
            lossy.len() < lossless.len(),
            "lossy {} bytes should be smaller than lossless {}",
            lossy.len(),
            lossless.len()
        );
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&lossy)
            .expect("decode lossy modular");
        let mut sse = 0.0f64;
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            for c in 0..3 {
                let d = got[i * 4 + c] as f64 - px[i * 3 + c] as f64;
                sse += d * d;
            }
        }
        let mse = sse / (w * h * 3) as f64;
        let psnr = if mse < 1e-12 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        assert!(
            psnr > 38.0,
            "lossy modular d=1 PSNR {psnr:.2} too low (mse {mse:.3})"
        );
        let lossless_bpp = lossless.len() as f64 * 8.0 / (w * h) as f64;
        let lossy_bpp = lossy.len() as f64 * 8.0 / (w * h) as f64;
        assert!(
            lossy_bpp < lossless_bpp * 0.7,
            "lossy {lossy_bpp:.2} bpp should beat 70% of lossless {lossless_bpp:.2}"
        );
    }

    #[test]
    fn encode_lossy_modular_multi_group_psnr() {
        // libjxl modular group_dim is 256 (512 only if both axes ≤ 400).
        // 640×520 is 3×3 groups, so the global tree is used.
        let w = 640u32;
        let h = 520u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x.wrapping_mul(3).wrapping_add(y)) as u8;
                px[i + 1] = 80 + ((x as i32 - 320) / 4).clamp(0, 80) as u8;
                px[i + 2] = (y.wrapping_mul(5)) as u8;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .modular(Some(true))
            .encode_rgb8(w, h, &px)
            .expect("lossy grouped");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode grouped lossy");
        let mut sse = 0.0f64;
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            for c in 0..3 {
                let d = got[i * 4 + c] as f64 - px[i * 3 + c] as f64;
                sse += d * d;
            }
        }
        let mse = sse / (w * h * 3) as f64;
        let psnr = if mse < 1e-12 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        let bpp = jxl.len() as f64 * 8.0 / (w * h) as f64;
        assert!(
            psnr > 28.0,
            "grouped lossy modular PSNR {psnr:.2} (mse {mse:.3}) {bpp:.2} bpp"
        );
        assert!(
            bpp < 4.0,
            "grouped lossy modular {bpp:.2} bpp too high (PSNR {psnr:.2})"
        );
    }

    #[test]
    fn encode_lossy_modular_global_tree_psnr() {
        // 1280×520 at group_dim 256 is 5×3 groups (always global).
        let w = 1280u32;
        let h = 520u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x.wrapping_mul(3).wrapping_add(y)) as u8;
                px[i + 1] = 80 + ((x as i32 - 640) / 8).clamp(0, 80) as u8;
                px[i + 2] = (y.wrapping_mul(5)) as u8;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .modular(Some(true))
            .encode_rgb8(w, h, &px)
            .expect("lossy global-tree");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode global-tree lossy");
        let mut sse = 0.0f64;
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            for c in 0..3 {
                let d = got[i * 4 + c] as f64 - px[i * 3 + c] as f64;
                sse += d * d;
            }
        }
        let mse = sse / (w * h * 3) as f64;
        let psnr = if mse < 1e-12 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        let bpp = jxl.len() as f64 * 8.0 / (w * h) as f64;
        assert!(
            psnr > 28.0,
            "global-tree lossy modular PSNR {psnr:.2} (mse {mse:.3}) {bpp:.2} bpp"
        );
        assert!(
            bpp < 4.0,
            "global-tree lossy modular {bpp:.2} bpp too high (PSNR {psnr:.2})"
        );
    }

    #[test]
    fn encode_lossy_modular_gray_decodes() {
        let w = 32u32;
        let h = 32u32;
        let mut px = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                px[(y * w + x) as usize] = (x * 7 + y * 3) as u8;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .encode_gray8(w, h, &px)
            .expect("lossy gray");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode lossy gray");
        assert_eq!(frames[0].width, w);
        assert!(frames[0].rgba.iter().any(|&b| b != 0));
    }

    #[test]
    fn encode_squeeze_16x16_rgb() {
        let w = 16u32;
        let h = 16u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x * 16) as u8;
                px[i + 1] = (y * 16) as u8;
                px[i + 2] = 80;
            }
        }
        let jxl = Encoder::new()
            .responsive(Some(true))
            .palette_colors(0)
            .encode_rgb8(w, h, &px)
            .expect("squeeze encode");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode squeeze");
        assert_eq!(frames[0].width, w);
        assert_eq!(frames[0].height, h);
        roundtrip(w, h, PixelFormat::Rgb, &px); // unsqueezed still works
        // Compare squeezed roundtrip with the same helper by encoding again...
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            let rec = [got[i * 4], got[i * 4 + 1], got[i * 4 + 2]];
            near(rec[0], px[i * 3], i, "sq r");
            near(rec[1], px[i * 3 + 1], i, "sq g");
            near(rec[2], px[i * 3 + 2], i, "sq b");
        }
    }

    #[test]
    fn encode_squeeze_gray() {
        let w = 17u32;
        let h = 9u32;
        let mut px = vec![0u8; (w * h) as usize];
        for (i, p) in px.iter_mut().enumerate() {
            *p = (i * 11) as u8;
        }
        let jxl = Encoder::new()
            .responsive(Some(true))
            .palette_colors(0)
            .encode_gray8(w, h, &px)
            .expect("squeeze gray");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode squeeze gray");
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            near(got[i * 4], px[i], i, "sq gray");
        }
    }

    #[test]
    fn encode_palette_few_colors() {
        let w = 32u32;
        let h = 32u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let on = (x / 8 + y / 8) % 2 == 0;
                px[i] = if on { 200 } else { 10 };
                px[i + 1] = if on { 30 } else { 180 };
                px[i + 2] = 90;
            }
        }
        let jxl = Encoder::new()
            .responsive(Some(false))
            .palette_colors(256)
            .encode_rgb8(w, h, &px)
            .expect("palette encode");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode palette");
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            near(got[i * 4], px[i * 3], i, "pal r");
            near(got[i * 4 + 1], px[i * 3 + 1], i, "pal g");
            near(got[i * 4 + 2], px[i * 3 + 2], i, "pal b");
        }
    }

    #[test]
    fn multiplier_parts_exact_hits_libjxl_q() {
        assert_eq!(multiplier_parts_exact(197), (0, 196, 197));
        assert_eq!(multiplier_parts_exact(8), (0, 7, 8));
        let (log, bits, v) = multiplier_parts_exact(192);
        assert_eq!(v, 192);
        assert_eq!((bits + 1) << log, 192);
        // XYB Y finest at d=1: 0.25 * 1 * 4.8 * 163.84 = 196.608 → 197.
        let q = squeeze_quant(1.0, 255.0, 0, 0, true, true);
        assert_eq!(q, 197);
    }

    #[test]
    fn encode_vardct_16x16_rgb_decodes() {
        let w = 16u32;
        let h = 16u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = 48 + (x * 8) as u8;
                px[i + 1] = 48 + (y * 8) as u8;
                px[i + 2] = 96;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .modular(Some(false))
            .encode_rgb8(w, h, &px)
            .expect("vardct encode");
        assert_eq!(&jxl[..2], [0xFF, 0x0A]);
        let img = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&jxl))
            .unwrap_or_else(|e| panic!("open vardct: {e:#}"));
        let _ = img
            .render_frame(0)
            .unwrap_or_else(|e| panic!("render: {e:#}"));
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .unwrap_or_else(|e| panic!("decode vardct: {e}"));
        assert_eq!(frames[0].width, w);
        assert_eq!(frames[0].height, h);
        assert!(frames[0].rgba.iter().any(|&b| b != 0));
        // Lossy, but a d=1 8×8 VarDCT of a smooth gradient should stay recognisable.
        let got = &frames[0].rgba;
        let mut sad = 0u64;
        for i in 0..(w * h) as usize {
            sad += got[i * 4].abs_diff(px[i * 3]) as u64;
            sad += got[i * 4 + 1].abs_diff(px[i * 3 + 1]) as u64;
            sad += got[i * 4 + 2].abs_diff(px[i * 3 + 2]) as u64;
        }
        let mae = sad as f64 / (w * h * 3) as f64;
        assert!(mae < 8.0, "VarDCT MAE {mae} too high");
    }

    #[test]
    fn encode_vardct_dct16_keeps_axis() {
        // Smooth 16×16 horizontal ramp should merge to DCT16. If HF is
        // transposed, columns vary instead of rows.
        let w = 16u32;
        let h = 16u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = (x * 16) as u8;
                px[i] = v;
                px[i + 1] = v;
                px[i + 2] = v;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .modular(Some(false))
            .encode_rgb8(w, h, &px)
            .expect("vardct ramp");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode ramp");
        let got = &frames[0].rgba;
        let row0: Vec<u8> = (0..16).map(|x| got[x * 4]).collect();
        let col0: Vec<u8> = (0..16).map(|y| got[(y * w as usize) * 4]).collect();
        let row_span = row0.iter().copied().max().unwrap() - row0.iter().copied().min().unwrap();
        let col_span = col0.iter().copied().max().unwrap() - col0.iter().copied().min().unwrap();
        assert!(
            row_span > 80 && col_span < 40,
            "DCT16 transposed or lost: row0={row0:?} col0={col0:?}"
        );
    }

    #[test]
    fn encode_vardct_keeps_axis() {
        // Period-2 vertical bars must stay vertical (AC scatter used to transpose).
        let w = 32u32;
        let h = 32u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = if x % 2 == 0 { 255 } else { 0 };
                px[i] = v;
                px[i + 1] = v;
                px[i + 2] = v;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .modular(Some(false))
            .encode_rgb8(w, h, &px)
            .expect("vardct bars");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode bars");
        let got = &frames[0].rgba;
        let row0 = (0..8).map(|x| got[x * 4]).collect::<Vec<_>>();
        let col0 = (0..8)
            .map(|y| got[(y * w as usize) * 4])
            .collect::<Vec<_>>();
        let row_flip = row0.windows(2).filter(|a| a[0].abs_diff(a[1]) > 40).count();
        let col_flip = col0.windows(2).filter(|a| a[0].abs_diff(a[1]) > 40).count();
        assert!(
            row_flip >= 3 && col_flip <= 1,
            "bars transposed or lost: row0={row0:?} col0={col0:?}"
        );
    }

    #[test]
    fn encode_vardct_multi_group_decodes() {
        // group_dim = 256 → two pass groups, still one LF group.
        let w = 300u32;
        let h = 16u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x % 251) as u8;
                px[i + 1] = (y * 12) as u8;
                px[i + 2] = 90;
            }
        }
        let jxl = Encoder::new()
            .distance(1.0)
            .modular(Some(false))
            .encode_rgb8(w, h, &px)
            .expect("vardct multi-group");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .unwrap_or_else(|e| panic!("decode vardct multi-group: {e}"));
        assert_eq!(frames[0].width, w);
        assert_eq!(frames[0].height, h);
        assert!(frames[0].rgba.iter().any(|&b| b != 0));
    }

    #[test]
    fn encode_animation_two_frames() {
        let w = 8u32;
        let h = 8u32;
        let mut a = vec![0u8; (w * h * 3) as usize];
        let mut b = vec![0u8; (w * h * 3) as usize];
        for p in a.as_chunks_mut::<3>().0 {
            p[0] = 200;
            p[1] = 10;
            p[2] = 10;
        }
        for p in b.as_chunks_mut::<3>().0 {
            p[0] = 10;
            p[1] = 10;
            p[2] = 200;
        }
        let jxl = Encoder::new()
            .responsive(Some(false))
            .palette_colors(0)
            .encode_animation(w, h, PixelFormat::Rgb, 1000, 1, 0, &[(40, &a), (40, &b)])
            .expect("anim encode");
        let info = Decoder::probe_bytes(&jxl).expect("probe");
        assert_eq!(info.num_keyframes, 2);
        assert!(info.animation_tps.is_some());
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode anim");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].duration, 40);
        assert_eq!(frames[1].duration, 40);
        near(frames[0].rgba[0], 200, 0, "f0 r");
        near(frames[1].rgba[2], 200, 0, "f1 b");
    }

    #[test]
    fn encode_16bit_gray_roundtrip() {
        let w = 8u32;
        let h = 8u32;
        let mut px = vec![0u16; (w * h) as usize];
        for (i, p) in px.iter_mut().enumerate() {
            *p = (i as u16) * 1000;
        }
        let jxl = Encoder::new()
            .palette_colors(0)
            .responsive(Some(false))
            .encode_u16(w, h, PixelFormat::Gray, &px)
            .expect("16-bit encode");
        let info = Decoder::probe_bytes(&jxl).expect("probe");
        assert_eq!(info.bit_depth, 16);
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode 16-bit");
        assert_eq!(frames[0].width, w);
        let got = &frames[0].rgba;
        near(got[0], 0, 0, "16 gray 0");
    }

    #[test]
    fn encode_squeeze_multi_group() {
        // group_dim at effort 7 is 512; 513×17 is two groups, squeeze on.
        let w = 513u32;
        let h = 17u32;
        let mut px = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                px[i] = (x % 251) as u8;
                px[i + 1] = (y * 12) as u8;
                px[i + 2] = 40;
            }
        }
        let jxl = Encoder::new()
            .effort(7)
            .responsive(Some(true))
            .palette_colors(0)
            .encode_rgb8(w, h, &px)
            .expect("squeeze multi-group");
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&jxl)
            .expect("decode squeeze multi-group");
        assert_eq!(frames[0].width, w);
        assert_eq!(frames[0].height, h);
        let got = &frames[0].rgba;
        for i in 0..(w * h) as usize {
            near(got[i * 4], px[i * 3], i, "sqmg r");
            near(got[i * 4 + 1], px[i * 3 + 1], i, "sqmg g");
            near(got[i * 4 + 2], px[i * 3 + 2], i, "sqmg b");
        }
    }
}
