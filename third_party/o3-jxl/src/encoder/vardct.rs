//! VarDCT encode: 8×8 DCT, default dequant, ANS HF, cjxl e7 loop filter.

use super::ans::{
    AnsHist, collect_ans_tokens_ctx_len, collect_ans_tokens_ctx_len_dist, write_ans_histograms_len,
    write_ans_tokens_ex,
};
use super::bits::{BitWriter, U32, off, u, write_u32};
use super::entropy::{
    LZ77_MIN_LENGTH, PrefixCode, hybrid_uint_400, pack_signed, write_cluster_map, write_histograms,
    write_ma_tree_gradient, write_residual_run,
};
use super::header::{
    ImageParams, write_frame_header, write_image_header, write_modular_header, write_toc,
};
use super::predict::gradient_residuals;
use crate::error::{Error, Result};
use std::simd::prelude::*;
use std::sync::OnceLock;

use crate::simd::{
    GABORISH_INV_MUL_VARDCT, dct_rect, dct_rect_from_plane, dct4_2d, dct8_2d, dct8_fwd_from_plane,
    dct16_fwd_from_plane, dct32_fwd_from_plane, gaborish_inverse_plane, idct_rect, idct2_2d,
    idct4_2d, idct8_factored_2d, idct16_2d, idct32_2d, srgb8_rgb_to_xyb_padded,
};

/// VarDCT default `group_size_shift = 1` → 256-pixel groups.
const GROUP_DIM: u32 = 256;
/// LF group is 8 pass-groups on a side → 256 LF samples.
const LF_GROUP_DIM: u32 = GROUP_DIM;

/// JPEG XL default DCT8 dequant bands (X, Y, B).
const DCT8_BANDS: [&[f32]; 3] = [
    &[3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
    &[560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
    &[512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
];

/// Default DCT16 dequant bands (jxl-oxide / libjxl `Dct16`).
const DCT16_BANDS: [&[f32]; 3] = [
    &[
        8996.873,
        -1.3000778,
        -0.4942453,
        -0.43909377,
        -0.6350102,
        -0.9017726,
        -1.6162099,
    ],
    &[
        3191.4836,
        -0.67424583,
        -0.80745816,
        -0.4492584,
        -0.3586544,
        -0.3132239,
        -0.37615025,
    ],
    &[
        1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914,
    ],
];

/// libjxl `DCTResampleScales<16, 2>` for `DCFromLowestFrequencies(DCT16)`.
const DCT16_LF_SCALE: [f32; 2] = [1.0, 0.901_764_2];
/// `DCTResampleScales<32, 4>`.
const DCT32_LF_SCALE: [f32; 4] = [1.0, 0.974_886_8, 0.901_764_2, 0.787_054_9];

/// Default DCT32 bands.
const DCT32_BANDS: [&[f32]; 3] = [
    &[
        15718.408,
        -1.025,
        -0.98,
        -0.9012,
        -0.4,
        -0.48819396,
        -0.421064,
        -0.27,
    ],
    &[
        7305.7637,
        -0.8041958,
        -0.76330364,
        -0.5566038,
        -0.49785304,
        -0.43699592,
        -0.40180868,
        -0.27321684,
    ],
    &[
        3803.5317,
        -3.0607336,
        -2.041327,
        -2.023565,
        -0.54953897,
        -0.4,
        -0.4,
        -0.3,
    ],
];
/// Default DCT16×8 / DCT8×16 bands.
const DCT16X8_BANDS: [&[f32]; 3] = [
    &[7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
    &[1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
    &[506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
];

/// `AcStrategyType` raw values.
const ACS_DCT8: u8 = 0;
const ACS_DCT4: u8 = 3;
const ACS_DCT16: u8 = 4;
const ACS_DCT32: u8 = 5;
const ACS_DCT16X8: u8 = 6;
const ACS_DCT8X16: u8 = 7;
const ACS_DCT4X8: u8 = 12;
const ACS_DCT8X4: u8 = 13;
const ACS_OCCUPIED: u8 = 255;

const COLOR_FACTOR: f32 = 84.0;
const Y_TO_B: f32 = 1.0;

/// Per-64×64 chroma-from-luma maps (`ytox_map`, `ytob_map`).
struct CflMaps {
    x: Vec<i32>,
    b: Vec<i32>,
    tw: usize,
    th: usize,
}

impl CflMaps {
    fn zeros(bw: usize, bh: usize) -> Self {
        let tw = bw.div_ceil(8).max(1);
        let th = bh.div_ceil(8).max(1);
        Self {
            x: vec![0; tw * th],
            b: vec![0; tw * th],
            tw,
            th,
        }
    }

    fn factors(&self, bx: usize, by: usize) -> (f32, f32) {
        let tx = (bx / 8).min(self.tw.saturating_sub(1));
        let ty = (by / 8).min(self.th.saturating_sub(1));
        let i = ty * self.tw + tx;
        (
            self.x[i] as f32 / COLOR_FACTOR,
            Y_TO_B + self.b[i] as f32 / COLOR_FACTOR,
        )
    }
}

/// `HfBlockContext`: default 15 clusters, or libjxl `FindBestBlockEntropyModel`.
struct BlockCtxModel {
    map: Vec<u8>,
    n_clusters: u8,
    qf_thresholds: Vec<u32>,
}

impl BlockCtxModel {
    fn default_map() -> Self {
        Self {
            map: BLOCK_CTX_MAP.to_vec(),
            n_clusters: 15,
            qf_thresholds: Vec::new(),
        }
    }

    fn is_default(&self) -> bool {
        self.n_clusters == 15
            && self.qf_thresholds.is_empty()
            && self.map.as_slice() == BLOCK_CTX_MAP
    }
}

pub fn encode_vardct_rgb(
    width: u32,
    height: u32,
    pixels: &[u8],
    distance: f32,
    effort: u8,
) -> Result<Vec<u8>> {
    if pixels.len() != width as usize * height as usize * 3 {
        return Err(Error::Encode("VarDCT RGB buffer size mismatch".into()));
    }
    let (gab, epf_iters) = vardct_loop_filter(distance, effort);
    let params = ImageParams {
        width,
        height,
        channels: 3,
        group_size_shift: 1,
        bit_depth: 8,
        animation: None,
        duration: 0,
        is_last: true,
        xyb_encoded: true,
        vardct: true,
        gab,
        epf_iters,
    };

    let bw = width.div_ceil(8) as usize;
    let bh = height.div_ceil(8) as usize;
    let pw = bw * 8;
    let ph = bh * 8;
    let (mut x_pl, mut y_pl, mut b_pl) = srgb8_rgb_to_xyb_padded(width, height, pixels, pw, ph);
    // libjxl `InitialQuantField` runs on *pre-gaborish* XYB. Hare+ sets
    // `global_scale` from `0.39/d` (not the AQ-field median). Gab-off scales
    // the IQF distance by 0.62.
    let (global_scale, quant_lf, base_hf_mul) = if effort >= 5 {
        distance_to_quant_hare(distance)
    } else {
        distance_to_quant(distance, 0.79)
    };
    let (mut hf_muls, mask1x1) = if effort >= 5 {
        let iqf_d = if gab { distance } else { distance * 0.62 };
        let (qf, mask) = initial_quant_field(&x_pl, &y_pl, &b_pl, pw, ph, bw, bh, iqf_d);
        let inv_gs = 65536.0 / global_scale as f32;
        let hf = qf
            .into_iter()
            .map(|q| (q * inv_gs + 0.5).clamp(1.0, 256.0) as i32)
            .collect();
        (hf, mask)
    } else {
        (vec![base_hf_mul; bw * bh], vec![1.0f32; pw * ph])
    };
    if gab {
        x_pl = gaborish_inverse_plane(&x_pl, pw, ph, GABORISH_INV_MUL_VARDCT[0]);
        y_pl = gaborish_inverse_plane(&y_pl, pw, ph, GABORISH_INV_MUL_VARDCT[1]);
        b_pl = gaborish_inverse_plane(&b_pl, pw, ph, GABORISH_INV_MUL_VARDCT[2]);
    }
    let cfl = if effort >= 5 {
        compute_cfl_maps(&x_pl, &y_pl, &b_pl, pw, bw, bh)
    } else {
        CflMaps::zeros(bw, bh)
    };
    let matrices = dct8_matrices();
    let qm_scale = [0.8f32, 1.0, 1.0]; // x_qm=3, b_qm=2
    // ac_inv for hf_mul = 1; per-block scale is applied at quantize time.
    let gs = global_scale as f32;
    let lf_inv = [
        1.0 / ((1.0 / 32.0) * 512.0 / (gs * quant_lf as f32)),
        1.0 / ((1.0 / 4.0) * 512.0 / (gs * quant_lf as f32)),
        1.0 / ((1.0 / 2.0) * 512.0 / (gs * quant_lf as f32)),
    ];
    let zigzag = dct8_zigzag_ac();
    // Default opsin quant_bias (X, Y, B). Decoder: |q|<=1 → q*bias, else q-0.145/q.
    const QUANT_BIAS: [f32; 3] = [1.0 - 0.054_650_073, 1.0 - 0.070_054_5, 1.0 - 0.049_935_103];
    const QUANT_BIAS_NUM: f32 = 0.145;
    let mut ac_inv = [[0f32; 64]; 3];
    for c in 0..3 {
        let mul = (65536.0 / gs) * qm_scale[c];
        for i in 0..64 {
            let m = matrices[c][i] * mul;
            ac_inv[c][i] = if m.abs() < 1e-12 { 0.0 } else { 1.0 / m };
        }
    }

    let mut lf_q = [
        vec![0i32; bw * bh],
        vec![0i32; bw * bh],
        vec![0i32; bw * bh],
    ];
    let mut ac_q = [
        vec![0i32; bw * bh * 63],
        vec![0i32; bw * bh * 63],
        vec![0i32; bw * bh * 63],
    ];
    dct_quant_all(
        bw,
        bh,
        pw,
        [&x_pl, &y_pl, &b_pl],
        &zigzag,
        lf_inv,
        &ac_inv,
        &hf_muls,
        QUANT_BIAS,
        QUANT_BIAS_NUM,
        &cfl,
        &mut lf_q,
        &mut ac_q,
    );
    let mut acs = vec![ACS_DCT8; bw * bh];
    if effort >= 5 {
        try_dct4x8(
            bw,
            bh,
            pw,
            [&x_pl, &y_pl, &b_pl],
            lf_inv,
            &hf_muls,
            distance,
            QUANT_BIAS,
            QUANT_BIAS_NUM,
            gs,
            &mut lf_q,
            &mut ac_q,
            &mut acs,
            &mask1x1,
            &cfl,
        );
        try_dct4(
            bw,
            bh,
            pw,
            [&x_pl, &y_pl, &b_pl],
            lf_inv,
            &hf_muls,
            distance,
            QUANT_BIAS,
            QUANT_BIAS_NUM,
            gs,
            &mut lf_q,
            &mut ac_q,
            &mut acs,
            &mask1x1,
            &cfl,
        );
    }
    let mut ac16 = [Vec::new(), Vec::new(), Vec::new()];
    let mut ac16_at = vec![u32::MAX; bw * bh];
    let mut ac32 = [Vec::new(), Vec::new(), Vec::new()];
    let mut ac32_at = vec![u32::MAX; bw * bh];
    let mut ac168 = [Vec::new(), Vec::new(), Vec::new()];
    let mut ac168_at = vec![u32::MAX; bw * bh];
    if effort >= 5 && std::env::var_os("O3_NO_VARBLOCK").is_none() {
        try_merge_dct16(
            bw,
            bh,
            pw,
            [&x_pl, &y_pl, &b_pl],
            lf_inv,
            &mut hf_muls,
            distance,
            QUANT_BIAS,
            QUANT_BIAS_NUM,
            gs,
            &mut lf_q,
            &ac_q,
            &mut acs,
            &mut ac16,
            &mut ac16_at,
            &mask1x1,
            &cfl,
        );
        try_merge_rect(
            bw,
            bh,
            pw,
            [&x_pl, &y_pl, &b_pl],
            lf_inv,
            &mut hf_muls,
            distance,
            QUANT_BIAS,
            QUANT_BIAS_NUM,
            gs,
            &mut lf_q,
            &ac_q,
            &mut acs,
            &mut ac168,
            &mut ac168_at,
            &mask1x1,
            &cfl,
        );
        try_merge_dct32(
            bw,
            bh,
            pw,
            [&x_pl, &y_pl, &b_pl],
            lf_inv,
            &mut hf_muls,
            distance,
            QUANT_BIAS,
            QUANT_BIAS_NUM,
            gs,
            &mut lf_q,
            &ac_q,
            &mut acs,
            &mut ac32,
            &mut ac32_at,
            &mask1x1,
            &cfl,
        );
    }
    if std::env::var_os("O3_ACS_STATS").is_some() {
        let mut n8 = 0u32;
        let mut n4 = 0u32;
        let mut n16 = 0u32;
        let mut n32 = 0u32;
        let mut nrect = 0u32;
        let mut n48 = 0u32;
        let mut nocc = 0u32;
        for &k in &acs {
            match k {
                ACS_DCT8 => n8 += 1,
                ACS_DCT4 => n4 += 1,
                ACS_DCT16 => n16 += 1,
                ACS_DCT32 => n32 += 1,
                ACS_DCT16X8 | ACS_DCT8X16 => nrect += 1,
                ACS_DCT4X8 | ACS_DCT8X4 => n48 += 1,
                ACS_OCCUPIED => nocc += 1,
                _ => {}
            }
        }
        eprintln!(
            "acs 8={n8} 4={n4} 48={n48} 16={n16} 32={n32} rect={nrect} occ={nocc} n={}",
            acs.len()
        );
    }
    let bcm = if effort >= 5 {
        find_best_block_entropy_model(bw, bh, &acs, &hf_muls, distance)
    } else {
        BlockCtxModel::default_map()
    };
    let coeff_orders = if effort >= 4 {
        compute_coeff_orders(
            bw, bh, &acs, &ac_q, &ac16, &ac16_at, &ac32, &ac32_at, &ac168, &ac168_at,
        )
    } else {
        CoeffOrders::default()
    };
    let (hf_hists, hf_map, hf_assign) = hist_from_hf_mixed(
        width,
        height,
        bw,
        &ac_q,
        &acs,
        &ac16,
        &ac16_at,
        &ac32,
        &ac32_at,
        &ac168,
        &ac168_at,
        &coeff_orders,
        &hf_muls,
        &bcm,
    );

    let mut lf_vals = Vec::new();
    for plane in &lf_q {
        let mut res = vec![0u32; bw * bh];
        gradient_residuals(bw, bh, plane, 1, &mut res);
        lf_vals.extend_from_slice(&res);
    }
    {
        let first: Vec<usize> = (0..bw * bh).filter(|&i| acs[i] != ACS_OCCUPIED).collect();
        let n = first.len();
        let mut info = vec![0i32; n * 2];
        for (k, &i) in first.iter().enumerate() {
            info[k] = i32::from(acs[i]);
            info[n + k] = hf_muls[i] - 1;
        }
        let mut res = vec![0u32; info.len()];
        gradient_residuals(n, 2, &info, 1, &mut res);
        lf_vals.extend_from_slice(&res);
    }
    lf_vals.push(pack_signed(4)); // EPF sharpness constant
    {
        let mut res = vec![0u32; cfl.x.len()];
        gradient_residuals(cfl.tw, cfl.th, &cfl.x, 1, &mut res);
        lf_vals.extend_from_slice(&res);
        gradient_residuals(cfl.tw, cfl.th, &cfl.b, 1, &mut res);
        lf_vals.extend_from_slice(&res);
    }
    let lf_code = code_from_values(&lf_vals);

    let num_groups = width.div_ceil(GROUP_DIM) * height.div_ceil(GROUP_DIM);
    let num_lf_groups = width.div_ceil(GROUP_DIM * 8) * height.div_ceil(GROUP_DIM * 8);
    let single = num_groups == 1;

    let mut sections = Vec::new();
    if single {
        // One TOC entry: LfGlobal + LfGroup + HfGlobal + PassGroup as a
        // *continuous* bitstream. Byte-padding between parts is a decode error
        // (the next modular header is then parsed as ANS).
        let mut w = BitWriter::with_capacity((width * height) as usize + 2048);
        encode_lf_global_vardct(&mut w, global_scale, quant_lf, &lf_code, &bcm);
        encode_lf_group_vardct(
            &mut w, width, height, bw, 0, &lf_q, &hf_muls, &acs, &lf_code, &cfl,
        );
        encode_hf_global(&mut w, &hf_hists, &hf_map, &coeff_orders, None, num_groups);
        encode_pass_group_vardct(
            &mut w,
            width,
            height,
            bw,
            0,
            &ac_q,
            &acs,
            &ac16,
            &ac16_at,
            &ac32,
            &ac32_at,
            &ac168,
            &ac168_at,
            &hf_hists,
            &hf_assign,
            &coeff_orders,
            None,
            &hf_muls,
            &bcm,
        );
        sections.push(w.finish());
    } else {
        let mut w = BitWriter::with_capacity(1024);
        encode_lf_global_vardct(&mut w, global_scale, quant_lf, &lf_code, &bcm);
        sections.push(w.finish());
        for g in 0..num_lf_groups {
            let mut w = BitWriter::with_capacity(4096);
            encode_lf_group_vardct(
                &mut w, width, height, bw, g, &lf_q, &hf_muls, &acs, &lf_code, &cfl,
            );
            sections.push(w.finish());
        }
        let mut w = BitWriter::with_capacity(1024);
        encode_hf_global(&mut w, &hf_hists, &hf_map, &coeff_orders, None, num_groups);
        sections.push(w.finish());
        let blocks_per_group = ((GROUP_DIM / 8) * (GROUP_DIM / 8)) as usize;
        for g in 0..num_groups {
            let mut w = BitWriter::with_capacity(blocks_per_group * 96 + 256);
            encode_pass_group_vardct(
                &mut w,
                width,
                height,
                bw,
                g,
                &ac_q,
                &acs,
                &ac16,
                &ac16_at,
                &ac32,
                &ac32_at,
                &ac168,
                &ac168_at,
                &hf_hists,
                &hf_assign,
                &coeff_orders,
                None,
                &hf_muls,
                &bcm,
            );
            sections.push(w.finish());
        }
    }

    let sizes: Vec<u32> = sections.iter().map(|s| s.len() as u32).collect();
    let mut out = BitWriter::with_capacity(64 + sizes.iter().map(|&s| s as usize).sum::<usize>());
    write_image_header(&mut out, params);
    write_frame_header(&mut out, params);
    write_toc(&mut out, &sizes);
    for s in &sections {
        out.append_bytes(s);
    }
    Ok(out.finish())
}

/// cjxl: gab on for VarDCT at Hare+ when d > 0.5; EPF iters from `{0.7, 1.5, 4.0}`.
fn vardct_loop_filter(distance: f32, effort: u8) -> (bool, u8) {
    let gab = effort >= 5 && distance > 0.5;
    let mut epf = 0u8;
    if distance >= 0.7 {
        epf += 1;
    }
    if distance >= 1.5 {
        epf += 1;
    }
    if distance >= 4.0 {
        epf += 1;
    }
    (gab, epf)
}

/// libjxl `ComputeGlobalScaleAndQuant` at Hare+: median = `0.39/d`, mad = 0.
fn distance_to_quant_hare(d: f32) -> (u32, u32, i32) {
    let d = d.clamp(0.01, 25.0);
    compute_global_scale_and_quant(initial_quant_dc(d), 0.39 / d, 0.0)
}

fn compute_global_scale_and_quant(quant_dc: f32, quant_median: f32, mad: f32) -> (u32, u32, i32) {
    const DENOM: f32 = 65536.0;
    const TARGET: f32 = 5.0;
    const NUMER: f32 = 4096.0;
    let mut scale = DENOM * (quant_median - mad) / TARGET;
    scale = scale.clamp(1.0, 32_768.0);
    let mut gs = scale as i32;
    let scaled_dc = (quant_dc * NUMER * 1.6) as i32;
    if gs > scaled_dc {
        gs = scaled_dc.max(1);
    }
    let global_scale = gs as u32;
    let inv_gs = DENOM / global_scale as f32;
    let quant_lf = (quant_dc * inv_gs + 0.5).min(65536.0).clamp(1.0, 65_535.0) as u32;
    let hf_mul = (quant_median * inv_gs + 0.5).clamp(1.0, 256.0) as i32;
    (global_scale, quant_lf, hf_mul)
}

fn find_best_cfl_mul(ys: &[f32], chroma: &[f32], base: f32) -> i32 {
    let n = ys.len() as f32;
    if n < 1.0 {
        return 0;
    }
    const DIST: f32 = 1e-9;
    const TZ: f32 = 2.6;
    let mut ca = 0.0f32;
    let mut cb = 0.0f32;
    for (&y, &ch) in ys.iter().zip(chroma) {
        let a = y / COLOR_FACTOR;
        let b = base * y - ch;
        ca += a * a;
        cb += a * b;
    }
    let mut x = -cb / (ca + n * DIST * 0.5).max(1e-12);
    if x >= TZ {
        x -= TZ;
    } else if x <= -TZ {
        x += TZ;
    } else {
        x = 0.0;
    }
    x.round().clamp(-128.0, 127.0) as i32
}

fn compute_cfl_maps(
    x_pl: &[f32],
    y_pl: &[f32],
    b_pl: &[f32],
    pw: usize,
    bw: usize,
    bh: usize,
) -> CflMaps {
    let tw = bw.div_ceil(8).max(1);
    let th = bh.div_ceil(8).max(1);
    let mut mx = vec![0i32; tw * th];
    let mut mb = vec![0i32; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let mut ys = Vec::new();
            let mut xs = Vec::new();
            let mut bs = Vec::new();
            let bx1 = ((tx + 1) * 8).min(bw);
            let by1 = ((ty + 1) * 8).min(bh);
            for by in ty * 8..by1 {
                for bx in tx * 8..bx1 {
                    let cy = dct8_from_plane(y_pl, pw, bx * 8, by * 8);
                    let cx = dct8_from_plane(x_pl, pw, bx * 8, by * 8);
                    let cb = dct8_from_plane(b_pl, pw, bx * 8, by * 8);
                    ys.extend_from_slice(&cy[1..]);
                    xs.extend_from_slice(&cx[1..]);
                    bs.extend_from_slice(&cb[1..]);
                }
            }
            let i = ty * tw + tx;
            mx[i] = find_best_cfl_mul(&ys, &xs, 0.0);
            mb[i] = find_best_cfl_mul(&ys, &bs, Y_TO_B);
        }
    }
    CflMaps {
        x: mx,
        b: mb,
        tw,
        th,
    }
}

/// Tile CFL on HF only. LLF (top-left `lf`×`lf` in `stride`) uses kb=1, kx=0.
#[allow(clippy::needless_range_loop)]
fn apply_cfl_triple_lf<const N: usize>(
    planes: &mut [[f32; N]; 3],
    kx: f32,
    kb: f32,
    lf: usize,
    stride: usize,
) {
    for i in 0..N {
        let yi = planes[1][i];
        let y = i / stride;
        let x = i % stride;
        if y < lf && x < lf {
            planes[2][i] -= yi;
        } else {
            planes[0][i] -= kx * yi;
            planes[2][i] -= kb * yi;
        }
    }
}

fn covered_blocks(acs: u8) -> usize {
    match acs {
        ACS_DCT16 => 4,
        ACS_DCT32 => 16,
        ACS_DCT16X8 | ACS_DCT8X16 => 2,
        ACS_OCCUPIED => 0,
        _ => 1,
    }
}

/// libjxl `FindBestBlockEntropyModel` (Squirrel / e7).
#[allow(clippy::needless_range_loop)]
fn find_best_block_entropy_model(
    bw: usize,
    bh: usize,
    acs: &[u8],
    hf_muls: &[i32],
    distance: f32,
) -> BlockCtxModel {
    let tot = bw * bh;
    let size_for_ctx = ((1 << 10) as f32 * distance).max(1.0) as usize;
    if tot < size_for_ctx {
        return BlockCtxModel::default_map();
    }
    let mut qf_counts = [0usize; 256];
    let mut qf_ord_counts = [[0usize; 256]; 13];
    let mut ord_counts = [0usize; 13];
    for i in 0..tot {
        let a = acs[i];
        if a == ACS_OCCUPIED {
            continue;
        }
        let ord = order_id(a);
        let n = covered_blocks(a);
        let qf = (hf_muls[i] - 1).clamp(0, 255) as usize;
        qf_counts[qf] += n;
        qf_ord_counts[ord][qf] += n;
        ord_counts[ord] += n;
    }
    let size_for_qf = ((1 << 13) as f32 * distance).max(1.0) as usize;
    let num_qf_segments = if tot < size_for_qf { 1 } else { 2 };
    let mut qft = Vec::new();
    let mut cumsum = 0usize;
    let mut next = 1usize;
    let mut last_cut = 256usize;
    let mut cut = tot * next / num_qf_segments;
    for j in 0..256 {
        cumsum += qf_counts[j];
        if cumsum > cut {
            if j != 0 {
                qft.push(j as u32);
            }
            last_cut = j;
            while cumsum > cut {
                next += 1;
                cut = tot * next / num_qf_segments;
            }
        } else if next > qft.len() + 1 && j > 0 && j - 1 == last_cut {
            qft.push(j as u32);
        }
    }
    let nseg = qft.len() + 1;
    let mut counts = vec![0usize; 13 * nseg];
    let mut qft_pos = 0usize;
    for j in 0..256 {
        if qft_pos < qft.len() && j == qft[qft_pos] as usize {
            qft_pos += 1;
        }
        for i in 0..13 {
            counts[qft_pos + i * nseg] += qf_ord_counts[i][j];
        }
    }
    let mut remap: Vec<u8> = (0..(13 * nseg) as u8).collect();
    let mut clusters = remap.clone();
    let nb_clusters = (tot / size_for_ctx / 2).clamp(2, 9);
    let nb_clusters_chroma = (tot / size_for_ctx / 3).clamp(1, 5);
    while clusters.len() > nb_clusters {
        clusters.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));
        let last = clusters.pop().unwrap();
        let keep = *clusters.last().unwrap();
        counts[keep as usize] += counts[last as usize];
        counts[last as usize] = 0;
        remap[last as usize] = keep;
    }
    for i in 0..remap.len() {
        let mut r = remap[i];
        while remap[r as usize] != r {
            r = remap[r as usize];
        }
        remap[i] = r;
    }
    let mut remap_remap = vec![u8::MAX; remap.len()];
    let mut num = 0u8;
    for i in 0..remap.len() {
        let r = remap[i] as usize;
        if remap_remap[r] == u8::MAX {
            remap_remap[r] = num;
            num += 1;
        }
        remap[i] = remap_remap[r];
    }
    let mut ctx_map = remap.clone();
    ctx_map.resize(remap.len() * 3, 0);
    let chroma_max = (nb_clusters_chroma as i32 - 1).max(0) as u8;
    for i in remap.len()..ctx_map.len() {
        ctx_map[i] = num + remap[i % remap.len()].min(chroma_max);
    }
    let n_clusters = ctx_map.iter().copied().max().unwrap_or(0) + 1;
    BlockCtxModel {
        map: ctx_map,
        n_clusters,
        qf_thresholds: qft,
    }
}

fn encode_hf_block_context(w: &mut BitWriter, bcm: &BlockCtxModel) {
    if bcm.is_default() {
        w.put_bool(true);
        return;
    }
    w.put_bool(false);
    for _ in 0..3 {
        w.put(4, 0);
    }
    w.put(4, bcm.qf_thresholds.len() as u64);
    for &t in &bcm.qf_thresholds {
        write_u32(
            w,
            t.saturating_sub(1),
            u(2),
            off(4, 3),
            off(12, 5),
            off(44, 8),
        );
    }
    write_cluster_map(w, &bcm.map, u32::from(bcm.n_clusters));
}

/// libjxl `RatioOfDerivativesOfCubicRootToSimpleGamma`.
#[allow(clippy::excessive_precision)]
fn ratio_gamma(v: f32, invert: bool) -> f32 {
    const SG_MUL: f32 = 226.772_161_535_089_14;
    const SG_MUL2: f32 = 1.0 / 73.377_132_366_608_82;
    const LOG2: f32 = std::f32::consts::LN_2;
    const SG_RET_MUL: f32 = SG_MUL2 * 18.658_093_213_5 * LOG2;
    const SG_V_OFFSET: f32 = 7.782_599_167_989_459;
    const EPS: f32 = 1e-2;
    let v = v.max(0.0);
    let v2 = v * v;
    let num = SG_RET_MUL * 3.0 * SG_MUL * v2 + EPS;
    let den = LOG2 * SG_MUL * v * v2 + (SG_V_OFFSET * LOG2 + EPS);
    if invert { num / den } else { den / num }
}

#[allow(clippy::excessive_precision)]
fn masking_sqrt(v: f32) -> f32 {
    const LOG_OFFSET: f32 = 27.505_837_037_000_106;
    const MUL: f32 = 211.665_679_735_036_78;
    0.25 * (v * (MUL * 1e8).sqrt() + LOG_OFFSET).sqrt()
}

/// libjxl `ComputeMask` on the 8×8 AQ map (modulates the exponent).
#[allow(clippy::excessive_precision)]
fn compute_mask(out_val: f32) -> f32 {
    const BASE: f32 = -0.7647;
    const MUL0: f32 = 0.800_617_628_627_417_6;
    const MUL2: f32 = 17.350_365_616_318_63;
    const MUL3: f32 = 6.794_325_051_737_649;
    const MUL4: f32 = 9.470_873_562_437_895;
    const OFFSET2: f32 = 302.595_878_155_797_27;
    const OFFSET3: f32 = 3.717_963_562_614_077;
    const OFFSET4: f32 = 0.25 * OFFSET3;
    let v1 = (out_val * MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + OFFSET2);
    let v3 = 1.0 / (v1 * v1 + OFFSET3);
    let v4 = 1.0 / (v1 * v1 + OFFSET4);
    BASE + MUL4 * v4 + MUL2 * v2 + MUL3 * v3
}

fn hf_modulation(y: &[f32], pw: usize, x: usize, y0: usize, ph: usize) -> f32 {
    const VALMIN: f32 = 0.0206;
    const MUL: f32 = -0.38;
    const OFFSET: f32 = 0.42;
    let mut sum = 0.0f32;
    for dy in 0..8 {
        let yy = (y0 + dy).min(ph.saturating_sub(1));
        // libjxl: last row of the 8×8 compares against itself (vertical diff 0).
        let ynext = if dy == 7 {
            yy
        } else {
            (y0 + dy + 1).min(ph.saturating_sub(1))
        };
        let row = yy * pw;
        let next = ynext * pw;
        for dx in 0..8 {
            let p = y[row + x + dx];
            if dx < 7 {
                sum += VALMIN.min((p - y[row + x + dx + 1]).abs());
            }
            sum += VALMIN.min((p - y[next + x + dx]).abs());
        }
    }
    sum * MUL + OFFSET
}

#[allow(clippy::excessive_precision)]
fn gamma_modulation(x_pl: &[f32], y_pl: &[f32], pw: usize, x: usize, y0: usize) -> f32 {
    const BIAS: f32 = 0.16;
    const GAMMA: f32 = 0.100_561_333_719_269_7;
    let mut overall = 0.0f32;
    for dy in 0..8 {
        let row = (y0 + dy) * pw + x;
        for dx in 0..8 {
            let iny = y_pl[row + dx] + BIAS;
            let inx = x_pl[row + dx];
            overall += ratio_gamma(iny - inx, true);
            overall += ratio_gamma(iny + inx, true);
        }
    }
    GAMMA * (overall * (0.5 / 64.0)).max(1e-12).log2()
}

#[allow(clippy::excessive_precision)]
fn blue_modulation(
    x_pl: &[f32],
    y_pl: &[f32],
    b_pl: &[f32],
    pw: usize,
    x: usize,
    y0: usize,
) -> f32 {
    const LIMIT: f32 = 0.027_121_074_570_634_722;
    const OFFSET: f32 = 0.084_381_641_171_960_5;
    const MUL: f32 = 0.142_070_003_584_391_59;
    const MAX_LIMIT: f32 = 15.398_788_439_047_934;
    let mut sum = 0.0f32;
    for dy in 0..8 {
        let row = (y0 + dy) * pw + x;
        for dx in 0..8 {
            let p_x = x_pl[row + dx];
            let p_b = b_pl[row + dx];
            let p_y = y_pl[row + dx] + OFFSET + p_x.abs();
            if p_b > p_y {
                sum += (p_b - p_y).min(LIMIT);
            }
        }
    }
    if sum >= 32.0 * LIMIT {
        sum = 64.0 * LIMIT - sum;
    }
    if sum >= MAX_LIMIT * LIMIT {
        sum = MAX_LIMIT * LIMIT;
    }
    sum * MUL
}

fn store_min4(v: f32, min: &mut [f32; 4]) {
    if v >= min[3] {
        return;
    }
    if v < min[0] {
        min[3] = min[2];
        min[2] = min[1];
        min[1] = min[0];
        min[0] = v;
    } else if v < min[1] {
        min[3] = min[2];
        min[2] = min[1];
        min[1] = v;
    } else if v < min[2] {
        min[3] = min[2];
        min[2] = v;
    } else {
        min[3] = v;
    }
}

/// libjxl `FuzzyErosion`: 3×3 four-smallest weighted sum, 2× downsample by summing.
#[allow(clippy::excessive_precision)]
fn fuzzy_erosion(from: &[f32], fw: usize, fh: usize, distance: f32) -> Vec<f32> {
    let tw = fw / 2;
    let th = fh / 2;
    let mut to = vec![0.0f32; tw * th];
    let mul = if distance < 2.0 {
        (2.0 - distance) * 0.5
    } else {
        0.0
    };
    let mut k0 = 0.125;
    let mut k1 = 0.10 + mul * -0.10;
    let mut k2 = 0.09 + mul * -0.09;
    let mut k3 = 0.06 + mul * -0.06;
    const TOTAL: f32 = 0.299_597_057_840_549_57;
    let norm = TOTAL / (k0 + k1 + k2 + k3);
    k0 *= norm;
    k1 *= norm;
    k2 *= norm;
    k3 *= norm;
    for fy in 0..fh {
        let y = fy;
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(fh - 1);
        for fx in 0..fw {
            let x = fx;
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(fw - 1);
            let mut m = [
                from[y * fw + x],
                from[y * fw + xm1],
                from[y * fw + xp1],
                from[ym1 * fw + xm1],
            ];
            m.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            store_min4(from[ym1 * fw + x], &mut m);
            store_min4(from[ym1 * fw + xp1], &mut m);
            store_min4(from[yp1 * fw + xm1], &mut m);
            store_min4(from[yp1 * fw + x], &mut m);
            store_min4(from[yp1 * fw + xp1], &mut m);
            let v = k0 * m[0] + k1 * m[1] + k2 * m[2] + k3 * m[3];
            let o = (fy / 2) * tw + fx / 2;
            if fx % 2 == 0 && fy % 2 == 0 {
                to[o] = v;
            } else {
                to[o] += v;
            }
        }
    }
    to
}

/// libjxl `AdaptiveQuantizationMap` / `InitialQuantField` (scalar, full image).
/// Returns `(quant_field, mask1x1)`.
#[allow(clippy::too_many_arguments)]
fn initial_quant_field(
    x_pl: &[f32],
    y_pl: &[f32],
    b_pl: &[f32],
    pw: usize,
    ph: usize,
    bw: usize,
    bh: usize,
    distance: f32,
) -> (Vec<f32>, Vec<f32>) {
    const AC_QUANT: f32 = 0.725;
    const MATCH_GAMMA: f32 = 0.019;
    const LIMIT: f32 = 0.2;
    let scale = AC_QUANT / distance.max(0.01);
    let mut pre = vec![0.0f32; (pw / 4) * (ph / 4)];
    let mut row = vec![0.0f32; pw];
    for y in 0..ph {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(ph - 1);
        let row_y = &y_pl[y * pw..][..pw];
        let row_t = &y_pl[y1 * pw..][..pw];
        let row_b = &y_pl[y2 * pw..][..pw];
        for x in 0..pw {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(pw - 1);
            let base = 0.25 * (row_b[x] + row_t[x] + row_y[x1] + row_y[x2]);
            let gammac = ratio_gamma(row_y[x] + MATCH_GAMMA, false);
            let mut diff = gammac * (row_y[x] - base);
            diff = (diff * diff).min(LIMIT);
            let v = masking_sqrt(diff);
            if y % 4 == 0 {
                row[x] = v;
            } else {
                row[x] += v;
            }
        }
        if y % 4 == 3 {
            let out_y = y / 4;
            for x in 0..pw / 4 {
                pre[out_y * (pw / 4) + x] =
                    (row[x * 4] + row[x * 4 + 1] + row[x * 4 + 2] + row[x * 4 + 3]) * 0.25;
            }
        }
    }
    let mut aq = fuzzy_erosion(&pre, pw / 4, ph / 4, distance);
    let mut dampen = 1.0f32;
    if distance >= 2.0 {
        dampen = (1.0 - (distance - 2.0) / 12.0).max(0.0);
    }
    let mul = scale * dampen;
    let add = (1.0 - dampen) * 0.48 * scale;
    for by in 0..bh {
        let y = by * 8;
        for bx in 0..bw {
            let x = bx * 8;
            let i = by * bw + bx;
            let mut out_val = compute_mask(aq[i]);
            out_val += hf_modulation(y_pl, pw, x, y, ph);
            out_val += gamma_modulation(x_pl, y_pl, pw, x, y);
            out_val += blue_modulation(x_pl, y_pl, b_pl, pw, x, y);
            aq[i] = out_val.exp() * mul + add;
        }
    }
    // libjxl `mask1x1`: 1 / (log1p(|γ·laplacian|) + 0.01), used by EstimateEntropy.
    let mut mask = vec![1.0f32; pw * ph];
    for y in 0..ph {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(ph - 1);
        let row = &y_pl[y * pw..][..pw];
        let row_t = &y_pl[y1 * pw..][..pw];
        let row_b = &y_pl[y2 * pw..][..pw];
        for x in 0..pw {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(pw - 1);
            let base = 0.25 * (row_b[x] + row_t[x] + row[x1] + row[x2]);
            let gammac = ratio_gamma(row[x] + MATCH_GAMMA, false);
            let diff = (gammac * (row[x] - base)).abs().ln_1p();
            mask[y * pw + x] = 1.0 / (diff + 0.01);
        }
    }
    (aq, mask)
}

/// libjxl-style quant: `quant_ac = 0.725 / d`, `global_scale = 65536 * quant_ac / 5`,
/// per-block `hf_mul` from the quant field, `quant_lf` from `InitialQuantDC`.
#[allow(clippy::too_many_arguments)]
fn dct_quant_all(
    bw: usize,
    bh: usize,
    pw: usize,
    planes: [&[f32]; 3],
    zigzag: &[u8; 63],
    lf_inv: [f32; 3],
    ac_inv: &[[f32; 64]; 3],
    hf_muls: &[i32],
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    cfl: &CflMaps,
    lf_q: &mut [Vec<i32>; 3],
    ac_q: &mut [Vec<i32>; 3],
) {
    let row_work = |by: usize, lf_rows: [&mut [i32]; 3], ac_rows: [&mut [i32]; 3]| {
        let y0 = by * 8;
        for bx in 0..bw {
            let x0 = bx * 8;
            let hf = hf_muls[by * bw + bx] as f32;
            let mut coefs = [
                dct8_fwd_from_plane(planes[0], pw, x0, y0),
                dct8_fwd_from_plane(planes[1], pw, x0, y0),
                dct8_fwd_from_plane(planes[2], pw, x0, y0),
            ];
            let (kx, kb) = cfl.factors(bx, by);
            let kx_v = Simd::<f32, 8>::splat(kx);
            let kb_v = Simd::<f32, 8>::splat(kb);
            // LF uses global correlation (kx=0, kb=1). Tile CFL is HF-only.
            {
                let y = coefs[1][0].to_array();
                let mut x = coefs[0][0].to_array();
                let mut b = coefs[2][0].to_array();
                b[0] -= y[0];
                for k in 1..8 {
                    x[k] -= kx * y[k];
                    b[k] -= kb * y[k];
                }
                coefs[0][0] = Simd::from_array(x);
                coefs[2][0] = Simd::from_array(b);
            }
            #[allow(clippy::needless_range_loop)]
            for row in 1..8 {
                let y = coefs[1][row];
                coefs[0][row] -= y * kx_v;
                coefs[2][row] -= y * kb_v;
            }
            for c in 0..3 {
                let dc = coefs[c][0].to_array()[0];
                lf_rows[c][bx] = (dc * lf_inv[c]).round() as i32;
                let mut spatial = [0i32; 64];
                let hf_v = Simd::<f32, 8>::splat(hf);
                for row in 0..8 {
                    let t =
                        coefs[c][row] * Simd::<f32, 8>::from_slice(&ac_inv[c][row * 8..]) * hf_v;
                    let ta = t.to_array();
                    for k in 0..8 {
                        spatial[row * 8 + k] =
                            quantize_with_bias(ta[k], quant_bias[c], quant_bias_num);
                    }
                }
                let ac = &mut ac_rows[c][bx * 63..bx * 63 + 63];
                for (k, &idx) in zigzag.iter().enumerate() {
                    ac[k] = spatial[idx as usize];
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let _ = bh;
        let [lf0, lf1, lf2] = lf_q;
        let [ac0, ac1, ac2] = ac_q;
        lf0.par_chunks_mut(bw)
            .zip(lf1.par_chunks_mut(bw))
            .zip(lf2.par_chunks_mut(bw))
            .zip(ac0.par_chunks_mut(bw * 63))
            .zip(ac1.par_chunks_mut(bw * 63))
            .zip(ac2.par_chunks_mut(bw * 63))
            .enumerate()
            .for_each(|(by, (((((l0, l1), l2), a0), a1), a2))| {
                row_work(by, [l0, l1, l2], [a0, a1, a2]);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let [lf0, lf1, lf2] = lf_q;
        let [ac0, ac1, ac2] = ac_q;
        for by in 0..bh {
            let s = by * bw;
            let e = s + bw;
            let as_ = by * bw * 63;
            let ae = as_ + bw * 63;
            row_work(
                by,
                [&mut lf0[s..e], &mut lf1[s..e], &mut lf2[s..e]],
                [&mut ac0[as_..ae], &mut ac1[as_..ae], &mut ac2[as_..ae]],
            );
        }
    }
}

/// Invert JPEG XL HF quant bias, then round to int.
fn quantize_with_bias(t: f32, qb: f32, qbn: f32) -> i32 {
    if !t.is_finite() {
        return 0;
    }
    if t.abs() < 0.5 * qb {
        return 0;
    }
    if t.abs() < 1.5 * qb {
        return if t >= 0.0 { 1 } else { -1 };
    }
    (t + qbn / t).round() as i32
}

fn distance_to_quant(d: f32, ac_quant: f32) -> (u32, u32, i32) {
    let d = d.clamp(0.01, 25.0);
    let quant_ac = ac_quant / d;
    let global_scale = (65536.0 * quant_ac / 5.0).round().clamp(1.0, 32_768.0) as u32;
    let inv_gs = 65536.0 / global_scale as f32;
    let quant_dc = initial_quant_dc(d);
    let quant_lf = (quant_dc * inv_gs + 0.5).round().clamp(1.0, 65_535.0) as u32;
    let hf_mul = (quant_ac * inv_gs + 0.5).round().clamp(1.0, 256.0) as i32;
    (global_scale, quant_lf, hf_mul)
}

/// libjxl `InitialQuantDC`: `kDcQuant / max(0.5 d, min(d, 0.3 (d/0.3)^0.83))`.
fn initial_quant_dc(d: f32) -> f32 {
    const DC_MUL: f32 = 0.3;
    const DC_POW: f32 = 0.83;
    #[allow(clippy::excessive_precision)]
    const DC_QUANT: f32 = 1.095_924_047_623_553;
    let dc_target = (0.5 * d).max((DC_MUL * (d / DC_MUL).powf(DC_POW)).min(d));
    (DC_QUANT / dc_target).min(50.0)
}

fn dct8_matrices() -> &'static [[f32; 64]; 3] {
    static M: OnceLock<[[f32; 64]; 3]> = OnceLock::new();
    M.get_or_init(|| [dct8_matrix(0), dct8_matrix(1), dct8_matrix(2)])
}

fn dct8_matrix(ch: usize) -> [f32; 64] {
    // Default Dct encoding stores interpolated *weights*; the decoder applies
    // the reciprocal as the dequant matrix (`need_recip`).
    let mut m = dct_quant_weights(DCT8_BANDS[ch], 8, 8);
    for x in &mut m {
        *x = 1.0 / *x;
    }
    m
}

fn dct_quant_weights(params: &[f32], width: u32, height: u32) -> [f32; 64] {
    fn mult(x: f32) -> f32 {
        if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
    }
    let mut bands = Vec::with_capacity(params.len());
    let mut last = params[0];
    bands.push(last);
    for &val in &params[1..] {
        last *= mult(val);
        bands.push(last);
    }
    let mut ret = [1.0f32; 64];
    let max = std::f32::consts::SQRT_2 + 1e-6;
    let len = bands.len();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            let pos = (dx * dx + dy * dy).sqrt() * (len - 1) as f32 / max;
            let idx = (pos as usize).min(len - 2);
            let frac = pos - idx as f32;
            let a = bands[idx];
            let b = bands[idx + 1];
            ret[y * width as usize + x] = a * (b / a).powf(frac.clamp(0.0, 1.0));
        }
    }
    ret
}

/// libjxl `enc_ac_strategy.cc`: merge 2×2 DCT8 into DCT16 when cheaper.
#[allow(clippy::too_many_arguments)]
fn try_merge_dct16(
    bw: usize,
    bh: usize,
    pw: usize,
    planes: [&[f32]; 3],
    lf_inv: [f32; 3],
    hf_muls: &mut [i32],
    distance: f32,
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    gs: f32,
    lf_q: &mut [Vec<i32>; 3],
    ac_q: &[Vec<i32>; 3],
    acs: &mut [u8],
    ac16: &mut [Vec<i32>; 3],
    ac16_at: &mut [u32],
    mask: &[f32],
    cfl: &CflMaps,
) {
    let order = natural_order_16();
    let mat = dct16_matrices();
    let mat8 = dct8_matrices();
    let zigzag = dct8_zigzag_ac();
    let qm_scale = [0.8f32, 1.0, 1.0];
    // C: mul8x8 = k8x8mul2 + k8x8mul1 / (d + k8x8base), entropy_mul16X16 = 1.35
    let mul8 = 1.0 - 0.4 / (distance + 1.4);
    const MUL16: f32 = 1.35;
    let (cost_delta, zeros_mul, loss_mul) = entropy_scales(distance);
    for by in (0..bh.saturating_sub(1)).step_by(2) {
        for bx in (0..bw.saturating_sub(1)).step_by(2) {
            let i00 = by * bw + bx;
            let mut max_abs_y = 0i32;
            let mut all8 = true;
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (by + dy) * bw + (bx + dx);
                    if acs[i] != ACS_DCT8 {
                        all8 = false;
                    }
                    for &v in &ac_q[1][i * 63..i * 63 + 63] {
                        max_abs_y = max_abs_y.max(v.unsigned_abs() as i32);
                    }
                }
            }
            if !all8 {
                continue;
            }
            // Nyquist-ish 8×8s (period-2 bars): DCT16 + EPF washes them even when the
            // encoder-domain residual looks cheap.
            if max_abs_y > 64 {
                continue;
            }
            let mut cost8 = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (by + dy) * bw + (bx + dx);
                    cost8 += estimate_8x8(
                        planes,
                        pw,
                        bx + dx,
                        by + dy,
                        i,
                        lf_q,
                        ac_q,
                        hf_muls[i],
                        gs,
                        mat8,
                        qm_scale,
                        &zigzag,
                        lf_inv,
                        cost_delta,
                        zeros_mul,
                        loss_mul,
                        mask,
                        cfl,
                    );
                }
            }
            cost8 *= mul8;
            // libjxl `AdjustQuantField` at d≤1.54: max of the covered 8×8s.
            let hf_i = hf_muls[i00]
                .max(hf_muls[i00 + 1])
                .max(hf_muls[i00 + bw])
                .max(hf_muls[i00 + bw + 1]);
            let hf = hf_i as f32;
            let mut coefs = [
                dct16_fwd_from_plane(planes[0], pw, bx * 8, by * 8),
                dct16_fwd_from_plane(planes[1], pw, bx * 8, by * 8),
                dct16_fwd_from_plane(planes[2], pw, bx * 8, by * 8),
            ];
            let (kx, kb) = cfl.factors(bx, by);
            apply_cfl_triple_lf(&mut coefs, kx, kb, 2, 16);
            let mut hf16 = [vec![0i32; 252], vec![0i32; 252], vec![0i32; 252]];
            let mut spatial = coefs;
            for c in 0..3 {
                // Match 8×8: t = orig * hf / (mat * 65536/gs * qm). Decoder rec is
                // q * mat * 65536 / (gs * hf) * qm; multiplying hf on both sides.
                let mul = (65536.0 / gs) * qm_scale[c];
                for (k, &(dx, dy)) in order[4..].iter().enumerate() {
                    let idx = dx as usize * 16 + dy as usize;
                    let m = mat[c][idx] * mul;
                    let t = if m.abs() < 1e-12 {
                        0.0
                    } else {
                        coefs[c][idx] / m * hf
                    };
                    hf16[c][k] = quantize_with_bias(t, quant_bias[c], quant_bias_num);
                    spatial[c][idx] = coefs[c][idx] - hf16[c][k] as f32 * m / hf.max(1e-6);
                }
                spatial[c][0] = 0.0;
                spatial[c][1] = 0.0;
                spatial[c][16] = 0.0;
                spatial[c][17] = 0.0;
                idct16_2d(&mut spatial[c]);
                apply_spatial_mask(&mut spatial[c], mask, pw, bx * 8, by * 8, 16, 16);
            }
            let cost16 = hf_cost(&hf16[0], cost_delta, zeros_mul)
                + hf_cost(&hf16[1], cost_delta, zeros_mul)
                + hf_cost(&hf16[2], cost_delta, zeros_mul);
            let qn = quant_norm_field(
                &[
                    hf_muls[i00],
                    hf_muls[i00 + 1],
                    hf_muls[i00 + bw],
                    hf_muls[i00 + bw + 1],
                ],
                gs,
            );
            let loss16 = eighth_loss(
                [&spatial[0][..], &spatial[1][..], &spatial[2][..]],
                256,
                qn,
                loss_mul,
            );
            if cost16 * MUL16 + loss16 >= cost8 {
                continue;
            }
            let slot = (ac16[0].len() / 252) as u32;
            for c in 0..3 {
                ac16[c].extend_from_slice(&hf16[c]);
                let dc = dc_from_lowest_16(&coefs[c]);
                lf_q[c][i00] = (dc[0] * lf_inv[c]).round() as i32;
                lf_q[c][i00 + 1] = (dc[1] * lf_inv[c]).round() as i32;
                lf_q[c][i00 + bw] = (dc[2] * lf_inv[c]).round() as i32;
                lf_q[c][i00 + bw + 1] = (dc[3] * lf_inv[c]).round() as i32;
            }
            acs[i00] = ACS_DCT16;
            acs[i00 + 1] = ACS_OCCUPIED;
            acs[i00 + bw] = ACS_OCCUPIED;
            acs[i00 + bw + 1] = ACS_OCCUPIED;
            hf_muls[i00] = hf_i;
            ac16_at[i00] = slot;
        }
    }
}

/// libjxl `entropy_mul16X8 = 1.25`: DCT16×8 (vertical pair) / DCT8×16 (horizontal).
#[allow(clippy::too_many_arguments)]
fn try_merge_rect(
    bw: usize,
    bh: usize,
    pw: usize,
    planes: [&[f32]; 3],
    lf_inv: [f32; 3],
    hf_muls: &mut [i32],
    distance: f32,
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    gs: f32,
    lf_q: &mut [Vec<i32>; 3],
    ac_q: &[Vec<i32>; 3],
    acs: &mut [u8],
    ac168: &mut [Vec<i32>; 3],
    ac168_at: &mut [u32],
    mask: &[f32],
    cfl: &CflMaps,
) {
    let order = natural_order(16, 8);
    let mat = dct16x8_matrices();
    let qm_scale = [0.8f32, 1.0, 1.0];
    let mul8 = 1.0 - 0.4 / (distance + 1.4);
    const MUL: f32 = 1.25;
    // Vertical pairs → DCT16×8 (8×16 px).
    for by in (0..bh.saturating_sub(1)).step_by(2) {
        for bx in 0..bw {
            let i0 = by * bw + bx;
            if acs[i0] != ACS_DCT8 || acs[i0 + bw] != ACS_DCT8 {
                continue;
            }
            let (cost8, max_abs) = pair_cost8(ac_q, bw, bx, by, true);
            if max_abs > 64 {
                continue;
            }
            let hf_i = hf_muls[i0].max(hf_muls[i0 + bw]);
            let hf = hf_i as f32;
            if !try_one_rect(
                planes,
                pw,
                bx,
                by,
                8,
                16,
                ACS_DCT16X8,
                true,
                &order,
                mat,
                hf,
                gs,
                qm_scale,
                quant_bias,
                quant_bias_num,
                lf_inv,
                cost8,
                mul8,
                MUL,
                distance,
                lf_q,
                acs,
                ac168,
                ac168_at,
                bw,
                i0,
                &[i0 + bw],
                mask,
                cfl,
            ) {
                continue;
            }
            hf_muls[i0] = hf_i;
        }
    }
    // Horizontal pairs → DCT8×16 (16×8 px).
    for by in 0..bh {
        for bx in (0..bw.saturating_sub(1)).step_by(2) {
            let i0 = by * bw + bx;
            if acs[i0] != ACS_DCT8 || acs[i0 + 1] != ACS_DCT8 {
                continue;
            }
            let (cost8, max_abs) = pair_cost8(ac_q, bw, bx, by, false);
            if max_abs > 64 {
                continue;
            }
            let hf_i = hf_muls[i0].max(hf_muls[i0 + 1]);
            let hf = hf_i as f32;
            if try_one_rect(
                planes,
                pw,
                bx,
                by,
                16,
                8,
                ACS_DCT8X16,
                false,
                &order,
                mat,
                hf,
                gs,
                qm_scale,
                quant_bias,
                quant_bias_num,
                lf_inv,
                cost8,
                mul8,
                MUL,
                distance,
                lf_q,
                acs,
                ac168,
                ac168_at,
                bw,
                i0,
                &[i0 + 1],
                mask,
                cfl,
            ) {
                hf_muls[i0] = hf_i;
            }
        }
    }
}

fn pair_cost8(ac_q: &[Vec<i32>; 3], bw: usize, bx: usize, by: usize, vertical: bool) -> (f32, i32) {
    let cells = if vertical {
        [by * bw + bx, (by + 1) * bw + bx]
    } else {
        [by * bw + bx, by * bw + bx + 1]
    };
    let mut cost = 0.0f32;
    let mut max_abs = 0i32;
    for i in cells {
        for &v in &ac_q[1][i * 63..i * 63 + 63] {
            max_abs = max_abs.max(v.unsigned_abs() as i32);
        }
        cost += ac_q
            .iter()
            .map(|p| hf_cost(&p[i * 63..i * 63 + 63], 10.83, 9.31))
            .sum::<f32>();
    }
    (cost, max_abs)
}

#[allow(clippy::too_many_arguments)]
fn try_one_rect(
    planes: [&[f32]; 3],
    pw: usize,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    kind: u8,
    transpose: bool,
    order: &[(u16, u16)],
    mat: &[[f32; 128]; 3],
    hf: f32,
    gs: f32,
    qm_scale: [f32; 3],
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    lf_inv: [f32; 3],
    cost8: f32,
    mul8: f32,
    mul_rect: f32,
    distance: f32,
    lf_q: &mut [Vec<i32>; 3],
    acs: &mut [u8],
    ac168: &mut [Vec<i32>; 3],
    ac168_at: &mut [u32],
    bw: usize,
    i0: usize,
    extra: &[usize],
    mask: &[f32],
    cfl: &CflMaps,
) -> bool {
    let mut coefs = [
        dct_rect_from_plane(planes[0], pw, bx * 8, by * 8, w, h),
        dct_rect_from_plane(planes[1], pw, bx * 8, by * 8, w, h),
        dct_rect_from_plane(planes[2], pw, bx * 8, by * 8, w, h),
    ];
    let (kx, kb) = cfl.factors(bx, by);
    {
        let lf = if h == 16 && w == 8 {
            [0usize, w]
        } else {
            [0, 1]
        };
        let [x, y, b] = &mut coefs;
        let n = x.len().min(y.len()).min(b.len());
        for i in 0..n {
            let yi = y[i];
            if i == lf[0] || i == lf[1] {
                b[i] -= yi;
            } else {
                x[i] -= kx * yi;
                b[i] -= kb * yi;
            }
        }
    }
    let mut hf16 = [vec![0i32; 126], vec![0i32; 126], vec![0i32; 126]];
    let mut spatial = [coefs[0].clone(), coefs[1].clone(), coefs[2].clone()];
    for c in 0..3 {
        let mul = (65536.0 / gs) * qm_scale[c];
        for (k, &(dx, dy)) in order[2..].iter().enumerate() {
            let (col, row) = if transpose {
                (dy as usize, dx as usize)
            } else {
                (dx as usize, dy as usize)
            };
            let idx = row * w + col;
            // Dequant is filled as height=8, width=16 → index dy*16+dx.
            let midx = (dy as usize * 16 + dx as usize).min(127);
            let m = mat[c][midx] * mul;
            let t = if idx >= coefs[c].len() || m.abs() < 1e-12 {
                0.0
            } else {
                coefs[c][idx] / m * hf
            };
            hf16[c][k] = quantize_with_bias(t, quant_bias[c], quant_bias_num);
            if idx < spatial[c].len() {
                spatial[c][idx] = coefs[c][idx] - hf16[c][k] as f32 * m / hf.max(1e-6);
            }
        }
        // LF pair lives in the first two natural-order slots.
        if h == 16 && w == 8 {
            spatial[c][0] = 0.0;
            if spatial[c].len() > w {
                spatial[c][w] = 0.0;
            }
        } else {
            spatial[c][0] = 0.0;
            if spatial[c].len() > 1 {
                spatial[c][1] = 0.0;
            }
        }
        idct_rect(&mut spatial[c], w, h);
        apply_spatial_mask(&mut spatial[c], mask, pw, bx * 8, by * 8, w, h);
    }
    let (cost_delta, zeros_mul, loss_mul) = entropy_scales(distance);
    let cost = hf_cost(&hf16[0], cost_delta, zeros_mul)
        + hf_cost(&hf16[1], cost_delta, zeros_mul)
        + hf_cost(&hf16[2], cost_delta, zeros_mul);
    let qn = (hf * gs / 65536.0).max(1e-6);
    let loss = eighth_loss(
        [&spatial[0][..], &spatial[1][..], &spatial[2][..]],
        w * h,
        qn,
        loss_mul,
    );
    if cost * mul_rect + loss >= cost8 * mul8 {
        return false;
    }
    let slot = (ac168[0].len() / 126) as u32;
    for c in 0..3 {
        ac168[c].extend_from_slice(&hf16[c]);
        let dc = dc_from_lowest_rect(&coefs[c], w, h);
        lf_q[c][i0] = (dc.0 * lf_inv[c]).round() as i32;
        if extra.len() == 1 {
            lf_q[c][extra[0]] = (dc.1 * lf_inv[c]).round() as i32;
        }
    }
    acs[i0] = kind;
    for &e in extra {
        acs[e] = ACS_OCCUPIED;
    }
    ac168_at[i0] = slot;
    let _ = bw;
    true
}

fn dc_from_lowest_rect(coeff: &[f32], w: usize, h: usize) -> (f32, f32) {
    // DCT16×8: 2×1 LF (vertical). DCT8×16: 1×2 LF (horizontal).
    let s1 = DCT16_LF_SCALE[1];
    if h == 16 && w == 8 {
        let a = coeff[0];
        let b = coeff[w] * s1;
        (a + b, a - b)
    } else {
        let a = coeff[0];
        let b = coeff[1] * s1;
        (a + b, a - b)
    }
}

/// libjxl `entropy_mul32X32 = 1.5`.
#[allow(clippy::too_many_arguments)]
fn try_merge_dct32(
    bw: usize,
    bh: usize,
    pw: usize,
    planes: [&[f32]; 3],
    lf_inv: [f32; 3],
    hf_muls: &mut [i32],
    distance: f32,
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    gs: f32,
    lf_q: &mut [Vec<i32>; 3],
    ac_q: &[Vec<i32>; 3],
    acs: &mut [u8],
    ac32: &mut [Vec<i32>; 3],
    ac32_at: &mut [u32],
    mask: &[f32],
    cfl: &CflMaps,
) {
    let order = natural_order(32, 32);
    let mat = dct32_matrices();
    let mat8 = dct8_matrices();
    let zigzag = dct8_zigzag_ac();
    let qm_scale = [0.8f32, 1.0, 1.0];
    let mul8 = 1.0 - 0.4 / (distance + 1.4);
    const MUL32: f32 = 1.5;
    let (cost_delta, zeros_mul, loss_mul) = entropy_scales(distance);
    for by in (0..bh.saturating_sub(3)).step_by(4) {
        for bx in (0..bw.saturating_sub(3)).step_by(4) {
            let mut ok = true;
            let mut max_abs = 0i32;
            for dy in 0..4 {
                for dx in 0..4 {
                    let i = (by + dy) * bw + (bx + dx);
                    if acs[i] != ACS_DCT8 {
                        ok = false;
                    }
                    for &v in &ac_q[1][i * 63..i * 63 + 63] {
                        max_abs = max_abs.max(v.unsigned_abs() as i32);
                    }
                }
            }
            if !ok || max_abs > 64 {
                continue;
            }
            let mut cost8 = 0.0f32;
            for dy in 0..4 {
                for dx in 0..4 {
                    let i = (by + dy) * bw + (bx + dx);
                    cost8 += estimate_8x8(
                        planes,
                        pw,
                        bx + dx,
                        by + dy,
                        i,
                        lf_q,
                        ac_q,
                        hf_muls[i],
                        gs,
                        mat8,
                        qm_scale,
                        &zigzag,
                        lf_inv,
                        cost_delta,
                        zeros_mul,
                        loss_mul,
                        mask,
                        cfl,
                    );
                }
            }
            cost8 *= mul8;
            let i00 = by * bw + bx;
            let mut hf_i = 1i32;
            for dy in 0..4 {
                for dx in 0..4 {
                    hf_i = hf_i.max(hf_muls[(by + dy) * bw + (bx + dx)]);
                }
            }
            let hf = hf_i as f32;
            let mut coefs = [
                dct32_fwd_from_plane(planes[0], pw, bx * 8, by * 8),
                dct32_fwd_from_plane(planes[1], pw, bx * 8, by * 8),
                dct32_fwd_from_plane(planes[2], pw, bx * 8, by * 8),
            ];
            let (kx, kb) = cfl.factors(bx, by);
            apply_cfl_triple_lf(&mut coefs, kx, kb, 4, 32);
            let mut hf32 = [vec![0i32; 1008], vec![0i32; 1008], vec![0i32; 1008]];
            let mut spatial = coefs;
            for c in 0..3 {
                let mul = (65536.0 / gs) * qm_scale[c];
                for (k, &(dx, dy)) in order[16..].iter().enumerate() {
                    let idx = dx as usize * 32 + dy as usize;
                    let m = mat[c][idx] * mul;
                    let t = if m.abs() < 1e-12 {
                        0.0
                    } else {
                        coefs[c][idx] / m * hf
                    };
                    hf32[c][k] = quantize_with_bias(t, quant_bias[c], quant_bias_num);
                    spatial[c][idx] = coefs[c][idx] - hf32[c][k] as f32 * m / hf.max(1e-6);
                }
                for y in 0..4 {
                    for x in 0..4 {
                        spatial[c][y * 32 + x] = 0.0;
                    }
                }
                idct32_2d(&mut spatial[c]);
                apply_spatial_mask(&mut spatial[c], mask, pw, bx * 8, by * 8, 32, 32);
            }
            let cost32 = hf_cost(&hf32[0], cost_delta, zeros_mul)
                + hf_cost(&hf32[1], cost_delta, zeros_mul)
                + hf_cost(&hf32[2], cost_delta, zeros_mul);
            let mut qf = [0i32; 16];
            for dy in 0..4 {
                for dx in 0..4 {
                    qf[dy * 4 + dx] = hf_muls[(by + dy) * bw + (bx + dx)];
                }
            }
            let loss32 = eighth_loss(
                [&spatial[0][..], &spatial[1][..], &spatial[2][..]],
                1024,
                quant_norm_field(&qf, gs),
                loss_mul,
            );
            if cost32 * MUL32 + loss32 >= cost8 {
                continue;
            }
            let slot = (ac32[0].len() / 1008) as u32;
            for c in 0..3 {
                ac32[c].extend_from_slice(&hf32[c]);
                let dc = dc_from_lowest_32(&coefs[c]);
                for dy in 0..4 {
                    for dx in 0..4 {
                        lf_q[c][(by + dy) * bw + (bx + dx)] =
                            (dc[dy * 4 + dx] * lf_inv[c]).round() as i32;
                    }
                }
            }
            acs[i00] = ACS_DCT32;
            for dy in 0..4 {
                for dx in 0..4 {
                    if dy != 0 || dx != 0 {
                        acs[(by + dy) * bw + (bx + dx)] = ACS_OCCUPIED;
                    }
                }
            }
            hf_muls[i00] = hf_i;
            ac32_at[i00] = slot;
        }
    }
}

fn entropy_scales(distance: f32) -> (f32, f32, f32) {
    let ratio = (distance + 0.137) / 1.137;
    (
        10.833_273 * ratio.powf(0.367),
        9.308_906 * ratio.powf(0.510),
        1.2 * ratio.powf(0.337),
    )
}

fn ceil_log2_nz(x: u32) -> u32 {
    if x <= 1 { 0 } else { (x - 1).ilog2() + 1 }
}

/// libjxl EstimateEntropy token cost: `cost_delta * Σ√q + zeros_mul * nbits`.
fn hf_cost(v: &[i32], cost_delta: f32, zeros_mul: f32) -> f32 {
    let mut s = 0.0f32;
    let mut nz = 0u32;
    for &x in v {
        if x != 0 {
            nz += 1;
            s += (x.unsigned_abs() as f32).sqrt();
        }
    }
    let nbits = ceil_log2_nz(nz + 1) + 1;
    s * cost_delta + zeros_mul * (ceil_log2_nz(nbits + 17) + nbits) as f32
}

fn apply_spatial_mask(
    spatial: &mut [f32],
    mask: &[f32],
    stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let mrow = (y0 + y) * stride + x0;
        let srow = y * w;
        for x in 0..w {
            spatial[srow + x] *= mask[mrow + x];
        }
    }
}

/// libjxl `kChannelMul` X/Y/B = 10.2, 1, 1.03 applied inside the 8th-norm.
fn eighth_loss(res: [&[f32]; 3], n_pixels: usize, quant: f32, info_loss_mul: f32) -> f32 {
    const CH: [f32; 3] = [10.2, 1.0, 1.03];
    let mut s = 0.0f32;
    for c in 0..3 {
        let w = CH[c];
        for &e in res[c] {
            let a = (e * w).abs();
            let a2 = a * a;
            s += a2 * a2 * a2 * a2;
        }
    }
    let mean = s / (n_pixels as f32).max(1.0);
    let eighth = mean.max(0.0).powf(0.125);
    info_loss_mul * eighth * n_pixels as f32 / quant.max(1e-6)
}

fn quant_norm_field(hf_muls: &[i32], gs: f32) -> f32 {
    let scale = gs / 65536.0;
    match hf_muls {
        [] => 1e-6,
        [q] => (*q as f32 * scale).max(1e-6),
        [a, b] => ((*a).max(*b) as f32 * scale).max(1e-6),
        qs => {
            let mut s = 0.0f32;
            for &q in qs {
                let v = q as f32 * scale;
                let v2 = v * v;
                let v4 = v2 * v2;
                let v8 = v4 * v4;
                s += v8 * v8;
            }
            (s / qs.len() as f32).powf(1.0 / 16.0).max(1e-6)
        }
    }
}

fn dct8_from_plane(plane: &[f32], stride: usize, x: usize, y0: usize) -> [f32; 64] {
    let mut b = [0f32; 64];
    for row in 0..8 {
        b[row * 8..row * 8 + 8].copy_from_slice(&plane[(y0 + row) * stride + x..][..8]);
    }
    dct8_2d(&mut b);
    b
}

/// Token cost of the already-quantized 8×8 plus spatial 8th-norm of its residual.
#[allow(clippy::too_many_arguments)]
fn estimate_8x8(
    planes: [&[f32]; 3],
    pw: usize,
    bx: usize,
    by: usize,
    i: usize,
    lf_q: &[Vec<i32>; 3],
    ac_q: &[Vec<i32>; 3],
    hf_mul: i32,
    gs: f32,
    matrices: &[[f32; 64]; 3],
    qm_scale: [f32; 3],
    zigzag: &[u8; 63],
    lf_inv: [f32; 3],
    cost_delta: f32,
    zeros_mul: f32,
    loss_mul: f32,
    mask: &[f32],
    cfl: &CflMaps,
) -> f32 {
    let mut orig = [
        dct8_from_plane(planes[0], pw, bx * 8, by * 8),
        dct8_from_plane(planes[1], pw, bx * 8, by * 8),
        dct8_from_plane(planes[2], pw, bx * 8, by * 8),
    ];
    let (kx, kb) = cfl.factors(bx, by);
    apply_cfl_triple_lf(&mut orig, kx, kb, 1, 8);
    let mut qs = Vec::with_capacity(3 * 64);
    let mut spatial = orig;
    let hf = (hf_mul as f32).max(1e-6);
    for c in 0..3 {
        // Decoder: rec = q * mat * 65536 / (gs * hf) * qm. Do not fold hf into `m`.
        let mul = (65536.0 / gs) * qm_scale[c];
        let dc_rec = lf_q[c][i] as f32 / lf_inv[c];
        spatial[c][0] = orig[c][0] - dc_rec;
        qs.push(lf_q[c][i]);
        for (k, &z) in zigzag.iter().enumerate() {
            let m = matrices[c][z as usize] * mul;
            let q = ac_q[c][i * 63 + k];
            spatial[c][z as usize] = orig[c][z as usize] - q as f32 * m / hf;
            qs.push(q);
        }
        idct8_factored_2d(&mut spatial[c]);
        apply_spatial_mask(&mut spatial[c], mask, pw, bx * 8, by * 8, 8, 8);
    }
    let token = hf_cost(&qs, cost_delta, zeros_mul);
    token
        + eighth_loss(
            [&spatial[0][..], &spatial[1][..], &spatial[2][..]],
            64,
            quant_norm_field(&[hf_mul], gs),
            loss_mul,
        )
}

fn transpose4x4(b: &mut [f32; 16]) {
    for y in 0..4 {
        for x in (y + 1)..4 {
            b.swap(y * 4 + x, x * 4 + y);
        }
    }
}

/// libjxl `TransformFromPixels(DCT4)`: four 4×4 DCTs + 2×2 DCT of the DCs.
///
/// Each 4×4 is stored transposed vs [`dct4_2d`]'s row-major `(fy, fx)`: jxl-oxide
/// `transform_dct4` gathers `scratch.get_mut(iy, ix) = coeff.get(qx+ix*2, qy+iy*2)`,
/// which is a transpose relative to libjxl `block[iy*4+ix]`.
fn dct4_8x8_fwd(plane: &[f32], stride: usize, x0: usize, y0: usize) -> [f32; 64] {
    let mut coeff = [0f32; 64];
    for qy in 0..2 {
        for qx in 0..2 {
            let mut b = [0f32; 16];
            for iy in 0..4 {
                let src = (y0 + qy * 4 + iy) * stride + x0 + qx * 4;
                b[iy * 4..iy * 4 + 4].copy_from_slice(&plane[src..src + 4]);
            }
            dct4_2d(&mut b);
            transpose4x4(&mut b);
            for iy in 0..4 {
                for ix in 0..4 {
                    coeff[(qy + iy * 2) * 8 + (qx + ix * 2)] = b[iy * 4 + ix];
                }
            }
        }
    }
    // Decoder `aux_idct2` is unnormalized Hadamard H (H²=4I). Encode with H/4.
    let (c00, c01, c10, c11) = (coeff[0], coeff[1], coeff[8], coeff[9]);
    coeff[0] = (c00 + c01 + c10 + c11) * 0.25;
    coeff[1] = (c00 + c01 - c10 - c11) * 0.25;
    coeff[8] = (c00 - c01 + c10 - c11) * 0.25;
    coeff[9] = (c00 - c01 - c10 + c11) * 0.25;
    coeff
}

fn dct4_8x8_inv(coeff: &mut [f32; 64]) {
    let (c00, c01, c10, c11) = (coeff[0], coeff[1], coeff[8], coeff[9]);
    coeff[0] = c00 + c01 + c10 + c11;
    coeff[1] = c00 + c01 - c10 - c11;
    coeff[8] = c00 - c01 + c10 - c11;
    coeff[9] = c00 - c01 - c10 + c11;
    let mut scratch = [0f32; 64];
    for qy in 0..2 {
        for qx in 0..2 {
            let mut b = [0f32; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    b[iy * 4 + ix] = coeff[(qy + iy * 2) * 8 + (qx + ix * 2)];
                }
            }
            transpose4x4(&mut b);
            idct4_2d(&mut b);
            for iy in 0..4 {
                for ix in 0..4 {
                    scratch[(qy * 4 + iy) * 8 + qx * 4 + ix] = b[iy * 4 + ix];
                }
            }
        }
    }
    *coeff = scratch;
}

fn dct4_matrices() -> &'static [[f32; 64]; 3] {
    static M: OnceLock<[[f32; 64]; 3]> = OnceLock::new();
    M.get_or_init(|| [dct4_matrix(0), dct4_matrix(1), dct4_matrix(2)])
}

fn dct4_matrix(ch: usize) -> [f32; 64] {
    const BANDS: [&[f32]; 3] = [
        &[2200.0, 0.0, 0.0, 0.0],
        &[392.0, 0.0, 0.0, 0.0],
        &[112.0, -0.25, -0.25, -0.5],
    ];
    let mat4 = dct_quant_weights_n(BANDS[ch], 4, 4);
    let mut ret = [1.0f32; 64];
    for y in 0..4 {
        for x in 0..4 {
            let v = mat4[y * 4 + x];
            ret[y * 16 + x * 2] = v;
            ret[y * 16 + x * 2 + 1] = v;
            ret[(y * 2 + 1) * 8 + x * 2] = v;
            ret[(y * 2 + 1) * 8 + x * 2 + 1] = v;
        }
    }
    for x in &mut ret {
        *x = 1.0 / x.max(1e-12);
    }
    ret
}

/// libjxl `FindBest8x8Transform`: DCT4×4 vs DCT8 (entropy_mul 1.08/0.8).
#[allow(clippy::too_many_arguments)]
fn try_dct4(
    bw: usize,
    bh: usize,
    pw: usize,
    planes: [&[f32]; 3],
    lf_inv: [f32; 3],
    hf_muls: &[i32],
    distance: f32,
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    gs: f32,
    lf_q: &mut [Vec<i32>; 3],
    ac_q: &mut [Vec<i32>; 3],
    acs: &mut [u8],
    mask: &[f32],
    cfl: &CflMaps,
) {
    let mat4 = dct4_matrices();
    let mat8 = dct8_matrices();
    let zigzag8 = dct8_zigzag_ac();
    let zigzag4 = dct4_zigzag_ac();
    let qm_scale = [0.8f32, 1.0, 1.0];
    const MUL4: f32 = 1.08 / 0.8;
    let (cost_delta, zeros_mul, loss_mul) = entropy_scales(distance);
    for by in 0..bh {
        for bx in 0..bw {
            let i = by * bw + bx;
            if acs[i] != ACS_DCT8 {
                continue;
            }
            let cost8 = estimate_8x8(
                planes, pw, bx, by, i, lf_q, ac_q, hf_muls[i], gs, mat8, qm_scale, &zigzag8,
                lf_inv, cost_delta, zeros_mul, loss_mul, mask, cfl,
            );
            let hf = hf_muls[i] as f32;
            let (kx, kb) = cfl.factors(bx, by);
            let mut orig = [
                dct4_8x8_fwd(planes[0], pw, bx * 8, by * 8),
                dct4_8x8_fwd(planes[1], pw, bx * 8, by * 8),
                dct4_8x8_fwd(planes[2], pw, bx * 8, by * 8),
            ];
            apply_cfl_triple_lf(&mut orig, kx, kb, 1, 8);
            let mut qs = Vec::with_capacity(3 * 64);
            let mut spatial = orig;
            let mut ac_store = [[0i32; 63]; 3];
            let mut lf_store = [0i32; 3];
            for c in 0..3 {
                let mul = (65536.0 / gs) * qm_scale[c];
                lf_store[c] = (orig[c][0] * lf_inv[c]).round() as i32;
                let dc_rec = lf_store[c] as f32 / lf_inv[c];
                spatial[c][0] = orig[c][0] - dc_rec;
                qs.push(lf_store[c]);
                for (k, &z) in zigzag4.iter().enumerate() {
                    let m = mat4[c][z as usize] * mul;
                    let t = if m.abs() < 1e-12 {
                        0.0
                    } else {
                        orig[c][z as usize] / m * hf
                    };
                    let q = quantize_with_bias(t, quant_bias[c], quant_bias_num);
                    ac_store[c][k] = q;
                    spatial[c][z as usize] = orig[c][z as usize] - q as f32 * m / hf.max(1e-6);
                    qs.push(q);
                }
                dct4_8x8_inv(&mut spatial[c]);
                apply_spatial_mask(&mut spatial[c], mask, pw, bx * 8, by * 8, 8, 8);
            }
            let token = hf_cost(&qs, cost_delta, zeros_mul);
            let loss = eighth_loss(
                [&spatial[0][..], &spatial[1][..], &spatial[2][..]],
                64,
                quant_norm_field(&[hf_muls[i]], gs),
                loss_mul,
            );
            // C EstimateEntropy: `entropy *= entropy_mul` then `+= loss`.
            if token * MUL4 + loss >= cost8 {
                continue;
            }
            acs[i] = ACS_DCT4;
            for c in 0..3 {
                lf_q[c][i] = lf_store[c];
                ac_q[c][i * 63..i * 63 + 63].copy_from_slice(&ac_store[c]);
            }
        }
    }
}

fn transpose8x8(b: &mut [f32; 64]) {
    for y in 0..8 {
        for x in (y + 1)..8 {
            b.swap(y * 8 + x, x * 8 + y);
        }
    }
}

/// libjxl `TransformFromPixels(DCT4X8)`: two 4×8 DCTs + 1-D mix of the DCs.
fn dct4x8_8x8_fwd(plane: &[f32], stride: usize, x0: usize, y0: usize) -> [f32; 64] {
    let mut coeff = [0f32; 64];
    for qy in 0..2 {
        let mut b = [0f32; 32];
        for iy in 0..4 {
            let src = (y0 + qy * 4 + iy) * stride + x0;
            b[iy * 8..iy * 8 + 8].copy_from_slice(&plane[src..src + 8]);
        }
        dct_rect(&mut b, 8, 4);
        for iy in 0..4 {
            for ix in 0..8 {
                coeff[(qy + iy * 2) * 8 + ix] = b[iy * 8 + ix];
            }
        }
    }
    let (c0, c1) = (coeff[0], coeff[8]);
    coeff[0] = (c0 + c1) * 0.5;
    coeff[8] = (c0 - c1) * 0.5;
    coeff
}

fn dct4x8_8x8_inv(coeff: &mut [f32; 64]) {
    let (c0, c1) = (coeff[0], coeff[8]);
    coeff[0] = c0 + c1;
    coeff[8] = c0 - c1;
    let mut scratch = [0f32; 64];
    for qy in 0..2 {
        let mut b = [0f32; 32];
        for iy in 0..4 {
            for ix in 0..8 {
                b[iy * 8 + ix] = coeff[(qy + iy * 2) * 8 + ix];
            }
        }
        idct_rect(&mut b, 8, 4);
        for iy in 0..4 {
            scratch[(qy * 4 + iy) * 8..(qy * 4 + iy) * 8 + 8]
                .copy_from_slice(&b[iy * 8..iy * 8 + 8]);
        }
    }
    *coeff = scratch;
}

fn dct8x4_8x8_fwd(plane: &[f32], stride: usize, x0: usize, y0: usize) -> [f32; 64] {
    let mut pix = [0f32; 64];
    for row in 0..8 {
        pix[row * 8..row * 8 + 8].copy_from_slice(&plane[(y0 + row) * stride + x0..][..8]);
    }
    transpose8x8(&mut pix);
    // DCT4×8 of the transposed 8×8 (libjxl DCT8×4).
    let mut coeff = [0f32; 64];
    for qy in 0..2 {
        let mut b = [0f32; 32];
        for iy in 0..4 {
            b[iy * 8..iy * 8 + 8].copy_from_slice(&pix[(qy * 4 + iy) * 8..][..8]);
        }
        dct_rect(&mut b, 8, 4);
        for iy in 0..4 {
            for ix in 0..8 {
                coeff[(qy + iy * 2) * 8 + ix] = b[iy * 8 + ix];
            }
        }
    }
    let (c0, c1) = (coeff[0], coeff[8]);
    coeff[0] = (c0 + c1) * 0.5;
    coeff[8] = (c0 - c1) * 0.5;
    coeff
}

fn dct8x4_8x8_inv(coeff: &mut [f32; 64]) {
    dct4x8_8x8_inv(coeff);
    transpose8x8(coeff);
}

fn dct4x8_matrices() -> &'static [[f32; 64]; 3] {
    static M: OnceLock<[[f32; 64]; 3]> = OnceLock::new();
    M.get_or_init(|| [dct4x8_matrix(0), dct4x8_matrix(1), dct4x8_matrix(2)])
}

fn dct4x8_matrix(ch: usize) -> [f32; 64] {
    const BANDS: [&[f32]; 3] = [
        &[2198.0505, -0.96269625, -0.7619425, -0.65511405],
        &[764.36554, -0.926302, -0.967523, -0.2784529],
        &[527.10754, -1.4594386, -1.4500821, -1.5843723],
    ];
    let mat = dct_quant_weights_n(BANDS[ch], 8, 4);
    let mut ret = [1.0f32; 64];
    for y in 0..4 {
        for x in 0..8 {
            let v = mat[y * 8 + x];
            ret[y * 16 + x] = v;
            ret[(y * 2 + 1) * 8 + x] = v;
        }
    }
    for x in &mut ret {
        *x = 1.0 / x.max(1e-12);
    }
    ret
}

/// libjxl `FindBest8x8Transform` DCT4×8 / DCT8×4 (entropy_mul 0.859/0.8).
#[allow(clippy::too_many_arguments)]
fn try_dct4x8(
    bw: usize,
    bh: usize,
    pw: usize,
    planes: [&[f32]; 3],
    lf_inv: [f32; 3],
    hf_muls: &[i32],
    distance: f32,
    quant_bias: [f32; 3],
    quant_bias_num: f32,
    gs: f32,
    lf_q: &mut [Vec<i32>; 3],
    ac_q: &mut [Vec<i32>; 3],
    acs: &mut [u8],
    mask: &[f32],
    cfl: &CflMaps,
) {
    let mat48 = dct4x8_matrices();
    let mat8 = dct8_matrices();
    let zigzag8 = dct8_zigzag_ac();
    let zigzag4 = dct4_zigzag_ac();
    let qm_scale = [0.8f32, 1.0, 1.0];
    const MUL48: f32 = 0.859_316_4 / 0.8;
    let (cost_delta, zeros_mul, loss_mul) = entropy_scales(distance);
    for by in 0..bh {
        for bx in 0..bw {
            let i = by * bw + bx;
            if acs[i] != ACS_DCT8 {
                continue;
            }
            let cost8 = estimate_8x8(
                planes, pw, bx, by, i, lf_q, ac_q, hf_muls[i], gs, mat8, qm_scale, &zigzag8,
                lf_inv, cost_delta, zeros_mul, loss_mul, mask, cfl,
            );
            let hf = hf_muls[i] as f32;
            let (kx, kb) = cfl.factors(bx, by);
            let mut best = cost8;
            let mut best_kind = ACS_DCT8;
            let mut best_ac = [[0i32; 63]; 3];
            let mut best_lf = [0i32; 3];
            for (kind, fwd, inv) in [
                (
                    ACS_DCT4X8,
                    dct4x8_8x8_fwd as fn(&[f32], usize, usize, usize) -> [f32; 64],
                    dct4x8_8x8_inv as fn(&mut [f32; 64]),
                ),
                (ACS_DCT8X4, dct8x4_8x8_fwd, dct8x4_8x8_inv),
            ] {
                let mut orig = [
                    fwd(planes[0], pw, bx * 8, by * 8),
                    fwd(planes[1], pw, bx * 8, by * 8),
                    fwd(planes[2], pw, bx * 8, by * 8),
                ];
                apply_cfl_triple_lf(&mut orig, kx, kb, 1, 8);
                let mut qs = Vec::with_capacity(3 * 64);
                let mut spatial = orig;
                let mut ac_store = [[0i32; 63]; 3];
                let mut lf_store = [0i32; 3];
                for c in 0..3 {
                    let mul = (65536.0 / gs) * qm_scale[c];
                    lf_store[c] = (orig[c][0] * lf_inv[c]).round() as i32;
                    let dc_rec = lf_store[c] as f32 / lf_inv[c];
                    spatial[c][0] = orig[c][0] - dc_rec;
                    qs.push(lf_store[c]);
                    for (k, &z) in zigzag4.iter().enumerate() {
                        let m = mat48[c][z as usize] * mul;
                        let t = if m.abs() < 1e-12 {
                            0.0
                        } else {
                            orig[c][z as usize] / m * hf
                        };
                        let q = quantize_with_bias(t, quant_bias[c], quant_bias_num);
                        ac_store[c][k] = q;
                        spatial[c][z as usize] = orig[c][z as usize] - q as f32 * m / hf.max(1e-6);
                        qs.push(q);
                    }
                    let mut sp = spatial[c];
                    inv(&mut sp);
                    apply_spatial_mask(&mut sp, mask, pw, bx * 8, by * 8, 8, 8);
                    spatial[c] = sp;
                }
                let token = hf_cost(&qs, cost_delta, zeros_mul);
                let loss = eighth_loss(
                    [&spatial[0][..], &spatial[1][..], &spatial[2][..]],
                    64,
                    quant_norm_field(&[hf_muls[i]], gs),
                    loss_mul,
                );
                let cost = token * MUL48 + loss;
                if cost < best {
                    best = cost;
                    best_kind = kind;
                    best_ac = ac_store;
                    best_lf = lf_store;
                }
            }
            if best_kind == ACS_DCT8 {
                continue;
            }
            acs[i] = best_kind;
            for c in 0..3 {
                lf_q[c][i] = best_lf[c];
                ac_q[c][i * 63..i * 63 + 63].copy_from_slice(&best_ac[c]);
            }
        }
    }
}

/// libjxl `DCFromLowestFrequencies(DCT16)`: scaled 2×2 IDCT of the LL 2×2.
fn dc_from_lowest_16(coeff: &[f32; 256]) -> [f32; 4] {
    let mut b = [
        coeff[0] * DCT16_LF_SCALE[0] * DCT16_LF_SCALE[0],
        coeff[1] * DCT16_LF_SCALE[0] * DCT16_LF_SCALE[1],
        coeff[16] * DCT16_LF_SCALE[1] * DCT16_LF_SCALE[0],
        coeff[17] * DCT16_LF_SCALE[1] * DCT16_LF_SCALE[1],
    ];
    idct2_2d(&mut b);
    b
}

fn natural_order_16() -> &'static [(u16, u16)] {
    static O: OnceLock<Vec<(u16, u16)>> = OnceLock::new();
    O.get_or_init(|| natural_order(16, 16))
}

/// jxl-oxide `const_compute_natural_order`.
fn natural_order(bw: usize, bh: usize) -> Vec<(u16, u16)> {
    let y_scale = bw / bh.max(1);
    let lbw = bw / 8;
    let lbh = bh / 8;
    let mut ret = vec![(0u16, 0u16); bw * bh];
    let mut idx = 0usize;
    while idx < lbw * lbh {
        let x = idx % lbw;
        let y = idx / lbw;
        ret[idx] = (x as u16, y as u16);
        idx += 1;
    }
    let mut dist = 1usize;
    while dist < 2 * bw {
        let margin = dist.saturating_sub(bw);
        let mut order = margin;
        while order < dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            order += 1;
            if x < lbw && y < lbw {
                continue;
            }
            if y_scale != 0 && y % y_scale != 0 {
                continue;
            }
            if idx < ret.len() {
                ret[idx] = (x as u16, (y / y_scale.max(1)) as u16);
                idx += 1;
            }
        }
        dist += 1;
    }
    ret.truncate(idx);
    ret
}

fn dc_from_lowest_32(coeff: &[f32; 1024]) -> [f32; 16] {
    let s = DCT32_LF_SCALE;
    let mut b = [0f32; 16];
    for y in 0..4 {
        for x in 0..4 {
            b[y * 4 + x] = coeff[y * 32 + x] * s[y] * s[x];
        }
    }
    idct4_2d(&mut b);
    b
}

fn dct16_matrices() -> &'static [[f32; 256]; 3] {
    static M: OnceLock<[[f32; 256]; 3]> = OnceLock::new();
    M.get_or_init(|| [dct16_matrix(0), dct16_matrix(1), dct16_matrix(2)])
}

fn dct16_matrix(ch: usize) -> [f32; 256] {
    let v = dct_quant_weights_n(DCT16_BANDS[ch], 16, 16);
    let mut m = [1.0f32; 256];
    for (i, x) in v.iter().enumerate() {
        m[i] = 1.0 / *x;
    }
    m
}

fn dct32_matrices() -> &'static [[f32; 1024]; 3] {
    static M: OnceLock<[[f32; 1024]; 3]> = OnceLock::new();
    M.get_or_init(|| [dct32_matrix(0), dct32_matrix(1), dct32_matrix(2)])
}

fn dct32_matrix(ch: usize) -> [f32; 1024] {
    let v = dct_quant_weights_n(DCT32_BANDS[ch], 32, 32);
    let mut m = [1.0f32; 1024];
    for (i, x) in v.iter().enumerate() {
        m[i] = 1.0 / *x;
    }
    m
}

fn dct16x8_matrices() -> &'static [[f32; 128]; 3] {
    static M: OnceLock<[[f32; 128]; 3]> = OnceLock::new();
    M.get_or_init(|| [dct16x8_matrix(0), dct16x8_matrix(1), dct16x8_matrix(2)])
}

fn dct16x8_matrix(ch: usize) -> [f32; 128] {
    let v = dct_quant_weights_n(DCT16X8_BANDS[ch], 16, 8);
    let mut m = [1.0f32; 128];
    for (i, x) in v.iter().enumerate() {
        m[i] = 1.0 / *x;
    }
    m
}

fn dct_quant_weights_n(params: &[f32], width: u32, height: u32) -> Vec<f32> {
    fn mult(x: f32) -> f32 {
        if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
    }
    let mut bands = Vec::with_capacity(params.len());
    let mut last = params[0];
    bands.push(last);
    for &val in &params[1..] {
        last *= mult(val);
        bands.push(last);
    }
    let mut ret = vec![1.0f32; (width * height) as usize];
    let max = std::f32::consts::SQRT_2 + 1e-6;
    let len = bands.len();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            let pos = (dx * dx + dy * dy).sqrt() * (len - 1) as f32 / max;
            let idx = (pos as usize).min(len - 2);
            let frac = pos - idx as f32;
            let a = bands[idx];
            let b = bands[idx + 1];
            ret[y * width as usize + x] = a * (b / a).powf(frac.clamp(0.0, 1.0));
        }
    }
    ret
}

/// Zigzag AC indices into a row-major 8×8 (skips DC at 0).
fn dct8_zigzag_ac() -> [u8; 63] {
    let mut ret = [0u8; 63];
    let mut idx = 0usize;
    let mut dist = 1usize;
    while dist < 16 {
        let margin = dist.saturating_sub(8);
        let mut order = margin;
        while order < dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            order += 1;
            if x < 1 && y < 1 {
                continue;
            }
            if x < 8 && y < 8 {
                // Decoder writes order (dx, dy) to grid(dx, dy) = (col, row).
                // `dct8_fwd_from_rows` emits coef[fx][fy] in [row=fx][col=fy], so
                // scatter with x*8+y (not y*8+x) or AC is transposed.
                ret[idx] = (x * 8 + y) as u8;
                idx += 1;
            }
        }
        dist += 1;
    }
    debug_assert_eq!(idx, 63);
    ret
}

/// DCT4/Identity/AFV `need_transpose = false`: natural `(x, y)` is written to
/// `grid(x, y)` without the DCT8 swap, so scatter is `y * 8 + x`.
fn dct4_zigzag_ac() -> [u8; 63] {
    let mut ret = [0u8; 63];
    let mut idx = 0usize;
    let mut dist = 1usize;
    while dist < 16 {
        let margin = dist.saturating_sub(8);
        let mut order = margin;
        while order < dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            order += 1;
            if x < 1 && y < 1 {
                continue;
            }
            if x < 8 && y < 8 {
                ret[idx] = (y * 8 + x) as u8;
                idx += 1;
            }
        }
        dist += 1;
    }
    debug_assert_eq!(idx, 63);
    ret
}

fn acc_token(freqs: &mut Vec<u64>, v: u32) {
    let (t, _, _) = hybrid_uint_400(v);
    let t = t as usize;
    if t >= freqs.len() {
        freqs.resize(t + 1, 0);
    }
    freqs[t] += 1;
}

fn finish_code(mut freqs: Vec<u64>) -> PrefixCode {
    if freqs.iter().all(|&x| x == 0) {
        freqs.resize(1, 0);
        freqs[0] = 1;
    }
    for slot in freqs.iter_mut().take(19) {
        *slot = slot.saturating_add(1);
    }
    PrefixCode::from_freqs(&freqs)
}

fn finish_ans(mut freqs: Vec<u64>) -> AnsHist {
    if freqs.iter().all(|&x| x == 0) {
        freqs.resize(1, 0);
        freqs[0] = 1;
    }
    for slot in freqs.iter_mut().take(19) {
        *slot = slot.saturating_add(1);
    }
    AnsHist::from_freqs(&freqs)
}

/// Default `HfBlockContext.block_ctx_map` (Y / X / B × 13 order ids).
const BLOCK_CTX_MAP: [u8; 39] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7, 8, 9,
    9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];

fn order_id(acs: u8) -> usize {
    match acs {
        ACS_DCT4 | ACS_DCT4X8 | ACS_DCT8X4 => 1,
        ACS_DCT16 => 2,
        ACS_DCT32 => 3,
        ACS_DCT16X8 | ACS_DCT8X16 => 4,
        _ => 0,
    }
}

fn block_ctx(acs: u8, c: usize, hf_mul: i32, bcm: &BlockCtxModel) -> u8 {
    let ch = match c {
        1 => 0,
        0 => 1,
        _ => 2,
    };
    let mut hf_idx = 0usize;
    for &t in &bcm.qf_thresholds {
        if hf_mul > t as i32 {
            hf_idx += 1;
        }
    }
    let nseg = bcm.qf_thresholds.len() + 1;
    let idx = (ch * 13 + order_id(acs)) * nseg + hf_idx;
    *bcm.map.get(idx).unwrap_or(&0)
}

/// jxl-oxide `COEFF_FREQ_CONTEXT` / `COEFF_NUM_NONZERO_CONTEXT`.
const COEFF_FREQ_CONTEXT: [u32; 63] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
    20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27,
    27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
    0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152, 180,
    180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206,
];

const HF_COEFF_SPLIT: usize = 80;

fn raw_hf_ctx(bc: u8, kind: u8) -> usize {
    bc as usize * 3 + kind as usize
}

/// Decoder map: 37×N nnz contexts then N×458 coeff contexts.
fn hf_cluster_map(assign: &[u8], n_bc: usize) -> Vec<u8> {
    let n_bc = n_bc.max(1);
    let mut m = vec![0u8; 495 * n_bc];
    for pred in 0..37 {
        for bc in 0..n_bc {
            m[bc + pred * n_bc] = assign[raw_hf_ctx(bc as u8, 0)];
        }
    }
    for bc in 0..n_bc {
        let base = bc * 458 + 37 * n_bc;
        for i in 0..458 {
            let kind = 1 + u8::from(i >= HF_COEFF_SPLIT);
            m[base + i] = assign[raw_hf_ctx(bc as u8, kind)];
        }
    }
    m
}

fn hist_shannon_bits(f: &[u64]) -> f64 {
    let n: u64 = f.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut h = nf * nf.log2();
    for &c in f {
        if c > 0 {
            h -= c as f64 * (c as f64).log2();
        }
    }
    h
}

fn add_hist(dst: &mut Vec<u64>, src: &[u64]) {
    if src.len() > dst.len() {
        dst.resize(src.len(), 0);
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Greedy merge of similar histograms down to `max_clusters` (Shannon extra-bits).
fn greedy_cluster(freqs: &[Vec<u64>], max_clusters: usize) -> (Vec<Vec<u64>>, Vec<u8>) {
    let n = freqs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut merged = freqs.to_vec();
    let mut active: Vec<usize> = (0..n)
        .filter(|&i| freqs[i].iter().any(|&c| c > 0))
        .collect();
    if active.is_empty() {
        return (vec![vec![1u64]], vec![0u8; n]);
    }
    const HEADER: f64 = 80.0;
    while active.len() > max_clusters {
        let mut best = f64::INFINITY;
        let mut pair = (0usize, 1usize);
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let a = &merged[active[i]];
                let b = &merged[active[j]];
                let mut sum = a.clone();
                add_hist(&mut sum, b);
                let extra =
                    hist_shannon_bits(&sum) - hist_shannon_bits(a) - hist_shannon_bits(b) - HEADER;
                if extra < best {
                    best = extra;
                    pair = (i, j);
                }
            }
        }
        let (i, j) = pair;
        let a = active[i];
        let b = active[j];
        let src = merged[b].clone();
        add_hist(&mut merged[a], &src);
        parent[b] = a;
        active.swap_remove(j);
    }
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut remap = vec![u8::MAX; n];
    let mut out = Vec::new();
    for &i in &active {
        let r = find(&mut parent, i);
        if remap[r] == u8::MAX {
            remap[r] = out.len() as u8;
            out.push(std::mem::take(&mut merged[r]));
        }
    }
    if out.is_empty() {
        out.push(vec![1u64]);
    }
    let mut assign = vec![0u8; n];
    for i in 0..n {
        if freqs[i].iter().all(|&c| c == 0) {
            continue;
        }
        let r = find(&mut parent, i);
        debug_assert_ne!(remap[r], u8::MAX);
        assign[i] = remap[r];
    }
    (out, assign)
}

#[cfg(test)]
mod order_tests {
    use super::{
        dct4_8x8_fwd, dct4_8x8_inv, dct4_zigzag_ac, dct4x8_8x8_fwd, dct4x8_8x8_inv, dct8_zigzag_ac,
        dct8x4_8x8_fwd, dct8x4_8x8_inv, lehmer_suffix, perm_from_zero_counts,
    };

    #[test]
    fn dct4_zigzag_is_not_transposed() {
        let z4 = dct4_zigzag_ac();
        let z8 = dct8_zigzag_ac();
        // Natural first AC is (x, y) = (1, 0): DCT4 stays at [0*8+1], DCT8 swaps to [1*8+0].
        assert_eq!(z4[0], 1);
        assert_eq!(z8[0], 8);
        assert_eq!(z4[1], 8);
        assert_eq!(z8[1], 1);
    }

    #[test]
    fn dct4_8x8_roundtrip() {
        let plane: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.17).sin() * 3.0 + 0.4)
            .collect();
        let mut c = dct4_8x8_fwd(&plane, 8, 0, 0);
        dct4_8x8_inv(&mut c);
        for (i, (got, want)) in c.iter().zip(plane.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4, "idx {i}: got {got} want {want}");
        }
    }

    #[test]
    fn dct4x8_8x8_roundtrip() {
        let plane: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.17).sin() * 3.0 + 0.4)
            .collect();
        let mut c = dct4x8_8x8_fwd(&plane, 8, 0, 0);
        dct4x8_8x8_inv(&mut c);
        for (i, (got, want)) in c.iter().zip(plane.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4, "idx {i}: got {got} want {want}");
        }
        let mut c = dct8x4_8x8_fwd(&plane, 8, 0, 0);
        dct8x4_8x8_inv(&mut c);
        for (i, (got, want)) in c.iter().zip(plane.iter()).enumerate() {
            assert!(
                (got - want).abs() < 2e-4,
                "8x4 idx {i}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn lehmer_identity_is_all_zero() {
        let perm: Vec<u16> = (0..64).collect();
        let code = lehmer_suffix(&perm, 1);
        assert!(code.iter().all(|&c| c == 0), "{code:?}");
    }

    #[test]
    fn lehmer_swap_two_ac() {
        // Keep DC, swap natural 1 and 2.
        let mut perm: Vec<u16> = (0..8).collect();
        perm.swap(1, 2);
        let code = lehmer_suffix(&perm, 1);
        // remaining = [1,2,3,4,5,6,7]; pick 2 → index 1; then pick 1 → index 0.
        assert_eq!(code[0], 1);
        assert_eq!(code[1], 0);
    }

    #[test]
    fn perm_from_zeros_keeps_llf_and_sorts_ac() {
        // two AC bins: more zeros at scan 0 than scan 1 → scan 1 first.
        let zeros = [10i32, 1, 5];
        let perm = perm_from_zero_counts(&zeros, 1, 4);
        assert_eq!(perm[0], 0);
        assert_eq!(&perm[1..], &[2, 3, 1]);
    }
}

#[cfg(test)]
mod cluster_tests {
    use super::greedy_cluster;

    #[test]
    fn merged_away_raw_keeps_root_cluster() {
        let mut freqs = vec![vec![0u64; 8]; 5];
        freqs[0][1] = 10;
        freqs[1][1] = 9;
        freqs[2][3] = 8;
        freqs[3][3] = 7;
        freqs[4][5] = 6;
        let (out, assign) = greedy_cluster(&freqs, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(assign.len(), 5);
        for (i, f) in freqs.iter().enumerate() {
            if f.iter().any(|&c| c > 0) {
                assert!((assign[i] as usize) < out.len());
            }
        }
        // Every non-empty raw hist's tokens live in its assigned cluster.
        for (i, f) in freqs.iter().enumerate() {
            let cl = assign[i] as usize;
            for (t, &c) in f.iter().enumerate() {
                if c > 0 {
                    assert!(
                        out[cl].get(t).copied().unwrap_or(0) >= c,
                        "raw {i} token {t} missing from cluster {cl}"
                    );
                }
            }
        }
    }
}

fn nblocks_log(acs: u8) -> u32 {
    match acs {
        ACS_DCT16 => 2,
        ACS_DCT32 => 4,
        ACS_DCT16X8 | ACS_DCT8X16 => 1,
        _ => 0,
    }
}

fn hf_coeff_ctx(nnz_left: u32, scan_idx: usize, prev_nz: bool, nblog: u32) -> usize {
    let nz = ((nnz_left.saturating_sub(1) >> nblog) as usize).min(62);
    let idx = (scan_idx >> nblog).min(62);
    let ctx = (COEFF_NUM_NONZERO_CONTEXT[nz] + COEFF_FREQ_CONTEXT[idx]) * 2 + u32::from(prev_nz);
    ctx as usize
}

const HF_LZ77_DIST_CLUSTER: u8 = 6;

/// libjxl `ComputeCoeffOrder`: permutation of natural-order indices per (order, channel).
#[derive(Clone, Debug)]
struct CoeffOrders {
    used: u16,
    /// Empty vec = default natural order.
    perm: [[Vec<u16>; 3]; 13],
}

impl Default for CoeffOrders {
    fn default() -> Self {
        Self {
            used: 0,
            perm: core::array::from_fn(|_| [Vec::new(), Vec::new(), Vec::new()]),
        }
    }
}

fn order_size(order_id: usize) -> usize {
    match order_id {
        2 => 256,
        3 => 1024,
        4 => 128,
        _ => 64,
    }
}

fn order_llf(order_id: usize) -> usize {
    match order_id {
        2 => 4,
        3 => 16,
        4 => 2,
        _ => 1,
    }
}

fn coeff_order_ctx(x: u32) -> u8 {
    let v = if x >= 0x8000_0000 {
        32
    } else {
        (x + 1).next_power_of_two().trailing_zeros()
    };
    v.min(7) as u8
}

fn lehmer_suffix(perm: &[u16], skip: usize) -> Vec<u32> {
    let size = perm.len();
    let mut temp: Vec<u16> = (skip as u16..size as u16).collect();
    let mut code = Vec::with_capacity(size.saturating_sub(skip));
    for &p in &perm[skip..] {
        let idx = temp.iter().position(|&x| x == p).expect("perm");
        code.push(idx as u32);
        temp.remove(idx);
    }
    code
}

fn perm_from_zero_counts(zeros: &[i32], llf: usize, size: usize) -> Vec<u16> {
    let inv = 1.0 / (size as f32).sqrt();
    let q = |z: i32| (z as f32 * inv + 0.1) as u32;
    let mut idx: Vec<u16> = (0..size as u16).collect();
    idx[llf..].sort_by(|&a, &b| {
        let za = q(zeros[a as usize - llf]);
        let zb = q(zeros[b as usize - llf]);
        za.cmp(&zb).then(a.cmp(&b))
    });
    idx
}

fn count_zeros_in_slices(slices: &[&[i32]], ac_len: usize) -> Vec<i32> {
    let mut z = vec![0i32; ac_len];
    for s in slices {
        for (k, &v) in s.iter().enumerate() {
            z[k] += i32::from(v == 0);
        }
    }
    z
}

#[allow(clippy::too_many_arguments)]
fn collect_order_slices<'a>(
    acs: &[u8],
    want: u8,
    ac_q: &'a [Vec<i32>; 3],
    ac16: &'a [Vec<i32>; 3],
    ac16_at: &[u32],
    ac32: &'a [Vec<i32>; 3],
    ac32_at: &[u32],
    ac168: &'a [Vec<i32>; 3],
    ac168_at: &[u32],
    c: usize,
) -> Vec<&'a [i32]> {
    let n = acs.len();
    let mut out = Vec::new();
    match want {
        ACS_DCT8 | ACS_DCT4 | ACS_DCT4X8 | ACS_DCT8X4 => {
            for (i, &a) in acs.iter().enumerate() {
                if a == want
                    || (want == ACS_DCT4 && matches!(a, ACS_DCT4 | ACS_DCT4X8 | ACS_DCT8X4))
                {
                    out.push(&ac_q[c][i * 63..i * 63 + 63]);
                }
            }
        }
        ACS_DCT16 => {
            for i in 0..n {
                if acs[i] == ACS_DCT16 {
                    let s = ac16_at[i] as usize * 252;
                    out.push(&ac16[c][s..s + 252]);
                }
            }
        }
        ACS_DCT32 => {
            for i in 0..n {
                if acs[i] == ACS_DCT32 {
                    let s = ac32_at[i] as usize * 1008;
                    out.push(&ac32[c][s..s + 1008]);
                }
            }
        }
        ACS_DCT16X8 | ACS_DCT8X16 => {
            for i in 0..n {
                if acs[i] == ACS_DCT16X8 || acs[i] == ACS_DCT8X16 {
                    let s = ac168_at[i] as usize * 126;
                    out.push(&ac168[c][s..s + 126]);
                }
            }
        }
        _ => {}
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn compute_coeff_orders(
    bw: usize,
    bh: usize,
    acs: &[u8],
    ac_q: &[Vec<i32>; 3],
    ac16: &[Vec<i32>; 3],
    ac16_at: &[u32],
    ac32: &[Vec<i32>; 3],
    ac32_at: &[u32],
    ac168: &[Vec<i32>; 3],
    ac168_at: &[u32],
) -> CoeffOrders {
    let mut out = CoeffOrders::default();
    // libjxl: default orders for images smaller than 5×5 blocks.
    if bw < 5 && bh < 5 {
        return out;
    }
    let specs: [(usize, u8); 5] = [
        (0, ACS_DCT8),
        (1, ACS_DCT4),
        (2, ACS_DCT16),
        (3, ACS_DCT32),
        (4, ACS_DCT16X8),
    ];
    for (oid, want) in specs {
        let size = order_size(oid);
        let llf = order_llf(oid);
        let ac_len = size - llf;
        let mut any = false;
        for c in 0..3 {
            let slices = collect_order_slices(
                acs, want, ac_q, ac16, ac16_at, ac32, ac32_at, ac168, ac168_at, c,
            );
            if slices.is_empty() {
                continue;
            }
            any = true;
            let zeros = count_zeros_in_slices(&slices, ac_len);
            let perm = perm_from_zero_counts(&zeros, llf, size);
            if perm.iter().enumerate().any(|(i, &p)| p as usize != i) {
                out.perm[oid][c] = perm;
            }
        }
        if any && (0..3).any(|c| !out.perm[oid][c].is_empty()) {
            // Encode all three channels when any one is non-default.
            for c in 0..3 {
                if out.perm[oid][c].is_empty() {
                    out.perm[oid][c] = (0..size as u16).collect();
                }
            }
            out.used |= 1 << oid;
        }
    }
    out
}

fn reorder_ac(slice: &[i32], perm: &[u16], llf: usize) -> Vec<i32> {
    let mut out = vec![0i32; slice.len()];
    for i in 0..slice.len() {
        out[i] = slice[perm[i + llf] as usize - llf];
    }
    out
}

fn encode_coeff_orders(w: &mut BitWriter, orders: &CoeffOrders) {
    write_u32(
        w,
        u32::from(orders.used),
        U32::Const(0x5F),
        U32::Const(0x13),
        U32::Const(0),
        u(13),
    );
    if orders.used == 0 {
        return;
    }
    let mut vals = Vec::new();
    let mut ctxs = Vec::new();
    for oid in 0..13 {
        if orders.used & (1 << oid) == 0 {
            continue;
        }
        let size = order_size(oid) as u32;
        let skip = order_llf(oid);
        for c in 0..3 {
            let perm = &orders.perm[oid][c];
            let code = lehmer_suffix(perm, skip);
            let mut end = code.len();
            while end > 0 && code[end - 1] == 0 {
                end -= 1;
            }
            vals.push(end as u32);
            ctxs.push(coeff_order_ctx(size));
            let mut last = 0u32;
            for &v in &code[..end] {
                vals.push(v);
                ctxs.push(coeff_order_ctx(last));
                last = v;
            }
        }
    }
    let mut freqs = vec![vec![0u64; 32]; 8];
    super::entropy::for_each_token_ctx_len(
        &vals,
        &ctxs,
        false,
        LZ77_MIN_LENGTH,
        |cluster, token, _, _| {
            let i = cluster as usize;
            if i < freqs.len() {
                let t = token as usize;
                if t >= freqs[i].len() {
                    freqs[i].resize(t + 1, 0);
                }
                freqs[i][t] += 1;
            }
        },
    );
    let hists: Vec<AnsHist> = freqs.into_iter().map(finish_ans).collect();
    let map: Vec<u8> = (0..8).collect();
    write_ans_histograms_len(w, &hists, Some(&map), None, LZ77_MIN_LENGTH);
    let mut tokens = Vec::new();
    collect_ans_tokens_ctx_len(&vals, &ctxs, false, None, LZ77_MIN_LENGTH, &mut tokens);
    write_ans_tokens_ex(w, &hists, None, &tokens);
}

#[allow(clippy::too_many_arguments)]
fn hist_from_hf_mixed(
    width: u32,
    height: u32,
    bw: usize,
    ac_q: &[Vec<i32>; 3],
    acs: &[u8],
    ac16: &[Vec<i32>; 3],
    ac16_at: &[u32],
    ac32: &[Vec<i32>; 3],
    ac32_at: &[u32],
    ac168: &[Vec<i32>; 3],
    ac168_at: &[u32],
    orders: &CoeffOrders,
    hf_muls: &[i32],
    bcm: &BlockCtxModel,
) -> (Vec<AnsHist>, Vec<u8>, Vec<u8>) {
    let n_raw = bcm.n_clusters as usize * 3;
    let identity: Vec<u8> = (0..n_raw as u8).collect();
    let mut freqs = vec![vec![0u64; 32]; n_raw];
    let num_groups = width.div_ceil(GROUP_DIM) * height.div_ceil(GROUP_DIM);
    for g in 0..num_groups {
        let (vals, ctxs) = hf_group_stream(
            width, height, bw, g, ac_q, acs, ac16, ac16_at, ac32, ac32_at, ac168, ac168_at,
            &identity, orders, hf_muls, bcm,
        );
        super::entropy::for_each_token_ctx_len(
            &vals,
            &ctxs,
            false,
            LZ77_MIN_LENGTH,
            |cluster, token, _, _| {
                let i = cluster as usize;
                if i < freqs.len() {
                    let t = token as usize;
                    if t >= freqs[i].len() {
                        freqs[i].resize(t + 1, 0);
                    }
                    freqs[i][t] += 1;
                }
            },
        );
    }
    let (merged, assign) = greedy_cluster(&freqs, 8);
    let map = hf_cluster_map(&assign, bcm.n_clusters as usize);
    let hists = merged.into_iter().map(finish_ans).collect();
    (hists, map, assign)
}

fn code_from_values(vals: &[u32]) -> PrefixCode {
    let mut freqs = vec![0u64; 32];
    for &v in vals {
        acc_token(&mut freqs, v);
    }
    finish_code(freqs)
}

fn count_nonzero(v: &[i32]) -> u32 {
    let mut n = 0u32;
    let mut i = 0;
    while i + 8 <= v.len() {
        let x = Simd::<i32, 8>::from_slice(&v[i..]);
        n += x.simd_ne(Simd::splat(0)).to_bitmask().count_ones();
        i += 8;
    }
    while i < v.len() {
        n += u32::from(v[i] != 0);
        i += 1;
    }
    n
}

fn encode_lf_global_vardct(
    w: &mut BitWriter,
    global_scale: u32,
    quant_lf: u32,
    lf_code: &PrefixCode,
    bcm: &BlockCtxModel,
) {
    w.put_bool(true); // LfChannelDequantization all_default
    write_u32(
        w,
        global_scale,
        off(1, 11),
        off(2049, 11),
        off(4097, 12),
        off(8193, 16),
    );
    write_u32(
        w,
        quant_lf,
        U32::Const(16),
        off(1, 5),
        off(1, 8),
        off(1, 16),
    );
    encode_hf_block_context(w, bcm);
    w.put_bool(true); // LfChannelCorrelation all_default (kx=0, kb=1)

    w.put_bool(true); // global MA tree
    write_ma_tree_gradient(w, 1, 0, 0);
    write_histograms(w, std::slice::from_ref(lf_code), None);
}

#[allow(clippy::too_many_arguments)]
fn encode_lf_group_vardct(
    w: &mut BitWriter,
    width: u32,
    height: u32,
    bw: usize,
    lf_g: u32,
    lf_q: &[Vec<i32>; 3],
    hf_muls: &[i32],
    acs: &[u8],
    code: &PrefixCode,
    cfl: &CflMaps,
) {
    let lf_dim = GROUP_DIM * 8;
    let lf_per_row = width.div_ceil(lf_dim).max(1);
    let gx = lf_g % lf_per_row;
    let gy = lf_g / lf_per_row;
    // LF samples are one per 8×8; an LF group covers LF_GROUP_DIM of them.
    let x0 = (gx * LF_GROUP_DIM) as usize;
    let y0 = (gy * LF_GROUP_DIM) as usize;
    let gw = ((width.div_ceil(8) as usize) - x0).min(LF_GROUP_DIM as usize);
    let gh = ((height.div_ceil(8) as usize) - y0).min(LF_GROUP_DIM as usize);

    w.put(2, 0); // extra_precision = 0

    // Modular: Y, X, B (channel order 1,0,2)
    write_modular_header(w, &[]);
    for &c in &[1usize, 0, 2] {
        let ch = crop_i32(&lf_q[c], bw, x0, y0, gw, gh);
        let mut res = vec![0u32; gw * gh];
        gradient_residuals(gw, gh, &ch, 1, &mut res);
        write_residual_run(w, code, &res, false);
    }

    // HfMetadata
    let n8 = gw * gh;
    let mut first_sel = Vec::new();
    let mut first_mul = Vec::new();
    for y in 0..gh {
        for x in 0..gw {
            let i = (y0 + y) * bw + (x0 + x);
            if acs[i] != ACS_OCCUPIED {
                first_sel.push(i32::from(acs[i]));
                first_mul.push(hf_muls[i]);
            }
        }
    }
    let nb = first_sel.len();
    let nbits = (n8.max(1)).next_power_of_two().trailing_zeros();
    w.put(nbits, (nb as u32).saturating_sub(1) as u64);

    write_modular_header(w, &[]);
    let lf_w = (gw * 8).min((width as usize).saturating_sub(gx as usize * lf_dim as usize));
    let lf_h = (gh * 8).min((height as usize).saturating_sub(gy as usize * lf_dim as usize));
    let cfl_w = lf_w.div_ceil(64).max(1);
    let cfl_h = lf_h.div_ceil(64).max(1);
    let tx0 = (x0 / 8).min(cfl.tw.saturating_sub(1));
    let ty0 = (y0 / 8).min(cfl.th.saturating_sub(1));
    let sw = cfl_w.min(cfl.tw - tx0).max(1);
    let sh = cfl_h.min(cfl.th - ty0).max(1);
    let mut cx = crop_i32(&cfl.x, cfl.tw, tx0, ty0, sw, sh);
    let mut cb = crop_i32(&cfl.b, cfl.tw, tx0, ty0, sw, sh);
    if sw != cfl_w || sh != cfl_h {
        let mut px = vec![0i32; cfl_w * cfl_h];
        let mut pb = vec![0i32; cfl_w * cfl_h];
        for y in 0..sh {
            px[y * cfl_w..y * cfl_w + sw].copy_from_slice(&cx[y * sw..y * sw + sw]);
            pb[y * cfl_w..y * cfl_w + sw].copy_from_slice(&cb[y * sw..y * sw + sw]);
        }
        cx = px;
        cb = pb;
    }
    encode_i32_plane(w, code, &cx, cfl_w, cfl_h);
    encode_i32_plane(w, code, &cb, cfl_w, cfl_h);
    encode_block_info(w, code, &first_sel, &first_mul);
    // EPF sharpness 0..=7; 4 is the default LUT midpoint (cjxl computes this
    // per-block; a constant 4 applies the filter instead of disabling it).
    encode_const_plane(w, code, gw, gh, 4);
}

fn encode_block_info(w: &mut BitWriter, code: &PrefixCode, dct_select: &[i32], hf_muls: &[i32]) {
    let nb = dct_select.len();
    debug_assert_eq!(hf_muls.len(), nb);
    let mut plane = vec![0i32; nb * 2];
    plane[..nb].copy_from_slice(dct_select);
    for i in 0..nb {
        plane[nb + i] = hf_muls[i] - 1;
    }
    let mut res = vec![0u32; nb * 2];
    gradient_residuals(nb, 2, &plane, 1, &mut res);
    write_residual_run(w, code, &res, false);
}

#[allow(dead_code)]
fn encode_zero_plane(w: &mut BitWriter, code: &PrefixCode, width: usize, height: usize) {
    encode_const_plane(w, code, width, height, 0);
}

fn encode_i32_plane(
    w: &mut BitWriter,
    code: &PrefixCode,
    plane: &[i32],
    width: usize,
    height: usize,
) {
    debug_assert_eq!(plane.len(), width * height);
    let mut res = vec![0u32; width * height];
    gradient_residuals(width, height, plane, 1, &mut res);
    write_residual_run(w, code, &res, false);
}

fn encode_const_plane(
    w: &mut BitWriter,
    code: &PrefixCode,
    width: usize,
    height: usize,
    value: i32,
) {
    let n = width * height;
    let mut res = vec![0u32; n];
    let plane = vec![value; n];
    gradient_residuals(width, height, &plane, 1, &mut res);
    write_residual_run(w, code, &res, false);
}

fn crop_i32(plane: &[i32], stride: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<i32> {
    let mut out = vec![0i32; w * h];
    for row in 0..h {
        let src = (y + row) * stride + x;
        out[row * w..(row + 1) * w].copy_from_slice(&plane[src..src + w]);
    }
    out
}

fn encode_hf_global(
    w: &mut BitWriter,
    hists: &[AnsHist],
    map: &[u8],
    orders: &CoeffOrders,
    dist: Option<&AnsHist>,
    num_groups: u32,
) {
    w.put_bool(true); // default dequant matrices
    let nbits = num_groups.next_power_of_two().trailing_zeros();
    w.put(nbits, 0); // num_hf_presets - 1 = 0 → 1 preset
    encode_coeff_orders(w, orders);
    write_ans_histograms_len(w, hists, Some(map), dist, LZ77_MIN_LENGTH);
}

#[allow(clippy::too_many_arguments)]
fn encode_pass_group_vardct(
    w: &mut BitWriter,
    width: u32,
    height: u32,
    bw: usize,
    g: u32,
    ac_q: &[Vec<i32>; 3],
    acs: &[u8],
    ac16: &[Vec<i32>; 3],
    ac16_at: &[u32],
    ac32: &[Vec<i32>; 3],
    ac32_at: &[u32],
    ac168: &[Vec<i32>; 3],
    ac168_at: &[u32],
    hists: &[AnsHist],
    assign: &[u8],
    orders: &CoeffOrders,
    dist: Option<&AnsHist>,
    hf_muls: &[i32],
    bcm: &BlockCtxModel,
) {
    let (vals, ctxs) = hf_group_stream(
        width, height, bw, g, ac_q, acs, ac16, ac16_at, ac32, ac32_at, ac168, ac168_at, assign,
        orders, hf_muls, bcm,
    );
    let mut tokens = Vec::with_capacity(vals.len() + 8);
    collect_ans_tokens_ctx_len_dist(
        &vals,
        &ctxs,
        dist.is_some(),
        dist.map(|_| HF_LZ77_DIST_CLUSTER),
        LZ77_MIN_LENGTH,
        0,
        &mut tokens,
    );
    write_ans_tokens_ex(w, hists, dist, &tokens);
}

#[allow(clippy::too_many_arguments)]
fn hf_group_stream(
    width: u32,
    height: u32,
    bw: usize,
    g: u32,
    ac_q: &[Vec<i32>; 3],
    acs: &[u8],
    ac16: &[Vec<i32>; 3],
    ac16_at: &[u32],
    ac32: &[Vec<i32>; 3],
    ac32_at: &[u32],
    ac168: &[Vec<i32>; 3],
    ac168_at: &[u32],
    assign: &[u8],
    orders: &CoeffOrders,
    hf_muls: &[i32],
    bcm: &BlockCtxModel,
) -> (Vec<u32>, Vec<u8>) {
    let gpr = width.div_ceil(GROUP_DIM).max(1);
    let gx = g % gpr;
    let gy = g / gpr;
    let x0 = (gx * (GROUP_DIM / 8)) as usize;
    let y0 = (gy * (GROUP_DIM / 8)) as usize;
    let gw = (bw - x0).min((GROUP_DIM / 8) as usize);
    let bh = height.div_ceil(8) as usize;
    let gh = (bh - y0).min((GROUP_DIM / 8) as usize);
    let mut values = Vec::with_capacity(gw * gh * 48);
    let mut ctxs = Vec::with_capacity(gw * gh * 48);
    let push = |values: &mut Vec<u32>,
                ctxs: &mut Vec<u8>,
                acs_id: u8,
                oid: usize,
                c: usize,
                slice: &[i32],
                hf_mul: i32| {
        let perm = &orders.perm[oid][c];
        if perm.is_empty() {
            hf_push_slice(values, ctxs, acs_id, c, slice, assign, hf_mul, bcm);
        } else {
            let re = reorder_ac(slice, perm, order_llf(oid));
            hf_push_slice(values, ctxs, acs_id, c, &re, assign, hf_mul, bcm);
        }
    };
    for by in y0..y0 + gh {
        for bx in x0..x0 + gw {
            let i = by * bw + bx;
            match acs[i] {
                ACS_OCCUPIED => {}
                ACS_DCT4 | ACS_DCT4X8 | ACS_DCT8X4 => {
                    for &c in &[1usize, 0, 2] {
                        push(
                            &mut values,
                            &mut ctxs,
                            acs[i],
                            1,
                            c,
                            &ac_q[c][i * 63..i * 63 + 63],
                            hf_muls[i],
                        );
                    }
                }
                ACS_DCT16 => {
                    let s = ac16_at[i] as usize * 252;
                    for &c in &[1usize, 0, 2] {
                        push(
                            &mut values,
                            &mut ctxs,
                            ACS_DCT16,
                            2,
                            c,
                            &ac16[c][s..s + 252],
                            hf_muls[i],
                        );
                    }
                }
                ACS_DCT32 => {
                    let s = ac32_at[i] as usize * 1008;
                    for &c in &[1usize, 0, 2] {
                        push(
                            &mut values,
                            &mut ctxs,
                            ACS_DCT32,
                            3,
                            c,
                            &ac32[c][s..s + 1008],
                            hf_muls[i],
                        );
                    }
                }
                ACS_DCT16X8 | ACS_DCT8X16 => {
                    let s = ac168_at[i] as usize * 126;
                    for &c in &[1usize, 0, 2] {
                        push(
                            &mut values,
                            &mut ctxs,
                            acs[i],
                            4,
                            c,
                            &ac168[c][s..s + 126],
                            hf_muls[i],
                        );
                    }
                }
                _ => {
                    for &c in &[1usize, 0, 2] {
                        push(
                            &mut values,
                            &mut ctxs,
                            ACS_DCT8,
                            0,
                            c,
                            &ac_q[c][i * 63..i * 63 + 63],
                            hf_muls[i],
                        );
                    }
                }
            }
        }
    }
    (values, ctxs)
}

#[allow(clippy::too_many_arguments)]
fn hf_push_slice(
    values: &mut Vec<u32>,
    ctxs: &mut Vec<u8>,
    acs: u8,
    c: usize,
    slice: &[i32],
    assign: &[u8],
    hf_mul: i32,
    bcm: &BlockCtxModel,
) {
    let nnz = count_nonzero(slice);
    let bc = block_ctx(acs, c, hf_mul, bcm);
    values.push(nnz);
    ctxs.push(assign[raw_hf_ctx(bc, 0)]);
    if nnz == 0 {
        return;
    }
    let nblog = nblocks_log(acs);
    let mut left = nnz;
    let mut prev = nnz <= (1 << nblog) * 4;
    for (scan, &v) in slice.iter().enumerate() {
        if left == 0 {
            break;
        }
        let ctx = hf_coeff_ctx(left, scan, prev, nblog);
        let kind = 1 + u8::from(ctx >= HF_COEFF_SPLIT);
        values.push(pack_signed(v));
        ctxs.push(assign[raw_hf_ctx(bc, kind)]);
        prev = v != 0;
        if prev {
            left -= 1;
        }
    }
}
