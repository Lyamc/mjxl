//! Image, frame, and TOC headers for modular lossless.

use super::bits::{
    BitWriter, U32, off, u, write_dim, write_enum, write_toc_size, write_u32, write_u64,
};

#[derive(Clone, Copy, Debug)]
pub struct ImageParams {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub group_size_shift: u32,
    pub bit_depth: u32,
    pub animation: Option<AnimHeader>,
    pub duration: u32,
    pub is_last: bool,
    pub xyb_encoded: bool,
    pub vardct: bool,
    /// Decoder Gaborish. cjxl: on for VarDCT when effort ≤ Hare and d > 0.5.
    pub gab: bool,
    /// EPF iterations 0..=3. cjxl: count of distance thresholds `{0.7, 1.5, 4.0}`.
    pub epf_iters: u8,
}

/// Image-level animation header (`tps = num/den`, `0` loops = forever).
#[derive(Clone, Copy, Debug)]
pub struct AnimHeader {
    pub tps_num: u32,
    pub tps_den: u32,
    pub num_loops: u32,
}

impl ImageParams {
    pub fn group_dim(&self) -> u32 {
        128 << self.group_size_shift
    }

    pub fn num_groups(&self) -> u32 {
        let dim = self.group_dim();
        self.width.div_ceil(dim) * self.height.div_ceil(dim)
    }

    pub fn num_lf_groups(&self) -> u32 {
        let dim = self.group_dim() * 8;
        self.width.div_ceil(dim) * self.height.div_ceil(dim)
    }

    pub fn groups_per_row(&self) -> u32 {
        self.width.div_ceil(self.group_dim())
    }

    pub fn is_single_group(&self) -> bool {
        self.num_groups() == 1
    }

    pub fn has_alpha(&self) -> bool {
        self.channels == 2 || self.channels == 4
    }

    pub fn is_gray(&self) -> bool {
        self.channels <= 2
    }
}

pub fn write_signature(w: &mut BitWriter) {
    w.put(16, 0x0AFF);
}

pub fn write_size_header(w: &mut BitWriter, width: u32, height: u32) {
    w.put_bool(false); // not div8
    write_dim(w, height);
    w.put(3, 0); // ratio = 0 (explicit width)
    write_dim(w, width);
}

pub fn write_image_metadata(w: &mut BitWriter, p: ImageParams) {
    w.put_bool(false); // all_default
    let extra = p.animation.is_some();
    w.put_bool(extra);
    if extra {
        w.put(3, 0); // orientation = 1 (stored as 1+u(3))
        w.put_bool(false); // have_intr_size
        w.put_bool(false); // have_preview
        w.put_bool(true); // have_animation
        let a = p.animation.unwrap();
        write_u32(
            w,
            a.tps_num,
            U32::Const(100),
            U32::Const(1000),
            off(1, 10),
            off(1, 30),
        );
        write_u32(
            w,
            a.tps_den,
            U32::Const(1),
            U32::Const(1001),
            off(1, 8),
            off(1, 10),
        );
        write_u32(w, a.num_loops, U32::Const(0), u(3), u(16), u(32));
        w.put_bool(false); // have_timecodes
    }
    w.put_bool(false); // bit_depth.float_sample
    write_u32(
        w,
        p.bit_depth,
        U32::Const(8),
        U32::Const(10),
        U32::Const(12),
        off(1, 6),
    );
    // 16-bit colour + RCT overflows signed i16 (Co = R−B). XYB modular
    // integers (X×65536) also overflow i16.
    w.put_bool(p.bit_depth < 16 && !p.xyb_encoded);
    if p.has_alpha() {
        write_u32(w, 1, U32::Const(0), U32::Const(1), off(2, 4), off(1, 12));
        if p.bit_depth == 8 {
            w.put_bool(true); // extra channel all_default (8-bit alpha)
        } else {
            w.put_bool(false);
            write_enum(w, 0); // Alpha
            w.put_bool(false); // integer
            write_u32(
                w,
                p.bit_depth,
                U32::Const(8),
                U32::Const(10),
                U32::Const(12),
                off(1, 6),
            );
            write_u32(w, 0, U32::Const(0), U32::Const(3), U32::Const(4), off(1, 3)); // dim_shift
            write_u32(w, 0, U32::Const(0), u(4), off(16, 5), off(48, 10)); // empty name
            w.put_bool(false); // alpha not associated
        }
    } else {
        write_u32(w, 0, U32::Const(0), U32::Const(1), off(2, 4), off(1, 12));
    }
    w.put_bool(p.xyb_encoded);
    if p.is_gray() {
        w.put_bool(false); // colour_encoding not all_default
        w.put_bool(false); // want_icc
        write_enum(w, 1); // Grey
        write_enum(w, 1); // D65
        w.put_bool(false); // not gamma
        write_enum(w, 13); // sRGB TF
        write_enum(w, 1); // relative intent
    } else {
        w.put_bool(true); // colour_encoding all_default (sRGB)
    }
    if extra {
        w.put_bool(true); // tone_mapping all_default
    }
    write_u64(w, 0); // extensions
    w.put_bool(true); // default_m (default transform data)
}

pub fn write_image_header(w: &mut BitWriter, p: ImageParams) {
    write_signature(w);
    write_size_header(w, p.width, p.height);
    write_image_metadata(w, p);
    w.zero_pad_to_byte();
}

pub fn write_frame_header(w: &mut BitWriter, p: ImageParams) {
    w.put_bool(false); // all_default
    w.put(2, 0); // regular frame
    w.put_bool(!p.vardct); // 0 = VarDCT, 1 = modular
    if p.vardct {
        write_u64(w, 0x80); // skip adaptive LF smoothing
    } else {
        write_u64(w, 0);
    }
    if !p.xyb_encoded {
        w.put_bool(false); // not YCbCr
    }
    write_u32(
        w,
        1,
        U32::Const(1),
        U32::Const(2),
        U32::Const(4),
        U32::Const(8),
    ); // upsampling
    if p.has_alpha() {
        write_u32(
            w,
            1,
            U32::Const(1),
            U32::Const(2),
            U32::Const(4),
            U32::Const(8),
        );
    }
    if !p.vardct {
        w.put(2, p.group_size_shift as u64);
    }
    if p.xyb_encoded && p.vardct {
        w.put(3, 3); // x_qm_scale default
        w.put(3, 2); // b_qm_scale default
    }
    write_u32(w, 1, U32::Const(1), U32::Const(2), U32::Const(3), off(4, 3)); // one pass
    w.put_bool(false); // no crop
    write_u32(w, 0, U32::Const(0), U32::Const(1), U32::Const(2), off(3, 2)); // kReplace
    if p.has_alpha() {
        write_u32(w, 0, U32::Const(0), U32::Const(1), U32::Const(2), off(3, 2));
    }
    if p.animation.is_some() {
        write_u32(w, p.duration, U32::Const(0), U32::Const(1), u(8), u(32));
    }
    w.put_bool(p.is_last);
    if !p.is_last {
        w.put(2, 0); // save_as_reference
    }
    write_u32(w, 0, U32::Const(0), u(4), off(16, 5), off(48, 10)); // empty name
    write_restoration_filter(w, p);
    write_u64(w, 0); // FH extensions
}

/// JPEG XL `LoopFilter`. Defaults are gab=true and epf_iters=2 (`all_default`).
fn write_restoration_filter(w: &mut BitWriter, p: ImageParams) {
    let epf = p.epf_iters.min(3);
    let all_default = p.gab && epf == 2;
    w.put_bool(all_default);
    if all_default {
        return;
    }
    w.put_bool(p.gab);
    if p.gab {
        w.put_bool(false); // gab not custom
    }
    w.put(2, epf as u64);
    if epf > 0 {
        if p.vardct {
            w.put_bool(false); // epf_sharp not custom
        }
        w.put_bool(false); // epf_weight not custom
        w.put_bool(false); // epf_sigma not custom
        if !p.vardct {
            w.put_f16(1.0); // epf_sigma_for_modular
        }
    }
    write_u64(w, 0); // RF extensions
}

pub fn write_toc(w: &mut BitWriter, sizes: &[u32]) {
    w.put_bool(false); // no permutation
    w.zero_pad_to_byte();
    for &s in sizes {
        write_toc_size(w, s);
    }
    w.zero_pad_to_byte();
}

/// Modular transforms applied in encoder order (decoder inverts in reverse).
#[derive(Clone, Copy, Debug)]
pub enum ModularTransform {
    /// RCT type 6 (YCoCg-R) starting at `begin_c`.
    RctYCoCg { begin_c: u32 },
    /// Squeeze with `num_sq = 0` (decoder synthesizes default params).
    SqueezeDefault,
    /// Lossless palette: `nb_deltas = 0`, Gradient `d_pred`.
    Palette {
        begin_c: u32,
        num_c: u32,
        nb_colours: u32,
    },
}

/// Modular group header: use global tree, default WP, listed transforms.
pub fn write_modular_header(w: &mut BitWriter, transforms: &[ModularTransform]) {
    write_modular_header_ex(w, transforms, true);
}

/// `use_global_tree = false` so the group carries its own MA tree + histograms.
pub fn write_local_modular_prefix(w: &mut BitWriter) {
    write_modular_header_ex(w, &[], false);
}

fn write_modular_header_ex(
    w: &mut BitWriter,
    transforms: &[ModularTransform],
    use_global_tree: bool,
) {
    w.put_bool(use_global_tree);
    w.put_bool(true); // default WP
    write_u32(
        w,
        transforms.len() as u32,
        U32::Const(0),
        U32::Const(1),
        off(2, 4),
        off(18, 8),
    );
    for tr in transforms {
        match *tr {
            ModularTransform::RctYCoCg { begin_c } => {
                w.put(2, 0); // RCT
                write_u32(w, begin_c, u(3), off(8, 6), off(72, 10), off(1096, 13));
                write_u32(w, 6, U32::Const(6), u(2), off(2, 4), off(10, 6));
            }
            ModularTransform::SqueezeDefault => {
                w.put(2, 2); // Squeeze
                write_u32(w, 0, U32::Const(0), off(1, 4), off(9, 6), off(41, 8));
            }
            ModularTransform::Palette {
                begin_c,
                num_c,
                nb_colours,
            } => {
                w.put(2, 1); // Palette
                write_u32(w, begin_c, u(3), off(8, 6), off(72, 10), off(1096, 13));
                write_u32(
                    w,
                    num_c,
                    U32::Const(1),
                    U32::Const(3),
                    U32::Const(4),
                    off(1, 13),
                );
                write_u32(
                    w,
                    nb_colours,
                    u(8),
                    off(256, 10),
                    off(1280, 12),
                    off(5376, 16),
                );
                write_u32(w, 0, U32::Const(0), off(1, 8), off(257, 10), off(1281, 16));
                w.put(4, 5); // d_pred = Gradient
            }
        }
    }
}
