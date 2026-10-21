//! JPEG XL DCT-II (forward) and DCT-III (inverse) for 2×2, 4×4, and 8×8.
//!
//! Factorization and scaling match jxl-oxide / libjxl so these can plug into
//! VarDCT. Lane-parallel butterflies: each SIMD lane is an independent column.

use std::simd::Simd;

use super::util::{F32x8, F32x16, apply_rows_4, apply_rows_8, transpose8_x2};

const SEC4_0: f32 = 0.541_196_1;
const SEC4_1: f32 = 1.306_563;
const SQRT2: f32 = std::f32::consts::SQRT_2;

#[allow(clippy::excessive_precision)]
const SEC8: [f32; 4] = [
    0.509_795_579_104_159_2,
    0.601_344_886_935_045_3,
    0.899_976_223_136_415_6,
    2.562_915_447_741_505_5,
];

/// In-place 2×2 DCT-II (row-major 4 samples).
pub fn dct2_2d(block: &mut [f32; 4]) {
    dct2_2d_dir(block, true);
}

/// In-place 2×2 DCT-III (inverse).
pub fn idct2_2d(block: &mut [f32; 4]) {
    dct2_2d_dir(block, false);
}

fn dct2_2d_dir(block: &mut [f32; 4], forward: bool) {
    let mul = if forward { 0.5 } else { 1.0 };
    // Rows.
    for row in block.as_chunks_mut::<2>().0 {
        let (a, b) = (row[0], row[1]);
        row[0] = (a + b) * mul;
        row[1] = (a - b) * mul;
    }
    // Columns.
    for col in 0..2 {
        let a = block[col];
        let b = block[2 + col];
        block[col] = (a + b) * mul;
        block[2 + col] = (a - b) * mul;
    }
}

/// Apply [`dct2_2d`] / [`idct2_2d`] to every contiguous 4-sample block.
pub fn dct2_2d_blocks(blocks: &mut [f32], forward: bool) {
    let (chunks, rem) = blocks.as_chunks_mut::<4>();
    assert!(
        rem.is_empty(),
        "2×2 DCT buffer length must be a multiple of 4, got {}",
        blocks.len()
    );
    for chunk in chunks {
        if forward {
            dct2_2d(chunk);
        } else {
            idct2_2d(chunk);
        }
    }
}

/// In-place 4×4 DCT-II.
pub fn dct4_2d(block: &mut [f32; 16]) {
    apply_rows_4(block, dct4_fwd_lanes);
}

/// In-place 4×4 DCT-III (inverse).
pub fn idct4_2d(block: &mut [f32; 16]) {
    apply_rows_4(block, dct4_inv_lanes);
}

/// Apply 4×4 DCT/IDCT to every contiguous 16-sample block.
pub fn dct4_2d_blocks(blocks: &mut [f32], forward: bool) {
    let (chunks, rem) = blocks.as_chunks_mut::<16>();
    assert!(
        rem.is_empty(),
        "4×4 DCT buffer length must be a multiple of 16, got {}",
        blocks.len()
    );
    for chunk in chunks {
        if forward {
            dct4_2d(chunk);
        } else {
            idct4_2d(chunk);
        }
    }
}

/// libjxl `WcMultipliers<16>` = 1/(2 cos((k+½)π/16)).
#[allow(clippy::excessive_precision)]
const SEC16: [f32; 8] = [
    0.502_419_286_188_155_7,
    0.522_498_614_939_688_9,
    0.566_944_034_816_357_7,
    0.646_821_783_359_990_1,
    0.788_154_623_451_250_2,
    1.060_677_685_990_347,
    1.722_447_098_238_334_2,
    5.101_148_618_689_155,
];

/// 1-D 8-point DCT-II matching [`dct8_fwd_lanes`].
fn dct8_1d(v: [f32; 8]) -> [f32; 8] {
    let input0 = [
        (v[0] + v[7]) * 0.5,
        (v[1] + v[6]) * 0.5,
        (v[2] + v[5]) * 0.5,
        (v[3] + v[4]) * 0.5,
    ];
    let input1 = [
        (v[0] - v[7]) * (SEC8[0] * 0.5),
        (v[1] - v[6]) * (SEC8[1] * 0.5),
        (v[2] - v[5]) * (SEC8[2] * 0.5),
        (v[3] - v[4]) * (SEC8[3] * 0.5),
    ];
    let output0 = dct4_1d_fwd(input0);
    let mut output1 = dct4_1d_fwd(input1);
    output1[0] *= SQRT2;
    [
        output0[0],
        output1[0] + output1[1],
        output0[1],
        output1[1] + output1[2],
        output0[2],
        output1[2] + output1[3],
        output0[3],
        output1[3],
    ]
}

fn dct4_1d_fwd(v: [f32; 4]) -> [f32; 4] {
    let sum03 = v[0] + v[3];
    let sum12 = v[1] + v[2];
    let tmp0 = (v[0] - v[3]) * SEC4_0;
    let tmp1 = (v[1] - v[2]) * SEC4_1;
    let out0 = (tmp0 + tmp1) * 0.25;
    let out1 = (tmp0 - tmp1) * 0.25;
    [
        (sum03 + sum12) * 0.25,
        out0 * SQRT2 + out1,
        (sum03 - sum12) * 0.25,
        out1,
    ]
}

/// 1-D 16-point DCT-II (jxl-oxide recursive split, libjxl `ComputeScaledDCT<16>`).
fn dct16_1d(v: [f32; 16]) -> [f32; 16] {
    let mut in0 = [0f32; 8];
    let mut in1 = [0f32; 8];
    for i in 0..8 {
        in0[i] = (v[i] + v[15 - i]) * 0.5;
        in1[i] = (v[i] - v[15 - i]) * 0.5 * SEC16[i];
    }
    let out0 = dct8_1d(in0);
    let mut out1 = dct8_1d(in1);
    out1[0] *= SQRT2;
    for i in 0..7 {
        out1[i] += out1[i + 1];
    }
    let mut y = [0f32; 16];
    for i in 0..8 {
        y[i * 2] = out0[i];
        y[i * 2 + 1] = out1[i];
    }
    y
}

fn dct4_1d_inv(v: [f32; 4]) -> [f32; 4] {
    let tmp0 = v[1] * SQRT2;
    let tmp1 = v[1] + v[3];
    let out0 = (tmp0 + tmp1) * SEC4_0;
    let out1 = (tmp0 - tmp1) * SEC4_1;
    let sum02 = v[0] + v[2];
    let sub02 = v[0] - v[2];
    [sum02 + out0, sub02 + out1, sub02 - out1, sum02 - out0]
}

fn dct8_1d_inv(v: [f32; 8]) -> [f32; 8] {
    let input0 = [v[0], v[2], v[4], v[6]];
    let input1 = [v[1] * SQRT2, v[3] + v[1], v[5] + v[3], v[7] + v[5]];
    let output0 = dct4_1d_inv(input0);
    let output1 = dct4_1d_inv(input1);
    let mut out = [0f32; 8];
    for (idx, &sec) in SEC8.iter().enumerate() {
        let r = output1[idx] * sec;
        out[idx] = output0[idx] + r;
        out[7 - idx] = output0[idx] - r;
    }
    out
}

/// Inverse of [`dct16_1d`] (jxl-oxide recursive IDCT).
fn dct16_1d_inv(v: [f32; 16]) -> [f32; 16] {
    let mut in0 = [0f32; 8];
    let mut in1 = [0f32; 8];
    for i in 0..8 {
        in0[i] = v[i * 2];
        in1[i] = v[i * 2 + 1];
    }
    for i in 1..8 {
        in1[8 - i] += in1[8 - i - 1];
    }
    in1[0] *= SQRT2;
    let out0 = dct8_1d_inv(in0);
    let mut out1 = dct8_1d_inv(in1);
    for i in 0..8 {
        out1[i] *= SEC16[i];
    }
    let mut y = [0f32; 16];
    for i in 0..8 {
        y[i] = out0[i] + out1[i];
        y[15 - i] = out0[i] - out1[i];
    }
    y
}

/// Inverse of [`dct16_2d`].
pub fn idct16_2d(block: &mut [f32; 256]) {
    let mut rows = [[0f32; 16]; 16];
    for y in 0..16 {
        rows[y].copy_from_slice(&block[y * 16..y * 16 + 16]);
        rows[y] = dct16_1d_inv(rows[y]);
    }
    let mut t = [[0f32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            t[x][y] = rows[y][x];
        }
    }
    for row in &mut t {
        *row = dct16_1d_inv(*row);
    }
    for y in 0..16 {
        for x in 0..16 {
            block[y * 16 + x] = t[x][y];
        }
    }
}

/// In-place 16×16 DCT-II. Same row/column order as [`dct8_2d`].
pub fn dct16_2d(block: &mut [f32; 256]) {
    let mut rows = [[0f32; 16]; 16];
    for y in 0..16 {
        rows[y].copy_from_slice(&block[y * 16..y * 16 + 16]);
        rows[y] = dct16_1d(rows[y]);
    }
    // Transpose.
    let mut t = [[0f32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            t[x][y] = rows[y][x];
        }
    }
    for row in &mut t {
        *row = dct16_1d(*row);
    }
    for y in 0..16 {
        for x in 0..16 {
            block[y * 16 + x] = t[x][y];
        }
    }
}

/// Load a 16×16 from a strided plane and run [`dct16_2d`].
pub fn dct16_fwd_from_plane(plane: &[f32], stride: usize, x: usize, y0: usize) -> [f32; 256] {
    let mut block = [0f32; 256];
    for row in 0..16 {
        let src = (y0 + row) * stride + x;
        block[row * 16..row * 16 + 16].copy_from_slice(&plane[src..src + 16]);
    }
    dct16_2d(&mut block);
    block
}

/// libjxl `WcMultipliers<32>`.
#[allow(clippy::excessive_precision)]
const SEC32: [f32; 16] = [
    0.500_602_998_235_196_3,
    0.505_470_959_897_543_6,
    0.515_447_309_922_624_6,
    0.531_042_591_089_784_1,
    0.553_103_896_034_444_5,
    0.582_934_968_206_133_9,
    0.622_504_123_035_664_8,
    0.674_808_341_455_005_7,
    0.744_536_271_002_298_6,
    0.839_349_645_415_526_8,
    0.972_568_237_861_960_8,
    1.169_439_933_432_884_7,
    1.484_164_616_314_166_2,
    2.057_781_009_953_411,
    3.407_608_418_468_719,
    10.190_008_123_548_033,
];

fn dct32_1d(v: [f32; 32]) -> [f32; 32] {
    let mut in0 = [0f32; 16];
    let mut in1 = [0f32; 16];
    for i in 0..16 {
        in0[i] = (v[i] + v[31 - i]) * 0.5;
        in1[i] = (v[i] - v[31 - i]) * 0.5 * SEC32[i];
    }
    let out0 = dct16_1d(in0);
    let mut out1 = dct16_1d(in1);
    out1[0] *= SQRT2;
    for i in 0..15 {
        out1[i] += out1[i + 1];
    }
    let mut y = [0f32; 32];
    for i in 0..16 {
        y[i * 2] = out0[i];
        y[i * 2 + 1] = out1[i];
    }
    y
}

fn dct32_1d_inv(v: [f32; 32]) -> [f32; 32] {
    let mut in0 = [0f32; 16];
    let mut in1 = [0f32; 16];
    for i in 0..16 {
        in0[i] = v[i * 2];
        in1[i] = v[i * 2 + 1];
    }
    for i in 1..16 {
        in1[16 - i] += in1[16 - i - 1];
    }
    in1[0] *= SQRT2;
    let out0 = dct16_1d_inv(in0);
    let mut out1 = dct16_1d_inv(in1);
    for i in 0..16 {
        out1[i] *= SEC32[i];
    }
    let mut y = [0f32; 32];
    for i in 0..16 {
        y[i] = out0[i] + out1[i];
        y[31 - i] = out0[i] - out1[i];
    }
    y
}

/// Inverse of [`dct32_2d`].
pub fn idct32_2d(block: &mut [f32; 1024]) {
    let mut rows = [[0f32; 32]; 32];
    for y in 0..32 {
        rows[y].copy_from_slice(&block[y * 32..y * 32 + 32]);
        rows[y] = dct32_1d_inv(rows[y]);
    }
    let mut t = [[0f32; 32]; 32];
    for y in 0..32 {
        for x in 0..32 {
            t[x][y] = rows[y][x];
        }
    }
    for row in &mut t {
        *row = dct32_1d_inv(*row);
    }
    for y in 0..32 {
        for x in 0..32 {
            block[y * 32 + x] = t[x][y];
        }
    }
}

/// In-place 32×32 DCT-II.
pub fn dct32_2d(block: &mut [f32; 1024]) {
    let mut rows = [[0f32; 32]; 32];
    for y in 0..32 {
        rows[y].copy_from_slice(&block[y * 32..y * 32 + 32]);
        rows[y] = dct32_1d(rows[y]);
    }
    let mut t = [[0f32; 32]; 32];
    for y in 0..32 {
        for x in 0..32 {
            t[x][y] = rows[y][x];
        }
    }
    for row in &mut t {
        *row = dct32_1d(*row);
    }
    for y in 0..32 {
        for x in 0..32 {
            block[y * 32 + x] = t[x][y];
        }
    }
}

pub fn dct32_fwd_from_plane(plane: &[f32], stride: usize, x: usize, y0: usize) -> [f32; 1024] {
    let mut block = [0f32; 1024];
    for row in 0..32 {
        let src = (y0 + row) * stride + x;
        block[row * 32..row * 32 + 32].copy_from_slice(&plane[src..src + 32]);
    }
    dct32_2d(&mut block);
    block
}

/// Rectangular DCT: `h` rows of `w` (row-major). Used for DCT16×8 / DCT8×16.
pub fn dct_rect(block: &mut [f32], w: usize, h: usize) {
    debug_assert_eq!(block.len(), w * h);
    match (w, h) {
        (16, 16) => {
            let mut b = [0f32; 256];
            b.copy_from_slice(block);
            dct16_2d(&mut b);
            block.copy_from_slice(&b);
        }
        (32, 32) => {
            let mut b = [0f32; 1024];
            b.copy_from_slice(block);
            dct32_2d(&mut b);
            block.copy_from_slice(&b);
        }
        (8, 16) => {
            for y in 0..16 {
                let mut r = [0f32; 8];
                r.copy_from_slice(&block[y * 8..y * 8 + 8]);
                let r = dct8_1d(r);
                block[y * 8..y * 8 + 8].copy_from_slice(&r);
            }
            for x in 0..8 {
                let mut c = [0f32; 16];
                for y in 0..16 {
                    c[y] = block[y * 8 + x];
                }
                c = dct16_1d(c);
                for y in 0..16 {
                    block[y * 8 + x] = c[y];
                }
            }
        }
        (16, 8) => {
            for y in 0..8 {
                let mut r = [0f32; 16];
                r.copy_from_slice(&block[y * 16..y * 16 + 16]);
                r = dct16_1d(r);
                block[y * 16..y * 16 + 16].copy_from_slice(&r);
            }
            for x in 0..16 {
                let mut c = [0f32; 8];
                for y in 0..8 {
                    c[y] = block[y * 16 + x];
                }
                c = dct8_1d(c);
                for y in 0..8 {
                    block[y * 16 + x] = c[y];
                }
            }
        }
        (8, 4) => {
            for y in 0..4 {
                let mut r = [0f32; 8];
                r.copy_from_slice(&block[y * 8..y * 8 + 8]);
                let r = dct8_1d(r);
                block[y * 8..y * 8 + 8].copy_from_slice(&r);
            }
            for x in 0..8 {
                let mut c = [0f32; 4];
                for y in 0..4 {
                    c[y] = block[y * 8 + x];
                }
                c = dct4_1d_fwd(c);
                for y in 0..4 {
                    block[y * 8 + x] = c[y];
                }
            }
        }
        (4, 8) => {
            for y in 0..8 {
                let mut r = [0f32; 4];
                r.copy_from_slice(&block[y * 4..y * 4 + 4]);
                let r = dct4_1d_fwd(r);
                block[y * 4..y * 4 + 4].copy_from_slice(&r);
            }
            for x in 0..4 {
                let mut c = [0f32; 8];
                for y in 0..8 {
                    c[y] = block[y * 4 + x];
                }
                c = dct8_1d(c);
                for y in 0..8 {
                    block[y * 4 + x] = c[y];
                }
            }
        }
        _ => panic!("unsupported DCT {w}x{h}"),
    }
}

/// Inverse of [`dct_rect`].
pub fn idct_rect(block: &mut [f32], w: usize, h: usize) {
    debug_assert_eq!(block.len(), w * h);
    match (w, h) {
        (16, 16) => {
            let mut b = [0f32; 256];
            b.copy_from_slice(block);
            idct16_2d(&mut b);
            block.copy_from_slice(&b);
        }
        (32, 32) => {
            let mut b = [0f32; 1024];
            b.copy_from_slice(block);
            idct32_2d(&mut b);
            block.copy_from_slice(&b);
        }
        (8, 16) => {
            for x in 0..8 {
                let mut c = [0f32; 16];
                for y in 0..16 {
                    c[y] = block[y * 8 + x];
                }
                c = dct16_1d_inv(c);
                for y in 0..16 {
                    block[y * 8 + x] = c[y];
                }
            }
            for y in 0..16 {
                let mut r = [0f32; 8];
                r.copy_from_slice(&block[y * 8..y * 8 + 8]);
                let r = dct8_1d_inv(r);
                block[y * 8..y * 8 + 8].copy_from_slice(&r);
            }
        }
        (16, 8) => {
            for x in 0..16 {
                let mut c = [0f32; 8];
                for y in 0..8 {
                    c[y] = block[y * 16 + x];
                }
                c = dct8_1d_inv(c);
                for y in 0..8 {
                    block[y * 16 + x] = c[y];
                }
            }
            for y in 0..8 {
                let mut r = [0f32; 16];
                r.copy_from_slice(&block[y * 16..y * 16 + 16]);
                r = dct16_1d_inv(r);
                block[y * 16..y * 16 + 16].copy_from_slice(&r);
            }
        }
        (8, 4) => {
            for x in 0..8 {
                let mut c = [0f32; 4];
                for y in 0..4 {
                    c[y] = block[y * 8 + x];
                }
                c = dct4_1d_inv(c);
                for y in 0..4 {
                    block[y * 8 + x] = c[y];
                }
            }
            for y in 0..4 {
                let mut r = [0f32; 8];
                r.copy_from_slice(&block[y * 8..y * 8 + 8]);
                let r = dct8_1d_inv(r);
                block[y * 8..y * 8 + 8].copy_from_slice(&r);
            }
        }
        (4, 8) => {
            for x in 0..4 {
                let mut c = [0f32; 8];
                for y in 0..8 {
                    c[y] = block[y * 4 + x];
                }
                c = dct8_1d_inv(c);
                for y in 0..8 {
                    block[y * 4 + x] = c[y];
                }
            }
            for y in 0..8 {
                let mut r = [0f32; 4];
                r.copy_from_slice(&block[y * 4..y * 4 + 4]);
                let r = dct4_1d_inv(r);
                block[y * 4..y * 4 + 4].copy_from_slice(&r);
            }
        }
        _ => panic!("unsupported IDCT {w}x{h}"),
    }
}

pub fn dct_rect_from_plane(
    plane: &[f32],
    stride: usize,
    x: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Vec<f32> {
    let mut block = vec![0f32; w * h];
    for row in 0..h {
        let src = (y0 + row) * stride + x;
        block[row * w..row * w + w].copy_from_slice(&plane[src..src + w]);
    }
    dct_rect(&mut block, w, h);
    block
}

/// In-place 8×8 DCT-II (JPEG XL / jxl-oxide scaling).
pub fn dct8_2d(block: &mut [f32; 64]) {
    let out = dct8_fwd_from_rows(super::util::load8(block));
    super::util::store8(block, out);
}

/// Load an 8×8 from a strided plane and run [`dct8_fwd_from_rows`].
pub fn dct8_fwd_from_plane(plane: &[f32], stride: usize, x: usize, y0: usize) -> [Simd<f32, 8>; 8] {
    dct8_fwd_from_rows(core::array::from_fn(|row| {
        F32x8::from_slice(&plane[(y0 + row) * stride + x..])
    }))
}

/// 8×8 DCT-II on 8 SIMD rows (lanes = columns). Stays in registers.
pub fn dct8_fwd_from_rows(rows: [Simd<f32, 8>; 8]) -> [Simd<f32, 8>; 8] {
    let t = dct8_fwd_lanes(super::util::transpose8(rows));
    dct8_fwd_lanes(super::util::transpose8(t))
}

/// Two adjacent 8×8 DCTs: lanes 0..7 are block 0, 8..15 are block 1.
pub fn dct8_fwd_from_plane_2(
    plane: &[f32],
    stride: usize,
    x: usize,
    y0: usize,
) -> [Simd<f32, 16>; 8] {
    dct8_fwd_from_rows16(core::array::from_fn(|row| {
        F32x16::from_slice(&plane[(y0 + row) * stride + x..])
    }))
}

fn dct8_fwd_from_rows16(rows: [Simd<f32, 16>; 8]) -> [Simd<f32, 16>; 8] {
    let t = dct8_fwd_lanes(transpose8_x2(rows));
    dct8_fwd_lanes(transpose8_x2(t))
}

/// In-place 8×8 DCT-III using the same factorization as [`dct8_2d`].
///
/// [`super::idct8_2d`] is a Loeffler IDCT of the same transform; this variant
/// is the exact inverse of [`dct8_2d`].
pub fn idct8_factored_2d(block: &mut [f32; 64]) {
    apply_rows_8(block, dct8_inv_lanes);
}

/// Apply 8×8 DCT-II to every contiguous 64-sample block.
pub fn dct8_2d_blocks(blocks: &mut [f32]) {
    let (chunks, rem) = blocks.as_chunks_mut::<64>();
    assert!(
        rem.is_empty(),
        "8×8 DCT buffer length must be a multiple of 64, got {}",
        blocks.len()
    );
    for chunk in chunks {
        dct8_2d(chunk);
    }
}

fn dct4_fwd_lanes<S>(v: [S; 4]) -> [S; 4]
where
    S: Copy + core::ops::Add<Output = S> + core::ops::Sub<Output = S> + core::ops::Mul<Output = S>,
    f32: BroadcastTo<S>,
{
    let sec0 = broadcast(SEC4_0);
    let sec1 = broadcast(SEC4_1);
    let quarter = broadcast(0.25);
    let sqrt2 = broadcast(SQRT2);

    let sum03 = v[0] + v[3];
    let sum12 = v[1] + v[2];
    let tmp0 = (v[0] - v[3]) * sec0;
    let tmp1 = (v[1] - v[2]) * sec1;
    let out0 = (tmp0 + tmp1) * quarter;
    let out1 = (tmp0 - tmp1) * quarter;
    [
        (sum03 + sum12) * quarter,
        out0 * sqrt2 + out1,
        (sum03 - sum12) * quarter,
        out1,
    ]
}

fn dct4_inv_lanes<S>(v: [S; 4]) -> [S; 4]
where
    S: Copy + core::ops::Add<Output = S> + core::ops::Sub<Output = S> + core::ops::Mul<Output = S>,
    f32: BroadcastTo<S>,
{
    let sec0 = broadcast(SEC4_0);
    let sec1 = broadcast(SEC4_1);
    let sqrt2 = broadcast(SQRT2);

    let tmp0 = v[1] * sqrt2;
    let tmp1 = v[1] + v[3];
    let out0 = (tmp0 + tmp1) * sec0;
    let out1 = (tmp0 - tmp1) * sec1;
    let sum02 = v[0] + v[2];
    let sub02 = v[0] - v[2];
    [sum02 + out0, sub02 + out1, sub02 - out1, sum02 - out0]
}

fn dct8_fwd_lanes<const N: usize>(v: [Simd<f32, N>; 8]) -> [Simd<f32, N>; 8] {
    type S<const M: usize> = Simd<f32, M>;
    let half = S::<N>::splat(0.5);
    let input0 = [
        (v[0] + v[7]) * half,
        (v[1] + v[6]) * half,
        (v[2] + v[5]) * half,
        (v[3] + v[4]) * half,
    ];
    let input1 = [
        (v[0] - v[7]) * S::<N>::splat(SEC8[0] * 0.5),
        (v[1] - v[6]) * S::<N>::splat(SEC8[1] * 0.5),
        (v[2] - v[5]) * S::<N>::splat(SEC8[2] * 0.5),
        (v[3] - v[4]) * S::<N>::splat(SEC8[3] * 0.5),
    ];
    let output0 = dct4_fwd_lanes(input0);
    let mut output1 = dct4_fwd_lanes(input1);
    output1[0] *= S::<N>::splat(SQRT2);
    [
        output0[0],
        output1[0] + output1[1],
        output0[1],
        output1[1] + output1[2],
        output0[2],
        output1[2] + output1[3],
        output0[3],
        output1[3],
    ]
}

fn dct8_inv_lanes(v: [F32x8; 8]) -> [F32x8; 8] {
    let input0 = [v[0], v[2], v[4], v[6]];
    let input1 = [
        v[1] * F32x8::splat(SQRT2),
        v[3] + v[1],
        v[5] + v[3],
        v[7] + v[5],
    ];
    let output0 = dct4_inv_lanes(input0);
    let output1 = dct4_inv_lanes(input1);
    let mut out = [F32x8::splat(0.0); 8];
    for (idx, &sec) in SEC8.iter().enumerate() {
        let r = output1[idx] * F32x8::splat(sec);
        out[idx] = output0[idx] + r;
        out[7 - idx] = output0[idx] - r;
    }
    out
}

/// Trait so `dct4_*_lanes` can run on both `f32x4` and `f32x8`.
trait BroadcastTo<S> {
    fn splat_to(self) -> S;
}

impl<const N: usize> BroadcastTo<Simd<f32, N>> for f32 {
    fn splat_to(self) -> Simd<f32, N> {
        Simd::splat(self)
    }
}

fn broadcast<S>(x: f32) -> S
where
    f32: BroadcastTo<S>,
{
    x.splat_to()
}

/// Horizontal 4-point DCT on a single `f32x4` (one row). Used by tests.
#[cfg(test)]
fn dct4_vec_roundtrip() {
    let mut block = [
        1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 3.0, 0.0, 4.0, 1.0, 2.0, 2.0,
    ];
    let orig = block;
    dct4_2d(&mut block);
    idct4_2d(&mut block);
    for (a, b) in orig.iter().zip(block.iter()) {
        assert!((a - b).abs() < 1e-4, "orig={a} got={b}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::idct::idct8_2d;
    use crate::simd::util::{F32x4, transpose4};

    fn fill<const N: usize>() -> [f32; N] {
        core::array::from_fn(|i| ((i as f32) * 0.17).sin() * 3.0)
    }

    fn assert_close<const N: usize>(a: &[f32; N], b: &[f32; N], eps: f32) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < eps, "lhs={x} rhs={y}");
        }
    }

    #[test]
    fn dct2_roundtrip() {
        let orig = fill::<4>();
        let mut b = orig;
        dct2_2d(&mut b);
        idct2_2d(&mut b);
        assert_close(&orig, &b, 1e-5);
    }

    #[test]
    fn dct4_roundtrip() {
        dct4_vec_roundtrip();
        let orig = fill::<16>();
        let mut b = orig;
        dct4_2d(&mut b);
        idct4_2d(&mut b);
        assert_close(&orig, &b, 1e-4);
    }

    #[test]
    fn dct4x8_rect_roundtrip() {
        let orig = fill::<32>();
        let mut b = orig.to_vec();
        dct_rect(&mut b, 8, 4);
        idct_rect(&mut b, 8, 4);
        for (a, c) in orig.iter().zip(b.iter()) {
            assert!((a - c).abs() < 1e-4, "lhs={a} rhs={c}");
        }
        let orig = fill::<32>();
        let mut b = orig.to_vec();
        dct_rect(&mut b, 4, 8);
        idct_rect(&mut b, 4, 8);
        for (a, c) in orig.iter().zip(b.iter()) {
            assert!((a - c).abs() < 1e-4, "lhs={a} rhs={c}");
        }
    }

    #[test]
    fn dct16_roundtrip() {
        let orig = fill::<256>();
        let mut b = orig;
        dct16_2d(&mut b);
        idct16_2d(&mut b);
        assert_close(&orig, &b, 2e-4);
    }

    #[test]
    fn dct32_roundtrip() {
        let orig = fill::<1024>();
        let mut b = orig;
        dct32_2d(&mut b);
        idct32_2d(&mut b);
        assert_close(&orig, &b, 4e-4);
    }

    #[test]
    fn dct16_dc_is_mean() {
        let mut b = [0.4f32; 256];
        dct16_2d(&mut b);
        assert!((b[0] - 0.4).abs() < 1e-5, "{}", b[0]);
        for v in b.iter().skip(1) {
            assert!(v.abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn dct32_dc_is_mean() {
        let mut b = [0.25f32; 1024];
        dct32_2d(&mut b);
        assert!((b[0] - 0.25).abs() < 1e-5, "{}", b[0]);
        for v in b.iter().skip(1) {
            assert!(v.abs() < 1e-4, "{v}");
        }
    }

    #[test]
    fn dct8_roundtrip() {
        let orig = fill::<64>();
        let mut b = orig;
        dct8_2d(&mut b);
        idct8_factored_2d(&mut b);
        assert_close(&orig, &b, 2e-4);
    }

    #[test]
    fn dct8_two_block_matches_singles() {
        let mut plane = [0f32; 8 * 16];
        for y in 0..8 {
            for x in 0..16 {
                plane[y * 16 + x] = ((x + y) as f32) * 0.13 - 0.7;
            }
        }
        let pair = dct8_fwd_from_plane_2(&plane, 16, 0, 0);
        let a = dct8_fwd_from_plane(&plane, 16, 0, 0);
        let b = dct8_fwd_from_plane(&plane, 16, 8, 0);
        for row in 0..8 {
            let p = pair[row].to_array();
            let aa = a[row].to_array();
            let bb = b[row].to_array();
            for k in 0..8 {
                assert!((p[k] - aa[k]).abs() < 1e-5, "b0 r{row} c{k}");
                assert!((p[k + 8] - bb[k]).abs() < 1e-5, "b1 r{row} c{k}");
            }
        }
    }

    #[test]
    fn dct8_then_loeffler_idct_roundtrip() {
        let orig = fill::<64>();
        let mut b = orig;
        dct8_2d(&mut b);
        idct8_2d(&mut b);
        assert_close(&orig, &b, 2e-4);
    }

    #[test]
    fn dct4_lanes_match_scalar_row() {
        let row = [0.5f32, -1.0, 2.0, 0.25];
        let mut block = [0.0f32; 16];
        block[..4].copy_from_slice(&row);
        // Isolate a single 1D pass: apply_rows_4 does 2D. Compare via 4 independent
        // copies of the same row so 2D IDCT of DC-in-one-row is still defined.
        let v = F32x4::from_array(row);
        let lanes = [v, v, v, v];
        let t = transpose4(lanes[0], lanes[1], lanes[2], lanes[3]);
        let fwd = dct4_fwd_lanes(t);
        let back = dct4_inv_lanes(fwd);
        let out = transpose4(back[0], back[1], back[2], back[3]);
        assert_close(&out[0].to_array(), &row, 1e-4);
    }
}
