//! Batched 8×8 inverse DCT using `std::simd`.
//!
//! The 1D kernel is the Loeffler factorization used by libjxl / jxl-rs. The 2D
//! transform loads an 8×8 block as eight `f32x8` row vectors, transposes so the
//! butterflies run lane-parallel, then transposes back and repeats for columns.

use std::simd::StdFloat;

use super::util::{F32x8, apply_rows_8};

const SQRT2: f32 = std::f32::consts::SQRT_2;
// Loeffler constants as published; extra digits document the source values.
#[allow(clippy::excessive_precision)]
const K_COS_4: f32 = 0.541_196_100_146_197_0;
#[allow(clippy::excessive_precision)]
const K_SIN_4: f32 = 1.306_562_964_876_376_4;
#[allow(clippy::excessive_precision)]
const K_SEC_1: f32 = 0.509_795_579_104_159_2;
#[allow(clippy::excessive_precision)]
const K_SEC_2: f32 = 0.601_344_886_935_045_3;
#[allow(clippy::excessive_precision)]
const K_SEC_3: f32 = 0.899_976_223_136_415_6;
#[allow(clippy::excessive_precision)]
const K_SEC_4: f32 = 2.562_915_447_741_505_5;

/// Apply an in-place 8×8 2D inverse DCT to a single block (row-major, 64 samples).
pub fn idct8_2d(block: &mut [f32; 64]) {
    apply_rows_8(block, idct8_1d_lanes);
}

/// Apply [`idct8_2d`] to every contiguous 64-sample block in `blocks`.
///
/// # Panics
/// Panics if `blocks.len()` is not a multiple of 64.
pub fn idct8_2d_blocks(blocks: &mut [f32]) {
    let (chunks, rem) = blocks.as_chunks_mut::<64>();
    assert!(
        rem.is_empty(),
        "IDCT buffer length must be a multiple of 64, got {}",
        blocks.len()
    );
    for chunk in chunks {
        idct8_2d(chunk);
    }
}

/// Eight independent 8-point IDCTs, one per SIMD lane.
fn idct8_1d_lanes(v: [F32x8; 8]) -> [F32x8; 8] {
    let sqrt2 = F32x8::splat(SQRT2);
    let k_cos4 = F32x8::splat(K_COS_4);
    let k_sin4 = F32x8::splat(K_SIN_4);
    let k_sec1 = F32x8::splat(K_SEC_1);
    let k_sec2 = F32x8::splat(K_SEC_2);
    let k_sec3 = F32x8::splat(K_SEC_3);
    let k_sec4 = F32x8::splat(K_SEC_4);

    let v8 = v[0] + v[4];
    let v9 = v[0] - v[4];
    let v10 = v[2] + v[6];
    let v11 = v[2] * sqrt2;
    let v12 = v11 + v10;
    let v13 = v11 - v10;

    let v14 = v12.mul_add(k_cos4, v8);
    let v15 = v8 - v12 * k_cos4;
    let v16 = v13.mul_add(k_sin4, v9);
    let v17 = v9 - v13 * k_sin4;

    let v18 = v[1] + v[3];
    let v19 = v[3] + v[5];
    let v20 = v[5] + v[7];
    let v21 = v[1] * sqrt2;
    let v22 = v21 + v19;
    let v23 = v21 - v19;
    let v24 = v18 + v20;
    let v25 = v18 * sqrt2;
    let v26 = v25 + v24;
    let v27 = v25 - v24;

    let v28 = v26.mul_add(k_cos4, v22);
    let v29 = v22 - v26 * k_cos4;
    let v30 = v27.mul_add(k_sin4, v23);
    let v31 = v23 - v27 * k_sin4;

    let v32 = v28.mul_add(k_sec1, v14);
    let v33 = v14 - v28 * k_sec1;
    let v34 = v30.mul_add(k_sec2, v16);
    let v35 = v16 - v30 * k_sec2;
    let v36 = v31.mul_add(k_sec3, v17);
    let v37 = v17 - v31 * k_sec3;
    let v38 = v29.mul_add(k_sec4, v15);
    let v39 = v15 - v29 * k_sec4;

    [v32, v34, v36, v38, v39, v37, v35, v33]
}

/// Scalar Loeffler 8-point IDCT (reference for tests and odd-sized leftovers).
#[cfg_attr(not(test), allow(dead_code))]
fn idct8_1d_scalar(v: [f32; 8]) -> [f32; 8] {
    let v8 = v[0] + v[4];
    let v9 = v[0] - v[4];
    let v10 = v[2] + v[6];
    let v11 = v[2] * SQRT2;
    let v12 = v11 + v10;
    let v13 = v11 - v10;

    let v14 = v12.mul_add(K_COS_4, v8);
    let v15 = v8 - v12 * K_COS_4;
    let v16 = v13.mul_add(K_SIN_4, v9);
    let v17 = v9 - v13 * K_SIN_4;

    let v18 = v[1] + v[3];
    let v19 = v[3] + v[5];
    let v20 = v[5] + v[7];
    let v21 = v[1] * SQRT2;
    let v22 = v21 + v19;
    let v23 = v21 - v19;
    let v24 = v18 + v20;
    let v25 = v18 * SQRT2;
    let v26 = v25 + v24;
    let v27 = v25 - v24;

    let v28 = v26.mul_add(K_COS_4, v22);
    let v29 = v22 - v26 * K_COS_4;
    let v30 = v27.mul_add(K_SIN_4, v23);
    let v31 = v23 - v27 * K_SIN_4;

    let v32 = v28.mul_add(K_SEC_1, v14);
    let v33 = v14 - v28 * K_SEC_1;
    let v34 = v30.mul_add(K_SEC_2, v16);
    let v35 = v16 - v30 * K_SEC_2;
    let v36 = v31.mul_add(K_SEC_3, v17);
    let v37 = v17 - v31 * K_SEC_3;
    let v38 = v29.mul_add(K_SEC_4, v15);
    let v39 = v15 - v29 * K_SEC_4;

    [v32, v34, v36, v38, v39, v37, v35, v33]
}

#[cfg_attr(not(test), allow(dead_code))]
fn idct8_2d_scalar(block: &mut [f32; 64]) {
    let mut tmp = [0.0f32; 64];
    for row in 0..8 {
        let mut v = [0.0f32; 8];
        v.copy_from_slice(&block[row * 8..row * 8 + 8]);
        let out = idct8_1d_scalar(v);
        tmp[row * 8..row * 8 + 8].copy_from_slice(&out);
    }
    for col in 0..8 {
        let mut v = [0.0f32; 8];
        for row in 0..8 {
            v[row] = tmp[row * 8 + col];
        }
        let out = idct8_1d_scalar(v);
        for row in 0..8 {
            block[row * 8 + col] = out[row];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose8_is_involution() {
        let rows: [F32x8; 8] = core::array::from_fn(|i| {
            F32x8::from_array(core::array::from_fn(|j| (i * 8 + j) as f32))
        });
        let twice = crate::simd::util::transpose8(crate::simd::util::transpose8(rows));
        for (got, want) in twice.iter().zip(rows.iter()) {
            assert_eq!(got.to_array(), want.to_array());
        }
    }

    #[test]
    fn simd_idct_matches_scalar() {
        let mut block = [0.0f32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = (i as f32 * 0.13).sin();
        }
        let mut scalar = block;
        idct8_2d_scalar(&mut scalar);
        idct8_2d(&mut block);
        for (a, b) in scalar.iter().zip(block.iter()) {
            assert!((a - b).abs() < 1e-5, "scalar={a} simd={b}");
        }
    }

    #[test]
    fn idct_blocks_rejects_bad_length() {
        let mut buf = [0.0f32; 63];
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idct8_2d_blocks(&mut buf);
        }));
        assert!(ok.is_err());
    }
}
