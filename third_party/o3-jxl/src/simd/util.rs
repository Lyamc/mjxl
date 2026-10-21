//! Shared portable-SIMD helpers (transpose, load/store of DCT blocks).

use std::simd::prelude::*;

pub(crate) type F32x4 = Simd<f32, 4>;
pub(crate) type F32x8 = Simd<f32, 8>;
pub(crate) type F32x16 = Simd<f32, 16>;

pub(crate) fn transpose4(a: F32x4, b: F32x4, c: F32x4, d: F32x4) -> [F32x4; 4] {
    let (ac0, ac1) = a.interleave(c);
    let (bd0, bd1) = b.interleave(d);
    let (r0, r1) = ac0.interleave(bd0);
    let (r2, r3) = ac1.interleave(bd1);
    [r0, r1, r2, r3]
}

pub(crate) fn lo(v: F32x8) -> F32x4 {
    simd_swizzle!(v, [0, 1, 2, 3])
}

pub(crate) fn hi(v: F32x8) -> F32x4 {
    simd_swizzle!(v, [4, 5, 6, 7])
}

pub(crate) fn join(l: F32x4, h: F32x4) -> F32x8 {
    simd_swizzle!(l, h, [0, 1, 2, 3, 4, 5, 6, 7])
}

pub(crate) fn lo16(v: F32x16) -> F32x8 {
    simd_swizzle!(v, [0, 1, 2, 3, 4, 5, 6, 7])
}

pub(crate) fn hi16(v: F32x16) -> F32x8 {
    simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15])
}

pub(crate) fn join16(l: F32x8, h: F32x8) -> F32x16 {
    simd_swizzle!(l, h, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Independent 8×8 transpose on each half of 16-wide rows (two adjacent blocks).
pub(crate) fn transpose8_x2(v: [F32x16; 8]) -> [F32x16; 8] {
    let lo = transpose8([
        lo16(v[0]),
        lo16(v[1]),
        lo16(v[2]),
        lo16(v[3]),
        lo16(v[4]),
        lo16(v[5]),
        lo16(v[6]),
        lo16(v[7]),
    ]);
    let hi = transpose8([
        hi16(v[0]),
        hi16(v[1]),
        hi16(v[2]),
        hi16(v[3]),
        hi16(v[4]),
        hi16(v[5]),
        hi16(v[6]),
        hi16(v[7]),
    ]);
    [
        join16(lo[0], hi[0]),
        join16(lo[1], hi[1]),
        join16(lo[2], hi[2]),
        join16(lo[3], hi[3]),
        join16(lo[4], hi[4]),
        join16(lo[5], hi[5]),
        join16(lo[6], hi[6]),
        join16(lo[7], hi[7]),
    ]
}

pub(crate) fn transpose8(v: [F32x8; 8]) -> [F32x8; 8] {
    let a = transpose4(lo(v[0]), lo(v[1]), lo(v[2]), lo(v[3]));
    let b = transpose4(hi(v[0]), hi(v[1]), hi(v[2]), hi(v[3]));
    let c = transpose4(lo(v[4]), lo(v[5]), lo(v[6]), lo(v[7]));
    let d = transpose4(hi(v[4]), hi(v[5]), hi(v[6]), hi(v[7]));
    [
        join(a[0], c[0]),
        join(a[1], c[1]),
        join(a[2], c[2]),
        join(a[3], c[3]),
        join(b[0], d[0]),
        join(b[1], d[1]),
        join(b[2], d[2]),
        join(b[3], d[3]),
    ]
}

pub(crate) fn load4(block: &[f32; 16]) -> [F32x4; 4] {
    core::array::from_fn(|i| F32x4::from_slice(&block[i * 4..]))
}

pub(crate) fn store4(block: &mut [f32; 16], rows: [F32x4; 4]) {
    for (i, row) in rows.iter().enumerate() {
        row.copy_to_slice(&mut block[i * 4..]);
    }
}

pub(crate) fn load8(block: &[f32; 64]) -> [F32x8; 8] {
    core::array::from_fn(|i| F32x8::from_slice(&block[i * 8..]))
}

pub(crate) fn store8(block: &mut [f32; 64], rows: [F32x8; 8]) {
    for (i, row) in rows.iter().enumerate() {
        row.copy_to_slice(&mut block[i * 8..]);
    }
}

pub(crate) fn apply_rows_4(block: &mut [f32; 16], f: impl Fn([F32x4; 4]) -> [F32x4; 4]) {
    let rows = load4(block);
    let t = transpose4(rows[0], rows[1], rows[2], rows[3]);
    let t = f(t);
    let rows = transpose4(t[0], t[1], t[2], t[3]);
    let rows = f(rows);
    store4(block, rows);
}

pub(crate) fn apply_rows_8(block: &mut [f32; 64], f: impl Fn([F32x8; 8]) -> [F32x8; 8]) {
    let rows = load8(block);
    let t = f(transpose8(rows));
    let rows = f(transpose8(t));
    store8(block, rows);
}
