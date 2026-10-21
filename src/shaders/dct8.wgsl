// 8×8 DCT-II / DCT-III matching o3-jxl `dct8_2d` / `idct8_factored_2d`.
// One workgroup is one block. The (0,0) thread runs the separable 1-D passes.

const SQRT2: f32 = 1.41421356237;
const SEC4_0: f32 = 0.5411961;
const SEC4_1: f32 = 1.306563;
const SEC8_0: f32 = 0.5097955791041592;
const SEC8_1: f32 = 0.6013448869350453;
const SEC8_2: f32 = 0.8999762231364156;
const SEC8_3: f32 = 2.5629154477415055;

struct Params {
    num_blocks: u32,
    base: u32,
    inverse: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> blk: array<f32, 64>;

fn dct4_fwd(v: array<f32, 4>) -> array<f32, 4> {
    let sum03 = v[0] + v[3];
    let sum12 = v[1] + v[2];
    let tmp0 = (v[0] - v[3]) * SEC4_0;
    let tmp1 = (v[1] - v[2]) * SEC4_1;
    let out0 = (tmp0 + tmp1) * 0.25;
    let out1 = (tmp0 - tmp1) * 0.25;
    return array<f32, 4>(
        (sum03 + sum12) * 0.25,
        out0 * SQRT2 + out1,
        (sum03 - sum12) * 0.25,
        out1,
    );
}

fn dct8_fwd(v: array<f32, 8>) -> array<f32, 8> {
    let input0 = array<f32, 4>(
        (v[0] + v[7]) * 0.5,
        (v[1] + v[6]) * 0.5,
        (v[2] + v[5]) * 0.5,
        (v[3] + v[4]) * 0.5,
    );
    let input1 = array<f32, 4>(
        (v[0] - v[7]) * (SEC8_0 * 0.5),
        (v[1] - v[6]) * (SEC8_1 * 0.5),
        (v[2] - v[5]) * (SEC8_2 * 0.5),
        (v[3] - v[4]) * (SEC8_3 * 0.5),
    );
    let output0 = dct4_fwd(input0);
    var output1 = dct4_fwd(input1);
    output1[0] = output1[0] * SQRT2;
    return array<f32, 8>(
        output0[0],
        output1[0] + output1[1],
        output0[1],
        output1[1] + output1[2],
        output0[2],
        output1[2] + output1[3],
        output0[3],
        output1[3],
    );
}

fn dct4_inv(v: array<f32, 4>) -> array<f32, 4> {
    let tmp0 = v[1] * SQRT2;
    let tmp1 = v[1] + v[3];
    let out0 = (tmp0 + tmp1) * SEC4_0;
    let out1 = (tmp0 - tmp1) * SEC4_1;
    let sum02 = v[0] + v[2];
    let sub02 = v[0] - v[2];
    return array<f32, 4>(sum02 + out0, sub02 + out1, sub02 - out1, sum02 - out0);
}

fn dct8_inv(v: array<f32, 8>) -> array<f32, 8> {
    let input0 = array<f32, 4>(v[0], v[2], v[4], v[6]);
    let input1 = array<f32, 4>(v[1] * SQRT2, v[3] + v[1], v[5] + v[3], v[7] + v[5]);
    let output0 = dct4_inv(input0);
    let output1 = dct4_inv(input1);
    var out: array<f32, 8>;
    let r0 = output1[0] * SEC8_0;
    let r1 = output1[1] * SEC8_1;
    let r2 = output1[2] * SEC8_2;
    let r3 = output1[3] * SEC8_3;
    out[0] = output0[0] + r0;
    out[7] = output0[0] - r0;
    out[1] = output0[1] + r1;
    out[6] = output0[1] - r1;
    out[2] = output0[2] + r2;
    out[5] = output0[2] - r2;
    out[3] = output0[3] + r3;
    out[4] = output0[3] - r3;
    return out;
}

fn xform(v: array<f32, 8>) -> array<f32, 8> {
    if (params.inverse != 0u) {
        return dct8_inv(v);
    }
    return dct8_fwd(v);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let in_range = wid.x < params.num_blocks;
    let b = params.base + wid.x;
    let li = lid.y * 8u + lid.x;
    if (in_range) {
        blk[li] = input[b * 64u + li];
    }
    workgroupBarrier();
    if (in_range && lid.x == 0u && lid.y == 0u) {
        for (var r = 0u; r < 8u; r = r + 1u) {
            var v: array<f32, 8>;
            for (var c = 0u; c < 8u; c = c + 1u) {
                v[c] = blk[r * 8u + c];
            }
            v = xform(v);
            for (var c = 0u; c < 8u; c = c + 1u) {
                blk[r * 8u + c] = v[c];
            }
        }
        for (var c = 0u; c < 8u; c = c + 1u) {
            var v: array<f32, 8>;
            for (var r = 0u; r < 8u; r = r + 1u) {
                v[r] = blk[r * 8u + c];
            }
            v = xform(v);
            for (var r = 0u; r < 8u; r = r + 1u) {
                blk[r * 8u + c] = v[r];
            }
        }
    }
    workgroupBarrier();
    if (in_range) {
        output[b * 64u + li] = blk[li];
    }
}
