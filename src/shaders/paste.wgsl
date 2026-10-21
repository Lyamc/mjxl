// Paste a packed RGB rect into an RGBA8 canvas (one u32 per pixel, little-endian).

struct Params {
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> canvas: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_of(word: u32, i: u32) -> u32 {
    let shift = (i & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.w * params.h;
    let i = gid.x;
    if (i >= n || params.w == 0u) {
        return;
    }
    let lx = i % params.w;
    let ly = i / params.w;
    let di = (params.y + ly) * params.canvas_w + (params.x + lx);
    let o = i * 3u;
    let r = byte_of(src[o >> 2u], o);
    let g = byte_of(src[(o + 1u) >> 2u], o + 1u);
    let b = byte_of(src[(o + 2u) >> 2u], o + 2u);
    canvas[di] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
