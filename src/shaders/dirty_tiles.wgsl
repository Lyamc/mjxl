// Per-tile mean absolute RGB difference. One thread, one 16×16 tile.
// Byte loads assume a little-endian packing of the RGB buffer into u32 words.

struct Params {
    width: u32,
    height: u32,
    tile: u32,
    threshold: u32,
    tw: u32,
    th: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> prev: array<u32>;
@group(0) @binding(1) var<storage, read> curr: array<u32>;
@group(0) @binding(2) var<storage, read_write> mask: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn byte_of(word: u32, i: u32) -> u32 {
    let shift = (i & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn abs_diff(a: u32, b: u32) -> u32 {
    return select(a - b, b - a, b > a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.tw * params.th;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let tx = i % params.tw;
    let ty = i / params.tw;
    let x0 = tx * params.tile;
    let y0 = ty * params.tile;
    let x1 = min(x0 + params.tile, params.width);
    let y1 = min(y0 + params.tile, params.height);

    var sad = 0u;
    var count = 0u;
    for (var y = y0; y < y1; y = y + 1u) {
        for (var x = x0; x < x1; x = x + 1u) {
            let o = (y * params.width + x) * 3u;
            sad = sad + abs_diff(byte_of(prev[o >> 2u], o), byte_of(curr[o >> 2u], o));
            sad = sad + abs_diff(byte_of(prev[(o + 1u) >> 2u], o + 1u), byte_of(curr[(o + 1u) >> 2u], o + 1u));
            sad = sad + abs_diff(byte_of(prev[(o + 2u) >> 2u], o + 2u), byte_of(curr[(o + 2u) >> 2u], o + 2u));
            count = count + 3u;
        }
    }
    let mae = sad / max(count, 1u);
    mask[i] = select(0u, 1u, count > 0u && mae > params.threshold);
}
