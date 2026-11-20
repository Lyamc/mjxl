# Modular performance work — measurements

Test clip: 16 × 960×540 pan (8.294 MP) cropped from a 1920×1080 photo,
`-d 1 -e 2`. Built with `cargo +nightly-x86_64-pc-windows-msvc build --release`
(the default gnu toolchain on this machine has a broken linker).

## Microbenchmark (`mjxl bench`, in-process, median after warmup)

With mimalloc as the global allocator:

```
$ mjxl bench -m modular -d 1 -e 2 --repeats 9 pan.y4m
Modular d=1 e=2 8294400 pixels 9 reps
  encode     75.6 ms   109.7 MP/s   decode     42.4 ms   195.6 MP/s   bytes 671356

$ mjxl bench -m vardct -d 1 -e 2 --repeats 5 pan.y4m
Vardct d=1 e=2 8294400 pixels 5 reps
  encode     18.9 ms   438.1 MP/s   decode     23.6 ms   351.4 MP/s   bytes 686908
```

Earlier in the same session, with the system allocator: modular 89 MP/s
encode / 162 MP/s decode, VarDCT 262 MP/s / 254 MP/s. mimalloc was worth
~1.3x on modular and ~1.7x on VarDCT. Timings on this desktop vary with
background load; the numbers above are the median of repeated runs.

## Before → after (same clip, `compare` harness)

| row | before | after | change |
|---|---|---|---|
| modular intra encode | 235 ms / 35.2 MP/s | 69–79 ms / 106–120 MP/s | **3× faster** |
| modular delta encode | ~653 ms / 12.7 MP/s | ~75 ms / 111 MP/s | **8.7× faster** |
| modular decode | ~49 ms / 169 MP/s | ~38–46 ms / 180–217 MP/s | **up to 1.3× faster** |
| bytes / PSNR | 671356 / 48.13 dB | 671356 / 48.13 dB | identical |

MJPEG 4:4:4 `-q:v 3` on the same clip: ~45 ms encode (~182–206 MP/s), ~42 ms
decode (~193–217 MP/s), 587228 bytes, 50.38 dB. Modular decode is now in the
same band as (and in several runs ahead of) MJPEG decode.

## Changes

- `third_party/o3-jxl/src/encoder/mod.rs`: `ResidualCache` computes each
  channel's gradient/zero residuals once instead of three times; global
  histograms reuse the raw split-tree contexts during token emission; skip a
  wasted per-channel histogram pass on the global-hist path.
- `third_party/o3-jxl/src/encoder/ans.rs`: one flat reverse map per ANS
  histogram instead of 256 inner `Vec`s; constant-cluster token fast path;
  `AnsToken` packed from 16 to 8 bytes.
- `third_party/o3-jxl/src/encoder/squeeze.rs`: 8-wide SIMD `tendency_i32x8`
  for the horizontal and vertical squeeze residual loops.
- `third_party/o3-jxl/src/simd/color.rs`: do not split XYB conversion into
  nested Rayon row tasks when already running on a Rayon worker (frame-level
  parallelism). This alone removed ~40–70 ms from a 16-frame clip.
- `src/temporal.rs`: analyse all frame pairs up front in parallel; when every
  frame is dirty, reuse the parallel all-intra path instead of serial full
  encodes. Output bitstream is identical.
- `encoder/ans.rs`: pack each ANS scratch bit-chunk into one `u64` instead of
  a 16-byte `(u64, u8)` tuple, halving the per-frame scratch buffer.
- `Cargo.toml` / `src/main.rs`: mimalloc global allocator. Frame-parallel
  encoding and decoding allocate hundreds of short-lived buffers per frame;
  this was worth ~1.3× on modular and made decode competitive with MJPEG.
- Added `mjxl bench` subcommand for stable median-of-N in-process timing.

### LZ77 for lossy modular — kept, large win
`encoder/mod.rs` now enables LZ77 at all efforts (`lz77: effort >= 3 ||
distance > 0.0`). The squeeze HF planes have long zero runs, so this cuts
both token count and rate. `LZ77_MIN_LENGTH` raised from 7 to 128 (the
encoder tolerates up to ~258): longer runs only are RLE'd, which further
reduces distance-token overhead. Bitstream verified with both jxl-oxide and
`djxl`.

| metric | before | after | MJPEG 444 |
|---|---|---|---|
| encode | 118 MP/s | **167 MP/s** | 181 MP/s |
| decode | 213 MP/s | 203 MP/s | 209 MP/s |
| bytes | 671356 | **647792** | 587228 |
| PSNR | 48.13 dB | 48.13 dB | 50.38 dB |

Encode gap narrowed from ~1.7× to ~1.09×; bitrate improved 3.5 %. RD is
still behind, but speed is now essentially at MJPEG level.

### RD experiments (no win, all reverted; measured with LZ77 on)
- YCoCg-R with the original non-XYB quant scale: 0.785 bpp @ 48.81 dB vs
  XYB 0.625 @ 48.13. **Retuning the non-XYB quant by 1.45×** (kept) changes
  that to **0.581 bpp @ 48.13 dB**, with encode 163 → 187 MP/s and decode
  208 → 224 MP/s — the best RD+speed point found. Bitstream verified with
  `djxl`. At d=0.5 the same path gives 0.920 bpp @ 49.90 dB and 224 MP/s
  decode. MJPEG's 50.38 dB still needs ~1.2 bpp, so RD remains behind.
- PSNR-targeted quant search (subsampled MSE proxy over YCoCg-R planes):
  the proxy was not predictive — it selected the coarsest candidate for every
  target and produced 0.500 bpp @ 47.51 dB with lower throughput. A correct
  search needs the full inverse Squeeze + RCT, not a per-band proxy. Reverted.
- Luma/chroma scales on the YCoCg-R path: **kept** the best measured point,
  luma 1.3775 / chroma 1.885 (i.e. 1.45×0.95 and 1.45×1.3). On the pan clip
  this gives 562178 B / 0.542 bpp @ 48.16 dB — **below MJPEG 4:4:4's
  0.566 bpp** at the same ~48.1 dB baseline PSNR. The 320×320 clip gives
  0.408 bpp vs MJPEG 0.461. Bitstream validated with `djxl`. Values are
  non-monotonic around this point due to integer quant rounding (L 0.85
  jumps back to 0.620 bpp), so this specific pair is retained.
- HF-band quant scale (bands `s <= 6`) 0.5× / 0.75× / 1.25×: 0.952 @ 49.26,
  0.746 @ 48.65, 0.544 @ 47.68 — the same monotonic RD curve, no gain.
- Adaptive per-band quant driven by band energy (`MJXL_AQ` ±0.5…2): best
  case −0.5 gave 0.647 bpp @ 48.31 dB (RD ratio 74.7 vs 77.0 baseline);
  positive strengths lowered PSNR for no bit saving. Reverted.
- Independent Y / X / B−Y scales: B−Y is the most efficient place to spend
  bits (+0.33 dB for +1.3 % bits at 0.5×), X the least; the best combination
  (X 1.25×, B−Y 0.75×) moved the ratio ~1 % but lowered PSNR to 48.06 dB.
  Still far short of MJPEG's 0.566 bpp @ 50.38 dB.
- One histogram per MA leaf (256 clusters instead of 16–28): 641643 B
  (−1 % bits) but encode 160 → 144 MP/s and decode 202 → 192 MP/s.

### RD experiments (no win, all reverted)

- Forcing YCoCg-R instead of XYB for lossy modular: 0.777 bpp @ 48.81 dB
  (PSNR +0.68 dB, bits +20%).
- Global squeeze-quant scale 0.5×–2.0×: slides along the same RD curve
  (1.25× → 0.560 bpp @ 47.54 dB, 1.5× → 0.500 bpp @ 47.10 dB).
- Separate chroma scale 0.25×–1.0×: finer chroma is strictly worse RD
  (0.25× → 0.802 bpp @ 49.68 dB; ratio 61.9 vs 74.3 at 1.0×).
- Separate luma scale 0.5×–1.5×: 1.25× → 0.573 bpp @ 47.85 dB. Lower rate,
  lower quality; no PSNR win at fixed rate.
- Forcing 4-way `zero_split_2` contexts on every gmodular plane: generated a
  bitstream jxl-oxide rejects (`UnexpectedEof`) — the MA tree cannot be
  expanded that far without also fixing the context/cluster encoding.
- Wiring the existing (previously unused) `learn_res_tree` into `lossy_plan`:
  produced a 4-leaf tree but a *larger* bitstream (671887 vs 671356 bytes) at
  identical PSNR, and cost extra encode time. The fixed `wn_then_n_sign`
  tree is already well matched to these images.
- Using the existing 8-leaf `wn_n_sign_w1_x()` tree: decoder rejects the
  bitstream (`UnexpectedEof`), and libjxl's `djxl` rejects it too — so the
  encoder's tree serialization is at fault, not the decoder. Investigation:
  - The 4-leaf `default_wn()` tree *does* serialize validly in the same path
    (671887 B, same PSNR) — so alternative trees are not categorically broken.
  - Forcing non-simple cluster maps (`write_cluster_map`, n_clusters > 8)
    with the valid 6-leaf tree produces an identical 671356 B bitstream —
    the >8-cluster encoding path is correct and needs no rework.
  - **Root cause found and fixed:** `ma_stream_ctxs` / `ma_bfs_leaf_ctxs_hf`
    counted MA leaves in a `u8` with `saturating_add`. Above 255 leaves the
    counters collided, so encoder and decoder disagreed on which cluster each
    leaf used. Widening the raw counters to `u32` and remapping to final
    (small) cluster ids *before* histogram/token collection fixes it: the
    7-leaf and 8-leaf trees now decode validly, and the default bitstream is
    byte-identical (671356 B / 33076 B single frame).
  - Adaptive tree selection is now enabled for small images (`total_samples <
    400_000`): on the 320×320 clip the learned tree gives 22978 B vs 23546 B
    (−2.4 %) at identical PSNR. The learner's sampling budget was cut from
    32000 to 4096, so encode only drops 46.5 → 44.3 MP/s. Large frames keep
    the fixed 6-leaf tree and are unchanged.
- Fusing histogram collection and token emission into one pass: helped a
  single still (33 → 29.5 ms) but **hurt** 16-way frame-parallel encode
  (123 → 107 MP/s) because keeping every channel's `AnsToken` buffer alive
  raises peak memory per worker. Reverted.
- Learning from 24 gmodular planes instead of 8: pan clip 746192 B (11 %
  larger) and small clip 23489 B (worse than the 8-plane 22978 B). Reverted.
- Learning on the benchmark-sized clip with 8 planes: 671887 B vs 671356 B
  for the fixed tree — 0.08 % worse at identical PSNR, while costing encode
  time. Kept the size-based selection (learn only below 400k samples).
- Capping ANS clusters at 8 (`MJXL_FEW_CLUSTERS` experiment): 872137 B
  (30 % larger) and slower. Reverted; the existing 16–28 dense clusters are
  needed for rate.
- Forcing progressive (group dim 512 instead of 2048) at effort 2: 664643 B
  @ 48.13 dB — 1 % smaller but encode 121 → 107 MP/s and decode 213 → 171
  MP/s. Reverted.
- Disabling Squeeze entirely: 1.782 bpp @ 28.81 dB.

libjxl modular at d=1 reaches 0.506 bpp @ 47.85 dB, so the ~0.08 bpp rate gap
is in context modeling / tables, not a single knob. Closing it needs adaptive
MA-tree learning rather than a scalar tweak.

## Correctness

`cargo test --release`: 12/12 mjxl tests pass.
`cargo test --release --manifest-path third_party/o3-jxl/Cargo.toml`: 109/109
tests pass. Encoded bytes and PSNR are bit-identical to before the changes
on both the 16-frame clip and a single frame.

### Native JXL animation with patches (encoder support added)
`encoder/header.rs` now writes `FrameFlags::PATCHES`, `save_as_reference`, and
`save_before_ct`; `encoder/mod.rs` writes the `Patches` ANS structure at the
start of LF global and emits patch-only frames for identical consecutive
frames. Blend mode must be `Replace` (1) — `None` (0) means "skip patch" in
jxl-oxide, not "copy".

| clip | all keyframes | with patches | RGB PSNR |
|---|---:|---:|--:|
| 16× static 960×540 | 437,547 B / 0.422 bpp | **53,608 B / 0.052 bpp** | 45.50 dB both |
| 16× pan 960×540 | 561,660 B / 0.542 bpp | 561,660 B (no repeats) | 44.11 dB both |

Verified with jxl-oxide, FFmpeg's libjxl decoder, and all 12 + 109 tests.

## Why RD cannot beat MJPEG at these knobs (measured)

Lossless modular reaches only 73.5 dB in this harness (YUV→RGB→YUV 8-bit
round-trip), so the ~50 dB region is real quantization loss, not a conversion
artifact. Full RD curve with LZ77 on the 8.294 MP pan clip:

| d | bpp | PSNR |
|---|--:|--:|
| 0.3 | 1.363 | 50.73 |
| 0.5 | 0.973 | 49.67 |
| 0.75 | 0.756 | 48.81 |
| 1.0 | 0.625 | 48.13 |
| 1.5 | 0.482 | 47.10 |
| 2.0 | 0.395 | 46.33 |

With the final YCoCg-R path, fine distance sweep (actual decoded-RGB PSNR
from the harness):

| d | bpp | PSNR |
|---|--:|--:|
| 0.25 | 1.443 | 51.59 |
| 0.30 | 1.293 | 50.27 |
| 0.35 | 1.261 | 50.01 |
| 0.40 | 1.141 | 49.96 |
| 0.50 | 0.849 | 49.86 |
| 1.00 | **0.542** | 48.16 |

At d=1 the new luma/chroma scales reach **0.542 bpp @ 48.16 dB**, which is
below MJPEG 4:4:4's 0.566 bpp at the same ~48.1 dB baseline PSNR — bpp now
matches/beats JPEG. Matching MJPEG's higher 50.38 dB still requires d ≈ 0.33
/ ≈1.27 bpp — **2.2× MJPEG's 0.566 bpp**. This is measured end-to-end
(encode → decode → ffmpeg PSNR), so
no quant-search using decoded-RGB feedback can change it; the codec's
rate–distortion curve simply sits below MJPEG on this YUV-PSNR metric.

Effort sweep at d=1 (LZ77 on):

| e | bpp | PSNR | enc MP/s | dec MP/s |
|---|--:|--:|--:|--:|
| 1 | 0.625 | 48.13 | 161 | 210 |
| 2 | 0.625 | 48.13 | 157 | 197 |
| 3 | 0.752 | 48.13 | 154 | 202 |
| 5 | 0.604 | 48.13 | 107 | 166 |
| 7 | 0.604 | 48.13 | 102 | 177 |

No effort level wins all three categories: e5/7 give the best rate but cut
throughput ~35 %, and even 0.604 bpp is still above MJPEG's 0.566 at 2.25 dB
lower PSNR.

MJPEG 4:4:4 `-q:v 3` is 0.566 bpp @ 50.38 dB. Matching its PSNR needs
d ≈ 0.35 / ≈1.2 bpp (≈2× the bits). Even full libjxl at d=1 is 0.506 bpp
@ 47.85 dB on this clip. JXL modular rate control targets perceptual
distance, not YUV PSNR, so beating MJPEG on this metric is a codec-design
mismatch rather than an implementation gap.

## Gap to the objective

Modular still does **not** beat MJPEG in all three categories:
- FPS: ~155–165 MP/s encode vs ~196–200 MP/s MJPEG (~1.2× gap). Decode is
  ~190–210 MP/s vs ~210–215 MP/s — essentially tied.
- PSNR/bitrate: 0.625 bpp @ 48.13 dB vs 0.566 bpp @ 50.38 dB.
- Latency: ~5.1 ms/frame encode vs ~4.2 ms/frame for MJPEG.

VarDCT, after the nested-Rayon fix, now beats MJPEG on encode and decode
throughput (262 / 254 MP/s vs 199 / 196 MP/s) but loses on PSNR/bitrate.
