# mjxl

Motion JPEG XL: independently coded [o3-jxl](third_party/o3-jxl) stills in a timed sample stream (`.mjxl`). The MJPEG / Motion JPEG 2000 contract — every frame is a keyframe — with JPEG XL as the still codec.

See [DESIGN.md](DESIGN.md) for the container investigation. The still codec is vendored under `third_party/o3-jxl` (package name stays `o3-jxl`).

## Build

Nightly Rust (`portable_simd`, pinned in `rust-toolchain.toml`).

```bash
cargo build --release
```

`jxl-oxide` is a git dependency. `.cargo/config.toml` fetches it without submodules.

## Encode / decode

```bash
# Y4M → .mjxl. Skip/dirty-rect on by default (several rectangles per frame when the changes are apart).
# -m vardct (default, JPEG-like) or -m modular (YCoCg-R + Gradient).
cargo run --release -- encode -d 1.0 -m vardct clip.y4m vd.mjxl
cargo run --release -- encode -d 1.0 -m modular --progressive clip.y4m mod.mjxl
cargo run --release -- encode --intra -d 1.0 clip.y4m intra.mjxl

# Back to 4:4:4 Y4M. --prefix 0.25 renders a progressive prefix of each sample.
cargo run --release -- decode out.mjxl out.y4m
cargo run --release -- decode --prefix 0.25 mod.mjxl preview.y4m

# DC / 2×2 / full 8×8 VarDCT-style passes (Vulkan compute when an adapter exists).
cargo run --release -- levels clip.y4m

cargo run --release -- info out.mjxl
```

`--backend auto|cpu|gpu` selects the compute device for dirty-tile search, rectangle paste, and the frequency passes. `auto` is the default. Entropy coding stays in `o3-jxl` on the CPU: Vulkan Video's standardized codecs are H.264, H.265, AV1, and VP9, and there is no JPEG XL profile, so the GPU path is compute (Vulkan first, then DX12 or Metal).

Each sample is a complete `.jxl`. `djxl` on a dumped frame should work.

## Compare

Needs `ffmpeg` on `PATH`. Stills versus libjxl also need `cjxl` and `djxl`. OpenJPEG / OpenJPH are not required; JPEG 2000 is FFmpeg's `jpeg2000` encoder.

```bash
# One still. In-process mjxl (VarDCT and modular) vs one-shot cjxl/djxl.
# Prints bytes, bpp, encode/decode ms, MP/s, and PSNR. Median of --repeats after a warmup.
cargo run --release -- stills photo.png -d 1 -e 7

# A Y4M against MJPEG 4:4:4, MJPEG 4:2:0, and JPEG 2000.
# encMP/s and decMP/s, bpp, PSNR. The cjxl row, when present, is per-frame process spawn.
cargo run --release -- compare clip.y4m -d 1 --mjpeg-q 3 --j2k-q 5
```

Quality knobs (`-q:v` vs `-d`) are not the same units. The table is one operating point, not a matched-rate bake-off.

## Performance

Release build, 2026-10-20, Windows, Intel Core i9-13900K. Dirty-rectangle search used Vulkan on an NVIDIA GeForce RTX 3060. Motion decode in the table is CPU. libjxl is 0.12.0. FFmpeg provides `mjpeg` and `jpeg2000`.

Settings: `-d 1` and `-e 7` for JPEG XL, MJPEG `-q:v 3`, JPEG 2000 `-q:v 5`. Those knobs are not the same unit. Stills PSNR is RGB. Motion PSNR is FFmpeg's `psnr` average on full-range 4:4:4 (`yuvj444p`).

libjxl still times include process startup and file IO. mjxl still times are in-process, codestream only, median of 3 after one warmup. The motion libjxl row is one `cjxl` / `djxl` process per frame, so its MP/s is not a stills speed score. Its PSNR is `djxl`. This tree's decoder does not match `djxl` on libjxl VarDCT (about 24 dB RGB apart on one checked frame).

### Stills vs libjxl

`bike.png`, 1280×1600 (2.048 MP).

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT | 645837 | 2.523 | 768.4 | 105.4 | 2.7 | 19.4 | 36.15 |
| mjxl modular | 622900 | 2.433 | 479.4 | 71.0 | 4.3 | 28.9 | 35.25 |
| libjxl 0.12.0 | 539057 | 2.106 | 281.1 | 29.3 | 7.3 | 69.9 | 36.44 |

`kodim23.png`, 768×512 (0.393 MP). Same settings. Process startup is a larger share of the libjxl time on this smaller image.

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT | 82884 | 1.686 | 144.1 | 41.4 | 2.7 | 9.5 | 39.28 |
| mjxl modular | 69617 | 1.416 | 42.7 | 18.1 | 9.2 | 21.7 | 37.41 |
| libjxl 0.12.0 | 59191 | 1.204 | 66.0 | 13.0 | 6.0 | 30.2 | 39.54 |

### Motion vs MJPEG and JPEG 2000

16 frames, 960×540, a 20×25 px/frame crop sliding across `bike.png` (8.294 MP). One pass. The shift dirties the whole frame, so the delta rows are the same bytes as all-intra and the search only adds encode time.

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT intra | 2637908 | 2.544 | 3567.6 | 1426.3 | 2.3 | 5.8 | 41.96 |
| mjxl VarDCT delta | 2637908 | 2.544 | 5681.3 | 1416.5 | 1.5 | 5.9 | 41.96 |
| mjxl modular intra | 2520487 | 2.431 | 1707.7 | 488.6 | 4.9 | 17.0 | 41.63 |
| mjxl modular delta | 2520487 | 2.431 | 1735.0 | 493.5 | 4.8 | 16.8 | 41.63 |
| MJPEG 4:4:4 | 2396468 | 2.311 | 48.8 | 55.8 | 169.8 | 148.7 | 42.27 |
| MJPEG 4:2:0 | 2000538 | 1.930 | 48.3 | 50.4 | 171.7 | 164.4 | 41.17 |
| libjxl, per frame | 2192578 | 2.115 | 1070.3 | 285.9 | 7.7 | 29.0 | 42.11 |
| JPEG 2000 | 5155836 | 4.973 | 462.1 | 217.1 | 17.9 | 38.2 | 42.68 |

Dirty rectangles on a clip that actually repeats pixels: 48 frames of `anim-icos`, 128×128 (0.786 MP). Delta fires (`rect=6`, `patches=48`) and saves a little rate. At this size the MP/s column is overhead, not the codec's photographic throughput. The pan table above is the speed comparison.

| codec | bytes | bpp | enc MP/s | dec MP/s | PSNR | note |
|---|--:|--:|--:|--:|--:|---|
| mjxl VarDCT intra | 398672 | 4.056 | 1.8 | 0.5 | 40.29 | all intra |
| mjxl VarDCT delta | 393962 | 4.008 | 1.8 | 0.2 | 39.93 | rect=6 patches=48 |
| mjxl modular intra | 305775 | 3.111 | 3.1 | 3.9 | 39.22 | all intra |
| mjxl modular delta | 302110 | 3.073 | 2.2 | 3.7 | 38.96 | rect=6 patches=48 |
| MJPEG 4:4:4 | 313696 | 3.191 | 22.4 | 29.7 | 42.29 | `-q:v 3` |
| MJPEG 4:2:0 | 236976 | 2.411 | 22.7 | 25.3 | 38.95 | `-q:v 3` |
| libjxl, per frame | 251703 | 2.560 | 0.8 | 1.5 | 40.61 | `djxl` PSNR |
| JPEG 2000 | 537444 | 5.467 | 13.2 | 16.9 | 44.06 | `-q:v 5` |

## Status

Both still modes are supported: **VarDCT** (lossy RGB, the MJPEG analogue) and **modular** (lossless and lossy residuals). `--progressive` writes smaller modular groups (and Squeeze when the distance is positive) so `decode --prefix` can paint a partial sample; VarDCT prefixes show LF groups before HF. `levels` renders DC, low-frequency, and full 8×8 passes on the GPU. Dirty rectangles are a list, not one bounding box, and `--closed-loop` diffs against the previous reconstruction. Container is not ISOBMFF. No audio. No motion-vector copy. Feeding `compare` a full-range `yuvj444p` Y4M keeps the PSNR in one range; untagged `yuv444p` from FFmpeg is limited-range and this reader treats untagged `C444` as full range.

## License

MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.
