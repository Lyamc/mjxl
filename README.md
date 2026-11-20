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

## MJPEG player / desktop streaming

`tools/mjpeg-player.ps1` wraps ffplay and ffmpeg for the MJPEG/Matroska path
(Windows, `gdigrab` desktop capture). The receiver binds first, so the stream
never outruns the player.

```powershell
# Play an MJPEG-in-MKV file (ffplay and mpv both handle this container).
.\tools\mjpeg-player.ps1 play -File .\test.mkv

# Desktop -> player over UDP (defaults: full 1280x720 desktop, 60 fps, q=5).
.\tools\mjpeg-player.ps1 stream                        # udp://127.0.0.1:3322
.\tools\mjpeg-player.ps1 stream -Seconds 10 -Quality 3

# Measure delivered frame rate end-to-end.
.\tools\mjpeg-player.ps1 bench -Seconds 4
#   Delivered 240 frames in 4.09s -> 58.7 fps (full desktop, q=5)

# Latency modes: normal (buffered) / low (default) / ultra (zero queue).
.\tools\mjpeg-player.ps1 stream -Latency ultra

# Optional: drop late frames instead of showing them behind schedule.
.\tools\mjpeg-player.ps1 stream -FrameDrop
```

Equivalent raw commands:

```bash
ffplay -autoexit -f matroska -i udp://127.0.0.1:3322
ffmpeg -f gdigrab -framerate 10 -i desktop -vf scale=640:-2 \
  -c:v mjpeg -q:v 5 -f matroska 'udp://127.0.0.1:3322?pkt_size=65536'
```

Performance on this machine (1280×720 desktop, MJPEG q=5): capture is the
only real-time constraint — MJPEG encoding alone runs >20× realtime. The
receiver decodes a 5-second stream as exactly 300 frames (60 fps), ~28 Mbit/s.
`bench` reports ~58–60 fps wall clock (it includes sender startup). gdigrab's
`-framerate` is an upper bound; 120 yields a steady 60 fps on a 60 Hz desktop,
while 60 yields ~58.

**Latency / timing.** ffplay's default `probesize` is 5 MB (~1.4 s at 28 Mbit/s)
and its video queue builds to ~1.5 MB (~0.44 s). The sender now uses
`-use_wallclock_as_timestamps 1` (gdigrab's own PTS drift) and the player uses
`-fflags +genpts+nobuffer -probesize 32768 -analyzeduration 0 -fpsprobesize 0`.
The Matroska muxer uses `-cluster_time_limit 50 -flush_packets 1`. With those
fixes the late-frame count drops from 10–19 to 1–2 without any special sync
mode. `ultra` adds `-sync video` (video master clock) and reaches **zero late
frames with a zero-length video queue** — no framedrop needed. `-FrameDrop`
adds `-framedrop` if you explicitly want late frames discarded.

Measured mid-stream (4 s, 1280×720 @ 60 fps):

| mode | video queue | late frames (`fd`) |
|---|---:|---:|
| normal | 1550 KB (~0.44 s) | 1 |
| low (default) | ~90 KB (~0.03 s) | 2 |
| ultra | 0 KB | **0** |

Verified: a pre-recorded `test.mkv` (MJPEG in Matroska) played end-to-end in
ffplay; mpv also reads MJPEG-in-MKV (`mpv --vo=null test.mkv`).

## Quality of life

- Shell completions: `mjxl completions bash|zsh|fish|powershell|elvish` prints
  a script to stdout (e.g. `mjxl completions powershell | Out-String | iex`).
- `--json` (global) makes `compare`, `bench`, `stills`, and `info` emit a single
  machine-readable object on stdout. Progress and diagnostics stay on stderr.
  `compare` rows carry `codec`, `bytes`, `bpp`, `encode_ms`, `decode_ms`,
  `enc_mps`, `dec_mps`, `psnr`, and `note`; `bench` carries the same timings
  plus the coder and settings.
- `compare` and `stills` clean up their temporary work directory on exit.
  Pass `--work-dir` to keep control of the location, or `--keep-work-dir` on
  `compare` to preserve the auto-generated one for inspection.

- `mjxl verify clip.y4m -m modular -d 1` round-trips a clip in-process and
  reports RGB PSNR (mean and per-frame minimum) without needing ffmpeg.
- `mjxl anim clip.y4m out.jxl -m`-style native JXL animation. Identical
  consecutive frames are emitted as **patch frames** that copy the whole
  canvas from the previous reference frame instead of being re-encoded:
  16 identical 960×540 frames go from 437,547 B (0.422 bpp) to **53,608 B
  (0.052 bpp)** at identical RGB PSNR (45.50 dB). Confirmed decodable by
  jxl-oxide, libjxl and FFmpeg.

```bash
mjxl verify clip.y4m -m modular -d 1 -e 2
mjxl --json compare -m modular -d 1 -e 2 clip.y4m > results.json
mjxl --json bench -m modular -d 1 -e 2 --repeats 9 clip.y4m
mjxl --json info out.mjxl | jq '.displayed[0].samples[0].size'
mjxl completions bash > /etc/bash_completion.d/mjxl
```

`-e` is 1..=10, the same scale as `cjxl -e`. The default is 2: coefficient reordering on top of flat 8×8 VarDCT. On the stills in the effort table below that is about 1% smaller than effort 1, at the same RGB PSNR, for a few milliseconds. Effort 3 adds the 4×4 / 4×8 search, which saves much less per millisecond. Effort 7 is the full VarDCT path, and 8..=10 match it. `compare` and `stills` take `-e` as well.

## Compare

Needs `ffmpeg` on `PATH`. Stills versus libjxl also need `cjxl` and `djxl`. OpenJPEG / OpenJPH are not required; JPEG 2000 is FFmpeg's `jpeg2000` encoder.

```bash
# One still. In-process mjxl (VarDCT and modular) vs one-shot cjxl/djxl.
# Prints bytes, bpp, encode/decode ms, MP/s, and PSNR. Median of --repeats after a warmup.
cargo run --release -- stills photo.png -d 1 -e 7

# A Y4M against MJPEG 4:4:4, MJPEG 4:2:0, and JPEG 2000.
# encMP/s and decMP/s, bpp, PSNR. The cjxl row, when present, is per-frame process spawn.
# -e 7 matches the tables below. Omit -e to use the default, 2.
cargo run --release -- compare clip.y4m -d 1 -e 7 --mjpeg-q 3 --j2k-q 5
```

Quality knobs (`-q:v` vs `-d`) are not the same units. The table is one operating point, not a matched-rate bake-off.

## Performance

Release build, 2026-10-20, Windows, Intel Core i9-13900K. Dirty-rectangle search used Vulkan on an NVIDIA GeForce RTX 3060. Motion decode in the table is CPU. libjxl is 0.12.0. FFmpeg provides `mjpeg` and `jpeg2000`.

Settings for the tables in this section: `-d 1` and `-e 7` for JPEG XL, MJPEG `-q:v 3`, JPEG 2000 `-q:v 5`. Those knobs are not the same unit. Stills PSNR is RGB. Motion PSNR is FFmpeg's `psnr` average on full-range 4:4:4 (`yuvj444p`). Encode, `compare`, and `stills` default to `-e 2`. Pass `-e 7` to reproduce these tables.

libjxl still times include process startup and file IO. mjxl still times are in-process, codestream only, median of 3 after one warmup. The motion libjxl row is one `cjxl` / `djxl` process per frame, so its MP/s is not a stills speed score. Its PSNR is `djxl`. This tree's decoder does not match `djxl` on libjxl VarDCT (about 24 dB RGB apart on one checked frame).

Effort 7 VarDCT encode runs inverse Gaborish in SIMD and the block-size search on the rayon pool. On these clips the codestream size and PSNR match the previous encoder. The effort-7 motion table below is from 2026-10-20. On that path MJPEG encode stays faster: it is an 8×8 DCT plus Huffman, and effort 7 still runs the adaptive quant field, several DCT sizes, and ANS.

Default effort is 2 (flat 8×8, coefficient orders, no gaborish). Intra frames of one clip encode and decode together, and the decoder builds each default dequantization matrix once per process instead of rebuilding DCT8 through DCT256 on every frame. The 2026-10-24 tables are that path. A single still does not get the cross-frame overlap, so its MP/s is lower than the pan.

### Effort

VarDCT turns tools on in this order. Nothing below a level is turned back off. 7..=10 run the same tools. There is no butteraugli quant search above 7.

| effort | VarDCT adds |
|---|---|
| 1 | Flat 8×8, flat quant |
| 2 | Coefficient orders (default) |
| 3 | 4×4 and 4×8 / 8×4 search |
| 4 | 16×16 and 8×16 / 16×8 |
| 5 | Gaborish, adaptive quant, chroma-from-luma |
| 6 | Block-context model |
| 7–10 | 32×32 search |

Gaborish stays on the same step as the quant field. Turning the blur on alone, without that field, raised `bike.png` from 2.59 bpp to 3.34 bpp at distance 1.

Lossy modular does not follow that ladder. Group size is libjxl's automatic choice at effort ≥ 5 (256 pixels, or 512 when both axes are ≤ 400). Effort 3–4 uses one step larger, and 1–2 two steps larger, capped at 1024. LZ77 for lossy residuals starts at 3, so 1 and 2 share a codestream, 3 and 4 share one, and 5..=10 share one. Lossless modular also enables palette search at 5 and Squeeze at 8.

`bike.png` and `kodim23.png`, `-d 1`, one encode after a warmup (not the median-of-3 used below). Bytes and PSNR are exact for this encoder. A few milliseconds of the times are run noise.

| effort | bike bytes | bike ms | bike PSNR | kodim bytes | kodim ms | kodim PSNR |
|---|--:|--:|--:|--:|--:|--:|
| 1 | 683799 | 31.1 | 36.51 | 75309 | 8.2 | 38.61 |
| 2 | 677856 | 35.7 | 36.51 | 74444 | 9.2 | 38.61 |
| 3 | 675875 | 51.3 | 36.58 | 74177 | 15.7 | 38.73 |
| 4 | 663078 | 66.4 | 36.39 | 71425 | 20.4 | 38.75 |
| 5 | 651310 | 91.1 | 36.15 | 84726 | 28.2 | 39.28 |
| 6 | 645837 | 95.6 | 36.15 | 82893 | 29.4 | 39.28 |
| 7 | 645837 | 91.7 | 36.15 | 82884 | 27.4 | 39.28 |

Effort 2 is the default because it is the light step that still pays. Against effort 1 it drops 5,943 bytes on bike (0.87%) and 865 bytes on kodim (1.1%), PSNR unchanged, for about 5 ms and 1 ms. Effort 3 then drops 1,981 and 267 bytes for about 16 ms and 7 ms. Effort 4 saves more, and on bike it also loses 0.19 dB. Effort 5 is the perceptual stack: bike gets smaller and loses another 0.24 dB, while kodim gets larger and gains 0.53 dB. The 32×32 search at effort 7 changes kodim by 9 bytes and does not change bike, so bike at effort 7 matches bike at effort 6.

Lossy modular on the same runs: bike effort 1–2 is 667846 bytes, 35.24 dB, about 191 ms; effort 3–4 is 621725 bytes, 35.25 dB, about 250 ms; effort 5–10 is 622900 bytes, 35.25 dB, about 237 ms. Kodim effort 1–2 is 68013 bytes, effort 3–4 is 68377, effort 5–10 is 69617, all 37.41 dB. The effort-7 bike modular row in the stills table is that last modular point.

### Default effort on the pan (2026-10-24)

Same 16×960×540 pan (8.294 MP), release build, `-d 1` with no `-e` (effort 2), MJPEG `-q:v 3`. Two `compare` runs, default coding (VarDCT and modular). VarDCT intra bytes and PSNR match across the runs.

| run | codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | mjxl VarDCT intra, e=2 | 2744331 | 2.647 | 55.6 | 52.7 | 149.1 | 157.4 | 42.46 |
| 1 | MJPEG 4:4:4 | 2396468 | 2.311 | 48.0 | 63.2 | 172.7 | 131.1 | 42.27 |
| 2 | mjxl VarDCT intra, e=2 | 2744331 | 2.647 | 65.8 | 55.9 | 126.1 | 148.3 | 42.46 |
| 2 | MJPEG 4:4:4 | 2396468 | 2.311 | 50.0 | 57.4 | 165.7 | 144.6 | 42.27 |

Half of MJPEG 4:4:4 encode on those runs is 86.4 and 82.9 MP/s. One third of MJPEG 4:4:4 decode is 43.7 and 48.2 MP/s. VarDCT intra is above both, PSNR is 0.19 dB above MJPEG, and bpp is 1.15× MJPEG (limit 1.4×). The delta rows are the same bytes as intra: the pan dirties every frame, and the dirty search stays serial, so delta encode is 43.5 and 39.6 MP/s on these two runs.

### Stills vs libjxl

`bike.png`, 1280×1600 (2.048 MP). Effort 2, median of 3 after one warmup, 2026-10-24.

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT, e=2 | 677856 | 2.648 | 23.1 | 26.0 | 88.6 | 78.7 | 36.51 |
| mjxl modular, e=2 | 667846 | 2.609 | 192.4 | 63.8 | 10.6 | 32.1 | 35.24 |
| libjxl 0.12.0, e=2 | 570543 | 2.229 | 45.1 | 24.3 | 45.4 | 84.2 | 36.34 |

Effort 7, same image, remeasured 2026-10-24. VarDCT bytes and PSNR match the 2026-10-20 codestream (645837 bytes, 36.15 dB). Decode is faster because the dequant matrices are cached. Kodim below was not remeasured.

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT, e=7 | 645837 | 2.523 | 87.4 | 32.4 | 23.4 | 63.2 | 36.15 |
| mjxl modular, e=7 | 622900 | 2.433 | 235.1 | 53.1 | 8.7 | 38.6 | 35.25 |
| libjxl 0.12.0, e=7 | 539057 | 2.106 | 280.6 | 28.4 | 7.3 | 72.0 | 36.44 |

`kodim23.png`, 768×512 (0.393 MP). Effort 7, measured 2026-10-20, not remeasured with the dequant cache. Process startup is a larger share of the libjxl time on this smaller image.

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT | 82884 | 1.686 | 27.7 | 40.1 | 14.2 | 9.8 | 39.28 |
| mjxl modular | 69617 | 1.416 | 41.4 | 18.3 | 9.5 | 21.4 | 37.41 |
| libjxl 0.12.0 | 59191 | 1.204 | 65.6 | 11.8 | 6.0 | 33.2 | 39.54 |

### Motion vs MJPEG and JPEG 2000

Effort 7, 2026-10-20. The effort-2 runs are in the table above. 16 frames, 960×540, a 20×25 px/frame crop sliding across `bike.png` (8.294 MP). One pass. The shift dirties the whole frame, so the delta rows are the same bytes as all-intra and the search only adds encode time.

| codec | bytes | bpp | enc ms | dec ms | enc MP/s | dec MP/s | PSNR |
|---|--:|--:|--:|--:|--:|--:|--:|
| mjxl VarDCT intra | 2637908 | 2.544 | 518.2 | 672.7 | 16.0 | 12.3 | 41.96 |
| mjxl VarDCT delta | 2637908 | 2.544 | 550.8 | 674.5 | 15.1 | 12.3 | 41.96 |
| mjxl modular intra | 2520487 | 2.431 | 936.0 | 337.1 | 8.9 | 24.6 | 41.63 |
| mjxl modular delta | 2520487 | 2.431 | 987.7 | 338.2 | 8.4 | 24.5 | 41.63 |
| MJPEG 4:4:4 | 2396468 | 2.311 | 45.3 | 54.0 | 183.0 | 153.7 | 42.27 |
| MJPEG 4:2:0 | 2000538 | 1.930 | 44.0 | 47.4 | 188.6 | 175.1 | 41.17 |
| libjxl, per frame | 2192578 | 2.115 | 1003.7 | 232.1 | 8.3 | 35.7 | 42.11 |
| JPEG 2000 | 5155836 | 4.973 | 458.8 | 218.6 | 18.1 | 37.9 | 42.68 |

Dirty rectangles on a clip that actually repeats pixels: 48 frames of `anim-icos`, 128×128 (0.786 MP). Delta fires (`rect=6`, `patches=48`) and saves a little rate. At this size the MP/s column is overhead, not the codec's photographic throughput. The pan table above is the speed comparison.

| codec | bytes | bpp | enc MP/s | dec MP/s | PSNR | note |
|---|--:|--:|--:|--:|--:|---|
| mjxl VarDCT intra | 398672 | 4.056 | 2.4 | 0.5 | 40.29 | all intra |
| mjxl VarDCT delta | 393962 | 4.008 | 2.1 | 0.5 | 39.93 | rect=6 patches=48 |
| mjxl modular intra | 305775 | 3.111 | 5.5 | 6.3 | 39.22 | all intra |
| mjxl modular delta | 302110 | 3.073 | 3.7 | 6.4 | 38.96 | rect=6 patches=48 |
| MJPEG 4:4:4 | 313696 | 3.191 | 24.5 | 29.0 | 42.29 | `-q:v 3` |
| MJPEG 4:2:0 | 236976 | 2.411 | 25.1 | 28.5 | 38.95 | `-q:v 3` |
| libjxl, per frame | 251703 | 2.560 | 0.9 | 2.0 | 40.61 | `djxl` PSNR |
| JPEG 2000 | 537444 | 5.467 | 14.2 | 18.8 | 44.06 | `-q:v 5` |

## WebAssembly demo

`third_party/o3-jxl` builds to WASM with a `wasm-bindgen` progressive-decode
session (`JxlSession`: `loadJxl`, `decodeTo`, `copyRgba`, `psnrVsFull`). A Trunk
demo lives beside it:

```bash
cd third_party/o3-jxl
trunk build --release
trunk serve            # http://127.0.0.1:8081
```

The demo decodes any prefix of a `.jxl` locally in the browser, with a bundled
`dice.jxl` sample and file upload. Only the **decoder** is exposed to JS today —
the encoder is compiled into the WASM binary but has no `wasm-bindgen` bindings.

Note: `mjxl` itself (the CLI / container / UDP streamer) is **not** WASM
compatible — it depends on clap, wgpu, rayon, filesystem and sockets.

## Development

Layout: `src/` is the mjxl CLI and container/temporal/GPU layers;
`third_party/o3-jxl/` is the vendored still codec (modular + VarDCT);
`third_party/jxl-vardct/` is a local patch of jxl-oxide's dequant cache;
`tools/` holds the MJPEG player / desktop-stream helper; `DESIGN.md` explains
the container investigation and `PERF.md` records measured results and the
experiments that were rejected.

Formatting and tests are checked in CI on Windows, Linux and macOS:

```bash
cargo fmt --all -- --check
cargo build --release
cargo test --release
cargo test --release --manifest-path third_party/o3-jxl/Cargo.toml
```

The pinned nightly in `rust-toolchain.toml` is required for `portable_simd`.
`MJXL_PROFILE=1` prints per-phase encode timings; `O3_PROFILE=1` prints decoder
phase timings. Benchmarks are in-process via `mjxl bench` and use median-of-N;
`mjxl compare` runs the external MJPEG / JPEG 2000 / libjxl baselines.

## Status

Both still modes are supported: **VarDCT** (lossy RGB, the MJPEG analogue) and **modular** (lossless and lossy residuals). `--progressive` writes smaller modular groups (and Squeeze when the distance is positive) so `decode --prefix` can paint a partial sample; VarDCT prefixes show LF groups before HF. `levels` renders DC, low-frequency, and full 8×8 passes on the GPU. Dirty rectangles are a list, not one bounding box, and `--closed-loop` diffs against the previous reconstruction. Container is not ISOBMFF. No audio. No motion-vector copy. Feeding `compare` a full-range `yuvj444p` Y4M keeps the PSNR in one range; untagged `yuv444p` from FFmpeg is limited-range and this reader treats untagged `C444` as full range.

## License

MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.
