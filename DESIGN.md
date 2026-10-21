# Motion JPEG XL — investigation

MJPEG is not a video codec with motion compensation. It is **independently coded stills in a timed container**. Motion JPEG 2000 is the same contract with JPEG 2000. This crate asks what that contract looks like if the still codec is JPEG XL.

## The three systems

| | MJPEG | Motion JPEG 2000 | Motion JPEG XL (this crate) |
|---|---|---|---|
| Still codec | JPEG (ISO 10918, usually 8×8 DCT, Huffman, 4:2:0 or 4:2:2) | JPEG 2000 Part 1 (DWT, EBCOT) | JPEG XL (ISO 18181) — 8×8 VarDCT or modular, via `o3-jxl` |
| Temporal tools | none | none | **none** (even though JXL animation *can* reference previous frames) |
| Random access | every frame is a keyframe | every frame is a keyframe | every frame is a complete still |
| Typical container | AVI (`MJPG`), QuickTime, raw `mjpg` | ISOBMFF `.mj2` / `.mjp2` (brand `mjp2`) | `.mjxl` sample stream (v0); native JXL animation is a later path |
| Why people used it | cheap encode, edit-friendly, cameras, capture cards | archive, lossless, better still coding than JPEG | MJPEG-class *speed* with a 2020s still codec |

MPEG-family codecs (H.264 intra, ProRes, DNxHD) overlap the same “all intra” job. They are not the comparison set here. The question is: **replace the still codec inside the MJPEG contract, keep the contract.**

## What MJPEG actually stores

A Motion-JPEG AVI is an AVI stream whose samples are JPEG/JFIF (or AVI1 APP0) bitstreams. Every sample has the keyframe flag. Seeking is “decode that JPEG.” Capture cards emitted this because a JPEG encoder is a still-image pipeline run at frame rate.

FFmpeg’s encoder (`-c:v mjpeg`) is that pipeline:

```bash
ffmpeg -i in.y4m -c:v mjpeg -pix_fmt yuvj444p -q:v 3 out.avi
```

Pixel formats are YUV (`yuvj420p` / `422` / `444`). Default capture-style encodes are **4:2:0**. Quality is a JPEG quantizer (`-q:v` 2..=31, 2 is finest). Speed on this machine is the MJPEG bar `o3-jxl` has been chasing (~100–400 MP/s class; 1080p60 ≈ 124 MP/s).

Limits that matter for a rematch:

- 8-bit.
- Usually chroma subsampled.
- No lossless (there is lossless JPEG; almost nobody muxes it as MJPEG).
- No HDR.
- Blocking at low rate.

## What Motion JPEG 2000 actually stores

ISO/IEC 15444-3 / ITU-T T.802. File is ISOBMFF (same family as MP4/JP2), brands `mjp2` / `mj2s`. Each video sample is a JPEG 2000 *codestream* (not a full JP2). No inter-frame prediction.

That is the clean analogue: **ISOBMFF timed samples of independent still codestreams.**

FFmpeg on this machine has a native `jpeg2000` encoder (not Kakadu, not OpenJPEG’s `libopenjpeg` wrapper). OpenJPEG `opj_compress` and OpenJPH `ojph_compress` (HTJ2K) are also installed. HTJ2K is the “make JPEG 2000 as fast as JPEG” effort; it is the closest *speed* cousin of what we want from JXL, but it is still a wavelet + precinct bitstream, not DCT.

```bash
ffmpeg -i in.y4m -c:v jpeg2000 -pix_fmt rgb24 out.mov
opj_compress -i frame.pgm -o frame.j2k
ojph_compress -i frame.ppm -o frame.jph
```

JPEG 2000 wins on:

- lossless (reversible 5/3) and lossy (9/7) in one family
- 4:4:4, high bit depth, cinema/archive profiles
- resolution scalability (decode 1:2, 1:4 without the rest)

It loses on:

- encode/decode cost vs Huffman JPEG (EBCOT)
- player/editor support compared with MJPEG
- FFmpeg’s native encoder is not Kakadu; numbers will reflect that

## What JPEG XL already has (and is *not* MJPEG)

A JPEG XL *codestream* can be an animation: `ImageHeader` has ticks-per-second and loop count; each `FrameHeader` has a duration, blend mode, crop, and optional references (patches, previous frame as base). The optional ISOBMFF wrapper (18181-2) adds Exif/XMP and a `jxli` keyframe index. That is an **animated image**, not a video track of independent samples.

`o3-jxl::Encoder::encode_animation` already writes that animation header. Today it is **modular-only** (no VarDCT frames). Each frame is independently coded (blend = replace, no references), so it is already “MJPEG-shaped” for lossless / lossy modular. Lossy RGB at MJPEG rates wants **VarDCT per frame**, which `encode_vardct_rgb` only emits as a complete still (its own image header).

v0 was all-intra. v2 samples can be **skips** (`size=0`, repeat the previous canvas) or **dirty rects** (a smaller JXL still pasted at `x,y`). That's "only write what changed." It is open-loop (compare to the *source* previous frame, not the reconstruction). Integer motion search that *copies* a block from a non-zero MV needs JXL patches; not in yet. `--intra` disables this and is the MJPEG-shaped row.

Each sample is encoded with one of two still coders, both kept:

| | VarDCT (`-m vardct`) | Modular (`-m modular`) |
|---|---|---|
| JXL mode | 8×8 DCT in XYB | YCoCg-R + Gradient (± Squeeze/palette) |
| Analogue | MJPEG / JPEG | closer to MJPEG2000’s “different still transform” |
| Lossy | `-d > 0` (default for RGB) | `-d > 0 -m modular` (quantized residuals) |
| Lossless | n/a (use modular `-d 0`) | `-d 0` |
| Next work | keep it correct and fast vs MJPEG | MA, skip in residual domain, better RD |

`compare` default is **both**. Future effort goes to modular; VarDCT stays as the DCT baseline.

There is no widely deployed “JXL video sample” fourcc in MP4. 18181-2 is an image container that happens to hold an animated codestream. Muxing JXL into Matroska as `V_JPEGXL` would be nice later; it is not required to compare codecs.

## Proposed v0: a sample stream of complete stills

File `.mjxl`:

```
magic     "MJXL"          4 bytes
version   u16le = 1
flags     u16le           bit0 = lossless modular; otherwise VarDCT
width     u32le
height    u32le
fps_num   u32le
fps_den   u32le
nframes   u32le
reserved  u32le = 0
index[n]  { offset u64le, size u32le, duration_ticks u32le }
payload   concatenation of complete JPEG XL stills (each starts 0xFF 0x0A)
```

Offsets are from the start of the file. Every sample is a decoder-valid `.jxl` you can `djxl` in isolation. That is the MJPEG property: **cut at a sample boundary, get a picture.**

This is deliberately *not* ISOBMFF yet. MJ2 is ISOBMFF because JPEG 2000 already lived there. We can grow a `mjlx` brand later without changing the still codec. v0 exists so a comparison harness can run this week.

Encode path:

1. Read Y4M (or a PNG sequence) → RGB8 frames.
2. For each frame, `o3-jxl::Encoder` (VarDCT if `-d > 0` and RGB; modular otherwise).
3. Append the still and an index entry.

Decode path: walk the index, `Decoder::decode_bytes` each sample, write Y4M `C444`.

Independence rules (must hold or it is not MJPEG):

- no `save_as_reference`
- blend = replace, full canvas, no crop
- no patches from a hidden frame
- duration > 0 on every displayed frame

## How to compare

Same RGB frames in, three all-intra bitstreams out. Do **not** let MJPEG silently drop to 4:2:0 while JXL stays 4:4:4 and then call it a quality win.

| Row | Encoder | Container | Notes |
|---|---|---|---|
| MJPEG 444 | `ffmpeg -c:v mjpeg -pix_fmt yuvj444p -q:v Q` | AVI | Huffman JPEG, full chroma |
| MJPEG 420 | `ffmpeg -c:v mjpeg -pix_fmt yuvj420p -q:v Q` | AVI | the historically common MJPEG |
| JPEG 2000 | `ffmpeg -c:v jpeg2000 -pix_fmt rgb24` | MOV | native FFmpeg J2K (slow, not Kakadu) |
| OpenJPEG | `opj_compress` per frame + concat | dir of `.jp2` | Part 1 reference |
| HTJ2K | `ojph_compress` per frame | dir of `.jph` | the fair *speed* J2K |
| mjxl | this crate | `.mjxl` | `o3-jxl` stills |
| cjxl anim | `cjxl` animation | `.jxl` | libjxl, may use tools we don't |

Metrics, all on a warmed process / clip:

- **size** and **bits per pixel** (whole sequence)
- **encode wall time** and **MP/s** (pixels × frames / seconds)
- **decode wall time** and **MP/s**
- **PSNR-RGB** vs the RGB source (ffmpeg `psnr` filter)
- optional VMAF (this FFmpeg is built with `--enable-libvmaf`)

Quality knobs are **not** equivalent. JPEG `-q:v 3`, JPEG XL `-d 1`, JPEG 2000 layer rates are different psychovisual units. Sweep them. Report the RD point, don't pretend `d=1` is `q=3`.

Suggested first clips:

- `~/Build/daala/samples/test1_24f_320.y4m` — 24 frames, 320×320, fast iterate
- `~/Build/daala/samples/test1_8f_1080.y4m` — 8 frames, 1080×1080

`o3-jxl` VarDCT is still a minimal 8×8 (no Gabor/EPF/varblocks). On the 320² clip at `-d 1` it is **much** larger and **much** worse PSNR than FFmpeg MJPEG `-q:v 3`. That is an encoder gap, not a container gap. The compare command therefore also muxes **`cjxl -d 1` stills** into the same `.mjxl` wrapper when `cjxl` is on `PATH`, so the table can separate “JPEG XL the format” from “our VarDCT.”

```bash
cargo run --release -- compare ~/Build/daala/samples/test1_24f_320.y4m
```

The tables below are a historical Apple M-series run (24 frames × 320² = 2.46 MP, in-process mjxl vs FFmpeg subprocesses). They are not the Windows numbers in the README. Quality knobs are **not** matched:

| codec | bpp | enc MP/s | PSNR-YUV avg |
|---|--:|--:|--:|
After fixing a **transposed AC scatter** (vertical bars decoded as horizontal — that was the 20 dB hole), plus default CFL (`B−Y`) and HF quant-bias invert:

**24×320²** (`test1_24f_320.y4m`, MJPEG `-q:v 3` ≈ JPEG q90):

| codec | bpp | enc MP/s | PSNR |
|---|--:|--:|--:|
| mjxl-intra `-d 1` | 2.84 | 73 | **43.4** |
| mjxl-delta `-d 1` | 0.12 | 300 | 43.2 |
| MJPEG 4:4:4 `-q:v 3` | 2.45 | 51 | 42.8 |
| MJPEG 4:2:0 `-q:v 3` | 2.02 | 63 | 40.9 |
| cjxl `-d 1` (via our pack) | 2.23 | ~3* | 35.5 |
| JPEG 2000 `-q:v 5` | 5.22 | 9 | 44.4 |

**8×1080²** (`test1_8f_1080.y4m`):

| codec | bpp | enc MP/s | PSNR |
|---|--:|--:|--:|
| mjxl-intra `-d 1` | 1.74 | 108 | **46.1** |
| mjxl-delta `-d 1` | 0.22 | 261 | 46.1 |
| MJPEG 4:4:4 `-q:v 3` | 1.45 | 140 | 46.0 |
| MJPEG 4:2:0 `-q:v 3` | 1.23 | 165 | 44.1 |

Delta `skip=N-1` on these clips means the sequence barely moves at 16×16 MAE>4; that's the skip path working, not motion compensation. Intra is the fair MJPEG rematch: **PSNR matches MJPEG 444; bpp is still ~20% worse; encode is in the same band, decode is not.** Still-file vs cjxl at `d=1` on a 256² photo is now ~33.6 dB both sides; we're ~45% larger. Varblocks / ANS / AQ / Gabor are the remaining bpp gap.

## Speed expectation

`o3-jxl` warmed VarDCT on this CPU: ~280 MP/s on a smooth megapixel, ~50 MP/s on busy photos, decode ~70–120 MP/s. 1080p60 is 124 MP/s. MJPEG in libjpeg-turbo is often faster than that on the same silicon. JPEG 2000 Part 1 will lose the speed contest; HTJ2K might not.

A single CLI process still pays rayon startup. Compare in-process or with several frames so that cost amortizes.

## What we are not doing in v0

- Audio.
- ISOBMFF / Matroska mux.
- Temporal prediction, patches, LF-only frames.
- Extending `o3-jxl::encode_animation` to VarDCT (native `.jxl` anim). Worth doing once the rematch is real; it is a smaller file (one image header) and `djxl` plays it as animation.
- Claiming feature parity with `cjxl -d 1` animation (Gabor, EPF, varblocks).

## Decision

v0 is a **timed index of independent JPEG XL stills**. That is Motion JPEG, with the 1992 still codec swapped for 18181. Compare it with FFmpeg MJPEG and FFmpeg/OpenJPEG/OpenJPH JPEG 2000 on the same RGB frames. If the numbers are boring, the contract was still the right one to measure.
