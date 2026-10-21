# o3-jxl

Vendored into mjxl (`third_party/o3-jxl`) so a fresh clone builds without a sibling checkout. The package name stays `o3-jxl`.

High-performance JPEG XL codec in Rust, aimed at native and WebAssembly. Numerically heavy stages this crate owns run through [`std::simd`](https://doc.rust-lang.org/std/simd/) portable SIMD (WASM SIMD128, NEON, SSE/AVX) rather than a GPU compute backend.

A GPU path was a poor fit: WebGPU is optional in browsers, upload/readback dominated the old quantize pass, and the IDCT shader was never wired into VarDCT. SIMD is the acceleration that actually ships everywhere this crate needs to run.

**Today:** decoder (jxl-oxide bitstream + SIMD pack) plus a from-scratch **encoder** with both still modes: **modular** (lossless and lossy YCoCg-R + Squeeze; the path we will grow) and **8×8 VarDCT** for lossy RGB (`-m 0` or auto when `-d`/`-q` is lossy RGB). Lossy modular uses libjxl Squeeze quant tables (fine LF, coarse HF, Zero predictor on residuals) rather than one Gradient quant for every plane. SIMD YCoCg-R, Gradient / Zero residuals, prefix codes, LZ77 RLE, default **Squeeze**, lossless **palette**, 8- and 16-bit, APNG animation. CLI flags follow cjxl (`-q`/`-d`/`-e`/`-p`/`-R`/`-m`) and djxl (PAM/PPM/PGM/PFM/PGX, JPEG reconstruction, EXIF/XMP). SIMD kernels also cover 2×4/8 DCT, XYB, YCbCr, Gaborish. Any truncated prefix can be rendered (`ProgressiveDecoder`).

libjxl still leads on JPEG lossless transcoding (`-j`), HDR/EXR, Gabor/EPF-on encode, and variable DCT block sizes.

## Requirements

- Nightly Rust (for `#![feature(portable_simd)]`). A `rust-toolchain.toml` pins `nightly`.
- Edition 2024

`jxl-oxide` is fetched automatically from GitHub as a Cargo git dependency. Submodule fetch is disabled in `.cargo/config.toml` (only the library crates are needed, not jxl-oxide’s testdata).

Optional reference repos (`jxl-rs`, `libjxl`, `core_jpeg_decoder`) may be cloned locally for study; they are not build dependencies and are gitignored.

## Architecture

| Stage | Backend |
|-------|---------|
| Container / codestream parsing | CPU (jxl-oxide) |
| Entropy decoding, modular synthesis, VarDCT render | CPU (jxl-oxide) |
| Modular lossless encode (headers, MA tree, prefix codes) | CPU — `src/encoder/` |
| YCoCg-R RCT, Gradient / Zero residuals, pack-signed | `std::simd` — `src/encoder/predict.rs` (`i32x8` / `i32x4`) |
| 2×2 / 4×4 / 8×8 DCT-II and DCT-III | `std::simd` — `src/simd/dct.rs`, `src/simd/idct.rs` |
| XYB ↔ linear sRGB, YCbCr ↔ RGB | `std::simd` — `src/simd/color.rs` (`f32x8`) |
| Gaborish 3×3, 2× chroma upsample | `std::simd` — `src/simd/gabor.rs` |
| Linear → sRGB tone mapping & RGBA8 pack | `std::simd` — `src/simd/quantize.rs` (`f32x16` / `u8x16`) |

DCT / color / Gaborish kernels are unit-tested against scalar references. Encode uses them for 8×8 VarDCT and XYB; jxl-oxide still owns entropy decode and VarDCT *render*. On an M-series CPU, warmed VarDCT encode of a smooth 1 MP frame is about **280 MP/s** (1080p60-class); a busy photographic megapixel is entropy-bound at about **50 MP/s**. Decode of those bitstreams is about **120 MP/s** (smooth) / **70 MP/s** (busy) vs ~150 MP/s in `djxl`. Hybrid-uint tokens of small AC use a 256-entry LUT, and decode packing scales jxl-oxide’s 0..=1 display samples to u8 without a second sRGB curve.

## Build

```bash
cargo build --release
```

Run tests (SIMD IDCT / quantize plus a sample decode):

```bash
cargo test
```

### WebAssembly

SIMD128 is enabled for `wasm32-unknown-unknown` via `.cargo/config.toml`. Disable the default features so the CLI and rayon thread pool are not pulled in (wasm32 has no host threads):

```bash
cargo build --release --target wasm32-unknown-unknown --no-default-features --lib
```

From other Rust-on-wasm crates, depend on `o3-jxl` and call [`Decoder::decode_bytes`](src/decoder.rs) or [`ProgressiveDecoder`](src/progressive.rs) — no filesystem required.

```toml
o3-jxl = { path = "../o3-jxl", default-features = false }
```

### Progressive demo (Trunk)

Same idea as the DIMG demo: load a JXL, then scrub any prefix of the bitstream.

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk
# wasm32 linking uses Homebrew `wasm-ld` (see `.cargo/config.toml`)
NO_COLOR=true trunk serve
```

Opens `http://127.0.0.1:8081`. Default still is `samples/jxl/dice.jxl`; Open another `.jxl` to add it to the session list. (`NO_COLOR=true` avoids a Trunk 0.21 / clap clash when `NO_COLOR=1` is set in the environment.)

## Usage

```bash
# Encode (cjxl-like flags). Default distance 0 = lossless.
cargo run --release -- encode samples/dice.png /tmp/dice.jxl
cargo run --release -- encode -q 90 -e 7 samples/dice.png /tmp/dice-q90.jxl
cargo run --release -- encode -d 1.0 -m 0 samples/dice.png /tmp/dice-vd.jxl
cargo run --release -- encode -d 1.0 -p samples/dice.png /tmp/dice-p.jxl
cargo run --release -- encode -R 1 --modular-palette-colors 0 samples/dice.png /tmp/dice-sq.jxl

# Decode (djxl-like). Format from extension: png, apng, pam, ppm, pgm, pfm, pgx, jpg, exif, xmp
cargo run --release -- decode samples/jxl/dice.jxl out.png
cargo run --release -- decode samples/jxl/dice.jxl out.pfm
cargo run --release -- decode samples/jxl/dice.jxl out.pgx
cargo run --release -- decode --bits_per_sample 16 samples/jxl/dice.jxl out.png
cargo run --release -- decode --allow_partial_files truncated.jxl preview.png

# Metadata (jxlinfo-like)
cargo run --release -- info samples/jxl/dice.jxl

# Truncated / progressive prefix
cargo run --release -- decode samples/jxl/dice.jxl --max-bytes 8000 preview.png

# Benchmark
cargo run --release -- bench samples/jxl/dice.jxl

# Animation — one PNG per frame
cargo run --release -- decode samples/jxl/anim-icos.jxl --all-frames
```

### vs cjxl / djxl

| Feature | o3-jxl | libjxl |
|---------|--------|--------|
| Decode any JXL (VarDCT, modular, animation, HDR) | yes (jxl-oxide) | yes |
| Truncated / progressive render | yes | `--allow_partial_files` |
| PNG / PAM / PPM / PGM / PFM / PGX / APNG out | yes | yes (also EXR) |
| JPEG reconstruction | yes (when the file has jbrd) | yes |
| EXIF / XMP extract | yes | yes |
| EXR out | no | yes |
| Modular lossless encode | yes | yes |
| Modular lossy (`-q` / `-d`) | yes (quantized residuals) | yes (richer) |
| VarDCT lossy encode (`-m 0`) | yes (8×8, Gabor/EPF off) | yes (varblocks, Gabor/EPF) |
| Effort 1–10 / `-p` groups | yes | yes |
| Squeeze (`-R` / `--responsive`) | yes (default params) | yes |
| Palette (`--modular-palette-colors`) | yes (lossless) | yes (lossy too) |
| 16-bit integer encode | yes | yes |
| 16-bit PNG / PAM out (`--bits_per_sample`) | yes | yes |
| Animation encode (APNG) | yes (modular) | yes |
| JPEG input (re-encode pixels) | yes | yes |
| JPEG lossless transcode (`-j`) | no | yes |

Compare with libjxl:

```bash
djxl samples/jxl/dice.jxl dice-djxl.png
```

## Library

```rust
use o3_jxl::{Decoder, Encoder, PixelFormat, write_png};

let lossless = Encoder::new().encode_rgb8(width, height, &rgb)?;
let q90 = Encoder::new().quality(90.0).effort(7).encode_rgb8(width, height, &rgb)?;
let modular_lossy = Encoder::new().distance(1.0).modular(Some(true)).encode_rgb8(width, height, &rgb)?;
let squeezed = Encoder::new().responsive(Some(true)).encode_rgb8(width, height, &rgb)?;
let anim = Encoder::new().encode_animation(width, height, PixelFormat::Rgb, 1000, 1, 0, &[(40, &frame0), (40, &frame1)])?;

let frames = Decoder::new()
    .request_srgb(true)
    .decode_file("samples/jxl/dice.jxl")?;

write_png("out.png", &frames[0])?;

// Any prefix — None until the header + some groups exist
let preview = o3_jxl::ProgressiveDecoder::decode_prefix(&jxl_bytes, 8_000, true)?;

// WASM / in-memory
let frames = Decoder::new()
    .request_srgb(true)
    .decode_bytes(&jxl_bytes)?;
```

## Project layout

```
o3-jxl/
├── Cargo.toml
├── rust-toolchain.toml   # nightly (portable_simd)
├── src/
│   ├── lib.rs            # Public API
│   ├── main.rs           # CLI (feature `cli`)
│   ├── decoder.rs        # Decode orchestration
│   ├── encoder/          # Modular encode (YCoCg, Squeeze, palette, Gradient)
│   └── simd/             # std::simd DCT, color, Gaborish, sRGB/RGBA kernels
└── samples/
    └── jxl/              # Committed JXL test inputs (git)
```

## Samples

`samples/jxl/` contains the JXL files tracked in git:

| File | Description |
|------|-------------|
| `dice.jxl` | 800×600 still image with alpha |
| `parrot.jxl` | JPEG XL macaw (cjxl `-q 90 -e 10 -m 1 -p`) |
| `kodim23.jxl` | Kodak kodim23, same encode settings |
| `avenches.jxl` | Avenches (IENA) from [Valin’s Daala compare](https://jmvalin.ca/daala/revisiting/compare.html) |
| `bike.jxl` | Bike, same source set |
| `crepuscular.jxl` | Crepuscular, same source set |
| `woman.jxl` | Woman, same source set |
| `anim-icos.jxl` | 48-frame animation |
| `Webkit-logo-P3.jxl` | Wide-gamut / P3 test |
| `zoltan-tasi-CLJeQCr2F_A-unsplash.jxl` | Large photo |

Other formats (PNG, WebP, JPEG, etc.) and decode outputs may exist locally under `samples/` but are gitignored.

## License

MIT OR Apache-2.0
