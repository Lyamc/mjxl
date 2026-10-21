use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{ArgGroup, Parser, Subcommand};
use o3_jxl::{
    Decoder, Encoder, PixelFormat, ProgressiveDecoder, extract_exif, extract_xmp,
    quality_to_distance, reconstruct_jpeg, write_apng, write_pam_depth, write_pfm, write_pgm,
    write_png, write_png_depth, write_ppm,
};

#[derive(Parser)]
#[command(
    name = "o3-jxl",
    version,
    about = "JPEG XL codec (std::simd). Subcommands mirror cjxl/djxl flags where implemented."
)]
struct Cli {
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
    #[arg(long, global = true)]
    quiet: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Decode a JXL file (djxl-compatible flags).
    Decode {
        /// Input .jxl (`-` = stdin).
        input: PathBuf,
        /// Output path; format from extension unless `--output_format` (default: `.png`).
        output: Option<PathBuf>,
        /// png, apng, pam, ppm, pgm, pnm, pfm, jpg, jpeg, exif, xmp.
        #[arg(long = "output_format", alias = "output-format")]
        output_format: Option<String>,
        /// Output bit depth (djxl `--bits_per_sample`). 0 = from the file. 8 or 16.
        #[arg(
            long = "bits_per_sample",
            alias = "bits-per-sample",
            default_value_t = 0
        )]
        bits_per_sample: i32,
        /// Decode all animation frames (numbered PNGs).
        #[arg(long = "all_frames", alias = "all-frames")]
        all_frames: bool,
        /// Decode only the first N bytes (truncated / progressive prefix).
        #[arg(long)]
        max_bytes: Option<usize>,
        /// Allow truncated files (same as `--max_bytes` of the full file).
        #[arg(long = "allow_partial_files", alias = "allow-partial-files")]
        allow_partial_files: bool,
        /// Decode pixels instead of reconstructing a transcoded JPEG.
        #[arg(short = 'j', long)]
        pixels_to_jpeg: bool,
    },
    /// Print image metadata (jxlinfo-like).
    Info { input: PathBuf },
    /// Benchmark decode throughput (`djxl` --num_reps analogue).
    Bench {
        input: PathBuf,
        #[arg(short, long, default_value_t = 3)]
        iterations: u32,
    },
    /// Encode an image to JXL (cjxl-compatible flags).
    #[command(group(ArgGroup::new("qual").args(["quality", "distance"])))]
    Encode {
        /// Input PNG / APNG / JPEG / PAM / PPM / PGM.
        input: PathBuf,
        /// Output .jxl (default: input name with .jxl).
        output: Option<PathBuf>,
        /// Quality 0..=100. 100 = lossless. Maps to `--distance` like cjxl.
        #[arg(short = 'q', long)]
        quality: Option<f32>,
        /// Butteraugli distance. 0 = lossless. Default 0 (unlike cjxl's 1.0).
        #[arg(short = 'd', long)]
        distance: Option<f32>,
        /// Effort 1..=10 (default 7). Higher = smaller groups / LZ77.
        #[arg(short = 'e', long, default_value_t = 7)]
        effort: u8,
        /// Smaller groups for progressive / truncated decode.
        #[arg(short = 'p', long)]
        progressive: bool,
        /// 1 = modular. 0 = VarDCT (lossy RGB). Omit to auto (VarDCT when `-d`/`-q` is lossy RGB).
        #[arg(short = 'm', long)]
        modular: Option<u8>,
        /// Squeeze (`cjxl -R` / `--responsive`). Omit for auto (lossy or effort ≥ 8).
        #[arg(short = 'R', long)]
        responsive: Option<u8>,
        /// Max palette colours (0 = disable). Default 256.
        #[arg(
            long = "modular_palette_colors",
            alias = "modular-palette-colors",
            default_value_t = 256
        )]
        modular_palette_colors: u32,
    },
}

fn main() -> o3_jxl::Result<()> {
    let cli = Cli::parse();
    let quiet = cli.quiet;
    match cli.command {
        Commands::Decode {
            input,
            output,
            output_format,
            bits_per_sample,
            all_frames,
            max_bytes,
            allow_partial_files,
            pixels_to_jpeg,
        } => cmd_decode(
            input,
            output,
            output_format,
            bits_per_sample,
            all_frames,
            max_bytes,
            allow_partial_files,
            pixels_to_jpeg,
            quiet,
        ),
        Commands::Info { input } => cmd_info(input),
        Commands::Bench { input, iterations } => cmd_bench(input, iterations, quiet),
        Commands::Encode {
            input,
            output,
            quality,
            distance,
            effort,
            progressive,
            modular,
            responsive,
            modular_palette_colors,
        } => cmd_encode(
            input,
            output,
            quality,
            distance,
            effort,
            progressive,
            modular,
            responsive,
            modular_palette_colors,
            quiet,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_decode(
    input: PathBuf,
    output: Option<PathBuf>,
    output_format: Option<String>,
    bits_per_sample: i32,
    all_frames: bool,
    max_bytes: Option<usize>,
    allow_partial_files: bool,
    pixels_to_jpeg: bool,
    quiet: bool,
) -> o3_jxl::Result<()> {
    let data = if input.as_os_str() == "-" {
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        std::fs::read(&input)?
    };

    let out = output.unwrap_or_else(|| {
        if input.as_os_str() == "-" {
            PathBuf::from("out.png")
        } else {
            input.with_extension("png")
        }
    });
    let fmt = output_format
        .as_deref()
        .or_else(|| out.extension().and_then(|s| s.to_str()))
        .unwrap_or("png")
        .to_ascii_lowercase();

    if matches!(fmt.as_str(), "jpg" | "jpeg") && !pixels_to_jpeg && input.as_os_str() != "-" {
        match reconstruct_jpeg(&input, &out) {
            Ok(()) => {
                if !quiet {
                    eprintln!("Reconstructed JPEG → {}", out.display());
                }
                return Ok(());
            }
            Err(_) => {
                if !quiet {
                    eprintln!("JPEG reconstruction unavailable; decoding pixels");
                }
            }
        }
    }
    if fmt == "exif" {
        let some = extract_exif(&input)?;
        match some {
            Some(bytes) => std::fs::write(&out, bytes)?,
            None => {
                return Err(o3_jxl::Error::Encode("no EXIF box in file".into()));
            }
        }
        return Ok(());
    }
    if matches!(fmt.as_str(), "xmp" | "xml") {
        let some = extract_xmp(&input)?;
        match some {
            Some(bytes) => std::fs::write(&out, bytes)?,
            None => return Err(o3_jxl::Error::Encode("no XMP/XML box in file".into())),
        }
        return Ok(());
    }

    let start = Instant::now();
    let want_srgb = !matches!(fmt.as_str(), "pfm");
    let src_bits = Decoder::probe_bytes(&data)
        .map(|i| i.bit_depth)
        .unwrap_or(8);
    let frames = if let Some(max_bytes) = max_bytes {
        let n = max_bytes.min(data.len());
        let frame =
            ProgressiveDecoder::decode_prefix(&data, n, true)?.ok_or(o3_jxl::Error::NoFrames)?;
        if !quiet {
            eprintln!("Using {n} / {} bytes", data.len());
        }
        vec![frame]
    } else if allow_partial_files {
        let frame = ProgressiveDecoder::decode_prefix(&data, data.len(), true)?
            .ok_or(o3_jxl::Error::NoFrames)?;
        vec![frame]
    } else {
        Decoder::new()
            .request_srgb(want_srgb)
            .keep_linear(matches!(fmt.as_str(), "pfm") || src_bits > 8 || bits_per_sample >= 16)
            .decode_bytes(&data)?
    };
    let elapsed = start.elapsed();

    if !quiet {
        let mpix = frames[0].pixel_count() as f64 / 1e6;
        eprintln!(
            "Decoded {} frame(s) in {:.2} ms ({:.2} MP/s)",
            frames.len(),
            elapsed.as_secs_f64() * 1000.0,
            mpix / elapsed.as_secs_f64()
        );
    }

    let bits = if bits_per_sample <= 0 {
        if matches!(fmt.as_str(), "pam" | "ppm" | "pgm" | "pnm") {
            8
        } else if src_bits > 8 {
            16
        } else {
            8
        }
    } else if bits_per_sample >= 16 {
        16
    } else {
        8
    };

    if all_frames && frames.len() > 1 {
        let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let parent = out.parent().unwrap_or(Path::new(".")).to_path_buf();
        for (i, frame) in frames.iter().enumerate() {
            let path = parent.join(format!("{stem}-{i:04}.png"));
            write_png_depth(&path, frame, bits)?;
            if !quiet {
                eprintln!("Wrote {}", path.display());
            }
        }
    } else if frames.len() > 1 && matches!(fmt.as_str(), "png" | "apng") {
        let tps = 1000;
        write_apng(&out, &frames, tps)?;
        if !quiet {
            eprintln!("Wrote {} ({} frames)", out.display(), frames.len());
        }
    } else {
        write_decoded(&out, &fmt, &frames[0], bits)?;
        if !quiet {
            eprintln!("Wrote {}", out.display());
        }
    }

    Ok(())
}

fn write_decoded(
    path: &Path,
    fmt: &str,
    frame: &o3_jxl::DecodedFrame,
    bits: u8,
) -> o3_jxl::Result<()> {
    match fmt {
        "png" | "apng" => write_png_depth(path, frame, bits),
        "pam" => write_pam_depth(path, frame, bits),
        "ppm" | "pnm" => write_ppm(path, frame),
        "pgm" => write_pgm(path, frame),
        "pfm" => write_pfm(path, frame),
        "pgx" => o3_jxl::write_pgx(path, frame),
        "jpg" | "jpeg" => write_png(path, frame),
        other => Err(o3_jxl::Error::Encode(format!(
            "unsupported output format '{other}' (png, apng, pam, ppm, pgm, pnm, pfm, jpg, exif, xmp)"
        ))),
    }
}

fn cmd_info(input: PathBuf) -> o3_jxl::Result<()> {
    let info = Decoder::probe(&input)?;
    println!("{}×{}", info.width, info.height);
    println!("keyframes: {}", info.num_keyframes);
    println!("alpha: {}", info.has_alpha);
    println!("bit depth: {}", info.bit_depth);
    println!("xyb: {}", info.xyb_encoded);
    println!("encoding: {}", info.encoding);
    println!("icc: {}", info.has_icc);
    println!("jpeg reconstruction: {}", info.jpeg_reconstruction);
    if let Some(tps) = info.animation_tps {
        println!("animation tps: {tps}");
    }
    Ok(())
}

fn cmd_bench(input: PathBuf, iterations: u32, quiet: bool) -> o3_jxl::Result<()> {
    let data = std::fs::read(&input)?;
    let info = Decoder::probe_bytes(&data)?;
    let pixels = info.width as u64 * info.height as u64;
    let decoder = Decoder::new().request_srgb(true).keep_linear(false);
    let mut times = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = decoder.decode_bytes(&data)?;
        times.push(start.elapsed());
    }
    let best = times.iter().min().unwrap().as_secs_f64();
    let mpix = pixels as f64 / 1e6;
    if !quiet {
        eprintln!("best {:.2} ms ({:.2} MP/s)", best * 1000.0, mpix / best);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_encode(
    input: PathBuf,
    output: Option<PathBuf>,
    quality: Option<f32>,
    distance: Option<f32>,
    effort: u8,
    progressive: bool,
    modular: Option<u8>,
    responsive: Option<u8>,
    modular_palette_colors: u32,
    quiet: bool,
) -> o3_jxl::Result<()> {
    if modular == Some(0) && (quality == Some(100.0) || distance == Some(0.0)) {
        return Err(o3_jxl::Error::Encode(
            "VarDCT requires a positive distance (lossy); use -d or -q < 100".into(),
        ));
    }
    let image = read_image_pixels(&input)?;
    let mut enc = Encoder::new()
        .effort(effort)
        .progressive(progressive)
        .palette_colors(modular_palette_colors)
        .modular(modular.map(|m| m != 0));
    if let Some(r) = responsive {
        enc = enc.responsive(Some(r != 0));
    }
    if let Some(q) = quality {
        enc = enc.quality(q);
    } else if let Some(d) = distance {
        enc = enc.distance(d);
    }
    let start = Instant::now();
    let (width, height, format, nframes, jxl) = match image {
        InputImage::U8 {
            width,
            height,
            format,
            pixels,
        } => (
            width,
            height,
            format,
            1usize,
            enc.encode(width, height, format, &pixels)?,
        ),
        InputImage::U16 {
            width,
            height,
            format,
            pixels,
        } => (
            width,
            height,
            format,
            1usize,
            enc.encode_u16(width, height, format, &pixels)?,
        ),
        InputImage::Anim {
            width,
            height,
            format,
            tps_num,
            tps_den,
            num_loops,
            frames,
        } => {
            let refs: Vec<(u32, &[u8])> = frames.iter().map(|(d, p)| (*d, p.as_slice())).collect();
            (
                width,
                height,
                format,
                frames.len(),
                enc.encode_animation(width, height, format, tps_num, tps_den, num_loops, &refs)?,
            )
        }
    };
    let elapsed = start.elapsed();
    let out = output.unwrap_or_else(|| input.with_extension("jxl"));
    std::fs::write(&out, &jxl)?;
    if !quiet {
        let npx = width as f64 * height as f64 * nframes as f64;
        let bpp = (jxl.len() as f64) * 8.0 / npx;
        let d = quality.map(quality_to_distance).or(distance).unwrap_or(0.0);
        eprintln!(
            "Encoded {}×{} {:?} ×{nframes} d={d:.3} e={effort} in {:.2} ms ({:.3} bpp, {} bytes) → {}",
            width,
            height,
            format,
            elapsed.as_secs_f64() * 1000.0,
            bpp,
            jxl.len(),
            out.display()
        );
    }
    Ok(())
}

enum InputImage {
    U8 {
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: Vec<u8>,
    },
    U16 {
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: Vec<u16>,
    },
    Anim {
        width: u32,
        height: u32,
        format: PixelFormat,
        tps_num: u32,
        tps_den: u32,
        num_loops: u32,
        frames: Vec<(u32, Vec<u8>)>,
    },
}

fn read_image_pixels(path: &Path) -> o3_jxl::Result<InputImage> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" | "apng" | "" => read_png_pixels(path),
        "jpg" | "jpeg" => read_jpeg_pixels(path),
        "ppm" | "pgm" | "pam" | "pnm" | "pbm" => read_pnm_pixels(path),
        other => Err(o3_jxl::Error::Encode(format!(
            "unsupported input '{other}' (png, apng, jpeg, ppm, pgm, pam, pnm)"
        ))),
    }
}

fn read_png_pixels(path: &Path) -> o3_jxl::Result<InputImage> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| o3_jxl::Error::Encode(e.to_string()))?;
    #[allow(clippy::collapsible_if)]
    if let Some(ac) = reader.info().animation_control {
        if ac.num_frames > 1 {
            return read_apng_frames(reader, ac);
        }
    }
    let mut buf = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or_else(|| o3_jxl::Error::Encode("PNG dimensions overflow".into()))?
    ];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| o3_jxl::Error::Encode(e.to_string()))?;
    buf.truncate(info.buffer_size());
    let format = match info.color_type {
        png::ColorType::Grayscale => PixelFormat::Gray,
        png::ColorType::GrayscaleAlpha => PixelFormat::GrayA,
        png::ColorType::Rgb => PixelFormat::Rgb,
        png::ColorType::Rgba => PixelFormat::Rgba,
        other => {
            return Err(o3_jxl::Error::Encode(format!(
                "unsupported PNG color type {other:?}"
            )));
        }
    };
    match info.bit_depth {
        png::BitDepth::Eight => Ok(InputImage::U8 {
            width: info.width,
            height: info.height,
            format,
            pixels: buf,
        }),
        png::BitDepth::Sixteen => {
            let mut px = vec![0u16; buf.len() / 2];
            for (i, p) in px.iter_mut().enumerate() {
                *p = u16::from_ne_bytes([buf[i * 2], buf[i * 2 + 1]]);
            }
            Ok(InputImage::U16 {
                width: info.width,
                height: info.height,
                format,
                pixels: px,
            })
        }
        other => Err(o3_jxl::Error::Encode(format!(
            "unsupported PNG bit depth {other:?}"
        ))),
    }
}

fn read_jpeg_pixels(path: &Path) -> o3_jxl::Result<InputImage> {
    let file = std::fs::File::open(path)?;
    let mut decoder = jpeg_decoder::Decoder::new(std::io::BufReader::new(file));
    let pixels = decoder
        .decode()
        .map_err(|e| o3_jxl::Error::Encode(e.to_string()))?;
    let info = decoder
        .info()
        .ok_or_else(|| o3_jxl::Error::Encode("JPEG has no image info".into()))?;
    let format = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => PixelFormat::Gray,
        jpeg_decoder::PixelFormat::RGB24 => PixelFormat::Rgb,
        other => {
            return Err(o3_jxl::Error::Encode(format!(
                "unsupported JPEG pixel format {other:?}"
            )));
        }
    };
    Ok(InputImage::U8 {
        width: info.width as u32,
        height: info.height as u32,
        format,
        pixels,
    })
}

fn read_apng_frames(
    mut reader: png::Reader<std::io::BufReader<std::fs::File>>,
    ac: png::AnimationControl,
) -> o3_jxl::Result<InputImage> {
    let width = reader.info().width;
    let height = reader.info().height;
    let color = reader.info().color_type;
    let mut canvas = vec![0u8; width as usize * height as usize * 4];
    let mut frames = Vec::with_capacity(ac.num_frames as usize);
    for i in 0..ac.num_frames {
        if i > 0 {
            reader
                .next_frame_info()
                .map_err(|e| o3_jxl::Error::Encode(e.to_string()))?;
        }
        let fctl = reader.info().frame_control.unwrap_or_default();
        let mut buf =
            vec![
                0;
                reader
                    .output_buffer_size()
                    .ok_or_else(|| o3_jxl::Error::Encode("PNG dimensions overflow".into()))?
            ];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| o3_jxl::Error::Encode(e.to_string()))?;
        buf.truncate(info.buffer_size());
        let fw = info.width;
        let fh = info.height;
        let src = expand_png_to_rgba(&buf, fw, fh, color);
        let snapshot = canvas.clone();
        blend_over(
            &mut canvas,
            width,
            &src,
            fw,
            fh,
            fctl.x_offset,
            fctl.y_offset,
            fctl.blend_op,
        );
        let den = if fctl.delay_den == 0 {
            100
        } else {
            u32::from(fctl.delay_den)
        };
        let ticks = (u32::from(fctl.delay_num) * 1000 / den).max(1);
        frames.push((ticks, canvas.clone()));
        match fctl.dispose_op {
            png::DisposeOp::None => {}
            png::DisposeOp::Background => {
                clear_rect(&mut canvas, width, fctl.x_offset, fctl.y_offset, fw, fh);
            }
            png::DisposeOp::Previous => canvas = snapshot,
        }
    }
    Ok(InputImage::Anim {
        width,
        height,
        format: PixelFormat::Rgba,
        tps_num: 1000,
        tps_den: 1,
        num_loops: ac.num_plays,
        frames,
    })
}

fn expand_png_to_rgba(buf: &[u8], w: u32, h: u32, color: png::ColorType) -> Vec<u8> {
    let npx = w as usize * h as usize;
    let mut out = vec![255u8; npx * 4];
    match color {
        png::ColorType::Rgba => out.copy_from_slice(buf),
        png::ColorType::Rgb => {
            for i in 0..npx {
                out[i * 4] = buf[i * 3];
                out[i * 4 + 1] = buf[i * 3 + 1];
                out[i * 4 + 2] = buf[i * 3 + 2];
            }
        }
        png::ColorType::Grayscale => {
            for i in 0..npx {
                let g = buf[i];
                out[i * 4] = g;
                out[i * 4 + 1] = g;
                out[i * 4 + 2] = g;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..npx {
                let g = buf[i * 2];
                out[i * 4] = g;
                out[i * 4 + 1] = g;
                out[i * 4 + 2] = g;
                out[i * 4 + 3] = buf[i * 2 + 1];
            }
        }
        _ => {}
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn blend_over(
    canvas: &mut [u8],
    cw: u32,
    src: &[u8],
    fw: u32,
    fh: u32,
    x0: u32,
    y0: u32,
    op: png::BlendOp,
) {
    let cw = cw as usize;
    for y in 0..fh as usize {
        for x in 0..fw as usize {
            let dx = x0 as usize + x;
            let dy = y0 as usize + y;
            let di = (dy * cw + dx) * 4;
            let si = (y * fw as usize + x) * 4;
            if di + 3 >= canvas.len() || si + 3 >= src.len() {
                continue;
            }
            match op {
                png::BlendOp::Source => {
                    canvas[di..di + 4].copy_from_slice(&src[si..si + 4]);
                }
                png::BlendOp::Over => {
                    let a = src[si + 3] as u16;
                    if a == 255 {
                        canvas[di..di + 4].copy_from_slice(&src[si..si + 4]);
                    } else if a > 0 {
                        let ia = 255 - a;
                        for c in 0..3 {
                            canvas[di + c] =
                                ((src[si + c] as u16 * a + canvas[di + c] as u16 * ia) / 255) as u8;
                        }
                        canvas[di + 3] = canvas[di + 3].max(src[si + 3]);
                    }
                }
            }
        }
    }
}

fn clear_rect(canvas: &mut [u8], cw: u32, x0: u32, y0: u32, fw: u32, fh: u32) {
    let cw = cw as usize;
    for y in 0..fh as usize {
        for x in 0..fw as usize {
            let di = ((y0 as usize + y) * cw + x0 as usize + x) * 4;
            if di + 3 < canvas.len() {
                canvas[di..di + 4].fill(0);
            }
        }
    }
}

fn read_pnm_pixels(path: &Path) -> o3_jxl::Result<InputImage> {
    let data = std::fs::read(path)?;
    parse_pnm(&data)
}

fn parse_pnm(data: &[u8]) -> o3_jxl::Result<InputImage> {
    let mut i = 0usize;
    skip_ws_comments(data, &mut i);
    if i + 2 > data.len() || data[i] != b'P' {
        return Err(o3_jxl::Error::Encode("not a PNM file".into()));
    }
    let kind = data[i + 1];
    i += 2;
    skip_ws_comments(data, &mut i);
    if kind == b'7' {
        return parse_pam(data, i);
    }
    let w = parse_u32(data, &mut i)?;
    skip_ws_comments(data, &mut i);
    let h = parse_u32(data, &mut i)?;
    skip_ws_comments(data, &mut i);
    let maxv = parse_u32(data, &mut i)?;
    if maxv != 255 {
        return Err(o3_jxl::Error::Encode(
            "only maxval 255 PNM is supported".into(),
        ));
    }
    if i < data.len() && data[i] == b'\n' {
        i += 1;
    }
    let npx = w as usize * h as usize;
    match kind {
        b'5' => {
            let pixels = data[i..]
                .get(..npx)
                .ok_or_else(|| o3_jxl::Error::Encode("truncated PGM".into()))?;
            Ok(InputImage::U8 {
                width: w,
                height: h,
                format: PixelFormat::Gray,
                pixels: pixels.to_vec(),
            })
        }
        b'6' => {
            let n = npx * 3;
            let pixels = data[i..]
                .get(..n)
                .ok_or_else(|| o3_jxl::Error::Encode("truncated PPM".into()))?;
            Ok(InputImage::U8 {
                width: w,
                height: h,
                format: PixelFormat::Rgb,
                pixels: pixels.to_vec(),
            })
        }
        _ => Err(o3_jxl::Error::Encode(
            "unsupported PNM (need P5/P6/P7 binary)".into(),
        )),
    }
}

fn parse_pam(data: &[u8], mut i: usize) -> o3_jxl::Result<InputImage> {
    let mut w = 0u32;
    let mut h = 0u32;
    let mut depth = 0u32;
    let mut maxv = 0u32;
    loop {
        skip_ws_comments(data, &mut i);
        let line_end = data[i..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| i + p)
            .unwrap_or(data.len());
        let line = std::str::from_utf8(&data[i..line_end]).unwrap_or("");
        i = (line_end + 1).min(data.len());
        if line.starts_with("ENDHDR") {
            break;
        }
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next()) {
            (Some("WIDTH"), Some(v)) => w = v.parse().unwrap_or(0),
            (Some("HEIGHT"), Some(v)) => h = v.parse().unwrap_or(0),
            (Some("DEPTH"), Some(v)) => depth = v.parse().unwrap_or(0),
            (Some("MAXVAL"), Some(v)) => maxv = v.parse().unwrap_or(0),
            _ => {}
        }
    }
    if maxv != 255 {
        return Err(o3_jxl::Error::Encode(
            "only maxval 255 PAM is supported".into(),
        ));
    }
    let n = w as usize * h as usize * depth as usize;
    let pixels = data[i..]
        .get(..n)
        .ok_or_else(|| o3_jxl::Error::Encode("truncated PAM".into()))?
        .to_vec();
    let format = match depth {
        1 => PixelFormat::Gray,
        2 => PixelFormat::GrayA,
        3 => PixelFormat::Rgb,
        4 => PixelFormat::Rgba,
        _ => {
            return Err(o3_jxl::Error::Encode(format!(
                "unsupported PAM depth {depth}"
            )));
        }
    };
    Ok(InputImage::U8 {
        width: w,
        height: h,
        format,
        pixels,
    })
}

fn skip_ws_comments(data: &[u8], i: &mut usize) {
    loop {
        while *i < data.len() && data[*i].is_ascii_whitespace() {
            *i += 1;
        }
        if *i < data.len() && data[*i] == b'#' {
            while *i < data.len() && data[*i] != b'\n' {
                *i += 1;
            }
            continue;
        }
        break;
    }
}

fn parse_u32(data: &[u8], i: &mut usize) -> o3_jxl::Result<u32> {
    skip_ws_comments(data, i);
    let start = *i;
    while *i < data.len() && data[*i].is_ascii_digit() {
        *i += 1;
    }
    std::str::from_utf8(&data[start..*i])
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| o3_jxl::Error::Encode("invalid PNM number".into()))
}
