use std::path::PathBuf;
use std::time::Instant;

// Frame-parallel encoding allocates hundreds of short-lived buffers per
// frame; a thread-local allocator keeps those off the global malloc lock.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::{Parser, Subcommand, ValueEnum};
use mjxl::compare::{self, Coding, CompareOpts};
use mjxl::container::{self, FLAG_LOSSLESS, FLAG_PROGRESSIVE};
use mjxl::decode::{self, DecodeOpts};
use mjxl::gpu::{self, Backend};
use mjxl::progressive;
use mjxl::still;
use mjxl::temporal::{self, SequenceOpts};
use mjxl::y4m;
use o3_jxl::{Encoder, PixelFormat};

#[derive(Parser)]
#[command(
    name = "mjxl",
    version,
    about = "Motion JPEG XL: independent JXL frames, dirty rectangles, modular / VarDCT / progressive."
)]
struct Cli {
    /// Emit machine-readable JSON on stdout (compare/bench/stills/info/verify).
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode a Y4M into `.mjxl` (complete JXL stills, or dirty rectangles).
    Encode {
        input: PathBuf,
        output: Option<PathBuf>,
        /// Butteraugli distance. 0 = lossless modular. Default 1.
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        /// Encoder effort 1..=10 (`cjxl -e`). Default 2. Effort 7 is the full path.
        #[arg(short = 'e', long, default_value_t = o3_jxl::DEFAULT_EFFORT)]
        effort: u8,
        /// `vardct` (lossy RGB, JPEG-like) or `modular` (YCoCg-R + Gradient).
        #[arg(short = 'm', long, value_enum, default_value_t = StillCoding::Vardct)]
        coding: StillCoding,
        /// Disable skip / dirty-rect (MJPEG-style all-intra).
        #[arg(long)]
        intra: bool,
        /// Smaller groups, and Squeeze on lossy modular, so a prefix can paint.
        #[arg(long)]
        progressive: bool,
        /// Diff against the previous reconstruction instead of the source frame.
        #[arg(long)]
        closed_loop: bool,
        /// `auto` uses Vulkan compute for the dirty search when an adapter exists.
        #[arg(long, value_enum, default_value_t = BackendCli::Auto)]
        backend: BackendCli,
    },
    /// Decode `.mjxl` to 4:4:4 Y4M.
    Decode {
        input: PathBuf,
        output: Option<PathBuf>,
        /// Render this fraction of each JXL sample (progressive). Omit for a full decode.
        #[arg(long)]
        prefix: Option<f32>,
        #[arg(long, value_enum, default_value_t = BackendCli::Auto)]
        backend: BackendCli,
    },
    /// VarDCT-style progressive passes of a Y4M: DC, 2×2 low frequency, full 8×8.
    Levels {
        input: PathBuf,
        /// Directory for `{stem}-dc.y4m`, `-lf.y4m`, and `-full.y4m`.
        output: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = BackendCli::Auto)]
        backend: BackendCli,
    },
    Info {
        input: PathBuf,
    },
    /// Encode the same clip with MJPEG, JPEG 2000, and mjxl; print size / time / PSNR.
    Compare {
        input: PathBuf,
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        /// Encoder effort, same scale as `cjxl -e`. Default 2. Effort 7 is the full path.
        #[arg(short = 'e', long, default_value_t = o3_jxl::DEFAULT_EFFORT)]
        effort: u8,
        /// FFmpeg MJPEG `-q:v` (2=finest). 3 ≈ JPEG quality 90 on this encoder.
        #[arg(long, default_value_t = 3)]
        mjpeg_q: i32,
        #[arg(long, default_value_t = 5)]
        j2k_q: i32,
        #[arg(long)]
        work_dir: Option<PathBuf>,
        /// Keep the temporary work directory instead of removing it.
        #[arg(long)]
        keep_work_dir: bool,
        /// Which still coder(s) to measure. Default both.
        #[arg(short = 'm', long, value_enum, default_value_t = CodingCli::Both)]
        coding: CodingCli,
    },
    /// In-process, median-of-N encode/decode bench for one still coder.
    Bench {
        input: PathBuf,
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        #[arg(short = 'e', long, default_value_t = o3_jxl::DEFAULT_EFFORT)]
        effort: u8,
        #[arg(short = 'm', long, value_enum, default_value_t = StillCoding::Modular)]
        coding: StillCoding,
        #[arg(long, default_value_t = 5)]
        repeats: u32,
    },
    /// Native JXL animation, using patches for repeated frames.
    Anim {
        input: PathBuf,
        output: Option<PathBuf>,
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        #[arg(short = 'e', long, default_value_t = o3_jxl::DEFAULT_EFFORT)]
        effort: u8,
    },
    /// Encode → decode a Y4M in-process and report round-trip PSNR.
    Verify {
        input: PathBuf,
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        #[arg(short = 'e', long, default_value_t = o3_jxl::DEFAULT_EFFORT)]
        effort: u8,
        #[arg(short = 'm', long, value_enum, default_value_t = StillCoding::Modular)]
        coding: StillCoding,
    },
    /// Print a shell completion script to stdout.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    /// Still-image speed, bpp, and PSNR versus libjxl (`cjxl` / `djxl`).
    Stills {
        input: PathBuf,
        /// Butteraugli distance. 0 encodes lossless modular only.
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        /// Encoder effort, same scale as `cjxl -e`. 1 is fastest. Default 2.
        #[arg(short = 'e', long, default_value_t = o3_jxl::DEFAULT_EFFORT)]
        effort: u8,
        /// Timed repetitions after one warmup. The median is reported.
        #[arg(long, default_value_t = 3)]
        repeats: u32,
        #[arg(long)]
        work_dir: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum StillCoding {
    Vardct,
    Modular,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CodingCli {
    Vardct,
    Modular,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BackendCli {
    Auto,
    Cpu,
    Gpu,
}

impl BackendCli {
    fn backend(self) -> Backend {
        match self {
            BackendCli::Auto => Backend::Auto,
            BackendCli::Cpu => Backend::Cpu,
            BackendCli::Gpu => Backend::Gpu,
        }
    }
}

fn main() -> mjxl::Result<()> {
    let cli = Cli::parse();
    let json = cli.json;
    match cli.command {
        Commands::Encode {
            input,
            output,
            distance,
            effort,
            coding,
            intra,
            progressive,
            closed_loop,
            backend,
        } => cmd_encode(
            input,
            output,
            distance,
            effort,
            coding,
            intra,
            progressive,
            closed_loop,
            backend.backend(),
        ),
        Commands::Decode {
            input,
            output,
            prefix,
            backend,
        } => cmd_decode(input, output, prefix, backend.backend()),
        Commands::Levels {
            input,
            output,
            backend,
        } => cmd_levels(input, output, backend.backend()),
        Commands::Info { input } => cmd_info(input, json),
        Commands::Anim {
            input,
            output,
            distance,
            effort,
        } => cmd_anim(input, output, distance, effort),
        Commands::Verify {
            input,
            distance,
            effort,
            coding,
        } => cmd_verify(input, distance, effort, coding, json),
        Commands::Completions { shell } => {
            clap_complete::generate(
                shell,
                &mut <Cli as clap::CommandFactory>::command(),
                "mjxl",
                &mut std::io::stdout(),
            );
            Ok(())
        }
        Commands::Compare {
            input,
            distance,
            effort,
            mjpeg_q,
            j2k_q,
            work_dir,
            keep_work_dir,
            coding,
        } => cmd_compare(
            input,
            distance,
            effort,
            mjpeg_q,
            j2k_q,
            work_dir,
            keep_work_dir,
            coding,
            json,
        ),
        Commands::Bench {
            input,
            distance,
            effort,
            coding,
            repeats,
        } => cmd_bench(input, distance, effort, coding, repeats, json),
        Commands::Stills {
            input,
            distance,
            effort,
            repeats,
            work_dir,
        } => cmd_stills(input, distance, effort, repeats, work_dir, json),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn num_or_null(v: Option<f64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

fn print_json_rows(rows: &[compare::Row], pixels: u64) {
    let mut s = String::from("[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"codec\":{},\"bytes\":{},\"bpp\":{:.6},\"encode_ms\":{:.3},\"decode_ms\":{:.3},\"enc_mps\":{:.3},\"dec_mps\":{:.3},\"psnr\":{},\"note\":{}}}",
            json_str(&r.name),
            r.bytes,
            r.bytes as f64 * 8.0 / pixels.max(1) as f64,
            r.encode_ms,
            r.decode_ms,
            (pixels as f64 / 1e6) / (r.encode_ms / 1000.0).max(1e-9),
            (pixels as f64 / 1e6) / (r.decode_ms / 1000.0).max(1e-9),
            num_or_null(r.psnr),
            json_str(&r.note),
        ));
    }
    s.push(']');
    println!("{s}");
}

fn cmd_encode(
    input: PathBuf,
    output: Option<PathBuf>,
    distance: f32,
    effort: u8,
    coding: StillCoding,
    intra: bool,
    progressive: bool,
    closed_loop: bool,
    backend: Backend,
) -> mjxl::Result<()> {
    let mut reader = y4m::open_path(&input)?;
    let info = reader.info.clone();
    let mut frames = Vec::new();
    while let Some(rgb) = reader.next_rgb()? {
        frames.push(rgb);
    }
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let modular = matches!(coding, StillCoding::Modular) || distance <= 0.0;
    let mut enc = Encoder::new()
        .distance(distance)
        .effort(effort)
        .modular(Some(modular))
        .progressive(progressive);
    if progressive && modular && distance > 0.0 {
        enc = enc.responsive(Some(true));
    }
    let gpu = gpu::select(backend)?;
    let device = gpu.as_ref().map(|g| g.info.as_str()).unwrap_or("cpu");
    let t = Instant::now();
    let coded = temporal::encode_sequence(
        &enc,
        info.width,
        info.height,
        &frames,
        &SequenceOpts {
            intra,
            closed_loop,
            backend,
        },
    )?;
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut flags = if distance <= 0.0 { FLAG_LOSSLESS } else { 0 };
    if !intra {
        flags |= container::FLAG_TEMPORAL;
    }
    if modular {
        flags |= container::FLAG_MODULAR;
    }
    if progressive {
        flags |= FLAG_PROGRESSIVE;
    }
    let out = output.unwrap_or_else(|| input.with_extension("mjxl"));
    let mut f = std::fs::File::create(&out)?;
    container::write_coded(
        &mut f,
        flags,
        info.width,
        info.height,
        info.fps_num,
        info.fps_den,
        &coded,
    )?;
    let bytes: usize = coded
        .iter()
        .flat_map(|c| c.patches.iter())
        .map(|p| p.jxl.len())
        .sum();
    let nskip = coded.iter().filter(|c| c.is_skip()).count();
    let nrect = coded
        .iter()
        .filter(|c| !c.is_skip() && !c.is_full(info.width, info.height))
        .count();
    let npatch: usize = coded.iter().map(|c| c.patches.len()).sum();
    let pixels = frames.len() as f64 * info.width as f64 * info.height as f64;
    eprintln!(
        "Encoded {}×{} ×{} {} d={distance:.3} skip={nskip} rect={nrect} patches={npatch} dirty={device} in {:.2} ms ({:.1} MP/s, {:.3} bpp, {} payload bytes) → {}",
        info.width,
        info.height,
        frames.len(),
        if modular { "modular" } else { "vardct" },
        encode_ms,
        (pixels / 1e6) / (encode_ms / 1000.0).max(1e-9),
        (bytes as f64) * 8.0 / pixels,
        bytes,
        out.display()
    );
    Ok(())
}

fn cmd_decode(
    input: PathBuf,
    output: Option<PathBuf>,
    prefix: Option<f32>,
    backend: Backend,
) -> mjxl::Result<()> {
    let blob = std::fs::read(&input)?;
    let out = output.unwrap_or_else(|| input.with_extension("y4m"));
    let t = Instant::now();
    let decoded = decode::decode_rgb(&blob, &DecodeOpts { prefix, backend })?;
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut f = std::fs::File::create(&out)?;
    let info = y4m::Info {
        width: decoded.header.width,
        height: decoded.header.height,
        fps_num: decoded.header.fps_num,
        fps_den: decoded.header.fps_den,
        chroma: y4m::Chroma::C444,
        limited: false,
    };
    y4m::write_sequence(&mut f, &info, &decoded.frames)?;
    let pixels = decoded.frames.len() as f64 * info.width as f64 * info.height as f64;
    eprintln!(
        "Decoded {} frames in {:.2} ms ({:.1} MP/s) → {}",
        decoded.frames.len(),
        ms,
        (pixels / 1e6) / (ms / 1000.0).max(1e-9),
        out.display()
    );
    Ok(())
}

fn cmd_levels(input: PathBuf, output: Option<PathBuf>, backend: Backend) -> mjxl::Result<()> {
    let mut reader = y4m::open_path(&input)?;
    let info = reader.info.clone();
    let mut dc = Vec::new();
    let mut low = Vec::new();
    let mut full = Vec::new();
    let t = Instant::now();
    while let Some(rgb) = reader.next_rgb()? {
        let passes = progressive::frequency_passes(&rgb, info.width, info.height, backend)?;
        dc.push(passes.dc);
        low.push(passes.low);
        full.push(passes.full);
    }
    if dc.is_empty() {
        return Err("no frames".into());
    }
    let dir = output.unwrap_or_else(|| {
        input
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf()
    });
    std::fs::create_dir_all(&dir)?;
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("frame");
    for (name, frames) in [("dc", &dc), ("lf", &low), ("full", &full)] {
        let path = dir.join(format!("{stem}-{name}.y4m"));
        let mut f = std::fs::File::create(&path)?;
        y4m::write_sequence(&mut f, &info, frames)?;
        eprintln!("{} frames → {}", frames.len(), path.display());
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let gpu = gpu::select(backend)?;
    eprintln!(
        "Frequency passes via {} in {ms:.1} ms",
        gpu.as_ref().map(|g| g.info.as_str()).unwrap_or("cpu")
    );
    Ok(())
}

fn cmd_info(input: PathBuf, json: bool) -> mjxl::Result<()> {
    let blob = std::fs::read(&input)?;
    let h = container::read_header(&blob)?;
    let payload: u64 = h.samples.iter().map(|s| s.size as u64).sum();
    let mut flag_s = String::new();
    if h.flags & container::FLAG_LOSSLESS != 0 {
        flag_s.push_str(" lossless");
    }
    if h.flags & container::FLAG_TEMPORAL != 0 {
        flag_s.push_str(" temporal");
    }
    if h.flags & container::FLAG_MODULAR != 0 {
        flag_s.push_str(" modular");
    } else {
        flag_s.push_str(" vardct");
    }
    if h.flags & FLAG_PROGRESSIVE != 0 {
        flag_s.push_str(" progressive");
    }
    let groups = container::display_frames(&h);
    let shown = groups.len();
    if json {
        let mut s = format!(
            "{{\"version\":{},\"width\":{},\"height\":{},\"frames\":{},\"samples\":{},\"fps\":{:.6},\"flags\":\"0x{:x}\",\"payload_bytes\":{},\"displayed\":[",
            h.version,
            h.width,
            h.height,
            shown,
            h.nframes(),
            h.fps(),
            h.flags,
            payload
        );
        for (fi, group) in groups.iter().enumerate() {
            if fi > 0 {
                s.push(',');
            }
            let dur: u32 = group.iter().map(|x| x.duration_ticks).sum();
            s.push_str(&format!(
                "{{\"index\":{},\"duration_ticks\":{},\"samples\":[",
                fi, dur
            ));
            for (si, sample) in group.iter().enumerate() {
                if si > 0 {
                    s.push(',');
                }
                s.push_str(&format!(
                    "{{\"offset\":{},\"size\":{},\"duration_ticks\":{},\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"skip\":{}}}",
                    sample.offset, sample.size, sample.duration_ticks, sample.x, sample.y,
                    sample.w, sample.h, sample.size == 0
                ));
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        println!("{s}");
        return Ok(());
    }
    let human = if payload >= 1 << 20 {
        format!("{:.1} MiB", payload as f64 / (1 << 20) as f64)
    } else if payload >= 1 << 10 {
        format!("{:.1} KiB", payload as f64 / (1 << 10) as f64)
    } else {
        format!("{payload} B")
    };
    println!(
        "v{} {}×{}  {} displayed / {} samples  {:.3} fps  flags=0x{:x} ({})  payload {} bytes ({human})",
        h.version,
        h.width,
        h.height,
        shown,
        h.nframes(),
        h.fps(),
        h.flags,
        flag_s.trim(),
        payload
    );
    for (fi, group) in groups.iter().enumerate() {
        let dur: u32 = group.iter().map(|s| s.duration_ticks).sum();
        println!("  frame {fi} dur={dur} rects={}", group.len());
        for s in group {
            let kind = if s.size == 0 { "skip" } else { "jxl" };
            println!(
                "    {kind} off={} size={} dur={} @{},{} {}×{}",
                s.offset, s.size, s.duration_ticks, s.x, s.y, s.w, s.h
            );
        }
    }
    Ok(())
}

fn cmd_bench(
    input: PathBuf,
    distance: f32,
    effort: u8,
    coding: StillCoding,
    repeats: u32,
    json: bool,
) -> mjxl::Result<()> {
    let mut reader = y4m::open_path(&input)?;
    let info = reader.info.clone();
    let mut frames = Vec::new();
    while let Some(rgb) = reader.next_rgb()? {
        frames.push(rgb);
    }
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let pixels = frames.len() as u64 * info.width as u64 * info.height as u64;
    let modular = matches!(coding, StillCoding::Modular) || distance <= 0.0;
    let enc = Encoder::new()
        .distance(distance)
        .effort(effort)
        .modular(Some(modular));
    let opts = SequenceOpts {
        intra: true,
        closed_loop: false,
        backend: Backend::Cpu,
    };
    let mut flags = if distance <= 0.0 { FLAG_LOSSLESS } else { 0 };
    if modular {
        flags |= container::FLAG_MODULAR;
    }
    let to_blob = |coded: &[temporal::Coded]| -> mjxl::Result<Vec<u8>> {
        let mut blob = Vec::new();
        container::write_coded(
            &mut blob,
            flags,
            info.width,
            info.height,
            info.fps_num,
            info.fps_den,
            coded,
        )?;
        Ok(blob)
    };
    let dopts = DecodeOpts {
        prefix: None,
        backend: Backend::Cpu,
    };

    // Warm up both Rayon and the decoder before timing.
    let warm = to_blob(&temporal::encode_sequence(
        &enc,
        info.width,
        info.height,
        &frames,
        &opts,
    )?)?;
    let _ = decode::decode_rgb(&warm, &dopts)?;

    let mut enc_ms = Vec::new();
    let mut dec_ms = Vec::new();
    let mut last_blob = warm;
    for _ in 0..repeats.max(1) {
        let t = Instant::now();
        let coded = temporal::encode_sequence(&enc, info.width, info.height, &frames, &opts)?;
        enc_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        last_blob = to_blob(&coded)?;
        let t = Instant::now();
        let _ = decode::decode_rgb(&last_blob, &dopts)?;
        dec_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let em = median(&mut enc_ms);
    let dm = median(&mut dec_ms);
    let mp = |ms: f64| (pixels as f64 / 1e6) / (ms / 1000.0);
    if json {
        println!(
            "{{\"coder\":\"{coding:?}\",\"distance\":{distance},\"effort\":{effort},\"pixels\":{pixels},\"repeats\":{repeats},\"encode_ms\":{em:.3},\"decode_ms\":{dm:.3},\"enc_mps\":{:.3},\"dec_mps\":{:.3},\"bytes\":{}}}",
            mp(em),
            mp(dm),
            last_blob.len()
        );
        return Ok(());
    }
    println!("{coding:?} d={distance} e={effort} {pixels} pixels {repeats} reps");
    println!(
        "  encode {em:8.1} ms {:7.1} MP/s   decode {dm:8.1} ms {:7.1} MP/s   bytes {}",
        mp(em),
        mp(dm),
        last_blob.len()
    );
    Ok(())
}

fn cmd_anim(
    input: PathBuf,
    output: Option<PathBuf>,
    distance: f32,
    effort: u8,
) -> mjxl::Result<()> {
    let mut reader = y4m::open_path(&input)?;
    let info = reader.info.clone();
    let mut frames = Vec::new();
    while let Some(rgb) = reader.next_rgb()? {
        frames.push(rgb);
    }
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let enc = Encoder::new()
        .distance(distance)
        .effort(effort)
        .modular(Some(true));
    let refs: Vec<(u32, &[u8])> = frames.iter().map(|f| (1u32, f.as_slice())).collect();
    let t = Instant::now();
    let blob = enc.encode_animation(
        info.width,
        info.height,
        PixelFormat::Rgb,
        info.fps_num,
        info.fps_den,
        0,
        &refs,
    )?;
    let enc_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let decoded = o3_jxl::Decoder::new()
        .request_srgb(true)
        .keep_linear(false)
        .decode_bytes(&blob)?;
    let dec_ms = t.elapsed().as_secs_f64() * 1000.0;
    let ndec = decoded.len();
    let psnrs: Vec<f64> = decoded
        .iter()
        .enumerate()
        .take(frames.len())
        .filter_map(|(i, f)| compare::psnr_rgb8(&frames[i], &still::rgba_to_rgb(&f.rgba)))
        .collect();
    let psnr = if psnrs.is_empty() {
        f64::NAN
    } else {
        psnrs.iter().sum::<f64>() / psnrs.len() as f64
    };
    let out = output.unwrap_or_else(|| input.with_extension("jxl"));
    std::fs::write(&out, &blob)?;
    let pixels = frames.len() as f64 * info.width as f64 * info.height as f64;
    println!(
        "Native JXL animation: {} frames, {} bytes, {:.3} bpp, encode {:.1} ms ({:.1} MP/s), decode {:.1} ms ({:.1} MP/s), decoded {} frames → {}",
        frames.len(),
        blob.len(),
        blob.len() as f64 * 8.0 / pixels,
        enc_ms,
        (pixels / 1e6) / (enc_ms / 1000.0).max(1e-9),
        dec_ms,
        (pixels / 1e6) / (dec_ms / 1000.0).max(1e-9),
        ndec,
        out.display()
    );
    println!("  RGB PSNR {psnr:.2} dB");
    Ok(())
}

fn cmd_verify(
    input: PathBuf,
    distance: f32,
    effort: u8,
    coding: StillCoding,
    json: bool,
) -> mjxl::Result<()> {
    let mut reader = y4m::open_path(&input)?;
    let info = reader.info.clone();
    let mut frames = Vec::new();
    while let Some(rgb) = reader.next_rgb()? {
        frames.push(rgb);
    }
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let modular = matches!(coding, StillCoding::Modular) || distance <= 0.0;
    let enc = Encoder::new()
        .distance(distance)
        .effort(effort)
        .modular(Some(modular));
    let coded = temporal::encode_sequence(
        &enc,
        info.width,
        info.height,
        &frames,
        &SequenceOpts {
            intra: true,
            closed_loop: false,
            backend: Backend::Cpu,
        },
    )?;
    let mut flags = if distance <= 0.0 { FLAG_LOSSLESS } else { 0 };
    if modular {
        flags |= container::FLAG_MODULAR;
    }
    let mut blob = Vec::new();
    container::write_coded(
        &mut blob,
        flags,
        info.width,
        info.height,
        info.fps_num,
        info.fps_den,
        &coded,
    )?;
    let decoded = decode::decode_rgb(
        &blob,
        &DecodeOpts {
            prefix: None,
            backend: Backend::Cpu,
        },
    )?
    .frames;
    let n = frames.len().min(decoded.len());
    let psnrs: Vec<f64> = (0..n)
        .filter_map(|i| compare::psnr_rgb8(&frames[i], &decoded[i]))
        .collect();
    let mean = if psnrs.is_empty() {
        f64::NAN
    } else {
        psnrs.iter().sum::<f64>() / psnrs.len() as f64
    };
    let min = psnrs.iter().copied().fold(f64::INFINITY, f64::min);
    if json {
        println!(
            "{{\"coder\":\"{coding:?}\",\"distance\":{distance},\"effort\":{effort},\"frames\":{n},\"bytes\":{},\"bpp\":{:.6},\"psnr_mean\":{},\"psnr_min\":{}}}",
            blob.len(),
            blob.len() as f64 * 8.0
                / (frames.len() as f64 * info.width as f64 * info.height as f64),
            mean,
            min,
        );
    } else {
        println!(
            "Verified {n} frames {coding:?} d={distance} e={effort}: {} bytes, {:.3} bpp, PSNR mean {:.2} dB (min {:.2} dB)",
            blob.len(),
            blob.len() as f64 * 8.0
                / (frames.len() as f64 * info.width as f64 * info.height as f64),
            mean,
            min
        );
    }
    if n != frames.len() || n != decoded.len() {
        return Err(format!(
            "frame count mismatch: {} in, {} out",
            frames.len(),
            decoded.len()
        )
        .into());
    }
    Ok(())
}

fn cmd_compare(
    input: PathBuf,
    distance: f32,
    effort: u8,
    mjpeg_q: i32,
    j2k_q: i32,
    work_dir: Option<PathBuf>,
    keep_work_dir: bool,
    coding: CodingCli,
    json: bool,
) -> mjxl::Result<()> {
    let owned_dir = work_dir.is_none();
    let work_dir = work_dir
        .unwrap_or_else(|| std::env::temp_dir().join(format!("mjxl-{}", std::process::id())));
    let opts = CompareOpts {
        distance,
        effort,
        mjpeg_q,
        j2k_q,
        work_dir: work_dir.clone(),
        coding: match coding {
            CodingCli::Vardct => Coding::Vardct,
            CodingCli::Modular => Coding::Modular,
            CodingCli::Both => Coding::Both,
        },
    };
    if !json {
        eprintln!("work dir {}", work_dir.display());
    }
    let result = compare::run(&input, &opts);
    if owned_dir && !keep_work_dir {
        let _ = std::fs::remove_dir_all(&work_dir);
    }
    let (rows, pixels) = result?;
    if json {
        print_json_rows(&rows, pixels);
    } else {
        println!(
            "{:.3} MP  mjxl -d {distance} -e {effort}  mjpeg -q:v {mjpeg_q}  j2k -q:v {j2k_q}  coding={coding:?}",
            pixels as f64 / 1e6
        );
        compare::print_table(&rows, pixels);
    }
    Ok(())
}

fn cmd_stills(
    input: PathBuf,
    distance: f32,
    effort: u8,
    repeats: u32,
    work_dir: Option<PathBuf>,
    json: bool,
) -> mjxl::Result<()> {
    let owned_dir = work_dir.is_none();
    let work_dir = work_dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("mjxl-stills-{}", std::process::id()))
    });
    let opts = compare::StillOpts {
        distance,
        effort,
        repeats,
        work_dir: work_dir.clone(),
    };
    if !json {
        eprintln!("work dir {}", work_dir.display());
    }
    let result = compare::run_still(&input, &opts);
    if owned_dir {
        let _ = std::fs::remove_dir_all(&work_dir);
    }
    let (rows, pixels) = result?;
    if json {
        print_json_rows(&rows, pixels);
    } else {
        println!(
            "{:.3} MP  still d={distance} effort={effort} repeats={repeats}",
            pixels as f64 / 1e6
        );
        compare::print_table(&rows, pixels);
    }
    Ok(())
}
