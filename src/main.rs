use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use mjxl::compare::{self, Coding, CompareOpts};
use mjxl::container::{self, FLAG_LOSSLESS, FLAG_PROGRESSIVE};
use mjxl::decode::{self, DecodeOpts};
use mjxl::gpu::{self, Backend};
use mjxl::progressive;
use mjxl::temporal::{self, SequenceOpts};
use mjxl::y4m;
use o3_jxl::Encoder;

#[derive(Parser)]
#[command(
    name = "mjxl",
    version,
    about = "Motion JPEG XL: independent JXL frames, dirty rectangles, modular / VarDCT / progressive."
)]
struct Cli {
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
        #[arg(short = 'e', long, default_value_t = 7)]
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
        /// FFmpeg MJPEG `-q:v` (2=finest). 3 ≈ JPEG quality 90 on this encoder.
        #[arg(long, default_value_t = 3)]
        mjpeg_q: i32,
        #[arg(long, default_value_t = 5)]
        j2k_q: i32,
        #[arg(long)]
        work_dir: Option<PathBuf>,
        /// Which still coder(s) to measure. Default both.
        #[arg(short = 'm', long, value_enum, default_value_t = CodingCli::Both)]
        coding: CodingCli,
    },
    /// Still-image speed, bpp, and PSNR versus libjxl (`cjxl` / `djxl`).
    Stills {
        input: PathBuf,
        /// Butteraugli distance. 0 encodes lossless modular only.
        #[arg(short = 'd', long, default_value_t = 1.0)]
        distance: f32,
        /// Encoder effort, same scale as `cjxl -e`. Default 7.
        #[arg(short = 'e', long, default_value_t = 7)]
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
    match Cli::parse().command {
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
        Commands::Info { input } => cmd_info(input),
        Commands::Compare {
            input,
            distance,
            mjpeg_q,
            j2k_q,
            work_dir,
            coding,
        } => cmd_compare(input, distance, mjpeg_q, j2k_q, work_dir, coding),
        Commands::Stills {
            input,
            distance,
            effort,
            repeats,
            work_dir,
        } => cmd_stills(input, distance, effort, repeats, work_dir),
    }
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

fn cmd_info(input: PathBuf) -> mjxl::Result<()> {
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
    let shown = container::display_frames(&h).len();
    println!(
        "v{} {}×{}  {} displayed / {} samples  {:.3} fps  flags=0x{:x} ({})  payload {} bytes",
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
    for (fi, group) in container::display_frames(&h).iter().enumerate() {
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

fn cmd_compare(
    input: PathBuf,
    distance: f32,
    mjpeg_q: i32,
    j2k_q: i32,
    work_dir: Option<PathBuf>,
    coding: CodingCli,
) -> mjxl::Result<()> {
    let work_dir = work_dir
        .unwrap_or_else(|| std::env::temp_dir().join(format!("mjxl-{}", std::process::id())));
    let opts = CompareOpts {
        distance,
        mjpeg_q,
        j2k_q,
        work_dir: work_dir.clone(),
        coding: match coding {
            CodingCli::Vardct => Coding::Vardct,
            CodingCli::Modular => Coding::Modular,
            CodingCli::Both => Coding::Both,
        },
    };
    eprintln!("work dir {}", work_dir.display());
    let (rows, pixels) = compare::run(&input, &opts)?;
    println!(
        "{:.3} MP  mjxl -d {distance}  mjpeg -q:v {mjpeg_q}  j2k -q:v {j2k_q}  coding={coding:?}",
        pixels as f64 / 1e6
    );
    compare::print_table(&rows, pixels);
    Ok(())
}

fn cmd_stills(
    input: PathBuf,
    distance: f32,
    effort: u8,
    repeats: u32,
    work_dir: Option<PathBuf>,
) -> mjxl::Result<()> {
    let work_dir = work_dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("mjxl-stills-{}", std::process::id()))
    });
    let opts = compare::StillOpts {
        distance,
        effort,
        repeats,
        work_dir: work_dir.clone(),
    };
    eprintln!("work dir {}", work_dir.display());
    let (rows, pixels) = compare::run_still(&input, &opts)?;
    println!(
        "{:.3} MP  still d={distance} effort={effort} repeats={repeats}",
        pixels as f64 / 1e6
    );
    compare::print_table(&rows, pixels);
    Ok(())
}
