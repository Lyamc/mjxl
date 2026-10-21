//! Run MJPEG, JPEG 2000, and mjxl on the same RGB frames, and stills against libjxl.

use crate::container::{self, FLAG_LOSSLESS};
use crate::decode::{self, DecodeOpts};
use crate::error::{Error, Result};
use crate::gpu::Backend;
use crate::temporal::{self, SequenceOpts};
use crate::y4m::{self, Info};
use o3_jxl::Encoder;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Coding {
    Vardct,
    Modular,
    Both,
}

pub struct CompareOpts {
    pub distance: f32,
    pub mjpeg_q: i32,
    pub j2k_q: i32,
    pub work_dir: PathBuf,
    pub coding: Coding,
}

pub struct StillOpts {
    pub distance: f32,
    pub effort: u8,
    pub repeats: u32,
    pub work_dir: PathBuf,
}

#[derive(Debug)]
pub struct Row {
    pub name: String,
    pub bytes: u64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub psnr: Option<f64>,
    pub note: String,
}

/// Returns (rows, total_pixels).
pub fn run(input: &Path, opts: &CompareOpts) -> Result<(Vec<Row>, u64)> {
    std::fs::create_dir_all(&opts.work_dir)?;
    let mut reader = y4m::open_path(input)?;
    let info = reader.info.clone();
    let mut frames = Vec::new();
    while let Some(rgb) = reader.next_rgb()? {
        frames.push(rgb);
    }
    if frames.is_empty() {
        return Err(Error::from("no frames in y4m"));
    }
    let n = frames.len() as u64;
    let pixels = n * info.width as u64 * info.height as u64;

    let ref_y4m = opts.work_dir.join("ref.y4m");
    write_rgb_sequence(&ref_y4m, &info, &frames)?;
    if let Ok(Some(gpu)) = crate::gpu::select(Backend::Auto) {
        eprintln!("dirty search: {}", gpu.info);
    } else {
        eprintln!("dirty search: cpu");
    }

    let mut rows = Vec::new();
    let run_vd = matches!(opts.coding, Coding::Vardct | Coding::Both);
    let run_mod = matches!(opts.coding, Coding::Modular | Coding::Both);
    if run_vd {
        rows.push(encode_mjxl(
            &info, &frames, opts, &ref_y4m, pixels, true, false,
        )?);
        rows.push(encode_mjxl(
            &info, &frames, opts, &ref_y4m, pixels, false, false,
        )?);
    }
    if run_mod {
        rows.push(encode_mjxl(
            &info, &frames, opts, &ref_y4m, pixels, true, true,
        )?);
        rows.push(encode_mjxl(
            &info, &frames, opts, &ref_y4m, pixels, false, true,
        )?);
    }

    rows.push(ffmpeg_codec(
        "mjpeg-444",
        &ref_y4m,
        &opts.work_dir.join("mjpeg444.avi"),
        &opts.work_dir.join("mjpeg444.y4m"),
        &[
            "-c:v",
            "mjpeg",
            "-pix_fmt",
            "yuvj444p",
            "-q:v",
            &opts.mjpeg_q.to_string(),
        ],
        pixels,
    )?);

    rows.push(ffmpeg_codec(
        "mjpeg-420",
        &ref_y4m,
        &opts.work_dir.join("mjpeg420.avi"),
        &opts.work_dir.join("mjpeg420.y4m"),
        &[
            "-c:v",
            "mjpeg",
            "-pix_fmt",
            "yuvj420p",
            "-q:v",
            &opts.mjpeg_q.to_string(),
        ],
        pixels,
    )?);

    if which("cjxl") && which("djxl") {
        match encode_cjxl(&info, opts, &ref_y4m, pixels) {
            Ok(row) => rows.push(row),
            Err(e) => eprintln!("skip libjxl: {e}"),
        }
    }

    rows.push(ffmpeg_codec(
        "jpeg2000",
        &ref_y4m,
        &opts.work_dir.join("j2k.mov"),
        &opts.work_dir.join("j2k.y4m"),
        &[
            "-c:v",
            "jpeg2000",
            "-pix_fmt",
            "rgb24",
            "-q:v",
            &opts.j2k_q.to_string(),
        ],
        pixels,
    )?);

    Ok((rows, pixels))
}

fn write_rgb_sequence(path: &Path, info: &Info, frames: &[Vec<u8>]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    y4m::write_sequence(&mut f, info, frames)?;
    Ok(())
}

fn encode_mjxl(
    info: &Info,
    frames: &[Vec<u8>],
    opts: &CompareOpts,
    ref_y4m: &Path,
    _pixels: u64,
    intra: bool,
    modular: bool,
) -> Result<Row> {
    let enc = Encoder::new()
        .distance(opts.distance)
        .modular(Some(modular));
    let t = Instant::now();
    let coded = temporal::encode_sequence(
        &enc,
        info.width,
        info.height,
        frames,
        &SequenceOpts {
            intra,
            closed_loop: false,
            backend: Backend::Auto,
        },
    )?;
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut flags = if opts.distance <= 0.0 {
        FLAG_LOSSLESS
    } else {
        0
    };
    if !intra {
        flags |= crate::container::FLAG_TEMPORAL;
    }
    if modular {
        flags |= crate::container::FLAG_MODULAR;
    }
    let mode = if modular { "mod" } else { "vdct" };
    let name = if intra {
        format!("mjxl-{mode}-intra")
    } else {
        format!("mjxl-{mode}-delta")
    };
    let out_path = opts.work_dir.join(format!("{name}.mjxl"));
    let mut file = std::fs::File::create(&out_path)?;
    container::write_coded(
        &mut file,
        flags,
        info.width,
        info.height,
        info.fps_num,
        info.fps_den,
        &coded,
    )?;
    drop(file);
    let bytes = std::fs::metadata(&out_path)?.len();

    let blob = std::fs::read(&out_path)?;
    let t = Instant::now();
    let decoded = decode::decode_rgb(
        &blob,
        &DecodeOpts {
            prefix: None,
            backend: Backend::Cpu,
        },
    )?
    .frames;
    let decode_ms = t.elapsed().as_secs_f64() * 1000.0;

    let recon = opts.work_dir.join(format!("{name}.y4m"));
    write_rgb_sequence(&recon, info, &decoded)?;
    let psnr = ffmpeg_psnr(ref_y4m, &recon)?;
    let nskip = coded.iter().filter(|c| c.is_skip()).count();
    let nrect = coded
        .iter()
        .filter(|c| !c.is_skip() && !c.is_full(info.width, info.height))
        .count();
    Ok(Row {
        name: format!("{name} d={}", opts.distance),
        bytes,
        encode_ms,
        decode_ms,
        psnr,
        note: if intra {
            if modular {
                "modular all-intra".into()
            } else {
                "VarDCT all-intra".into()
            }
        } else {
            let npatch: usize = coded.iter().map(|c| c.patches.len()).sum();
            format!("skip={nskip} rect={nrect} patches={npatch}")
        },
    })
}

fn which(name: &str) -> bool {
    let finder = if cfg!(windows) { "where" } else { "which" };
    Command::new(finder)
        .arg(name)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn encode_cjxl(info: &Info, opts: &CompareOpts, ref_y4m: &Path, _pixels: u64) -> Result<Row> {
    let png_dir = opts.work_dir.join("cjxl-png");
    std::fs::create_dir_all(&png_dir)?;
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(ref_y4m)
        .arg(png_dir.join("%04d.png"))
        .stdin(Stdio::null())
        .status()?;
    if !st.success() {
        return Err(Error::from("ffmpeg png dump for cjxl failed"));
    }
    let mut pngs: Vec<PathBuf> = std::fs::read_dir(&png_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    pngs.sort();
    let jxl_dir = opts.work_dir.join("cjxl-jxl");
    std::fs::create_dir_all(&jxl_dir)?;
    let t = Instant::now();
    let mut stills = Vec::new();
    for (i, png) in pngs.iter().enumerate() {
        let out = jxl_dir.join(format!("{i:04}.jxl"));
        let st = Command::new("cjxl")
            .args(["-d", &opts.distance.to_string(), "-e", "7", "--quiet"])
            .arg(png)
            .arg(&out)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !st.success() {
            return Err(Error::from("cjxl failed"));
        }
        stills.push(std::fs::read(&out)?);
    }
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let pairs: Vec<(u32, &[u8])> = stills.iter().map(|s| (1u32, s.as_slice())).collect();
    let out_path = opts.work_dir.join("cjxl.mjxl");
    let mut file = std::fs::File::create(&out_path)?;
    container::write_file(
        &mut file,
        0,
        info.width,
        info.height,
        info.fps_num,
        info.fps_den,
        &pairs,
    )?;
    drop(file);
    let bytes = std::fs::metadata(&out_path)?.len();

    // Score libjxl with djxl. mjxl's own decoder does not match djxl on these
    // VarDCT streams, so a PSNR from `decode_rgb` would not be libjxl's quality.
    let ppm_dir = opts.work_dir.join("cjxl-ppm");
    std::fs::create_dir_all(&ppm_dir)?;
    let mut frames = Vec::with_capacity(stills.len());
    let t = Instant::now();
    for i in 0..stills.len() {
        let src = jxl_dir.join(format!("{i:04}.jxl"));
        let dst = ppm_dir.join(format!("{i:04}.ppm"));
        djxl_decode(&src, &dst)?;
        let (_, _, rgb) = read_ppm(&dst)?;
        frames.push(rgb);
    }
    let decode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let recon = opts.work_dir.join("cjxl.y4m");
    write_rgb_sequence(&recon, info, &frames)?;
    let psnr = ffmpeg_psnr(ref_y4m, &recon)?;
    Ok(Row {
        name: format!("libjxl d={}", opts.distance),
        bytes,
        encode_ms,
        decode_ms,
        psnr,
        note: "cjxl/djxl per frame; PSNR is djxl; not a speed score".into(),
    })
}

fn ffmpeg_codec(
    name: &str,
    src: &Path,
    encoded: &Path,
    decoded: &Path,
    extra: &[&str],
    _pixels: u64,
) -> Result<Row> {
    let t = Instant::now();
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(src)
        .args(extra)
        .arg(encoded)
        .stdin(Stdio::null())
        .status()?;
    if !st.success() {
        return Err(Error::from(format!("ffmpeg encode {name} failed")));
    }
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let bytes = std::fs::metadata(encoded)?.len();

    let t = Instant::now();
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(encoded)
        .args(["-pix_fmt", "yuvj444p"])
        .arg(decoded)
        .stdin(Stdio::null())
        .status()?;
    if !st.success() {
        return Err(Error::from(format!("ffmpeg decode {name} failed")));
    }
    let decode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let psnr = ffmpeg_psnr(src, decoded)?;
    Ok(Row {
        name: name.into(),
        bytes,
        encode_ms,
        decode_ms,
        psnr,
        note: extra.join(" "),
    })
}

fn ffmpeg_psnr(ref_y4m: &Path, dist: &Path) -> Result<Option<f64>> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(dist)
        .arg("-i")
        .arg(ref_y4m)
        .args(["-lavfi", "psnr", "-f", "null", "-"])
        .output()?;
    let err = String::from_utf8_lossy(&out.stderr);
    for line in err.lines() {
        if let Some(i) = line.find("average:") {
            let rest = &line[i + "average:".len()..];
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                return Ok(Some(v));
            }
        }
    }
    Ok(None)
}

pub fn print_table(rows: &[Row], pixels: u64) {
    println!(
        "{:<26} {:>10} {:>8} {:>9} {:>9} {:>8} {:>8} {:>7}  note",
        "codec", "bytes", "bpp", "enc_ms", "dec_ms", "encMP/s", "decMP/s", "PSNR"
    );
    for r in rows {
        let bpp = (r.bytes as f64) * 8.0 / pixels.max(1) as f64;
        let psnr = r
            .psnr
            .map(|v| format!("{v:5.2}"))
            .unwrap_or_else(|| "  n/a".into());
        println!(
            "{:<26} {:>10} {:>8.3} {:>9.1} {:>9.1} {:>8.1} {:>8.1} {:>7}  {}",
            r.name,
            r.bytes,
            bpp,
            r.encode_ms,
            r.decode_ms,
            megapixels_per_sec(pixels, r.encode_ms),
            megapixels_per_sec(pixels, r.decode_ms),
            psnr,
            r.note
        );
    }
}

fn megapixels_per_sec(pixels: u64, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    (pixels as f64 / 1e6) / (ms / 1000.0)
}

/// In-process mjxl stills versus one-shot `cjxl` / `djxl` on the same RGB.
///
/// Times are the median of `opts.repeats` after one warmup. libjxl times
/// include process startup and file IO. mjxl times are the codestream only.
pub fn run_still(input: &Path, opts: &StillOpts) -> Result<(Vec<Row>, u64)> {
    std::fs::create_dir_all(&opts.work_dir)?;
    let (w, h, rgb) = load_rgb_via_ffmpeg(input, &opts.work_dir)?;
    let pixels = w as u64 * h as u64;
    let repeats = opts.repeats.max(1);
    let mut rows = Vec::new();
    if opts.distance > 0.0 {
        rows.push(bench_mjxl_still(&rgb, w, h, opts, repeats, false)?);
    }
    rows.push(bench_mjxl_still(&rgb, w, h, opts, repeats, true)?);
    if which("cjxl") && which("djxl") {
        match bench_libjxl_still(&rgb, w, h, opts, repeats) {
            Ok(row) => rows.push(row),
            Err(e) => eprintln!("skip libjxl: {e}"),
        }
    } else {
        eprintln!("skip libjxl: cjxl and djxl not both on PATH");
    }
    Ok((rows, pixels))
}

fn bench_mjxl_still(
    rgb: &[u8],
    w: u32,
    h: u32,
    opts: &StillOpts,
    repeats: u32,
    modular: bool,
) -> Result<Row> {
    let enc = Encoder::new()
        .distance(opts.distance)
        .effort(opts.effort)
        .modular(Some(modular));
    let warmup = enc.encode_rgb8(w, h, rgb)?;
    let _ = crate::still::decode_jxl_rgb(&warmup)?;
    let mut enc_ms = Vec::with_capacity(repeats as usize);
    let mut last = warmup;
    for _ in 0..repeats {
        let t = Instant::now();
        last = enc.encode_rgb8(w, h, rgb)?;
        enc_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let _ = crate::still::decode_jxl_rgb(&last)?;
    let mut dec_ms = Vec::with_capacity(repeats as usize);
    let mut decoded = Vec::new();
    for _ in 0..repeats {
        let t = Instant::now();
        let (_, _, frame) = crate::still::decode_jxl_rgb(&last)?;
        dec_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        decoded = frame;
    }
    let mode = if modular || opts.distance <= 0.0 {
        "modular"
    } else {
        "vardct"
    };
    Ok(Row {
        name: format!("mjxl-{mode} d={}", opts.distance),
        bytes: last.len() as u64,
        encode_ms: median_ms(&enc_ms),
        decode_ms: median_ms(&dec_ms),
        psnr: psnr_rgb8(rgb, &decoded),
        note: format!(
            "in-process codestream, median of {repeats}, e={}",
            opts.effort
        ),
    })
}

fn bench_libjxl_still(rgb: &[u8], w: u32, h: u32, opts: &StillOpts, repeats: u32) -> Result<Row> {
    let src = opts.work_dir.join("still-src.ppm");
    let jxl = opts.work_dir.join("libjxl.jxl");
    let dec = opts.work_dir.join("libjxl-dec.ppm");
    write_ppm(&src, w, h, rgb)?;
    let d = opts.distance.to_string();
    let e = opts.effort.to_string();
    cjxl_encode(&src, &jxl, &d, &e)?;
    djxl_decode(&jxl, &dec)?;
    let mut enc_ms = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let _ = std::fs::remove_file(&jxl);
        let t = Instant::now();
        cjxl_encode(&src, &jxl, &d, &e)?;
        enc_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let bytes = std::fs::metadata(&jxl)?.len();
    let mut dec_ms = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let _ = std::fs::remove_file(&dec);
        let t = Instant::now();
        djxl_decode(&jxl, &dec)?;
        dec_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let (_, _, decoded) = read_ppm(&dec)?;
    Ok(Row {
        name: format!("libjxl d={}", opts.distance),
        bytes,
        encode_ms: median_ms(&enc_ms),
        decode_ms: median_ms(&dec_ms),
        psnr: psnr_rgb8(rgb, &decoded),
        note: format!(
            "cjxl/djxl process + file IO, median of {repeats}, e={}",
            opts.effort
        ),
    })
}

fn cjxl_encode(src: &Path, dst: &Path, distance: &str, effort: &str) -> Result<()> {
    let out = Command::new("cjxl")
        .args(["-d", distance, "-e", effort, "--quiet"])
        .arg(src)
        .arg(dst)
        .stdin(Stdio::null())
        .output()?;
    tool_status("cjxl", &out)
}

fn djxl_decode(src: &Path, dst: &Path) -> Result<()> {
    let out = Command::new("djxl")
        .arg("--quiet")
        .arg(src)
        .arg(dst)
        .stdin(Stdio::null())
        .output()?;
    tool_status("djxl", &out)
}

fn tool_status(cmd: &str, out: &std::process::Output) -> Result<()> {
    if out.status.success() {
        return Ok(());
    }
    let err = &out.stderr;
    let start = err.len().saturating_sub(300);
    let tail = String::from_utf8_lossy(&err[start..]);
    Err(Error::from(format!("{cmd} failed: {tail}")))
}

fn load_rgb_via_ffmpeg(input: &Path, work: &Path) -> Result<(u32, u32, Vec<u8>)> {
    let ppm = work.join("input.ppm");
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-frames:v", "1", "-pix_fmt", "rgb24"])
        .arg(&ppm)
        .stdin(Stdio::null())
        .status()?;
    if !st.success() {
        return Err(Error::from("ffmpeg rgb ppm dump failed"));
    }
    read_ppm(&ppm)
}

fn write_ppm(path: &Path, w: u32, h: u32, rgb: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    Ok(())
}

fn read_ppm(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    let data = std::fs::read(path)?;
    let mut i = 0usize;
    let magic = ppm_token(&data, &mut i)?;
    if magic != "P6" {
        return Err(Error::from(format!("expected P6 ppm, got {magic}")));
    }
    let w: u32 = ppm_token(&data, &mut i)?
        .parse()
        .map_err(|_| Error::from("bad ppm width"))?;
    let h: u32 = ppm_token(&data, &mut i)?
        .parse()
        .map_err(|_| Error::from("bad ppm height"))?;
    let maxval: u32 = ppm_token(&data, &mut i)?
        .parse()
        .map_err(|_| Error::from("bad ppm maxval"))?;
    if maxval != 255 {
        return Err(Error::from(format!("ppm maxval {maxval}, want 255")));
    }
    if i < data.len() && data[i].is_ascii_whitespace() {
        i += 1;
    }
    let need = w as usize * h as usize * 3;
    if data.len() < i + need {
        return Err(Error::from("truncated ppm"));
    }
    Ok((w, h, data[i..i + need].to_vec()))
}

fn ppm_token(data: &[u8], i: &mut usize) -> Result<String> {
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
    let start = *i;
    while *i < data.len() && !data[*i].is_ascii_whitespace() {
        *i += 1;
    }
    if start == *i {
        return Err(Error::from("truncated ppm header"));
    }
    Ok(String::from_utf8_lossy(&data[start..*i]).into_owned())
}

fn median_ms(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn psnr_rgb8(a: &[u8], b: &[u8]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut sse = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = f64::from(*x) - f64::from(*y);
        sse += d * d;
    }
    if sse < 1e-9 {
        return Some(99.0);
    }
    Some(10.0 * (255.0 * 255.0 / (sse / a.len() as f64)).log10())
}

#[cfg(test)]
mod ppm_tests {
    use super::*;

    #[test]
    fn ppm_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mjxl-ppm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.ppm");
        let rgb = vec![0u8, 1, 2, 255, 128, 64];
        write_ppm(&path, 2, 1, &rgb).unwrap();
        let (w, h, out) = read_ppm(&path).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, rgb);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
