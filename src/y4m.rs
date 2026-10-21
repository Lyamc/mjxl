//! Minimal Y4M reader (4:2:0 / 4:4:4 → RGB8) and 4:4:4 writer.

use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Chroma {
    C420,
    C444,
}

#[derive(Clone, Debug)]
pub struct Info {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub chroma: Chroma,
    /// `true` if luma is studio range (16–235). MPEG-2 chroma tags.
    pub limited: bool,
}

pub struct Reader<R: BufRead> {
    inner: R,
    pub info: Info,
}

impl<R: BufRead> Reader<R> {
    pub fn open(mut inner: R) -> Result<Self> {
        let mut header = String::new();
        inner.read_line(&mut header)?;
        if !header.starts_with("YUV4MPEG2") {
            return Err(Error::from("not a Y4M file"));
        }
        let mut width = 0u32;
        let mut height = 0u32;
        let mut fps_num = 24u32;
        let mut fps_den = 1u32;
        let mut chroma = Chroma::C420;
        let mut limited = true;
        let mut range_full = None;
        for tok in header.split_whitespace().skip(1) {
            match tok.as_bytes() {
                [b'W', rest @ ..] => width = parse_u32(rest)?,
                [b'H', rest @ ..] => height = parse_u32(rest)?,
                [b'F', rest @ ..] => {
                    let s = std::str::from_utf8(rest).unwrap_or("");
                    let mut it = s.split(':');
                    fps_num = it.next().unwrap_or("24").parse().unwrap_or(24);
                    fps_den = it.next().unwrap_or("1").parse().unwrap_or(1).max(1);
                }
                t if t.starts_with(b"C420jpeg") => {
                    chroma = Chroma::C420;
                    limited = false;
                }
                t if t.starts_with(b"C420") => {
                    chroma = Chroma::C420;
                    limited = true;
                }
                t if t.starts_with(b"C444") => {
                    chroma = Chroma::C444;
                    // Untagged C444 is full range (what this crate writes, and
                    // FFmpeg's yuvj444p before the XCOLORRANGE token is seen).
                    limited = t.windows(4).any(|w| w == b"mpeg");
                }
                t if t.starts_with(b"XCOLORRANGE=") => {
                    range_full = Some(t.ends_with(b"FULL"));
                }
                _ => {}
            }
        }
        if let Some(full) = range_full {
            limited = !full;
        }
        if width == 0 || height == 0 {
            return Err(Error::from("Y4M missing W/H"));
        }
        Ok(Self {
            inner,
            info: Info {
                width,
                height,
                fps_num,
                fps_den: fps_den.max(1),
                chroma,
                limited,
            },
        })
    }

    /// Next frame as packed RGB8. `None` at EOF.
    pub fn next_rgb(&mut self) -> Result<Option<Vec<u8>>> {
        let mut tag = String::new();
        let n = self.inner.read_line(&mut tag)?;
        if n == 0 {
            return Ok(None);
        }
        if !tag.starts_with("FRAME") {
            return Err(Error::from(format!("expected FRAME, got {}", tag.trim())));
        }
        let w = self.info.width as usize;
        let h = self.info.height as usize;
        let (ys, us, vs) = match self.info.chroma {
            Chroma::C444 => (w * h, w * h, w * h),
            Chroma::C420 => {
                let cw = w.div_ceil(2);
                let ch = h.div_ceil(2);
                (w * h, cw * ch, cw * ch)
            }
        };
        let mut y = vec![0u8; ys];
        let mut u = vec![0u8; us];
        let mut v = vec![0u8; vs];
        self.inner.read_exact(&mut y)?;
        self.inner.read_exact(&mut u)?;
        self.inner.read_exact(&mut v)?;
        Ok(Some(yuv_to_rgb(
            w,
            h,
            &y,
            &u,
            &v,
            self.info.chroma,
            self.info.limited,
        )))
    }
}

pub fn open_path(path: &std::path::Path) -> Result<Reader<BufReader<std::fs::File>>> {
    let f = std::fs::File::open(path)?;
    Reader::open(BufReader::new(f))
}

pub fn write_sequence<W: Write>(w: &mut W, info: &Info, frames: &[Vec<u8>]) -> Result<()> {
    write_header(w, info)?;
    for rgb in frames {
        write_rgb_frame(w, info.width, info.height, rgb)?;
    }
    Ok(())
}

pub fn write_header<W: Write>(w: &mut W, info: &Info) -> Result<()> {
    writeln!(
        w,
        "YUV4MPEG2 W{} H{} F{}:{} Ip A0:0 C444 XCOLORRANGE=FULL",
        info.width, info.height, info.fps_num, info.fps_den
    )?;
    Ok(())
}

/// Write one RGB8 frame as 4:4:4 Y4M (full range, BT.601 JPEG).
pub fn write_rgb_frame<W: Write>(w: &mut W, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    let n = width as usize * height as usize;
    if rgb.len() < n * 3 {
        return Err(Error::from("RGB buffer short"));
    }
    writeln!(w, "FRAME")?;
    let mut y = vec![0u8; n];
    let mut u = vec![0u8; n];
    let mut v = vec![0u8; n];
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        y[i] = (0.299 * r + 0.587 * g + 0.114 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
        u[i] = ((-0.168736 * r - 0.331264 * g + 0.5 * b) + 128.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        v[i] = ((0.5 * r - 0.418688 * g - 0.081312 * b) + 128.0)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    w.write_all(&y)?;
    w.write_all(&u)?;
    w.write_all(&v)?;
    Ok(())
}

fn parse_u32(b: &[u8]) -> Result<u32> {
    let s = std::str::from_utf8(b).map_err(|_| Error::from("bad Y4M number"))?;
    s.parse().map_err(|_| Error::from("bad Y4M number"))
}

fn yuv_to_rgb(
    w: usize,
    h: usize,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    chroma: Chroma,
    limited: bool,
) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let cw = match chroma {
        Chroma::C420 => w.div_ceil(2),
        Chroma::C444 => w,
    };
    for j in 0..h {
        for i in 0..w {
            let yp = y[j * w + i] as f32;
            let (ui, vi) = match chroma {
                Chroma::C444 => (j * w + i, j * w + i),
                Chroma::C420 => {
                    let cx = i / 2;
                    let cy = j / 2;
                    let o = cy * cw + cx;
                    (o, o)
                }
            };
            let up = u[ui] as f32;
            let vp = v[vi] as f32;
            let (yf, cb, cr) = if limited {
                (
                    (yp - 16.0) * (255.0 / 219.0),
                    (up - 128.0) * (255.0 / 224.0),
                    (vp - 128.0) * (255.0 / 224.0),
                )
            } else {
                (yp, up - 128.0, vp - 128.0)
            };
            let r = yf + 1.402 * cr;
            let g = yf - 0.344136 * cb - 0.714136 * cr;
            let b = yf + 1.772 * cb;
            let o = (j * w + i) * 3;
            rgb[o] = r.round().clamp(0.0, 255.0) as u8;
            rgb[o + 1] = g.round().clamp(0.0, 255.0) as u8;
            rgb[o + 2] = b.round().clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}
