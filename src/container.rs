//! Timed index of independently coded JPEG XL stills.

use crate::error::{Error, Result};
use std::io::{Read, Write};

pub const MAGIC: &[u8; 4] = b"MJXL";
/// One rectangle (or a full frame, or a skip) per sample. Duration is always ≥ 1.
pub const VERSION: u16 = 2;
/// Same index as v2, but a displayed frame may be several samples. Every sample
/// except the last in that frame has `duration_ticks == 0` (paste, do not present).
pub const VERSION_MULTI: u16 = 3;
/// Bit 0: all samples are modular lossless. Otherwise encoder used VarDCT (lossy RGB) or mixed.
pub const FLAG_LOSSLESS: u16 = 1;
/// Bit 1: samples may be skips (`size=0`) or dirty rects (`w,h` smaller than the canvas).
pub const FLAG_TEMPORAL: u16 = 2;
/// Bit 2: samples are modular JXL (lossless or lossy residuals), not VarDCT.
pub const FLAG_MODULAR: u16 = 4;
/// Bit 3: stills were written with smaller groups / Squeeze so a prefix can paint.
pub const FLAG_PROGRESSIVE: u16 = 8;

#[derive(Clone, Debug)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    pub duration_ticks: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub version: u16,
    pub flags: u16,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub samples: Vec<Sample>,
}

impl Header {
    pub fn nframes(&self) -> usize {
        self.samples.len()
    }

    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den.max(1) as f64
    }

    /// Bytes from file start to the first sample payload.
    pub fn payload_origin(&self) -> u64 {
        index_bytes(self.samples.len())
    }
}

/// Group samples into displayed frames. A `duration_ticks` of 0 (v3) means
/// "paste this rectangle, then keep reading". v1/v2 files have duration ≥ 1,
/// so each sample is its own frame.
pub fn display_frames(header: &Header) -> Vec<Vec<&Sample>> {
    let mut frames = Vec::new();
    let mut cur = Vec::new();
    for s in &header.samples {
        cur.push(s);
        if s.duration_ticks > 0 {
            frames.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        frames.push(cur);
    }
    frames
}

const FIXED: usize = 4 + 2 + 2 + 4 * 6; // magic + ver + flags + 6×u32
const SAMPLE: usize = 8 + 4 + 4 + 8; // offset, size, dur, x,y,w,h as u16

pub fn index_bytes(nframes: usize) -> u64 {
    (FIXED + nframes * SAMPLE) as u64
}

pub fn write_file<W: Write>(
    w: W,
    flags: u16,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    frames: &[(u32, &[u8])],
) -> Result<()> {
    let coded: Vec<crate::temporal::Coded> = frames
        .iter()
        .map(|&(d, data)| crate::temporal::Coded {
            duration: d,
            patches: vec![crate::temporal::Patch {
                jxl: data.to_vec(),
                x: 0,
                y: 0,
                w: width,
                h: height,
            }],
        })
        .collect();
    write_coded(w, flags, width, height, fps_num, fps_den, &coded)
}

pub fn write_coded<W: Write>(
    mut w: W,
    flags: u16,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    frames: &[crate::temporal::Coded],
) -> Result<()> {
    if frames.is_empty() {
        return Err(Error::from("need at least one frame"));
    }
    if frames[0].is_skip() {
        return Err(Error::from("first sample cannot be a skip"));
    }
    let flat = flatten(frames, width, height)?;
    let version = if flat.iter().any(|s| s.duration_ticks == 0) {
        VERSION_MULTI
    } else {
        VERSION
    };
    let origin = index_bytes(flat.len());
    let mut samples = Vec::with_capacity(flat.len());
    let mut off = origin;
    for f in &flat {
        samples.push(Sample {
            offset: off,
            size: f.jxl.len() as u32,
            duration_ticks: f.duration_ticks,
            x: f.x,
            y: f.y,
            w: f.w,
            h: f.h,
        });
        off += f.jxl.len() as u64;
    }

    w.write_all(MAGIC)?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;
    w.write_all(&width.to_le_bytes())?;
    w.write_all(&height.to_le_bytes())?;
    w.write_all(&fps_num.to_le_bytes())?;
    w.write_all(&fps_den.max(1).to_le_bytes())?;
    w.write_all(&(flat.len() as u32).to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?;
    for s in &samples {
        w.write_all(&s.offset.to_le_bytes())?;
        w.write_all(&s.size.to_le_bytes())?;
        w.write_all(&s.duration_ticks.to_le_bytes())?;
        w.write_all(&(s.x as u16).to_le_bytes())?;
        w.write_all(&(s.y as u16).to_le_bytes())?;
        w.write_all(&(s.w as u16).to_le_bytes())?;
        w.write_all(&(s.h as u16).to_le_bytes())?;
    }
    for f in &flat {
        w.write_all(&f.jxl)?;
    }
    Ok(())
}

struct Flat {
    jxl: Vec<u8>,
    duration_ticks: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

fn flatten(frames: &[crate::temporal::Coded], width: u32, height: u32) -> Result<Vec<Flat>> {
    let mut flat = Vec::new();
    for f in frames {
        if f.patches.is_empty() {
            flat.push(Flat {
                jxl: Vec::new(),
                duration_ticks: f.duration.max(1),
                x: 0,
                y: 0,
                w: width,
                h: height,
            });
            continue;
        }
        let last = f.patches.len() - 1;
        for (i, p) in f.patches.iter().enumerate() {
            if p.jxl.len() > u32::MAX as usize {
                return Err(Error::from("frame exceeds 4 GiB"));
            }
            if p.jxl.len() < 2 || p.jxl[0] != 0xFF || p.jxl[1] != 0x0A {
                return Err(Error::from("sample is not a naked JPEG XL codestream"));
            }
            if p.x > u16::MAX as u32
                || p.y > u16::MAX as u32
                || p.w > u16::MAX as u32
                || p.h > u16::MAX as u32
            {
                return Err(Error::from("rectangle does not fit in the u16 index"));
            }
            flat.push(Flat {
                jxl: p.jxl.clone(),
                duration_ticks: if i == last { f.duration.max(1) } else { 0 },
                x: p.x,
                y: p.y,
                w: if p.w == 0 { width } else { p.w },
                h: if p.h == 0 { height } else { p.h },
            });
        }
    }
    Ok(flat)
}

pub fn read_header(data: &[u8]) -> Result<Header> {
    if data.len() < FIXED {
        return Err(Error::from("truncated mjxl header"));
    }
    if &data[0..4] != MAGIC {
        return Err(Error::from("not an mjxl file (bad magic)"));
    }
    let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
    if version != 1 && version != VERSION && version != VERSION_MULTI {
        return Err(Error::from(format!("unsupported mjxl version {version}")));
    }
    let sample_bytes = if version == 1 { 16 } else { SAMPLE };
    let flags = u16::from_le_bytes(data[6..8].try_into().unwrap());
    let width = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let fps_num = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let fps_den = u32::from_le_bytes(data[20..24].try_into().unwrap()).max(1);
    let nframes = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
    let need = FIXED + nframes * sample_bytes;
    if data.len() < need {
        return Err(Error::from("truncated mjxl index"));
    }
    let mut samples = Vec::with_capacity(nframes);
    let mut p = FIXED;
    for _ in 0..nframes {
        let offset = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
        let size = u32::from_le_bytes(data[p + 8..p + 12].try_into().unwrap());
        let duration_ticks = u32::from_le_bytes(data[p + 12..p + 16].try_into().unwrap());
        let (x, y, w, h) = if version == 1 {
            (0, 0, width, height)
        } else {
            (
                u16::from_le_bytes(data[p + 16..p + 18].try_into().unwrap()) as u32,
                u16::from_le_bytes(data[p + 18..p + 20].try_into().unwrap()) as u32,
                u16::from_le_bytes(data[p + 20..p + 22].try_into().unwrap()) as u32,
                u16::from_le_bytes(data[p + 22..p + 24].try_into().unwrap()) as u32,
            )
        };
        samples.push(Sample {
            offset,
            size,
            duration_ticks,
            x,
            y,
            w: if w == 0 { width } else { w },
            h: if h == 0 { height } else { h },
        });
        p += sample_bytes;
    }
    Ok(Header {
        version,
        flags,
        width,
        height,
        fps_num,
        fps_den,
        samples,
    })
}

pub fn sample_bytes<'a>(file: &'a [u8], s: &Sample) -> Result<&'a [u8]> {
    let start = s.offset as usize;
    let end = start.saturating_add(s.size as usize);
    if end > file.len() {
        return Err(Error::from("sample extends past end of file"));
    }
    Ok(&file[start..end])
}

pub fn read_file<R: Read>(mut r: R) -> Result<(Header, Vec<u8>)> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    let header = read_header(&buf)?;
    Ok((header, buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_index() {
        let jxl = {
            let mut v = vec![0xFF, 0x0A];
            v.extend_from_slice(&[1, 2, 3, 4]);
            v
        };
        let mut out = Vec::new();
        write_file(
            &mut out,
            0,
            16,
            16,
            24,
            1,
            &[(1, jxl.as_slice()), (1, jxl.as_slice())],
        )
        .unwrap();
        let h = read_header(&out).unwrap();
        assert_eq!(h.width, 16);
        assert_eq!(h.nframes(), 2);
        assert_eq!(sample_bytes(&out, &h.samples[0]).unwrap()[0], 0xFF);
        assert_eq!(sample_bytes(&out, &h.samples[1]).unwrap().len(), jxl.len());
        assert_eq!(h.version, VERSION);
        assert_eq!(display_frames(&h).len(), 2);
    }

    #[test]
    fn two_rects_are_one_display_frame() {
        use crate::temporal::{Coded, Patch};
        let a = vec![0xFF, 0x0A, 1];
        let b = vec![0xFF, 0x0A, 2, 3];
        let coded = [Coded {
            duration: 1,
            patches: vec![
                Patch {
                    jxl: a.clone(),
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
                Patch {
                    jxl: b.clone(),
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            ],
        }];
        let mut out = Vec::new();
        write_coded(&mut out, FLAG_TEMPORAL, 32, 32, 24, 1, &coded).unwrap();
        let h = read_header(&out).unwrap();
        assert_eq!(h.version, VERSION_MULTI);
        assert_eq!(h.samples.len(), 2);
        assert_eq!(h.samples[0].duration_ticks, 0);
        assert_eq!(h.samples[1].duration_ticks, 1);
        assert_eq!(h.samples[0].x, 0);
        assert_eq!(h.samples[1].x, 8);
        assert_eq!(display_frames(&h).len(), 1);
        assert_eq!(sample_bytes(&out, &h.samples[0]).unwrap(), a.as_slice());
        assert_eq!(sample_bytes(&out, &h.samples[1]).unwrap(), b.as_slice());
    }
}
