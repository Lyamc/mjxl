use std::io::{Cursor, Read};
use std::path::Path;

use jxl_oxide::{EnumColourEncoding, JxlImage, Render};

use crate::error::{Error, Result};
use crate::image::{DecodedFrame, ImageInfo};
use crate::simd::{floats_unit_to_rgba, planar_encoded_to_rgba};

/// High-level JPEG XL decoder.
///
/// Bitstream parsing, entropy decode, and VarDCT/modular synthesis come from
/// [jxl-oxide]. RGBA packing (0..=1 display samples → u8) uses [`std::simd`].
#[derive(Debug, Clone)]
pub struct Decoder {
    request_srgb: bool,
    /// When `false`, skip the interleaved linear buffer (PNG/bench). Default `true`.
    keep_linear: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            request_srgb: false,
            keep_linear: true,
        }
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request conversion to sRGB before packing to 8-bit RGBA.
    pub fn request_srgb(mut self, enable: bool) -> Self {
        self.request_srgb = enable;
        self
    }

    /// Keep the interleaved 0..=1 buffer on [`DecodedFrame::linear`].
    pub fn keep_linear(mut self, enable: bool) -> Self {
        self.keep_linear = enable;
        self
    }

    pub fn probe(path: impl AsRef<Path>) -> Result<ImageInfo> {
        let image = JxlImage::builder().open(path).map_err(Error::ReadJxl)?;
        Ok(image_info_from(&image))
    }

    pub fn probe_bytes(data: &[u8]) -> Result<ImageInfo> {
        Self::probe_reader(Cursor::new(data))
    }

    pub fn probe_reader(reader: impl Read) -> Result<ImageInfo> {
        let image = JxlImage::builder().read(reader).map_err(Error::ReadJxl)?;
        Ok(image_info_from(&image))
    }

    pub fn decode_file(&self, path: impl AsRef<Path>) -> Result<Vec<DecodedFrame>> {
        let mut image = JxlImage::builder().open(path).map_err(Error::ReadJxl)?;
        self.decode_image(&mut image)
    }

    /// Decode from an in-memory JXL bitstream. Preferred entry point for WASM.
    pub fn decode_bytes(&self, data: &[u8]) -> Result<Vec<DecodedFrame>> {
        self.decode_reader(Cursor::new(data))
    }

    pub fn decode_reader(&self, reader: impl Read) -> Result<Vec<DecodedFrame>> {
        let mut image = JxlImage::builder().read(reader).map_err(Error::ReadJxl)?;
        self.decode_image(&mut image)
    }

    pub fn decode_image(&self, image: &mut JxlImage) -> Result<Vec<DecodedFrame>> {
        if self.request_srgb {
            apply_srgb(image);
        }
        let renders = render_all_keyframes(image)?;
        let w = image.width();
        let h = image.height();
        Ok(renders
            .iter()
            .map(|r| pack_render(r, self.keep_linear, w, h))
            .collect())
    }
}

pub(crate) fn apply_srgb(image: &mut JxlImage) {
    image.request_color_encoding(EnumColourEncoding::srgb(
        jxl_oxide::color::RenderingIntent::Relative,
    ));
}

pub(crate) fn image_info_from(image: &JxlImage) -> ImageInfo {
    let meta = &image.image_header().metadata;
    let encoding = image
        .frame_header(0)
        .map(|h| match h.encoding {
            jxl_oxide::frame::Encoding::Modular => "modular",
            jxl_oxide::frame::Encoding::VarDct => "vardct",
        })
        .unwrap_or("unknown");
    let jpeg_reconstruction = match image.jpeg_reconstruction_status() {
        jxl_oxide::JpegReconstructionStatus::Available => "available",
        jxl_oxide::JpegReconstructionStatus::Unavailable => "unavailable",
        jxl_oxide::JpegReconstructionStatus::Invalid => "invalid",
        jxl_oxide::JpegReconstructionStatus::NeedMoreData => "need-more-data",
    };
    ImageInfo {
        width: image.width(),
        height: image.height(),
        num_keyframes: image.num_loaded_keyframes(),
        has_alpha: image.pixel_format().has_alpha(),
        bit_depth: meta.bit_depth.bits_per_sample(),
        animation_tps: meta
            .animation
            .as_ref()
            .map(|a| a.tps_numerator / a.tps_denominator.max(1)),
        xyb_encoded: meta.xyb_encoded,
        encoding,
        has_icc: image.original_icc().is_some(),
        jpeg_reconstruction,
    }
}

fn render_all_keyframes(image: &JxlImage) -> Result<Vec<Render>> {
    let count = image.num_loaded_keyframes();
    if count == 0 {
        return Err(Error::NoFrames);
    }
    (0..count)
        .map(|i| image.render_frame(i).map_err(Error::ReadJxl))
        .collect()
}

pub(crate) fn collect_stream_floats(render: &Render) -> (u32, u32, u32, Vec<f32>) {
    let mut stream = render.stream();
    let width = stream.width();
    let height = stream.height();
    let channels = stream.channels();
    let mut floats = vec![0f32; (width * height * channels) as usize];
    stream.write_to_buffer(&mut floats);
    (width, height, channels, floats)
}

pub(crate) fn render_to_rgba(render: &Render) -> DecodedFrame {
    let st = render.stream();
    pack_render(render, true, st.width(), st.height())
}

fn pack_render(render: &Render, keep_linear: bool, width: u32, height: u32) -> DecodedFrame {
    if let Some(frame) = pack_planar_fast(render, keep_linear, width, height) {
        return frame;
    }
    let (width, height, channels, floats) = collect_stream_floats(render);
    let rgba = floats_unit_to_rgba(width, height, channels, &floats);
    DecodedFrame {
        width,
        height,
        rgba,
        linear: if keep_linear { floats } else { Vec::new() },
        linear_channels: channels,
        duration: render.duration(),
    }
}

/// SIMD pack from planar F32 grids when orientation is identity.
fn pack_planar_fast(render: &Render, keep_linear: bool, w: u32, h: u32) -> Option<DecodedFrame> {
    if render.orientation() != 1 {
        return None;
    }
    let wu = w as usize;
    let hu = h as usize;
    let colors = render.color_channels();
    if colors.is_empty() {
        return None;
    }
    let planes: Vec<&[f32]> = colors
        .iter()
        .map(|c| {
            let g = c.as_float()?;
            if g.width() < wu || g.height() < hu {
                return None;
            }
            Some(g.buf())
        })
        .collect::<Option<Vec<_>>>()?;
    let stride = colors[0].as_float()?.width();
    if planes.iter().any(|p| p.len() < stride * hu) {
        return None;
    }

    let mut alpha = None;
    let (ecs, ebufs) = render.extra_channels();
    for (ec, buf) in ecs.iter().zip(ebufs.iter()) {
        if ec.is_alpha() {
            let g = buf.as_float()?;
            if g.width() < wu || g.height() < hu {
                return None;
            }
            alpha = Some(g.buf());
            break;
        }
    }

    if planes.len() != 3 {
        return None;
    }
    let (rgba, linear, linear_channels) = planar_encoded_to_rgba(
        w,
        h,
        planes[0],
        planes[1],
        planes[2],
        alpha,
        stride,
        keep_linear,
    );
    Some(DecodedFrame {
        width: w,
        height: h,
        rgba,
        linear,
        linear_channels,
        duration: render.duration(),
    })
}

pub fn write_png(path: impl AsRef<Path>, frame: &DecodedFrame) -> Result<()> {
    write_png_depth(path, frame, 8)
}

pub fn write_png_depth(path: impl AsRef<Path>, frame: &DecodedFrame, bits: u8) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    write_png_writer_depth(std::io::BufWriter::new(file), frame, bits)
}

pub fn write_png_writer<W: std::io::Write>(w: W, frame: &DecodedFrame) -> Result<()> {
    write_png_writer_depth(w, frame, 8)
}

/// `bits` is 8 or 16 (djxl `--bits_per_sample`).
pub fn write_png_writer_depth<W: std::io::Write>(
    w: W,
    frame: &DecodedFrame,
    bits: u8,
) -> Result<()> {
    let mut encoder = png::Encoder::new(w, frame.width, frame.height);
    encoder.set_color(png::ColorType::Rgba);
    match bits {
        16 => {
            encoder.set_depth(png::BitDepth::Sixteen);
            let data = crate::simd::floats_unit_to_rgba16(
                frame.width,
                frame.height,
                frame.linear_channels,
                &frame.linear,
            );
            let mut packed = vec![0u8; data.len() * 2];
            for (i, &s) in data.iter().enumerate() {
                packed[i * 2..i * 2 + 2].copy_from_slice(&s.to_be_bytes());
            }
            let mut writer = encoder
                .write_header()
                .map_err(|e| Error::WritePng(e.to_string()))?;
            writer
                .write_image_data(&packed)
                .map_err(|e| Error::WritePng(e.to_string()))?;
        }
        _ => {
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(|e| Error::WritePng(e.to_string()))?;
            writer
                .write_image_data(&frame.rgba)
                .map_err(|e| Error::WritePng(e.to_string()))?;
        }
    }
    Ok(())
}

/// Animated PNG (djxl `.png` / `.apng` for multi-frame).
pub fn write_apng(path: impl AsRef<Path>, frames: &[DecodedFrame], tps: u32) -> Result<()> {
    if frames.is_empty() {
        return Err(Error::NoFrames);
    }
    let w = frames[0].width;
    let h = frames[0].height;
    let file = std::fs::File::create(path.as_ref())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .set_animated(frames.len() as u32, 0)
        .map_err(|e| Error::WritePng(e.to_string()))?;
    let tps = tps.max(1);
    let mut writer = encoder
        .write_header()
        .map_err(|e| Error::WritePng(e.to_string()))?;
    for frame in frames {
        let delay = frame.duration.min(u32::from(u16::MAX));
        writer
            .set_frame_delay(delay as u16, tps.min(u32::from(u16::MAX)) as u16)
            .map_err(|e| Error::WritePng(e.to_string()))?;
        writer
            .write_image_data(&frame.rgba)
            .map_err(|e| Error::WritePng(e.to_string()))?;
    }
    writer
        .finish()
        .map_err(|e| Error::WritePng(e.to_string()))?;
    Ok(())
}

/// PGX (djxl): one unsigned 8-bit file per RGB channel next to `path`.
pub fn write_pgx(path: impl AsRef<Path>, frame: &DecodedFrame) -> Result<()> {
    let path = path.as_ref();
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let parent = path.parent().unwrap_or(Path::new("."));
    let w = frame.width;
    let h = frame.height;
    let names = ["_0.pgx", "_1.pgx", "_2.pgx"];
    for (c, suffix) in names.iter().enumerate() {
        let p = parent.join(format!("{stem}{suffix}"));
        let mut f = std::fs::File::create(&p)?;
        use std::io::Write;
        writeln!(f, "PG ML + 8 {w} {h}")?;
        for i in 0..(w * h) as usize {
            f.write_all(&[frame.rgba[i * 4 + c]])?;
        }
    }
    Ok(())
}

/// Portable float map (RGB, little-endian, bottom-up) — djxl `.pfm`.
pub fn write_pfm(path: impl AsRef<Path>, frame: &DecodedFrame) -> Result<()> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let ch = frame.linear_channels as usize;
    let mut f = std::fs::File::create(path.as_ref())?;
    use std::io::Write;
    write!(f, "PF\n{w} {h}\n-1.0\n")?;
    // PFM is bottom-up.
    for y in (0..h).rev() {
        for x in 0..w {
            let i = y * w + x;
            let (r, g, b) = match ch {
                1 => {
                    let v = frame.linear[i];
                    (v, v, v)
                }
                3 => {
                    let o = i * 3;
                    (frame.linear[o], frame.linear[o + 1], frame.linear[o + 2])
                }
                4 => {
                    let o = i * 4;
                    (frame.linear[o], frame.linear[o + 1], frame.linear[o + 2])
                }
                _ => (0.0, 0.0, 0.0),
            };
            f.write_all(&r.to_le_bytes())?;
            f.write_all(&g.to_le_bytes())?;
            f.write_all(&b.to_le_bytes())?;
        }
    }
    Ok(())
}

fn rgb_from_rgba(frame: &DecodedFrame) -> Vec<u8> {
    frame
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

fn gray_from_rgba(frame: &DecodedFrame) -> Vec<u8> {
    frame
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            ((u16::from(p[0]) * 77 + u16::from(p[1]) * 150 + u16::from(p[2]) * 29) / 256) as u8
        })
        .collect()
}

/// Netpbm P6 RGB.
pub fn write_ppm(path: impl AsRef<Path>, frame: &DecodedFrame) -> Result<()> {
    let rgb = rgb_from_rgba(frame);
    let mut f = std::fs::File::create(path.as_ref())?;
    use std::io::Write;
    write!(f, "P6\n{} {}\n255\n", frame.width, frame.height)?;
    f.write_all(&rgb)?;
    Ok(())
}

/// Netpbm P5 grayscale.
pub fn write_pgm(path: impl AsRef<Path>, frame: &DecodedFrame) -> Result<()> {
    let g = gray_from_rgba(frame);
    let mut f = std::fs::File::create(path.as_ref())?;
    use std::io::Write;
    write!(f, "P5\n{} {}\n255\n", frame.width, frame.height)?;
    f.write_all(&g)?;
    Ok(())
}

/// PAM RGB_ALPHA (djxl `.pam`). `bits` is 8 or 16.
pub fn write_pam(path: impl AsRef<Path>, frame: &DecodedFrame) -> Result<()> {
    write_pam_depth(path, frame, 8)
}

pub fn write_pam_depth(path: impl AsRef<Path>, frame: &DecodedFrame, bits: u8) -> Result<()> {
    let mut f = std::fs::File::create(path.as_ref())?;
    use std::io::Write;
    if bits >= 16 {
        let rgba16 = crate::simd::floats_unit_to_rgba16(
            frame.width,
            frame.height,
            frame.linear_channels,
            &frame.linear,
        );
        write!(
            f,
            "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
            frame.width, frame.height
        )?;
        for s in rgba16 {
            f.write_all(&s.to_be_bytes())?;
        }
    } else {
        write!(
            f,
            "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
            frame.width, frame.height
        )?;
        f.write_all(&frame.rgba)?;
    }
    Ok(())
}

/// Reconstruct the original JPEG if the JXL is a lossless JPEG recompress.
pub fn reconstruct_jpeg(path: impl AsRef<Path>, jpeg_out: impl AsRef<Path>) -> Result<()> {
    let image = JxlImage::builder().open(path).map_err(Error::ReadJxl)?;
    let file = std::fs::File::create(jpeg_out.as_ref())?;
    image.reconstruct_jpeg(file).map_err(Error::ReadJxl)
}

/// First EXIF payload from the container, if present.
pub fn extract_exif(path: impl AsRef<Path>) -> Result<Option<Vec<u8>>> {
    let image = JxlImage::builder().open(path).map_err(Error::ReadJxl)?;
    match image.aux_boxes().first_exif() {
        Ok(jxl_oxide::AuxBoxData::Data(exif)) => Ok(Some(exif.payload().to_vec())),
        Ok(_) => Ok(None),
        Err(e) => Err(Error::ReadJxl(e)),
    }
}

/// First XMP/XML box, if present.
pub fn extract_xmp(path: impl AsRef<Path>) -> Result<Option<Vec<u8>>> {
    let image = JxlImage::builder().open(path).map_err(Error::ReadJxl)?;
    match image.aux_boxes().first_xml() {
        jxl_oxide::AuxBoxData::Data(xml) => Ok(Some(xml.to_vec())),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_dice_sample() {
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_file("samples/jxl/dice.jxl")
            .expect("decode dice.jxl");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].width, 800);
        assert_eq!(frames[0].height, 600);
        assert_eq!(frames[0].rgba.len(), 800 * 600 * 4);
        assert!(frames[0].rgba.iter().any(|&b| b != 0));
    }

    #[test]
    fn write_pfm_and_png16_from_dice() {
        let frames = Decoder::new()
            .request_srgb(true)
            .decode_file("samples/jxl/dice.jxl")
            .expect("decode");
        let dir = std::env::temp_dir();
        let pfm = dir.join("o3-jxl-dice.pfm");
        write_pfm(&pfm, &frames[0]).expect("pfm");
        let bytes = std::fs::read(&pfm).unwrap();
        assert!(bytes.starts_with(b"PF\n800 600\n"));
        let png16 = dir.join("o3-jxl-dice16.png");
        write_png_depth(&png16, &frames[0], 16).expect("png16");
        let dec = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(&png16).unwrap(),
        ));
        let reader = dec.read_info().unwrap();
        assert_eq!(reader.info().bit_depth, png::BitDepth::Sixteen);
        let _ = std::fs::remove_file(pfm);
        let _ = std::fs::remove_file(png16);
    }

    #[test]
    fn decode_bytes_matches_file() {
        let bytes = std::fs::read("samples/jxl/dice.jxl").unwrap();
        let from_bytes = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&bytes)
            .unwrap();
        let from_file = Decoder::new()
            .request_srgb(true)
            .decode_file("samples/jxl/dice.jxl")
            .unwrap();
        assert_eq!(from_bytes[0].rgba, from_file[0].rgba);
    }
}
