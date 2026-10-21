//! Decode any prefix of a JPEG XL bitstream.
//!
//! Once the container header is available, [`ProgressiveDecoder::render`] tries
//! [`jxl_oxide::JxlImage::render_loading_frame`] so a truncated progressive
//! (or grouped sequential) file still produces a full-canvas image. Extra bytes
//! refine the same reconstruction.

use jxl_oxide::{InitializeResult, JxlImage, UninitializedJxlImage};

use crate::decoder::{apply_srgb, image_info_from, render_to_rgba};
use crate::error::{Error, Result};
use crate::image::{DecodedFrame, ImageInfo};

enum Inner {
    Uninit(UninitializedJxlImage),
    Init(JxlImage),
}

/// Streaming decoder: [`push`](Self::push) any number of bytes, then
/// [`render`](Self::render). Shrinking the prefix requires a new decoder
/// (JPEG XL state cannot rewind).
pub struct ProgressiveDecoder {
    inner: Option<Inner>,
    request_srgb: bool,
    bytes_in: usize,
}

impl Default for ProgressiveDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressiveDecoder {
    pub fn new() -> Self {
        Self {
            inner: Some(Inner::Uninit(JxlImage::builder().build_uninit())),
            request_srgb: false,
            bytes_in: 0,
        }
    }

    pub fn request_srgb(mut self, enable: bool) -> Self {
        self.request_srgb = enable;
        self
    }

    /// Bytes handed to [`push`](Self::push) so far.
    pub fn bytes_fed(&self) -> usize {
        self.bytes_in
    }

    /// Header has been parsed; [`info`](Self::info) is `Some`.
    pub fn has_header(&self) -> bool {
        matches!(self.inner, Some(Inner::Init(_)))
    }

    /// Codestream is complete (no missing frames or groups).
    pub fn is_complete(&self) -> bool {
        match &self.inner {
            Some(Inner::Init(image)) => image.is_loading_done(),
            _ => false,
        }
    }

    pub fn info(&self) -> Option<ImageInfo> {
        match &self.inner {
            Some(Inner::Init(image)) => Some(image_info_from(image)),
            _ => None,
        }
    }

    /// Append bitstream bytes. Returns how many bytes this call accepted.
    pub fn push(&mut self, bytes: &[u8]) -> Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let mut rest = bytes;
        while !rest.is_empty() {
            let n = self.feed(rest)?;
            if n == 0 {
                break;
            }
            rest = &rest[n.min(rest.len())..];
            self.promote()?;
        }
        self.bytes_in += bytes.len();
        Ok(bytes.len())
    }

    /// Render whatever has been fed. `Ok(None)` means the prefix is too short
    /// for a canvas (no header, or header but no groups yet).
    pub fn render(&mut self) -> Result<Option<DecodedFrame>> {
        self.promote()?;
        let Some(Inner::Init(image)) = self.inner.as_mut() else {
            return Ok(None);
        };
        if image.is_loading_done() && image.num_loaded_keyframes() > 0 {
            let render = image.render_frame(0).map_err(Error::ReadJxl)?;
            return Ok(Some(render_to_rgba(&render)));
        }
        match image.render_loading_frame() {
            Ok(render) => Ok(Some(render_to_rgba(&render))),
            Err(_) => Ok(None),
        }
    }

    /// Decode `data[..max_bytes]` in one shot.
    pub fn decode_prefix(
        data: &[u8],
        max_bytes: usize,
        request_srgb: bool,
    ) -> Result<Option<DecodedFrame>> {
        let n = max_bytes.min(data.len());
        let mut dec = Self::new().request_srgb(request_srgb);
        dec.push(&data[..n])?;
        dec.render()
    }

    fn feed(&mut self, bytes: &[u8]) -> Result<usize> {
        match self.inner.as_mut() {
            Some(Inner::Uninit(image)) => image.feed_bytes(bytes).map_err(Error::ReadJxl),
            Some(Inner::Init(image)) => image.feed_bytes(bytes).map_err(Error::ReadJxl),
            None => unreachable!("decoder inner missing"),
        }
    }

    fn promote(&mut self) -> Result<bool> {
        match self.inner.take() {
            Some(Inner::Uninit(uninit)) => match uninit.try_init().map_err(Error::ReadJxl)? {
                InitializeResult::NeedMoreData(uninit) => {
                    self.inner = Some(Inner::Uninit(uninit));
                    Ok(false)
                }
                InitializeResult::Initialized(mut image) => {
                    if self.request_srgb {
                        apply_srgb(&mut image);
                    }
                    self.inner = Some(Inner::Init(image));
                    Ok(true)
                }
            },
            other => {
                self.inner = other;
                Ok(true)
            }
        }
    }
}

/// RGB PSNR of two RGBA buffers (alpha ignored).
pub fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 4 {
        return 0.0;
    }
    let mut sse = 0.0;
    let mut count = 0.0;
    let mut i = 0;
    while i + 3 < n {
        for c in 0..3 {
            let d = a[i + c] as f64 - b[i + c] as f64;
            sse += d * d;
            count += 1.0;
        }
        i += 4;
    }
    if sse < 1e-9 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 / (sse / count)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Decoder;

    const DICE: &str = "samples/jxl/dice.jxl";

    fn load_dice() -> Vec<u8> {
        std::fs::read(DICE).expect("samples/jxl/dice.jxl")
    }

    #[test]
    fn full_prefix_matches_decode_bytes() {
        let data = load_dice();
        let full = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&data)
            .unwrap();
        let prefix = ProgressiveDecoder::decode_prefix(&data, data.len(), true)
            .unwrap()
            .expect("full prefix should render");
        assert_eq!(prefix.width, full[0].width);
        assert_eq!(prefix.height, full[0].height);
        assert_eq!(prefix.rgba, full[0].rgba);
    }

    #[test]
    fn tiny_prefix_is_none() {
        let data = load_dice();
        let frame = ProgressiveDecoder::decode_prefix(&data, 16, true).unwrap();
        assert!(frame.is_none());
    }

    #[test]
    fn truncated_prefixes_eventually_render() {
        let data = load_dice();
        let full = Decoder::new()
            .request_srgb(true)
            .decode_bytes(&data)
            .unwrap()
            .remove(0);

        let mut first = None;
        let mut last_psnr = 0.0;
        for pct in [2, 5, 10, 15, 25, 40, 60, 80, 100] {
            let n = (data.len() * pct / 100).max(1);
            let frame = ProgressiveDecoder::decode_prefix(&data, n, true).unwrap();
            if let Some(frame) = frame {
                assert_eq!(frame.width, full.width);
                assert_eq!(frame.height, full.height);
                assert_eq!(frame.rgba.len(), full.rgba.len());
                let psnr = psnr_rgb(&frame.rgba, &full.rgba);
                if first.is_none() {
                    first = Some((pct, n, psnr));
                }
                last_psnr = psnr;
            }
        }
        let (pct, n, psnr) = first.expect("dice.jxl should render at some truncated size");
        assert!(
            pct < 100,
            "only the complete file rendered; progressive groups never appeared"
        );
        assert!(n < data.len());
        assert!(
            last_psnr > 40.0,
            "full prefix PSNR vs decode_bytes was {last_psnr}"
        );
        let _ = psnr;
    }

    #[test]
    fn incremental_push_matches_oneshot() {
        let data = load_dice();
        let n = data.len() / 2;
        let oneshot = ProgressiveDecoder::decode_prefix(&data, n, true).unwrap();

        let mut inc = ProgressiveDecoder::new().request_srgb(true);
        let mut off = 0;
        while off < n {
            let end = (off + 1024).min(n);
            inc.push(&data[off..end]).unwrap();
            off = end;
        }
        let incremental = inc.render().unwrap();
        match (oneshot, incremental) {
            (None, None) => {}
            (Some(a), Some(b)) => assert_eq!(a.rgba, b.rgba),
            (oneshot, incremental) => panic!(
                "oneshot vs incremental: {} vs {}",
                oneshot.is_some(),
                incremental.is_some()
            ),
        }
    }
}
