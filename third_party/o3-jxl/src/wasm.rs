//! wasm-bindgen session for the Trunk progressive-decode demo.

use wasm_bindgen::prelude::*;

use crate::progressive::{ProgressiveDecoder, psnr_rgb};

#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Incremental JXL session: load a bitstream, then `decodeTo(n)` any prefix.
#[wasm_bindgen]
pub struct JxlSession {
    stream: Vec<u8>,
    pos: usize,
    decoder: ProgressiveDecoder,
    rgba: Vec<u8>,
    reference: Vec<u8>,
    width: u32,
    height: u32,
    header_bytes: u32,
    first_frame_bytes: u32,
}

#[wasm_bindgen]
impl JxlSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JxlSession {
        JxlSession {
            stream: Vec::new(),
            pos: 0,
            decoder: ProgressiveDecoder::new().request_srgb(true),
            rgba: Vec::new(),
            reference: Vec::new(),
            width: 0,
            height: 0,
            header_bytes: 0,
            first_frame_bytes: 0,
        }
    }

    #[wasm_bindgen(js_name = streamLen)]
    pub fn stream_len(&self) -> u32 {
        self.stream.len() as u32
    }

    #[wasm_bindgen(js_name = headerSize)]
    pub fn header_size(&self) -> u32 {
        self.header_bytes
    }

    #[wasm_bindgen(js_name = firstFrameSize)]
    pub fn first_frame_size(&self) -> u32 {
        self.first_frame_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn complete(&self) -> bool {
        self.decoder.is_complete()
    }

    /// Load a complete JXL bitstream. Returns the length in bytes.
    #[wasm_bindgen(js_name = loadJxl)]
    pub fn load_jxl(&mut self, bytes: &[u8]) -> Result<u32, JsValue> {
        if bytes.is_empty() {
            return Err(js_err("empty bitstream"));
        }
        self.stream = bytes.to_vec();
        self.reset_decoder();

        let mut scan = ProgressiveDecoder::new().request_srgb(true);
        scan.push(&self.stream).map_err(js_err)?;
        let full = scan
            .render()
            .map_err(js_err)?
            .ok_or_else(|| js_err("JXL did not render"))?;
        self.width = full.width;
        self.height = full.height;
        self.reference = full.rgba;
        self.rgba = vec![0u8; self.reference.len()];

        self.header_bytes = 0;
        self.first_frame_bytes = self.stream.len() as u32;
        let mut probe = ProgressiveDecoder::new().request_srgb(true);
        let mut off = 0usize;
        const STEP: usize = 256;
        while off < self.stream.len() {
            let end = (off + STEP).min(self.stream.len());
            probe.push(&self.stream[off..end]).map_err(js_err)?;
            off = end;
            if self.header_bytes == 0 && probe.has_header() {
                self.header_bytes = off as u32;
            }
            if probe.render().map_err(js_err)?.is_some() {
                self.first_frame_bytes = off as u32;
                break;
            }
        }
        if self.header_bytes == 0 {
            self.header_bytes = self.first_frame_bytes;
        }

        self.reset_decoder();
        Ok(self.stream.len() as u32)
    }

    /// Decode prefix `stream[0..n]`. Growing `n` is incremental; shrinking resets.
    #[wasm_bindgen(js_name = decodeTo)]
    pub fn decode_to(&mut self, n: u32) -> Result<(), JsValue> {
        if self.stream.is_empty() {
            return Err(js_err("no bitstream"));
        }
        let n = (n as usize).min(self.stream.len());
        if n == self.pos && !self.rgba.iter().all(|&b| b == 0) {
            return Ok(());
        }
        if n < self.pos {
            self.reset_decoder();
        }
        if n > self.pos {
            self.decoder
                .push(&self.stream[self.pos..n])
                .map_err(js_err)?;
            self.pos = n;
        }
        if let Some(frame) = self.decoder.render().map_err(js_err)? {
            self.width = frame.width;
            self.height = frame.height;
            self.rgba = frame.rgba;
        }
        Ok(())
    }

    /// Copy current RGBA into `dest` (`width * height * 4` bytes).
    #[wasm_bindgen(js_name = copyRgba)]
    pub fn copy_rgba(&self, dest: &mut [u8]) -> Result<(), JsValue> {
        if dest.len() < self.rgba.len() {
            return Err(js_err("RGBA buffer too small"));
        }
        dest[..self.rgba.len()].copy_from_slice(&self.rgba);
        Ok(())
    }

    /// Copy the full-file decode into `dest` (same layout as [`copy_rgba`](Self::copy_rgba)).
    #[wasm_bindgen(js_name = copyFull)]
    pub fn copy_full(&self, dest: &mut [u8]) -> Result<(), JsValue> {
        if dest.len() < self.reference.len() {
            return Err(js_err("RGBA buffer too small"));
        }
        dest[..self.reference.len()].copy_from_slice(&self.reference);
        Ok(())
    }

    /// PSNR of the current reconstruction vs the full-file decode (RGB only).
    #[wasm_bindgen(js_name = psnrVsFull)]
    pub fn psnr_vs_full(&self) -> f64 {
        psnr_rgb(&self.rgba, &self.reference)
    }
}

impl JxlSession {
    fn reset_decoder(&mut self) {
        self.decoder = ProgressiveDecoder::new().request_srgb(true);
        self.pos = 0;
    }
}
