/// Decoded frame in 8-bit sRGB (or grayscale expanded to RGB).
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// RGBA8 pixels, row-major (sRGB OETF applied).
    pub rgba: Vec<u8>,
    /// Interleaved 0..=1 samples from the renderer (1, 3, or 4 channels).
    pub linear: Vec<f32>,
    pub linear_channels: u32,
    pub duration: u32,
}

impl DecodedFrame {
    pub fn channels(&self) -> usize {
        4
    }

    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Image metadata extracted without full decode.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub num_keyframes: usize,
    pub has_alpha: bool,
    pub bit_depth: u32,
    pub animation_tps: Option<u32>,
    /// `true` if the codestream uses XYB.
    pub xyb_encoded: bool,
    /// First-frame coding: `"modular"`, `"vardct"`, or `"unknown"`.
    pub encoding: &'static str,
    pub has_icc: bool,
    /// JPEG reconstruction box: available / unavailable / invalid / need-more-data.
    pub jpeg_reconstruction: &'static str,
}
