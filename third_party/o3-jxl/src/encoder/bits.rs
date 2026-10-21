//! LSB-first bit writer matching JPEG XL / jxl-oxide.

/// JPEG XL bitstream writer (least-significant bit first).
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    scratch: u64,
    nbits: u32,
}

impl BitWriter {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            scratch: 0,
            nbits: 0,
        }
    }

    /// Write `n` low bits of `bits`. `n` must be `<= 56`.
    #[inline(always)]
    pub fn put(&mut self, n: u32, bits: u64) {
        debug_assert!(n <= 56);
        debug_assert!(n == 0 || bits < (1u64 << n) || n == 64);
        if n == 0 {
            return;
        }
        self.scratch |= bits << self.nbits;
        self.nbits += n;
        if self.nbits >= 8 {
            let nbytes = self.nbits / 8;
            let len = self.buf.len();
            self.buf.reserve(8);
            unsafe {
                let p = self.buf.as_mut_ptr().add(len);
                p.cast::<u64>().write_unaligned(self.scratch.to_le());
                self.buf.set_len(len + nbytes as usize);
            }
            self.scratch >>= nbytes * 8;
            self.nbits -= nbytes * 8;
        }
    }

    #[inline]
    pub fn put_bool(&mut self, v: bool) {
        self.put(1, u64::from(v));
    }

    pub fn zero_pad_to_byte(&mut self) {
        if self.nbits != 0 {
            self.put(8 - self.nbits, 0);
        }
    }

    /// Finish and return the byte buffer, zero-padded to a byte boundary.
    pub fn finish(mut self) -> Vec<u8> {
        self.zero_pad_to_byte();
        self.buf
    }

    pub fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if self.nbits == 0 {
            self.buf.extend_from_slice(bytes);
            return;
        }
        for &b in bytes {
            self.put(8, b as u64);
        }
    }

    /// IEEE binary16, same bit layout as libjxl `F16Coder::Write`.
    pub fn put_f16(&mut self, v: f32) {
        self.put(16, f32_to_f16_bits(v) as u64);
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | (mant >> 13) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign;
    }
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((new_exp as u16) << 10) | (mant >> 13) as u16
}

/// A U32 distribution selector as in the JPEG XL spec.
#[derive(Clone, Copy, Debug)]
pub enum U32 {
    Const(u32),
    Bits { offset: u32, n: u32 },
}

impl U32 {
    fn try_encode(self, value: u32) -> Option<(u32, u64)> {
        match self {
            U32::Const(c) => (value == c).then_some((0, 0)),
            U32::Bits { offset, n } => {
                if value >= offset {
                    let v = value - offset;
                    if n >= 32 || v < (1u32 << n) {
                        Some((n, v as u64))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }
}

/// `u(n)` starting at 0.
#[inline]
pub const fn u(n: u32) -> U32 {
    U32::Bits { offset: 0, n }
}

/// `offset + u(n)`.
#[inline]
pub const fn off(offset: u32, n: u32) -> U32 {
    U32::Bits { offset, n }
}

pub fn write_u32(w: &mut BitWriter, value: u32, d0: U32, d1: U32, d2: U32, d3: U32) {
    let specs = [d0, d1, d2, d3];
    for (sel, spec) in specs.iter().enumerate() {
        if let Some((n, bits)) = spec.try_encode(value) {
            w.put(2, sel as u64);
            if n > 0 {
                w.put(n, bits);
            }
            return;
        }
    }
    panic!("U32 value {value} does not fit the given distribution");
}

/// JPEG XL `U64`.
pub fn write_u64(w: &mut BitWriter, value: u64) {
    if value == 0 {
        w.put(2, 0);
        return;
    }
    if value <= 16 {
        w.put(2, 1);
        w.put(4, value - 1);
        return;
    }
    if value <= 272 {
        w.put(2, 2);
        w.put(8, value - 17);
        return;
    }
    w.put(2, 3);
    w.put(12, value & 0xFFF);
    let mut v = value >> 12;
    let mut shift = 12;
    while v > 0 {
        w.put(1, 1);
        if shift == 60 {
            w.put(4, v);
            return;
        }
        w.put(8, v & 0xFF);
        v >>= 8;
        shift += 8;
    }
    w.put(1, 0);
}

/// JPEG XL enum: `U32(0, 1, 2 + u(4), 18 + u(6))`.
pub fn write_enum(w: &mut BitWriter, value: u32) {
    write_u32(
        w,
        value,
        U32::Const(0),
        U32::Const(1),
        off(2, 4),
        off(18, 6),
    );
}

/// SizeHeader `U32(1+u(9), 1+u(13), 1+u(18), 1+u(30))`.
pub fn write_dim(w: &mut BitWriter, size: u32) {
    assert!((1..(1 << 30)).contains(&size));
    write_u32(w, size, off(1, 9), off(1, 13), off(1, 18), off(1, 30));
}

/// TOC entry size `U32(u(10), 1024+u(14), 17408+u(22), 4211712+u(30))`.
pub fn write_toc_size(w: &mut BitWriter, size: u32) {
    write_u32(
        w,
        size,
        u(10),
        off(1024, 14),
        off(17408, 22),
        off(4_211_712, 30),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsb_first_bytes() {
        let mut w = BitWriter::new();
        w.put(16, 0x0AFF);
        let b = w.finish();
        assert_eq!(b, [0xFF, 0x0A]);
    }

    #[test]
    fn put_many_small_symbols() {
        let mut w = BitWriter::new();
        for i in 0..1000u64 {
            w.put(3, i & 7);
        }
        let b = w.finish();
        assert!(!b.is_empty());
        let mut w2 = BitWriter::new();
        for i in 0..1000u64 {
            w2.put(3, i & 7);
        }
        assert_eq!(b, w2.finish());
    }

    #[test]
    fn u32_small_dim() {
        let mut w = BitWriter::new();
        write_dim(&mut w, 8);
        // selector 00, 9 bits of 7
        let b = w.finish();
        assert_eq!(b[0] & 0b11, 0);
    }
}
