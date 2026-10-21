//! Lossless palette transform (all colour + extra channels together).

use super::squeeze::EncChannel;

/// Palette meta-channel plus a single index plane.
pub struct PalettePlanes {
    pub meta: EncChannel,
    pub index: EncChannel,
    pub num_c: u32,
    pub nb_colours: u32,
}

/// Build a palette if the image has few unique packed pixels.
///
/// `max_colours == 0` disables. Indices are `0..nb_colours`.
pub fn try_palette(
    width: usize,
    height: usize,
    channels: usize,
    pixels: &[u8],
    max_colours: u32,
) -> Option<PalettePlanes> {
    if max_colours == 0 || !matches!(channels, 1..=4) {
        return None;
    }
    let npx = width * height;
    if npx == 0 || pixels.len() != npx * channels {
        return None;
    }
    // Tiny images: palette header often costs more than it saves.
    if npx < 64 {
        return None;
    }

    use std::collections::HashMap;
    let mut map: HashMap<u32, u16> = HashMap::with_capacity(max_colours.min(256) as usize);
    let mut order: Vec<u32> = Vec::with_capacity(max_colours.min(256) as usize);
    let mut indices = vec![0i32; npx];
    for (i, idx) in indices.iter_mut().enumerate() {
        let key = pack_pixel(pixels, i, channels);
        if let Some(&id) = map.get(&key) {
            *idx = id as i32;
        } else {
            if map.len() as u32 >= max_colours {
                return None;
            }
            let id = map.len() as u16;
            map.insert(key, id);
            order.push(key);
            *idx = id as i32;
        }
    }
    let nb = map.len() as u32;
    if nb <= 1 || nb as usize >= npx / 2 {
        return None;
    }

    let mut pal = vec![0i32; nb as usize * channels];
    for (id, &key) in order.iter().enumerate() {
        let bytes = key.to_le_bytes();
        for c in 0..channels {
            pal[c * nb as usize + id] = bytes[c] as i32;
        }
    }

    Some(PalettePlanes {
        meta: EncChannel::meta(nb, channels as u32, pal),
        index: EncChannel::from_plane(width as u32, height as u32, indices),
        num_c: channels as u32,
        nb_colours: nb,
    })
}

fn pack_pixel(pixels: &[u8], i: usize, channels: usize) -> u32 {
    let o = i * channels;
    let mut v = 0u32;
    for c in 0..channels {
        v |= (pixels[o + c] as u32) << (8 * c);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palettes_few_colors() {
        let mut px = vec![0u8; 16 * 16 * 3];
        for i in 0..256 {
            let o = i * 3;
            let c = if i < 128 { 0u8 } else { 255 };
            px[o] = c;
            px[o + 1] = 0;
            px[o + 2] = 255 - c;
        }
        let p = try_palette(16, 16, 3, &px, 256).expect("palette");
        assert_eq!(p.nb_colours, 2);
        assert_eq!(p.num_c, 3);
        assert_eq!(p.meta.width, 2);
        assert_eq!(p.meta.height, 3);
    }

    #[test]
    fn skips_when_too_many_colors() {
        let mut px = vec![0u8; 16 * 16 * 3];
        for i in 0..256 {
            px[i * 3] = i as u8;
            px[i * 3 + 1] = (i / 2) as u8;
            px[i * 3 + 2] = 3;
        }
        assert!(try_palette(16, 16, 3, &px, 8).is_none());
    }
}
