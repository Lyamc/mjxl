//! JPEG XL rANS (12-bit) for modular residual tokens.
//!
//! General histograms pick shift 0/6/12 (libjxl kFast). Unary is written as
//! binary with a dummy second symbol so mixed-cluster streams never hit
//! jxl-oxide's single-symbol shortcut.
#![allow(dead_code, unused, clippy::all)]

use super::bits::BitWriter;
use super::entropy::{
    LZ77_MIN_LENGTH, for_each_token, for_each_token_ctx, for_each_token_ctx_len, write_cluster_map,
    write_integer_config_400, write_lz77_header_len,
};

const ANS_LOG: u32 = 12;
const ANS_TAB: u32 = 1 << ANS_LOG;
const ANS_SIGNATURE: u32 = 0x13;
const LOG_ALPHA: u32 = 8;
const TABLE: usize = 1 << LOG_ALPHA;
const LOG_BUCKET: u32 = ANS_LOG - LOG_ALPHA;
const BUCKET: u32 = 1 << LOG_BUCKET;
const RECIP_PREC: u32 = 32 + ANS_LOG;

/// One ANS distribution (`log_alphabet_size = 8`).
#[derive(Clone, Debug)]
pub struct AnsHist {
    freq: Vec<u16>,
    ifreq: Vec<u64>,
    reverse_map: Vec<Vec<u16>>,
    alphabet: u32,
    kind: HistKind,
}

#[derive(Clone, Copy, Debug)]
enum HistKind {
    Unary(u32),
    Binary { v0: u32, v1: u32, p0: u16 },
    Flat { alphabet: u32 },
    General { shift: u8 },
}

#[derive(Clone, Copy, Debug)]
pub struct AnsToken {
    pub cluster: u8,
    pub token: u32,
    pub extra_n: u32,
    pub extra: u32,
}

impl AnsHist {
    pub fn from_freqs(freqs: &[u64]) -> Self {
        let used: Vec<usize> = freqs
            .iter()
            .enumerate()
            .filter(|(_, f)| **f > 0)
            .map(|(i, _)| i)
            .collect();
        match used.as_slice() {
            [] => Self::unary(0),
            &[s] => Self::unary(s as u32),
            &[a, b] => Self::binary(freqs, a as u32, b as u32),
            _ => Self::general(freqs),
        }
    }

    fn unary(sym: u32) -> Self {
        // A true unary histogram takes jxl-oxide's single-symbol shortcut,
        // which does not share an alias table with general/binary clusters in
        // the same stream (LZ77 distance, or a constant plane next to Y).
        let other = if sym == 0 { 1 } else { 0 };
        let n = other.max(sym) as usize + 1;
        let mut freqs = vec![0u64; n];
        freqs[sym as usize] = u64::from(ANS_TAB - 1);
        freqs[other as usize] = 1;
        Self::binary(&freqs, other.min(sym), other.max(sym))
    }

    fn binary(freqs: &[u64], a: u32, b: u32) -> Self {
        let fa = freqs.get(a as usize).copied().unwrap_or(0);
        let fb = freqs.get(b as usize).copied().unwrap_or(0);
        let tot = fa.saturating_add(fb).max(1);
        let mut p0 = ((fa * u64::from(ANS_TAB) + tot / 2) / tot) as u16;
        p0 = p0.clamp(1, (ANS_TAB - 1) as u16);
        let mut freq = vec![0u16; TABLE];
        freq[a as usize] = p0;
        freq[b as usize] = (ANS_TAB as u16) - p0;
        Self::from_dist(freq, HistKind::Binary { v0: a, v1: b, p0 })
    }

    fn general(freqs: &[u64]) -> Self {
        // libjxl kFast tries shift 0 / 6 / 12. Shift 0 is powers of two;
        // 6 and 12 keep more of the true distribution.
        let mut best: Option<(f64, Vec<u16>, u8)> = None;
        for shift in [0u8, 6, 12] {
            if let Some(counts) = quantize_counts(freqs, u32::from(shift)) {
                let cost = ans_data_cost(freqs, &counts);
                if best.as_ref().is_none_or(|(c, _, _)| cost < *c) {
                    best = Some((cost, counts, shift));
                }
            }
        }
        let Some((_, counts, shift)) = best else {
            return Self::flat(freqs);
        };
        Self::from_dist(counts, HistKind::General { shift })
    }

    /// Even distribution over `0..alphabet` (ISO “flat” ANS histogram).
    fn flat(freqs: &[u64]) -> Self {
        let alphabet = freqs
            .iter()
            .rposition(|&f| f > 0)
            .map(|i| (i + 1) as u32)
            .unwrap_or(1)
            .clamp(1, TABLE as u32);
        let base = ANS_TAB / alphabet;
        let leftover = ANS_TAB % alphabet;
        let mut counts = vec![0u16; TABLE];
        for i in 0..alphabet as usize {
            counts[i] = (base + u32::from(i < leftover as usize)) as u16;
        }
        Self::from_dist(counts, HistKind::Flat { alphabet })
    }

    fn from_dist(freq: Vec<u16>, kind: HistKind) -> Self {
        let alphabet = freq
            .iter()
            .rposition(|&c| c > 0)
            .map(|i| i + 1)
            .unwrap_or(1) as u32;
        let mut ifreq = vec![0u64; TABLE];
        let mut reverse_map = vec![Vec::new(); TABLE];
        for (s, &f) in freq.iter().enumerate() {
            if f > 0 {
                ifreq[s] = ((1u64 << RECIP_PREC) + u64::from(f) - 1) / u64::from(f);
                reverse_map[s] = vec![0u16; f as usize];
            }
        }
        fill_reverse_map(&freq, &mut reverse_map);
        debug_assert!(
            {
                let mut seen = vec![false; ANS_TAB as usize];
                let mut ok = true;
                for rm in &reverse_map {
                    for &slot in rm {
                        let i = slot as usize;
                        if i >= seen.len() || seen[i] {
                            ok = false;
                            break;
                        }
                        seen[i] = true;
                    }
                }
                ok && seen.iter().all(|&b| b)
            },
            "reverse_map must cover 0..4096 exactly once"
        );
        Self {
            freq,
            ifreq,
            reverse_map,
            alphabet,
            kind,
        }
    }

    fn put_symbol(&self, state: &mut u32, token: u32) -> (u16, u8) {
        // jxl-oxide's single-symbol histogram is a no-op on the 12-bit state
        // (and never renormalises). Using the alias reverse_map here desyncs
        // as soon as a unary cluster is mixed with a general one (LZ77 dist).
        if let HistKind::Unary(_) = self.kind {
            debug_assert_eq!(self.freq[token as usize], ANS_TAB as u16);
            return (0, 0);
        }
        let f = u32::from(self.freq[token as usize]);
        debug_assert!(f > 0, "ANS token {token} has freq 0");
        let mut nbits = 0u8;
        let mut bits = 0u16;
        if (*state >> (32 - ANS_LOG)) >= f {
            bits = (*state & 0xffff) as u16;
            *state >>= 16;
            nbits = 16;
        }
        let v = *state / f;
        let off = *state % f;
        let slot = self.reverse_map[token as usize][off as usize];
        *state = (v << ANS_LOG) + u32::from(slot);
        (bits, nbits)
    }
}

#[derive(Clone, Copy)]
struct Bucket {
    alias_symbol: u16,
    alias_offset: u16,
    alias_cutoff: u16,
}

fn fill_reverse_map(freq: &[u16], reverse_map: &mut [Vec<u16>]) {
    let mut buckets = vec![
        Bucket {
            alias_symbol: 0,
            alias_offset: 0,
            alias_cutoff: 0,
        };
        TABLE
    ];
    let mut underfull = Vec::new();
    let mut overfull = Vec::new();
    for (i, &d) in freq.iter().enumerate() {
        buckets[i].alias_cutoff = d;
        buckets[i].alias_symbol = i as u16;
        match d.cmp(&(BUCKET as u16)) {
            std::cmp::Ordering::Less => underfull.push(i),
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => overfull.push(i),
        }
    }
    for i in freq.len()..TABLE {
        buckets[i].alias_cutoff = 0;
        underfull.push(i);
    }
    while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
        let by = BUCKET as u16 - buckets[u].alias_cutoff;
        buckets[o].alias_cutoff -= by;
        buckets[u].alias_symbol = o as u16;
        buckets[u].alias_offset = buckets[o].alias_cutoff;
        match buckets[o].alias_cutoff.cmp(&(BUCKET as u16)) {
            std::cmp::Ordering::Less => underfull.push(o),
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => overfull.push(o),
        }
    }
    for (i, b) in buckets.iter_mut().enumerate() {
        if b.alias_cutoff == BUCKET as u16 {
            b.alias_symbol = i as u16;
            b.alias_offset = 0;
            b.alias_cutoff = 0;
        } else {
            b.alias_offset -= b.alias_cutoff;
        }
    }

    let mask = BUCKET - 1;
    for idx in 0..ANS_TAB {
        let i = (idx >> LOG_BUCKET) as usize;
        let pos = idx & mask;
        let b = buckets[i];
        let map_to_alias = pos >= u32::from(b.alias_cutoff);
        let (sym, off) = if map_to_alias {
            (u32::from(b.alias_symbol), u32::from(b.alias_offset) + pos)
        } else {
            (i as u32, pos)
        };
        let rm = &mut reverse_map[sym as usize];
        if (off as usize) < rm.len() {
            rm[off as usize] = idx as u16;
        }
    }
}

fn floor_log2(x: u32) -> u32 {
    debug_assert!(x > 0);
    31 - x.leading_zeros()
}

/// libjxl `GetPopulationCountPrecision`.
fn popcount_precision(logcount: u32, shift: u32) -> u32 {
    let r = (shift as i32) - (((ANS_LOG - logcount) >> 1) as i32);
    if r < 0 { 0 } else { (r as u32).min(logcount) }
}

fn quantize_one(c: u32, shift: u32) -> u32 {
    if c <= 1 {
        return c.max(1);
    }
    let logcount = floor_log2(c);
    let bitcount = popcount_precision(logcount, shift);
    let drop = logcount - bitcount;
    let step = 1u32 << drop;
    let base = 1u32 << logcount;
    let max_k = (1u32 << bitcount) - 1;
    let k = ((c - base + step / 2) / step).min(max_k);
    base + k * step
}

fn quantize_counts(freqs: &[u64], shift: u32) -> Option<Vec<u16>> {
    let n = freqs.len().min(TABLE);
    let total: u64 = freqs.iter().take(n).sum::<u64>().max(1);
    let mut omit = freqs
        .iter()
        .take(n)
        .enumerate()
        .max_by_key(|&(i, f)| (*f, std::cmp::Reverse(i)))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut counts = vec![0u16; TABLE];
    let mut acc = 0u32;
    for i in 0..n {
        if i == omit || freqs[i] == 0 {
            continue;
        }
        let mut c = ((freqs[i] * u64::from(ANS_TAB) + total / 2) / total) as u32;
        c = quantize_one(c.max(1), shift);
        counts[i] = c as u16;
        acc = acc.saturating_add(c);
    }
    while acc >= ANS_TAB {
        let mut best = None;
        for (i, &c) in counts.iter().enumerate() {
            if i != omit && c >= 2 {
                let next = quantize_one(u32::from(c) - 1, shift);
                if next < u32::from(c) && next > 0 {
                    best = match best {
                        Some((_, bc)) if u32::from(c) <= bc => best,
                        _ => Some((i, u32::from(c))),
                    };
                }
            }
        }
        let Some((i, c)) = best else { return None };
        let next = quantize_one(c - 1, shift).max(1);
        acc -= c - next;
        counts[i] = next as u16;
    }
    let rest = ANS_TAB.saturating_sub(acc);
    if rest == 0 {
        return None;
    }
    // Huffman 13 is RLE. If another bin already needs 12 bits, make that omit
    // so omit_width stays ≤ 12 (libjxl RebalanceHistogram).
    for i in 0..omit {
        if u32::from(counts[i]) >= 2048 {
            counts[omit] = counts[i];
            omit = i;
            break;
        }
    }
    counts[omit] = rest as u16;
    let mut max_bw = 0u8;
    for (i, &c) in counts.iter().enumerate() {
        if i != omit && c > 0 {
            max_bw = max_bw.max((floor_log2(u32::from(c)) + 1) as u8);
        }
    }
    if max_bw >= 12 {
        return None;
    }
    Some(counts)
}

fn ans_data_cost(freqs: &[u64], counts: &[u16]) -> f64 {
    let n = freqs.len().min(counts.len());
    let mut bits = 0.0;
    for i in 0..n {
        if freqs[i] == 0 {
            continue;
        }
        let c = u32::from(counts[i]).max(1);
        bits += freqs[i] as f64 * (ANS_LOG as f64 - (c as f64).log2());
    }
    bits
}

/// Elias-gamma-like `shift` (libjxl `method = shift + 1`).
fn write_shift(w: &mut BitWriter, shift: u8) {
    let method = u32::from(shift) + 1;
    let upper_bound_log = 3u32; // FloorLog2(ANS_LOG + 1) = FloorLog2(13)
    let log = floor_log2(method);
    w.put(log, (1u64 << log) - 1);
    if log != upper_bound_log {
        w.put_bool(false);
    }
    w.put(log, ((1u64 << log) - 1) & u64::from(method));
}

fn write_u8(w: &mut BitWriter, n: u32) {
    if n == 0 {
        w.put_bool(false);
        return;
    }
    w.put_bool(true);
    let nbits = 31 - n.leading_zeros();
    w.put(3, nbits as u64);
    w.put(nbits, (n - (1 << nbits)) as u64);
}

/// libjxl `kBitWidthLengths` / `kBitWidthSymbols` (LSB-first). Index 13 is RLE.
const BW_LEN: [u8; 14] = [5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7];
const BW_SYM: [u8; 14] = [17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65];

fn write_bit_width(w: &mut BitWriter, width: u8) {
    let want = width.min(12) as usize;
    w.put(u32::from(BW_LEN[want]), u64::from(BW_SYM[want]));
}

fn write_ans_histogram(w: &mut BitWriter, h: &AnsHist) {
    match h.kind {
        HistKind::Unary(sym) => {
            w.put_bool(true);
            w.put_bool(false);
            write_u8(w, sym);
        }
        HistKind::Binary { v0, v1, p0 } => {
            w.put_bool(true);
            w.put_bool(true);
            write_u8(w, v0);
            write_u8(w, v1);
            w.put(12, p0 as u64);
        }
        HistKind::Flat { alphabet } => {
            w.put_bool(false);
            w.put_bool(true);
            write_u8(w, alphabet.max(1) - 1);
        }
        HistKind::General { shift } => {
            w.put_bool(false);
            w.put_bool(false);
            write_shift(w, shift);
            let alphabet = h.alphabet.max(3);
            write_u8(w, alphabet - 3);
            let omit = h
                .freq
                .iter()
                .enumerate()
                .take(alphabet as usize)
                .max_by_key(|&(i, f)| (*f, std::cmp::Reverse(i)))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut bit_width = vec![0u8; alphabet as usize];
            let mut omit_width = 10u8;
            for i in 0..alphabet as usize {
                if i != omit && h.freq[i] > 0 {
                    bit_width[i] = (floor_log2(u32::from(h.freq[i])) + 1) as u8;
                    omit_width = omit_width.max(bit_width[i] + u8::from(i < omit));
                }
            }
            // Huffman 13 is RLE, never an omit width.
            bit_width[omit] = omit_width.min(12);
            for &bw in &bit_width {
                write_bit_width(w, bw);
            }
            if shift != 0 {
                let sh = u32::from(shift);
                for i in 0..alphabet as usize {
                    if i == omit || bit_width[i] <= 1 {
                        continue;
                    }
                    let logcount = u32::from(bit_width[i]) - 1;
                    let bitcount = popcount_precision(logcount, sh);
                    let drop = logcount - bitcount;
                    let extra = (u32::from(h.freq[i]) >> drop) - (1 << bitcount);
                    w.put(bitcount, extra as u64);
                }
            }
        }
    }
}

pub fn write_ans_histograms(
    w: &mut BitWriter,
    codes: &[AnsHist],
    cluster_of: Option<&[u8]>,
    dist: Option<&AnsHist>,
) {
    write_ans_histograms_len(w, codes, cluster_of, dist, LZ77_MIN_LENGTH);
}

/// All `num_dist` contexts share one ANS histogram (simple clustering, 0 bits).
pub fn write_ans_histograms_collapsed(w: &mut BitWriter, num_dist: u32, hist: &AnsHist) {
    w.put_bool(false); // no lz77
    if num_dist > 1 {
        w.put_bool(true); // simple clustering
        w.put(2, 0); // 0 bits → every context in cluster 0
    }
    w.put_bool(false); // ANS
    w.put(2, (LOG_ALPHA - 5) as u64);
    write_integer_config_400_ans(w);
    write_ans_histogram(w, hist);
}

pub fn write_ans_histograms_len(
    w: &mut BitWriter,
    codes: &[AnsHist],
    cluster_of: Option<&[u8]>,
    dist: Option<&AnsHist>,
    min_length: u32,
) {
    if dist.is_some() {
        w.put_bool(true);
        write_lz77_header_len(w, min_length);
    } else {
        w.put_bool(false);
    }
    let n_data = codes.len() as u32;
    let n_clusters = n_data + u32::from(dist.is_some());
    let mut ids: Vec<u8> = if let Some(co) = cluster_of {
        co.to_vec()
    } else {
        (0..n_data).map(|i| i as u8).collect()
    };
    if dist.is_some() {
        ids.push(n_data as u8);
    }
    if ids.len() > 1 {
        write_cluster_map(w, &ids, n_clusters);
    }
    w.put_bool(false); // ANS, not prefix
    w.put(2, (LOG_ALPHA - 5) as u64);
    for _ in 0..n_clusters {
        write_integer_config_400_ans(w);
    }
    for c in codes {
        write_ans_histogram(w, c);
    }
    if let Some(d) = dist {
        write_ans_histogram(w, d);
    }
}

fn write_integer_config_400_ans(w: &mut BitWriter) {
    // log_alphabet_size = 8 → split_exponent field is 4 bits.
    write_integer_config_400(w);
}

pub fn write_ans_tokens(w: &mut BitWriter, hists: &[AnsHist], tokens: &[AnsToken]) {
    write_ans_tokens_ex(w, hists, None, tokens);
}

/// One ANS stream: extra bits then PutSymbol, reverse order, then 32-bit state.
/// `dist` is the LZ77 distance histogram (cluster `codes.len()`).
pub fn write_ans_tokens_ex(
    w: &mut BitWriter,
    codes: &[AnsHist],
    dist: Option<&AnsHist>,
    tokens: &[AnsToken],
) {
    let hist = |cluster: u8| -> &AnsHist {
        let i = cluster as usize;
        if i < codes.len() {
            &codes[i]
        } else {
            dist.expect("ANS distance cluster")
        }
    };
    let mut chunks: Vec<(u64, u8)> = Vec::new();
    let mut allbits = 0u64;
    let mut numall = 0u32;
    let mut addbits = |bits: u64, nbits: u32| {
        if nbits == 0 {
            return;
        }
        if numall + nbits > 56 {
            chunks.push((allbits, numall as u8));
            allbits = 0;
            numall = 0;
        }
        allbits <<= nbits;
        allbits |= bits;
        numall += nbits;
    };
    let mut state = ANS_SIGNATURE << 16;
    for t in tokens.iter().rev() {
        addbits(t.extra as u64, t.extra_n);
        let (bits, nbits) = hist(t.cluster).put_symbol(&mut state, t.token);
        addbits(bits as u64, nbits as u32);
    }
    w.put(32, state as u64);
    w.put(numall, allbits);
    for (bits, n) in chunks.iter().rev() {
        w.put(*n as u32, *bits);
    }
}

pub fn collect_ans_tokens(
    residuals: &[u32],
    lz77: bool,
    cluster: u8,
    dist_cluster: Option<u8>,
    out: &mut Vec<AnsToken>,
) {
    collect_ans_tokens_len(residuals, lz77, cluster, dist_cluster, LZ77_MIN_LENGTH, out);
}

pub fn collect_ans_tokens_len(
    residuals: &[u32],
    lz77: bool,
    cluster: u8,
    dist_cluster: Option<u8>,
    min_length: u32,
    out: &mut Vec<AnsToken>,
) {
    let ctxs = vec![cluster; residuals.len()];
    collect_ans_tokens_ctx_len(residuals, &ctxs, lz77, dist_cluster, min_length, out);
}

pub fn collect_ans_tokens_ctx(
    residuals: &[u32],
    ctxs: &[u8],
    lz77: bool,
    dist_cluster: Option<u8>,
    out: &mut Vec<AnsToken>,
) {
    collect_ans_tokens_ctx_len(residuals, ctxs, lz77, dist_cluster, LZ77_MIN_LENGTH, out);
}

pub fn collect_ans_tokens_ctx_len(
    residuals: &[u32],
    ctxs: &[u8],
    lz77: bool,
    dist_cluster: Option<u8>,
    min_length: u32,
    out: &mut Vec<AnsToken>,
) {
    collect_ans_tokens_ctx_len_dist(residuals, ctxs, lz77, dist_cluster, min_length, 1, out);
}

/// `dist_token` is the raw ANS symbol for the LZ77 distance cluster.
/// Modular uses 1 (special-distance table with `dist_multiplier = width`).
/// HF uses 0 (`dist_multiplier = 0` → value 0, then decoder `+ 1` → RLE).
pub fn collect_ans_tokens_ctx_len_dist(
    residuals: &[u32],
    ctxs: &[u8],
    lz77: bool,
    dist_cluster: Option<u8>,
    min_length: u32,
    dist_token: u32,
    out: &mut Vec<AnsToken>,
) {
    for_each_token_ctx_len(
        residuals,
        ctxs,
        lz77,
        min_length,
        |cluster, token, extra_n, extra| {
            out.push(AnsToken {
                cluster,
                token,
                extra_n,
                extra,
            });
            if lz77 && token >= super::entropy::LZ77_MIN_SYMBOL {
                if let Some(dc) = dist_cluster {
                    out.push(AnsToken {
                        cluster: dc,
                        token: dist_token,
                        extra_n: 0,
                        extra: 0,
                    });
                }
            }
        },
    );
}

pub fn ans_rle_distance() -> AnsHist {
    // Two used symbols so the decoder builds a real alias table instead of the
    // single-symbol shortcut. Token 1 is the RLE distance; token 0 is unused.
    AnsHist::from_freqs(&[1, 1023])
}

pub fn ans_from_residuals(residuals: &[u32], lz77: bool) -> AnsHist {
    let cap = if lz77 { 257 } else { 32 };
    let mut freqs = vec![0u64; cap];
    for_each_token(residuals, lz77, |token, _, _| {
        let t = token as usize;
        if t >= freqs.len() {
            freqs.resize(t + 1, 0);
        }
        freqs[t] += 1;
    });
    if freqs.iter().all(|&x| x == 0) {
        freqs[0] = 1;
    }
    let n = freqs.len().min(19);
    for slot in freqs.iter_mut().take(n) {
        *slot = slot.saturating_add(1);
    }
    AnsHist::from_freqs(&freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_hist_roundtrip_state() {
        let h = AnsHist::from_freqs(&[0, 10]);
        assert!(matches!(h.kind, HistKind::Binary { .. }));
        assert!(h.freq[1] > 4000);
    }

    #[test]
    fn writes_histogram_bytes() {
        let h = AnsHist::from_freqs(&[100, 50, 10, 5, 1]);
        let mut w = BitWriter::with_capacity(64);
        write_ans_histograms(&mut w, &[h], None, None);
        assert!(!w.finish().is_empty());
    }

    #[test]
    fn put_symbol_near_unary_stays_in_range() {
        let h = AnsHist::from_freqs(&[0, 99]);
        let mut state = ANS_SIGNATURE << 16;
        for _ in 0..64 {
            let _ = h.put_symbol(&mut state, 1);
            assert!(state != 0);
        }
    }

    #[test]
    fn general_hist_sparse_lz77_alphabet() {
        let mut freqs = vec![0u64; 242];
        freqs[0] = 400;
        freqs[1] = 40;
        freqs[2] = 8;
        freqs[224] = 12;
        freqs[241] = 3;
        let h = AnsHist::from_freqs(&freqs);
        assert!(matches!(h.kind, HistKind::General { .. }));
        assert_eq!(
            parse_general_counts(&h).as_slice(),
            &h.freq[..h.alphabet as usize]
        );
    }

    #[test]
    fn general_hist_counts_match_decoder() {
        let h = AnsHist::from_freqs(&[80, 40, 20, 10, 5]);
        assert!(matches!(h.kind, HistKind::General { .. }));
        let got = parse_general_counts(&h);
        assert_eq!(
            got.as_slice(),
            &h.freq[..h.alphabet as usize],
            "decoder counts vs encoder freq"
        );
    }

    fn parse_general_counts(h: &AnsHist) -> Vec<u16> {
        let mut w = BitWriter::with_capacity(256);
        write_ans_histogram(&mut w, h);
        let bytes = w.finish();
        let mut br = TestBits::new(&bytes);
        assert_eq!(br.get(1), 0);
        assert_eq!(br.get(1), 0);
        let mut len = 0usize;
        while len < 3 {
            if br.get(1) == 1 {
                len += 1;
            } else {
                break;
            }
        }
        let shift = br.get(len as u32) + (1 << len) - 1;
        let alphabet = read_u8_test(&mut br) as usize + 3;
        let mut codes = vec![0u16; alphabet];
        let mut omit_log = 0u16;
        let mut omit_pos = 0usize;
        for i in 0..alphabet {
            codes[i] = read_prefix_test(&mut br);
            if i == 0 || codes[i] > omit_log {
                omit_log = codes[i];
                omit_pos = i;
            }
        }
        let mut dist = vec![0u16; alphabet];
        let mut acc = 0u32;
        for i in 0..alphabet {
            if i == omit_pos || codes[i] == 0 {
                continue;
            }
            if codes[i] == 1 {
                dist[i] = 1;
            } else {
                let zeros = codes[i] as i16 - 1;
                let bitcount = (shift as i16 - ((12 - zeros) >> 1)).clamp(0, zeros) as u32;
                let extra = br.get(bitcount);
                dist[i] = ((1u32 << zeros) + (extra << (zeros as u32 - bitcount))) as u16;
            }
            acc += u32::from(dist[i]);
        }
        dist[omit_pos] = (ANS_TAB - acc) as u16;
        dist
    }

    fn read_u8_test(br: &mut TestBits<'_>) -> u32 {
        if br.get(1) == 0 {
            0
        } else {
            let n = br.get(3);
            if n == 0 { 1 } else { (1 << n) + br.get(n) }
        }
    }

    fn read_prefix_test(br: &mut TestBits<'_>) -> u16 {
        match br.get(3) {
            0 => 10,
            1 => {
                for val in [4, 0, 11, 13] {
                    if br.get(1) == 1 {
                        return val;
                    }
                }
                12
            }
            2 => 7,
            3 => {
                if br.get(1) == 1 {
                    1
                } else {
                    3
                }
            }
            4 => 6,
            5 => 8,
            6 => 9,
            7 => {
                if br.get(1) == 1 {
                    2
                } else {
                    5
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn put_symbol_general_many_tokens() {
        let h = AnsHist::from_freqs(&[80, 40, 20, 10, 5]);
        assert!(matches!(h.kind, HistKind::General { .. }));
        let seq: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let tokens: Vec<AnsToken> = seq
            .iter()
            .map(|&token| AnsToken {
                cluster: 0,
                token,
                extra_n: 0,
                extra: 0,
            })
            .collect();
        let mut w = BitWriter::with_capacity(512);
        write_ans_tokens(&mut w, std::slice::from_ref(&h), &tokens);
        let bytes = w.finish();
        let mut br = TestBits::new(&bytes);
        let mut st = br.get(32);
        let mut out = Vec::new();
        for _ in 0..seq.len() {
            let (sym, next) = h.decode_symbol(st, &mut br);
            st = next;
            out.push(sym);
        }
        assert_eq!(st, ANS_SIGNATURE << 16, "final ANS state");
        assert_eq!(out, seq);
    }

    #[test]
    fn put_symbol_flat_many_tokens() {
        let h = AnsHist::flat(&[80, 40, 20, 10, 5]);
        let seq: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let tokens: Vec<AnsToken> = seq
            .iter()
            .map(|&token| AnsToken {
                cluster: 0,
                token,
                extra_n: 0,
                extra: 0,
            })
            .collect();
        let mut w = BitWriter::with_capacity(512);
        write_ans_tokens(&mut w, std::slice::from_ref(&h), &tokens);
        let bytes = w.finish();
        let mut br = TestBits::new(&bytes);
        let mut st = br.get(32);
        let mut out = Vec::new();
        for _ in 0..seq.len() {
            let (sym, next) = h.decode_symbol(st, &mut br);
            st = next;
            out.push(sym);
        }
        assert_eq!(st, ANS_SIGNATURE << 16, "final ANS state");
        assert_eq!(out, seq);
    }

    struct TestBits<'a> {
        bytes: &'a [u8],
        bit: usize,
    }
    impl<'a> TestBits<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, bit: 0 }
        }
        fn get(&mut self, n: u32) -> u32 {
            let mut v = 0u32;
            for i in 0..n {
                let b = self.bit / 8;
                let o = self.bit % 8;
                let bit = u32::from(self.bytes.get(b).copied().unwrap_or(0) >> o) & 1;
                v |= bit << i;
                self.bit += 1;
            }
            v
        }
    }

    impl AnsHist {
        fn decode_symbol(&self, mut state: u32, br: &mut TestBits<'_>) -> (u32, u32) {
            let idx = state & (ANS_TAB - 1);
            let mut found = None;
            for (sym, rm) in self.reverse_map.iter().enumerate() {
                if let Some(off) = rm.iter().position(|&s| u32::from(s) == idx) {
                    found = Some((sym as u32, off as u32, self.freq[sym]));
                    break;
                }
            }
            let (sym, off, freq) = found.expect("alias slot");
            let next = (state >> ANS_LOG) * u32::from(freq) + off;
            state = if next < (1 << 16) {
                (next << 16) | br.get(16)
            } else {
                next
            };
            (sym, state)
        }
    }
}
