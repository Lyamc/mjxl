//! Prefix-code entropy coding (hybrid uint, Huffman, MA tree).

use super::bits::{BitWriter, U32, off, write_u32};

const MAX_PREFIX_BITS: u8 = 15;

/// HybridUint config 000: `token = 0` if `value==0`, else `1+floor(log2(value))`
/// with `floor(log2(value))` extra bits.
#[inline]
pub fn hybrid_uint_000(value: u32) -> (u32, u32, u32) {
    if value == 0 {
        return (0, 0, 0);
    }
    let n = 31 - value.leading_zeros();
    let bits = value - (1 << n);
    (1 + n, n, bits)
}

#[inline]
pub fn pack_signed(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

#[inline]
pub fn unpack_signed(x: u32) -> i32 {
    ((x >> 1) as i32) ^ -((x & 1) as i32)
}

/// Huffman code lengths for `freqs`, unused symbols get length 0.
pub fn huffman_lengths(freqs: &[u64], max_len: u8) -> Vec<u8> {
    let n = freqs.len();
    let mut lengths = vec![0u8; n];
    let used: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();
    match used.len() {
        0 => return lengths,
        1 => {
            // Complex prefix codes cannot represent a single-symbol alphabet
            // (length 0); callers should use the simple-histogram path.
            lengths[used[0]] = 1;
            return lengths;
        }
        _ => {}
    }

    let mut parent = vec![usize::MAX; 2 * used.len()];
    let mut heap: Vec<(u64, usize)> = used
        .iter()
        .enumerate()
        .map(|(compact, &sym)| (freqs[sym], compact))
        .collect();
    let mut next = used.len();

    fn pop_min(heap: &mut Vec<(u64, usize)>) -> (u64, usize) {
        let (min_i, _) = heap
            .iter()
            .enumerate()
            .min_by_key(|(_, x)| (x.0, x.1))
            .unwrap();
        heap.swap_remove(min_i)
    }

    while heap.len() > 1 {
        let (f0, a) = pop_min(&mut heap);
        let (f1, b) = pop_min(&mut heap);
        parent[a] = next;
        parent[b] = next;
        heap.push((f0.saturating_add(f1), next));
        next += 1;
    }

    // Walk from last internal node down. Easier: for each leaf, count hops to root.
    for compact in 0..used.len() {
        let mut d = 0u8;
        let mut node = compact;
        while parent[node] != usize::MAX {
            d = d.saturating_add(1);
            node = parent[node];
            if d == 64 {
                break;
            }
        }
        lengths[used[compact]] = d.max(1);
    }

    limit_lengths(&mut lengths, max_len);
    kraft_repair(&mut lengths, max_len);
    lengths
}

fn limit_lengths(lengths: &mut [u8], max_len: u8) {
    for l in lengths.iter_mut() {
        if *l > max_len {
            *l = max_len;
        }
    }
}

fn kraft_space(lengths: &[u8], max_len: u8) -> i32 {
    let target = 1i32 << max_len;
    let mut acc = 0i32;
    for &l in lengths {
        if l > 0 {
            acc += 1 << (max_len - l);
        }
    }
    acc - target
}

fn kraft_repair(lengths: &mut [u8], max_len: u8) {
    let used: Vec<usize> = (0..lengths.len()).filter(|&i| lengths[i] > 0).collect();
    if used.len() < 2 {
        return;
    }
    for _ in 0..4096 {
        let s = kraft_space(lengths, max_len);
        if s == 0 {
            return;
        }
        if s > 0 {
            // Too much capacity: lengthen the shortest code.
            let i = *used.iter().min_by_key(|&&i| lengths[i]).unwrap();
            if lengths[i] < max_len {
                lengths[i] += 1;
            } else {
                // All at max_len — should only happen if used.len() > 2^max_len.
                return;
            }
        } else {
            // Too little capacity: shorten the longest code.
            let i = *used.iter().max_by_key(|&&i| lengths[i]).unwrap();
            if lengths[i] > 1 {
                lengths[i] -= 1;
            } else {
                return;
            }
        }
    }
}

fn bit_reverse(nbits: u32, bits: u16) -> u16 {
    bits.reverse_bits() >> (16 - nbits)
}

/// Canonical prefix codes (bit-reversed for LSB-first write).
pub fn canonical_codes(nbits: &[u8]) -> Vec<u16> {
    const MAX: usize = 16;
    let mut counts = [0u16; MAX];
    for &l in nbits {
        if l > 0 {
            counts[l as usize] += 1;
        }
    }
    let mut next = [0u16; MAX];
    let mut code = 0u16;
    for i in 1..MAX {
        code = (code + counts[i - 1]) << 1;
        next[i] = code;
    }
    let mut bits = vec![0u16; nbits.len()];
    for (i, &l) in nbits.iter().enumerate() {
        if l > 0 {
            let c = next[l as usize];
            next[l as usize] = c.wrapping_add(1);
            bits[i] = bit_reverse(l as u32, c);
        }
    }
    bits
}

/// A prefix histogram we can emit and later encode symbols with.
///
/// Hybrid-uint 000 of values `0..=255` (covers `|AC| <= 128`) is a 256-entry
/// LUT so the HF inner loop never recomputes `log2` or indexes `Vec`s.
#[derive(Clone, Debug)]
pub struct PrefixCode {
    pub nbits: Vec<u8>,
    pub bits: Vec<u16>,
    /// One used symbol → 0-bit codes; skip `write_symbol`.
    trivial: bool,
    /// Combined prefix+extra length for `hybrid_uint_000(0..=255)`.
    lut_n: [u8; 256],
    /// Combined prefix+extra bits (LSB-first prefix, extra shifted by `nbits`).
    lut_b: [u32; 256],
    lut400_n: [u8; 256],
    lut400_b: [u32; 256],
}

fn hybrid_lut(
    nbits: &[u8],
    bits: &[u16],
    trivial: bool,
    hybrid: fn(u32) -> (u32, u32, u32),
) -> ([u8; 256], [u32; 256]) {
    let mut lut_n = [0u8; 256];
    let mut lut_b = [0u32; 256];
    for v in 0..256u32 {
        let (token, extra_n, extra) = hybrid(v);
        let t = token as usize;
        if trivial {
            lut_n[v as usize] = extra_n as u8;
            lut_b[v as usize] = extra;
            continue;
        }
        let n = nbits.get(t).copied().unwrap_or(0) as u32;
        let code = bits.get(t).copied().unwrap_or(0) as u32;
        lut_n[v as usize] = (n + extra_n) as u8;
        lut_b[v as usize] = code | (extra << n);
    }
    (lut_n, lut_b)
}

impl PrefixCode {
    pub fn from_freqs(freqs: &[u64]) -> Self {
        let nbits = huffman_lengths(freqs, MAX_PREFIX_BITS);
        let bits = canonical_codes(&nbits);
        let trivial = nbits.iter().filter(|&&l| l > 0).count() <= 1;
        let (lut_n, lut_b) = hybrid_lut(&nbits, &bits, trivial, hybrid_uint_000);
        let (lut400_n, lut400_b) = hybrid_lut(&nbits, &bits, trivial, hybrid_uint_400);
        Self {
            nbits,
            bits,
            trivial,
            lut_n,
            lut_b,
            lut400_n,
            lut400_b,
        }
    }

    pub fn alphabet_size(&self) -> u32 {
        let mut n = self.nbits.len();
        while n > 1 && self.nbits[n - 1] == 0 {
            n -= 1;
        }
        n.max(1) as u32
    }

    /// Prefix token plus hybrid extra bits in one `put`.
    #[inline(always)]
    pub fn write_token(&self, w: &mut BitWriter, token: u32, extra_n: u32, extra: u32) {
        if self.trivial {
            if extra_n > 0 {
                w.put(extra_n, extra as u64);
            }
            return;
        }
        let t = token as usize;
        let n = self.nbits[t] as u32;
        debug_assert!(n > 0, "token {token} has no code");
        w.put(n + extra_n, self.bits[t] as u64 | ((extra as u64) << n));
    }

    #[inline(always)]
    pub fn write_hybrid000(&self, w: &mut BitWriter, value: u32) {
        if value < 256 {
            w.put(
                self.lut_n[value as usize] as u32,
                self.lut_b[value as usize] as u64,
            );
            return;
        }
        let (token, extra_n, extra) = hybrid_uint_000(value);
        self.write_token(w, token, extra_n, extra);
    }

    /// Hybrid uint 400 (values 0..=15 are raw tokens). Better for quantized residuals.
    #[inline(always)]
    pub fn write_hybrid400(&self, w: &mut BitWriter, value: u32) {
        if value < 256 {
            w.put(
                self.lut400_n[value as usize] as u32,
                self.lut400_b[value as usize] as u64,
            );
            return;
        }
        let (token, extra_n, extra) = hybrid_uint_400(value);
        self.write_token(w, token, extra_n, extra);
    }
}

/// IntegerConfig split_exp=4, msb=0, lsb=0 against prefix `log_alphabet_size = 15`.
pub fn write_integer_config_400(w: &mut BitWriter) {
    w.put(4, 4); // split_exponent
    w.put(3, 0); // msb_in_token, add_log2_ceil(4) = 3
    w.put(3, 0); // lsb_in_token
}

/// Write alphabet size as in `DecoderInner::parse` (prefix-code path).
pub fn write_alphabet_size(w: &mut BitWriter, size: u32) {
    assert!((1..=(1 << 15)).contains(&size));
    if size == 1 {
        w.put_bool(false);
        return;
    }
    for n in 0..16u32 {
        let base = 1 + (1u32 << n);
        if size >= base && (size - base) < (1u32 << n) {
            w.put_bool(true);
            w.put(4, n as u64);
            w.put(n, (size - base) as u64);
            return;
        }
    }
    // Round up to 1 + 2^n.
    let n = (size - 1).next_power_of_two().trailing_zeros();
    let base = 1 + (1u32 << n);
    w.put_bool(true);
    w.put(4, n as u64);
    w.put(n, (size.saturating_sub(base)) as u64);
}

/// Simple prefix histogram: 1..=4 symbols.
fn write_simple_histogram(w: &mut BitWriter, nbits: &[u8], alphabet: u32) {
    let symbols: Vec<usize> = nbits
        .iter()
        .enumerate()
        .filter(|(_, l)| **l > 0)
        .map(|(i, _)| i)
        .collect();
    let nsym = symbols.len();
    assert!((1..=4).contains(&nsym));

    w.put(2, 1); // HSKIP = 1 (simple)
    w.put(2, (nsym as u64) - 1);

    let alphabet_bits = (alphabet.next_power_of_two().trailing_zeros()).max(1);
    match nsym {
        1 => {
            w.put(alphabet_bits, symbols[0] as u64);
        }
        2 => {
            w.put(alphabet_bits, symbols[0] as u64);
            w.put(alphabet_bits, symbols[1] as u64);
        }
        3 => {
            w.put(alphabet_bits, symbols[0] as u64);
            w.put(alphabet_bits, symbols[1] as u64);
            w.put(alphabet_bits, symbols[2] as u64);
        }
        4 => {
            w.put(alphabet_bits, symbols[0] as u64);
            w.put(alphabet_bits, symbols[1] as u64);
            w.put(alphabet_bits, symbols[2] as u64);
            w.put(alphabet_bits, symbols[3] as u64);
            let lens: Vec<u8> = symbols.iter().map(|&s| nbits[s]).collect();
            let tree_selector = lens != [2, 2, 2, 2];
            w.put_bool(tree_selector);
            // tree_selector true → lengths 1,2,3,3; false → 2,2,2,2.
            // We only take this path when lengths match one of those.
        }
        _ => unreachable!(),
    }
}

/// Code-length-code: U32(0, 4, 3, 8) plus extra bits for 1/2/5.
fn write_clcl_length(w: &mut BitWriter, len: u8) {
    match len {
        0 => w.put(2, 0),
        1 => {
            w.put(2, 3);
            w.put(1, 1);
            w.put(1, 0);
        }
        2 => {
            w.put(2, 3);
            w.put(1, 0);
        }
        3 => w.put(2, 2),
        4 => w.put(2, 1),
        5 => {
            w.put(2, 3);
            w.put(1, 1);
            w.put(1, 1);
        }
        _ => panic!("code-length code length {len} out of 0..=5"),
    }
}

fn write_complex_histogram(w: &mut BitWriter, nbits: &[u8], alphabet: u32) {
    const ORDER: [usize; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    let mut cl_freqs = [0u64; 18];
    let n = (alphabet as usize).min(nbits.len());
    for &l in &nbits[..n] {
        cl_freqs[l as usize] += 1;
    }
    // Ensure the code-length Huffman is valid (at least two non-zero, or one).
    let cl_nbits = huffman_lengths(&cl_freqs, 5);
    // Force lengths into 0..=5 (already).

    w.put(2, 0); // HSKIP = 0

    // Write code-length-code lengths until Kraft of those lengths is 1.
    let mut bitacc = 0usize;
    let mut wrote = 0usize;
    for (k, &idx) in ORDER.iter().enumerate() {
        let len = cl_nbits[idx];
        write_clcl_length(w, len);
        wrote = k + 1;
        if len != 0 {
            bitacc += 32 >> len;
            if bitacc == 32 {
                break;
            }
        }
    }
    debug_assert!(bitacc == 32 || cl_freqs.iter().filter(|&&f| f > 0).count() <= 1);

    let cl_bits = canonical_codes(&cl_nbits);
    let cl_nonzero = cl_nbits.iter().filter(|&&l| l > 0).count();

    let write_len_sym = |w: &mut BitWriter, sym: u8| {
        if cl_nonzero == 1 {
            // Decoder uses a 0-bit single-symbol code-length alphabet.
            return;
        }
        let l = cl_nbits[sym as usize];
        debug_assert!(l > 0, "no code-length code for data length {sym}");
        w.put(l as u32, cl_bits[sym as usize] as u64);
    };

    let mut data_acc = 0usize;
    for &l in nbits.iter().take(n) {
        write_len_sym(w, l);
        if l != 0 {
            data_acc += 1 << (MAX_PREFIX_BITS as usize).saturating_sub(l as usize);
            if data_acc == (1 << MAX_PREFIX_BITS) {
                break;
            }
        }
    }
    let _ = (wrote, bitacc);
}

impl PrefixCode {
    pub fn write_histogram(&self, w: &mut BitWriter, alphabet: u32) {
        let n = alphabet as usize;
        let slice = if n <= self.nbits.len() {
            &self.nbits[..n]
        } else {
            &self.nbits
        };
        let nz: Vec<usize> = slice
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > 0)
            .map(|(i, _)| i)
            .collect();

        // Simple histograms assign lengths in *write order*, not by sorted length.
        let simple_ok = match nz.len() {
            1 => true,
            2 => self.nbits[nz[0]] == 1 && self.nbits[nz[1]] == 1,
            3 => self.nbits[nz[0]] == 1 && self.nbits[nz[1]] == 2 && self.nbits[nz[2]] == 2,
            4 => {
                let ls = [
                    self.nbits[nz[0]],
                    self.nbits[nz[1]],
                    self.nbits[nz[2]],
                    self.nbits[nz[3]],
                ];
                ls == [2, 2, 2, 2] || ls == [1, 2, 3, 3]
            }
            _ => false,
        };

        if simple_ok && nz.len() <= 4 {
            write_simple_histogram(w, slice, alphabet);
        } else {
            write_complex_histogram(w, slice, alphabet);
        }
    }
}

/// IntegerConfig 000 against a prefix-code alphabet (`log_alphabet_size = 15`).
pub fn write_integer_config_000(w: &mut BitWriter) {
    // split_exponent_bits = ceil(log2(15+1)) = 4
    w.put(4, 0);
    // split_exponent == 0 != 15, so msb/lsb follow with 0-bit fields.
}

/// HybridUint 400: `token = value` if `value < 16`, else `16 + floor(log2(value)) - 4`.
#[inline]
pub fn hybrid_uint_400(value: u32) -> (u32, u32, u32) {
    if value < 16 {
        (value, 0, 0)
    } else {
        let n = 31 - value.leading_zeros();
        (16 + n - 4, n, value - (1 << n))
    }
}

pub const LZ77_MIN_SYMBOL: u32 = 224;
pub const LZ77_MIN_LENGTH: u32 = 7;

/// All `num_dist` contexts share one prefix histogram (simple clustering → 0).
#[allow(dead_code)]
pub fn write_histograms_collapsed(w: &mut BitWriter, num_dist: u32, code: &PrefixCode) {
    w.put_bool(false); // no lz77
    if num_dist > 1 {
        w.put_bool(true); // simple clustering
        w.put(2, 0); // nbits = 0 → every context in cluster 0
    }
    w.put_bool(true); // prefix
    write_integer_config_400(w);
    let s = code.alphabet_size();
    write_alphabet_size(w, s);
    if s > 1 {
        code.write_histogram(w, s);
    }
}

/// Identity clustering, prefix codes, hybrid-uint 000. `dist` is the LZ77
/// distance histogram (enables LZ77). `codes` are the MA-leaf symbol codes.
pub fn write_histograms(w: &mut BitWriter, codes: &[PrefixCode], dist: Option<&PrefixCode>) {
    let n_leaves = codes.len() as u32;
    if dist.is_some() {
        w.put_bool(true); // lz77
        write_lz77_header(w);
    } else {
        w.put_bool(false);
    }

    let num = n_leaves + u32::from(dist.is_some());
    if num > 1 {
        let ids: Vec<u8> = (0..num).map(|i| i as u8).collect();
        write_cluster_map(w, &ids, num);
    }
    w.put_bool(true); // use_prefix_code
    for _ in 0..num {
        write_integer_config_400(w);
    }

    let mut all: Vec<&PrefixCode> = Vec::with_capacity(num as usize);
    all.extend(codes);
    if let Some(d) = dist {
        all.push(d);
    }
    let sizes: Vec<u32> = all.iter().map(|c| c.alphabet_size()).collect();
    for &s in &sizes {
        write_alphabet_size(w, s);
    }
    for (code, &s) in all.iter().zip(&sizes) {
        if s > 1 {
            code.write_histogram(w, s);
        }
    }
}

/// Map `cluster_of.len()` MA-leaf contexts onto `codes` (dense cluster ids).
/// LZ77 distance is an extra context in its own last cluster.
#[allow(dead_code)]
pub fn write_histograms_clustered(
    w: &mut BitWriter,
    cluster_of: &[u8],
    codes: &[PrefixCode],
    dist: Option<&PrefixCode>,
) {
    debug_assert!(!cluster_of.is_empty() && !codes.is_empty());
    if dist.is_some() {
        w.put_bool(true);
        write_lz77_header(w);
    } else {
        w.put_bool(false);
    }

    let n_clusters = codes.len() as u32 + u32::from(dist.is_some());
    let dist_cluster = codes.len() as u32;
    let mut ids = cluster_of.to_vec();
    if dist.is_some() {
        ids.push(dist_cluster as u8);
    }
    if ids.len() > 1 {
        write_cluster_map(w, &ids, n_clusters);
    }
    w.put_bool(true);
    for _ in 0..n_clusters {
        write_integer_config_400(w);
    }

    let mut all: Vec<&PrefixCode> = Vec::with_capacity(n_clusters as usize);
    all.extend(codes);
    if let Some(d) = dist {
        all.push(d);
    }
    let sizes: Vec<u32> = all.iter().map(|c| c.alphabet_size()).collect();
    for &s in &sizes {
        write_alphabet_size(w, s);
    }
    for (code, &s) in all.iter().zip(&sizes) {
        if s > 1 {
            code.write_histogram(w, s);
        }
    }
}

pub(crate) fn write_lz77_header(w: &mut BitWriter) {
    write_lz77_header_len(w, LZ77_MIN_LENGTH);
}

pub(crate) fn write_lz77_header_len(w: &mut BitWriter, min_length: u32) {
    write_u32(
        w,
        LZ77_MIN_SYMBOL,
        U32::Const(224),
        U32::Const(512),
        U32::Const(4096),
        off(8, 15),
    );
    write_u32(
        w,
        min_length,
        U32::Const(3),
        U32::Const(4),
        off(5, 2),
        off(9, 8),
    );
    w.put(4, 4);
    w.put(3, 0);
    w.put(3, 0);
}

/// Cluster map: simple (≤ 8 clusters) or a nested prefix code (ISO 18181).
///
/// Maps with more than 8 clusters use MTF. HF's default 15-cluster layout is
/// long runs of one id (coeff contexts) plus a 0..14 cycle (nnz contexts);
/// MTF turns the runs into zeros.
pub(crate) fn write_cluster_map(w: &mut BitWriter, ids: &[u8], n_clusters: u32) {
    debug_assert!(ids.len() > 1);
    debug_assert!((1..=256).contains(&n_clusters));
    if n_clusters <= 8 {
        w.put_bool(true); // simple
        let nbits = if n_clusters <= 1 {
            0
        } else {
            (n_clusters - 1).ilog2() + 1
        };
        w.put(2, nbits as u64);
        for &c in ids {
            debug_assert!((c as u32) < n_clusters);
            w.put(nbits, c as u64);
        }
        return;
    }
    w.put_bool(false); // not simple
    w.put_bool(true); // MTF
    w.put_bool(false); // nested decoder: no LZ77
    w.put_bool(true); // prefix
    write_integer_config_000(w);
    let encoded = mtf_encode(ids);
    let max_id = encoded.iter().copied().max().unwrap_or(0) as u32;
    let (max_tok, _, _) = hybrid_uint_000(max_id);
    let mut tf = vec![0u64; max_tok as usize + 1];
    for &id in &encoded {
        let (t, _, _) = hybrid_uint_000(id as u32);
        tf[t as usize] += 1;
    }
    let code = PrefixCode::from_freqs(&tf);
    let alph = code.alphabet_size();
    write_alphabet_size(w, alph);
    if alph > 1 {
        code.write_histogram(w, alph);
    }
    for &id in &encoded {
        code.write_hybrid000(w, id as u32);
    }
}

fn mtf_encode(ids: &[u8]) -> Vec<u8> {
    let mut table: [u8; 256] = core::array::from_fn(|i| i as u8);
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        let pos = table.iter().position(|&x| x == id).unwrap();
        out.push(pos as u8);
        let v = table[pos];
        table.copy_within(0..pos, 1);
        table[0] = v;
    }
    out
}

/// Distance histogram that always emits token 1 (RLE distance).
#[allow(dead_code)]
pub fn rle_distance_code() -> PrefixCode {
    let mut freqs = vec![0u64; 2];
    freqs[1] = 1;
    PrefixCode::from_freqs(&freqs)
}

/// Visit each emitted entropy token (raw hybrid-000 token, or LZ77 token).
pub fn for_each_token(residuals: &[u32], lz77: bool, emit: impl FnMut(u32, u32, u32)) {
    for_each_token_len(residuals, lz77, LZ77_MIN_LENGTH, emit);
}

pub fn for_each_token_len(
    residuals: &[u32],
    lz77: bool,
    min_length: u32,
    mut emit: impl FnMut(u32, u32, u32),
) {
    let mut i = 0;
    while i < residuals.len() {
        if lz77 && residuals[i] == 0 {
            let mut k = 1;
            while i + k < residuals.len() && residuals[i + k] == 0 {
                k += 1;
            }
            if k > min_length as usize {
                let (t, n, b) = hybrid_uint_400(0);
                emit(t, n, b);
                let extra = (k as u32 - 1) - min_length;
                let (t, n, b) = hybrid_uint_400(extra);
                emit(LZ77_MIN_SYMBOL + t, n, b);
                i += k;
                continue;
            }
            for _ in 0..k {
                let (t, n, b) = hybrid_uint_400(0);
                emit(t, n, b);
            }
            i += k;
            continue;
        }
        let (t, n, b) = hybrid_uint_400(residuals[i]);
        emit(t, n, b);
        i += 1;
    }
}

/// Like [`for_each_token`], but each emit also gets the MA cluster of that
/// token. LZ77 *length* is attributed to the second pixel of the run (the
/// decoder reads it from that pixel's cluster).
pub fn for_each_token_ctx(
    residuals: &[u32],
    ctxs: &[u8],
    lz77: bool,
    emit: impl FnMut(u8, u32, u32, u32),
) {
    for_each_token_ctx_len(residuals, ctxs, lz77, LZ77_MIN_LENGTH, emit);
}

pub fn for_each_token_ctx_len(
    residuals: &[u32],
    ctxs: &[u8],
    lz77: bool,
    min_length: u32,
    mut emit: impl FnMut(u8, u32, u32, u32),
) {
    debug_assert_eq!(residuals.len(), ctxs.len());
    let mut i = 0;
    while i < residuals.len() {
        let c0 = ctxs[i];
        if lz77 && residuals[i] == 0 {
            let mut k = 1;
            while i + k < residuals.len() && residuals[i + k] == 0 {
                k += 1;
            }
            if k > min_length as usize {
                let (t, n, b) = hybrid_uint_400(0);
                emit(c0, t, n, b);
                let extra = (k as u32 - 1) - min_length;
                let (t, n, b) = hybrid_uint_400(extra);
                emit(ctxs[i + 1], LZ77_MIN_SYMBOL + t, n, b);
                i += k;
                continue;
            }
            for j in 0..k {
                let (t, n, b) = hybrid_uint_400(0);
                emit(ctxs[i + j], t, n, b);
            }
            i += k;
            continue;
        }
        let (t, n, b) = hybrid_uint_400(residuals[i]);
        emit(c0, t, n, b);
        i += 1;
    }
}

/// Write packed residuals, using LZ77 RLE for zero runs of length ≥ `min_length + 1`.
pub fn write_residual_run(w: &mut BitWriter, code: &PrefixCode, residuals: &[u32], lz77: bool) {
    if !lz77 {
        for &v in residuals {
            code.write_hybrid400(w, v);
        }
        return;
    }
    for_each_token(residuals, true, |token, extra_n, extra| {
        code.write_token(w, token, extra_n, extra);
    });
}

/// One MA-tree leaf: predictor + residual multiplier `(mul_bits+1)<<mul_log`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaLeaf {
    /// 0 = Zero, 5 = Gradient.
    pub predictor: u32,
    pub mul_log: u32,
    pub mul_bits: u32,
    /// Extra split on this Zero-predictor leaf (`0` = none).
    ///
    /// JPEG XL property index: `3` = `x > 0`, `4` = `|N| > 0`, `5` = `|W| > 0`.
    /// Written as `property + 1` in the tree uint (0 means leaf).
    pub split_prop: u8,
    pub split_val: i32,
    /// Nested split on the `|W|×|N|` 4-way (0 = none). Compact tree: property ≤ 6.
    pub nest_prop: u8,
    pub nest_val: i32,
}

impl MaLeaf {
    pub fn gradient(mul_log: u32, mul_bits: u32) -> Self {
        Self {
            predictor: 5,
            mul_log,
            mul_bits,
            split_prop: 0,
            split_val: 0,
            nest_prop: 0,
            nest_val: 0,
        }
    }

    pub fn zero(mul_log: u32, mul_bits: u32) -> Self {
        Self {
            predictor: 0,
            mul_log,
            mul_bits,
            split_prop: 0,
            split_val: 0,
            nest_prop: 0,
            nest_val: 0,
        }
    }

    /// Squeeze HF: split on property 5 (`|W| > 0`).
    #[allow(dead_code)]
    pub fn zero_abs_n(mul_log: u32, mul_bits: u32) -> Self {
        Self::zero_split(mul_log, mul_bits, 5)
    }

    pub fn zero_split(mul_log: u32, mul_bits: u32, split_prop: u8) -> Self {
        Self::zero_split_2(mul_log, mul_bits, split_prop, 0, 4, 0)
    }

    pub fn zero_split_2(
        mul_log: u32,
        mul_bits: u32,
        split_prop: u8,
        split_val: i32,
        nest_prop: u8,
        nest_val: i32,
    ) -> Self {
        Self {
            predictor: 0,
            mul_log,
            mul_bits,
            split_prop,
            split_val,
            nest_prop,
            nest_val,
        }
    }

    #[allow(dead_code)]
    pub fn ctx_span(self) -> usize {
        if self.nest_prop > 0 {
            4
        } else if self.split_prop > 0 {
            2
        } else {
            1
        }
    }
}

/// Shape of a small MA tree split on channel index (property 0).
///
/// Leaf order is **preorder** (left / `prop > value` first) and must match the
/// histogram context order.
#[derive(Clone, Debug)]
pub enum MaShape {
    /// 1..=4 Gradient leaves, same multiplier. Lossless / VarDCT LF.
    PerChannel { n: u32, mul_log: u32, mul_bits: u32 },
    /// Per-stream-local channel leaves.
    ///
    /// `gmod` is stream 0. `lf` is modular LF groups (`0 < stream_idx <= stream_cut`).
    /// `pass` is pass groups (`stream_idx > stream_cut`, or `stream_idx > 0` if `lf`
    /// is empty). Empty `lf` and `pass` → one chain (local index = global).
    Leaves {
        gmod: Vec<MaLeaf>,
        lf: Vec<MaLeaf>,
        pass: Vec<MaLeaf>,
        /// Property-1 split between LF and pass streams (`2 * num_lf_groups`).
        stream_cut: u32,
        /// Wrap the chain in `property > 0` (JPEG XL property index; `0` = none).
        /// Left subtree is copied as the right subtree so flatten-after-prune
        /// leaves a 2-node remaining tree per channel (not a mid-spine insert).
        prop_split: u8,
        /// Learned residual tree cloned onto each HF Zero leaf. `None` = the
        /// fixed `|W|` then `|N|` 4-way.
        hf: Option<ResTree>,
    },
}

/// Residual MA subtree (no channel/stream). Leaf `id` is the shared cluster slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResTree {
    pub nodes: Vec<ResNode>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResNode {
    Split {
        prop: u8,
        val: i32,
        left: u8,
        right: u8,
    },
    Leaf {
        id: u8,
    },
}

impl ResTree {
    pub fn n_leaves(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, ResNode::Leaf { .. }))
            .count()
    }

    pub fn max_u(&self) -> u32 {
        let mut m = 0u32;
        for n in &self.nodes {
            if let ResNode::Split { prop, val, .. } = *n {
                m = m.max(u32::from(prop) + 1);
                m = m.max(pack_signed(val));
            }
        }
        m
    }

    /// Dense leaf ids `0..n-1` in this tree's BFS order.
    pub fn compact_ids(&mut self) {
        let mut q = std::collections::VecDeque::new();
        q.push_back(0u8);
        let mut id = 0u8;
        let mut seen = vec![false; self.nodes.len()];
        while let Some(i) = q.pop_front() {
            let i = i as usize;
            if i >= self.nodes.len() || seen[i] {
                continue;
            }
            seen[i] = true;
            match self.nodes[i] {
                ResNode::Leaf { .. } => {
                    self.nodes[i] = ResNode::Leaf { id };
                    id = id.saturating_add(1);
                }
                ResNode::Split { left, right, .. } => {
                    q.push_back(left);
                    q.push_back(right);
                }
            }
        }
    }

    /// 8 compact leaves: 6-way signed-N, then `|W|>1` on `|W|>0 ∧ |N|=0`
    /// and `x>0` on both-zero.
    #[allow(dead_code)]
    pub fn wn_n_sign_w1_x() -> Self {
        Self {
            nodes: vec![
                ResNode::Split {
                    prop: 5,
                    val: 0,
                    left: 1,
                    right: 2,
                },
                ResNode::Split {
                    prop: 4,
                    val: 0,
                    left: 3,
                    right: 4,
                },
                ResNode::Split {
                    prop: 4,
                    val: 0,
                    left: 5,
                    right: 6,
                },
                ResNode::Split {
                    prop: 6,
                    val: 0,
                    left: 7,
                    right: 8,
                },
                ResNode::Split {
                    prop: 5,
                    val: 1,
                    left: 11,
                    right: 12,
                },
                ResNode::Split {
                    prop: 6,
                    val: 0,
                    left: 9,
                    right: 10,
                },
                ResNode::Split {
                    prop: 3,
                    val: 0,
                    left: 13,
                    right: 14,
                },
                ResNode::Leaf { id: 0 },
                ResNode::Leaf { id: 1 },
                ResNode::Leaf { id: 3 },
                ResNode::Leaf { id: 4 },
                ResNode::Leaf { id: 2 },
                ResNode::Leaf { id: 6 },
                ResNode::Leaf { id: 5 },
                ResNode::Leaf { id: 7 },
            ],
        }
    }

    /// `|W|>0` then `|N|>0` then `N>0` on the `|N|>0` branches (6 leaves).
    pub fn wn_then_n_sign() -> Self {
        Self {
            nodes: vec![
                ResNode::Split {
                    prop: 5,
                    val: 0,
                    left: 1,
                    right: 2,
                },
                ResNode::Split {
                    prop: 4,
                    val: 0,
                    left: 3,
                    right: 4,
                },
                ResNode::Split {
                    prop: 4,
                    val: 0,
                    left: 5,
                    right: 6,
                },
                ResNode::Split {
                    prop: 6,
                    val: 0,
                    left: 7,
                    right: 8,
                },
                ResNode::Leaf { id: 2 },
                ResNode::Split {
                    prop: 6,
                    val: 0,
                    left: 9,
                    right: 10,
                },
                ResNode::Leaf { id: 5 },
                ResNode::Leaf { id: 0 },
                ResNode::Leaf { id: 1 },
                ResNode::Leaf { id: 3 },
                ResNode::Leaf { id: 4 },
            ],
        }
    }

    pub fn default_wn() -> Self {
        Self {
            nodes: vec![
                ResNode::Split {
                    prop: 5,
                    val: 0,
                    left: 1,
                    right: 2,
                },
                ResNode::Split {
                    prop: 4,
                    val: 0,
                    left: 3,
                    right: 4,
                },
                ResNode::Split {
                    prop: 4,
                    val: 0,
                    left: 5,
                    right: 6,
                },
                ResNode::Leaf { id: 0 },
                ResNode::Leaf { id: 1 },
                ResNode::Leaf { id: 2 },
                ResNode::Leaf { id: 3 },
            ],
        }
    }
}

/// Which chain a global channel belongs to in [`MaShape::Leaves`].
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum MaChain {
    Gmod,
    Lf,
    Pass,
}

/// Preorder of non-empty (chain, length) pairs. Must match [`write_ma_tree`].
#[allow(dead_code)]
pub fn ma_tree_preorder(n_gmod: usize, n_lf: usize, n_pass: usize) -> Vec<(MaChain, usize)> {
    let g = n_gmod > 0;
    let l = n_lf > 0;
    let p = n_pass > 0;
    match (g, l, p) {
        (true, false, false) => vec![(MaChain::Gmod, n_gmod)],
        (false, true, false) => vec![(MaChain::Lf, n_lf)],
        (false, false, true) => vec![(MaChain::Pass, n_pass)],
        (true, true, false) => vec![(MaChain::Lf, n_lf), (MaChain::Gmod, n_gmod)],
        (true, false, true) => vec![(MaChain::Pass, n_pass), (MaChain::Gmod, n_gmod)],
        (false, true, true) => vec![(MaChain::Pass, n_pass), (MaChain::Lf, n_lf)],
        (true, true, true) => vec![
            (MaChain::Pass, n_pass),
            (MaChain::Lf, n_lf),
            (MaChain::Gmod, n_gmod),
        ],
        (false, false, false) => Vec::new(),
    }
}

/// Histogram context of local channel `local` in a right-spine chain of `len`
/// starting at `ctx_base` (preorder: highest local index first).
#[inline]
#[allow(dead_code)]
pub fn ma_chain_leaf_ctx(ctx_base: usize, len: usize, local: usize) -> usize {
    debug_assert!(local < len);
    ctx_base + len - 1 - local
}

/// MA tree: one Gradient leaf per channel, split on channel index (property 0).
///
/// Tree entropy uses a 4-symbol equal-length simple prefix + hybrid uint 000,
/// covering integer values 0..=7.
pub fn write_ma_tree_gradient(w: &mut BitWriter, n_channels: u32, mul_log: u32, mul_bits: u32) {
    write_ma_tree(
        w,
        &MaShape::PerChannel {
            n: n_channels,
            mul_log,
            mul_bits,
        },
    );
}

pub fn write_ma_tree(w: &mut BitWriter, shape: &MaShape) {
    if let MaShape::Leaves {
        gmod,
        lf,
        pass,
        stream_cut,
        prop_split,
        hf,
    } = shape
    {
        write_ma_tree_leaves(w, gmod, lf, pass, *stream_cut, *prop_split, hf.as_ref());
        return;
    }

    write_tree_header_compact(w);
    match shape {
        MaShape::PerChannel {
            n,
            mul_log,
            mul_bits,
        } => {
            assert!((1..=4).contains(n));
            let leaf = MaLeaf::gradient(*mul_log, *mul_bits);
            match *n {
                1 => ma_leaf(w, leaf),
                2 => {
                    ma_decision(w, 0);
                    ma_leaf(w, leaf);
                    ma_leaf(w, leaf);
                }
                3 => {
                    ma_decision(w, 1);
                    ma_leaf(w, leaf);
                    ma_decision(w, 0);
                    ma_leaf(w, leaf);
                    ma_leaf(w, leaf);
                }
                4 => {
                    ma_decision(w, 1);
                    ma_decision(w, 2);
                    ma_leaf(w, leaf);
                    ma_leaf(w, leaf);
                    ma_decision(w, 0);
                    ma_leaf(w, leaf);
                    ma_leaf(w, leaf);
                }
                _ => unreachable!(),
            }
        }
        MaShape::Leaves { .. } => unreachable!(),
    }
}

fn write_tree_header_compact(w: &mut BitWriter) {
    w.put_bool(false); // no lz77
    w.put_bool(true); // simple cluster map
    w.put(2, 0); // 0 bits → every context in cluster 0
    w.put_bool(true); // prefix
    write_integer_config_000(w);
    w.put(6, 0b100011); // alphabet size 4
    w.put(2, 1); // HSKIP simple
    w.put(2, 3); // nsym = 4
    w.put(2, 0);
    w.put(2, 1);
    w.put(2, 2);
    w.put(2, 3);
    w.put_bool(false); // lengths 2,2,2,2
}

/// Right-spine splits on local channel: `ch > n-2` → leaf n-1, else …
///
/// When more than one stream kind has channels, split on property 1
/// (`stream_idx`) first so pass/LF local index 0 is not confused with Y LF.
fn write_ma_tree_leaves(
    w: &mut BitWriter,
    gmod: &[MaLeaf],
    lf: &[MaLeaf],
    pass: &[MaLeaf],
    stream_cut: u32,
    _prop_split: u8,
    hf: Option<&ResTree>,
) {
    assert!(!gmod.is_empty() || !lf.is_empty() || !pass.is_empty());
    let mut max_u = 0u32;
    for chain in [gmod, lf, pass] {
        if chain.len() >= 2 {
            max_u = max_u.max(pack_signed((chain.len() - 2) as i32));
        }
    }
    let need_cut = !lf.is_empty() && !pass.is_empty();
    if need_cut {
        max_u = max_u.max(pack_signed(stream_cut as i32));
    }
    if _prop_split > 0 {
        max_u = max_u.max(u32::from(_prop_split) + 1);
    }
    for chain in [gmod, lf, pass] {
        for l in chain {
            max_u = max_u.max(l.predictor).max(l.mul_log).max(l.mul_bits);
            if l.split_prop > 0 {
                max_u = max_u.max(u32::from(l.split_prop) + 1);
                max_u = max_u.max(pack_signed(l.split_val));
            }
            if l.nest_prop > 0 {
                max_u = max_u.max(u32::from(l.nest_prop) + 1);
                max_u = max_u.max(pack_signed(l.nest_val));
            }
        }
    }
    if let Some(t) = hf {
        max_u = max_u.max(t.max_u());
    }
    let wide = max_u > 7;
    let code = if wide {
        Some(write_tree_header_wide(w, alphabet_for_tree(max_u)))
    } else {
        write_tree_header_compact(w);
        None
    };
    let code = code.as_ref();
    write_enc_ma(w, build_stream_tree(gmod, lf, pass, stream_cut, hf), code);
}

fn alphabet_for_tree(max_u: u32) -> u32 {
    let (tok, _, _) = hybrid_uint_000(max_u);
    (tok + 1).max(8).next_power_of_two().min(32)
}

fn uint_tree(w: &mut BitWriter, wide: Option<&PrefixCode>, v: u32) {
    if let Some(code) = wide {
        code.write_hybrid000(w, v);
    } else {
        ma_tree_uint(w, v);
    }
}

fn write_tree_header_wide(w: &mut BitWriter, alphabet: u32) -> PrefixCode {
    w.put_bool(false);
    w.put_bool(true);
    w.put(2, 0);
    w.put_bool(true);
    write_integer_config_000(w);
    write_alphabet_size(w, alphabet);
    // Mixed freqs so the code-length alphabet has ≥2 symbols (the all-equal
    // length-3 histogram hits a single-symbol CL path that we do not trust).
    let n = alphabet as usize;
    let mut freqs = vec![0u64; n];
    for (i, f) in freqs.iter_mut().enumerate() {
        *f = 1u64 << 6u32.saturating_sub(i as u32).min(6);
    }
    if n >= 2 {
        freqs[n - 1] = freqs[n - 1].max(1);
        if freqs[n - 1] == freqs[0] {
            freqs[0] = freqs[0].saturating_add(1);
        }
    }
    let code = PrefixCode::from_freqs(&freqs);
    debug_assert_eq!(code.alphabet_size(), alphabet);
    debug_assert!(code.nbits.iter().take(n).all(|&l| l > 0));
    code.write_histogram(w, alphabet);
    code
}

/// MA tree node. Serialized in BFS (libjxl `to_decode` queue), not DFS.
#[derive(Clone, Debug)]
enum EncMa {
    D {
        prop: u32,
        val: i32,
        l: Box<EncMa>,
        r: Box<EncMa>,
    },
    L {
        leaf: MaLeaf,
        chain: u8,
        ch: usize,
        /// `|W|` then `|N|`: 0 = both >0, 1 = `|N|=0`, 2 = `|W|=0`, 3 = both 0.
        quad: u8,
    },
}

fn two_prop_quads(chain: u8, ch: usize, l: MaLeaf) -> EncMa {
    let leaf = |quad: u8| {
        Box::new(EncMa::L {
            leaf: l,
            chain,
            ch,
            quad,
        })
    };
    EncMa::D {
        prop: u32::from(l.split_prop),
        val: l.split_val,
        l: Box::new(EncMa::D {
            prop: u32::from(l.nest_prop),
            val: l.nest_val,
            l: leaf(0),
            r: leaf(1),
        }),
        r: Box::new(EncMa::D {
            prop: u32::from(l.nest_prop),
            val: l.nest_val,
            l: leaf(2),
            r: leaf(3),
        }),
    }
}

fn instantiate_res(tree: &ResTree, chain: u8, ch: usize, l: MaLeaf) -> EncMa {
    fn rec(tree: &ResTree, idx: usize, chain: u8, ch: usize, l: MaLeaf) -> EncMa {
        match tree.nodes[idx] {
            ResNode::Leaf { id } => EncMa::L {
                leaf: l,
                chain,
                ch,
                quad: id,
            },
            ResNode::Split {
                prop,
                val,
                left,
                right,
            } => EncMa::D {
                prop: u32::from(prop),
                val,
                l: Box::new(rec(tree, left as usize, chain, ch, l)),
                r: Box::new(rec(tree, right as usize, chain, ch, l)),
            },
        }
    }
    rec(tree, 0, chain, ch, l)
}

fn wrap_leaf(chain: u8, ch: usize, l: MaLeaf, hf: Option<&ResTree>) -> EncMa {
    if l.split_prop > 0
        && let Some(t) = hf
    {
        return instantiate_res(t, chain, ch, l);
    }
    if l.nest_prop > 0 && l.split_prop > 0 {
        two_prop_quads(chain, ch, l)
    } else if l.split_prop > 0 {
        EncMa::D {
            prop: u32::from(l.split_prop),
            val: l.split_val,
            l: Box::new(EncMa::L {
                leaf: l,
                chain,
                ch,
                quad: 0,
            }),
            r: Box::new(EncMa::L {
                leaf: l,
                chain,
                ch,
                quad: 1,
            }),
        }
    } else {
        EncMa::L {
            leaf: l,
            chain,
            ch,
            quad: 0,
        }
    }
}

/// Right-spine on property 0: `ch > i-1` → leaf `i`, else continue.
fn build_spine(leaves: &[MaLeaf], chain: u8, hf: Option<&ResTree>) -> EncMa {
    let mut node = wrap_leaf(chain, 0, leaves[0], hf);
    for (rest, &leaf) in leaves.iter().enumerate().skip(1) {
        node = EncMa::D {
            prop: 0,
            val: (rest - 1) as i32,
            l: Box::new(wrap_leaf(chain, rest, leaf, hf)),
            r: Box::new(node),
        };
    }
    node
}

/// Stream-idx splits then per-kind channel spines. Serialized in BFS.
fn build_stream_tree(
    gmod: &[MaLeaf],
    lf: &[MaLeaf],
    pass: &[MaLeaf],
    stream_cut: u32,
    hf: Option<&ResTree>,
) -> EncMa {
    let g = (!gmod.is_empty()).then(|| build_spine(gmod, 0, hf));
    let l = (!lf.is_empty()).then(|| build_spine(lf, 1, hf));
    let p = (!pass.is_empty()).then(|| build_spine(pass, 2, hf));
    match (g, l, p) {
        (Some(g), None, None) => g,
        (None, Some(l), None) => l,
        (None, None, Some(p)) => p,
        (Some(g), None, Some(p)) => EncMa::D {
            prop: 1,
            val: 0,
            l: Box::new(p),
            r: Box::new(g),
        },
        (None, Some(l), Some(p)) => EncMa::D {
            prop: 1,
            val: stream_cut as i32,
            l: Box::new(p),
            r: Box::new(l),
        },
        (Some(g), Some(l), None) => EncMa::D {
            prop: 1,
            val: 0,
            l: Box::new(l),
            r: Box::new(g),
        },
        (Some(g), Some(l), Some(p)) => EncMa::D {
            prop: 1,
            val: 0,
            l: Box::new(EncMa::D {
                prop: 1,
                val: stream_cut as i32,
                l: Box::new(p),
                r: Box::new(l),
            }),
            r: Box::new(g),
        },
        (None, None, None) => unreachable!("empty MA tree"),
    }
}

/// BFS leaf contexts: four `|W|×|N|` slots per local channel.
///
/// Index `(!|W|>0)*2 + (!|N|>0)`. Unsplit fills every slot with the same ctx.
#[allow(dead_code)]
pub fn ma_bfs_leaf_ctxs(leaves: &[MaLeaf]) -> Vec<[u8; 8]> {
    ma_bfs_leaf_ctxs_hf(leaves, None)
}

pub fn ma_bfs_leaf_ctxs_hf(leaves: &[MaLeaf], hf: Option<&ResTree>) -> Vec<[u8; 8]> {
    let mut out = vec![[0u8; 8]; leaves.len()];
    let mut q = std::collections::VecDeque::new();
    q.push_back(build_spine(leaves, 0, hf));
    let mut ctx = 0u8;
    while let Some(n) = q.pop_front() {
        match n {
            EncMa::D { l, r, .. } => {
                q.push_back(*l);
                q.push_back(*r);
            }
            EncMa::L { ch, quad, .. } => {
                let q = quad.min(7) as usize;
                out[ch][q] = ctx;
                if leaves[ch].split_prop == 0 {
                    out[ch] = [ctx; 8];
                }
                ctx = ctx.saturating_add(1);
            }
        }
    }
    out
}

/// BFS leaf contexts for a global tree with stream-idx splits.
///
/// Chain 0 = gmodular, 1 = LF groups, 2 = pass groups. Local channel index
/// inside each chain matches the right-spine (same as [`ma_bfs_leaf_ctxs`]).
pub fn ma_stream_leaf_count(
    gmod: &[MaLeaf],
    lf: &[MaLeaf],
    pass: &[MaLeaf],
    stream_cut: u32,
    hf: Option<&ResTree>,
) -> usize {
    fn count(n: &EncMa) -> usize {
        match n {
            EncMa::L { .. } => 1,
            EncMa::D { l, r, .. } => count(l) + count(r),
        }
    }
    count(&build_stream_tree(gmod, lf, pass, stream_cut, hf))
}

pub fn ma_stream_ctxs(
    gmod: &[MaLeaf],
    lf: &[MaLeaf],
    pass: &[MaLeaf],
    stream_cut: u32,
    hf: Option<&ResTree>,
) -> [Vec<[u8; 8]>; 3] {
    let mut maps = [
        vec![[0u8; 8]; gmod.len()],
        vec![[0u8; 8]; lf.len()],
        vec![[0u8; 8]; pass.len()],
    ];
    let chains = [gmod, lf, pass];
    let mut q = std::collections::VecDeque::new();
    q.push_back(build_stream_tree(gmod, lf, pass, stream_cut, hf));
    let mut ctx = 0u8;
    while let Some(n) = q.pop_front() {
        match n {
            EncMa::D { l, r, .. } => {
                q.push_back(*l);
                q.push_back(*r);
            }
            EncMa::L {
                chain, ch, quad, ..
            } => {
                let chain = chain.min(2) as usize;
                let q = quad.min(7) as usize;
                if ch < maps[chain].len() {
                    maps[chain][ch][q] = ctx;
                    if chains[chain].get(ch).is_some_and(|l| l.split_prop == 0) {
                        maps[chain][ch] = [ctx; 8];
                    }
                }
                ctx = ctx.saturating_add(1);
            }
        }
    }
    maps
}

fn write_enc_ma(w: &mut BitWriter, root: EncMa, wide: Option<&PrefixCode>) {
    fn uint(w: &mut BitWriter, wide: Option<&PrefixCode>, v: u32) {
        uint_tree(w, wide, v);
    }
    fn write_leaf(w: &mut BitWriter, wide: Option<&PrefixCode>, l: MaLeaf) {
        uint(w, wide, 0);
        uint(w, wide, l.predictor);
        uint(w, wide, 0);
        uint(w, wide, l.mul_log);
        uint(w, wide, l.mul_bits);
    }
    let mut q = std::collections::VecDeque::new();
    q.push_back(root);
    while let Some(n) = q.pop_front() {
        match n {
            EncMa::D { prop, val, l, r } => {
                uint(w, wide, prop + 1);
                uint(w, wide, pack_signed(val));
                q.push_back(*l);
                q.push_back(*r);
            }
            EncMa::L { leaf, .. } => write_leaf(w, wide, leaf),
        }
    }
}

fn ma_tree_uint(w: &mut BitWriter, value: u32) {
    match value {
        0 => w.put(2, 0b00),
        1 => w.put(2, 0b10),
        2 => w.put(3, 0b001),
        3 => w.put(3, 0b101),
        4 => w.put(4, 0b0011),
        5 => w.put(4, 0b0111),
        6 => w.put(4, 0b1011),
        7 => w.put(4, 0b1111),
        _ => panic!("MA tree uint {value} > 7"),
    }
}

fn ma_decision(w: &mut BitWriter, value: i32) {
    ma_tree_uint(w, 1); // property 0 (channel) encoded as 1
    ma_tree_uint(w, pack_signed(value));
}

fn ma_leaf(w: &mut BitWriter, leaf: MaLeaf) {
    assert!(leaf.predictor <= 7 && leaf.mul_log <= 7 && leaf.mul_bits <= 7);
    ma_tree_uint(w, 0);
    ma_tree_uint(w, leaf.predictor);
    ma_tree_uint(w, 0);
    ma_tree_uint(w, leaf.mul_log);
    ma_tree_uint(w, leaf.mul_bits);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_signed_roundtrip_table() {
        fn unpack(x: u32) -> i32 {
            let s = -((x & 1) as i32);
            ((x >> 1) as i32) ^ s
        }
        for v in -32..32 {
            assert_eq!(unpack(pack_signed(v)), v);
        }
    }

    #[test]
    fn bfs_leaf_ctxs_split_on_channel_zero() {
        let split = MaLeaf::zero_split(0, 0, 5);
        let plain = MaLeaf::zero(0, 0);
        // D ch>0; left = ch1 leaf; right = |W| then |N| on ch0.
        let ctx = ma_bfs_leaf_ctxs(&[split, plain]);
        assert_eq!(ctx[1], [0; 8]);
        assert_eq!(&ctx[0][..4], &[1, 2, 3, 4]);
        assert_eq!(split.ctx_span(), 4);
        assert_eq!(plain.ctx_span(), 1);
    }

    #[test]
    fn stream_tree_pass_then_gmod_bfs() {
        let pass = MaLeaf::zero_split(0, 0, 5);
        let gmod = MaLeaf::zero(0, 0);
        let [g, l, p] = ma_stream_ctxs(&[gmod], &[], &[pass], 0, None);
        assert!(l.is_empty());
        // D stream>0; left = |W|×|N| on pass; right = gmod leaf. BFS hits gmod
        // after queuing the two |N| nodes, so gmod is ctx 0.
        assert_eq!(g[0], [0; 8]);
        assert!(p[0][0] != p[0][3]);
        let mut seen = p[0].to_vec();
        seen.push(0);
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn stream_tree_six_leaf_res_fills_all_slots() {
        let pass = MaLeaf::zero_split_2(0, 0, 5, 0, 4, 0);
        let gmod = MaLeaf::zero(0, 0);
        let hf = ResTree::wn_then_n_sign();
        let [g, l, p] = ma_stream_ctxs(&[gmod], &[], &[pass], 0, Some(&hf));
        assert!(l.is_empty());
        assert_eq!(g[0], [0; 8]);
        let slots = &p[0][..6];
        let mut seen = slots.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(
            seen.len(),
            6,
            "6 residual leaves need 6 ctx slots, got {slots:?}"
        );
        assert!(
            !slots.contains(&0),
            "gmod is ctx 0; HF slots must not reuse it: {slots:?}"
        );
        let n = ma_stream_leaf_count(&[gmod], &[], &[pass], 0, Some(&hf));
        assert_eq!(n, 7);
    }

    #[test]
    fn stream_tree_eight_leaf_res_fills_all_slots() {
        let pass = MaLeaf::zero_split_2(0, 0, 5, 0, 4, 0);
        let gmod = MaLeaf::zero(0, 0);
        let hf = ResTree::wn_n_sign_w1_x();
        assert_eq!(hf.n_leaves(), 8);
        let [g, l, p] = ma_stream_ctxs(&[gmod], &[], &[pass], 0, Some(&hf));
        assert!(l.is_empty());
        assert_eq!(g[0], [0; 8]);
        let slots = &p[0][..8];
        let mut seen = slots.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(
            seen.len(),
            8,
            "8 residual leaves need 8 ctx slots, got {slots:?}"
        );
        let n = ma_stream_leaf_count(&[gmod], &[], &[pass], 0, Some(&hf));
        assert_eq!(n, 9);
    }

    #[test]
    fn hybrid_000_token_range() {
        assert_eq!(hybrid_uint_000(0), (0, 0, 0));
        assert_eq!(hybrid_uint_000(1), (1, 0, 0));
        assert_eq!(hybrid_uint_000(2), (2, 1, 0));
        assert_eq!(hybrid_uint_000(3), (2, 1, 1));
        assert_eq!(hybrid_uint_000(4), (3, 2, 0));
        assert_eq!(hybrid_uint_400(0), (0, 0, 0));
        assert_eq!(hybrid_uint_400(15), (15, 0, 0));
        assert_eq!(hybrid_uint_400(16), (16, 4, 0));
    }

    #[test]
    fn huffman_kraft() {
        let freqs = [100u64, 50, 25, 10, 5, 1, 0, 0];
        let l = huffman_lengths(&freqs, 15);
        let mut acc = 0i32;
        for &x in &l {
            if x > 0 {
                acc += 1 << (15 - x);
            }
        }
        assert_eq!(acc, 1 << 15);
    }

    #[test]
    fn ma_tree_preorder_pass_then_gmod() {
        let order = ma_tree_preorder(4, 0, 2);
        assert!(matches!(
            order.as_slice(),
            [(MaChain::Pass, 2), (MaChain::Gmod, 4)]
        ));
        assert_eq!(ma_chain_leaf_ctx(0, 2, 0), 1);
        assert_eq!(ma_chain_leaf_ctx(0, 2, 1), 0);
        assert_eq!(ma_chain_leaf_ctx(2, 4, 0), 5);
    }

    #[test]
    fn eight_equal_freqs_are_length_3() {
        let l = huffman_lengths(&[1u64; 8], 15);
        assert_eq!(l, vec![3, 3, 3, 3, 3, 3, 3, 3]);
        let code = PrefixCode::from_freqs(&[1u64; 8]);
        assert_eq!(code.alphabet_size(), 8);
    }

    #[test]
    fn histograms_nonsimple_cluster_map_writes() {
        let codes: Vec<PrefixCode> = (0..12)
            .map(|i| PrefixCode::from_freqs(&[10 + i as u64, 5, 2, 1]))
            .collect();
        let mut w = BitWriter::with_capacity(256);
        write_histograms(&mut w, &codes, None);
        let bytes = w.finish();
        assert!(
            bytes.len() > 8,
            "12-cluster prefix tables should emit bytes"
        );
    }

    #[test]
    fn hybrid000_lut_matches_token_path() {
        let freqs: Vec<u64> = (0..16).map(|i| 32 - i).collect();
        let code = PrefixCode::from_freqs(&freqs);
        for v in 0..256u32 {
            let (token, extra_n, extra) = hybrid_uint_000(v);
            let n = code.nbits[token as usize] as u32;
            let bits = code.bits[token as usize] as u32 | (extra << n);
            assert_eq!(code.lut_n[v as usize] as u32, n + extra_n);
            assert_eq!(code.lut_b[v as usize], bits);
        }
    }
}
