//! Optimized longest prefix matcher for 16-byte constrained tokens
//!
//! Specialized variant of the longest prefix matcher designed for OnPair16's
//! length-constrained tokens. The 16-byte limit enables aggressive optimizations:
//! - Simplified data structures with inline suffix storage
//! - More efficient long patterns matching with bitwise operations
//! - Transition to static representation for parsing phase
//!
//! Provides both dynamic (training) and static (parsing) implementations.

use rustc_hash::FxHashMap;
use ptr_hash::{bucket_fn::Linear, PtrHash, PtrHashParams};

/// Number of suffix bytes stored inline in bucket entries
const N_INLINE_SUFFIXES: usize = 4;
/// Maximum entries per bucket before reorganization
const MAX_BUCKET_SIZE: usize = 128;

/// Bit masks for extracting prefixes of different lengths (little-endian)
const MASKS: [u64; 9] = [
    0x0000000000000000, // 0 bytes
    0x00000000000000FF, // 1 byte
    0x000000000000FFFF, // 2 bytes
    0x0000000000FFFFFF, // 3 bytes
    0x00000000FFFFFFFF, // 4 bytes
    0x000000FFFFFFFFFF, // 5 bytes
    0x0000FFFFFFFFFFFF, // 6 bytes
    0x00FFFFFFFFFFFFFF, // 7 bytes
    0xFFFFFFFFFFFFFFFF, // 8 bytes
];

/// Dynamic longest prefix matcher for 16-byte constrained patterns
/// 
/// Used during OnPair16's training phase. Optimized for incremental pattern insertion
/// with length constraint enabling simplified data structures and faster operations.
pub struct LongestPrefixMatcher16 {
    dictionary: FxHashMap<(u64, u8), u16>,              // (prefix, length) → token ID
    buckets: FxHashMap<u64, Vec<(u64, u8, u16)>>,       // 8-byte prefix → (suffix, len, ID) entries
}

impl LongestPrefixMatcher16{   
    /// Creates a new empty matcher for training phase
    pub fn new() -> Self {
        Self {
            dictionary: FxHashMap::default(),
            buckets: FxHashMap::default(),
        }
    }

    /// Inserts a pattern with length constraint checking
    /// 
    /// Returns false if the pattern would cause bucket overflow.
    /// 
    /// # Length-based storage strategy
    /// - ≤8 bytes: Direct dictionary storage
    /// - >8 bytes: Bucketed with 8-byte prefix + up to 8-byte suffix
    #[inline]
    pub fn insert(&mut self, data: &[u8], id: u16) -> bool {
        let length = data.len();

        if length <= 8 {
            // Short pattern: direct hash table storage
            let value = bytes_to_u64_le(data, length);
            self.dictionary.insert((value, length as u8), id);
            return true;
        }
        else {
            // Long pattern: bucketed storage with overflow protection
            let prefix = bytes_to_u64_le(data, 8);
            let bucket = self.buckets.entry(prefix).or_default();

            // Reject patterns that would cause bucket overflow
            if bucket.len() > MAX_BUCKET_SIZE {
                return false;
            }

            let suffix_len = length - 8;
            let suffix = bytes_to_u64_le(&data[8..], suffix_len);
            
            bucket.push((suffix, suffix_len as u8, id));
            // Sort by suffix length (longest first) for greedy matching
            bucket.sort_unstable_by(|&a, &b| b.1.cmp(&a.1));
            return true;
        }
    }

    /// Finds longest matching pattern with 16-byte constraint optimization
    #[inline]
    pub fn find_longest_match(&self, data: &[u8]) -> Option<(u16, usize)> {
        // Phase 1: Long pattern search (>8 bytes)
        if data.len() > 8 {
            let suffix_len = data.len().min(16) - 8;
            let prefix = bytes_to_u64_le(&data, 8);
            let suffix = bytes_to_u64_le(&data[8..], suffix_len);
            
            if let Some(bucket) = self.buckets.get(&prefix) {
                for &(entry_suffix, entry_suffix_len, entry_id) in bucket {
                    // Fast bitwise prefix comparison
                    if is_prefix(suffix, entry_suffix, suffix_len, entry_suffix_len as usize) {
                        return Some((entry_id, 8 + entry_suffix_len as usize));
                    }
                }
            }
        }

        // Phase 2: Short pattern search (≤8 bytes)
        let mut prefix = bytes_to_u64_le(&data, 8);
        for length in (1..=8.min(data.len())).rev() {
            prefix = prefix & MASKS[length];
            if let Some(&id) = self.dictionary.get(&(prefix, length as u8)) {
                return Some((id, length));
            }
        }

        None
    }

    /// Converts dynamic matcher to optimized static representation
    /// 
    /// Transitions from training-optimized data structures to parsing-optimized
    /// structures. The static version uses more efficient layouts optimized for
    /// read-only access during the parsing phase.
    pub fn finalize(&self) -> StaticLongestPrefixMatcher16 {
        let mut long_dictionary = FxHashMap::default();
        let mut long_buckets = Vec::new();
        
        for (&prefix, bucket) in self.buckets.iter() {
            let (answer_id, answer_length) = self.find_longest_match(&prefix.to_le_bytes()).unwrap();
            let offset = long_buckets.len() as u16;
            let mut n_suffixes: u16 = 0;
            
            let mut inline_suffixes: [u64; N_INLINE_SUFFIXES] = [0; N_INLINE_SUFFIXES];
            let mut inline_lengths: [u8; N_INLINE_SUFFIXES] = [0; N_INLINE_SUFFIXES];
            let mut inline_ids: [u16; N_INLINE_SUFFIXES] = [0; N_INLINE_SUFFIXES];

            for i in 0..N_INLINE_SUFFIXES.min(bucket.len()) {
                inline_suffixes[i] = bucket[i].0;
                inline_lengths[i] = bucket[i].1;
                inline_ids[i] = bucket[i].2;
                n_suffixes += 1;
            }

            for &(suffix, len, id) in bucket.iter().skip(N_INLINE_SUFFIXES) {
                long_buckets.push((suffix, len, id));
                n_suffixes += 1;
            }
 
            let info_long_match = LongMatchInfo {
                prefix,
                answer_id,
                answer_length: answer_length as u8,
                n_suffixes,
                inline_suffixes,
                inline_lengths,
                inline_ids,
                offset,
            };

            long_dictionary.insert(prefix, info_long_match);
        }

        let mut short_dictionary = FxHashMap::default();
        
        for (&(prefix, length), &id) in self.dictionary.iter() {
            if length == 8 {
                if long_dictionary.contains_key(&prefix) {
                    continue;
                }

                let info_long_match = LongMatchInfo {
                    prefix,
                    answer_id: id,
                    answer_length: length,
                    n_suffixes: 0,
                    inline_suffixes: [0; N_INLINE_SUFFIXES],
                    inline_lengths: [0; N_INLINE_SUFFIXES],
                    inline_ids: [0; N_INLINE_SUFFIXES],
                    offset: 0,
                };

                long_dictionary.insert(prefix, info_long_match);

                continue;
            }

            short_dictionary.insert((prefix, length), id);
        }

        let prefixes = long_dictionary.keys().copied().collect::<Vec<_>>();
        let mut params = PtrHashParams::default_fast();
        params.remap = false;
        let long_phf = PtrHash::new(&prefixes, params);
        let max = prefixes.iter()
            .map(|prefix| long_phf.index_no_remap(prefix))
            .fold(0, |acc, idx| acc.max(idx));

        let mut long_info = vec![LongMatchInfo::default(); max as usize + 1];
        for (prefix, &p) in long_dictionary.iter() {
            let index = long_phf.index_no_remap(prefix) as usize;
            long_info[index] = p;
        }

        StaticLongestPrefixMatcher16 {
            short_dictionary,
            long_phf,
            long_info,
            long_buckets,
        }
    }
}

/// Cache-aligned metadata for efficient long pattern matching
/// 
/// Includes inline storage for up to N_INLINE_SUFFIXES patterns to minimize
/// memory indirection during the parsing phase.
#[repr(align(64))] // Ensure 64-byte alignment
#[derive(Default, Copy, Clone)]
struct LongMatchInfo{
    pub prefix: u64,                                   // 8-byte prefix key
    pub inline_suffixes: [u64; N_INLINE_SUFFIXES],     // Inline suffix storage  
    pub inline_lengths: [u8; N_INLINE_SUFFIXES],       // Corresponding lengths
    pub inline_ids: [u16; N_INLINE_SUFFIXES],          // Corresponding token IDs
    pub n_suffixes: u16,                               // Total number of suffixes
    pub offset: u16,                                   // Offset into overflow storage
    pub answer_id: u16,                                // Default answer for prefix match
    pub answer_length: u8,                             // Default answer length
}

/// Static (read-only) longest prefix matcher optimized for parsing phase
/// 
/// Immutable data structure optimized for maximum query performance during
/// string parsing. Uses perfect hash functions and inline storage to minimize
/// memory indirection and cache misses.
pub struct StaticLongestPrefixMatcher16{
    short_dictionary: FxHashMap<(u64, u8), u16>,    // Short pattern lookup table
    long_phf: PtrHash<u64, Linear>,                 // Perfect hash for long pattern prefixes  
    long_info: Vec<LongMatchInfo>,                  // Long pattern metadata with inline storage
    long_buckets: Vec<(u64, u8, u16)>,              // Overflow storage for long patterns
}

impl StaticLongestPrefixMatcher16{
    /// Optimized longest match search for parsing phase
    /// 
    /// High-performance implementation leveraging static optimizations:
    /// - Perfect hash function eliminates hash collisions
    /// - Inline suffix storage reduces memory indirection
    #[inline]
    pub fn find_longest_match(&self, data: &[u8]) -> Option<(u16, usize)> {
        // Phase 1: Long pattern search using perfect hash function
        if data.len() >= 8 {
            let suffix_len = data.len().min(16) - 8;
            let prefix = bytes_to_u64_le(&data, 8);
            let suffix = bytes_to_u64_le(&data[8..], suffix_len);

            let long_answer = self.compute_long_answer(prefix, suffix, suffix_len);
            if long_answer.is_some() {
                return long_answer;
            }
        }

        // Phase 2: Short pattern search
        let mut prefix = bytes_to_u64_le(&data, 8);
        for length in (1..=7.min(data.len())).rev() {
            prefix = prefix & MASKS[length];
            if let Some(&id) = self.short_dictionary.get(&(prefix, length as u8)) {
                return Some((id, length));
            }
        }

        None
    }

    /// Optimized long pattern resolution with inline storage
    #[inline]
    pub fn compute_long_answer(&self, prefix: u64, suffix: u64, suffix_len: usize) -> Option<(u16, usize)> {
        let index = self.long_phf.index_no_remap(&prefix);

        // Perfect hash validation - ensure we found the right prefix
        if index >= self.long_info.len() || prefix != self.long_info[index].prefix {
            return None;
        }

        let long_info = &self.long_info[index];

        // Phase 1: Check inline suffixes
        for i in 0..N_INLINE_SUFFIXES.min(long_info.n_suffixes as usize) {
            let inline_suffix = long_info.inline_suffixes[i as usize];
            let inline_id = long_info.inline_ids[i as usize];
            let inline_len = long_info.inline_lengths[i as usize] as usize;
            if is_prefix(suffix, inline_suffix, suffix_len, inline_len) {
                return Some((inline_id, 8 + inline_len));
            }
        }

        // Phase 2: Check overflow bucket if inline storage insufficient
        if long_info.n_suffixes as usize > N_INLINE_SUFFIXES {
            let start = long_info.offset as usize;
            let end = start + long_info.n_suffixes as usize - N_INLINE_SUFFIXES;

            for i in start..end {
                let item = &self.long_buckets[i];
                if is_prefix(suffix, item.0, suffix_len, item.1 as usize) {
                    return Some((item.2, 8 + item.1 as usize));
                }
            }
        }

        // Phase 3: Fallback to default prefix match
        return Some((long_info.answer_id, long_info.answer_length as usize));
    }
}

/// Converts byte sequence to little-endian u64 with length masking
#[inline(always)]
fn bytes_to_u64_le(bytes: &[u8], len: usize) -> u64 {
    let ptr = bytes.as_ptr();
    let value = unsafe {
        *(ptr as *const u64)
    };

    value & MASKS[len]
}

/// Fast prefix checking using bitwise operations
#[inline(always)]
fn is_prefix(text: u64, prefix: u64, text_size: usize, prefix_size: usize) -> bool {
    prefix_size <= text_size && shared_prefix_size(text, prefix) >= prefix_size
}

/// Bitwise shared prefix length calculation
#[inline(always)]
fn shared_prefix_size(a: u64, b: u64) -> usize {
    ((a ^ b).trailing_zeros() >> 3) as usize
}
