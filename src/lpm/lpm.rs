//! Longest Prefix Matcher for OnPair
//!
//! Provides efficient longest prefix matching using a hybrid approach:
//! - **Short matches** (≤8 bytes): Direct hash table lookup
//! - **Long matches** (>8 bytes): Bucketed by 8-byte prefix with suffix verification

use rustc_hash::FxHashMap;

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

/// Threshold for switching from direct lookup to bucketed approach
const MIN_MATCH: usize = 8;

/// Hybrid longest prefix matcher supporting arbitrary-length patterns
/// 
/// Combines direct hash lookup for short patterns with bucketed search for long patterns.
/// Optimized for OnPair's token discovery phase where most patterns are short but
/// long patterns provide significant compression benefits.
/// 
/// # Type Parameters
/// - `V`: Token ID type (typically `u16` for OnPair)
pub struct LongestPrefixMatcher<V> {
    long_match_buckets: FxHashMap<u64, Vec<V>>,     // 8-byte prefix → candidate token IDs
    short_match_lookup: FxHashMap<(u64, u8), V>,    // (prefix, length) → token ID
    dictionary: Vec<u8>,                            // Suffix storage for long patterns
    end_positions: Vec<u32>,                        // Boundary positions in dictionary
}

impl<V> LongestPrefixMatcher<V> 
where 
    V: Copy + Into<usize>,
{   
    /// Creates a new empty longest prefix matcher
    pub fn new() -> Self {
        Self {
            long_match_buckets: FxHashMap::default(),
            short_match_lookup: FxHashMap::default(),
            dictionary: Vec::with_capacity(1024 * 1024),
            end_positions: vec![0],
        }
    }

    /// Inserts a new pattern with associated token ID
    /// 
    /// Automatically chooses storage strategy based on pattern length:
    /// - Short patterns (≤8 bytes): Direct hash table insertion
    /// - Long patterns (>8 bytes): Bucketed by 8-byte prefix with suffix storage
    /// 
    /// Long pattern buckets are kept sorted by pattern length (descending) for
    /// efficient longest-match-first lookup during matching.
    #[inline]
    pub fn insert(&mut self, entry: &[u8], id: V) {
        if entry.len() > MIN_MATCH {
            // Long pattern: store 8-byte prefix in bucket, suffix in dictionary
            let prefix = Self::bytes_to_u64_le(&entry, MIN_MATCH);
            self.dictionary.extend_from_slice(&entry[MIN_MATCH..]);
            self.end_positions.push(self.dictionary.len() as u32);

            let bucket = self.long_match_buckets.entry(prefix).or_default();
            bucket.push(id);
            // Sort by pattern length (longest first) for greedy matching
            bucket.sort_unstable_by(|&id1, &id2| {
                let len1 = self.end_positions[id1.into() + 1] as usize 
                           - self.end_positions[id1.into()] as usize;
                let len2 = self.end_positions[id2.into() + 1] as usize 
                           - self.end_positions[id2.into()] as usize;
                len2.cmp(&len1)
            });
        } else {
            // Short pattern: direct hash table lookup
            let prefix = Self::bytes_to_u64_le(&entry, entry.len());
            self.short_match_lookup.insert((prefix, entry.len() as u8), id);
            self.end_positions.push(self.dictionary.len() as u32);
        }
    }

    /// Finds the longest matching pattern for the given input data
    /// 
    /// Returns the token ID and match length for the longest pattern that matches
    /// the beginning of the input data. Uses two-phase search:
    /// 
    /// 1. **Long pattern search**: Check bucketed patterns (>8 bytes) first for longest matches
    /// 2. **Short pattern search**: Check direct lookup patterns (≤8 bytes) in decreasing length order
    #[inline]
    pub fn find_longest_match(&self, data: &[u8]) -> Option<(V, usize)> {
        // Phase 1: Long pattern search (>8 bytes) - check longest matches first
        if data.len() > MIN_MATCH {
            let prefix = Self::bytes_to_u64_le(&data, MIN_MATCH);
            
            if let Some(bucket) = self.long_match_buckets.get(&prefix) {
                for &id in bucket {
                    let dict_start = self.end_positions[id.into()] as usize;
                    let dict_end = self.end_positions[id.into() + 1] as usize;
                    let length = dict_end - dict_start;
                    // Verify suffix matches beyond the 8-byte prefix
                    if data[MIN_MATCH..].starts_with(&self.dictionary[dict_start..dict_end]) {
                        return Some((id, MIN_MATCH + length));
                    }
                }
            }
        }

        // Phase 2: Short pattern search (≤8 bytes) - longest to shortest
        for length in (1..=MIN_MATCH.min(data.len())).rev() {
            let prefix = Self::bytes_to_u64_le(&data, length);
            
            if let Some(&id) = self.short_match_lookup.get(&(prefix, length as u8)) {
                return Some((id, length));
            }
        }

        None
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
}