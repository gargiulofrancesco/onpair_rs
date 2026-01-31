//! Longest Prefix Matcher for OnPair
//!
//! Provides efficient longest prefix matching using a hybrid approach:
//! - Short matches (≤8 bytes): Direct hash table lookup in decreasing length order
//! - Long matches (>8 bytes): Skip the first 8 bytes via hash table, then trie lookup for suffixes

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

/// Length of the prefix used for indexing trie roots
const TRIE_PREFIX_LEN: usize = 8;

/// Trie node structure for suffix storage of long patterns
#[derive(Clone, Debug)]
struct TrieNode<V> {
    id: Option<V>,
    children: Vec<(u8, u32)>,
}

/// Hybrid longest prefix matcher supporting arbitrary-length patterns
/// 
/// Combines direct hash lookup for short patterns with trie lookup for long patterns.
/// Optimized for OnPair's token discovery phase where most patterns are short but
/// long patterns provide significant compression benefits.
/// 
/// Type Parameters
/// - `V`: Token ID type (typically `u16` for OnPair)
pub struct LongestPrefixMatcher<V> {
    /// Short patterns: (prefix, length) -> token ID
    short_match_lookup: FxHashMap<(u64, u8), V>,
    
    /// Long patterns Trie roots: 8-byte prefix -> index in node_pool
    long_match_roots: FxHashMap<u64, u32>,
    
    /// Pool of trie nodes (using indices instead of pointers)
    node_pool: Vec<TrieNode<V>>,
}

impl<V> LongestPrefixMatcher<V> 
where 
    V: Copy + Into<usize>,
{   
    /// Creates a new empty longest prefix matcher
    pub fn new() -> Self {
        Self {
            short_match_lookup: FxHashMap::default(),
            long_match_roots: FxHashMap::default(),
            node_pool: Vec::with_capacity(256 * 1024),
        }
    }

    /// Inserts a new pattern with associated token ID
    /// 
    /// Automatically chooses storage strategy based on pattern length:
    /// - Short patterns (≤8 bytes): Direct hash table insertion
    /// - Long patterns (>8 bytes): Trie storage for suffixes beyond 8-byte prefix
    #[inline]
    pub fn insert(&mut self, entry: &[u8], id: V) {
        if entry.len() <= TRIE_PREFIX_LEN {
            // Short pattern: direct hash table lookup
            let prefix = Self::bytes_to_u64_le(entry, entry.len());
            self.short_match_lookup.insert((prefix, entry.len() as u8), id);
        } else {
            // Long pattern: use trie for suffix
            let prefix = Self::bytes_to_u64_le(entry, TRIE_PREFIX_LEN);

            // Find or create root node for this 8-byte prefix
            let mut node_idx = *self.long_match_roots.entry(prefix).or_insert_with(|| {
                let idx = self.node_pool.len() as u32;
                self.node_pool.push(TrieNode { id: None, children: Vec::new() });
                idx
            });

            // Traverse/Create nodes for the suffix
            for &byte in &entry[TRIE_PREFIX_LEN..] {
                // Check if child exists for this byte
                let mut child_idx = None;
                
                // Scope the borrow of the node
                {
                    let node = &self.node_pool[node_idx as usize];
                    for &(c, idx) in &node.children {
                        if c == byte {
                            child_idx = Some(idx);
                            break;
                        }
                    }
                }

                if let Some(idx) = child_idx {
                    node_idx = idx;
                } else {
                    // Create new node
                    let new_idx = self.node_pool.len() as u32;
                    self.node_pool.push(TrieNode { id: None, children: Vec::new() });
                    
                    // Link parent to new node
                    self.node_pool[node_idx as usize].children.push((byte, new_idx));
                    node_idx = new_idx;
                }
            }

            // Set ID at the final node (leaf of this pattern)
            self.node_pool[node_idx as usize].id = Some(id);
        }
    }

    /// Finds the longest matching pattern for the given input data
    /// 
    /// Returns the token ID and match length for the longest pattern that matches
    /// the beginning of the input data. Uses two-phase search:
    /// 
    /// 1. Long pattern search: Use first 8 bytes to find trie root, then traverse trie for deepest match
    /// 2. Short pattern search: Check direct lookup patterns (≤8 bytes) in decreasing length order
    #[inline]
    pub fn find_longest_match(&self, data: &[u8]) -> Option<(V, usize)> {
        // Phase 1: Long pattern search (>8 bytes)
        if data.len() > TRIE_PREFIX_LEN {
            let prefix = Self::bytes_to_u64_le(data, TRIE_PREFIX_LEN);
            
            if let Some(&root_idx) = self.long_match_roots.get(&prefix) {
                let mut best_long_match = None;
                let mut current_idx = root_idx;
                let mut current_len = TRIE_PREFIX_LEN;
                
                // Traverse the trie to find the longest possible match
                for &byte in &data[TRIE_PREFIX_LEN..] {
                    let node = &self.node_pool[current_idx as usize];
                    
                    // Find child node for the current byte
                    let mut found_idx = None;
                    for &(c, idx) in &node.children {
                        if c == byte {
                            found_idx = Some(idx);
                            break;
                        }
                    }

                    if let Some(idx) = found_idx {
                        current_idx = idx;
                        current_len += 1;
                        
                        // If this node marks the end of a valid pattern, record it
                        // We keep going to see if there's a longer match
                        if let Some(id) = self.node_pool[current_idx as usize].id {
                            best_long_match = Some((id, current_len));
                        }
                    } else {
                        // Mismatch in trie, cannot go deeper
                        break;
                    }
                }

                if let Some(match_res) = best_long_match {
                    return Some(match_res);
                }
            }
        }

        // Phase 2: Short pattern search (≤8 bytes) - longest to shortest
        for length in (1..=TRIE_PREFIX_LEN.min(data.len())).rev() {
            let prefix = Self::bytes_to_u64_le(data, length);
            
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