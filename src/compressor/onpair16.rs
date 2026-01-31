use crate::lpm::{LongestPrefixMatcher16, StaticLongestPrefixMatcher16};
use rustc_hash::FxHashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Maximum token length constraint for optimization
const MAX_LENGTH: usize = 16;

pub struct OnPair16 {
    // Merging frequency threshold
    threshold: u16,

    // Compressed data storage
    compressed_data: Vec<u16>,           // Sequence of token IDs
    string_boundaries: Vec<usize>,       // End positions for each string
    
    // Dictionary storage  
    dictionary: Vec<u8>,                 // Raw token data
    token_boundaries: Vec<u32>,        // Token end positions in dictionary
}

impl OnPair16 {
    /// Creates a new compressor instance
    pub fn new(threshold: u16) -> Self {
        assert!(threshold > 1, "Threshold must be greater than 1");

        Self {
            threshold,
            compressed_data: Vec::new(),
            string_boundaries: Vec::new(),
            dictionary: Vec::new(),
            token_boundaries: Vec::new(),
        }
    }
    
    /// Creates a new compressor with capacity hints for better memory allocation
    pub fn with_capacity(threshold: u16, n_strings: usize, n_bytes: usize) -> Self {
        assert!(threshold > 1, "Threshold must be greater than 1");
        
        Self {
            threshold,
            compressed_data: Vec::with_capacity(n_bytes),
            string_boundaries: Vec::with_capacity(n_strings),
            dictionary: Vec::with_capacity(1024 * 1024),
            token_boundaries: Vec::with_capacity(1 << 16),
        }
    }

    /// Compresses a collection of strings
    /// 
    /// This is a convenience method that handles the flattening for you.
    pub fn compress_strings<S: AsRef<str>>(&mut self, strings: &[S]) {
        let (data, end_positions) = flatten_strings(strings);
        self.compress_bytes(&data, &end_positions);
    }

    /// Compresses pre-flattened byte data with end positions
    /// 
    /// The `end_positions` should be a prefix sum array starting with 0.
    /// For example, if you have strings of lengths [3, 2, 4], 
    /// then `end_positions` should be [0, 3, 5, 9].
    pub fn compress_bytes(&mut self, data: &[u8], end_positions: &[usize]) {
        let lpm = self.train_dictionary(data, end_positions);
        let static_lpm = lpm.finalize();
        self.parse_data(data, end_positions, &static_lpm);
    }

    /// Phase 1: Dictionary population
    /// 
    /// Uses longest prefix matching to parse training data and identify frequent
    /// adjacent token pairs.
    /// 
    /// # Algorithm
    /// 1. Initialize 256 single-byte tokens  
    /// 2. Parse shuffled training data with longest prefix matching
    /// 3. Track adjacent token pair frequencies
    /// 4. Merge frequent pairs into new tokens until dictionary full (65,536 tokens)
    fn train_dictionary(&mut self, data: &[u8], end_positions: &[usize]) -> LongestPrefixMatcher16 {
        self.token_boundaries.push(0);

        let mut frequency: FxHashMap<(u16, u16), u16> = FxHashMap::default();
        let mut lpm = LongestPrefixMatcher16::new();
        let mut next_token_id = 256;
    
        // Initialize the dictionary with single-byte tokens
        for i in 0..256 {
            let token = vec![i as u8];
            lpm.insert(&token, i as u16);
            self.dictionary.extend(&token);
            self.token_boundaries.push(self.dictionary.len() as u32);
        }

        // Shuffle entries
        let mut shuffled_indices: Vec<usize> = (0..end_positions.len()-1).collect();
        shuffled_indices.shuffle(&mut thread_rng());

        // Iterate over entries
        'outer: for &index in shuffled_indices.iter() {
            let start = end_positions[index];
            let end = end_positions[index + 1];

            if start == end {
                continue;
            }
    
            let (match_token_id, match_length) = lpm.find_longest_match(&data[start..end]).unwrap();
            let mut previous_token_id = match_token_id;
            let mut previous_length = match_length;

            let mut pos = start + previous_length;
    
            while pos < end {
                // Find the longest match
                let (match_token_id, match_length) = lpm.find_longest_match(&data[pos..end]).unwrap();

                let mut added_token = false;
                if match_length + previous_length <= MAX_LENGTH {
                    // Update token frequency and possibly merge tokens
                    *frequency.entry((previous_token_id, match_token_id)).or_insert(0) += 1;

                    if frequency[&(previous_token_id, match_token_id)] >= self.threshold {
                        let merged_token = &data[pos - previous_length..pos + match_length];
                        added_token = lpm.insert(merged_token, next_token_id);
                        if added_token {
                            self.dictionary.extend(merged_token);
                            self.token_boundaries.push(self.dictionary.len() as u32);
    
                            frequency.remove(&(previous_token_id, match_token_id));
                            previous_token_id = next_token_id;
                            previous_length = merged_token.len();

                            if next_token_id == u16::MAX {
                                break 'outer;
                            }

                            next_token_id += 1;
                        }
                    }
                }

                if !added_token {
                    previous_token_id = match_token_id;
                    previous_length = match_length;
                }
                
                pos += match_length;
            }
        }
    
        lpm
    }
    
    /// Phase 2: String compression using learned dictionary
    /// 
    /// Compresses each string independently by greedily applying longest prefix matching
    /// with the constructed dictionary. Each string becomes a sequence of token IDs.
    fn parse_data(&mut self, data: &[u8], end_positions: &[usize], lpm: &StaticLongestPrefixMatcher16) {
        self.string_boundaries.push(0);
    
        for window in end_positions.windows(2) {
            let start = window[0];
            let end = window[1];

            if start == end {
                self.string_boundaries.push(self.compressed_data.len());
                continue;
            }
    
            let mut pos = start;
            while pos < end {
                // Find the longest match
                let (token_id, length) = lpm.find_longest_match(&data[pos..end]).unwrap();
                self.compressed_data.push(token_id);
                pos += length;
            }
    
            self.string_boundaries.push(self.compressed_data.len());
        }
    }

    /// Decompresses a specific string by index
    /// 
    /// # Safety Warning
    /// This method uses unsafe memory operations for performance. All tokens are constrained
    /// to be at most 16 bytes long, and this method always copies exactly 16 bytes for each
    /// token regardless of the actual token length (for optimization).
    /// 
    /// The buffer must have sufficient space beyond the actual decompressed data to accommodate
    /// the full 16-byte copy for the last token, or undefined behavior will occur.
    #[inline]
    pub fn decompress_string(&mut self, index: usize, buffer: &mut [u8]) -> usize {
        let item_start = self.string_boundaries[index];
        let item_end = self.string_boundaries[index + 1];
        let dict_ptr = self.dictionary.as_ptr();
        let end_positions_ptr = self.token_boundaries.as_ptr();
        let mut size = 0;

        for &token_id in &self.compressed_data[item_start..item_end] {
            unsafe {
                // Access dictionary positions using raw pointers
                let dict_start = *end_positions_ptr.add(token_id as usize) as usize;
                let dict_end = *end_positions_ptr.add(token_id as usize + 1) as usize;
                let length = dict_end - dict_start;

                let src = dict_ptr.add(dict_start);
                let dst = buffer.as_mut_ptr().add(size);
                std::ptr::copy_nonoverlapping(src, dst, MAX_LENGTH);

                size += length;
            }
        }

        size
    }

    /// Decompresses all strings
    /// 
    /// # Safety Warning
    /// This method uses unsafe memory operations for performance. All tokens are constrained
    /// to be at most 16 bytes long, and this method always copies exactly 16 bytes for each
    /// token regardless of the actual token length (for optimization).
    /// 
    /// The buffer must have sufficient space beyond the actual decompressed data to accommodate
    /// the full 16-byte copy for the last token, or undefined behavior will occur.
    pub fn decompress_all(&self, buffer: &mut [u8]) -> usize {
        let dict_ptr = self.dictionary.as_ptr();
        let end_positions_ptr = self.token_boundaries.as_ptr();
        let mut size = 0;

        for &token_id in self.compressed_data.iter(){
            unsafe {
                let dict_start = *end_positions_ptr.add(token_id as usize) as usize;
                let dict_end = *end_positions_ptr.add(token_id as usize + 1) as usize;
                let length = dict_end - dict_start;

                let src = dict_ptr.add(dict_start);
                let dst = buffer.as_mut_ptr().add(size);
                std::ptr::copy_nonoverlapping(src, dst, MAX_LENGTH);

                size += length;
            }
        }

        size
    }

    /// Returns the total space (in bytes) used by the compressed data
    pub fn space_used(&self) -> usize {
        self.compressed_data.len() * std::mem::size_of::<u16>() + 
        self.dictionary.len() + 
        self.token_boundaries.len() * std::mem::size_of::<u32>()
    }

    /// Shrinks all internal buffers to fit their current contents
    pub fn shrink_to_fit(&mut self) {
        self.compressed_data.shrink_to_fit();
        self.string_boundaries.shrink_to_fit();
        self.dictionary.shrink_to_fit();
        self.token_boundaries.shrink_to_fit();
    }
}

/// Flattens a collection of strings into a single byte array with boundary positions
/// 
/// Returns a tuple of (flattened_data, end_positions) where end_positions is a 
/// prefix sum array starting with 0.
fn flatten_strings<S: AsRef<str>>(strings: &[S]) -> (Vec<u8>, Vec<usize>) {
    let total_len: usize = strings.iter().map(|s| s.as_ref().len()).sum();
    let mut data = Vec::with_capacity(total_len);
    let mut end_positions = Vec::with_capacity(strings.len() + 1);
    
    end_positions.push(0);
    
    for string in strings {
        data.extend_from_slice(string.as_ref().as_bytes());
        end_positions.push(data.len());
    }
    
    (data, end_positions)
}