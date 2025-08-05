# OnPair: Short Strings Compression for Fast Random Access

[![Paper](https://img.shields.io/badge/Paper-arXiv:2508.02280-blue)](https://arxiv.org/abs/2508.02280)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Rust implementation of **OnPair**, a compression algorithm designed for efficient random access on sequences of short strings.

## Overview

OnPair is a field-level compression algorithm designed for workloads requiring fast random access to individual strings in large collections. The compression process consists of two distinct phases:

- **Training Phase**: A longest prefix matching strategy is used to parse the input and identify frequent adjacent token pairs. When the frequency of a pair exceeds a predefined threshold, a new token is created to represent the merged pair. This continues until the dictionary is full or the input data is exhausted. The dictionary supports up to 65,536 tokens, with each token assigned a fixed 2-byte ID.
- **Parsing Phase**: Once the dictionary is constructed, each string is compressed independently into a sequence of token IDs by greedily applying longest prefix matching

OnPair16 is a variant that limits dictionary entries to a maximum length of 16 bytes. This constraint enables further optimizations in both longest prefix matching and decoding.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
onpair_rs = { git = "https://github.com/gargiulofrancesco/onpair_rs" }
```

## Quick Start

```rust
use onpair_rs::{OnPair, OnPair16};

fn main() {
    // Your string data
    let strings = vec![
        "user_000001",
        "user_000002", 
        "user_000003",
        "admin_001",
        "user_000004",
    ];

    let mut compressor = OnPair::new();
    
    // Compress all strings
    compressor.compress_strings(&strings);
    
    // Random access decompression
    let mut buffer = vec![0u8; 256];
    
    for i in 0..strings.len() {
        let size = compressor.decompress_string(i, &mut buffer);
        let decompressed = std::str::from_utf8(&buffer[..size]).unwrap();
        println!("String {}: {}", i, decompressed);
    }
}
```

## Building from Source

```bash
git clone https://github.com/gargiulofrancesco/onpair_rs
cd onpair_rs

# Build with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run the example
RUSTFLAGS="-C target-cpu=native" cargo run --example basic_usage --release
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Francesco Gargiulo** - [francesco.gargiulo@phd.unipi.it](mailto:francesco.gargiulo@phd.unipi.it)
- **Rossano Venturini** - [rossano.venturini@unipi.it](mailto:rossano.venturini@unipi.it)

*University of Pisa, Italy*
