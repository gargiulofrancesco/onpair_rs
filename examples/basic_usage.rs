use onpair_rs::{OnPair, OnPair16};

fn main() {
    // Simulate a database column with user IDs
    let strings = vec![
        "user_000001",
        "user_000002", 
        "user_000003",
        "admin_001",
        "user_000004",
        "user_000005",
        "guest_001",
        "user_000006",
        "admin_002",
        "user_000007",
    ];

    // Initialize onpair with capacity hints
    let n_strings = strings.len();
    let n_bytes = strings .iter().map(|s| s.len()).sum::<usize>();

    // Compress data using OnPair
    let mut onpair = OnPair::with_capacity(n_strings, n_bytes);
    onpair.compress_strings(&strings);

    // Compress data using OnPair16
    let mut onpair16 = OnPair16::with_capacity(n_strings, n_bytes);
    onpair16.compress_strings(&strings);

    // Decompress and verify the results
    let max_length = strings.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut buffer = vec![0u8; max_length + 16];
    for (i, user_id) in strings.iter().enumerate() {
        println!("\nString {}: \"{}\"", i, user_id);

        let size = onpair.decompress_string(i, &mut buffer);
        println!("- OnPair: \"{}\"", String::from_utf8_lossy(&buffer[..size]));

        let size = onpair16.decompress_string(i, &mut buffer);
        println!("- OnPair16: \"{}\"", String::from_utf8_lossy(&buffer[..size]));
    }
}