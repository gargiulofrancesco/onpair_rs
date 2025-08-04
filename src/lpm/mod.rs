pub mod lpm;
pub mod lpm16;

pub use lpm::LongestPrefixMatcher;
pub use lpm16::{LongestPrefixMatcher16, StaticLongestPrefixMatcher16};