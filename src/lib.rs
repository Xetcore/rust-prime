use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use std::path::Path; // Added for path checking
use std::fs::File; // For load_pre_tokenized_from_json
use std::io::Read; // For load_pre_tokenized_from_json

// This is the main library file.
// We will add functions and structures here later.

pub mod config;
pub mod common;
pub mod accelerator;
pub mod attention;
pub mod mlp;
pub mod model;
pub mod tokenizer; // This will now load src/tokenizer.rs

// Remove the gpt2_tokenizer_tests module from here, as tests should be in their respective files.
// The tests for TokenizerWrapper are in src/tokenizer.rs.
// The tests I previously added for a GPT2Tokenizer in *this* file (lib.rs) were misplaced
// if the intention was to test TokenizerWrapper from src/tokenizer.rs.