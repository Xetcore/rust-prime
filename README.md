# Rust Transformers GPT-2 & Native CLI

## Overview

This project contains two distinct Rust codebases related to GPT-2 model inference:

1.  **`rust-native-transformer` (Primary Implementation)**: Located in the `rust-native-transformer/` directory, this is a pure Rust, "from scratch" implementation of GPT-2. It includes:
    *   A **native library** with core components like a custom tensor engine, BPE tokenizer, model architecture, and weight loader.
    *   A **Command-Line Interface (CLI)** application that uses this internal library to perform text generation.
    *   **Philosophy**: Aims for transparency, minimal dependencies (no external C/C++ or large ML frameworks for core logic), and a deep understanding of transformer internals.

2.  **Experimental `ndarray`-based Library**: Located in the root `src/` directory (e.g., `src/model.rs`, `src/tokenizer.rs`), this is a separate, more experimental library.
    *   It utilizes the `ndarray` crate for tensor operations.
    *   It uses the external `tokenizers` crate (Hugging Face's Rust library) for tokenization, rather than a native implementation.
    *   **Status**: This part is less developed and serves as an alternative exploration of implementing transformer components in Rust, relying on some established Rust crates.

**Key Distinction**: The `rust-native-transformer` CLI uses its *own* internal library modules (found within `rust-native-transformer/src/`) and is the primary, functional part of this project. The library in the root `src/` is a separate codebase with a different design philosophy and dependency set.

The overall philosophy is to explore and implement transformer models in Rust, with `rust-native-transformer` being the flagship example of a "from scratch" approach.

## Features

The **`rust-native-transformer`** CLI and its underlying native library support:

*   **`.safetensors` Model Loading**: Securely loads GPT-2 model weights.
    *   Native BPE Tokenizer**: Implements the Byte Pair Encoding tokenizer used by GPT-2 from scratch.
    *   GPT-2 Architecture**: Core components of the GPT-2 model (Multi-Head Attention, Feed-Forward Networks, Layer Normalization, KV Caching) are implemented natively.
    *   Custom Tensor Engine**: Basic tensor operations implemented in pure Rust.
    *   Greedy Decoding**: Text generation using greedy sampling.
    *   Pure Rust**: No external C/C++ dependencies for core model logic.

For more details on the `rust-native-transformer` CLI features, see its dedicated [rust-native-transformer/README.md](rust-native-transformer/README.md).

## Project Structure

```
.
├── DEPLOY.md                       # Detailed CLI deployment and usage instructions
├── LICENSE                         # Project license file
├── README.md                       # This file: Main project overview
├── resources/                      # Shared tokenizer data (vocab, merges) and model configs
│   ├── config/
│   │   └── gpt2/
│   │       └── config.json         # Example GPT-2 model configuration for `rust-native-transformer`
│   └── tokenizer_data/
│       └── gpt2/
│           ├── gpt2-vocab.json     # GPT-2 vocabulary (used by `rust-native-transformer`)
│           ├── merges.txt          # GPT-2 merges rules (used by `rust-native-transformer`)
│           └── sample_token_ids.json # Sample token IDs for testing
├── rust-native-transformer/        # Primary "from scratch" GPT-2 implementation (CLI & its own library)
│   ├── README.md                   # Detailed CLI-specific README
│   ├── Cargo.toml
│   └── src/                        # Source files for `rust-native-transformer`
│       ├── lib.rs                  # Library entry point for `rust-native-transformer` components
│       ├── main.rs                 # CLI entry point
│       ├── model_loader.rs         # Logic for loading `.safetensors` model weights
│       ├── resonance_feedback.rs   # System for collecting and storing user feedback on model generations
│       ├── runtime_interface.rs    # CLI argument parsing and runtime setup
│       ├── tensor_engine.rs        # Custom tensor operations
│       ├── text_generator.rs       # Text generation logic (e.g., greedy decoding)
│       ├── tokenizer_core.rs       # Native BPE tokenizer implementation
│       └── transformer_core.rs     # Core GPT-2 model architecture
└── src/                            # Experimental `ndarray`-based library
    ├── lib.rs                      # Library entry point
    ├── attention.rs                # Attention mechanism using ndarray
    ├── cache_tier.rs               # KV Caching components
    ├── common.rs                   # Common utilities
    ├── config.rs                   # Configuration handling
    ├── gating.rs                   # Gating mechanisms (e.g. for MoE)
    ├── main.rs                     # Experimental main/test binary for this library
    ├── mlp.rs                      # MLP layers using ndarray
    ├── model.rs                    # Main model structure using ndarray components
    ├── moe.rs                      # Mixture of Experts components
    ├── orchestrator.rs             # Higher-level coordination
    ├── repl.rs                     # REPL interface for experimentation
    ├── repl_feedback.rs            # Feedback mechanisms for REPL
    ├── system_resources.rs         # System resource monitoring
    └── tokenizer.rs                # Wrapper for external `tokenizers` crate
```
*Note: The root `src/` directory contains an experimental library using `ndarray` and the `tokenizers` crate, distinct from the self-contained `rust-native-transformer`.*

## Getting Started

### CLI Application (`rust-native-transformer`)

The primary way to use this project is through the **`rust-native-transformer`** CLI. This allows you to load a GPT-2 model (converted to `.safetensors` format) and generate text based on a prompt using the native Rust engine.

**For full build, setup, model preparation, and execution instructions, please refer to [DEPLOY.md](DEPLOY.md) or the [rust-native-transformer/README.md](rust-native-transformer/README.md).**

### Library Usage (Optional)

#### Using `rust-native-transformer` components
The core components of the `rust-native-transformer` (like its native tokenizer or model parts) can be used as a library if needed. Ensure your `Cargo.toml` points to its path:

```toml
# In your Cargo.toml
# [dependencies]
# rust_native_transformer = { path = "path/to/this/repo/rust-native-transformer" }
```

Then, you can use its modules:
```rust
// In your Rust code (conceptual example):
/*
use rust_native_transformer::tokenizer_core;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab_path = Path::new("resources/tokenizer_data/gpt2/gpt2-vocab.json");
    let merges_path = Path::new("resources/tokenizer_data/gpt2/merges.txt");

    let vocab = tokenizer_core::load_vocab(vocab_path)?;
    let merges = tokenizer_core::load_merges(merges_path)?;

    let text = "Hello world!";
    let token_ids = tokenizer_core::encode(text, &vocab, &merges)?;
    println!("'{}' tokenized to: {:?}", text, token_ids);

    // ... further operations ...
    Ok(())
}
*/
```
Refer to `rust-native-transformer/src/main.rs` for comprehensive usage of its internal library components.

#### Using the experimental `ndarray`-based library (root `src/`)
The library in the root `src/` directory is more experimental and relies on `ndarray` and the external `tokenizers` crate. To use it, you would typically set it up as a dependency similarly:
```toml
# In your Cargo.toml
# [dependencies]
# my_gpt2_experimental_lib = { path = "path/to/this/repo/src" } // Assuming `name` in src/Cargo.toml (if it exists)
```
Usage would then depend on its specific module structure (e.g., `use my_gpt2_experimental_lib::model::Model;`). This library is less mature than `rust-native-transformer`.

## Current Status & Future Work

The **`rust-native-transformer`** is a minimally viable GPT-2 inference engine demonstrating core concepts in a pure Rust environment. Key functionalities are in place. For a detailed list of its current limitations and planned future work, please see the "Limitations" and "Future Work" sections in the [rust-native-transformer/README.md](rust-native-transformer/README.md).

The **experimental library in the root `src/` directory** is at an earlier stage. It explores an alternative implementation strategy using `ndarray` and the `tokenizers` crate, and its development is ongoing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
