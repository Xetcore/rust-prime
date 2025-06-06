# Deploying the Rust Native Transformer CLI

## 1. Introduction

This document provides guidance on setting up, building, and running the `rust-native-transformer` Command Line Interface (CLI). This CLI allows users to perform text generation using GPT-2 models.

## 2. Prerequisites

*   **Rust and Cargo**: You need Rust and its package manager, Cargo, installed. If you don't have them, you can install them from [rustup.rs](https://rustup.rs/).

## 3. Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd <repository-name>       # Replace <repository-name> with the cloned directory name
    ```

2.  **Navigate to the Project Directory**:
    The CLI project is located within the `rust-native-transformer` subdirectory.
    ```bash
    cd rust-native-transformer
    ```

## 4. Building the CLI

To build the CLI in release mode (optimized for performance), run the following command from within the `rust-native-transformer` directory:

```bash
cargo build --release
```

The compiled executable will be located at `target/release/rust_native_transformer_cli`.

## 5. Obtaining and Preparing a GPT-2 Model

The `rust-native-transformer` CLI requires a pre-trained GPT-2 model in the `.safetensors` format.

*   **Sourcing Models**: You can find compatible GPT-2 models (e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl') on the [Hugging Face Hub](https://huggingface.co/models). Look for models that have `.safetensors` versions available or can be converted.

*   **Converting Models to `.safetensors`**:
    If your model is in a different format (like PyTorch's `.bin`), you can convert it to `.safetensors` using Python libraries. You'll typically need `transformers` and `safetensors`.

    Here's a conceptual example Python snippet:
    ```python
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_model
    import os

    # Choose the model name (e.g., "gpt2", "gpt2-medium")
    model_name = "gpt2"
    output_dir = "converted_models" # Or any directory you prefer
    output_path = os.path.join(output_dir, f"{model_name}.safetensors")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the model from Hugging Face
    print(f"Loading model '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save the model in .safetensors format
    print(f"Saving model to '{output_path}'...")
    save_model(model, output_path)

    print(f"Model converted and saved to {output_path}")
    ```
    Install necessary libraries with `pip install transformers safetensors torch`.

*   **Placing the Model File**:
    Once you have your `your_model.safetensors` file, place it in a known location. We recommend creating a `models/` directory inside the `rust-native-transformer` project directory and placing your model there.
    For example: `rust-native-transformer/models/your_model.safetensors`.

## 6. Running the CLI

After building the CLI and preparing your model, you can run it from the `rust-native-transformer` directory.

Here's an example command:

```bash
./target/release/rust_native_transformer_cli \
    --model-path ./models/your_model.safetensors \
    --vocab-path ../resources/tokenizer_data/gpt2/gpt2-vocab.json \
    --merges-path ../resources/tokenizer_data/gpt2/merges.txt \
    --prompt "Hello, world! This is a test prompt." \
    --max-length 50 \
    --config-n-layer 12 \
    --config-n-head 12 \
    --config-n-embd 768 \
    --config-vocab-size 50257 \
    --config-block-size 1024 \
    --eos-token-id 50256
```

**Important**:
*   Replace `./models/your_model.safetensors` with the actual path to your model file.
*   The paths to `vocab-path` and `merges-path` are relative to the `rust-native-transformer` directory, assuming the `resources` directory is at the repository root.
*   The model configuration parameters (`--config-n-layer`, `--config-n-head`, etc.) must be set directly on the command line and should match the architecture of the specific GPT-2 variant you are using. The example values are for the standard 'gpt2' base model. You can refer to the `resources/config/gpt2/config.json` file for typical values for different GPT-2 model sizes, but this file is **not** directly loaded by the CLI.
*   **Adjust Model Configuration Parameters**: The `--config-n-layer`, `--config-n-head`, etc., parameters **must match** the architecture of the specific GPT-2 variant you are using (e.g., 'gpt2', 'gpt2-medium'). The example values are for the standard 'gpt2' base model. Refer to the model's configuration (often a `config.json` file) on Hugging Face Hub for the correct values.

## 7. Troubleshooting/Notes

*   **Rust Version**: The codebase might utilize features available in newer stable versions of Rust. If you encounter build issues, ensure your Rust installation is up-to-date (`rustup update`).
*   **SIMD Optimizations**: The project originally included experimental SIMD (Single Instruction, Multiple Data) optimizations for performance. These have been temporarily disabled to ensure broader compatibility across different CPUs. Future versions may re-introduce stable SIMD support.
*   **Feedback Store**: A mechanism for storing and utilizing feedback on model generations was part of earlier designs. This feature has been temporarily commented out of the codebase to focus on core text generation functionality. It might be re-introduced or redesigned in future iterations.
```
