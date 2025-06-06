// src/tensor_engine.rs

// Re-export core types from the new tensor_core_optimized crate.
// This allows other modules in the rust-native-transformer crate that previously
// depended on `crate::tensor_engine::Tensor` or `crate::tensor_engine::TensorError`
// to continue functioning without changing their use statements, as this module
// now acts as a facade for these types.
pub use tensor_core_optimized::{Tensor, TensorError};

// Any f32-specific or other functionality that was *not* moved to tensor_core_optimized
// but was part of the original tensor_engine's public API could remain here or be
// defined here. However, based on previous steps, most functionality including
// f32 impls and SIMD helpers were moved to tensor_core_optimized.

// The test module was also moved to tensor_core_optimized.
// This file is now primarily for re-exporting.
