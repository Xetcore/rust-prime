// src/model.rs
use ndarray::{s, Array, Array2, ArrayD, ArrayView1, Axis, Ix1, Ix2, IxDyn, ShapeError};
use safetensors::SafeTensors;
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
// Ensure TokenizerWrapper is used for the generate function's tokenizer argument
use crate::tokenizer::TokenizerWrapper; // Changed from tokenizers::Tokenizer

use crate::attention::MultiHeadAttention;
use crate::common::{KVCacheEntry, LayerKVCache, LayerNorm, ModelKVCache};
use crate::config::GPT2Config;
use crate::mlp::MLP;


#[derive(Debug)]
pub struct TransformerBlock {
    pub(crate) ln_1: LayerNorm,
    pub(crate) attn: MultiHeadAttention,
    pub(crate) ln_2: LayerNorm,
    pub(crate) mlp: MLP,
}

impl TransformerBlock {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let ln_1 = LayerNorm::new(config.n_embd, config.layer_norm_epsilon)?;
        let attn = MultiHeadAttention::new(config.n_head, config.n_embd)?;
        let ln_2 = LayerNorm::new(config.n_embd, config.layer_norm_epsilon)?;
        let n_inner = config.n_inner.unwrap_or(4 * config.n_embd); // GPT-2 default
        let mlp = MLP::new(config.n_embd, n_inner)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &ArrayD<f32>,
        attention_mask: Option<&ArrayD<f32>>,
        layer_kv_cache: Option<&mut LayerKVCache>,
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let ln_1_output = self.ln_1.forward(hidden_states)?;
        let attn_output = self
            .attn
            .forward(&ln_1_output, attention_mask, layer_kv_cache)?;
        let residual_1 = hidden_states + &attn_output; // Ensure ArrayD + &ArrayD is valid or adjust
        let ln_2_output = self.ln_2.forward(&residual_1)?;
        let mlp_output = self.mlp.forward(&ln_2_output)?;
        let output = &residual_1 + &mlp_output; // Ensure ArrayD + &ArrayD is valid
        Ok(output)
    }
}

#[derive(Debug)]
pub struct GPT2Model {
    pub(crate) wte_weight: ArrayD<f32>, // Token embeddings
    pub(crate) wpe_weight: ArrayD<f32>, // Positional embeddings
    pub(crate) h: Vec<TransformerBlock>,
    pub(crate) ln_f: LayerNorm,
    pub(crate) config: GPT2Config, // Store config
}

impl GPT2Model {
    pub fn new(config: GPT2Config) -> Result<Self, Box<dyn std::error::Error>> { // Take config by value
        let wte_weight =
            ArrayD::zeros(IxDyn(&[config.vocab_size as usize, config.n_embd as usize]));
        let wpe_weight =
            ArrayD::zeros(IxDyn(&[config.n_positions as usize, config.n_embd as usize]));

        let mut h = Vec::with_capacity(config.n_layer as usize);
        for _i in 0..config.n_layer {
            h.push(TransformerBlock::new(&config)?); // Pass config by reference
        }

        let ln_f = LayerNorm::new(config.n_embd, config.layer_norm_epsilon)?;

        Ok(Self {
            wte_weight,
            wpe_weight,
            h,
            ln_f,
            config, // Store the passed config
        })
    }

    pub fn forward(
        &self,
        input_ids: &Array2<i32>,
        attention_mask_option: Option<&ArrayD<f32>>,
        model_kv_cache: Option<&mut ModelKVCache>,
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];
        let current_seq_len = input_ids.shape()[1];
        let n_embd = self.wte_weight.shape()[1];

        let past_seq_len = match model_kv_cache.as_ref() {
            Some(cache) if !cache.is_empty() && !cache[0].is_empty() && !cache[0][0].key.is_empty() => {
                cache[0][0].key.shape()[1]
            }
            _ => 0,
        };

        if let Some(cache_ref) = model_kv_cache.as_mut() {
            if cache_ref.is_empty() {
                *cache_ref = vec![Vec::new(); self.h.len()];
            }
        }

        let mut token_embeddings_arr3 = Array::zeros((batch_size, current_seq_len, n_embd));
        for b in 0..batch_size {
            for s_idx in 0..current_seq_len {
                let token_id = input_ids[[b, s_idx]] as usize;
                if token_id >= self.wte_weight.shape()[0] {
                    return Err(format!(
                        "Token ID {} at [{},{}] is out of vocab size {}",
                        token_id,
                        b,
                        s_idx,
                        self.wte_weight.shape()[0]
                    )
                    .into());
                }
                let embedding_vector_view = self.wte_weight.slice(s![token_id, ..]);
                token_embeddings_arr3
                    .slice_mut(s![b, s_idx, ..])
                    .assign(&embedding_vector_view);
            }
        }
        let mut hidden_states = token_embeddings_arr3.into_dyn();

        let position_ids: Vec<usize> = (past_seq_len..past_seq_len + current_seq_len).collect();
        if !position_ids.is_empty() && *position_ids.last().unwrap_or(&0) >= self.wpe_weight.shape()[0] {
             return Err(format!(
                "Max position id ({}) exceeds maximum positional embeddings ({})",
                position_ids.last().unwrap_or(&0),
                self.wpe_weight.shape()[0]
            ).into());
        }

        if current_seq_len > 0 {
            let mut positional_embeddings_arr = Array::zeros((current_seq_len, n_embd));
            for (i, &pos_id) in position_ids.iter().enumerate() {
                let pos_embedding_slice = self.wpe_weight.slice(s![pos_id, ..]);
                positional_embeddings_arr
                    .slice_mut(s![i, ..])
                    .assign(&pos_embedding_slice);
            }
            let positional_embeddings_broadcastable = positional_embeddings_arr
                .into_shape((1, current_seq_len, n_embd))
                .map_err(|e: ShapeError| {
                    format!("Error reshaping positional_embeddings: {}", e.to_string())
                })?
                .into_dyn();
            hidden_states = hidden_states + positional_embeddings_broadcastable;
        }
        
        let mut current_model_kv_cache = model_kv_cache;
        for (block_idx, block) in self.h.iter().enumerate() {
            let layer_cache_opt = current_model_kv_cache
                .as_mut()
                .map(|cache| &mut cache[block_idx]);
            let effective_attention_mask = attention_mask_option;
            hidden_states =
                block.forward(&hidden_states, effective_attention_mask, layer_cache_opt)?;
        }

        hidden_states = self.ln_f.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    pub fn lm_head(
        &self,
        final_hidden_states: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let initial_shape = final_hidden_states.shape();
        if initial_shape.len() != 3 {
            return Err(format!(
                "Expected final_hidden_states to be 3D (batch, seq_len, n_embd), got shape: {:?}",
                initial_shape
            )
            .into());
        }
        let batch_size = initial_shape[0];
        let seq_len = initial_shape[1];
        let n_embd = initial_shape[2];
        let vocab_size = self.wte_weight.shape()[0];

        let wte_2d_view = self
            .wte_weight
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|e| {
                format!(
                    "Failed to view wte_weight as 2D: {}. Shape was {:?}",
                    e,
                    self.wte_weight.shape()
                )
            })?;
        let wte_transposed_view = wte_2d_view.t();

        let reshaped_hidden_states = final_hidden_states
            .view()
            .into_shape((batch_size * seq_len, n_embd))
            .map_err(|e: ShapeError| {
                format!(
                    "Error reshaping hidden_states: {}. Original shape: {:?}, Target shape: ({}, {})",
                    e,
                    initial_shape,
                    batch_size * seq_len,
                    n_embd
                )
            })?;

        let logits_2d = reshaped_hidden_states.dot(&wte_transposed_view);
        let original_logits_2d_shape = logits_2d.shape().to_vec();

        let logits = logits_2d
            .into_shape((batch_size, seq_len, vocab_size))
            .map_err(|e: ShapeError| {
                format!(
                    "Error reshaping logits: {}. Original shape: {:?}, Target shape: ({}, {}, {})",
                    e, original_logits_2d_shape, batch_size, seq_len, vocab_size
                )
            })?
            .into_dyn();
        Ok(logits)
    }

    pub fn load_weights(&mut self, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open(model_path)
            .map_err(|e| format!("Failed to open model file {}: {}", model_path, e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read model file {}: {}", model_path, e))?;
        
        let tensors_map = SafeTensors::deserialize(&buffer)
            .map_err(|e| format!("Failed to deserialize safetensors from buffer (path: {}): {}", model_path, e))?;

        self.wte_weight = load_specific_tensor(&tensors_map, "transformer.wte.weight", self.wte_weight.shape())?;
        self.wpe_weight = load_specific_tensor(&tensors_map, "transformer.wpe.weight", self.wpe_weight.shape())?;
        let n_embd_usize = self.wte_weight.shape()[1];

        for i in 0..self.h.len() {
            self.h[i].ln_1._weight = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.ln_1.weight", i), &[n_embd_usize])?;
            self.h[i].ln_1._bias = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.ln_1.bias", i), &[n_embd_usize])?;
            
            let n_inner_usize = self.h[i].mlp.c_fc_b.shape()[0];
            let loaded_fc_w = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.mlp.c_fc.weight", i), &[n_inner_usize, n_embd_usize])?;
            self.h[i].mlp.c_fc_w = loaded_fc_w.into_dimensionality::<Ix2>()?.t().into_owned().into_dyn();
            self.h[i].mlp.c_fc_b = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.mlp.c_fc.bias", i), &[n_inner_usize])?;
            
            let loaded_proj_w = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.mlp.c_proj.weight", i), &[n_embd_usize, n_inner_usize])?;
            self.h[i].mlp.c_proj_w = loaded_proj_w.into_dimensionality::<Ix2>()?.t().into_owned().into_dyn();
            self.h[i].mlp.c_proj_b = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.mlp.c_proj.bias", i), &[n_embd_usize])?;

            self.h[i].ln_2._weight = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.ln_2.weight", i), &[n_embd_usize])?;
            self.h[i].ln_2._bias = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.ln_2.bias", i), &[n_embd_usize])?;

            let loaded_attn_c_attn_w = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.attn.c_attn.weight", i), &[3 * n_embd_usize, n_embd_usize])?;
            self.h[i].attn.c_attn_w = loaded_attn_c_attn_w.into_dimensionality::<Ix2>()?.t().into_owned().into_dyn();
            self.h[i].attn.c_attn_b = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.attn.c_attn.bias", i), &[3 * n_embd_usize])?;
            self.h[i].attn.c_proj_w = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.attn.c_proj.weight", i), &[n_embd_usize, n_embd_usize])?;
            self.h[i].attn.c_proj_b = load_specific_tensor(&tensors_map, &format!("transformer.h.{}.attn.c_proj.bias", i), &[n_embd_usize])?;
        }
        self.ln_f._weight = load_specific_tensor(&tensors_map, "transformer.ln_f.weight", &[n_embd_usize])?;
        self.ln_f._bias = load_specific_tensor(&tensors_map, "transformer.ln_f.bias", &[n_embd_usize])?;
        Ok(())
    }

    pub fn generate(
        &self,
        tokenizer: &TokenizerWrapper, // Changed to TokenizerWrapper
        prompt_ids: &[i32],
        max_length: usize,
        eos_token_id: i32,
    ) -> Result<String, Box<dyn std::error::Error>> {
        if prompt_ids.is_empty() {
            return Err("Prompt cannot be empty.".into());
        }
        let mut current_token_sequence: Vec<i32> = prompt_ids.to_vec();
        if max_length <= current_token_sequence.len() {
            let token_ids_u32: Vec<u32> = current_token_sequence.iter().map(|&id| id as u32).collect();
            return tokenizer.decode(&token_ids_u32, true).map_err(|e| e.to_string().into());
        }

        let max_new_tokens = max_length - current_token_sequence.len();
        let mut model_kv_cache: ModelKVCache = Vec::new();

        if !current_token_sequence.is_empty() {
            let prefill_input_array = Array2::from_shape_vec((1, current_token_sequence.len()), current_token_sequence.clone())?;
            let prefill_hidden_states = self.forward(&prefill_input_array, None, Some(&mut model_kv_cache))?;
            let prefill_logits_dyn = self.lm_head(&prefill_hidden_states)?;
            let last_token_logits_view_dyn = prefill_logits_dyn.slice(s![0, current_token_sequence.len() - 1, ..]);
            let last_token_logits_view_1d = last_token_logits_view_dyn.view().into_dimensionality::<Ix1>()?;
            let predicted_token_idx = argmax(last_token_logits_view_1d);
            let predicted_token_id = predicted_token_idx as i32;
            current_token_sequence.push(predicted_token_id);
            if predicted_token_id == eos_token_id || current_token_sequence.len() >= max_length {
                let token_ids_u32: Vec<u32> = current_token_sequence.iter().map(|&id| id as u32).collect();
                return tokenizer.decode(&token_ids_u32, true).map_err(|e| e.to_string().into());
            }
        }
        
        for _ in 0..(max_new_tokens -1) {
            if current_token_sequence.is_empty() {
                return Err("Token sequence became empty during generation".into());
            }
            let last_predicted_token_id = *current_token_sequence.last().unwrap();
            let single_token_input_array = Array2::from_shape_vec((1, 1), vec![last_predicted_token_id])?;
            let hidden_states_gen = self.forward(&single_token_input_array, None, Some(&mut model_kv_cache))?;
            let logits_dyn_gen = self.lm_head(&hidden_states_gen)?;
            let next_token_logits_view_dyn = logits_dyn_gen.slice(s![0, 0, ..]);
            let next_token_logits_view_1d = next_token_logits_view_dyn.view().into_dimensionality::<Ix1>()?;
            let predicted_token_idx = argmax(next_token_logits_view_1d);
            let new_predicted_token_id = predicted_token_idx as i32;
            current_token_sequence.push(new_predicted_token_id);
            if new_predicted_token_id == eos_token_id || current_token_sequence.len() >= max_length {
                break;
            }
        }
        let token_ids_u32: Vec<u32> = current_token_sequence.iter().map(|&id| id as u32).collect();
        tokenizer.decode(&token_ids_u32, true).map_err(|e| e.to_string().into())
    }
}

fn argmax(array_view: ArrayView1<f32>) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in array_view.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    max_idx
}

fn load_specific_tensor(
    tensors: &SafeTensors,
    name: &str,
    expected_shape_usize: &[usize]
) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    let tensor_view = tensors.tensor(name)
        .map_err(|e| format!("Tensor '{}' not found: {}. Available: {:?}", name, e, tensors.names()))?;
    if tensor_view.shape() != expected_shape_usize {
        return Err(format!(
            "Shape mismatch for tensor '{}'. Expected {:?}, got {:?}.",
            name, expected_shape_usize, tensor_view.shape()
        ).into());
    }
    if tensor_view.dtype() != safetensors::Dtype::F32 {
        return Err(format!("Data type mismatch for '{}'. Expected F32, got {:?}.", name, tensor_view.dtype()).into());
    }
    let data_bytes = tensor_view.data();
    if data_bytes.len() != expected_shape_usize.iter().product::<usize>() * std::mem::size_of::<f32>() {
        return Err(format!("Data size mismatch for '{}'.", name).into());
    }
    let data_f32: Vec<f32> = data_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    ArrayD::from_shape_vec(IxDyn(expected_shape_usize), data_f32)
        .map_err(|e| format!("Failed to create ArrayD for '{}': {}", name, e).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::tokenizer::DUMMY_TOKENIZER_JSON; // For dummy tokenizer in generate test

    // Helper to create a test config
    fn create_test_config_for_model( // Renamed to avoid conflict if there was another one
        n_embd: i32, n_head: i32, n_layer: i32, vocab_size: i32, n_positions: i32
    ) -> GPT2Config {
        GPT2Config {
            vocab_size, n_layer, n_head, n_embd, n_positions,
            eos_token_id: 0, bos_token_id: 0, layer_norm_epsilon: 1e-5,
            n_inner: Some(4 * n_embd), activation_function: "gelu".to_string(),
            resid_pdrop: 0.1, embd_pdrop: 0.1, attn_pdrop: 0.1,
            initializer_range: 0.02, summary_type: "cls_index".to_string(),
            summary_use_proj: true, summary_activation: None,
            summary_proj_to_labels: None, summary_first_dropout: None,
            scale_attn_weights: Some(true), use_cache: Some(true),
            model_type: "gpt2".to_string(), ..Default::default()
        }
    }

    #[test]
    fn test_transformer_block_new_and_forward_shape() -> Result<(), Box<dyn std::error::Error>> {
        let config = create_test_config_for_model(4, 2, 1, 10, 10);
        let block = TransformerBlock::new(&config)?;
        let batch_size = 1; let seq_len = 5; let n_embd_usize = config.n_embd as usize;
        let hidden_states = ArrayD::zeros(IxDyn(&[batch_size, seq_len, n_embd_usize]));
        let output_no_cache = block.forward(&hidden_states, None, None)?;
        assert_eq!(output_no_cache.shape(), &[batch_size, seq_len, n_embd_usize]);
        
        let mut dummy_layer_kv_cache: LayerKVCache = Vec::new();
         for _ in 0..config.n_head {
             dummy_layer_kv_cache.push(KVCacheEntry {
                 key: ArrayD::zeros(IxDyn(&[batch_size, 0, n_embd_usize / config.n_head as usize])),
                 value: ArrayD::zeros(IxDyn(&[batch_size, 0, n_embd_usize / config.n_head as usize])),
             });
         }
        let output_with_cache = block.forward(&hidden_states, None, Some(&mut dummy_layer_kv_cache))?;
        assert_eq!(output_with_cache.shape(), &[batch_size, seq_len, n_embd_usize]);
        Ok(())
    }

    #[test]
    fn test_gpt2model_new() -> Result<(), Box<dyn std::error::Error>> {
        let config = create_test_config_for_model(8, 2, 2, 16, 20);
        let model = GPT2Model::new(config.clone())?; // Pass config by value
        assert_eq!(model.h.len(), config.n_layer as usize);
        assert_eq!(model.wte_weight.shape(), &[config.vocab_size as usize, config.n_embd as usize]);
        assert_eq!(model.wpe_weight.shape(), &[config.n_positions as usize, config.n_embd as usize]);
        Ok(())
    }


    #[test]
    fn test_gpt2model_forward_smoke() -> Result<(), Box<dyn std::error::Error>> {
        let config = create_test_config_for_model(8, 2, 2, 16, 20);
        let model = GPT2Model::new(config)?;
        let batch_size = 1; let seq_len = 5;
        let input_ids_data = (0..batch_size * seq_len).map(|i| (i % model.config.vocab_size) as i32).collect();
        let input_ids = Array2::from_shape_vec((batch_size, seq_len), input_ids_data)?;
        
        let output_no_cache = model.forward(&input_ids, None, None)?;
        assert_eq!(output_no_cache.shape(), &[batch_size, seq_len, model.config.n_embd as usize]);

        let mut model_kv_cache_data: ModelKVCache = Vec::new();
        let output_with_empty_cache = model.forward(&input_ids, None, Some(&mut model_kv_cache_data))?;
        assert_eq!(output_with_empty_cache.shape(), &[batch_size, seq_len, model.config.n_embd as usize]);
        assert_eq!(model_kv_cache_data.len(), model.config.n_layer as usize);
        Ok(())
    }

    #[test]
    fn test_gpt2_lm_head() -> Result<(), Box<dyn std::error::Error>> {
        let config = create_test_config_for_model(10, 2, 1, 50, 100);
        let mut model = GPT2Model::new(config)?;
        model.wte_weight = Array::ones((model.config.vocab_size as usize, model.config.n_embd as usize)).into_dyn();
        let batch_size = 2; let seq_len = 3; let n_embd_usize = model.config.n_embd as usize;
        let sample_hidden_states = ArrayD::ones(IxDyn(&[batch_size, seq_len, n_embd_usize]));
        let logits = model.lm_head(&sample_hidden_states)?;
        assert_eq!(logits.shape(), &[batch_size, seq_len, model.config.vocab_size as usize]);
        for val in logits.iter() {
            assert_abs_diff_eq!(*val, model.config.n_embd as f32, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_load_weights_and_generate() -> Result<(), Box<dyn std::error::Error>> {
        let config = GPT2Config { // Using actual config values for gpt2
            vocab_size: 50257, n_embd: 768, n_positions: 1024, n_layer: 12, n_head: 12,
            layer_norm_epsilon: 1e-5, eos_token_id: 50256, bos_token_id: 50256,
            n_inner: Some(4 * 768), activation_function: "gelu".to_string(),
             ..Default::default()
        };
        let mut model = GPT2Model::new(config.clone())?; // Clone config if needed later
        let model_path = "../../resources/model_data/gpt2/model.safetensors";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test_load_weights_and_generate: Model file not found at {}", model_path);
            return Ok(());
        }
        model.load_weights(model_path)?;

        // Use TokenizerWrapper with a dummy tokenizer for this test, as gpt2 files might not be available
        // or TokenizerWrapper itself might have issues from main crate.
        // This test primarily focuses on model.generate structure.
        let mut temp_file = tempfile::NamedTempFile::new().expect("Failed to create temporary file for test tokenizer");
        use std::io::Write;
        temp_file.write_all(DUMMY_TOKENIZER_JSON.as_bytes()).expect("Failed to write dummy tokenizer to temp file");
        let tokenizer_wrapper = TokenizerWrapper::new(temp_file.path())?;

        let prompt_text = "Hello, world";
        // Since we use DUMMY_TOKENIZER_JSON, the actual encoding doesn't matter as much as the flow.
        let prompt_ids: Vec<i32> = vec![1,2,3]; // Dummy IDs
        
        let max_length = prompt_ids.len() + 10;
        
        let generated_text = model.generate(&tokenizer_wrapper, &prompt_ids, max_length, config.eos_token_id)?;
        assert!(!generated_text.is_empty());
        // Length assertion might be tricky if EOS is generated early by actual model
        // For now, just check it's longer than prompt if it didn't hit EOS immediately.
        if !generated_text.ends_with("<|endoftext|>") || generated_text.split_whitespace().count() > prompt_text.split_whitespace().count() {
             assert!(generated_text.len() >= prompt_text.len());
        }
        Ok(())
    }
}