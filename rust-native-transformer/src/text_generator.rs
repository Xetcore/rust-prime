// src/text_generator.rs

use crate::transformer_core::{GPT2Model, TransformerError as ModelError};
use crate::tensor_engine::{Tensor, TensorError};
use crate::resonance_feedback::ResonanceFeedbackStore; // Added

// 1. TextGeneratorError Enum
#[derive(Debug)]
pub enum TextGeneratorError {
    ModelError(ModelError),
    TensorError(TensorError),
    InvalidInput(String),
    GenerationLimitReached(String), // If max_length is hit in a way that's an error condition elsewhere
}

impl std::fmt::Display for TextGeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextGeneratorError::ModelError(e) => write!(f, "Model error: {:?}", e), // Use {:?} if Display is not detailed
            TextGeneratorError::TensorError(e) => write!(f, "Tensor error: {:?}", e), // Same here
            TextGeneratorError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            TextGeneratorError::GenerationLimitReached(s) => write!(f, "Generation limit reached: {}", s),
        }
    }
}

impl std::error::Error for TextGeneratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TextGeneratorError::ModelError(ref e) => Some(e), // Assuming ModelError (TransformerError) impls Error
            TextGeneratorError::TensorError(ref e) => Some(e), // Assuming TensorError impls Error
            _ => None,
        }
    }
}

impl From<ModelError> for TextGeneratorError {
    fn from(err: ModelError) -> TextGeneratorError {
        TextGeneratorError::ModelError(err)
    }
}

impl From<TensorError> for TextGeneratorError {
    fn from(err: TensorError) -> TextGeneratorError {
        TextGeneratorError::TensorError(err)
    }
}

// 2. Local argmax helper function
fn argmax(data: &[f32]) -> Result<usize, TextGeneratorError> {
    if data.is_empty() {
        return Err(TextGeneratorError::InvalidInput("Cannot perform argmax on empty data slice".to_string()));
    }
    let mut max_val = data[0];
    let mut max_idx = 0;
    for (i, &val) in data.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    Ok(max_idx)
}

// 3. `generate` Function
pub fn generate(
    model: &GPT2Model,
    input_ids: Vec<u32>,
    max_length: usize,
    eos_token_id: u32,
    feedback_store: Option<&ResonanceFeedbackStore> // New parameter
) -> Result<Vec<u32>, TextGeneratorError> {
    // Placeholder for feedback usage
    if let Some(store) = feedback_store {
        // TODO: Future enhancement:
        // Query the feedback_store to potentially adjust generation strategy.
        // For example, retrieve recent positive/negative experiences
        // to influence logits or sampling parameters.
        let recent_feedback = store.get_recent_experiences(5); // Example query
        // Using a print statement for now to confirm it's accessible
        // In a real scenario, this might be logged or used more subtly.
        if !recent_feedback.is_empty() { // Only print if there's actually feedback
            println!("[TextGenerator] Feedback store available. Found {} recent entries (example).", recent_feedback.len());
        }
    }

    if input_ids.is_empty() {
        return Err(TextGeneratorError::InvalidInput("Input IDs cannot be empty.".to_string()));
    }
    if max_length == 0 {
        return Err(TextGeneratorError::InvalidInput("Max length must be greater than 0.".to_string()));
    }
    if input_ids.len() >= max_length {
        // Return a copy of input_ids if it's already at or beyond max_length
        return Ok(input_ids[..max_length.min(input_ids.len())].to_vec());
    }

    let mut generated_ids = input_ids.clone();

    for _ in 0..(max_length - input_ids.len()) {
        let current_sequence_length = generated_ids.len();
        
        // a. Prepare input tensor
        let input_tensor = Tensor::new(generated_ids.clone(), vec![1, current_sequence_length])?;

        // b. Call model.forward()
        // Logits tensor shape: `[1, current_sequence_length, vocab_size]`.
        let logits_tensor = model.forward(&input_tensor, None, None, None)?;

        // c. Extract logits for the *last token*
        // Logits for the last token are at the end of the `data` Vec.
        // If shape is [1, S, V], data is flat.
        // Logits for token s (0-indexed) start at index s * V.
        // We want logits for the last token, so s = current_sequence_length - 1.
        let vocab_size = model.config.vocab_size;
        let last_token_logits_start_idx = (current_sequence_length - 1) * vocab_size;
        let last_token_logits_end_idx = last_token_logits_start_idx + vocab_size;

        if logits_tensor.data.len() < last_token_logits_end_idx {
            return Err(TextGeneratorError::TensorError(TensorError::OutOfBounds(
                "Logits tensor data length is too small for extracting last token logits.".to_string()
            )));
        }
        let last_token_logits_slice = &logits_tensor.data[last_token_logits_start_idx..last_token_logits_end_idx];
        
        // d. Find the token ID with the highest probability (argmax)
        let next_token_id_usize = argmax(last_token_logits_slice)?;
        
        // e. Get the chosen `next_token_id` (as `u32`)
        let next_token_id = next_token_id_usize as u32; // Assuming vocab_size fits in u32

        // g. Append `next_token_id` to `generated_ids`.
        generated_ids.push(next_token_id);

        // f. If `next_token_id == eos_token_id`, break the loop.
        // Now checking after push, so EOS token is included in the output.
        if next_token_id == eos_token_id {
            break;
        }

        // h. If `generated_ids.len() >= max_length`, break the loop (already handled by loop condition).
        // This check is also implicitly handled by the main loop condition, but double check is fine.
        if generated_ids.len() >= max_length {
            break;
        }
    }

    Ok(generated_ids)
}


// 4. Unit Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer_core::{Config, GPT2Model}; // For creating a dummy model
    use std::collections::HashMap; // For dummy weights

    fn create_dummy_model_for_generation(vocab_size: usize, block_size: usize, n_embd: usize) -> GPT2Model {
        let config = Config {
            n_layer: 1,
            n_head: 1,
            n_embd,
            vocab_size,
            block_size,
            bias: true,
        };
        
        // Create minimal dummy weights
        let mut weights = HashMap::new();
        weights.insert("wte.weight".to_string(), Tensor::zeros(vec![vocab_size, n_embd]));
        weights.insert("wpe.weight".to_string(), Tensor::zeros(vec![block_size, n_embd]));
        
        weights.insert("h.0.attn.c_attn.weight".to_string(), Tensor::zeros(vec![n_embd, 3 * n_embd]));
        weights.insert("h.0.attn.c_attn.bias".to_string(), Tensor::zeros(vec![3 * n_embd]));
        weights.insert("h.0.attn.c_proj.weight".to_string(), Tensor::zeros(vec![n_embd, n_embd]));
        weights.insert("h.0.attn.c_proj.bias".to_string(), Tensor::zeros(vec![n_embd]));
        
        weights.insert("h.0.mlp.c_fc.weight".to_string(), Tensor::zeros(vec![n_embd, 4 * n_embd]));
        weights.insert("h.0.mlp.c_fc.bias".to_string(), Tensor::zeros(vec![4 * n_embd]));
        weights.insert("h.0.mlp.c_proj.weight".to_string(), Tensor::zeros(vec![4 * n_embd, n_embd]));
        weights.insert("h.0.mlp.c_proj.bias".to_string(), Tensor::zeros(vec![n_embd]));

        weights.insert("h.0.ln_1.weight".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("h.0.ln_1.bias".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("h.0.ln_2.weight".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("h.0.ln_2.bias".to_string(), Tensor::zeros(vec![n_embd]));
        
        weights.insert("ln_f.weight".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("ln_f.bias".to_string(), Tensor::zeros(vec![n_embd]));

        GPT2Model::new(config, weights).expect("Failed to create dummy GPT2Model for testing")
    }

    #[test]
    fn test_argmax_simple() {
        let data = vec![1.0, 0.0, 3.0, 2.0];
        assert_eq!(argmax(&data).unwrap(), 2);
        let data_neg = vec![-1.0, -5.0, -0.5];
        assert_eq!(argmax(&data_neg).unwrap(), 2);
    }

    #[test]
    fn test_argmax_empty() {
        let data: Vec<f32> = vec![];
        assert!(argmax(&data).is_err());
    }
    
    // More involved test: Mocking model.forward or using a real (dummy) model
    // This test relies on GPT2Model::forward producing logits of the correct shape.
    // The actual values in logits don't matter much for this test, only the argmax selection.
    #[test]
    fn test_generate_reaches_max_length() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        
        let input_ids = vec![1u32];
        let max_len = 3;
        let eos_id = 99; // Some ID that won't be generated by zero logits

        // We need to control what model.forward() returns for the last token logits.
        // Since we are using a real model with zero weights, it will likely always pick token 0.
        // This is fine for testing stopping conditions.

        let result = generate(&model, input_ids.clone(), max_len, eos_id, None);
        
        assert!(result.is_ok(), "Generation failed: {:?}", result.err());
        let generated_sequence = result.unwrap();
        assert_eq!(generated_sequence.len(), max_len);
        assert_eq!(generated_sequence[0], input_ids[0]); 
        // With zero weights, the model (after softmax) will likely produce uniform probabilities if biases are zero,
        // or argmax might consistently pick 0 if all logits are zero.
        // Let's assume it picks 0.
        assert_eq!(generated_sequence, vec![1, 0, 0]);
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        
        // To make this test deterministic for EOS, we'd need to mock model.forward
        // to return specific logits that make the EOS token the argmax.
        // This is complex without a mocking framework for the Tensor struct or GPT2Model.
        // For now, we'll assume a scenario where EOS might be generated.
        // If all logits are 0, argmax picks index 0. If eos_token_id is 0, it should stop.
        
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids = vec![1u32];
        let max_len = 5;
        let eos_id = 0; // Assume token 0 is EOS and will be generated by zero-weight model.

        let result = generate(&model, input_ids.clone(), max_len, eos_id, None);
        assert!(result.is_ok(), "Generation failed: {:?}", result.err());
        let generated_sequence = result.unwrap();
        
        // Expected: [1 (input), 0 (generated EOS)]
        assert_eq!(generated_sequence, vec![1, 0]);
        assert_eq!(generated_sequence.last().unwrap(), &eos_id);
        assert!(generated_sequence.len() < max_len);
    }

    #[test]
    fn test_generate_input_already_max_length() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        
        let input_ids = vec![1u32, 2, 3];
        let max_len = 3;
        let eos_id = 99;

        let result = generate(&model, input_ids.clone(), max_len, eos_id, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), input_ids);
    }

    #[test]
    fn test_generate_input_longer_than_max_length() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        
        let input_ids = vec![1u32, 2, 3, 4];
        let max_len = 3; // Shorter than input
        let eos_id = 99;

        let result = generate(&model, input_ids.clone(), max_len, eos_id, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1,2,3]); // Should truncate input
    }
    
    #[test]
    fn test_generate_empty_input_error() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids: Vec<u32> = vec![];
        let result = generate(&model, input_ids, 5, 0, None);
        assert!(matches!(result, Err(TextGeneratorError::InvalidInput(_))));
    }

    #[test]
    fn test_generate_max_length_zero() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids = vec![1u32];
        let max_len = 0;
        let eos_id = 99;

        let result = generate(&model, input_ids, max_len, eos_id, None);
        assert!(matches!(result, Err(TextGeneratorError::InvalidInput(msg)) if msg.contains("Max length must be greater than 0")));
    }

    #[test]
    fn test_generate_max_length_one_from_prompt_of_one() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        
        let input_ids = vec![1u32]; // Prompt of length 1
        let max_len = 1; // Max length is same as prompt length
        let eos_id = 99;

        let result = generate(&model, input_ids.clone(), max_len, eos_id, None);
        assert!(result.is_ok(), "Generation failed: {:?}", result.err());
        let generated_sequence = result.unwrap();
        // Expect only the input_ids back, as no new tokens should be generated.
        assert_eq!(generated_sequence, input_ids, "Should return prompt if max_length is same as prompt length");
        assert_eq!(generated_sequence.len(), 1);
    }

    #[test]
    fn test_generate_max_length_one_from_empty_prompt_is_error() {
        // This case is already covered by test_generate_empty_input_error
        // generate() returns Err if input_ids is empty.
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids: Vec<u32> = vec![];
        let max_len = 1;
        let eos_id = 99;
        let result = generate(&model, input_ids, max_len, eos_id, None);
        assert!(matches!(result, Err(TextGeneratorError::InvalidInput(_))));
    }
}
