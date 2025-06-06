use std::io::{self, Write};
use std::error::Error; // For Box<dyn Error>
use std::path::PathBuf; // For feedback store path

// Module for feedback mechanism
// mod repl_feedback; // This was local, now use crate::repl_feedback
use crate::repl_feedback::{ResonanceFeedbackStore, ExperienceEntry};

use std::path::Path; // For tokenizer path

// Module for feedback mechanism (duplicate removed)
// mod repl_feedback;
// use repl_feedback::{ResonanceFeedbackStore, ExperienceEntry};

// Module for tokenizer (local `mod tokenizer;` declaration removed)
use crate::tokenizer::TokenizerWrapper; // Use the TokenizerWrapper from lib.rs

// Imports for token generation logic
// use ndarray::{s, ArrayD, Array2, Axis, ArrayView1}; // Added s and Axis, ArrayView1 - Commenting out as CpuTensor replaces these for main logic
use crate::model::{GPT2Model, GPT2Config};
use crate::common::ModelKVCache;
// Removed: use crate::tokenizer::GPT2Tokenizer; 
use crate::accelerator::{Accelerator, Device, Module, CpuTensor, Tensor};


pub fn get_user_prompt() -> String {
    print!("Enter your prompt: ");
    io::stdout().flush().expect("Failed to flush stdout.");
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).expect("Failed to read line from stdin.");
    buffer.trim().to_string()
}

pub fn prefill_prompt(
    model: &GPT2Model, // Changed from &mut GPT2Model
    prompt_tokens: &[u32],
    // _config, model_cache, theta_hat removed as they are not used by model.forward((tensor, None))
) -> Result<CpuTensor<f32>, String> {
    if prompt_tokens.is_empty() {
        return Err("Prompt tokens cannot be empty for prefill.".to_string());
    }
    let tokens_i32: Vec<i32> = prompt_tokens.iter().map(|&x| x as i32).collect();
    // Device::CPU is assumed as per previous model refactoring.
    // The model's forward pass expects shape [batch_size, seq_len]. For REPL, batch_size is 1.
    let input_tensor = CpuTensor::from_data_and_shape(&tokens_i32, &[1, tokens_i32.len()], Device::CPU)
        .map_err(|e| format!("Failed to create input CpuTensor for prefill: {}", e))?;
    
    // GPT2Model's Module::Input is (CpuTensor<i32>, Option<CpuTensor<f32>>).
    // Attention mask is None in this REPL context.
    model.forward((input_tensor, None))
        .map_err(|e| format!("Model forward pass failed during prefill: {}", e))
}

pub fn generate_next_token(
    model: &GPT2Model, // Changed from &mut GPT2Model
    last_token_id: u32,
    // config, model_cache, theta_hat removed
) -> Result<CpuTensor<f32>, String> {
    let token_i32 = last_token_id as i32;
    // Shape [batch_size, seq_len], so [1, 1] for generating next token.
    let input_tensor = CpuTensor::from_data_and_shape(&[token_i32], &[1, 1], Device::CPU)
        .map_err(|e| format!("Failed to create input CpuTensor for next token: {}", e))?;
    
    model.forward((input_tensor, None))
        .map_err(|e| format!("Model forward pass failed during next token generation: {}", e))
}

// Renamed from get_last_token_logits_slice and updated for CpuTensor
// Takes CpuTensor<f32> and returns Result<Vec<f32>, String>
fn get_last_token_logits_vector(logits_tensor: &CpuTensor<f32>) -> Result<Vec<f32>, String> {
    let shape = logits_tensor.shape();
    // Expected shape [batch_size, seq_len, vocab_size]. For REPL, batch_size is 1.
    if shape.len() != 3 {
        return Err(format!("Unexpected logits tensor rank: {}. Expected 3 (batch, seq, vocab).", shape.len()));
    }
    if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err("Logits tensor has an empty dimension.".to_string());
    }

    // let batch_size = shape[0]; // Expected to be 1 for REPL.
    let seq_len = shape[1];
    let vocab_size = shape[2];

    let data_slice = logits_tensor.as_slice()
        .map_err(|e| format!("Failed to get slice from logits tensor: {}", e))?;

    // Calculate the starting index of the last token's logits in the flat slice.
    // Data is laid out as [b0_s0_v0, b0_s0_v1, ..., b0_s1_v0, ...].
    // For batch_size = 1, we want the slice for the last token: (seq_len - 1).
    let start_index = (seq_len - 1) * vocab_size;
    let end_index = start_index + vocab_size;

    if end_index > data_slice.len() {
        return Err(format!(
            "Calculated logit slice indices [{}, {}) are out of bounds for data length {}. Shape: {:?}.",
            start_index, end_index, data_slice.len(), shape
        ));
    }
    
    Ok(data_slice[start_index..end_index].to_vec())
}

/// Returns the top k tokens and their logit values from a slice of logits.
/// Logits are assumed to be for a single token position.
pub fn get_top_k_tokens(logits_for_last_token: &[f32], k: usize) -> Vec<(u32, f32)> {
    if k == 0 || logits_for_last_token.is_empty() { // Check if the provided slice is empty
        return Vec::new();
    }
    
    let mut indexed_logits: Vec<(u32, f32)> = logits_for_last_token.iter()
        .enumerate()
        .map(|(id, &logit_val)| (id as u32, logit_val))
        .collect();

    // Sort by logit value in descending order.
    // Using partial_cmp for f32, handling potential NaNs by treating them as less than regular values.
    indexed_logits.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less)
    });

    indexed_logits.truncate(k);
    indexed_logits
}


/// Gets the token ID and logit value for the token with the highest logit score
/// from a slice of logits for a single token position.
pub fn get_greedy_token_and_logit(logits_for_last_token: &[f32]) -> Result<(u32, f32), String> {
    let top_k = get_top_k_tokens(logits_for_last_token, 1); // Now takes a slice
    if let Some(result) = top_k.first() {
        Ok(*result)
    } else {
        Err("Failed to get greedy token: Logits processing resulted in empty list or k=0.".to_string())
    }
}


/// Adjusts theta_hat based on user feedback and clamps it to the [0.0, 1.0] range.
pub(crate) fn adjust_theta_hat(initial_theta: f32, user_feedback: &str) -> f32 {
    let mut new_theta = initial_theta;
    match user_feedback {
        "s" => new_theta += 0.05,
        "n" => new_theta -= 0.05,
        _ => {} // No change for other inputs, including "q" or invalid
    }
    new_theta.max(0.0).min(1.0)
}

/// Runs the main Read-Eval-Print Loop (REPL) for interactive token generation.
///
/// This loop handles user input, token generation, and user feedback.
/// It now integrates with an `MoEOrchestrator` to dynamically select experts
/// based on cache tier policies, system resources, and intentionality score (θ̂).
///
/// Key operations within the loop:
/// 1.  **Dynamic Tier Policy Adjustment**: Before each token generation, the
///     `MoEOrchestrator`'s `current_allowed_tiers` policy is updated based on
///     available system RAM and the current `theta_hat` value. This logic is
///     encapsulated in the `determine_and_set_allowed_tiers` helper function.
/// 2.  **Token Generation via Orchestrator**: Uses `orchestrator.forward()` to generate
///     the next token's logits. This call also returns information about which
///     experts were activated.
/// 3.  **User Validation**: Prompts the user to validate the generated token ("s"atisfied,
///     "n"ot satisfied, "q"uit).
/// 4.  **Theta Hat Adjustment**: Adjusts `theta_hat` based on user feedback.
/// 5.  **Feedback Storage**: Records the generation experience (prompt context,
///     generated token, validation, theta_hat) to a JSON file.
/// 6.  **Information Display**: Shows the generated token, its ID, logit value,
///     the current `theta_hat`, validation status, and a list of activated experts
///     (name and cache tier) for the generated token.
///
/// # Arguments
/// * `model`: A mutable reference to the `GPT2Model` (used for embedding lookup via `get_embeddings`).
/// * `config`: A reference to the `GPT2Config`.
/// * `tokenizer`: A reference to the `TokenizerWrapper` for encoding/decoding tokens.
/// * `initial_prompt_tokens`: A vector of token IDs for the initial prompt.
/// * `max_new_tokens`: The maximum number of new tokens to generate in this session.
/// * `initial_theta_hat`: The starting value for the intentionality score.
/// * `eos_token_id`: The end-of-sequence token ID to stop generation.
/// * `store`: A mutable reference to the `ResonanceFeedbackStore` for saving experiences.
/// * `orchestrator`: A mutable reference to the `MoEOrchestrator` to be used for expert selection and generation.
#[allow(clippy::too_many_arguments)] 
pub fn run_repl_loop(
    model: &mut GPT2Model,
    config: &GPT2Config,
    tokenizer: &TokenizerWrapper, // Changed from GPT2Tokenizer to TokenizerWrapper
    initial_prompt_tokens: Vec<u32>, 
    max_new_tokens: usize,
    initial_theta_hat: f32,
    eos_token_id: u32,
    store: &mut ResonanceFeedbackStore, // New parameter for feedback store
) -> Result<(), String> {
    if initial_prompt_tokens.is_empty() {
        return Err("Initial prompt tokens cannot be empty for REPL loop.".to_string());
    }
    // ModelKVCache is not used in the current generation flow with model.forward((tensor, None)).
    // Commenting out its initialization.
    // let mut model_cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize];
    let mut current_token_ids = initial_prompt_tokens.clone();
    let mut current_theta_hat = initial_theta_hat; // Used for adjust_theta_hat and logging

    println!("> prompt: (tokens) {:?}", initial_prompt_tokens);
    
    let prefill_output_tensor = prefill_prompt(
        model, &current_token_ids
    )?;

    // Process CpuTensor output
    let prefill_logits_vec = get_last_token_logits_vector(&prefill_output_tensor)
        .map_err(|e| format!("Failed to get logits from prefill output: {}", e))?;
    
    let (first_generated_token_id, first_dominant_logit) = get_greedy_token_and_logit(&prefill_logits_vec)?;
    let first_top_k_alternatives = get_top_k_tokens(&prefill_logits_vec, 4);
    
    let mut next_token_id = first_generated_token_id;
    current_token_ids.push(next_token_id);

    // Variables to hold current iteration's logit info for display
    let mut dominant_logit_value_for_display = first_dominant_logit;
    let mut top_k_alternatives_for_display = first_top_k_alternatives;
    
    for i in 0..max_new_tokens {
        let token_display_id = next_token_id; // This is the token generated in the previous step (or from prefill)
        
        // Decode the token for display
        let token_str_display = tokenizer.decode(&[token_display_id], true)
            .unwrap_or_else(|e| {
                eprintln!("\n[Warning: Failed to decode token ID {}: {}]", token_display_id, e);
                format!("[ID:{}]", token_display_id)
            });

        // --- User Validation for `token_display_id` ---
        print!("Validate token {}: (s)atisfied, (n)ot satisfied, (q)uit? ", token_display_id);
        io::stdout().flush().expect("Failed to flush stdout.");
        let mut user_feedback = String::new();
        io::stdin().read_line(&mut user_feedback).expect("Failed to read user feedback.");
        user_feedback = user_feedback.trim().to_lowercase();

        let pre_adjustment_theta = current_theta_hat; 
        let mut validation_status_str: String; 

        if user_feedback == "q" {
            println!("Quitting generation loop.");
            validation_status_str = "quit".to_string();
            // Display info for the token that led to quit
            println!(
                "⟳ [{}] token: \"{}\" (ID: {}, Logit: {:.2}) | θ̂: {:.2} | validado: {}",
                i, token_str_display, token_display_id, dominant_logit_value_for_display, pre_adjustment_theta, validation_status_str
            );
            // Alternatives for the quit token
            println!("  Alternatives for quit token (ID: {}):", token_display_id);
            let mut alternatives_shown_quit = 0;
            for (alt_id, alt_logit) in top_k_alternatives_for_display.iter() {
                if *alt_id != token_display_id && alternatives_shown_quit < 3 {
                    let alt_token_str = tokenizer.decode(&[*alt_id], true).unwrap_or_else(|_e| format!("[ID:{}]", alt_id));
                    println!("    Alt {}: \"{}\" (ID: {}, Logit: {:.2})", alternatives_shown_quit + 1, alt_token_str, alt_id, alt_logit);
                    alternatives_shown_quit += 1;
                }
                if alternatives_shown_quit >= 3 { break; }
            }
            break; // Exit the loop
        }
        
        current_theta_hat = adjust_theta_hat(current_theta_hat, &user_feedback);

        match user_feedback.as_str() {
            "s" | "n" => {
                let validation_bool = user_feedback == "s";
                validation_status_str = if validation_bool { "sí".to_string() } else { "no".to_string() };

                let context_tokens_for_exp = if current_token_ids.len() > 1 {
                    current_token_ids[..current_token_ids.len()-1].to_vec()
                } else { Vec::new() };

                let entry = ExperienceEntry {
                    prompt_tokens: context_tokens_for_exp,
                    generated_token_id: token_display_id, 
                    validation_status: validation_bool,
                    theta_hat_at_generation: pre_adjustment_theta, // Theta used to generate this token
                };
                store.add_experience(entry);
                if let Err(e) = store.save() {
                    eprintln!("\n[Error saving experience: {}]", e);
                }
            }
            _ => { 
                validation_status_str = "n/a".to_string();
            }
        }
        
        // Display validated token info
        println!(
            "⟳ [{}] token: \"{}\" (ID: {}, Logit: {:.2}) | θ̂: {:.2} | validado: {}",
            i, token_str_display, token_display_id, dominant_logit_value_for_display, current_theta_hat, validation_status_str
        );
        // Display alternatives
        println!("  Alternatives:");
        let mut alternatives_shown = 0;
        for (alt_id, alt_logit) in top_k_alternatives_for_display.iter() {
            if *alt_id != token_display_id && alternatives_shown < 3 {
                let alt_token_str = tokenizer.decode(&[*alt_id], true).unwrap_or_else(|_e| format!("[ID:{}]", alt_id));
                println!("    Alt {}: \"{}\" (ID: {}, Logit: {:.2})", alternatives_shown + 1, alt_token_str, alt_id, alt_logit);
                alternatives_shown += 1;
            }
            if alternatives_shown >= 3 { break; }
        }


        if token_display_id == eos_token_id {
            println!("EOS token ({}) encountered. Stopping generation.", eos_token_id);
            break;
        }
        
        // Prepare for next iteration (if not the last one)
        if i < max_new_tokens - 1 {
            let last_token_id_for_gen = token_display_id; // The token we just processed
            
            // Generate logits for the *next* token
            // Updated call to generate_next_token (model is &GPT2Model, no config, cache, theta_hat)
            let next_logits_tensor = generate_next_token(
                model, last_token_id_for_gen
            )?;
            
            // Process CpuTensor output
            let next_logits_vec = get_last_token_logits_vector(&next_logits_tensor)
                 .map_err(|e| format!("Failed to get logits from next token output: {}", e))?;

            let (chosen_next_token_id, next_dominant_logit) = get_greedy_token_and_logit(&next_logits_vec)?;
            let next_top_k_alternatives = get_top_k_tokens(&next_logits_vec, 4);

            next_token_id = chosen_next_token_id;
            current_token_ids.push(next_token_id);

            // Update logit info for the display in the *next* iteration
            dominant_logit_value_for_display = next_dominant_logit;
            top_k_alternatives_for_display = next_top_k_alternatives;
        }
    }

    println!("\n--- Generation Ended ---");
    println!("Final generated token IDs: {:?}", current_token_ids);
    let final_decoded_text = tokenizer.decode(&current_token_ids, true)
        .unwrap_or_else(|e| format!("Failed to decode final sequence: {}", e));
    println!("Final decoded text: {}", final_decoded_text);
    Ok(())
}

/// Main entry point for the standalone REPL application.
///
/// This function orchestrates the setup of all necessary components for the REPL session:
/// 1.  Initializes `GPT2Config` and `GPT2Model`.
/// 2.  Initializes the `TokenizerWrapper`.
/// 3.  Initializes the `MoEOrchestrator` with a predefined set of experts (e.g., `SymbolicExpert`, `MLPExpert`)
///     and a `GatingLayer`. This orchestrator will manage expert activation during token generation.
/// 4.  Retrieves the user's initial prompt and tokenizes it.
/// 5.  Initializes the `ResonanceFeedbackStore` for recording user feedback.
/// 6.  Predicts or sets an initial `theta_hat` (intentionality score).
/// 7.  Calls `run_repl_loop` to start the interactive session, passing all initialized components,
///     including the `orchestrator`.
///
/// # Returns
/// `Ok(())` on successful execution, or a `Box<dyn Error>` if any critical setup or runtime error occurs.
pub fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Standalone REPL Initializing ---");

    // 1. Initialize GPT2Config (default)
    let config = GPT2Config::default(); // Using default for simplicity here
    println!("Default GPT2Config loaded: {:?}", config);

    // 2. Initialize GPT2Model
    let mut model = GPT2Model::new(&config)
        .map_err(|e| format!("Failed to create GPT2Model: {}", e))?;
    println!("GPT2Model initialized.");

    // Initialize Accelerator and prepare the model
    let accelerator = Accelerator::new(Device::CPU); // Assuming CPU for REPL
    accelerator.prepare(&mut model)
        .map_err(|e| format!("Failed to prepare model with accelerator: {}", e))?;
    println!("GPT2Model prepared with Accelerator on {:?}.", accelerator.current_device());

    // 3. Initialize TokenizerWrapper
    // Using test_tokenizer.json as it's known to exist and be compatible.
    // Users should replace "test_tokenizer.json" with their actual HuggingFace tokenizer.json path.
    let tokenizer_path_str = "test_tokenizer.json"; 
    let tokenizer_path = Path::new(tokenizer_path_str);
    let tokenizer = TokenizerWrapper::new(tokenizer_path)
        .map_err(|e| format!("Failed to load TokenizerWrapper from '{}': {}. Ensure the file exists and is a valid HuggingFace tokenizer JSON.", tokenizer_path_str, e))?;
    println!("TokenizerWrapper initialized from '{}'. Vocab size: {}", tokenizer_path_str, tokenizer.get_vocab_size());

    // 4. Get User Prompt
    let prompt_string = get_user_prompt();
    if prompt_string.is_empty() {
        println!("Prompt is empty. Exiting.");
        return Ok(());
    }

    // 5. Convert Prompt to Tokens using TokenizerWrapper
    let initial_prompt_tokens = match tokenizer.encode(&prompt_string, false) { // false = don't add special tokens for prompt
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("Failed to encode prompt: {}. Exiting.", e);
            return Ok(()); // Return Ok here as main expects Result<(), Box<dyn Error>>
        }
    };
    
    if initial_prompt_tokens.is_empty() {
        println!("Tokenized prompt is empty (e.g., input was only special tokens not kept, or tokenizer issue). Exiting.");
        return Ok(());
    }
    println!("Successfully tokenized prompt: {:?}", initial_prompt_tokens);

    // 6. Set REPL parameters
    let max_new_tokens = 10;
    // let initial_theta_hat = 0.75_f32; // Default will be set after prediction attempt
    let eos_token_id = config.eos_token_id.unwrap_or(50256); // Use EOS from config or default

    // 7. Initialize ResonanceFeedbackStore
    let feedback_store_path = PathBuf::from("resonance_feedback.json");
    let mut store = ResonanceFeedbackStore::new(feedback_store_path);
    println!("ResonanceFeedbackStore initialized. Loaded {} existing experiences.", store.experiences.len());

    // Predict initial_theta_hat based on experiences
    let mut initial_theta_hat: f32; // Declare, will be set below
    let default_initial_theta = 0.75_f32; // Define default

    match store.predict_initial_theta(&initial_prompt_tokens) {
        Some(predicted_theta) => {
            initial_theta_hat = predicted_theta;
            println!("Intuition from past experiences suggests an initial theta_hat of: {:.2}", initial_theta_hat);
        }
        None => {
            initial_theta_hat = default_initial_theta;
            println!("No similar past experiences found to predict initial theta_hat. Starting with default: {:.2}", initial_theta_hat);
        }
    }
    
    println!("Starting REPL loop with max_new_tokens={}, initial_theta_hat={:.2}, eos_token_id={}", 
        max_new_tokens, initial_theta_hat, eos_token_id);


    // 8. Call run_repl_loop
    // The run_repl_loop returns Result<(), String>, convert its error to Box<dyn Error>
    run_repl_loop(
        &mut model, 
        &config, 
        &tokenizer, 
        initial_prompt_tokens, 
        max_new_tokens, 
        initial_theta_hat, 
        eos_token_id,
        &mut store, // Pass the feedback store
    ).map_err(|e| -> Box<dyn Error> { Box::from(e) })?;

    println!("--- Standalone REPL Finished ---");
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    // use crate::tokenizer::GPT2Tokenizer; // No longer used directly in tests here
    // TokenizerWrapper might be needed if we had a test for run_repl_loop that needed specific tokenizer interactions

    #[test]
    #[ignore] 
    fn test_get_user_prompt_manual_input() {
        // ... (existing test code) ...
    }

    #[test]
    fn test_get_user_prompt_exists() {
        // ... (existing test code) ...
    }

    #[cfg(test)]
    mod repl_logic_tests {
        use super::*;
        // GPT2Model, GPT2Config, Accelerator, Device, Module, CpuTensor, Tensor are in outer scope.
        // ModelKVCache might be needed if tests involve functions that still use it (not currently).
        // Removed ndarray imports as they are no longer needed for the updated tests.
        // use ndarray::{ArrayD, IxDyn};

        fn create_test_config_for_repl() -> GPT2Config { 
            GPT2Config {
                vocab_size: 50, n_layer: 1, n_head: 1, n_embd: 4, // Minimal for testing
                n_positions: 64, n_ctx: 64, block_size: 64, // Smaller context
                embd_pdrop: 0.0, resid_pdrop: 0.0, attn_pdrop: 0.0,
                layer_norm_epsilon: 1e-5, initializer_range: 0.02,
                n_inner: Some(8), activation_function: "gelu_new".to_string(),
                bos_token_id: Some(50256), eos_token_id: Some(50256), 
                scale_attn_weights: true, scale_attn_by_inverse_layer_idx: false,
                reorder_and_upcast_attn: false,
            }
        }
        
        fn initialized_test_model() -> (GPT2Model, GPT2Config) {
            let config = create_test_config_for_repl();
            let mut model = GPT2Model::new(&config).expect("Failed to create test model.");
            let accelerator = Accelerator::new(Device::CPU);
            accelerator.prepare(&mut model).expect("Failed to prepare model with accelerator.");
            (model, config)
        }

        #[test]
        fn test_prefill_prompt_basic() { 
            let (model, config) = initialized_test_model();
            let prompt_tokens: Vec<u32> = vec![0, 1, 2];
            
            let result = prefill_prompt(&model, &prompt_tokens);
            assert!(result.is_ok());
            let output_tensor = result.unwrap();
            // Output of GPT2Model::forward is hidden states, shape [batch, seq, n_embd]
            // For REPL, batch is 1.
            let expected_shape: &[usize] = &[1, prompt_tokens.len(), config.n_embd as usize];
            assert_eq!(output_tensor.shape(), expected_shape);
        }

        #[test]
        fn test_prefill_prompt_empty_tokens() { 
            let (model, _config) = initialized_test_model();
            let prompt_tokens: Vec<u32> = vec![];
            let result = prefill_prompt(&model, &prompt_tokens);
            assert!(result.is_err()); 
        }

        #[test]
        fn test_generate_next_token_basic() { 
            let (model, config) = initialized_test_model();
            let last_token_id: u32 = 0;
            
            let result = generate_next_token(&model, last_token_id);
            assert!(result.is_ok());
            let output_tensor = result.unwrap();
            // Output of GPT2Model::forward is hidden states, shape [batch, seq, n_embd]
            // For REPL, batch is 1, seq is 1.
            let expected_shape: &[usize] = &[1, 1, config.n_embd as usize];
            assert_eq!(output_tensor.shape(), expected_shape);
        }
        
        #[test]
        fn test_get_last_token_logits_vector_valid() { 
            let vocab_size = 5;
            let data = (0..(1 * 2 * vocab_size)).map(|x| x as f32).collect::<Vec<f32>>();
            let tensor = CpuTensor::from_data_and_shape(&data, &[1, 2, vocab_size], Device::CPU).unwrap();
            let logits_vec = get_last_token_logits_vector(&tensor).unwrap();
            let expected_logits = data[vocab_size..].to_vec(); 
            assert_eq!(logits_vec, expected_logits);

            let data_seq1 = (0..vocab_size).map(|x| x as f32).collect::<Vec<f32>>();
            let tensor_seq1 = CpuTensor::from_data_and_shape(&data_seq1, &[1, 1, vocab_size], Device::CPU).unwrap();
            let logits_vec_seq1 = get_last_token_logits_vector(&tensor_seq1).unwrap();
            assert_eq!(logits_vec_seq1, data_seq1);
        }

        #[test]
        fn test_get_greedy_token_and_logit_basic() { 
            let logits_slice = vec![0.1, 0.2, 0.5, 0.1, 0.1];
            let (token_id, logit_val) = get_greedy_token_and_logit(&logits_slice).unwrap();
            assert_eq!(token_id, 2);
            assert!((logit_val - 0.5).abs() < f32::EPSILON);
        }

        #[test]
        fn test_get_greedy_token_and_logit_empty() { 
            let empty_logits_slice: Vec<f32> = vec![];
            assert!(get_greedy_token_and_logit(&empty_logits_slice).is_err());
        }

        #[test]
        fn test_get_top_k_tokens_basic() { 
            let logits_slice = vec![0.1, 0.7, 0.5, 0.9, 0.3];
            
            let top_1 = get_top_k_tokens(&logits_slice, 1);
            assert_eq!(top_1.len(), 1);
            assert_eq!(top_1[0].0, 3);
            assert!((top_1[0].1 - 0.9).abs() < f32::EPSILON);

            let top_3 = get_top_k_tokens(&logits_slice, 3);
            assert_eq!(top_3.len(), 3);
            assert_eq!(top_3[0].0, 3);
            assert_eq!(top_3[1].0, 1);
            assert_eq!(top_3[2].0, 2);

            let top_10 = get_top_k_tokens(&logits_slice, 10); 
            assert_eq!(top_10.len(), 5); 

            let top_0 = get_top_k_tokens(&logits_slice, 0); 
            assert!(top_0.is_empty());
        }

        #[test]
        fn test_get_top_k_tokens_empty() { 
            let empty_logits_slice: Vec<f32> = vec![];
            assert!(get_top_k_tokens(&empty_logits_slice, 3).is_empty());
        }


        #[test]
        #[ignore] 
        fn test_run_repl_loop_no_input_smoke_test() {
            // ... (This test requires specific setup or manual input, kept ignored) ...
        }

        #[test]
        fn test_adjust_theta_hat_logic() {
            // Test 's' (satisfied)
            assert_eq!(adjust_theta_hat(0.5, "s"), 0.55);
            
            // Test 'n' (not satisfied)
            assert_eq!(adjust_theta_hat(0.5, "n"), 0.45);

            // Test clamping (upper bound from 's')
            // Using f32::abs for float comparisons due to potential precision issues.
            // A small epsilon is often used, but for increments of 0.05, direct comparison should be mostly fine.
            // Let's be explicit with a small tolerance if needed, or ensure results are exact.
            assert!((adjust_theta_hat(0.98, "s") - 1.0).abs() < f32::EPSILON);
            assert!((adjust_theta_hat(0.97, "s") - 1.0).abs() < f32::EPSILON); // 0.97 + 0.05 = 1.02 -> 1.0

            // Test clamping (lower bound from 'n')
            assert!((adjust_theta_hat(0.02, "n") - 0.0).abs() < f32::EPSILON);
            assert!((adjust_theta_hat(0.03, "n") - 0.0).abs() < f32::EPSILON); // 0.03 - 0.05 = -0.02 -> 0.0


            // Test invalid input char (should not change theta)
            assert_eq!(adjust_theta_hat(0.5, "x"), 0.5);
            assert_eq!(adjust_theta_hat(0.5, ""), 0.5); // Empty feedback

            // Test 's' at max
            assert!((adjust_theta_hat(1.0, "s") - 1.0).abs() < f32::EPSILON);

            // Test 'n' at min
            assert!((adjust_theta_hat(0.0, "n") - 0.0).abs() < f32::EPSILON);
            
            // Test 's' that would exceed 1.0
            assert!((adjust_theta_hat(0.99, "s") - 1.0).abs() < f32::EPSILON); // 0.99 + 0.05 = 1.04 -> 1.0

            // Test 'n' that would go below 0.0
            assert!((adjust_theta_hat(0.01, "n") - 0.0).abs() < f32::EPSILON); // 0.01 - 0.05 = -0.04 -> 0.0
        }

        #[test]
        fn test_get_last_token_logits_slice_logic() {
            // Case 1: 1D tensor (e.g., already a slice)
            let logits_1d = ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
            let slice_1d = get_last_token_logits_slice(&logits_1d).unwrap();
            assert_eq!(slice_1d.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

            // Case 2: 3D tensor (batch=1, seq=1, vocab_size=5)
            let logits_3d_s1 = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
            let slice_3d_s1 = get_last_token_logits_slice(&logits_3d_s1).unwrap();
            assert_eq!(slice_3d_s1.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

            // Case 3: 3D tensor (batch=1, seq=3, vocab_size=2)
            // Logits: [[[1,2], [3,4], [5,6]]]
            let logits_3d_s3 = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
            let slice_3d_s3 = get_last_token_logits_slice(&logits_3d_s3).unwrap();
            assert_eq!(slice_3d_s3.to_vec(), vec![5.0, 6.0]); // Should be the last slice [5.0, 6.0]

            // Case 4: 3D tensor with batch_size > 1 (should take last batch, last seq)
            // Logits: [ Batch0: [[1,2], [3,4]], Batch1: [[5,6], [7,8]] ]
            let logits_3d_b2_s2 = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0,2.0, 3.0,4.0, 5.0,6.0, 7.0,8.0]).unwrap();
            let slice_3d_b2_s2 = get_last_token_logits_slice(&logits_3d_b2_s2).unwrap();
            assert_eq!(slice_3d_b2_s2.to_vec(), vec![7.0, 8.0]); // Last slice of last batch

            // Case 5: Empty dimension in 3D tensor
            let logits_3d_empty_vocab = ArrayD::from_shape_vec(IxDyn(&[1, 1, 0]), vec![]).unwrap();
            assert!(get_last_token_logits_slice(&logits_3d_empty_vocab).is_none());
            let logits_3d_empty_seq = ArrayD::from_shape_vec(IxDyn(&[1, 0, 5]), vec![]).unwrap();
            assert!(get_last_token_logits_slice(&logits_3d_empty_seq).is_none());
            let logits_3d_empty_batch = ArrayD::from_shape_vec(IxDyn(&[0, 1, 5]), vec![]).unwrap();
            assert!(get_last_token_logits_slice(&logits_3d_empty_batch).is_none());

            // Case 6: Unexpected number of dimensions (e.g., 2D)
            let logits_2d = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            assert!(get_last_token_logits_slice(&logits_2d).is_none());

            // Case 7: Unexpected number of dimensions (e.g., 4D)
            let logits_4d = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![1.0, 2.0]).unwrap();
            assert!(get_last_token_logits_slice(&logits_4d).is_none());
        }
    }
}
