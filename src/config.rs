use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize, Debug, Default)]
pub struct GPT2Config {
    pub vocab_size: i32,
    #[serde(alias = "max_position_embeddings")]
    pub n_positions: i32,
    #[serde(alias = "hidden_size")]
    pub n_embd: i32,
    #[serde(alias = "num_hidden_layers")]
    pub n_layer: i32,
    #[serde(alias = "num_attention_heads")]
    pub n_head: i32,
    pub n_inner: Option<i32>,
    pub activation_function: String,
    pub resid_pdrop: f32,
    pub embd_pdrop: f32,
    pub attn_pdrop: f32,
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub summary_type: String,
    pub summary_use_proj: bool,
    pub summary_activation: Option<String>,
    pub summary_proj_to_labels: Option<bool>, // Not typically in gpt2 config.json, but good to have
    pub summary_first_dropout: Option<f32>, // Not typically in gpt2 config.json
    pub scale_attn_weights: Option<bool>, // Made optional to handle missing field
    #[serde(alias = "output_attentions", alias = "output_hidden_states")] // use_cache is more specific for generation
    pub use_cache: Option<bool>, // Made optional to handle missing field
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub model_type: String,
    // Add any other fields that might be in various gpt2 config.json files
    // For example, some configs might have 'id2label' or 'label2id' which would be Maps
    // Or 'architectures' which would be Vec<String>
}

impl GPT2Config {
    pub fn load(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if !Path::new(config_path).exists() {
            return Err(format!("Config file not found at: {}", config_path).into());
        }

        let mut file = File::open(config_path)
            .map_err(|e| format!("Failed to open config file {}: {}", config_path, e))?;
        
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("Failed to read config file {}: {}", config_path, e))?;
        
        let config: GPT2Config = serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to deserialize JSON from {}: {}", config_path, e))?;
        
        Ok(config)
    }
}