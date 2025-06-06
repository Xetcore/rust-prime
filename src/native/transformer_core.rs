// src/transformer_core.rs

use crate::tensor_engine::{Tensor, TensorError};
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::Arc;

// 0. Basic Setup
#[derive(Debug)]
use std::sync::Arc;

// 0. Basic Setup
#[derive(Debug)]
pub enum TransformerError {
    TensorError(TensorError),
    WeightNotFound(String),
    InvalidWeightShape(String),
    ConfigError(String),
    UnsupportedOperation(String), 
}

impl std::fmt::Display for TransformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformerError::TensorError(e) => write!(f, "Tensor error: {:?}", e),
            TransformerError::WeightNotFound(s) => write!(f, "Weight not found: {}", s),
            TransformerError::InvalidWeightShape(s) => write!(f, "Invalid weight shape: {}", s),
            TransformerError::ConfigError(s) => write!(f, "Configuration error: {}", s),
            TransformerError::UnsupportedOperation(s) => write!(f, "Unsupported operation: {}", s),
        }
    }
}

impl std::error::Error for TransformerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TransformerError::TensorError(ref e) => Some(e), 
            _ => None,
        }
    }
}

impl From<TensorError> for TransformerError {
    fn from(err: TensorError) -> TransformerError {
        TransformerError::TensorError(err)
    }
}

// 1. Config Struct
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,    
    pub n_head: usize,     
    pub n_embd: usize,     
    pub vocab_size: usize, 
    pub block_size: usize, 
    pub bias: bool, 
}

impl Config {
    pub fn head_dim(&self) -> usize {
        if self.n_embd == 0 || self.n_head == 0 { 
            return 0;
        }
        if self.n_embd % self.n_head != 0 {
            eprintln!("Warning: n_embd {} is not divisible by n_head {}. Head dimension may be incorrect.", self.n_embd, self.n_head);
        }
        self.n_embd / self.n_head
    }
}

#[allow(dead_code)]
pub(crate) mod tensor_ops {
    use super::*;

    pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        Tensor::matmul(a, b) 
    }

    pub fn softmax(a: &Tensor<f32>, axis: usize) -> Result<Tensor<f32>, TensorError> {
        a.softmax(axis)
    }

    pub fn layernorm(a: &Tensor<f32>, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<Tensor<f32>, TensorError> {
        a.layernorm(gamma, beta, epsilon)
    }

    pub fn gelu(a: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        a.gelu() 
    }
    
    pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if a.shape != b.shape {
            if b.rank() == 1 && a.shape.last() == b.shape.last() && a.rank() > 1 {
                let mut out_data = a.data.clone();
                let last_dim_size = b.shape[0];
                let num_vectors = a.data.len() / last_dim_size;
                for i in 0..num_vectors {
                    for j_idx in 0..last_dim_size {
                        out_data[i * last_dim_size + j_idx] += b.data[j_idx];
                    }
                }
                return Tensor::new(out_data, a.shape.clone());
            }
            return Err(TensorError::ShapeMismatch(format!(
                "Element-wise add requires identical shapes or broadcastable bias. Got {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let data = a.data.iter().zip(b.data.iter()).map(|(av, bv)| av + bv).collect();
        Tensor::new(data, a.shape.clone())
    }
    
    pub fn split_last_dim(tensor: &Tensor<f32>, num_chunks: usize) -> Result<Vec<Tensor<f32>>, TensorError> {
        if tensor.rank() == 0 {
            return Err(TensorError::InvalidDimension("Cannot split scalar tensor".to_string()));
        }
        if tensor.data.is_empty() && tensor.num_elements() == 0 {
            let last_dim_idx = tensor.rank().saturating_sub(1); 
            let last_dim_size = if tensor.shape.is_empty() { 0 } else { tensor.shape[last_dim_idx] };

            if last_dim_size == 0 && num_chunks > 0 { 
                 let mut new_shape = tensor.shape.clone();
                 if !new_shape.is_empty() { new_shape[last_dim_idx] = 0; }
                let mut chunks = Vec::new();
                for _ in 0..num_chunks {
                    chunks.push(Tensor::new(Vec::new(), new_shape.clone())?);
                }
                return Ok(chunks);
            } else if last_dim_size % num_chunks == 0 {
                let mut new_shape = tensor.shape.clone();
                 if !new_shape.is_empty() { new_shape[last_dim_idx] = last_dim_size / num_chunks; }
                let mut chunks = Vec::new();
                for _ in 0..num_chunks {
                    chunks.push(Tensor::new(Vec::new(), new_shape.clone())?);
                }
                return Ok(chunks);
            }
        }

        let last_dim_idx = tensor.rank() - 1;
        let last_dim_size = tensor.shape[last_dim_idx];

        if last_dim_size % num_chunks != 0 {
            return Err(TensorError::InvalidDimension(format!(
                "Last dimension size {} cannot be evenly split into {} chunks",
                last_dim_size, num_chunks
            )));
        }
        let chunk_size_last_dim = last_dim_size / num_chunks;
        
        let mut result_tensors = Vec::with_capacity(num_chunks);
        let mut new_shape_template = tensor.shape.clone();
        new_shape_template[last_dim_idx] = chunk_size_last_dim;
        
        if tensor.data.is_empty() && tensor.num_elements() > 0 {
             return Err(TensorError::ShapeMismatch("Tensor has non-zero shape product but empty data for split.".to_string()));
        }
        if tensor.data.is_empty() && tensor.num_elements() == 0 { 
            return Ok(result_tensors); 
        }

        let num_elements_per_chunk: usize = new_shape_template.iter().product();
        let num_outer_elements: usize = if tensor.rank() > 1 { tensor.shape[..last_dim_idx].iter().product() } else { 1 };

        for chunk_idx in 0..num_chunks {
            let mut chunk_data = Vec::with_capacity(num_elements_per_chunk);
            for outer_idx in 0..num_outer_elements {
                let original_data_start_offset = outer_idx * last_dim_size + chunk_idx * chunk_size_last_dim;
                chunk_data.extend_from_slice(&tensor.data[original_data_start_offset .. original_data_start_offset + chunk_size_last_dim]);
            }
            result_tensors.push(Tensor::new(chunk_data, new_shape_template.clone())?);
        }
        Ok(result_tensors)
    }

    pub fn split_dim1(tensor: &Tensor<f32>, num_chunks: usize) -> Result<Vec<Tensor<f32>>, TensorError> {
        if tensor.rank() != 2 {
            return Err(TensorError::InvalidDimension("split_dim1 expects a 2D tensor".to_string()));
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];

        if cols == 0 && num_chunks > 0 { 
            let mut chunks = Vec::new();
            for _ in 0..num_chunks {
                chunks.push(Tensor::new(Vec::new(), vec![rows, 0])?);
            }
            return Ok(chunks);
        }

        if cols % num_chunks != 0 {
            return Err(TensorError::InvalidDimension(format!(
                "Dimension 1 (cols) size {} cannot be evenly split into {} chunks",
                cols, num_chunks
            )));
        }
        let chunk_cols = cols / num_chunks;
        let mut result_tensors = Vec::with_capacity(num_chunks);

        if tensor.data.is_empty() && tensor.num_elements() > 0 {
            return Err(TensorError::ShapeMismatch("Tensor has non-zero shape product but empty data for split_dim1.".to_string()));
        }
        if tensor.data.is_empty() && tensor.num_elements() == 0 {
             for _ in 0..num_chunks {
                result_tensors.push(Tensor::new(Vec::new(), vec![rows, chunk_cols])?);
            }
            return Ok(result_tensors);
        }

        for i in 0..num_chunks {
            let mut chunk_data = Vec::with_capacity(rows * chunk_cols);
            for r in 0..rows {
                let start_col_in_original = i * chunk_cols;
                let original_row_start_idx = r * cols;
                chunk_data.extend_from_slice(&tensor.data[original_row_start_idx + start_col_in_original .. original_row_start_idx + start_col_in_original + chunk_cols]);
            }
            result_tensors.push(Tensor::new(chunk_data, vec![rows, chunk_cols])?);
        }
        Ok(result_tensors)
    }
    
    pub fn linear(input: &Tensor<f32>, weight: &Tensor<f32>, bias: Option<&Tensor<f32>>) -> Result<Tensor<f32>, TransformerError> {
        let din = weight.shape[0];
        let dout = weight.shape[1];
        if input.shape.last().unwrap_or(&0) != &din {
            return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(format!(
                "Linear input last dim {} != weight first dim {}",
                input.shape.last().unwrap_or(&0), din
            ))));
        }

        let mut out_shape = input.shape.clone();
        if out_shape.is_empty() && input.num_elements() == din { 
             out_shape = vec![1, din]; 
        } else if out_shape.is_empty() { 
            return Err(TransformerError::TensorError(TensorError::InvalidDimension("Scalar input not directly usable in linear layer without proper shape".to_string())));
        }
        let last_dim_idx = out_shape.len() - 1;
        out_shape[last_dim_idx] = dout;
        
        let original_rank = input.rank();
        let input_reshaped = if original_rank > 2 {
            let new_rows: usize = input.shape[..original_rank-1].iter().product();
            input.reshape(vec![new_rows, din])?
        } else if original_rank == 1 && input.shape[0] == din { 
            input.reshape(vec![1, din])? 
        }
         else {
            input.clone() 
        };

        let mut output = Tensor::matmul(&input_reshaped, weight)?;

        if let Some(b) = bias {
            if b.rank() != 1 || b.shape[0] != dout {
                 return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(format!("Bias shape {:?} incompatible with output dim {}", b.shape, dout))));
            }
            for r_idx in 0..output.shape[0] { 
                for c_idx in 0..output.shape[1] { 
                    let flat_idx = r_idx * output.shape[1] + c_idx;
                    output.data[flat_idx] += b.data[c_idx];
                }
            }
        }
        
        if original_rank > 2 || (original_rank == 1 && input.shape[0] == din) { 
            output = output.reshape(out_shape)?;
        }
        Ok(output)
    }

    pub fn embedding(ids: &Tensor<u32>, weight: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if weight.rank() != 2 {
            return Err(TensorError::InvalidDimension("Embedding weight matrix must be 2D".to_string()));
        }
        let vocab_size = weight.shape[0];
        let emb_dim = weight.shape[1];

        let mut out_shape = ids.shape.clone();
        out_shape.push(emb_dim);

        let mut out_data = Vec::with_capacity(ids.num_elements() * emb_dim);

        for id_val in &ids.data {
            let id_idx = *id_val as usize;
            if id_idx >= vocab_size {
                return Err(TensorError::OutOfBounds(format!("Token ID {} out of vocab size {}", id_idx, vocab_size)));
            }
            let embedding_vector_slice_start = id_idx * emb_dim;
            let embedding_vector_slice_end = embedding_vector_slice_start + emb_dim;
            out_data.extend_from_slice(&weight.data[embedding_vector_slice_start..embedding_vector_slice_end]);
        }
        Tensor::new(out_data, out_shape)
    }
}

// --- KV Cache Data Structures ---

#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub key: Tensor<f32>,   
    pub value: Tensor<f32>, 
}

pub type LayerKVCache = Vec<KVCacheEntry>; 
pub type ModelKVCache = Vec<LayerKVCache>;  

// MultiHeadAttention Module (Refactored for KV Cache)
pub struct MultiHeadAttention {
    w_q: Tensor<f32>,
    b_q: Tensor<f32>,
    w_k: Tensor<f32>,
    b_k: Tensor<f32>,
    w_v: Tensor<f32>,
    b_v: Tensor<f32>,
    c_proj_w: Tensor<f32>, 
    c_proj_b: Tensor<f32>, 
    config: Arc<Config>,
}

impl MultiHeadAttention {
    pub fn new(
        config: Arc<Config>,
        weights: &mut HashMap<String, Tensor<f32>>,
        prefix: &str,        
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;

        let w_q: Tensor<f32>;
        let b_q: Tensor<f32>;
        let w_k: Tensor<f32>;
        let b_k: Tensor<f32>;
        let w_v: Tensor<f32>;
        let b_v: Tensor<f32>;

        let w_q_key = format!("{}w_q.weight", prefix);
        let b_q_key = format!("{}w_q.bias", prefix);
        let w_k_key = format!("{}w_k.weight", prefix);
        let b_k_key = format!("{}w_k.bias", prefix);
        let w_v_key = format!("{}w_v.weight", prefix);
        let b_v_key = format!("{}w_v.bias", prefix);

        if weights.contains_key(&w_q_key) { 
            w_q = GPT2Model::get_weight(weights, &w_q_key, Some(&[n_embd, n_embd]))?;
            b_q = GPT2Model::get_weight(weights, &b_q_key, Some(&[n_embd]))?;
            w_k = GPT2Model::get_weight(weights, &w_k_key, Some(&[n_embd, n_embd]))?;
            b_k = GPT2Model::get_weight(weights, &b_k_key, Some(&[n_embd]))?;
            w_v = GPT2Model::get_weight(weights, &w_v_key, Some(&[n_embd, n_embd]))?;
            b_v = GPT2Model::get_weight(weights, &b_v_key, Some(&[n_embd]))?;
        } else {
            eprintln!("[MultiHeadAttention::new] Info: Separate Q,K,V weights not found for prefix '{}'. Attempting to load and split combined 'c_attn' weights.", prefix);
            let c_attn_w_combined = GPT2Model::get_weight(weights, &format!("{}c_attn.weight", prefix), Some(&[n_embd, 3 * n_embd]))?;
            let c_attn_b_combined = GPT2Model::get_weight(weights, &format!("{}c_attn.bias", prefix), Some(&[3 * n_embd]))?;

            let mut qkv_w_parts = tensor_ops::split_dim1(&c_attn_w_combined, 3)?;
            if qkv_w_parts.len() != 3 { return Err(TransformerError::InvalidWeightShape(format!("Failed to split c_attn.weight for {} into 3 parts", prefix)));}
            w_q = qkv_w_parts.remove(0); 
            w_k = qkv_w_parts.remove(0); 
            w_v = qkv_w_parts.remove(0);

            let mut qkv_b_parts = tensor_ops::split_last_dim(&c_attn_b_combined, 3)?;
            if qkv_b_parts.len() != 3 { return Err(TransformerError::InvalidWeightShape(format!("Failed to split c_attn.bias for {} into 3 parts", prefix)));}
            b_q = qkv_b_parts.remove(0);
            b_k = qkv_b_parts.remove(0);
            b_v = qkv_b_parts.remove(0);
        }
        
        let c_proj_w = GPT2Model::get_weight(weights, &format!("{}c_proj.weight", prefix), Some(&[n_embd, n_embd]))?;
        let c_proj_b = GPT2Model::get_weight(weights, &format!("{}c_proj.bias", prefix), Some(&[n_embd]))?;

        Ok(MultiHeadAttention {
            w_q, b_q,
            w_k, b_k,
            w_v, b_v,
            c_proj_w,
            c_proj_b,
            config,
        })
    }

    pub fn forward(
        &self, 
        x: &Tensor<f32>, 
        mask: Option<&Tensor<f32>>,
        theta_hat: Option<f32>,
        cache: Option<&mut LayerKVCache> 
    ) -> Result<Tensor<f32>, TransformerError> {
        let batch_size = x.shape[0];
        let current_seq_len = x.shape[1]; 
        let n_embd = self.config.n_embd;
        let n_head = self.config.n_head;
        let head_dim = self.config.head_dim();

        let q_all = tensor_ops::linear(x, &self.w_q, Some(&self.b_q))?;
        let k_all_current_input = tensor_ops::linear(x, &self.w_k, Some(&self.b_k))?;
        let v_all_current_input = tensor_ops::linear(x, &self.w_v, Some(&self.b_v))?;

        let q_heads = q_all.reshape(vec![batch_size, current_seq_len, n_head, head_dim])?
                           .permute_mha_qkv()?; 
        let k_heads_current = k_all_current_input.reshape(vec![batch_size, current_seq_len, n_head, head_dim])?
                                     .permute_mha_qkv()?;
        let v_heads_current = v_all_current_input.reshape(vec![batch_size, current_seq_len, n_head, head_dim])?
                                     .permute_mha_qkv()?;

        let mut k_for_attention_all_heads: Tensor<f32>;
        let mut v_for_attention_all_heads: Tensor<f32>;
        let mut effective_kv_seq_len = current_seq_len;

        if let Some(layer_cache_mut) = cache {
            // Sequential Initialization: Ensure cache structure is ready for all heads.
            // This must run before parallel operations on the cache.
            while layer_cache_mut.len() < n_head {
                layer_cache_mut.push(KVCacheEntry {
                    key: Tensor::new(Vec::new(), vec![batch_size, 0, head_dim])?,
                    value: Tensor::new(Vec::new(), vec![batch_size, 0, head_dim])?,
                });
            }

            // Create an immutable snapshot of the previous K/V states for parallel processing.
            let previous_kv_state: Vec<(Tensor<f32>, Tensor<f32>)> = layer_cache_mut
                .iter()
                .map(|entry| (entry.key.clone(), entry.value.clone()))
                .collect();

            // Parallel Computation of New K/V Pairs
            let computed_kv_updates: Result<Vec<(Tensor<f32>, Tensor<f32>)>, TransformerError> =
                (0..n_head).into_par_iter().map(|h_idx| {
                    let k_current_this_head = k_heads_current.slice_one_head_all_batches(h_idx)?;
                    let v_current_this_head = v_heads_current.slice_one_head_all_batches(h_idx)?;

                    // Access the previous state for this head from the snapshot
                    let (prev_k_this_head, prev_v_this_head) = &previous_kv_state[h_idx];

                    let k_to_cache = if prev_k_this_head.shape[1] == 0 {
                        k_current_this_head // No need to clone if it's the first entry
                    } else {
                        Tensor::concat(&[prev_k_this_head, &k_current_this_head], 1)?
                    };
                    let v_to_cache = if prev_v_this_head.shape[1] == 0 {
                        v_current_this_head // No need to clone
                    } else {
                        Tensor::concat(&[prev_v_this_head, &v_current_this_head], 1)?
                    };
                    Ok((k_to_cache, v_to_cache))
                }).collect();
            
            let computed_kv_updates = computed_kv_updates?;

            // Sequential Update of LayerKVCache
            let mut temp_k_list = Vec::with_capacity(n_head);
            let mut temp_v_list = Vec::with_capacity(n_head);

            for h_idx in 0..n_head {
                // Directly move computed tensors if possible, or clone if ownership rules require.
                // Assuming computed_kv_updates owns the tensors.
                layer_cache_mut[h_idx].key = computed_kv_updates[h_idx].0.clone(); // Clone if multiple uses, or if not owning
                layer_cache_mut[h_idx].value = computed_kv_updates[h_idx].1.clone();

                temp_k_list.push(layer_cache_mut[h_idx].key.clone());
                temp_v_list.push(layer_cache_mut[h_idx].value.clone());
            }

            k_for_attention_all_heads = Tensor::stack_heads_from_list(&temp_k_list)?;
            v_for_attention_all_heads = Tensor::stack_heads_from_list(&temp_v_list)?;

            if !temp_k_list.is_empty() {
                effective_kv_seq_len = temp_k_list[0].shape[1];
            }

        } else {
            k_for_attention_all_heads = k_heads_current;
            v_for_attention_all_heads = v_heads_current;
        }
        
        let k_t_final = k_for_attention_all_heads.permute_mha_kt()?;
        
        let scale = (head_dim as f32).sqrt();
        let inv_scale = 1.0 / scale; // Precompute for multiplication if preferred

        // Parallel computation of attention scores
        let att_scores_results: Result<Vec<Tensor<f32>>, TransformerError> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b_idx| {
                (0..n_head).into_par_iter().map(move |h_idx| {
                    let q_slice = q_heads.slice_mha(b_idx, h_idx)?;
                    let k_t_slice = k_t_final.slice_mha_for_kv(b_idx, h_idx, effective_kv_seq_len)?;

                    let mut scores_s = Tensor::matmul(&q_slice, &k_t_slice)?;

                    // Scale the scores
                    // Using scalar_mul would be cleaner if it's confirmed to be efficient
                    // or if this loop becomes a bottleneck later.
                    for val in scores_s.data.iter_mut() {
                        *val *= inv_scale; // Use multiplication by inverse for potential minor perf gain
                    }
                    Ok(scores_s)
                })
            })
            .collect();

        let att_scores_parts = att_scores_results?;

        // Assemble att_scores tensor (sequential)
        // Calculate expected capacity for att_scores_data_flat
        let expected_capacity = batch_size * n_head * current_seq_len * effective_kv_seq_len;
        let mut att_scores_data_flat = Vec::with_capacity(expected_capacity);
        for t in att_scores_parts { // att_scores_parts is Vec<Tensor<f32>>
            att_scores_data_flat.extend(t.data);
        }

        let mut att_scores = Tensor::new(att_scores_data_flat, vec![batch_size, n_head, current_seq_len, effective_kv_seq_len])?;

        if let Some(th_value) = theta_hat {
            att_scores = att_scores.scalar_mul(th_value)?;
        }

        if let Some(m) = mask { 
             if m.shape == vec![current_seq_len, effective_kv_seq_len] { 
                for b in 0..batch_size {
                    for h in 0..n_head {
                        for s_q_idx in 0..current_seq_len {      
                            for s_kv_idx in 0..effective_kv_seq_len { 
                                if *m.get(&[s_q_idx, s_kv_idx])? == 0.0 { 
                                    let val_ref = att_scores.get_mut(&[b,h,s_q_idx,s_kv_idx])?;
                                    *val_ref = f32::NEG_INFINITY;
                                }
                            }
                        }
                    }
                }
            } else { // Fallback for standard causal mask if no cache or first token
                 if current_seq_len == effective_kv_seq_len && m.shape == vec![current_seq_len, current_seq_len] {
                    for b in 0..batch_size {
                        for h in 0..n_head {
                            for s1 in 0..current_seq_len {
                                for s2 in 0..current_seq_len {
                                    if *m.get(&[s1, s2])? == 0.0 { 
                                        let val_ref = att_scores.get_mut(&[b,h,s1,s2])?;
                                        *val_ref = f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("[MHA] Warning: Mask shape {:?} not directly applicable for Q_len={} KV_len={}. Masking might be incorrect.", m.shape, current_seq_len, effective_kv_seq_len);
                }
            }
        }

        let att_probs = tensor_ops::softmax(&att_scores, 3)?;

        // Parallel computation of output attention values (AttnProbs @ V)
        let out_att_results: Result<Vec<Tensor<f32>>, TransformerError> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b_idx| {
                (0..n_head).into_par_iter().map(move |h_idx| {
                    let probs_slice = att_probs.slice_mha_custom(b_idx, h_idx, current_seq_len, effective_kv_seq_len)?;
                    let v_slice = v_for_attention_all_heads.slice_mha_for_kv(b_idx, h_idx, effective_kv_seq_len)?;
                    Tensor::matmul(&probs_slice, &v_slice)
                })
            })
            .collect();

        let out_att_parts = out_att_results?;

        // Assemble out_att tensor (sequential)
        let expected_out_capacity = batch_size * n_head * current_seq_len * head_dim;
        let mut out_att_data_flat = Vec::with_capacity(expected_out_capacity);
        for t in out_att_parts { // out_att_parts is Vec<Tensor<f32>>
            out_att_data_flat.extend(t.data);
        }
        let out_att = Tensor::new(out_att_data_flat, vec![batch_size, n_head, current_seq_len, head_dim])?;
        
        let out_reshaped = out_att.permute_mha_output()?
                                  .reshape(vec![batch_size, current_seq_len, n_embd])?;
        
        let final_output = tensor_ops::linear(&out_reshaped, &self.c_proj_w, Some(&self.c_proj_b))?;
        
        Ok(final_output)
    }
}

pub struct FeedForward {
    c_fc_w: Tensor<f32>,   
    c_fc_b: Tensor<f32>,   
    c_proj_w: Tensor<f32>, 
    c_proj_b: Tensor<f32>, 
}

impl FeedForward {
    pub fn new(
        c_fc_w: Tensor<f32>, 
        c_fc_b: Tensor<f32>, 
        c_proj_w: Tensor<f32>, 
        c_proj_b: Tensor<f32>,
        config: &Config 
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        let n_hidden = 4 * n_embd; 

        if c_fc_w.shape != vec![n_embd, n_hidden] {
            return Err(TransformerError::InvalidWeightShape(format!("c_fc_w shape mismatch: expected [{}, {}], got {:?}", n_embd, n_hidden, c_fc_w.shape)));
        }
        if c_fc_b.shape != vec![n_hidden] {
            return Err(TransformerError::InvalidWeightShape(format!("c_fc_b shape mismatch: expected [{}], got {:?}", n_hidden, c_fc_b.shape)));
        }
        if c_proj_w.shape != vec![n_hidden, n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_w shape mismatch: expected [{}, {}], got {:?}", n_hidden, n_embd, c_proj_w.shape)));
        }
        if c_proj_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_b shape mismatch: expected [{}], got {:?}", n_embd, c_proj_b.shape)));
        }
        
        Ok(FeedForward { c_fc_w, c_fc_b, c_proj_w, c_proj_b })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TransformerError> {
        let mut h = tensor_ops::linear(x, &self.c_fc_w, Some(&self.c_fc_b))?;
        h = h.gelu()?; 
        let output = tensor_ops::linear(&h, &self.c_proj_w, Some(&self.c_proj_b))?;
        Ok(output)
    }
}

pub struct Block {
    attn: MultiHeadAttention,
    mlp: FeedForward,
    ln_1_g: Tensor<f32>, 
    ln_1_b: Tensor<f32>, 
    ln_2_g: Tensor<f32>, 
    ln_2_b: Tensor<f32>, 
}

impl Block {
    pub fn new(
        attn: MultiHeadAttention, 
        mlp: FeedForward, 
        ln_1_g: Tensor<f32>, 
        ln_1_b: Tensor<f32>, 
        ln_2_g: Tensor<f32>, 
        ln_2_b: Tensor<f32>,
        config: &Config 
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        if ln_1_g.shape != vec![n_embd] || ln_1_b.shape != vec![n_embd] ||
           ln_2_g.shape != vec![n_embd] || ln_2_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape("LayerNorm weight shape mismatch".to_string()));
        }

        Ok(Block { attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b })
    }

    pub fn forward(
        &self, 
        x: &Tensor<f32>, 
        mask: Option<&Tensor<f32>>, 
        theta_hat: Option<f32>, 
        layer_cache: Option<&mut LayerKVCache>
    ) -> Result<Tensor<f32>, TransformerError> {
        let x_norm1 = tensor_ops::layernorm(x, &self.ln_1_g, &self.ln_1_b, 1e-5)?;
        let attn_output = self.attn.forward(&x_norm1, mask, theta_hat, layer_cache)?; 
        let x_plus_attn = tensor_ops::add(x, &attn_output)?; 
        
        let x_norm2 = tensor_ops::layernorm(&x_plus_attn, &self.ln_2_g, &self.ln_2_b, 1e-5)?;
        let mlp_output = self.mlp.forward(&x_norm2)?;
        let final_output = tensor_ops::add(&x_plus_attn, &mlp_output)?;
        
        Ok(final_output)
    }
}

pub struct GPT2Model {
    pub config: Arc<Config>, 
    wte: Tensor<f32>,    
    wpe: Tensor<f32>,    
    blocks: Vec<Block>,
    ln_f_g: Tensor<f32>, 
    ln_f_b: Tensor<f32>, 
}

impl GPT2Model {
    pub(crate) fn get_weight(weights: &mut HashMap<String, Tensor<f32>>, name: &str, expected_shape: Option<&[usize]>) -> Result<Tensor<f32>, TransformerError> {
        let weight_name = name.to_string(); 
        let weight = weights.remove(&weight_name).ok_or_else(|| TransformerError::WeightNotFound(name.to_string()))?;
        if let Some(shape) = expected_shape {
            if weight.shape != shape {
                return Err(TransformerError::InvalidWeightShape(format!(
                    "Weight {} shape mismatch: expected {:?}, got {:?}",
                    name, shape, weight.shape
                )));
            }
        }
        Ok(weight)
    }
    
    pub fn new(config: Config, mut weights: HashMap<String, Tensor<f32>>) -> Result<Self, TransformerError> {
        let conf = Arc::new(config);
        let n_layer = conf.n_layer;
        let n_embd = conf.n_embd;
        let vocab_size = conf.vocab_size;
        let block_size = conf.block_size;

        let wte = Self::get_weight(&mut weights, "wte.weight", Some(&[vocab_size, n_embd]))?;
        let wpe = Self::get_weight(&mut weights, "wpe.weight", Some(&[block_size, n_embd]))?;
        
        let mut blocks_vec = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let prefix = format!("h.{}.attn.", i);
            let attn = MultiHeadAttention::new(Arc::clone(&conf), &mut weights, &prefix)?;

            let mlp_c_fc_w = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_fc.weight", i), Some(&[n_embd, 4 * n_embd]))?;
            let mlp_c_fc_b = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_fc.bias", i), Some(&[4 * n_embd]))?;
            let mlp_c_proj_w = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_proj.weight", i), Some(&[4 * n_embd, n_embd]))?;
            let mlp_c_proj_b = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_proj.bias", i), Some(&[n_embd]))?;
            let mlp = FeedForward::new(mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b, &conf)?;

            let ln_1_g = Self::get_weight(&mut weights, &format!("h.{}.ln_1.weight", i), Some(&[n_embd]))?;
            let ln_1_b = Self::get_weight(&mut weights, &format!("h.{}.ln_1.bias", i), Some(&[n_embd]))?;
            let ln_2_g = Self::get_weight(&mut weights, &format!("h.{}.ln_2.weight", i), Some(&[n_embd]))?;
            let ln_2_b = Self::get_weight(&mut weights, &format!("h.{}.ln_2.bias", i), Some(&[n_embd]))?;
            
            blocks_vec.push(Block::new(attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b, &conf)?);
        }

        let ln_f_g = Self::get_weight(&mut weights, "ln_f.weight", Some(&[n_embd]))?;
        let ln_f_b = Self::get_weight(&mut weights, "ln_f.bias", Some(&[n_embd]))?;

        if !weights.is_empty() {
            eprintln!("[GPT2Model::new] Warning: Unused weights: {:?}", weights.keys().collect::<Vec<_>>());
        }

        Ok(GPT2Model {
            config: conf,
            wte,
            wpe,
            blocks: blocks_vec,
            ln_f_g,
            ln_f_b,
        })
    }

    pub fn forward(
        &self, 
        token_ids: &Tensor<u32>, 
        _mask: Option<&Tensor<f32>>, // Original mask, might need to be regenerated or passed carefully
        theta_hat: Option<f32>,
        mut model_cache: Option<&mut ModelKVCache>
    ) -> Result<Tensor<f32>, TransformerError> {
        let batch_size = token_ids.shape[0];
        let current_seq_len = token_ids.shape[1]; // S_q

        if current_seq_len > self.config.block_size { // This check is against S_q, not S_total
            return Err(TransformerError::ConfigError(format!(
                "Input sequence length {} exceeds model block size {}",
                current_seq_len, self.config.block_size
            )));
        }

        let token_embed = tensor_ops::embedding(token_ids, &self.wte)?; 
        
        // Positional embedding logic:
        // If caching, current_seq_len might be 1. We need absolute positions.
        let past_seq_len = if let Some(cache) = model_cache.as_ref() {
            if !cache.is_empty() && !cache[0].is_empty() && cache[0][0].key.shape.len() > 1 {
                cache[0][0].key.shape[1] // Seq_len from layer 0, head 0, key tensor [B, S, D]
            } else { 0 }
        } else { 0 };

        // Create position IDs for the current input tokens based on past_seq_len
        let pos_ids_data: Vec<u32> = (past_seq_len .. past_seq_len + current_seq_len).map(|p| p as u32).collect();
        if pos_ids_data.is_empty() && current_seq_len > 0 { // Should not happen if current_seq_len > 0
             return Err(TransformerError::ConfigError("Position IDs vector is empty for non-zero sequence length.".to_string()));
        }
        // Reshape pos_ids_data to match how embedding expects it, typically [current_seq_len] or [batch_size, current_seq_len]
        // If embedding op handles broadcasting of [current_seq_len] to batch, then [current_seq_len] is fine.
        // Our current embedding expects ids: [B,S] or [S]. We'll make it [current_seq_len] and rely on broadcasting/tiling later.
        let pos_ids_tensor_shape = if batch_size > 1 && current_seq_len > 0 { vec![batch_size, current_seq_len] } else { vec![current_seq_len] };
        let pos_ids_tensor_data = if batch_size > 1 && current_seq_len > 0 {
            pos_ids_data.iter().cycle().take(batch_size * current_seq_len).cloned().collect()
        } else {
            pos_ids_data
        };
        let pos_ids_tensor = Tensor::new(pos_ids_tensor_data, pos_ids_tensor_shape)?;
        
        let pos_embed_full = tensor_ops::embedding(&pos_ids_tensor, &self.wpe)?; 

        // If pos_embed_full is [S_q, E] and token_embed is [B, S_q, E], we need to broadcast/add.
        // If pos_embed_full is [B, S_q, E] (due to batch_size in pos_ids_tensor_shape), direct add is fine.
        let mut x = tensor_ops::add(&token_embed, &pos_embed_full)?;


        // Mask generation:
        let total_kv_seq_len = past_seq_len + current_seq_len;
        let attention_mask = if current_seq_len == 1 && past_seq_len > 0 { // Attending to all past, and self
            Tensor::new(vec![1.0f32; 1 * total_kv_seq_len], vec![1, total_kv_seq_len])? // [1, S_total]
        } else { // Causal mask for prefill or if no cache
            let mut mask_data = vec![1.0f32; current_seq_len * total_kv_seq_len];
            for i in 0..current_seq_len { // Query sequence (current tokens)
                for j_idx in 0..total_kv_seq_len { // Key/Value sequence (past + current)
                    if j_idx > (past_seq_len + i) { 
                        mask_data[i * total_kv_seq_len + j_idx] = 0.0;
                    }
                }
            }
            Tensor::new(mask_data, vec![current_seq_len, total_kv_seq_len])?
        };


        // Initialize model_cache if it's the first pass and cache is Some
        if let Some(cache_mut) = model_cache.as_mut() {
            if cache_mut.is_empty() { // First time this cache is used for this model
                for _ in 0..self.config.n_layer {
                    let mut layer_cache_init = Vec::with_capacity(self.config.n_head);
                    for _ in 0..self.config.n_head {
                        layer_cache_init.push(KVCacheEntry {
                            key: Tensor::new(Vec::new(), vec![batch_size, 0, self.config.head_dim()])?,
                            value: Tensor::new(Vec::new(), vec![batch_size, 0, self.config.head_dim()])?,
                        });
                    }
                    cache_mut.push(layer_cache_init);
                }
            } else if cache_mut.len() != self.config.n_layer {
                 return Err(TransformerError::ConfigError(format!(
                    "Provided model_cache has {} layers, but model expects {}",
                    cache_mut.len(), self.config.n_layer
                )));
            }
        }

        match model_cache {
            Some(cache_mut) => {
                for (i, block) in self.blocks.iter().enumerate() {
                    x = block.forward(&x, Some(&attention_mask), theta_hat, Some(&mut cache_mut[i]))?;
                }
            }
            None => {
                // Standard causal mask for non-cached scenario (total_kv_seq_len == current_seq_len)
                let simple_causal_mask = attention_mask; // Mask already calculated for S_q, S_kv=S_q
                for block in &self.blocks {
                    x = block.forward(&x, Some(&simple_causal_mask), theta_hat, None)?;
                }
            }
        }

        x = tensor_ops::layernorm(&x, &self.ln_f_g, &self.ln_f_b, 1e-5)?;
        
        // If only generating for the last token (current_seq_len == 1 and past_seq_len > 0),
        // take only the last token's activations for logit calculation.
        let x_for_logits = if current_seq_len == 1 && past_seq_len > 0 {
            // x is [B, 1, E]. This is already the last token's activations.
            x 
        } else if current_seq_len > 1 && past_seq_len == 0 {
            // This is a prefill, we need all logits.
            x
        } else {
            // Other cases or if full sequence logits are always needed by design
            x
        };
        
        let wte_t_data = Tensor::<f32>::transpose_data_generic(&self.wte.data, self.wte.shape[0], self.wte.shape[1]);
        let wte_t = Tensor::new(wte_t_data, vec![self.config.n_embd, self.config.vocab_size])?;
        
        let logits = tensor_ops::linear(&x_for_logits, &wte_t, None)?;

        Ok(logits)
    }
}

trait TensorExtMHA {
    fn permute_mha_qkv(&self) -> Result<Tensor<f32>, TransformerError>;      
    fn permute_mha_kt(&self) -> Result<Tensor<f32>, TransformerError>;       
    fn permute_mha_output(&self) -> Result<Tensor<f32>, TransformerError>;  
    fn slice_mha(&self, batch_idx: usize, head_idx: usize) -> Result<Tensor<f32>, TransformerError>; 
    fn slice_one_head_all_batches(&self, head_idx: usize) -> Result<Tensor<f32>, TransformerError>; 
    fn stack_heads_from_list(head_tensors: &[Tensor<f32>]) -> Result<Tensor<f32>, TransformerError>;
    fn slice_mha_for_kv(&self, batch_idx: usize, head_idx: usize, kv_seq_len: usize) -> Result<Tensor<f32>, TransformerError>; 
    fn slice_mha_custom(&self, batch_idx: usize, head_idx: usize, q_seq_len: usize, kv_seq_len: usize) -> Result<Tensor<f32>, TransformerError>; 
}

impl TensorExtMHA for Tensor<f32> {
    fn permute_mha_qkv(&self) -> Result<Tensor<f32>, TransformerError> { 
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_qkv expects 4D tensor".into()))); }
        let b = self.shape[0]; let s = self.shape[1]; let h = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![0.0; self.data.len()];
        let new_shape = vec![b, h, s, d];
        for b_i in 0..b {
            for s_i in 0..s {
                for h_i in 0..h {
                    for d_i in 0..d {
                        let old_idx = b_i*s*h*d + s_i*h*d + h_i*d + d_i;
                        let new_idx = b_i*h*s*d + h_i*s*d + s_i*d + d_i;
                        new_data[new_idx] = self.data[old_idx];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }

    fn permute_mha_kt(&self) -> Result<Tensor<f32>, TransformerError> { 
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_kt expects 4D tensor".into()))); }
        let b = self.shape[0]; let h = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![0.0; self.data.len()];
        let new_shape = vec![b, h, d, s];
        for b_i in 0..b {
            for h_i in 0..h {
                for s_i in 0..s {
                    for d_i in 0..d {
                        let old_idx = b_i*h*s*d + h_i*s*d + s_i*d + d_i;
                        let new_idx = b_i*h*d*s + h_i*d*s + d_i*s + s_i;
                        new_data[new_idx] = self.data[old_idx];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }
    
    fn permute_mha_output(&self) -> Result<Tensor<f32>, TransformerError> { 
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_output expects 4D tensor".into()))); }
        let b = self.shape[0]; let h = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![0.0; self.data.len()];
        let new_shape = vec![b, s, h, d];
         for b_i in 0..b {
            for h_i in 0..h {
                for s_i in 0..s {
                    for d_i in 0..d {
                        let old_idx = b_i*h*s*d + h_i*s*d + s_i*d + d_i;
                        let new_idx = b_i*s*h*d + s_i*h*d + h_i*d + d_i;
                        new_data[new_idx] = self.data[old_idx];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }

    fn slice_mha(&self, batch_idx: usize, head_idx: usize) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_mha expects 4D tensor [B,H,S,D]".into()))); }
        let s_dim = self.shape[2]; 
        let d_dim = self.shape[3]; 
        
        let mut slice_data = Vec::with_capacity(s_dim * d_dim);
        let h_total_in_tensor = self.shape[1]; 

        let offset = batch_idx * h_total_in_tensor * s_dim * d_dim + head_idx * s_dim * d_dim;

        if offset + (s_dim * d_dim) > self.data.len() {
            return Err(TransformerError::TensorError(TensorError::OutOfBounds(
                format!("slice_mha: Offset {} + slice_size {} > data_len {}. Shape: {:?}, B_idx: {}, H_idx: {}", 
                offset, s_dim*d_dim, self.data.len(), self.shape, batch_idx, head_idx)
            )));
        }
        slice_data.extend_from_slice(&self.data[offset .. offset + (s_dim * d_dim)]);
        Tensor::new(slice_data, vec![s_dim, d_dim]).map_err(TransformerError::from)
    }

    fn slice_one_head_all_batches(&self, head_idx: usize) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_one_head_all_batches expects 4D tensor [B,H,S,D]".into()))); }
        let b = self.shape[0]; let h_total = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        if head_idx >= h_total { return Err(TransformerError::TensorError(TensorError::OutOfBounds("head_idx out of bounds".into())));}

        let mut new_data = Vec::with_capacity(b * s * d);
        let new_shape = vec![b, s, d];

        for b_i in 0..b {
            for s_i in 0..s {
                for d_i in 0..d {
                    let old_idx = b_i*h_total*s*d + head_idx*s*d + s_i*d + d_i;
                    new_data.push(self.data[old_idx]);
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }
    
    fn stack_heads_from_list(head_tensors: &[Tensor<f32>]) -> Result<Tensor<f32>, TransformerError> {
        if head_tensors.is_empty() {
            return Err(TransformerError::TensorError(TensorError::InvalidDimension("Cannot stack empty list of head tensors if n_head > 0".into())));
        }
        let first_head = &head_tensors[0];
        if first_head.rank() != 3 {return Err(TransformerError::TensorError(TensorError::InvalidDimension("Head tensors for stacking must be 3D [B,S,D]".into())));}
        
        let b = first_head.shape[0]; let s = first_head.shape[1]; let d = first_head.shape[2];
        let h = head_tensors.len();
        let new_shape = vec![b,h,s,d];
        let total_elements = b*h*s*d;
        let mut new_data = if total_elements > 0 { vec![0.0; total_elements] } else { Vec::new() };

        for (h_idx, head_tensor) in head_tensors.iter().enumerate() {
            if head_tensor.shape != first_head.shape { 
                return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(
                    format!("All head tensors must have the same shape for stacking. Expected {:?}, got {:?} for head {}", 
                    first_head.shape, head_tensor.shape, h_idx)
                )));
            }
            if head_tensor.data.is_empty() && head_tensor.num_elements() > 0 {
                 return Err(TransformerError::TensorError(TensorError::ShapeMismatch(format!("Head tensor {} has non-zero shape product but empty data.", h_idx))));
            }
            if head_tensor.data.is_empty() { continue; } 

            for b_i in 0..b {
                for s_i in 0..s {
                    for d_i in 0..d {
                        let val_idx_in_head_tensor = b_i*s*d + s_i*d + d_i;
                        let target_idx_in_new_data = b_i*h*s*d + h_idx*s*d + s_i*d + d_i;
                        new_data[target_idx_in_new_data] = head_tensor.data[val_idx_in_head_tensor];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }

    fn slice_mha_for_kv(&self, batch_idx: usize, head_idx: usize, kv_seq_len: usize) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_mha_for_kv expects 4D tensor [B,H,S_kv,D]".into()))); }
        let d_dim = self.shape[3];
        let mut slice_data = Vec::with_capacity(kv_seq_len * d_dim);
        let h_total_in_tensor = self.shape[1];

        if self.shape[2] != kv_seq_len {
             return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(
                format!("KV tensor sequence length {} does not match expected kv_seq_len {} in slice_mha_for_kv. Full tensor shape: {:?}", self.shape[2], kv_seq_len, self.shape)
            )));
        }

        let offset = batch_idx * h_total_in_tensor * kv_seq_len * d_dim + head_idx * kv_seq_len * d_dim;

        if self.data.is_empty() && (kv_seq_len * d_dim > 0) {
             return Err(TransformerError::TensorError(TensorError::ShapeMismatch("slice_mha_for_kv: Input data is empty but slice size is non-zero".into())));
        }
        if self.data.is_empty() && kv_seq_len * d_dim == 0 { 
             return Tensor::new(Vec::new(), vec![kv_seq_len, d_dim]).map_err(TransformerError::from);
        }

        if offset + (kv_seq_len * d_dim) > self.data.len() {
             return Err(TransformerError::TensorError(TensorError::OutOfBounds(
                format!("slice_mha_for_kv: Offset {} + slice_size {} > data_len {}. Shape: {:?}, B_idx: {}, H_idx: {}", 
                offset, kv_seq_len*d_dim, self.data.len(), self.shape, batch_idx, head_idx)
            )));
        }
        slice_data.extend_from_slice(&self.data[offset .. offset + (kv_seq_len * d_dim)]);
        Tensor::new(slice_data, vec![kv_seq_len, d_dim]).map_err(TransformerError::from)
    }
    
    fn slice_mha_custom(&self, batch_idx: usize, head_idx: usize, q_seq_len: usize, kv_seq_len: usize) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_mha_custom expects 4D tensor [B,H,S_q,S_kv]".into()))); }
        if self.shape[2] != q_seq_len || self.shape[3] != kv_seq_len {
            return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(
                format!("Tensor S_q or S_kv dims ({},{}) do not match expected ({},{}) in slice_mha_custom. Full tensor shape: {:?}", 
                        self.shape[2], self.shape[3], q_seq_len, kv_seq_len, self.shape)
            )));
        }

        let mut slice_data = Vec::with_capacity(q_seq_len * kv_seq_len);
        let h_total_in_tensor = self.shape[1];
        let offset = batch_idx * h_total_in_tensor * q_seq_len * kv_seq_len + head_idx * q_seq_len * kv_seq_len;

        if self.data.is_empty() && (q_seq_len * kv_seq_len > 0) {
            return Err(TransformerError::TensorError(TensorError::ShapeMismatch("slice_mha_custom: Input data is empty but slice size is non-zero".into())));
        }
        if self.data.is_empty() && q_seq_len * kv_seq_len == 0 {
            return Tensor::new(Vec::new(), vec![q_seq_len, kv_seq_len]).map_err(TransformerError::from);
        }

        if offset + (q_seq_len * kv_seq_len) > self.data.len() {
             return Err(TransformerError::TensorError(TensorError::OutOfBounds(
                format!("slice_mha_custom: Offset {} + slice_size {} > data_len {}. Shape: {:?}, B_idx: {}, H_idx: {}", 
                offset, q_seq_len*kv_seq_len, self.data.len(), self.shape, batch_idx, head_idx)
            )));
        }
        slice_data.extend_from_slice(&self.data[offset .. offset + (q_seq_len * kv_seq_len)]);
        Tensor::new(slice_data, vec![q_seq_len, kv_seq_len]).map_err(TransformerError::from)
    }
}

impl Tensor<f32> {
    pub fn transpose_data_generic(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        if rows * cols != data.len() && !data.is_empty() { 
            eprintln!("Warning: Transpose data length mismatch. Data len: {}, rows*cols: {}", data.len(), rows*cols);
            return data.to_vec(); 
        }
        if data.is_empty() { return Vec::new(); }

        let mut new_data = vec![0.0; data.len()];
        for r in 0..rows {
            for c_idx in 0..cols { 
                new_data[c_idx * rows + r] = data[r * cols + c_idx];
            }
        }
        new_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_config() -> Config {
        Config {
            n_layer: 1, 
            n_head: 2,
            n_embd: 4, 
            vocab_size: 10,
            block_size: 8,
            bias: true,
        }
    }
    
    fn create_dummy_weights_for_model(config: &Config, use_split_qkv: bool) -> HashMap<String, Tensor<f32>> {
        let mut weights = HashMap::new();
        let n_embd = config.n_embd;
        let vocab_size = config.vocab_size;
        let block_size = config.block_size;

        weights.insert("wte.weight".to_string(), Tensor::zeros(vec![vocab_size, n_embd]));
        weights.insert("wpe.weight".to_string(), Tensor::zeros(vec![block_size, n_embd]));

        for i in 0..config.n_layer {
            let prefix = format!("h.{}.attn.", i);
            if use_split_qkv {
                weights.insert(format!("{}w_q.weight", prefix), Tensor::zeros(vec![n_embd, n_embd]));
                weights.insert(format!("{}w_q.bias", prefix), Tensor::zeros(vec![n_embd]));
                weights.insert(format!("{}w_k.weight", prefix), Tensor::zeros(vec![n_embd, n_embd]));
                weights.insert(format!("{}w_k.bias", prefix), Tensor::zeros(vec![n_embd]));
                weights.insert(format!("{}w_v.weight", prefix), Tensor::zeros(vec![n_embd, n_embd]));
                weights.insert(format!("{}w_v.bias", prefix), Tensor::zeros(vec![n_embd]));
            } else {
                weights.insert(format!("{}c_attn.weight", prefix), Tensor::zeros(vec![n_embd, 3 * n_embd]));
                weights.insert(format!("{}c_attn.bias", prefix), Tensor::zeros(vec![3 * n_embd]));
            }
            weights.insert(format!("{}c_proj.weight", prefix), Tensor::zeros(vec![n_embd, n_embd]));
            weights.insert(format!("{}c_proj.bias", prefix), Tensor::zeros(vec![n_embd]));
            
            weights.insert(format!("h.{}.mlp.c_fc.weight", i), Tensor::zeros(vec![n_embd, 4 * n_embd]));
            weights.insert(format!("h.{}.mlp.c_fc.bias", i), Tensor::zeros(vec![4 * n_embd]));
            weights.insert(format!("h.{}.mlp.c_proj.weight", i), Tensor::zeros(vec![4 * n_embd, n_embd]));
            weights.insert(format!("h.{}.mlp.c_proj.bias", i), Tensor::zeros(vec![n_embd]));

            weights.insert(format!("h.{}.ln_1.weight", i), Tensor::zeros(vec![n_embd])); 
            weights.insert(format!("h.{}.ln_1.bias", i), Tensor::zeros(vec![n_embd]));   
            weights.insert(format!("h.{}.ln_2.weight", i), Tensor::zeros(vec![n_embd])); 
            weights.insert(format!("h.{}.ln_2.bias", i), Tensor::zeros(vec![n_embd]));   
        }
        weights.insert("ln_f.weight".to_string(), Tensor::zeros(vec![n_embd])); 
        weights.insert("ln_f.bias".to_string(), Tensor::zeros(vec![n_embd]));   
        weights
    }

    #[test]
    fn test_config_creation() {
        let config = create_dummy_config();
        assert_eq!(config.n_layer, 1);
        assert_eq!(config.head_dim(), 2); 
    }

    #[test]
    fn test_mha_creation_valid_split_weights() {
        let config = Arc::new(create_dummy_config());
        let mut weights = create_dummy_weights_for_model(&config, true); 
        let mha = MultiHeadAttention::new(Arc::clone(&config), &mut weights, "h.0.attn.");
        assert!(mha.is_ok(), "MHA creation failed with split weights: {:?}", mha.err());
    }

    #[test]
    fn test_mha_creation_valid_combined_weights_fallback() {
        let config = Arc::new(create_dummy_config());
        let mut weights = create_dummy_weights_for_model(&config, false); 

        let mha = MultiHeadAttention::new(Arc::clone(&config), &mut weights, "h.0.attn.");
        assert!(mha.is_ok(), "MHA creation failed with combined weights fallback: {:?}", mha.err());
        assert!(weights.get("h.0.attn.c_attn.weight").is_none(), "c_attn.weight should be consumed");
    }
    
    #[test]
    fn test_mha_creation_missing_weights() {
        let config = Arc::new(create_dummy_config());
        let mut weights = HashMap::new(); 
        weights.insert("h.0.attn.c_proj.weight".to_string(), Tensor::zeros(vec![config.n_embd, config.n_embd])); 
        weights.insert("h.0.attn.c_proj.bias".to_string(), Tensor::zeros(vec![config.n_embd]));

        let mha = MultiHeadAttention::new(Arc::clone(&config), &mut weights, "h.0.attn.");
        assert!(mha.is_err());
        match mha.err().unwrap() {
            TransformerError::WeightNotFound(s) => {
                assert!(s.contains("w_q.weight") || s.contains("c_attn.weight"));
            },
            e => panic!("Unexpected error type for missing MHA weights: {:?}", e),
        }
    }

    #[test]
    fn test_block_creation_valid() {
        let config_obj = create_dummy_config();
        let config = Arc::new(config_obj); 
        let mut weights_for_block = create_dummy_weights_for_model(&config, true); 
        
        let attn_prefix = "h.0.attn.";
        let attn = MultiHeadAttention::new(Arc::clone(&config), &mut weights_for_block, attn_prefix).unwrap();
        
        let n_embd = config.n_embd;
        let mlp_c_fc_w = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.mlp.c_fc.weight"), Some(&[n_embd, 4*n_embd])).unwrap();
        let mlp_c_fc_b = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.mlp.c_fc.bias"), Some(&[4*n_embd])).unwrap();
        let mlp_c_proj_w = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.mlp.c_proj.weight"), Some(&[4*n_embd, n_embd])).unwrap();
        let mlp_c_proj_b = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.mlp.c_proj.bias"), Some(&[n_embd])).unwrap();
        let mlp = FeedForward::new(mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b, &config).unwrap(); 

        let ln_1_g = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.ln_1.weight"), Some(&[n_embd])).unwrap();
        let ln_1_b = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.ln_1.bias"), Some(&[n_embd])).unwrap();
        let ln_2_g = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.ln_2.weight"), Some(&[n_embd])).unwrap();
        let ln_2_b = GPT2Model::get_weight(&mut weights_for_block, &format!("h.0.ln_2.bias"), Some(&[n_embd])).unwrap();

        let block = Block::new(attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b, &config); 
        assert!(block.is_ok());
    }

    #[test]
    fn test_gpt2model_creation_valid_split_qkv_weights() {
        let config = create_dummy_config();
        let weights = create_dummy_weights_for_model(&config, true); 
        let model = GPT2Model::new(config, weights);
        assert!(model.is_ok(), "Model creation with split QKV weights failed: {:?}", model.err());
    }

    #[test]
    fn test_gpt2model_creation_valid_combined_qkv_weights() {
        let config = create_dummy_config();
        let weights = create_dummy_weights_for_model(&config, false); 
        let model = GPT2Model::new(config, weights);
        assert!(model.is_ok(), "Model creation with combined QKV weights (fallback) failed: {:?}", model.err());
    }

    #[test]
    fn test_gpt2model_creation_missing_weight_error() {
        let config = create_dummy_config();
        let mut weights = create_dummy_weights_for_model(&config, true); 
        weights.remove("wte.weight"); 
        let model = GPT2Model::new(config, weights);
        assert!(model.is_err());
        match model.err().unwrap() {
            TransformerError::WeightNotFound(s) => assert_eq!(s, "wte.weight"),
            e => panic!("Unexpected error type for missing weight: {:?}", e),
        }
    }
    
    #[test]
    fn test_gpt2model_creation_wrong_weight_shape_error() {
        let config = create_dummy_config();
        let mut weights = create_dummy_weights_for_model(&config, true);
        weights.insert("ln_f.bias".to_string(), Tensor::zeros(vec![config.n_embd + 1])); 
        let model = GPT2Model::new(config, weights);
        assert!(model.is_err());
        match model.err().unwrap() {
            TransformerError::InvalidWeightShape(s) => assert!(s.contains("ln_f.bias shape mismatch")),
            e => panic!("Unexpected error type for wrong weight shape: {:?}", e),
        }
    }

    #[test]
    fn test_gpt2model_forward_pass_mocked() {
        let config_obj = create_dummy_config();
        let weights = create_dummy_weights_for_model(&config_obj, false); 
        let model = GPT2Model::new(config_obj.clone(), weights).expect("Model creation should succeed");

        let batch_size = 1;
        let seq_len = config_obj.block_size / 2; 
        
        let token_ids_data: Vec<u32> = (0..(batch_size * seq_len) as u32)
                                        .map(|i| i % (config_obj.vocab_size as u32))
                                        .collect();
        let token_ids = Tensor::new(token_ids_data, vec![batch_size, seq_len]).unwrap();

        // Test without cache first
        let result_no_cache = model.forward(&token_ids, None, Some(1.0), None); 
        assert!(result_no_cache.is_ok(), "Forward pass (no cache) failed: {:?}", result_no_cache.err());
        if let Ok(logits) = result_no_cache {
            assert_eq!(logits.shape, vec![batch_size, seq_len, config_obj.vocab_size]);
        }

        // Test with cache
        let mut model_cache = ModelKVCache::new(); // Create an empty cache
        // Initialize model_cache for the first pass (GPT2Model::forward handles this internally now)
        // model.forward will initialize it if it's empty and passed as Some(&mut).
        
        // Pass 1 (Prefill)
        let result_cache_pass1 = model.forward(&token_ids, None, Some(1.0), Some(&mut model_cache));
        assert!(result_cache_pass1.is_ok(), "Forward pass (cache pass 1) failed: {:?}", result_cache_pass1.err());
         if let Ok(logits) = result_cache_pass1 {
            assert_eq!(logits.shape, vec![batch_size, seq_len, config_obj.vocab_size]);
        }

        // Pass 2 (Generate one new token)
        let next_token_id_data: Vec<u32> = vec![0]; // Dummy next token
        let next_token_ids = Tensor::new(next_token_id_data, vec![batch_size, 1]).unwrap();
        let result_cache_pass2 = model.forward(&next_token_ids, None, Some(1.0), Some(&mut model_cache));
        assert!(result_cache_pass2.is_ok(), "Forward pass (cache pass 2) failed: {:?}", result_cache_pass2.err());
        if let Ok(logits_pass2) = result_cache_pass2 {
            // Logits for the new token only
            assert_eq!(logits_pass2.shape, vec![batch_size, 1, config_obj.vocab_size]);
        }
        
        // Check if cache has been populated
        assert_eq!(model_cache.len(), config_obj.n_layer);
        if !model_cache.is_empty() {
            assert_eq!(model_cache[0].len(), config_obj.n_head);
            if !model_cache[0].is_empty() {
                let past_seq_len = model_cache[0][0].key.shape[1];
                assert_eq!(past_seq_len, seq_len + 1); // seq_len from first pass + 1 from second pass
            }
        }
    }
}
