// src/transformer_core.rs

use crate::tensor_engine::{Tensor, TensorError};
use half::f16; // Added for f16 support
use std::collections::HashMap;
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

// Model Data Type Enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDataType {
    F32,
    F16,
    BF16,
}

impl Default for ModelDataType {
    fn default() -> Self {
        ModelDataType::F32
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
    pub dtype: ModelDataType,
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

    // F32 Ops
    pub fn matmul_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        a.matmul(b)
    }

    pub fn softmax_f32(a: &mut Tensor<f32>, axis: usize) -> Result<(), TensorError> {
        a.softmax(Some(axis))
    }

    pub fn layernorm_f32(a: &mut Tensor<f32>, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<(), TensorError> {
        a.layernorm(gamma, beta, epsilon)
    }

    pub fn gelu_f32(a: &mut Tensor<f32>) -> Result<(), TensorError> {
        a.gelu() 
    }
    
    pub fn add_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
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
            return Err(TensorError::DimensionMismatch(format!(
                "Element-wise add_f32 requires identical shapes or broadcastable bias. Got {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let data = a.data.iter().zip(b.data.iter()).map(|(av, bv)| av + bv).collect();
        Tensor::new(data, a.shape.clone())
    }

    // F16 Ops
    pub fn matmul_f16(a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, TensorError> {
        a.matmul(b)
    }

    pub fn softmax_f16(a: &mut Tensor<f16>, axis: usize) -> Result<(), TensorError> {
        a.softmax(Some(axis))
    }

    pub fn layernorm_f16(a: &mut Tensor<f16>, gamma: &Tensor<f16>, beta: &Tensor<f16>, epsilon: f32) -> Result<(), TensorError> {
        a.layernorm(gamma, beta, epsilon)
    }

    pub fn gelu_f16(a: &mut Tensor<f16>) -> Result<(), TensorError> {
        a.gelu()
    }

    pub fn add_f16(a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, TensorError> {
        if a.shape != b.shape {
            if b.rank() == 1 && a.shape.last() == b.shape.last() && a.rank() > 1 {
                let mut out_data = a.data.clone();
                let last_dim_size = b.shape[0];
                let num_vectors = a.data.len() / last_dim_size;
                for i in 0..num_vectors {
                    for j_idx in 0..last_dim_size {
                        out_data[i * last_dim_size + j_idx] = out_data[i * last_dim_size + j_idx] + b.data[j_idx];
                    }
                }
                return Tensor::new(out_data, a.shape.clone());
            }
            return Err(TensorError::DimensionMismatch(format!(
                "Element-wise add_f16 requires identical shapes or broadcastable bias. Got {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let data: Vec<f16> = a.data.iter().zip(b.data.iter()).map(|(&av, &bv)| av + bv).collect();
        Tensor::new(data, a.shape.clone())
    }

    pub fn scalar_mul_f16(a: &mut Tensor<f16>, scalar: f16) -> Result<(), TensorError> {
        a.scalar_mul(scalar)
    }
    
    pub fn split_last_dim<T: Clone + Default + std::fmt::Debug + PartialEq + Send + Sync + Copy>(
        tensor: &Tensor<T>, num_chunks: usize
    ) -> Result<Vec<Tensor<T>>, TensorError> {
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
                    chunks.push(Tensor::new(Vec::<T>::new(), new_shape.clone())?);
                }
                return Ok(chunks);
            } else if last_dim_size % num_chunks == 0 {
                let mut new_shape = tensor.shape.clone();
                 if !new_shape.is_empty() { new_shape[last_dim_idx] = last_dim_size / num_chunks; }
                let mut chunks = Vec::new();
                for _ in 0..num_chunks {
                    chunks.push(Tensor::new(Vec::<T>::new(), new_shape.clone())?);
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

    pub fn split_dim1<T: Clone + Default + std::fmt::Debug + PartialEq + Send + Sync + Copy>(
        tensor: &Tensor<T>, num_chunks: usize
    ) -> Result<Vec<Tensor<T>>, TensorError> {
        if tensor.rank() != 2 {
            return Err(TensorError::InvalidDimension("split_dim1 expects a 2D tensor".to_string()));
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];

        if cols == 0 && num_chunks > 0 { 
            let mut chunks = Vec::new();
            for _ in 0..num_chunks {
                chunks.push(Tensor::new(Vec::<T>::new(), vec![rows, 0])?);
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
                result_tensors.push(Tensor::new(Vec::<T>::new(), vec![rows, chunk_cols])?);
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

    pub fn linear_f16(input: &Tensor<f16>, weight: &Tensor<f16>, bias: Option<&Tensor<f16>>) -> Result<Tensor<f16>, TransformerError> {
        let din = weight.shape[0];
        let dout = weight.shape[1];
        if input.shape.last().unwrap_or(&0) != &din {
            return Err(TransformerError::TensorError(TensorError::DimensionMismatch(format!(
                "Linear_f16 input last dim {} != weight first dim {}",
                input.shape.last().unwrap_or(&0), din
            ))));
        }

        let mut out_shape = input.shape.clone();
        if out_shape.is_empty() && input.num_elements() == din { 
             out_shape = vec![1, din]; 
        } else if out_shape.is_empty() { 
            return Err(TransformerError::TensorError(TensorError::InvalidDimension("Scalar input not directly usable in linear_f16 layer without proper shape".to_string())));
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

        let mut output = input_reshaped.matmul(weight)?;

        if let Some(b) = bias {
            if b.rank() != 1 || b.shape[0] != dout {
                 return Err(TransformerError::TensorError(TensorError::DimensionMismatch(format!("Bias_f16 shape {:?} incompatible with output dim {}", b.shape, dout))));
            }
            output = add_f16(&output, b)?;

        }
        
        if original_rank > 2 || (original_rank == 1 && input.shape[0] == din) { 
            output = output.reshape(out_shape)?;
        }
        Ok(output)
    }

    pub fn embedding_f16(ids: &Tensor<u32>, weight: &Tensor<f16>) -> Result<Tensor<f16>, TransformerError> {
        if weight.rank() != 2 {
            return Err(TransformerError::TensorError(TensorError::InvalidDimension("Embedding_f16 weight matrix must be 2D".to_string())));
        }
        let vocab_size = weight.shape[0];
        let emb_dim = weight.shape[1];

        let mut out_shape = ids.shape.clone();
        out_shape.push(emb_dim);

        let mut out_data = Vec::with_capacity(ids.num_elements() * emb_dim);

        for id_val in &ids.data {
            let id_idx = *id_val as usize;
            if id_idx >= vocab_size {
                return Err(TransformerError::TensorError(TensorError::OutOfBounds(format!("Token ID {} out of vocab size {}", id_idx, vocab_size))));
            }
            let embedding_vector_slice_start = id_idx * emb_dim;
            let embedding_vector_slice_end = embedding_vector_slice_start + emb_dim;
            out_data.extend_from_slice(&weight.data[embedding_vector_slice_start..embedding_vector_slice_end]);
        }
        Tensor::new(out_data, out_shape).map_err(TransformerError::from)
    }
}

// --- KV Cache Data Structures ---

#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub key: Tensor<f16>,
    pub value: Tensor<f16>,
}

pub type LayerKVCache = Vec<KVCacheEntry>; 
pub type ModelKVCache = Vec<LayerKVCache>;  

// MultiHeadAttention Module
pub struct MultiHeadAttention {
    w_q: Tensor<f16>,
    b_q: Tensor<f16>,
    w_k: Tensor<f16>,
    b_k: Tensor<f16>,
    w_v: Tensor<f16>,
    b_v: Tensor<f16>,
    c_proj_w: Tensor<f16>,
    c_proj_b: Tensor<f16>,
    config: Arc<Config>,
}

impl MultiHeadAttention {
    pub fn new(
        config: Arc<Config>,
        // These weights are expected to be f16 already, converted by GPT2Model::new
        w_q: Tensor<f16>, b_q: Tensor<f16>,
        w_k: Tensor<f16>, b_k: Tensor<f16>,
        w_v: Tensor<f16>, b_v: Tensor<f16>,
        c_proj_w: Tensor<f16>, c_proj_b: Tensor<f16>,
    ) -> Result<Self, TransformerError> {
        Ok(MultiHeadAttention {
            w_q, b_q,
            w_k, b_k,
            w_v, b_v,
            c_proj_w, c_proj_b,
            config,
        })
    }

    pub fn forward(
        &self, 
        x_f16: &Tensor<f16>,
        mask: Option<&Tensor<f32>>,
        theta_hat: Option<f32>,
        cache: Option<&mut LayerKVCache> 
    ) -> Result<Tensor<f16>, TransformerError> {
        let batch_size = x_f16.shape[0];
        let current_seq_len = x_f16.shape[1];
        let n_embd = self.config.n_embd;
        let n_head = self.config.n_head;
        let head_dim = self.config.head_dim();

        let q_all = tensor_ops::linear_f16(x_f16, &self.w_q, Some(&self.b_q))?;
        let k_all_current_input = tensor_ops::linear_f16(x_f16, &self.w_k, Some(&self.b_k))?;
        let v_all_current_input = tensor_ops::linear_f16(x_f16, &self.w_v, Some(&self.b_v))?;

        let q_heads = q_all.reshape(vec![batch_size, current_seq_len, n_head, head_dim])?
                           .permute_mha_qkv()?; 
        let k_heads_current_f16 = k_all_current_input.reshape(vec![batch_size, current_seq_len, n_head, head_dim])?
                                     .permute_mha_qkv()?;
        let v_heads_current_f16 = v_all_current_input.reshape(vec![batch_size, current_seq_len, n_head, head_dim])?
                                     .permute_mha_qkv()?;

        let k_for_attention_all_heads: Tensor<f16>;
        let v_for_attention_all_heads: Tensor<f16>;
        let mut effective_kv_seq_len = current_seq_len; 

        if let Some(layer_cache_mut) = cache {
            let mut temp_k_list = Vec::with_capacity(n_head);
            let mut temp_v_list = Vec::with_capacity(n_head);
            
            while layer_cache_mut.len() < n_head {
                 layer_cache_mut.push(KVCacheEntry { 
                    key: Tensor::new(Vec::<f16>::new(), vec![batch_size, 0, head_dim])?,
                    value: Tensor::new(Vec::<f16>::new(), vec![batch_size, 0, head_dim])?
                });
            }
            
            for h_idx in 0..n_head {
                let k_current_this_head_f16 = k_heads_current_f16.slice_one_head_all_batches(h_idx)?;
                let v_current_this_head_f16 = v_heads_current_f16.slice_one_head_all_batches(h_idx)?;

                let cache_entry = &mut layer_cache_mut[h_idx];
                
                let k_to_cache = if cache_entry.key.shape[1] == 0 { 
                    k_current_this_head_f16.clone()
                } else {
                    Tensor::concat(&[&cache_entry.key, &k_current_this_head_f16], 1)?
                };
                let v_to_cache = if cache_entry.value.shape[1] == 0 {
                    v_current_this_head_f16.clone()
                } else {
                    Tensor::concat(&[&cache_entry.value, &v_current_this_head_f16], 1)?
                };
                
                cache_entry.key = k_to_cache;
                cache_entry.value = v_to_cache;

                temp_k_list.push(cache_entry.key.clone());
                temp_v_list.push(cache_entry.value.clone());
            }
            k_for_attention_all_heads = Tensor::stack_heads_from_list_f16(&temp_k_list)?;
            v_for_attention_all_heads = Tensor::stack_heads_from_list_f16(&temp_v_list)?;
            if !temp_k_list.is_empty() {
                effective_kv_seq_len = temp_k_list[0].shape[1]; 
            }
        } else {
            k_for_attention_all_heads = k_heads_current_f16;
            v_for_attention_all_heads = v_heads_current_f16;
        }
        
        let k_t_final = k_for_attention_all_heads.permute_mha_kt()?;
        
        let mut att_scores_parts = Vec::with_capacity(batch_size * n_head);
        let scale_f16 = f16::from_f32((head_dim as f32).sqrt());
        let inv_scale_f16 = if scale_f16 == f16::from_f32(0.0) { f16::from_f32(1.0) } else { f16::from_f32(1.0) / scale_f16 };


        for b_idx in 0..batch_size {
            for h_idx in 0..n_head {
                let q_slice = q_heads.slice_mha(b_idx, h_idx)?; 
                let k_t_slice = k_t_final.slice_mha(b_idx, h_idx)?; 
                
                let mut scores_s = tensor_ops::matmul_f16(&q_slice, &k_t_slice)?;
                tensor_ops::scalar_mul_f16(&mut scores_s, inv_scale_f16)?;
                att_scores_parts.push(scores_s);
            }
        }
        let mut att_scores_data_flat = Vec::new();
        for t in att_scores_parts { att_scores_data_flat.extend(t.data); }
        let mut att_scores = Tensor::new(att_scores_data_flat, vec![batch_size, n_head, current_seq_len, effective_kv_seq_len])?;

        if let Some(th_value) = theta_hat {
            tensor_ops::scalar_mul_f16(&mut att_scores, f16::from_f32(th_value))?;
        }

        if let Some(m) = mask { 
             if m.shape == vec![current_seq_len, effective_kv_seq_len] { 
                for b in 0..batch_size {
                    for h in 0..n_head {
                        for s_q_idx in 0..current_seq_len {      
                            for s_kv_idx in 0..effective_kv_seq_len { 
                                if *m.get(&[s_q_idx, s_kv_idx])? == 0.0 { 
                                    let val_ref = att_scores.get_mut(&[b,h,s_q_idx,s_kv_idx])?;
                                    *val_ref = f16::NEG_INFINITY;
                                }
                            }
                        }
                    }
                }
            } else {
                 if current_seq_len == effective_kv_seq_len && m.shape == vec![current_seq_len, current_seq_len] {
                    for b in 0..batch_size {
                        for h in 0..n_head {
                            for s1 in 0..current_seq_len {
                                for s2 in 0..current_seq_len {
                                    if *m.get(&[s1, s2])? == 0.0 { 
                                        let val_ref = att_scores.get_mut(&[b,h,s1,s2])?;
                                        *val_ref = f16::NEG_INFINITY;
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

        tensor_ops::softmax_f16(&mut att_scores, 3)?;

        let mut out_att_parts = Vec::with_capacity(batch_size * n_head);
        for b_idx in 0..batch_size {
            for h_idx in 0..n_head {
                let probs_slice = att_scores.slice_mha_custom(b_idx, h_idx, current_seq_len, effective_kv_seq_len)?;
                let v_slice = v_for_attention_all_heads.slice_mha_for_kv(b_idx, h_idx, effective_kv_seq_len)?; 
                out_att_parts.push(tensor_ops::matmul_f16(&probs_slice, &v_slice)?);
            }
        }
        let mut out_att_data_flat = Vec::new();
        for t in out_att_parts { out_att_data_flat.extend(t.data); }
        let out_att = Tensor::new(out_att_data_flat, vec![batch_size, n_head, current_seq_len, head_dim])?;
        
        let out_reshaped = out_att.permute_mha_output()? 
                                  .reshape(vec![batch_size, current_seq_len, n_embd])?;
        
        let final_output = tensor_ops::linear_f16(&out_reshaped, &self.c_proj_w, Some(&self.c_proj_b))?;
        
        Ok(final_output)
    }
}

pub struct FeedForward {
    c_fc_w: Tensor<f16>,
    c_fc_b: Tensor<f16>,
    c_proj_w: Tensor<f16>,
    c_proj_b: Tensor<f16>,
}

impl FeedForward {
    pub fn new(
        // Expecting f16 weights directly from GPT2Model::new
        c_fc_w: Tensor<f16>,
        c_fc_b: Tensor<f16>,
        c_proj_w: Tensor<f16>,
        c_proj_b: Tensor<f16>,
        config: &Config // Keep config for validation if needed, though not strictly used if shapes are pre-validated
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        let n_hidden = 4 * n_embd; 

        // Validate shapes of incoming f16 tensors
        if c_fc_w.shape != vec![n_embd, n_hidden] {
            return Err(TransformerError::InvalidWeightShape(format!("c_fc_w (f16) shape mismatch: expected [{}, {}], got {:?}", n_embd, n_hidden, c_fc_w.shape)));
        }
        if c_fc_b.shape != vec![n_hidden] {
            return Err(TransformerError::InvalidWeightShape(format!("c_fc_b (f16) shape mismatch: expected [{}], got {:?}", n_hidden, c_fc_b.shape)));
        }
        if c_proj_w.shape != vec![n_hidden, n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_w (f16) shape mismatch: expected [{}, {}], got {:?}", n_hidden, n_embd, c_proj_w.shape)));
        }
        if c_proj_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_b (f16) shape mismatch: expected [{}], got {:?}", n_embd, c_proj_b.shape)));
        }
        
        Ok(FeedForward { c_fc_w, c_fc_b, c_proj_w, c_proj_b })
    }

    pub fn forward(&self, x: &Tensor<f16>) -> Result<Tensor<f16>, TransformerError> {
        let mut h = tensor_ops::linear_f16(x, &self.c_fc_w, Some(&self.c_fc_b))?;
        tensor_ops::gelu_f16(&mut h)?;
        let output = tensor_ops::linear_f16(&h, &self.c_proj_w, Some(&self.c_proj_b))?;
        Ok(output)
    }
}

pub struct Block {
    attn: MultiHeadAttention,
    mlp: FeedForward,
    ln_1_g: Tensor<f16>,
    ln_1_b: Tensor<f16>,
    ln_2_g: Tensor<f16>,
    ln_2_b: Tensor<f16>,
}

impl Block {
    pub fn new(
        attn: MultiHeadAttention, 
        mlp: FeedForward, 
        // Expecting f16 LayerNorm weights
        ln_1_g: Tensor<f16>,
        ln_1_b: Tensor<f16>,
        ln_2_g: Tensor<f16>,
        ln_2_b: Tensor<f16>,
        config: &Config 
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        // Validate shapes of incoming f16 tensors
        if ln_1_g.shape != vec![n_embd] || ln_1_b.shape != vec![n_embd] ||
           ln_2_g.shape != vec![n_embd] || ln_2_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape("LayerNorm (f16) weight shape mismatch".to_string()));
        }

        Ok(Block { attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b })
    }

    pub fn forward(
        &self, 
        x: &Tensor<f16>,
        mask: Option<&Tensor<f32>>, 
        theta_hat: Option<f32>, 
        layer_cache: Option<&mut LayerKVCache>
    ) -> Result<Tensor<f16>, TransformerError> {
        let mut x_norm1 = x.clone();
        tensor_ops::layernorm_f16(&mut x_norm1, &self.ln_1_g, &self.ln_1_b, 1e-5)?;

        let attn_output = self.attn.forward(&x_norm1, mask, theta_hat, layer_cache)?; 
        let x_plus_attn = tensor_ops::add_f16(x, &attn_output)?;
        
        let mut x_norm2 = x_plus_attn.clone();
        tensor_ops::layernorm_f16(&mut x_norm2, &self.ln_2_g, &self.ln_2_b, 1e-5)?;
        let mlp_output = self.mlp.forward(&x_norm2)?;
        let final_output = tensor_ops::add_f16(&x_plus_attn, &mlp_output)?;
        
        Ok(final_output)
    }
}

pub struct GPT2Model {
    pub config: Arc<Config>, 
    wte: Tensor<f16>,
    wpe: Tensor<f16>,
    blocks: Vec<Block>,
    ln_f_g: Tensor<f16>,
    ln_f_b: Tensor<f16>,
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

        // Load f32 weights then convert to f16 for storage
        let wte_f32 = Self::get_weight(&mut weights, "wte.weight", Some(&[vocab_size, n_embd]))?;
        let wpe_f32 = Self::get_weight(&mut weights, "wpe.weight", Some(&[block_size, n_embd]))?;

        let wte = wte_f32.to_f16_tensor()?;
        let wpe = wpe_f32.to_f16_tensor()?;
        
        let mut blocks_vec = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let prefix_attn = format!("h.{}.attn.", i);
            let w_q_f32: Tensor<f32>; let b_q_f32: Tensor<f32>;
            let w_k_f32: Tensor<f32>; let b_k_f32: Tensor<f32>;
            let w_v_f32: Tensor<f32>; let b_v_f32: Tensor<f32>;

            let w_q_key = format!("{}w_q.weight", prefix_attn);
            if weights.contains_key(&w_q_key) {
                w_q_f32 = Self::get_weight(&mut weights, &w_q_key, Some(&[n_embd, n_embd]))?;
                b_q_f32 = Self::get_weight(&mut weights, &format!("{}w_q.bias", prefix_attn), Some(&[n_embd]))?;
                w_k_f32 = Self::get_weight(&mut weights, &format!("{}w_k.weight", prefix_attn), Some(&[n_embd, n_embd]))?;
                b_k_f32 = Self::get_weight(&mut weights, &format!("{}w_k.bias", prefix_attn), Some(&[n_embd]))?;
                w_v_f32 = Self::get_weight(&mut weights, &format!("{}w_v.weight", prefix_attn), Some(&[n_embd, n_embd]))?;
                b_v_f32 = Self::get_weight(&mut weights, &format!("{}w_v.bias", prefix_attn), Some(&[n_embd]))?;
            } else {
                let c_attn_w_combined = Self::get_weight(&mut weights, &format!("{}c_attn.weight", prefix_attn), Some(&[n_embd, 3 * n_embd]))?;
                let c_attn_b_combined = Self::get_weight(&mut weights, &format!("{}c_attn.bias", prefix_attn), Some(&[3 * n_embd]))?;
                let mut qkv_w_parts = tensor_ops::split_dim1(&c_attn_w_combined, 3)?;
                w_q_f32 = qkv_w_parts.remove(0); w_k_f32 = qkv_w_parts.remove(0); w_v_f32 = qkv_w_parts.remove(0);
                let mut qkv_b_parts = tensor_ops::split_last_dim(&c_attn_b_combined, 3)?;
                b_q_f32 = qkv_b_parts.remove(0); b_k_f32 = qkv_b_parts.remove(0); b_v_f32 = qkv_b_parts.remove(0);
            }
            let c_proj_w_f32 = Self::get_weight(&mut weights, &format!("{}c_proj.weight", prefix_attn), Some(&[n_embd, n_embd]))?;
            let c_proj_b_f32 = Self::get_weight(&mut weights, &format!("{}c_proj.bias", prefix_attn), Some(&[n_embd]))?;

            let attn = MultiHeadAttention::new(
                Arc::clone(&conf),
                w_q_f32.to_f16_tensor()?, b_q_f32.to_f16_tensor()?,
                w_k_f32.to_f16_tensor()?, b_k_f32.to_f16_tensor()?,
                w_v_f32.to_f16_tensor()?, b_v_f32.to_f16_tensor()?,
                c_proj_w_f32.to_f16_tensor()?, c_proj_b_f32.to_f16_tensor()?
            )?;

            let mlp_c_fc_w_f32 = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_fc.weight", i), Some(&[n_embd, 4 * n_embd]))?;
            let mlp_c_fc_b_f32 = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_fc.bias", i), Some(&[4 * n_embd]))?;
            let mlp_c_proj_w_f32 = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_proj.weight", i), Some(&[4 * n_embd, n_embd]))?;
            let mlp_c_proj_b_f32 = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_proj.bias", i), Some(&[n_embd]))?;
            let mlp = FeedForward::new(
                mlp_c_fc_w_f32.to_f16_tensor()?, mlp_c_fc_b_f32.to_f16_tensor()?,
                mlp_c_proj_w_f32.to_f16_tensor()?, mlp_c_proj_b_f32.to_f16_tensor()?,
                &conf
            )?;

            let ln_1_g_f32 = Self::get_weight(&mut weights, &format!("h.{}.ln_1.weight", i), Some(&[n_embd]))?;
            let ln_1_b_f32 = Self::get_weight(&mut weights, &format!("h.{}.ln_1.bias", i), Some(&[n_embd]))?;
            let ln_2_g_f32 = Self::get_weight(&mut weights, &format!("h.{}.ln_2.weight", i), Some(&[n_embd]))?;
            let ln_2_b_f32 = Self::get_weight(&mut weights, &format!("h.{}.ln_2.bias", i), Some(&[n_embd]))?;
            
            blocks_vec.push(Block::new(
                attn, mlp,
                ln_1_g_f32.to_f16_tensor()?, ln_1_b_f32.to_f16_tensor()?,
                ln_2_g_f32.to_f16_tensor()?, ln_2_b_f32.to_f16_tensor()?,
                &conf
            )?);
        }

        let ln_f_g_f32 = Self::get_weight(&mut weights, "ln_f.weight", Some(&[n_embd]))?;
        let ln_f_b_f32 = Self::get_weight(&mut weights, "ln_f.bias", Some(&[n_embd]))?;

        if !weights.is_empty() {
            eprintln!("[GPT2Model::new] Warning: Unused weights: {:?}", weights.keys().collect::<Vec<_>>());
        }

        Ok(GPT2Model {
            config: conf,
            wte, wpe, // Already f16
            blocks: blocks_vec,
            ln_f_g: ln_f_g_f32.to_f16_tensor()?,
            ln_f_b: ln_f_b_f32.to_f16_tensor()?,
        })
    }

    pub fn forward(
        &self, 
        token_ids: &Tensor<u32>, 
        _mask: Option<&Tensor<f32>>,
        theta_hat: Option<f32>,
        mut model_cache: Option<&mut ModelKVCache>
    ) -> Result<Tensor<f16>, TransformerError> {
        let batch_size = token_ids.shape[0];
        let current_seq_len = token_ids.shape[1];

        if current_seq_len > self.config.block_size {
            return Err(TransformerError::ConfigError(format!(
                "Input sequence length {} exceeds model block size {}",
                current_seq_len, self.config.block_size
            )));
        }

        let token_embed = tensor_ops::embedding_f16(token_ids, &self.wte)?;
        
        let past_seq_len = if let Some(cache) = model_cache.as_ref() {
            if !cache.is_empty() && !cache[0].is_empty() && !cache[0][0].key.is_empty() && cache[0][0].key.shape.len() > 1 {
                cache[0][0].key.shape[1]
            } else { 0 }
        } else { 0 };

        let pos_ids_data: Vec<u32> = (past_seq_len .. past_seq_len + current_seq_len).map(|p| p as u32).collect();
        if pos_ids_data.is_empty() && current_seq_len > 0 {
             return Err(TransformerError::ConfigError("Position IDs vector is empty for non-zero sequence length.".to_string()));
        }

        let pos_ids_tensor_shape = if batch_size > 1 && current_seq_len > 0 { vec![batch_size, current_seq_len] } else { vec![current_seq_len] };
        let pos_ids_tensor_data = if batch_size > 1 && current_seq_len > 0 {
            pos_ids_data.iter().cycle().take(batch_size * current_seq_len).cloned().collect()
        } else {
            pos_ids_data
        };
        let pos_ids_tensor = Tensor::new(pos_ids_tensor_data, pos_ids_tensor_shape)?;
        
        let mut pos_embed_reshaped = tensor_ops::embedding_f16(&pos_ids_tensor, &self.wpe)?;

        let mut x : Tensor<f16>;
        if token_embed.num_elements() > 0 || pos_embed_reshaped.num_elements() > 0 {
            if pos_embed_reshaped.rank() == 2 && token_embed.rank() == 3 {
                if pos_embed_reshaped.shape[0] == token_embed.shape[1] && pos_embed_reshaped.shape[1] == token_embed.shape[2] {
                    let new_shape_for_pos = vec![1, pos_embed_reshaped.shape[0], pos_embed_reshaped.shape[1]];
                    pos_embed_reshaped = pos_embed_reshaped.reshape(new_shape_for_pos)?;
                } else if !(pos_embed_reshaped.num_elements() == 0 && token_embed.num_elements() == 0){
                    return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(
                        format!("Positional embedding shape {:?} not broadcastable to token embedding shape {:?} after attempting reshape to 3D",
                                pos_embed_reshaped.shape, token_embed.shape)
                    )));
                }
            }
            x = tensor_ops::add_f16(&token_embed, &pos_embed_reshaped)?;
        } else {
            x = token_embed;
        }

        let mut x_mut = x;

        let total_kv_seq_len = past_seq_len + current_seq_len;
        let attention_mask = if current_seq_len == 1 && past_seq_len > 0 {
            Tensor::new(vec![1.0f32; 1 * total_kv_seq_len], vec![1, total_kv_seq_len])?
        } else {
            let mut mask_data = vec![1.0f32; current_seq_len * total_kv_seq_len];
            for i in 0..current_seq_len {
                for j_idx in 0..total_kv_seq_len {
                    if j_idx > (past_seq_len + i) { 
                        mask_data[i * total_kv_seq_len + j_idx] = 0.0;
                    }
                }
            }
            Tensor::new(mask_data, vec![current_seq_len, total_kv_seq_len])?
        };

        if let Some(cache_mut) = model_cache.as_mut() {
            if cache_mut.is_empty() {
                for _ in 0..self.config.n_layer {
                    let mut layer_cache_init = Vec::with_capacity(self.config.n_head);
                    for _ in 0..self.config.n_head {
                        layer_cache_init.push(KVCacheEntry {
                            key: Tensor::new(Vec::<f16>::new(), vec![batch_size, 0, self.config.head_dim()])?,
                            value: Tensor::new(Vec::<f16>::new(), vec![batch_size, 0, self.config.head_dim()])?,
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
                    x_mut = block.forward(&x_mut, Some(&attention_mask), theta_hat, Some(&mut cache_mut[i]))?;
                }
            }
            None => {
                let simple_causal_mask = attention_mask;
                for block in &self.blocks {
                    x_mut = block.forward(&x_mut, Some(&simple_causal_mask), theta_hat, None)?;
                }
            }
        }
        
        let mut x_final_norm = x_mut;
        tensor_ops::layernorm_f16(&mut x_final_norm, &self.ln_f_g, &self.ln_f_b, 1e-5)?;

        let x_for_logits = if current_seq_len == 1 && past_seq_len > 0 {
            x_final_norm
        } else if current_seq_len > 1 && past_seq_len == 0 {
            x_final_norm
        } else if current_seq_len == 0 {
            return Tensor::new(Vec::new(), vec![batch_size, 0, self.config.vocab_size]).map_err(Into::into);
        } else {
            x_final_norm
        };
        
        let wte_t = {
            let wte_f16 = &self.wte;
            let data_f16 = Tensor::<f16>::transpose_data_generic_f16(&wte_f16.data, wte_f16.shape[0], wte_f16.shape[1]);
            Tensor::new(data_f16, vec![self.config.n_embd, self.config.vocab_size])?
        };
        
        let logits = tensor_ops::linear_f16(&x_for_logits, &wte_t, None)?;

        Ok(logits)
    }
}

// Trait for MHA specific tensor permutations and slicing.
// This might need to become generic or have f16 specific versions if internal MHA ops move to f16.
// For now, it's used by f32 tensors which are then converted if needed.
trait TensorExtMHA<T> { // Made generic over T
    fn permute_mha_qkv(&self) -> Result<Tensor<T>, TransformerError>;
    fn permute_mha_kt(&self) -> Result<Tensor<T>, TransformerError>;
    fn permute_mha_output(&self) -> Result<Tensor<T>, TransformerError>;
    fn slice_mha(&self, batch_idx: usize, head_idx: usize) -> Result<Tensor<T>, TransformerError>;
    fn slice_one_head_all_batches(&self, head_idx: usize) -> Result<Tensor<T>, TransformerError>;
    // stack_heads_from_list might need to be type specific if types inside list are fixed
    // fn stack_heads_from_list(head_tensors: &[Tensor<T>]) -> Result<Tensor<T>, TransformerError>;
    fn slice_mha_for_kv(&self, batch_idx: usize, head_idx: usize, kv_seq_len: usize) -> Result<Tensor<T>, TransformerError>;
    fn slice_mha_custom(&self, batch_idx: usize, head_idx: usize, q_seq_len: usize, kv_seq_len: usize) -> Result<Tensor<T>, TransformerError>;
}

impl<U: Clone + Default + Send + Sync + Copy + std::fmt::Debug + PartialEq + 'static> TensorExtMHA<U> for Tensor<U> {
    fn permute_mha_qkv(&self) -> Result<Tensor<U>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension(format!("permute_mha_qkv expects 4D tensor [B,S,H,D], got rank {}", self.rank())))); }
        let b = self.shape[0]; let s = self.shape[1]; let h = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![U::default(); self.data.len()];
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

    fn permute_mha_kt(&self) -> Result<Tensor<U>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_kt expects 4D tensor".into()))); }
        let b = self.shape[0]; let h = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![U::default(); self.data.len()];
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
    
    fn permute_mha_output(&self) -> Result<Tensor<U>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_output expects 4D tensor".into()))); }
        let b = self.shape[0]; let h = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![U::default(); self.data.len()];
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

    fn slice_mha(&self, batch_idx: usize, head_idx: usize) -> Result<Tensor<U>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_mha expects 4D tensor [B,H,S,D]".into()))); }
        let s_dim = self.shape[2]; 
        let d_dim = self.shape[3]; 
        
        let mut slice_data = Vec::with_capacity(s_dim * d_dim);
        let h_total_in_tensor = self.shape[1]; 

        let offset = batch_idx * h_total_in_tensor * s_dim * d_dim + head_idx * s_dim * d_dim;

        if self.data.is_empty() && s_dim * d_dim > 0 {
            return Err(TransformerError::TensorError(TensorError::ShapeMismatch("slice_mha: Input data is empty but slice size is non-zero".into())));
        }
        if self.data.is_empty() && s_dim * d_dim == 0 {
            return Tensor::new(Vec::new(), vec![s_dim, d_dim]).map_err(TransformerError::from);
        }

        if offset + (s_dim * d_dim) > self.data.len() {
            return Err(TransformerError::TensorError(TensorError::OutOfBounds(
                format!("slice_mha: Offset {} + slice_size {} > data_len {}. Shape: {:?}, B_idx: {}, H_idx: {}", 
                offset, s_dim*d_dim, self.data.len(), self.shape, batch_idx, head_idx)
            )));
        }
        slice_data.extend_from_slice(&self.data[offset .. offset + (s_dim * d_dim)]);
        Tensor::new(slice_data, vec![s_dim, d_dim]).map_err(TransformerError::from)
    }

    fn slice_one_head_all_batches(&self, head_idx: usize) -> Result<Tensor<U>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_one_head_all_batches expects 4D tensor [B,H,S,D]".into()))); }
        let b = self.shape[0]; let h_total = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        if head_idx >= h_total { return Err(TransformerError::TensorError(TensorError::OutOfBounds("head_idx out of bounds".into())));}

        let mut new_data = Vec::with_capacity(b * s * d);
        let new_shape = vec![b, s, d];

        for b_i in 0..b {
            for s_i in 0..s {
                for d_i in 0..d {
                    let old_idx = b_i*h_total*s*d + head_idx*s*d + s_i*d + d_i;
                    if old_idx < self.data.len() { // Added bounds check
                        new_data.push(self.data[old_idx]);
                    } else {
                        return Err(TransformerError::TensorError(TensorError::OutOfBounds("Index out of bounds during slice_one_head_all_batches".into())));
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }

    fn slice_mha_for_kv(&self, batch_idx: usize, head_idx: usize, kv_seq_len: usize) -> Result<Tensor<U>, TransformerError> {
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
    
    fn slice_mha_custom(&self, batch_idx: usize, head_idx: usize, q_seq_len: usize, kv_seq_len: usize) -> Result<Tensor<U>, TransformerError> {
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

// Specific stack_heads_from_list for f16, as TensorExtMHA is now generic and can't assume f32 for default()
impl Tensor<f16> {
    fn stack_heads_from_list_f16(head_tensors: &[Tensor<f16>]) -> Result<Tensor<f16>, TransformerError> {
        if head_tensors.is_empty() {
             // If n_head is 0, this might be valid. However, n_head should be > 0 typically.
             // Let's assume for now that if the list is empty, it's an error or needs context.
             // For an empty sequence (S=0), this might be okay.
             // If the first head tensor has S=0, then B*H*0*D = 0 elements.
             if head_tensors.get(0).map_or(true, |t| t.shape.get(1).map_or(true, |&s_dim| s_dim == 0) || t.shape.get(0).map_or(true, |&b_dim| b_dim ==0) )) {
                // This case implies S=0 or B=0. We can return an empty tensor with appropriate shape.
                // However, determining B, H, S, D is tricky if list is empty.
                // If list is not empty but S=0, then first_head.shape[1] is 0.
                // For now, let's require non-empty list if we expect heads.
                return Err(TransformerError::TensorError(TensorError::InvalidDimension("Cannot stack empty list of head tensors if n_head > 0".into())));
            }
        }
        let first_head = &head_tensors[0];
        if first_head.rank() != 3 {return Err(TransformerError::TensorError(TensorError::InvalidDimension("Head tensors for stacking must be 3D [B,S,D]".into())));}

        let b = first_head.shape[0]; let s = first_head.shape[1]; let d = first_head.shape[2];
        let h = head_tensors.len();
        let new_shape = vec![b,h,s,d];
        let total_elements = b*h*s*d;
        let mut new_data = if total_elements > 0 { vec![f16::from_f32(0.0); total_elements] } else { Vec::new() }; // Use f16 default

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
}


impl Tensor<f16> { // Add transpose_data_generic_f16 here
    pub fn transpose_data_generic_f16(data: &[f16], rows: usize, cols: usize) -> Vec<f16> {
        if rows * cols != data.len() && !data.is_empty() { 
            eprintln!("Warning: Transpose data length mismatch for f16. Data len: {}, rows*cols: {}", data.len(), rows*cols);
            return data.to_vec(); 
        }
        if data.is_empty() { return Vec::new(); }

        let mut new_data = vec![f16::from_f32(0.0); data.len()];
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
            dtype: ModelDataType::F32, // Added dtype
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

[end of rust-native-transformer/src/transformer_core.rs]
