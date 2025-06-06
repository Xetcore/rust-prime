use ndarray::{Array, ArrayD, Axis, Ix1, Ix2, Ix3, Ix4, Slice, ShapeError, s, concatenate};
use std::error::Error;
use crate::common::{KVCacheEntry, LayerKVCache};

// Helper function for softmax
fn softmax(input: &ArrayD<f32>, axis_index: usize) -> Result<ArrayD<f32>, Box<dyn Error>> {
    let axis = Axis(axis_index);
    let max_val = input.fold_axis(axis, f32::NEG_INFINITY, |&a, &b| a.max(b));
    let max_val_broadcastable = max_val.insert_axis(axis);
    
    let exp_values = (input - &max_val_broadcastable).mapv(f32::exp);
    
    let sum_exp_values = exp_values.sum_axis(axis);
    let sum_exp_values_broadcastable = sum_exp_values.insert_axis(axis);

    Ok(&exp_values / &sum_exp_values_broadcastable)
}

#[derive(Debug)]
pub struct MultiHeadAttention {
    pub(crate) n_head: i32,
    pub(crate) n_embd: i32,
    pub(crate) head_dim: i32,
    pub(crate) c_attn_w: ArrayD<f32>, // Shape: [n_embd, 3 * n_embd]
    pub(crate) c_attn_b: ArrayD<f32>, // Shape: [3 * n_embd]
    pub(crate) c_proj_w: ArrayD<f32>, // Shape: [n_embd, n_embd]
    pub(crate) c_proj_b: ArrayD<f32>, // Shape: [n_embd]
}

impl MultiHeadAttention {
    pub fn new(n_head: i32, n_embd: i32) -> Result<Self, Box<dyn Error>> {
        if n_embd % n_head != 0 {
            return Err(format!(
                "n_embd ({}) must be divisible by n_head ({})",
                n_embd, n_head
            )
            .into());
        }
        let head_dim = n_embd / n_head;

        let c_attn_w = Array::zeros((n_embd as usize, 3 * n_embd as usize)).into_dyn();
        let c_attn_b = Array::zeros((3 * n_embd as usize,)).into_dyn();
        let c_proj_w = Array::zeros((n_embd as usize, n_embd as usize)).into_dyn();
        let c_proj_b = Array::zeros((n_embd as usize,)).into_dyn();

        Ok(Self {
            n_head,
            n_embd,
            head_dim,
            c_attn_w,
            c_attn_b,
            c_proj_w,
            c_proj_b,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &ArrayD<f32>,
        attention_mask: Option<&ArrayD<f32>>,
        layer_kv_cache: Option<&mut LayerKVCache>,
    ) -> Result<ArrayD<f32>, Box<dyn Error>> {
        let initial_shape = hidden_states.shape();
        if initial_shape.len() != 3 {
            return Err(format!(
                "Expected hidden_states to be 3D (batch, seq_len, n_embd), got shape: {:?}",
                initial_shape
            )
            .into());
        }
        let batch_size = initial_shape[0];
        let seq_len_q = initial_shape[1]; // Sequence length of Query

        let hidden_states_reshaped_view = hidden_states
            .view()
            .into_shape((batch_size * seq_len_q, self.n_embd as usize))
            .map_err(|e: ShapeError| e.to_string())?;

        let c_attn_w_view = self.c_attn_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_attn_w as Ix2: {}", e))?;
        let qkv_projected = hidden_states_reshaped_view.dot(&c_attn_w_view);

        let c_attn_b_view = self.c_attn_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_attn_b as Ix1: {}", e))?;
        let qkv_biased = qkv_projected + &c_attn_b_view;

        let qkv = qkv_biased
            .into_shape((batch_size, seq_len_q, 3 * self.n_embd as usize))
            .map_err(|e: ShapeError| e.to_string())?
            .into_dyn();

        let q_current = qkv.slice_axis(Axis(2), Slice::from(0..(self.n_embd as usize))).to_owned();
        let k_current = qkv.slice_axis(Axis(2), Slice::from((self.n_embd as usize)..(2 * self.n_embd as usize))).to_owned();
        let v_current = qkv.slice_axis(Axis(2), Slice::from((2 * self.n_embd as usize)..(3 * self.n_embd as usize))).to_owned();

        let q_split = q_current
            .into_shape((batch_size, seq_len_q, self.n_head as usize, self.head_dim as usize))
            .map_err(|e: ShapeError| format!("Error reshaping Q: {}", e.to_string()))?
            .permuted_axes([0, 2, 1, 3]);

        let (k_split, v_split, seq_len_kv) = if let Some(cache) = layer_kv_cache {
            let mut k_head_list = Vec::with_capacity(self.n_head as usize);
            let mut v_head_list = Vec::with_capacity(self.n_head as usize);
            let mut current_max_kv_len = 0;

            for h in 0..(self.n_head as usize) {
                let k_current_h_unpermuted = k_current.view()
                    .into_shape((batch_size, seq_len_q, self.n_head as usize, self.head_dim as usize)).map_err(|e: ShapeError| e.to_string())?
                    .slice_axis(Axis(2), Slice::from(h..(h+1)))
                    .into_shape((batch_size, seq_len_q, self.head_dim as usize)).map_err(|e: ShapeError| e.to_string())?
                    .to_owned();

                 let v_current_h_unpermuted = v_current.view()
                    .into_shape((batch_size, seq_len_q, self.n_head as usize, self.head_dim as usize)).map_err(|e: ShapeError| e.to_string())?
                    .slice_axis(Axis(2), Slice::from(h..(h+1)))
                    .into_shape((batch_size, seq_len_q, self.head_dim as usize)).map_err(|e: ShapeError| e.to_string())?
                    .to_owned();

                let (k_h_combined, v_h_combined) = {
                    let cache_entry = &mut cache[h];
                    let k_past_h = &cache_entry.key;
                    let v_past_h = &cache_entry.value;

                    let k_combined_h_specific: ArrayD<f32>;
                    let v_combined_h_specific: ArrayD<f32>;

                    if k_past_h.shape()[1] == 0 {
                        k_combined_h_specific = k_current_h_unpermuted.into_dyn();
                        v_combined_h_specific = v_current_h_unpermuted.into_dyn();
                    } else {
                        let k_past_h_view = k_past_h.view().into_dimensionality::<Ix3>()
                            .map_err(|e| format!("Error viewing k_past_h as Ix3: {}", e))?;
                        let v_past_h_view = v_past_h.view().into_dimensionality::<Ix3>()
                            .map_err(|e| format!("Error viewing v_past_h as Ix3: {}", e))?;
                        
                        let k_current_h_view = k_current_h_unpermuted.view().into_dimensionality::<Ix3>()
                             .map_err(|e| format!("Error viewing k_current_h as Ix3: {}", e))?;
                        let v_current_h_view = v_current_h_unpermuted.view().into_dimensionality::<Ix3>()
                             .map_err(|e| format!("Error viewing v_current_h as Ix3: {}", e))?;

                        k_combined_h_specific = concatenate(Axis(1), &[k_past_h_view, k_current_h_view])
                            .map_err(|e| format!("Error concatenating K for head {}: {}", h, e))?.into_dyn();
                        v_combined_h_specific = concatenate(Axis(1), &[v_past_h_view, v_current_h_view])
                            .map_err(|e| format!("Error concatenating V for head {}: {}", h, e))?.into_dyn();
                    }
                    
                    cache_entry.key = k_combined_h_specific.clone();
                    cache_entry.value = v_combined_h_specific.clone();
                    (k_combined_h_specific, v_combined_h_specific)
                };
                
                current_max_kv_len = k_h_combined.shape()[1];
                k_head_list.push(k_h_combined.into_dimensionality::<Ix3>().unwrap());
                v_head_list.push(v_h_combined.into_dimensionality::<Ix3>().unwrap());
            }

            let k_stacked_views: Vec<_> = k_head_list.iter().map(|a| a.view().insert_axis(Axis(1))).collect();
            let k_combined_all_heads = concatenate(Axis(1), &k_stacked_views)
                .map_err(|e| format!("Error stacking K heads: {}", e))?;

            let v_stacked_views: Vec<_> = v_head_list.iter().map(|a| a.view().insert_axis(Axis(1))).collect();
            let v_combined_all_heads = concatenate(Axis(1), &v_stacked_views)
                .map_err(|e| format!("Error stacking V heads: {}", e))?;

            (k_combined_all_heads.into_dyn(), v_combined_all_heads.into_dyn(), current_max_kv_len)

        } else {
            let k_split_no_cache = k_current
                .into_shape((batch_size, seq_len_q, self.n_head as usize, self.head_dim as usize))
                .map_err(|e: ShapeError| format!("Error reshaping K (no cache): {}", e.to_string()))?
                .permuted_axes([0, 2, 1, 3]);
            let v_split_no_cache = v_current
                .into_shape((batch_size, seq_len_q, self.n_head as usize, self.head_dim as usize))
                .map_err(|e: ShapeError| format!("Error reshaping V (no cache): {}", e.to_string()))?
                .permuted_axes([0, 2, 1, 3]);
            (k_split_no_cache.into_dyn(), v_split_no_cache.into_dyn(), seq_len_q)
        };

        let k_split_transposed = k_split.permuted_axes(ndarray::IxDyn(&[0, 1, 3, 2]));

        let mut attn_scores_arr = Array::zeros((batch_size, self.n_head as usize, seq_len_q, seq_len_kv));
        let q_split_arr4 = q_split.view().into_dimensionality::<Ix4>()
            .map_err(|e| format!("Failed to view q_split as Ix4: {}", e))?;
        let k_split_transposed_arr4 = k_split_transposed.view().into_dimensionality::<Ix4>()
            .map_err(|e| format!("Failed to view k_split_transposed as Ix4: {}", e))?;

        for b in 0..batch_size {
            for h_idx in 0..(self.n_head as usize) {
                let q_slice = q_split_arr4.slice(s![b, h_idx, .., ..]);
                let k_t_slice = k_split_transposed_arr4.slice(s![b, h_idx, .., ..]);
                let score_slice = q_slice.dot(&k_t_slice);
                attn_scores_arr.slice_mut(s![b, h_idx, .., ..]).assign(&score_slice);
            }
        }
        
        let mut attn_scores = attn_scores_arr.into_dyn();
        attn_scores = attn_scores / (self.head_dim as f32).sqrt();

        if let Some(mask) = attention_mask {
            attn_scores = attn_scores + mask;
        } else {
            if seq_len_q > 1 || (seq_len_q == 1 && seq_len_kv > 1 && layer_kv_cache.is_none()) {
                 let mut causal_mask_slice = Array::from_elem((seq_len_q, seq_len_kv), 0.0f32);
                 for i in 0..seq_len_q {
                    for j in 0..seq_len_kv {
                        if j > i + (seq_len_kv - seq_len_q) {
                             causal_mask_slice[[i,j]] = f32::NEG_INFINITY;
                        }
                    }
                 }
                 let causal_mask = causal_mask_slice
                    .into_dyn()
                    .insert_axis(Axis(0)) 
                    .insert_axis(Axis(0)); 
                attn_scores = attn_scores + &causal_mask;

            } else if seq_len_q == 1 && seq_len_kv > 1 && layer_kv_cache.is_some() {
            }
        }

        let attn_probs = softmax(&attn_scores, 3)?; 

        let v_split_arr4 = v_split.view().into_dimensionality::<Ix4>()
            .map_err(|e| format!("Failed to view v_split as Ix4: {}", e))?;
        let attn_probs_arr4 = attn_probs.view().into_dimensionality::<Ix4>()
             .map_err(|e| format!("Failed to view attn_probs as Ix4: {}", e))?;
        
        let mut context_layer_arr = Array::zeros((batch_size, self.n_head as usize, seq_len_q, self.head_dim as usize));
        for b in 0..batch_size {
            for h_idx in 0..(self.n_head as usize) {
                let prob_slice = attn_probs_arr4.slice(s![b, h_idx, .., ..]);
                let v_slice = v_split_arr4.slice(s![b, h_idx, .., ..]);
                let ctx_slice = prob_slice.dot(&v_slice);
                context_layer_arr.slice_mut(s![b, h_idx, .., ..]).assign(&ctx_slice);
            }
        }

        let context_transposed = context_layer_arr.permuted_axes([0, 2, 1, 3]);
        let context_reshaped = context_transposed
            .as_standard_layout() 
            .into_shape((batch_size, seq_len_q, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping context layer: {}", e.to_string()))?;

        let context_reshaped_2d = context_reshaped
            .into_shape((batch_size * seq_len_q, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping context for final projection: {}", e.to_string()))?;

        let c_proj_w_view = self.c_proj_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_proj_w as Ix2: {}", e))?;
        let c_proj_b_view = self.c_proj_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_proj_b as Ix1: {}", e))?;

        let output_2d = context_reshaped_2d.dot(&c_proj_w_view) + &c_proj_b_view;
        
        let output = output_2d
            .into_shape((batch_size, seq_len_q, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping output: {}", e.to_string()))?
            .into_dyn();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use ndarray::Array; // Already imported at top level

    #[test]
    fn test_mha_new_valid_params() {
        let n_head = 12;
        let n_embd = 768;
        let mha_result = MultiHeadAttention::new(n_head, n_embd);
        assert!(mha_result.is_ok());
        let mha = mha_result.unwrap();

        assert_eq!(mha.n_head, n_head);
        assert_eq!(mha.n_embd, n_embd);
        assert_eq!(mha.head_dim, n_embd / n_head);
        assert_eq!(mha.c_attn_w.shape(), &[n_embd as usize, 3 * n_embd as usize]);
        assert_eq!(mha.c_attn_b.shape(), &[3 * n_embd as usize]);
        assert_eq!(mha.c_proj_w.shape(), &[n_embd as usize, n_embd as usize]);
        assert_eq!(mha.c_proj_b.shape(), &[n_embd as usize]);
    }

    #[test]
    fn test_mha_new_invalid_params() {
        assert!(MultiHeadAttention::new(12, 769).is_err(), "n_embd not divisible by n_head should be an error");
        assert!(MultiHeadAttention::new(0, 768).is_err(), "n_head cannot be zero");
        assert!(MultiHeadAttention::new(12, 0).is_err(), "n_embd cannot be zero if n_head is non-zero and implies head_dim > 0");
    }

    #[test]
    fn test_mha_forward_shape() {
        let n_head = 2;
        let n_embd = 4;
        let head_dim = n_embd / n_head; // Should be 2
        let mha = MultiHeadAttention::new(n_head, n_embd).unwrap();

        let batch_size = 1;
        let seq_len = 3;
        let input_hidden_states = ArrayD::zeros(ndarray::IxDyn(&[batch_size, seq_len, n_embd as usize]));
        
        let forward_result_no_mask_no_cache = mha.forward(&input_hidden_states, None, None);
        assert!(forward_result_no_mask_no_cache.is_ok(), "MHA forward (no mask, no cache) failed: {:?}", forward_result_no_mask_no_cache.err());
        let output_no_mask_no_cache = forward_result_no_mask_no_cache.unwrap();
        assert_eq!(output_no_mask_no_cache.shape(), &[batch_size, seq_len, n_embd as usize], "Output shape mismatch (no mask, no cache)");

        let attention_mask_shape = ndarray::IxDyn(&[batch_size, 1, seq_len, seq_len]);
        let attention_mask = ArrayD::zeros(attention_mask_shape);
        let forward_result_with_mask_no_cache = mha.forward(&input_hidden_states, Some(&attention_mask), None);
        assert!(forward_result_with_mask_no_cache.is_ok(), "MHA forward (with mask, no cache) failed: {:?}", forward_result_with_mask_no_cache.err());
        let output_with_mask_no_cache = forward_result_with_mask_no_cache.unwrap();
        assert_eq!(output_with_mask_no_cache.shape(), &[batch_size, seq_len, n_embd as usize], "Output shape mismatch (with mask, no cache)");
    
        let mut kv_cache: LayerKVCache = (0..n_head).map(|_| KVCacheEntry {
            key: Array::zeros((batch_size, seq_len -1, head_dim as usize)).into_dyn(),
            value: Array::zeros((batch_size, seq_len -1, head_dim as usize)).into_dyn(),
        }).collect();
        
        let current_hidden_states = ArrayD::zeros(ndarray::IxDyn(&[batch_size, 1, n_embd as usize]));
        let forward_result_with_cache = mha.forward(&current_hidden_states, None, Some(&mut kv_cache));
        assert!(forward_result_with_cache.is_ok(), "MHA forward (with cache) failed: {:?}", forward_result_with_cache.err());
        let output_with_cache = forward_result_with_cache.unwrap();
        assert_eq!(output_with_cache.shape(), &[batch_size, 1, n_embd as usize], "Output shape mismatch (with cache)");

        for head_cache in kv_cache.iter() {
            assert_eq!(head_cache.key.shape(), &[batch_size, seq_len, head_dim as usize]);
            assert_eq!(head_cache.value.shape(), &[batch_size, seq_len, head_dim as usize]);
        }
    }
}
