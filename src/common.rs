use ndarray::{ArrayD, Array1, Axis};

#[derive(Debug)]
pub struct LayerNorm {
    pub(crate) _weight: ArrayD<f32>, // gamma
    pub(crate) _bias: ArrayD<f32>,   // beta
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(n_embd: i32, epsilon: f32) -> Result<Self, Box<dyn std::error::Error>> {
        let weight = Array1::ones(n_embd as usize).into_dyn();
        let bias = Array1::zeros(n_embd as usize).into_dyn();
        
        Ok(Self { 
            _weight: weight,
            _bias: bias,
            epsilon,
        })
    }

    pub fn forward(&self, x: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let last_dim_idx = x.ndim() - 1;
        let axis = Axis(last_dim_idx);

        let mean = x.mean_axis(axis).ok_or("Failed to compute mean")?;
        let variance = x.var_axis(axis, 0.0);

        let mean_kept_dims = mean.insert_axis(axis);
        let variance_kept_dims = variance.insert_axis(axis);

        let x_minus_mean = x - &mean_kept_dims;
        let std_dev_inv = (&variance_kept_dims + self.epsilon).mapv(f32::sqrt).mapv(|v| 1.0 / v);
        
        let normalized_x = x_minus_mean * std_dev_inv;
        let y = normalized_x * &self._weight + &self._bias;
        
        Ok(y)
    }
}

// KV Cache related type definitions
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub key: ArrayD<f32>,   // Shape: [Batch, NumHeads, SeqLen, HeadDim] or [Batch, SeqLen, HeadDim] post-merge/split
    pub value: ArrayD<f32>, // Shape: Similar to key
}

// Cache for a single layer, containing entries for all its attention heads
pub type LayerKVCache = Vec<KVCacheEntry>; // Each KVCacheEntry is for one head in the layer

// Full cache for the model, vector of caches for each layer
pub type ModelKVCache = Vec<LayerKVCache>;


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, s};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_layer_norm_forward() -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = 4;
        let epsilon = 1e-5;
        let layer_norm = LayerNorm::new(n_embd, epsilon)?;

        let x_data = Array::from_shape_vec((2, 3, 4), vec![
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0, 10.0, 11.0, 12.0,
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0
        ])?.into_dyn();

        let y = layer_norm.forward(&x_data)?;
        assert_eq!(y.shape(), x_data.shape(), "Output shape mismatch");

        let y_mean_last_dim = y.mean_axis(Axis(y.ndim() - 1)).expect("Mean calculation failed for y");
        for val in y_mean_last_dim.iter() {
            assert_abs_diff_eq!(*val, 0.0, epsilon = 1e-6);
        }

        let y_var_last_dim = y.var_axis(Axis(y.ndim() - 1), 0.0); 
        for val in y_var_last_dim.iter() {
            assert_abs_diff_eq!(*val, 1.0, epsilon = 1e-6);
        }
        
        let first_val_output = y[[0_usize, 0_usize, 0_usize]];
        let x_slice_00 = x_data.slice(s![0_usize, 0_usize, ..]); 
        let mean_00 = x_slice_00.mean().expect("Mean calculation for slice failed"); 
        let var_00 = x_slice_00.var(0.0); 
        let x_000 = x_slice_00[0_usize]; 
        let expected_first_val = (x_000 - mean_00) / (var_00 + epsilon).sqrt();
        assert_abs_diff_eq!(first_val_output, expected_first_val, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_layer_norm_new() -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = 10;
        let epsilon = 1e-6;
        let ln = LayerNorm::new(n_embd, epsilon)?;

        assert_eq!(ln._weight.shape(), &[n_embd as usize]);
        assert_eq!(ln._bias.shape(), &[n_embd as usize]);
        assert_eq!(ln.epsilon, epsilon);

        assert!(ln._weight.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON));
        assert!(ln._bias.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON));
        Ok(())
    }
}
