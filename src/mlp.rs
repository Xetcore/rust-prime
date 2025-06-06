use ndarray::{arr0, s, Array, ArrayD, Ix1, Ix2, IxDyn, ShapeError}; // Added arr0, s, IxDyn
use std::error::Error;
use std::f32::consts::PI;

// GELU Activation Function (approximation from GPT-2 "gelu_new")
fn gelu(x: &ArrayD<f32>) -> ArrayD<f32> {
    let x_cubed = x.mapv(|v| v.powi(3));
    let coeff = (2.0 / PI).sqrt();
    // Intermediate calculations are element-wise
    let inner_term_values = (x + &(0.044715 * &x_cubed)).mapv(|val| val * coeff);
    let tanh_values = inner_term_values.mapv(|val| val.tanh());
    0.5 * x * (1.0 + &tanh_values)
}

#[derive(Debug)]
pub struct MLP {
    pub(crate) n_embd: i32,
    pub(crate) n_inner: i32,
    pub(crate) c_fc_w: ArrayD<f32>,   // Shape: [n_embd, n_inner]
    pub(crate) c_fc_b: ArrayD<f32>,   // Shape: [n_inner]
    pub(crate) c_proj_w: ArrayD<f32>, // Shape: [n_inner, n_embd]
    pub(crate) c_proj_b: ArrayD<f32>, // Shape: [n_embd]
}

impl MLP {
    pub fn new(n_embd: i32, n_inner: i32) -> Result<Self, Box<dyn Error>> {
        if n_embd <= 0 || n_inner <= 0 {
            return Err("n_embd and n_inner must be positive".into());
        }
        let c_fc_w = Array::zeros((n_embd as usize, n_inner as usize)).into_dyn();
        let c_fc_b = Array::zeros((n_inner as usize,)).into_dyn();
        let c_proj_w = Array::zeros((n_inner as usize, n_embd as usize)).into_dyn();
        let c_proj_b = Array::zeros((n_embd as usize,)).into_dyn();

        Ok(Self {
            n_embd,
            n_inner,
            c_fc_w,
            c_fc_b,
            c_proj_w,
            c_proj_b,
        })
    }

    pub fn forward(&self, hidden_states: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn Error>> {
        let initial_shape = hidden_states.shape();
        if initial_shape.len() != 3 {
            return Err(format!(
                "Expected hidden_states to be 3D (batch, seq_len, n_embd), got shape: {:?}",
                initial_shape
            )
            .into());
        }
        let batch_size = initial_shape[0];
        let seq_len = initial_shape[1];
        let n_embd_input = initial_shape[2];

        if n_embd_input != self.n_embd as usize {
            return Err(format!(
                "Input embedding dimension ({}) does not match model n_embd ({})",
                n_embd_input, self.n_embd
            ).into());
        }

        // Reshape hidden_states for matrix multiplication: [B, S, E] -> [B*S, E]
        let reshaped_hs = hidden_states
            .view()
            .into_shape((batch_size * seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping hidden_states: {}", e.to_string()))?;

        // First Linear Transformation: (B*S, E) @ (E, I) + (I) -> (B*S, I)
        let c_fc_w_view = self.c_fc_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_fc_w as Ix2: {}", e))?;
        let c_fc_b_view = self.c_fc_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_fc_b as Ix1: {}", e))?;
        let h_fc = reshaped_hs.dot(&c_fc_w_view) + &c_fc_b_view; // h_fc is Array<f32, Ix2>

        // Apply GELU Activation
        let h_activated = gelu(&h_fc.into_dyn()); // gelu expects &ArrayD, returns ArrayD

        // Second Linear Transformation (Projection): (B*S, I) @ (I, E) + (E) -> (B*S, E)
        let h_activated_2d = h_activated.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view h_activated as Ix2: {}", e))?;
        let c_proj_w_view = self.c_proj_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_proj_w as Ix2: {}", e))?;
        let c_proj_b_view = self.c_proj_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_proj_b as Ix1: {}", e))?;
        let output_2d = h_activated_2d.dot(&c_proj_w_view) + &c_proj_b_view; // output_2d is Array<f32, Ix2>

        // Reshape Output back to [B, S, E]
        let output = output_2d
            .into_shape((batch_size, seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping output: {}", e.to_string()))?
            .into_dyn(); // Convert to ArrayD for the final output type

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1, ArrayD, IxDyn, s}; // Added arr1, s for slice
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mlp_new() -> Result<(), Box<dyn Error>> {
        let n_embd = 768;
        let n_inner = 4 * n_embd;
        let mlp = MLP::new(n_embd, n_inner)?;

        assert_eq!(mlp.n_embd, n_embd);
        assert_eq!(mlp.n_inner, n_inner);
        assert_eq!(mlp.c_fc_w.shape(), &[n_embd as usize, n_inner as usize]);
        assert_eq!(mlp.c_fc_b.shape(), &[n_inner as usize]);
        assert_eq!(mlp.c_proj_w.shape(), &[n_inner as usize, n_embd as usize]);
        assert_eq!(mlp.c_proj_b.shape(), &[n_embd as usize]);
        Ok(())
    }

    #[test]
    fn test_mlp_new_invalid_params() {
        assert!(MLP::new(0, 3072).is_err());
        assert!(MLP::new(768, 0).is_err());
        assert!(MLP::new(-1, 3072).is_err()); // Added for negative check
        assert!(MLP::new(768, -1).is_err()); // Added for negative check
    }

    #[test]
    fn test_gelu_fn() {
        // Test scalar 0.0
        let x0 = arr0(0.0f32).into_dyn();
        let y0 = gelu(&x0);
        assert_abs_diff_eq!(y0.iter().next().unwrap(), &0.0, epsilon = 1e-6); // Changed to iter().next().unwrap()

        // Test scalar 1.0
        let x1 = arr0(1.0f32).into_dyn();
        let y1 = gelu(&x1);
        assert_abs_diff_eq!(y1.iter().next().unwrap(), &0.84136, epsilon = 1e-5); // Changed to iter().next().unwrap()

        // Test with a small array
        let x_arr = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
        let y_arr = gelu(&x_arr);
        let y_expected_vec = vec![
            0.5 * -1.0 * (1.0 + ((2.0/PI).sqrt() * (-1.0 + 0.044715 * (-1.0f32).powi(3)) ).tanh()),
            0.0,
            0.84136,
        ];
        for (val, expected) in y_arr.iter().zip(y_expected_vec.iter()) {
            assert_abs_diff_eq!(val, expected, epsilon = 1e-5);
        }
    }
    
    #[test]
    fn test_mlp_forward() -> Result<(), Box<dyn Error>> {
        let n_embd = 4;
        let n_inner = 2 * n_embd; // 8
        let batch_size = 2;
        let seq_len = 3;

        let mut mlp = MLP::new(n_embd, n_inner)?;
        
        mlp.c_fc_w = Array::ones((n_embd as usize, n_inner as usize)).into_dyn();
        mlp.c_fc_b = Array::zeros((n_inner as usize,)).into_dyn();
        mlp.c_proj_w = Array::ones((n_inner as usize, n_embd as usize)).into_dyn();
        mlp.c_proj_b = Array::zeros((n_embd as usize,)).into_dyn();

        let hidden_states_vec: Vec<f32> = (0..(batch_size * seq_len * n_embd as usize))
            .map(|x| x as f32 * 0.1)
            .collect();
        let hidden_states = ArrayD::from_shape_vec(IxDyn(&[batch_size, seq_len, n_embd as usize]), hidden_states_vec)?;

        let output = mlp.forward(&hidden_states)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, n_embd as usize]);
        
        let first_input_token_vector = hidden_states.slice(s![0,0,..]);
        let sum_first_input_emb_vector: f32 = first_input_token_vector.iter().sum();
        
        let gelu_of_sum = gelu(&arr0(sum_first_input_emb_vector).into_dyn()).iter().next().unwrap().clone();
        let final_expected_val = gelu_of_sum * n_inner as f32;

        let first_output_token_vector = output.slice(s![0,0,..]);
        for val in first_output_token_vector.iter() {
            assert_abs_diff_eq!(*val, final_expected_val, epsilon = 1e-4);
        }
        Ok(())
    }

    #[test]
    fn test_mlp_forward_shape_check() { // Renamed from main's test_mlp_forward_shape
        let n_embd = 4;
        let n_inner = 4 * n_embd;
        let mlp = MLP::new(n_embd, n_inner).unwrap();

        let batch_size = 1;
        let seq_len = 3;
        let input_hidden_states = ArrayD::zeros(IxDyn(&[batch_size, seq_len, n_embd as usize]));
        
        let forward_result = mlp.forward(&input_hidden_states);
        assert!(forward_result.is_ok(), "MLP forward failed: {:?}", forward_result.err());
        let output = forward_result.unwrap();
        
        assert_eq!(output.shape(), &[batch_size, seq_len, n_embd as usize], "Output shape mismatch");
    }

    // Value-based test is not feasible until the forward pass is implemented with actual logic.
    // #[test]
    // #[ignore] // Ignored because forward is not fully implemented
    // fn test_mlp_forward_values_simple() {
    //     // This test would require setting specific weights and biases
    //     // and manually calculating the expected output for a very small input.
    //     // For now, it's a placeholder.
    //     todo!("Implement value-based test for MLP forward pass");
    // }
}