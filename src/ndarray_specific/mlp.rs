use crate::accelerator::{CpuTensor, Device, Module, Tensor};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct MLP {
    c_fc_weight: CpuTensor<f32>,
    c_fc_bias: CpuTensor<f32>,
    c_proj_weight: CpuTensor<f32>,
    c_proj_bias: CpuTensor<f32>,
    device: Device,
    // Activation function (e.g., GELU) is typically applied in forward pass
}

impl MLP {
    pub fn new(
        c_fc_weight_data: &[f32],
        c_fc_weight_shape: &[usize],
        c_fc_bias_data: &[f32],
        c_fc_bias_shape: &[usize],
        c_proj_weight_data: &[f32],
        c_proj_weight_shape: &[usize],
        c_proj_bias_data: &[f32],
        c_proj_bias_shape: &[usize],
    ) -> Result<Self, Box<dyn Error>> {
        let c_fc_weight =
            CpuTensor::from_data_and_shape(c_fc_weight_data, c_fc_weight_shape, Device::CPU)?;
        let c_fc_bias =
            CpuTensor::from_data_and_shape(c_fc_bias_data, c_fc_bias_shape, Device::CPU)?;
        let c_proj_weight =
            CpuTensor::from_data_and_shape(c_proj_weight_data, c_proj_weight_shape, Device::CPU)?;
        let c_proj_bias =
            CpuTensor::from_data_and_shape(c_proj_bias_data, c_proj_bias_shape, Device::CPU)?;

        Ok(Self {
            c_fc_weight,
            c_fc_bias,
            c_proj_weight,
            c_proj_bias,
            device: Device::CPU,
        })
    }
}

use ndarray::{Array, ArrayD, Axis, IxDyn, s, Zip};
use libm::tanhf;

// GELU approximation
fn gelu_new(x: f32) -> f32 {
    0.5 * x * (1.0 + tanhf((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))))
}

impl Module for MLP {
    type Input = CpuTensor<f32>;
    type Output = CpuTensor<f32>;

    fn forward(&self, input_tensor: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        if input_tensor.device() != self.device {
            return Err(format!("MLP is on {:?} but input is on {:?}", self.device, input_tensor.device()).into());
        }
        if self.device != Device::CPU {
            return Err(format!("MLP on {:?} is not supported for forward pass, only CPU.", self.device).into());
        }

        let input_slice = input_tensor.as_slice()?;
        let original_shape = input_tensor.shape();

        let (batch_size, seq_len, n_embd) = match original_shape.len() {
            2 => (1, original_shape[0], original_shape[1]), // [seq_len, n_embd]
            3 => (original_shape[0], original_shape[1], original_shape[2]), // [batch, seq_len, n_embd]
            _ => return Err(format!("Unsupported input rank: {}. Expected 2 or 3.", original_shape.len()).into()),
        };
        
        // Reshape to 3D [batch_size, seq_len, n_embd] for consistent processing
        let input_arr_3d = Array::from_shape_vec((batch_size, seq_len, n_embd), input_slice.to_vec())?;

        // --- First Linear Layer (Feed-Forward) ---
        let fc_w_slice = self.c_fc_weight.as_slice()?;
        let fc_b_slice = self.c_fc_bias.as_slice()?;
        // c_fc_weight shape: [n_embd, n_inner], c_fc_bias shape: [n_inner]
        let fc_w_arr = Array::from_shape_vec(self.c_fc_weight.shape().to_vec(), fc_w_slice.to_vec())?;
        let fc_b_arr = Array::from_shape_vec(self.c_fc_bias.shape().to_vec(), fc_b_slice.to_vec())?;

        // h_fc: [batch_size, seq_len, n_inner]
        let mut h_fc = input_arr_3d.dot(&fc_w_arr);
        h_fc = h_fc + &fc_b_arr; // Broadcasting bias

        // --- Activation Function (GELU) ---
        let h_gelu = h_fc.mapv(gelu_new);

        // --- Second Linear Layer (Projection) ---
        let proj_w_slice = self.c_proj_weight.as_slice()?;
        let proj_b_slice = self.c_proj_bias.as_slice()?;
        // c_proj_weight shape: [n_inner, n_embd], c_proj_bias shape: [n_embd]
        let proj_w_arr = Array::from_shape_vec(self.c_proj_weight.shape().to_vec(), proj_w_slice.to_vec())?;
        let proj_b_arr = Array::from_shape_vec(self.c_proj_bias.shape().to_vec(), proj_b_slice.to_vec())?;
        
        // h_proj: [batch_size, seq_len, n_embd]
        let mut h_proj = h_gelu.dot(&proj_w_arr);
        h_proj = h_proj + &proj_b_arr; // Broadcasting bias

        // Reshape back to original rank if it was 2D
        let final_output_data = h_proj.into_raw_vec();
        let output_shape_vec = if original_shape.len() == 2 {
            vec![seq_len, n_embd] // [seq_len, n_embd]
        } else {
            vec![batch_size, seq_len, n_embd] // [batch_size, seq_len, n_embd]
        };
        
        CpuTensor::from_data_and_shape(&final_output_data, &output_shape_vec, self.device.clone())
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        if device != Device::CPU {
            return Err(format!(
                "MLP with CpuTensor weights cannot be moved to {:?}. Only CPU is supported.",
                device
            )
            .into());
        }

        self.c_fc_weight = self.c_fc_weight.to_device(device.clone())?;
        self.c_fc_bias = self.c_fc_bias.to_device(device.clone())?;
        self.c_proj_weight = self.c_proj_weight.to_device(device.clone())?;
        self.c_proj_bias = self.c_proj_bias.to_device(device.clone())?;
        self.device = device;
        Ok(())
    }

    fn current_device(&self) -> Device {
        self.device.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{CpuTensor, Device, Tensor}; 

    // Helper to create MLP for testing
    fn test_mlp(n_embd: usize, n_inner: usize) -> MLP {
        let c_fc_weight_data = vec![0.0f32; n_embd * n_inner];
        let c_fc_weight_shape = vec![n_embd, n_inner];
        let c_fc_bias_data = vec![0.0f32; n_inner];
        let c_fc_bias_shape = vec![n_inner];

        let c_proj_weight_data = vec![0.0f32; n_inner * n_embd];
        let c_proj_weight_shape = vec![n_inner, n_embd];
        let c_proj_bias_data = vec![0.0f32; n_embd];
        let c_proj_bias_shape = vec![n_embd];

        MLP::new(
            &c_fc_weight_data, &c_fc_weight_shape,
            &c_fc_bias_data, &c_fc_bias_shape,
            &c_proj_weight_data, &c_proj_weight_shape,
            &c_proj_bias_data, &c_proj_bias_shape,
        ).expect("Failed to create MLP for test")
    }

    #[test]
    fn test_mlp_forward_2d_input() {
        let n_embd = 12;
        let n_inner = 48; // Typically 4 * n_embd
        let seq_len = 5;
        let mlp = test_mlp(n_embd, n_inner);

        let hidden_states_data: Vec<f32> = (0..(seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = mlp.forward(hidden_states);
        assert!(result.is_ok(), "MLP forward failed for 2D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[seq_len, n_embd], "Output shape mismatch for 2D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 2D input.");
        
        // With zero weights and biases, the output of matmuls and GELU(0) = 0.
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros for 2D input.");
    }

    #[test]
    fn test_mlp_forward_3d_input() {
        let n_embd = 12;
        let n_inner = 48;
        let batch_size = 2;
        let seq_len = 5;
        let mlp = test_mlp(n_embd, n_inner);

        let hidden_states_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![batch_size, seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = mlp.forward(hidden_states);
        assert!(result.is_ok(), "MLP forward failed for 3D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd], "Output shape mismatch for 3D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 3D input.");

        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros for 3D input.");
    }

    #[test]
    fn test_mlp_to_device_cpu() {
        let n_embd = 12;
        let n_inner = 48;
        let mut mlp = test_mlp(n_embd, n_inner);
        assert_eq!(mlp.current_device(), Device::CPU);
        assert!(mlp.to_device(Device::CPU).is_ok());
        assert_eq!(mlp.current_device(), Device::CPU);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{CpuTensor, Device, Tensor}; // Module is already in super

    // Helper to create MLP for testing
    fn test_mlp(n_embd: usize, n_inner: usize) -> MLP {
        let c_fc_weight_data = vec![0.0f32; n_embd * n_inner];
        let c_fc_weight_shape = vec![n_embd, n_inner];
        let c_fc_bias_data = vec![0.0f32; n_inner];
        let c_fc_bias_shape = vec![n_inner];

        let c_proj_weight_data = vec![0.0f32; n_inner * n_embd];
        let c_proj_weight_shape = vec![n_inner, n_embd];
        let c_proj_bias_data = vec![0.0f32; n_embd];
        let c_proj_bias_shape = vec![n_embd];

        MLP::new(
            &c_fc_weight_data, &c_fc_weight_shape,
            &c_fc_bias_data, &c_fc_bias_shape,
            &c_proj_weight_data, &c_proj_weight_shape,
            &c_proj_bias_data, &c_proj_bias_shape,
        ).unwrap()
    }

    #[test]
    fn test_mlp_forward_2d_input() {
        let n_embd = 12;
        let n_inner = 48; // Typically 4 * n_embd
        let seq_len = 5;
        let mlp = test_mlp(n_embd, n_inner);

        let hidden_states_data: Vec<f32> = (0..(seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = mlp.forward(hidden_states);
        assert!(result.is_ok(), "MLP forward failed: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[seq_len, n_embd], "Output shape mismatch for 2D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 2D input.");

        // Verify some output values. Since weights are 0, and bias is 0,
        // output of first linear is 0. GELU(0) is 0. Output of second linear is 0.
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros as expected with zero weights/biases.");
    }

    #[test]
    fn test_mlp_forward_3d_input() {
        let n_embd = 12;
        let n_inner = 48;
        let batch_size = 2;
        let seq_len = 5;
        let mlp = test_mlp(n_embd, n_inner);

        let hidden_states_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![batch_size, seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = mlp.forward(hidden_states);
        assert!(result.is_ok(), "MLP forward failed for 3D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd], "Output shape mismatch for 3D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 3D input.");

        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros for 3D input.");
    }

    #[test]
    fn test_mlp_to_device_cpu() {
        let n_embd = 12;
        let n_inner = 48;
        let mut mlp = test_mlp(n_embd, n_inner);
        assert_eq!(mlp.current_device(), Device::CPU);
        assert!(mlp.to_device(Device::CPU).is_ok());
        assert_eq!(mlp.current_device(), Device::CPU);
    }
}