// src/accelerator.rs

use ndarray::ArrayD; // Or your preferred tensor library
use std::error::Error;

// Represents the computational device
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA(usize), // usize is the device ID
    Metal(usize), // usize is the device ID
    // Future: TPU, etc.
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU // Default to CPU
    }
}

// Trait for abstracting tensor operations across devices
pub trait Tensor<T: Clone>: std::fmt::Debug {
    // Returns the device this tensor resides on
    fn device(&self) -> Device;

    // Returns the shape of the tensor
    fn shape(&self) -> &[usize];

    // Returns the data as a slice (primarily for CPU tensors or after moving to CPU)
    fn as_slice(&self) -> Result<&[T], Box<dyn Error>>;
    
    // Returns the data as a mutable slice (primarily for CPU tensors or after moving to CPU)
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>>;

    // Creates a new tensor on the specified device with given data and shape
    // This is a simplified constructor; more sophisticated ones might be needed
    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized;

    // Moves the tensor to the specified device
    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized;
    
    // Placeholder for a generic tensor operation (example)
    // fn matmul(&self, other: &Self) -> Result<Self, Box<dyn Error>> where Self: Sized;
}

// A basic CPU tensor implementation (wrapping ndarray for now)
#[derive(Debug, Clone)]
pub struct CpuTensor<T: Clone + Default + std::fmt::Debug> {
    array: ArrayD<T>,
    device: Device,
}

impl<T: Clone + Default + std::fmt::Debug + 'static> Tensor<T> for CpuTensor<T> {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn as_slice(&self) -> Result<&[T], Box<dyn Error>> {
        if self.device != Device::CPU {
            return Err("Cannot get slice from non-CPU tensor directly.".into());
        }
        self.array.as_slice().ok_or_else(|| "Failed to get slice from ndarray.".into())
    }
    
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>> {
        if self.device != Device::CPU {
            return Err("Cannot get mutable slice from non-CPU tensor directly.".into());
        }
        self.array.as_slice_mut().ok_or_else(|| "Failed to get mutable slice from ndarray.".into())
    }

    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> {
        if device != Device::CPU {
            return Err("CpuTensor can only be created on CPU device.".into());
        }
        let array = ArrayD::from_shape_vec(shape.to_vec(), data.to_vec())
            .map_err(|e| format!("Failed to create ArrayD: {}", e))?;
        Ok(CpuTensor { array, device })
    }

    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> {
        if device == self.device {
            return Ok(self.clone());
        }
        match device {
            Device::CPU => Err("CpuTensor is already on CPU. Cannot move to CPU again if it's a different conceptual CPU (not supported).".into()),
            Device::CUDA(_) | Device::Metal(_) => {
                Err(format!("CpuTensor to {:?} conversion not directly supported by CpuTensor::to_device. Use GpuTensor::from_data_and_shape or specific conversion methods.", device).into())
            }
        }
    }
}

// Represents a tensor on a GPU device (placeholder)
#[derive(PartialEq, Eq, Hash)]
pub struct GpuTensor<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> {
    shape: Vec<usize>,
    device: Device,
    // Actual GPU buffer/pointer omitted for scaffolding
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor")
         .field("shape", &self.shape)
         .field("device", &self.device)
         .finish()
    }
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> Clone for GpuTensor<T> {
    fn clone(&self) -> Self {
        GpuTensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> Tensor<T> for GpuTensor<T> {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn as_slice(&self) -> Result<&[T], Box<dyn Error>> {
        Err("Cannot directly access slice from GpuTensor.".into())
    }
    
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>> {
        Err("Cannot directly access mutable slice from GpuTensor.".into())
    }

    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> {
        match device {
            Device::CPU => Err("GpuTensor cannot be created on CPU device.".into()),
            Device::CUDA(_) | Device::Metal(_) => {
                // Data isn't actually stored on GPU yet. This is a placeholder.
                Ok(GpuTensor {
                    shape: shape.to_vec(),
                    device,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }

    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized {
        if device == self.device {
            return Ok(self.clone());
        }
        
        match device {
            Device::CPU => {
                Err("GpuTensor to CpuTensor conversion not directly supported by GpuTensor::to_device. Use specific conversion methods or a unified tensor type.".into())
            }
            Device::CUDA(_) | Device::Metal(_) => {
                Ok(GpuTensor {
                    shape: self.shape.clone(),
                    device,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }
}

// Trait for models or model components that can be moved between devices
// and can perform a forward pass.
pub trait Module: std::fmt::Debug {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>>;
    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>>;
    fn current_device(&self) -> Device;
}

// The Accelerator struct: main entry point for users
#[derive(Debug)]
pub struct Accelerator {
    device: Device,
}

impl Accelerator {
    pub fn new(device: Device) -> Self {
        Accelerator { device }
    }

    pub fn default() -> Self {
        Accelerator { device: Device::default() }
    }

    pub fn current_device(&self) -> Device {
        self.device.clone()
    }

    pub fn prepare<M: Module>(&self, module: &mut M) -> Result<(), Box<dyn Error>> {
        module.to_device(self.device.clone())?;
        Ok(())
    }

    pub fn prepare_data<T: Clone + Default + std::fmt::Debug + 'static, Tsr: Tensor<T> + Clone>(
        &self, 
        tensor: &Tsr
    ) -> Result<Tsr, Box<dyn Error>> {
        if tensor.device() == self.device {
            Ok(tensor.clone())
        } else {
            tensor.to_device(self.device.clone())
        }
    }
}

// Example of a simple module (conceptual)
#[derive(Debug)]
struct SimpleLinear {
    weights: CpuTensor<f32>,
    bias: CpuTensor<f32>,
    device: Device,
}

impl SimpleLinear {
    #[allow(dead_code)]
    fn new(in_features: usize, out_features: usize) -> Result<Self, Box<dyn Error>> {
        let w_data = vec![0.0f32; in_features * out_features];
        let b_data = vec![0.0f32; out_features];
        
        let weights = CpuTensor::from_data_and_shape(&w_data, &[in_features, out_features], Device::CPU)?;
        let bias = CpuTensor::from_data_and_shape(&b_data, &[out_features], Device::CPU)?;
        
        Ok(SimpleLinear { weights, bias, device: Device::CPU })
    }
}

impl Module for SimpleLinear {
    type Input = CpuTensor<f32>;
    type Output = CpuTensor<f32>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        if self.device != input.device() {
            return Err(format!("Module on {:?} received input on {:?}", self.device, input.device()).into());
        }
        let output_shape = input.shape();
        let output_data_len = output_shape.iter().product();
        let output_data = vec![0.0f32; output_data_len];
        CpuTensor::from_data_and_shape(&output_data, output_shape, self.device.clone())
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        if device != Device::CPU {
            return Err(format!("SimpleLinear with CpuTensor weights cannot be moved to {:?}.", device).into());
        }
        self.weights = self.weights.to_device(device.clone())?;
        self.bias = self.bias.to_device(device.clone())?;
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

    #[test]
    fn device_creation() {
        let cpu = Device::CPU;
        let cuda0 = Device::CUDA(0);
        assert_eq!(cpu, Device::CPU);
        assert_eq!(cuda0, Device::CUDA(0));
    }

    #[test]
    fn cpu_tensor_creation_and_properties() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let cpu_tensor = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();
        
        assert_eq!(cpu_tensor.device(), Device::CPU);
        assert_eq!(cpu_tensor.shape(), &[2, 2]);
        assert_eq!(cpu_tensor.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_tensor_to_same_device() {
        let data = vec![1.0f32, 2.0];
        let shape = vec![1, 2];
        let tensor1 = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();
        let tensor2 = tensor1.to_device(Device::CPU).unwrap();
        assert_eq!(tensor2.device(), Device::CPU);
        assert_eq!(tensor2.as_slice().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn cpu_tensor_to_other_device_should_fail_for_now() {
        let data = vec![1.0f32, 2.0];
        let shape = vec![1, 2];
        let tensor1 = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();
        assert!(tensor1.to_device(Device::CUDA(0)).is_err());
    }
    
    #[test]
    fn simple_linear_module_creation_and_device() {
        let mut linear = SimpleLinear::new(10, 5).unwrap();
        assert_eq!(linear.current_device(), Device::CPU);
        
        // Test moving to the same device (CPU)
        assert!(linear.to_device(Device::CPU).is_ok());
        assert_eq!(linear.current_device(), Device::CPU);

        // Test moving to another device (should fail for this simple example)
        assert!(linear.to_device(Device::CUDA(0)).is_err());
    }

    #[test]
    fn accelerator_prepare_module() {
        let accelerator = Accelerator::new(Device::CPU);
        let mut linear = SimpleLinear::new(10, 5).unwrap();
        
        assert!(accelerator.prepare(&mut linear).is_ok());
        assert_eq!(linear.current_device(), Device::CPU);
    }

    #[test]
    fn accelerator_prepare_data() {
        let accelerator = Accelerator::new(Device::CPU);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let cpu_tensor = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();

        let prepared_tensor = accelerator.prepare_data(&cpu_tensor).unwrap();
        assert_eq!(prepared_tensor.device(), Device::CPU);
        assert_eq!(prepared_tensor.as_slice().unwrap(), cpu_tensor.as_slice().unwrap());
    }
}
