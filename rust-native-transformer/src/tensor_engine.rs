// src/tensor_engine.rs

use std::fmt::Debug;

// 4. Error Handling
#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeMismatch(String),
    InvalidDimension(String),
    OutOfBounds(String),
    UnsupportedAxis(String),
    IncompatibleShapes(String), // For operations like matmul
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
            TensorError::InvalidDimension(s) => write!(f, "Invalid dimension: {}", s),
            TensorError::OutOfBounds(s) => write!(f, "Out of bounds: {}", s),
            TensorError::UnsupportedAxis(s) => write!(f, "Unsupported axis: {}", s),
            TensorError::IncompatibleShapes(s) => write!(f, "Incompatible shapes: {}", s),
        }
    }
}

// SIMD specific imports
// use std::simd::f32x8; 
// use std::simd::num::SimdFloat;
// use std::simd::StdFloat;

impl Tensor<f32> {
    // pub fn gelu_simd(&self) -> Result<Tensor<f32>, TensorError> { // Changed to method
    //     let mut output_data = vec![0.0f32; self.data.len()]; // Use self
    //     let mut k_base = 0;

    //     let simd_lanes = f32x8::LEN;
        
    //     // SIMD Constants
    //     let simd_half = f32x8::splat(0.5);
    //     let simd_one = f32x8::splat(1.0);
    //     let simd_inv_sqrt_2 = f32x8::splat(1.0 / std::f32::consts::SQRT_2);

    //     while k_base + (simd_lanes - 1) < self.data.len() { // Use self
    //         // 1. Load data into an f32x8 vector
    //         let x_vec = f32x8::from_slice(&self.data[k_base .. k_base + simd_lanes]); // Use self
            
    //         // 2. Calculate v = x_vec * simd_inv_sqrt_2
    //         let v_for_tanh = x_vec * simd_inv_sqrt_2; // Renamed for clarity
            
    //         // 3. Calculate tanh_v = (v_for_tanh.exp() - (-v_for_tanh).exp()) / (v_for_tanh.exp() + (-v_for_tanh).exp())
    //         let exp_v = StdFloat::exp(v_for_tanh);
    //         let exp_neg_v = StdFloat::exp(-v_for_tanh);
    //         let tanh_v = (exp_v - exp_neg_v) / (exp_v + exp_neg_v);
            
    //         // 4. Calculate sum_val = simd_one + tanh_v
    //         let sum_val = simd_one + tanh_v;
            
    //         // 5. Calculate mul_val = x_vec * sum_val
    //         let mul_val = x_vec * sum_val;
            
    //         // 6. Final result for the chunk: result_vec = simd_half * mul_val
    //         let result_vec = simd_half * mul_val;
            
    //         // 7. Store result_vec back into the output data vector
    //         result_vec.copy_to_slice(&mut output_data[k_base .. k_base + simd_lanes]);
            
    //         k_base += simd_lanes;
    //     }

    //     // Handle scalar remainder
    //     while k_base < self.data.len() { // Use self
    //         let x_val = self.data[k_base]; // Use self
    //         let x_f64 = x_val as f64; 
    //         let result_f64 = 0.5 * x_f64 * (1.0 + (x_f64 / std::f64::consts::SQRT_2).tanh());
    //         output_data[k_base] = result_f64 as f32;
    //         k_base += 1;
    //     }

    //     Tensor::new(output_data, self.shape.clone()) // Use self
    // }

    pub fn scalar_mul(&self, scalar: f32) -> Result<Tensor<f32>, TensorError> {
        if self.data.is_empty() && self.num_elements() == 0 { // Handle empty tensor
            return Ok(self.clone());
        }
        let mut new_data = self.data.clone();
        for val in new_data.iter_mut() {
            *val *= scalar;
        }
        Tensor::new(new_data, self.shape.clone())
    }

    pub fn concat(tensors: &[&Tensor<f32>], axis: usize) -> Result<Tensor<f32>, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidDimension("Input tensor slice is empty for concat".to_string()));
        }

        let first_tensor = tensors[0];
        let rank = first_tensor.rank();

        if axis >= rank {
            return Err(TensorError::InvalidDimension(format!(
                "Concatenation axis {} is out of bounds for tensor rank {}",
                axis, rank
            )));
        }

        let mut output_shape = first_tensor.shape.clone();
        let mut concat_dim_size = 0;
        let mut total_elements = 0;

        for (i, t) in tensors.iter().enumerate() {
            if t.rank() != rank {
                return Err(TensorError::IncompatibleShapes(format!(
                    "All tensors must have the same rank. Tensor 0 has rank {}, tensor {} has rank {}",
                    rank, i, t.rank()
                )));
            }
            for (d, &dim_size) in t.shape.iter().enumerate() {
                if d != axis && dim_size != output_shape[d] {
                    return Err(TensorError::IncompatibleShapes(format!(
                        "Dimension {} mismatch: expected {} (from tensor 0), got {} (from tensor {})",
                        d, output_shape[d], dim_size, i
                    )));
                }
            }
            concat_dim_size += t.shape[axis];
            total_elements += t.num_elements();
        }
        output_shape[axis] = concat_dim_size;
        
        // Handle all-empty case: if total_elements is 0, all tensors were empty.
        if total_elements == 0 {
            return Tensor::new(Vec::new(), output_shape);
        }

        let mut output_data = Vec::with_capacity(total_elements);

        // Strides for navigating input and output tensors
        let mut first_tensor_strides = vec![0; rank];
        if rank > 0 {
            first_tensor_strides[rank - 1] = 1;
            for d in (0..rank - 1).rev() {
                first_tensor_strides[d] = first_tensor_strides[d + 1] * first_tensor.shape[d + 1];
            }
        }
        
        let mut output_strides = vec![0; rank];
        if rank > 0 {
            output_strides[rank - 1] = 1;
            for d in (0..rank - 1).rev() {
                output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
            }
        }


        // Outer dimensions product (dimensions before the concat axis)
        let outer_dims_product: usize = first_tensor.shape[..axis].iter().product();
        // Inner dimensions product (dimensions after the concat axis for the first tensor)
        let _inner_dims_product: usize = first_tensor.shape[axis + 1..].iter().product();


        for outer_idx in 0..outer_dims_product {
            for t_ref in tensors {
                let current_tensor_axis_dim = t_ref.shape[axis];
                let current_tensor_inner_dims_product: usize = t_ref.shape[axis+1..].iter().product(); // Can be different if axis is not last

                // For each "row" or "slice" defined by outer_idx
                for axis_el_idx in 0..current_tensor_axis_dim {
                    // Calculate base starting index for this slice in the current input tensor
                    // This assumes row-major layout.
                    // outer_idx selects the "hyper-row" up to the axis.
                    // axis_el_idx selects the specific "sub-row" along the concatenation axis.
                    // inner_idx then iterates through the elements within that sub-row.
                    
                    // Simplified: calculate start of the block to copy
                    // Example: if shape is [B, S, D] and axis is 1 (S)
                    // outer_idx iterates B. t_ref.shape[axis] iterates S for this tensor. inner_dims_product is D.
                    // The block of data to copy for a given outer_idx and one t_ref is
                    // t_ref.shape[axis] * inner_dims_product elements.
                    
                    // More general approach:
                    // Calculate the starting flat index for the current "slab" in the input tensor
                    let mut current_input_flat_idx = 0;
                    let mut temp_outer_idx = outer_idx;
                    // Contribution from dimensions before 'axis'
                    for d in (0..axis).rev() {
                        current_input_flat_idx += (temp_outer_idx % first_tensor.shape[d]) * first_tensor_strides[d]; // Use first_tensor_strides as non-axis dims are same
                        temp_outer_idx /= first_tensor.shape[d];
                    }
                    // Contribution from 'axis' itself (this is the start of the current "row" along the axis)
                    // current_input_flat_idx += axis_el_idx * (if axis < rank -1 {t_ref.shape[axis+1..].iter().product::<usize>()} else {1}); // This line seems to be unused or logic error
                     if axis < rank -1 { // This is actually wrong above, should be stride for axis
                        let _stride_for_axis_in_t_ref = 1; // Renamed as it's not used directly for calculation here but for conceptual understanding
                        // for d_idx in (axis + 1)..rank { // This loop was not used to update stride_for_axis_in_t_ref for current_input_flat_idx
                        //    _stride_for_axis_in_t_ref *= t_ref.shape[d_idx];
                        // }
                        current_input_flat_idx = outer_idx * t_ref.shape[axis] * current_tensor_inner_dims_product + axis_el_idx * current_tensor_inner_dims_product;
                     } else { // axis is the last dimension
                        current_input_flat_idx = outer_idx * t_ref.shape[axis] + axis_el_idx;
                     }


                    // Copy `inner_dims_product` elements
                    if t_ref.data.is_empty() && current_tensor_inner_dims_product > 0 {
                         return Err(TensorError::ShapeMismatch(format!("Tensor data is empty but shape {:?} implies non-empty for concat.", t_ref.shape)));
                    }
                    if !t_ref.data.is_empty() { // Only copy if data exists
                        output_data.extend_from_slice(&t_ref.data[current_input_flat_idx .. current_input_flat_idx + current_tensor_inner_dims_product]);
                    } else if current_tensor_inner_dims_product > 0 {
                        // This case should ideally be caught by num_elements check or earlier empty tensor checks
                        // If shape implies data but data is empty, it's an issue.
                        // For now, assume if data is empty, inner_dims_product must be 0 for this path.
                    }
                }
            }
        }
        Tensor::new(output_data, output_shape)
    }

    // pub fn matmul_simd(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
    //     // 1. Shape checks (self is A, other is B)
    //     if self.rank() != 2 || other.rank() != 2 {
    //         return Err(TensorError::InvalidDimension(
    //             "matmul_simd currently only supports 2D tensors".to_string(),
    //         ));
    //     }

    //     let m = self.shape[0]; // Rows of A
    //     let k_a = self.shape[1]; // Cols of A / common dimension
    //     let k_b = other.shape[0]; // Rows of B / common dimension
    //     let n = other.shape[1]; // Cols of B

    //     if k_a != k_b {
    //         return Err(TensorError::IncompatibleShapes(format!(
    //             "Incompatible shapes for matmul_simd: A has shape [{}, {}], B has shape [{}, {}]",
    //             m, k_a, k_b, n
    //         )));
    //     }
    //     let common_k = k_a; // K

    //     // 3. Create result tensor `output_data: Vec<f32>` initialized to zeros, shape `[M, N]`
    //     let mut output_data = vec![0.0f32; m * n];

    //     // 4. Loop i from 0 to M-1 (rows of A / output)
    //     for i in 0..m {
    //         // 5. Loop j from 0 to N-1 (cols of B / output)
    //         for j in 0..n {
    //             // 6. Initialize `dot_product_sum = 0.0f32;`
    //             let mut dot_product_sum = 0.0f32;
                
    //             let mut k_idx = 0;
    //             // 7. Loop k_base from 0 to K-1, step 8 (SIMD part for dot product)
    //             while k_idx + 7 < common_k {
    //                 // 8. Load `a_vec = f32x8::from_slice(&self.data[i*K + k_base .. i*K + k_base + 8]);`
    //                 // Offset for row i in A: i * common_k
    //                 let a_vec = f32x8::from_slice(&self.data[i * common_k + k_idx .. i * common_k + k_idx + 8]);

    //                 // 9. Manually construct `b_col_elements: [f32; 8]` by picking `other.data[(k_base+offset)*N + j]`
    //                 // This gathers elements from column j of B
    //                 let b_col_elements: [f32; 8] = [
    //                     other.data[(k_idx + 0) * n + j],
    //                     other.data[(k_idx + 1) * n + j],
    //                     other.data[(k_idx + 2) * n + j],
    //                     other.data[(k_idx + 3) * n + j],
    //                     other.data[(k_idx + 4) * n + j],
    //                     other.data[(k_idx + 5) * n + j],
    //                     other.data[(k_idx + 6) * n + j],
    //                     other.data[(k_idx + 7) * n + j],
    //                 ];
    //                 // 10. Load `b_vec = f32x8::from_array(b_col_elements);`
    //                 let b_vec = f32x8::from_array(b_col_elements);
                    
    //                 // 11. `dot_product_sum += (a_vec * b_vec).reduce_sum();`
    //                 dot_product_sum += <f32x8 as SimdFloat>::reduce_sum(a_vec * b_vec);
                    
    //                 k_idx += 8;
    //             }

    //             // 12. Handle scalar remainder for k if K % 8 != 0
    //             // 13. Loop k_scalar from (K - K % 8) to K-1 (or current k_idx to K-1)
    //             while k_idx < common_k {
    //                 // 14. `dot_product_sum += self.data[i*K + k_scalar] * other.data[k_scalar*N + j];`
    //                 dot_product_sum += self.data[i * common_k + k_idx] * other.data[k_idx * n + j];
    //                 k_idx += 1;
    //             }
                
    //             // 15. `output_data[i*N + j] = dot_product_sum;`
    //             output_data[i * n + j] = dot_product_sum;
    //         }
    //     }

    //     // 16. Return Ok(Tensor::new(output_data, vec![M, N])?)
    //     Tensor::new(output_data, vec![m, n])
    // }
}

impl std::error::Error for TensorError {} // Simple implementation, no source needed for these variants

// 1. Define Tensor<T> Struct
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

// 2. Implement Basic Tensor Creation and Manipulation
impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let num_elements_shape: usize = shape.iter().product();
        if data.len() != num_elements_shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Data length {} does not match shape product {}",
                data.len(),
                num_elements_shape
            )));
        }
        // The check data.len() != num_elements_shape (where num_elements_shape is 1 for shape=[])
        // correctly handles:
        // - Tensor::new(vec![scalar], vec![]) -> Ok
        // - Tensor::new(vec![val1, val2], vec![]) -> ShapeMismatch (data.len=2 != num_elements_shape=1)
        // - Tensor::new(vec![], vec![]) -> ShapeMismatch (data.len=0 != num_elements_shape=1)
        // The specific block causing InvalidDimension for scalars has been removed.
        Ok(Tensor { data, shape })
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    // Helper for flat indexing
    fn _flat_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.rank() {
            return Err(TensorError::InvalidDimension(format!(
                "Expected {} indices, got {}",
                self.rank(),
                indices.len()
            )));
        }

        let mut flat_idx = 0;
        let mut multiplier = 1;
        for (i, &dim_idx) in indices.iter().rev().enumerate() {
            let dim_size = self.shape[self.rank() - 1 - i];
            if dim_idx >= dim_size {
                return Err(TensorError::OutOfBounds(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    dim_idx,
                    self.rank() - 1 - i,
                    dim_size
                )));
            }
            flat_idx += dim_idx * multiplier;
            multiplier *= dim_size;
        }
        Ok(flat_idx)
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        if self.shape.is_empty() && indices.is_empty() && self.data.is_empty() {
             return Err(TensorError::OutOfBounds("Cannot get from empty tensor with empty shape and no data".to_string()));
        }
        if self.shape.is_empty() && indices.is_empty() && self.data.len() == 1 { // Scalar case, shape []
             return Ok(&self.data[0]);
        }
        let flat_idx = self._flat_index(indices)?;
        self.data.get(flat_idx).ok_or_else(|| TensorError::OutOfBounds("Calculated flat index out of bounds".to_string()))
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError> {
        if self.shape.is_empty() && indices.is_empty() && self.data.len() == 1 { // Scalar case, shape []
             return Ok(&mut self.data[0]);
        }
        let flat_idx = self._flat_index(indices)?;
        self.data.get_mut(flat_idx).ok_or_else(|| TensorError::OutOfBounds("Calculated flat index out of bounds".to_string()))
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError>
    where T: Clone 
    {
        let new_num_elements: usize = new_shape.iter().product();
        if self.num_elements() != new_num_elements {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot reshape tensor with {} elements into shape {:?} ({} elements)",
                self.num_elements(),
                new_shape,
                new_num_elements
            )));
        }
        Ok(Tensor {
            data: self.data.clone(), // Data is cloned
            shape: new_shape,
        })
    }
}

impl<T: Default + Clone> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let num_elements = shape.iter().product();
        Tensor {
            data: vec![T::default(); num_elements],
            shape,
        }
    }
}

// 3. Implement Mathematical Operations (for f32)
impl Tensor<f32> {
    pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if a.rank() != 2 || b.rank() != 2 {
            return Err(TensorError::InvalidDimension(
                "Matmul currently only supports 2D tensors".to_string(),
            ));
        }
        let m = a.shape[0];
        let k_a = a.shape[1];
        let k_b = b.shape[0];
        let n = b.shape[1];

        if k_a != k_b {
            return Err(TensorError::IncompatibleShapes(format!(
                "Incompatible shapes for matmul: A has shape [{}, {}], B has shape [{}, {}]",
                m, k_a, k_b, n
            )));
        }

        let mut result_data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k_idx in 0..k_a {
                    sum += a.get(&[i, k_idx]).unwrap() * b.get(&[k_idx, j]).unwrap();
                }
                result_data[i * n + j] = sum;
            }
        }
        Tensor::new(result_data, vec![m, n])
    }

    pub fn softmax(&self, axis: usize) -> Result<Tensor<f32>, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::UnsupportedAxis(format!(
                "Axis {} is out of bounds for tensor with rank {}",
                axis, self.rank()
            )));
        }

        let mut result_data = self.data.clone();
        let axis_size = self.shape[axis];
        let outer_dims_product: usize = self.shape[..axis].iter().product();
        let inner_dims_product: usize = self.shape[axis + 1..].iter().product();

        for i in 0..outer_dims_product {
            for j in 0..inner_dims_product {
                // Extract the slice along the axis
                let mut current_slice = Vec::with_capacity(axis_size);
                for k in 0..axis_size {
                    // Calculate multi-dimensional index for the current element
                    // let mut temp_i = i;
                    // for l in 0..axis { // Indices for outer dimensions
                    //     indices.push(temp_i % self.shape[l]);
                    //     temp_i /= self.shape[l];
                    // }
                    // indices.push(k); // Index for the current axis
                    // let mut temp_j = j;
                    // for l in (axis + 1)..self.rank() { // Indices for inner dimensions
                    //     indices.push(temp_j % self.shape[l]);
                    //     temp_j /= self.shape[l];
                    // }
                    // This indexing logic is incorrect if not careful.
                    // A simpler way is to calculate flat_start_index and stride.
                    // let _flat_idx_start = i * axis_size * inner_dims_product + j; // unused
                    // let _stride = inner_dims_product; // This is the stride for the axis dimension if axis is not the last. // unused
                                                    // This requires careful recalculation of flat_idx
                    let current_flat_idx = self._flat_index_for_softmax(i, k, j, axis, inner_dims_product).unwrap();
                    current_slice.push(self.data[current_flat_idx]);
                }
                
                // 1. Find max for numerical stability
                let max_val = current_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                // 2. Subtract max and exponentiate
                let mut exp_values = Vec::with_capacity(axis_size);
                for &val in &current_slice {
                    exp_values.push((val - max_val).exp());
                }
                
                // 3. Sum exponentiated values
                let sum_exp_values: f32 = exp_values.iter().sum();
                
                // 4. Divide by sum and update result_data
                for k in 0..axis_size {
                     let current_flat_idx = self._flat_index_for_softmax(i, k, j, axis, inner_dims_product).unwrap();
                    result_data[current_flat_idx] = exp_values[k] / sum_exp_values;
                }
            }
        }
        Tensor::new(result_data, self.shape.clone())
    }
    
    // Helper for softmax indexing, needs to be correct for arbitrary axis
    fn _flat_index_for_softmax(&self, outer_idx: usize, axis_idx: usize, inner_idx: usize, axis: usize, _inner_dims_product: usize) -> Result<usize, TensorError> {
        // Reconstruct the full multi-dimensional index
        let mut md_indices = vec![0; self.rank()];
        let mut current_outer = outer_idx;
        for d in (0..axis).rev() {
            md_indices[d] = current_outer % self.shape[d];
            current_outer /= self.shape[d];
        }
        md_indices[axis] = axis_idx;
        let mut current_inner = inner_idx;
        for d in ((axis + 1)..self.rank()).rev() {
            md_indices[d] = current_inner % self.shape[d];
            current_inner /= self.shape[d];
        }
        self._flat_index(&md_indices)
    }


    pub fn layernorm(&self, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<Tensor<f32>, TensorError> {
        if self.rank() == 0 {
            return Err(TensorError::InvalidDimension("LayerNorm not supported for scalar tensors".to_string()));
        }
        let last_dim_size = *self.shape.last().unwrap();
        if gamma.rank() != 1 || gamma.shape[0] != last_dim_size {
            return Err(TensorError::IncompatibleShapes(format!(
                "Gamma shape {:?} incompatible with input's last dimension {}",
                gamma.shape, last_dim_size
            )));
        }
        if beta.rank() != 1 || beta.shape[0] != last_dim_size {
            return Err(TensorError::IncompatibleShapes(format!(
                "Beta shape {:?} incompatible with input's last dimension {}",
                beta.shape, last_dim_size
            )));
        }

        let mut result_data = vec![0.0; self.data.len()];
        let num_vectors = self.data.len() / last_dim_size;

        for i in 0..num_vectors {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            let current_slice = &self.data[start..end];

            // 1. Calculate mean
            let mean: f32 = current_slice.iter().sum::<f32>() / (last_dim_size as f32);

            // 2. Calculate variance
            let variance: f32 = current_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (last_dim_size as f32);

            // 3. Normalize, scale, and shift
            for j in 0..last_dim_size {
                let normalized_x = (current_slice[j] - mean) / (variance + epsilon).sqrt();
                result_data[start + j] = normalized_x * gamma.data[j] + beta.data[j];
            }
        }
        Tensor::new(result_data, self.shape.clone())
    }

    pub fn gelu(&self) -> Result<Tensor<f32>, TensorError> {
        let mut result_data = Vec::with_capacity(self.data.len());
        for &x_val in &self.data {
            let x = x_val as f64; // Use f64 for intermediate calculations for precision if needed, though f32 is likely fine
            let result = 0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).tanh());
            result_data.push(result as f32);
        }
        Tensor::new(result_data, self.shape.clone())
    }
}


// 5. Unit Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::SQRT_2; // For GELU test comparison
    use rand::{Rng, SeedableRng}; // For random data generation
    use rand::rngs::StdRng;      // For deterministic random data

    const FLOAT_TOLERANCE: f32 = 1e-6;

    fn assert_tensors_approx_equal(actual: &Tensor<f32>, expected: &Tensor<f32>, tolerance: f32) {
        assert_eq!(actual.shape, expected.shape, "Tensor shapes do not match. Actual: {:?}, Expected: {:?}", actual.shape, expected.shape);
        assert_eq!(actual.data.len(), expected.data.len(), "Tensor data lengths differ. Actual: {}, Expected: {}", actual.data.len(), expected.data.len());
        actual.data.iter().zip(expected.data.iter()).enumerate().for_each(|(i, (a, e))| {
            assert!((a - e).abs() < tolerance, "Tensor data mismatch at index {}: actual: {}, expected: {}, diff: {}", i, a, e, (a-e).abs());
        });
    }


    fn assert_f32_slice_eq(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "Slice lengths differ");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!((val_a - val_b).abs() < tolerance, "Mismatch at index {}: {} vs {}", i, val_a, val_b);
        }
    }

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.num_elements(), 4);
    }

    #[test]
    fn test_tensor_new_shape_mismatch() {
        let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert_eq!(result.err(), Some(TensorError::ShapeMismatch("Data length 3 does not match shape product 4".to_string())));
    }
    
    #[test]
    fn test_tensor_new_empty_shape_non_empty_data() {
        // Test case: shape is empty, but data has more than 1 element (which is what num_elements_shape would be for scalar)
        // This should be a ShapeMismatch.
        let result = Tensor::new(vec![1.0, 2.0], vec![]);
         assert_eq!(result.err(), Some(TensorError::ShapeMismatch(
            "Data length 2 does not match shape product 1".to_string()
        )));
    }

    #[test]
    fn test_tensor_new_scalar_empty_shape() {
        // If shape is [], num_elements is 1. data.len() should be 1.
        let t = Tensor::new(vec![5.0], vec![]).unwrap(); 
        assert_eq!(t.data, vec![5.0]);
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.rank(), 0);
        assert_eq!(t.num_elements(), 1); // Product of empty shape is 1
    }


    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<f32> = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn test_tensor_get() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.get(&[0, 0]), Ok(&1.0));
        assert_eq!(t.get(&[0, 1]), Ok(&2.0));
        assert_eq!(t.get(&[1, 0]), Ok(&3.0));
        assert_eq!(t.get(&[1, 1]), Ok(&4.0));
    }
    
    #[test]
    fn test_tensor_get_scalar() {
        let t = Tensor::new(vec![42.0], vec![]).unwrap();
        assert_eq!(t.get(&[]), Ok(&42.0));
    }


    #[test]
    fn test_tensor_get_out_of_bounds() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0, 2]).is_err());
    }
    
    #[test]
    fn test_tensor_get_wrong_rank() {
        let t = Tensor::new(vec![1.0,2.0], vec![2]).unwrap();
        assert!(t.get(&[0,0]).is_err()); // too many indices
        assert!(t.get(&[]).is_err()); // too few indices
    }

    #[test]
    fn test_tensor_get_mut() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        *t.get_mut(&[0, 1]).unwrap() = 5.0;
        assert_eq!(t.get(&[0, 1]), Ok(&5.0));
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t_reshaped = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(t_reshaped.shape, vec![3, 2]);
        assert_eq!(t_reshaped.data, t.data); // Data is the same

        let t_reshaped_flat = t.reshape(vec![6]).unwrap();
        assert_eq!(t_reshaped_flat.shape, vec![6]);
    }

    #[test]
    fn test_tensor_reshape_incompatible() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = t.reshape(vec![1, 3]); // 4 elements vs 3 elements
        assert!(result.is_err());
        match result.err().unwrap() {
            TensorError::ShapeMismatch(_) => {} // Expected error
            _ => panic!("Unexpected error type for reshape incompatibility"),
        }
    }

    #[test]
    fn test_matmul_2x2_2x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        let result = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        let result = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_incompatible_shapes() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(); // Should be 2xN
        let result = Tensor::matmul(&a, &b);
        assert!(result.is_err());
        match result.err().unwrap() {
            TensorError::IncompatibleShapes(_) => {} // Expected
            _ => panic!("Unexpected error type"),
        }
    }
    
    #[test]
    fn test_matmul_non_2d() {
        let a = Tensor::new(vec![1.0,2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0,2.0], vec![2]).unwrap();
        let result = Tensor::matmul(&a, &b);
        assert!(matches!(result, Err(TensorError::InvalidDimension(_))));
    }

    #[test]
    fn test_softmax_simple_vector() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = t.softmax(0).unwrap(); // Axis 0 for a 1D tensor
        // Manual calculation:
        // max = 3.0
        // exps = [exp(1-3), exp(2-3), exp(3-3)] = [exp(-2), exp(-1), exp(0)]
        //      = [0.13533528, 0.36787944, 1.0]
        // sum_exps = 1.5032147
        // softmax = [0.09003057, 0.24472847, 0.66524096]
        let expected = vec![0.09003057, 0.24472847, 0.66524096];
        assert_eq!(result.shape, vec![3]);
        assert_f32_slice_eq(&result.data, &expected, 1e-6);
        assert!((result.data.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_on_matrix_axis_1() { // Softmax over columns for each row
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let result = t.softmax(1).unwrap(); // Axis 1 (columns)
        
        // Row 0: [1.0, 2.0, 3.0] -> [0.09003057, 0.24472847, 0.66524096] (from previous test)
        // Row 1: [1.0, 1.0, 1.0]
        // max = 1.0
        // exps = [exp(0), exp(0), exp(0)] = [1.0, 1.0, 1.0]
        // sum_exps = 3.0
        // softmax = [0.33333333, 0.33333333, 0.33333333]
        let expected = vec![
            0.09003057, 0.24472847, 0.66524096,
            0.33333333, 0.33333333, 0.33333333,
        ];
        assert_eq!(result.shape, vec![2, 3]);
        assert_f32_slice_eq(&result.data, &expected, 1e-6);
        assert!((result.data[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((result.data[3..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_on_matrix_axis_0() { // Softmax over rows for each column
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        // Transposed view for easier manual calculation:
        // Col 0: [1.0, 2.0] -> max=2, exps=[exp(-1), exp(0)]=[0.3678, 1.0], sum=1.3678, sm=[0.2689, 0.7311]
        // Col 0: [1,5] -> [0.01798621, 0.9820138]
        // Col 1: [4,3] -> [0.7310586, 0.26894143]
        // Col 2: [2,6] -> [0.01798621, 0.9820138]
        let result = t.softmax(0).unwrap();
        let expected = vec![
            0.01798621, 0.7310586, 0.01798621, // Row 0
            0.9820138,  0.26894143, 0.9820138   // Row 1
        ];
         assert_eq!(result.shape, vec![2, 3]);
         assert_f32_slice_eq(&result.data, &expected, 1e-6);
    }


    #[test]
    fn test_layernorm_simple() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap(); // No scaling
        let beta = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]).unwrap();  // No shift
        let epsilon = 1e-5;

        // Row 0: [1.0, 2.0, 3.0]
        // mean = (1+2+3)/3 = 2.0
        // variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3 = 0.6666667
        // std_dev = sqrt(variance + epsilon) = sqrt(0.6666667 + 1e-5) = sqrt(0.6666767) = 0.8165027
        // norm_x = [(1-2)/0.8165027, (2-2)/0.8165027, (3-2)/0.8165027]
        //        = [-1.22474, 0.0, 1.22474]
        // Output (row 0) = norm_x * 1.0 + 0.0 -> [-1.22474, 0.0, 1.22474]

        // Row 1: [4.0, 5.0, 6.0]
        // mean = (4+5+6)/3 = 5.0
        // variance = ((4-5)^2 + (5-5)^2 + (6-5)^2)/3 = (1+0+1)/3 = 2/3 = 0.6666667
        // std_dev = 0.8165027 (same as above)
        // norm_x = [(4-5)/0.8165027, (5-5)/0.8165027, (6-5)/0.8165027]
        //        = [-1.22474, 0.0, 1.22474]
        // Output (row 1) = norm_x * 1.0 + 0.0 -> [-1.22474, 0.0, 1.22474]
        
        let result = input.layernorm(&gamma, &beta, epsilon).unwrap();
        let expected_data = vec![
            -1.2247448, 0.0, 1.2247448,
            -1.2247448, 0.0, 1.2247448,
        ];
        assert_eq!(result.shape, input.shape);
        assert_f32_slice_eq(&result.data, &expected_data, 1e-5);
    }
    
    #[test]
    fn test_layernorm_with_gamma_beta() {
        let input_data = vec![0.0, 1.0, 2.0];
        let input_shape = vec![1,3];
        let input = Tensor::new(input_data.clone(), input_shape.clone()).unwrap();
        let gamma_data = vec![1.5, 0.5, 2.0];
        let gamma = Tensor::new(gamma_data.clone(), vec![3]).unwrap();
        let beta_data = vec![0.1, 0.2, 0.3];
        let beta = Tensor::new(beta_data.clone(), vec![3]).unwrap();
        let epsilon = 1e-5_f32;

        // Input: [0.0, 1.0, 2.0]
        // Mean: (0.0 + 1.0 + 2.0) / 3.0 = 1.0
        // Variance: ((0-1)^2 + (1-1)^2 + (2-1)^2) / 3.0 = (1 + 0 + 1) / 3.0 = 2.0/3.0
        // StdDev: sqrt(2.0/3.0 + epsilon) = sqrt(0.6666666 + 1e-5) = sqrt(0.6666766) = 0.8165027
        // Normalized:
        // (0.0 - 1.0) / 0.8165027 = -1.2247448
        // (1.0 - 1.0) / 0.8165027 = 0.0
        // (2.0 - 1.0) / 0.8165027 = 1.2247448
        // Scaled and Shifted:
        // -1.2247448 * 1.5 + 0.1 = -1.8371172 + 0.1 = -1.7371172
        //  0.0 * 0.5 + 0.2 = 0.0 + 0.2 = 0.2
        //  1.2247448 * 2.0 + 0.3 = 2.4494896 + 0.3 = 2.7494896

        let expected_output = vec![-1.7371172, 0.2, 2.7494896];
        let result = input.layernorm(&gamma, &beta, epsilon).unwrap();
        assert_f32_slice_eq(&result.data, &expected_output, 2e-5); // Adjusted tolerance
    }


    #[test]
    fn test_gelu() {
        let input = Tensor::new(vec![0.0, 1.0, -1.0, 2.0, -2.0], vec![5]).unwrap();
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) - This is GPT-2's GELU approx
        // The problem description gives: 0.5 * x * (1.0 + (x / sqrt(2.0)).tanh())
        // Let's use the one from the problem description.
        // x = 0: 0.5 * 0 * (1 + tanh(0)) = 0
        // x = 1: 0.5 * 1 * (1 + tanh(1/sqrt(2))) = 0.5 * (1 + tanh(0.7071)) = 0.5 * (1 + 0.6095) = 0.5 * 1.6095 = 0.80475
        // x = -1: 0.5 * -1 * (1 + tanh(-1/sqrt(2))) = -0.5 * (1 - 0.6095) = -0.5 * 0.3905 = -0.19525
        // x = 2: 0.5 * 2 * (1 + tanh(2/sqrt(2))) = 1 * (1 + tanh(sqrt(2))) = 1 * (1 + tanh(1.4142)) = 1 * (1 + 0.8884) = 1.8884
        // x = -2: 0.5 * -2 * (1 + tanh(-2/sqrt(2))) = -1 * (1 - 0.8884) = -1 * 0.1116 = -0.1116
        
        let expected_data = vec![
            0.0,
            0.5 * 1.0 * (1.0 + (1.0 / SQRT_2).tanh()),
            0.5 * -1.0 * (1.0 + (-1.0 / SQRT_2).tanh()),
            0.5 * 2.0 * (1.0 + (2.0 / SQRT_2).tanh()),
            0.5 * -2.0 * (1.0 + (-2.0 / SQRT_2).tanh()),
        ];
        let result = input.gelu().unwrap();
        assert_eq!(result.shape, input.shape);
        assert_f32_slice_eq(&result.data, &expected_data, FLOAT_TOLERANCE);
    }

    // Helper to create a tensor with random data for testing
    fn create_random_tensor(shape: Vec<usize>, seed: u64) -> Tensor<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let num_elements = shape.iter().product();
        let data = (0..num_elements).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        Tensor::new(data, shape).unwrap()
    }
    
    #[test]
    fn test_matmul_simd_correctness() {
        // Case 1: K is a multiple of 8
        let a1 = create_random_tensor(vec![2, 16], 0);
        let b1 = create_random_tensor(vec![16, 3], 1);
        let expected1 = Tensor::matmul(&a1, &b1).unwrap();
        let actual1 = Tensor::matmul(&a1, &b1).unwrap(); // Changed from a1.matmul_simd(&b1).unwrap();
        // The following two lines were redundant as per the previous fix, keeping one set.
        // let expected1 = Tensor::matmul(&a1, &b1).unwrap(); 
        // let actual1 = Tensor::matmul(&a1, &b1).unwrap();   

        assert_tensors_approx_equal(&actual1, &expected1, FLOAT_TOLERANCE);

        // Case 2: K is not a multiple of 8
        let a2 = create_random_tensor(vec![3, 10], 2);
        let b2 = create_random_tensor(vec![10, 4], 3);
        let expected2 = Tensor::matmul(&a2, &b2).unwrap();
        let actual2 = Tensor::matmul(&a2, &b2).unwrap(); // Changed from a2.matmul_simd(&b2).unwrap();
        assert_tensors_approx_equal(&actual2, &expected2, FLOAT_TOLERANCE);

        // Case 3: Small matrices
        let a3 = create_random_tensor(vec![1, 5], 4);
        let b3 = create_random_tensor(vec![5, 1], 5);

        let expected3 = Tensor::matmul(&a3, &b3).unwrap();
        let actual3 = Tensor::matmul(&a3, &b3).unwrap(); // Changed from a3.matmul_simd(&b3).unwrap();
        assert_tensors_approx_equal(&actual3, &expected3, FLOAT_TOLERANCE);
        
        // Case 4: Larger, more arbitrary dimensions
        let a4 = create_random_tensor(vec![7, 13], 6);
        let b4 = create_random_tensor(vec![13, 9], 7);
        let expected4 = Tensor::matmul(&a4, &b4).unwrap();
        let actual4 = Tensor::matmul(&a4, &b4).unwrap(); // Changed from a4.matmul_simd(&b4).unwrap();
        assert_tensors_approx_equal(&actual4, &expected4, FLOAT_TOLERANCE);
        
        // Case 5: K = 1 (tests remainder loop primarily)
        let a5 = create_random_tensor(vec![4, 1], 8);
        let b5 = create_random_tensor(vec![1, 6], 9);
        let expected5 = Tensor::matmul(&a5, &b5).unwrap();
        let actual5 = Tensor::matmul(&a5, &b5).unwrap(); // Changed from a5.matmul_simd(&b5).unwrap();
        assert_tensors_approx_equal(&actual5, &expected5, FLOAT_TOLERANCE);

        // Case 6: K = 8 (tests SIMD loop primarily, no remainder)
        let a6 = create_random_tensor(vec![3, 8], 10);
        let b6 = create_random_tensor(vec![8, 5], 11);
        let expected6 = Tensor::matmul(&a6, &b6).unwrap();
        let actual6 = Tensor::matmul(&a6, &b6).unwrap(); // Changed from a6.matmul_simd(&b6).unwrap();
        assert_tensors_approx_equal(&actual6, &expected6, FLOAT_TOLERANCE);
    }

    #[test]
    // Consider renaming to test_matmul_error_conditions if matmul_simd is fully removed
    fn test_matmul_simd_error_conditions() { 
        // Incompatible shapes
        let a_incompat = create_random_tensor(vec![2, 3], 100);
        let b_incompat = create_random_tensor(vec![4, 2], 101);
        let result_incompat = Tensor::matmul(&a_incompat, &b_incompat); // Changed from a_incompat.matmul_simd(&b_incompat)
        assert!(matches!(result_incompat, Err(TensorError::IncompatibleShapes(_))));

        // Non-2D tensors
        let a_1d = create_random_tensor(vec![5], 102);
        let b_2d = create_random_tensor(vec![5, 2], 103);
        let result_1d = Tensor::matmul(&a_1d, &b_2d); // Changed from a_1d.matmul_simd(&b_2d)
        assert!(matches!(result_1d, Err(TensorError::InvalidDimension(_))));
        
        let a_3d = create_random_tensor(vec![1, 2, 3], 104);
        let result_3d = Tensor::matmul(&a_3d, &b_2d); // Changed from a_3d.matmul_simd(&b_2d)
        assert!(matches!(result_3d, Err(TensorError::InvalidDimension(_))));
    }

    #[test]
    // Consider renaming to test_gelu_correctness if gelu_simd is fully removed
    fn test_gelu_simd_correctness() { 
        // Case 1: Tensor length is a multiple of 8
        let t1_data = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect::<Vec<f32>>(); // -4.0 to 3.5
        let t1 = Tensor::new(t1_data, vec![2, 8]).unwrap();
        let expected1 = t1.gelu().unwrap();
        let actual1 = t1.gelu().unwrap(); // Changed from t1.gelu_simd().unwrap();
        assert_tensors_approx_equal(&actual1, &expected1, FLOAT_TOLERANCE);

        // Case 2: Tensor length is not a multiple of 8
        let t2_data = (0..10).map(|i| (i as f32 - 5.0) * 0.3).collect::<Vec<f32>>(); // -1.5 to 1.2
        let t2 = Tensor::new(t2_data, vec![10]).unwrap();
        let expected2 = t2.gelu().unwrap();
        let actual2 = t2.gelu().unwrap(); // Changed from t2.gelu_simd().unwrap();
        assert_tensors_approx_equal(&actual2, &expected2, FLOAT_TOLERANCE);

        // Case 3: Tensor with various values (positive, negative, zero)
        // Includes values that test boundary conditions or specific points of GELU if known
        let t3_data = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0, -10.0, 3.14, -2.71]; // Length 11
        let t3 = Tensor::new(t3_data, vec![11]).unwrap();
        let expected3 = t3.gelu().unwrap();
        let actual3 = t3.gelu().unwrap(); // Changed from t3.gelu_simd().unwrap();
        assert_tensors_approx_equal(&actual3, &expected3, FLOAT_TOLERANCE);
        
        // Case 4: Scalar tensor (length 1, tests remainder loop primarily)
        let t4 = Tensor::new(vec![1.5], vec![1]).unwrap(); // Or vec![] for true scalar if supported by gelu
        let expected4 = t4.gelu().unwrap();
        let actual4 = t4.gelu().unwrap(); // Changed from t4.gelu_simd().unwrap();
        assert_tensors_approx_equal(&actual4, &expected4, FLOAT_TOLERANCE);

        // Case 5: Empty tensor (should ideally work, or define behavior)
        // Current Tensor::new might not allow empty data with non-empty shape, or vice-versa.
        // If shape is [0] or [2,0], num_elements is 0.
        let t5 = Tensor::new(Vec::<f32>::new(), vec![0]).unwrap_or_else(|_| Tensor::new(Vec::<f32>::new(), vec![2,0]).unwrap());
        let expected5 = t5.gelu().unwrap(); // gelu on empty tensor should yield empty tensor
        let actual5 = t5.gelu().unwrap(); // Changed from t5.gelu_simd().unwrap();
        assert_tensors_approx_equal(&actual5, &expected5, FLOAT_TOLERANCE);
        assert_eq!(actual5.data.len(), 0);
    }

    #[test]
    fn test_concat_simple_1d() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2]).unwrap();
        let result = Tensor::concat(&[&t1, &t2], 0).unwrap();
        assert_eq!(result.shape, vec![4]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_2d_axis0() { // Stack rows
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(); // [[1,2],[3,4]]
        let t2 = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();           // [[5,6]]
        let result = Tensor::concat(&[&t1, &t2], 0).unwrap();
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_2d_axis1() { // Stack columns
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(); // [[1,2],[3,4]]
        let t2 = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();           // [[5],[6]]
        let result = Tensor::concat(&[&t1, &t2], 1).unwrap();
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }
    
    #[test]
    fn test_concat_3d_kv_cache_style_axis1() { // e.g. [batch, seq_len, features]
        // Past KV: [1, 2, 4] (batch=1, past_seq_len=2, features=4)
        let past_kv = Tensor::new(
            (0..8).map(|x| x as f32).collect(), // 0,1,2,3, 4,5,6,7
            vec![1, 2, 4]
        ).unwrap();
        // New KV: [1, 1, 4] (batch=1, new_seq_len=1, features=4)
        let new_kv = Tensor::new(
            (8..12).map(|x| x as f32).collect(), // 8,9,10,11
            vec![1, 1, 4]
        ).unwrap();

        let result = Tensor::concat(&[&past_kv, &new_kv], 1).unwrap();
        assert_eq!(result.shape, vec![1, 3, 4]);
        let expected_data = (0..12).map(|x| x as f32).collect::<Vec<f32>>();
        assert_eq!(result.data, expected_data);
    }

    #[test]
    fn test_concat_4d_kv_cache_style_axis2() { // e.g. [batch, num_heads, seq_len, head_dim]
        // Past KV: [1, 2, 2, 3] (B=1, H=2, S_past=2, D_head=3)
        // Data: 0 .. 11
        let past_kv_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let past_kv = Tensor::new(past_kv_data, vec![1, 2, 2, 3]).unwrap();
        // H0, S0: [0,1,2], S1: [3,4,5]
        // H1, S0: [6,7,8], S1: [9,10,11]

        // New KV: [1, 2, 1, 3] (B=1, H=2, S_new=1, D_head=3)
        // Data: 12 .. 17
        let new_kv_data: Vec<f32> = (12..18).map(|x| x as f32).collect();
        let new_kv = Tensor::new(new_kv_data, vec![1, 2, 1, 3]).unwrap();
        // H0, S0: [12,13,14]
        // H1, S0: [15,16,17]
        
        let result = Tensor::concat(&[&past_kv, &new_kv], 2).unwrap(); // Concat along seq_len axis
        assert_eq!(result.shape, vec![1, 2, 3, 3]); // New shape [B, H, S_past+S_new, D_head]

        let expected_data = vec![
            // Batch 0
            // Head 0
            0.0, 1.0, 2.0, // Past S0
            3.0, 4.0, 5.0, // Past S1
            12.0, 13.0, 14.0, // New S0
            // Head 1
            6.0, 7.0, 8.0, // Past S0
            9.0, 10.0, 11.0, // Past S1
            15.0, 16.0, 17.0, // New S0
        ];
        assert_eq!(result.data, expected_data);
    }


    #[test]
    fn test_concat_error_empty_input() {
        let result = Tensor::concat(&[], 0);
        assert!(matches!(result, Err(TensorError::InvalidDimension(s)) if s.contains("Input tensor slice is empty")));
    }

    #[test]
    fn test_concat_error_mismatched_ranks() {
        let t1 = Tensor::new(vec![1.0], vec![1]).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = Tensor::concat(&[&t1, &t2], 0);
        assert!(matches!(result, Err(TensorError::IncompatibleShapes(s)) if s.contains("same rank")) );
    }

    #[test]
    fn test_concat_error_invalid_axis() {
        let t1 = Tensor::new(vec![1.0], vec![1]).unwrap();
        let result = Tensor::concat(&[&t1], 1); // Axis 1 for 1D tensor
        assert!(matches!(result, Err(TensorError::InvalidDimension(s)) if s.contains("out of bounds")));
    }

    #[test]
    fn test_concat_error_mismatched_shapes_non_axis() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let t2 = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap(); // Different size on axis 1
        let result = Tensor::concat(&[&t1, &t2], 0); // Concat along axis 0
        assert!(matches!(result, Err(TensorError::IncompatibleShapes(s)) if s.contains("Dimension 1 mismatch")));
    }
}
