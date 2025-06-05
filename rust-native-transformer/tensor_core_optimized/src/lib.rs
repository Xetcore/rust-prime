use std::fmt::Debug;
use num_traits::{FloatCore, NumAssign};
use half::{f16, bf16};

// TensorError Enum Definition
#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeMismatch(String),
    InvalidDimension(String),
    OutOfBounds(String),
    UnsupportedAxis(String),
    IncompatibleShapes(String),
    UnsupportedOperation(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
            TensorError::InvalidDimension(s) => write!(f, "Invalid dimension: {}", s),
            TensorError::OutOfBounds(s) => write!(f, "Out of bounds: {}", s),
            TensorError::UnsupportedAxis(s) => write!(f, "Unsupported axis: {}", s),
            TensorError::IncompatibleShapes(s) => write!(f, "Incompatible shapes: {}", s),
            TensorError::UnsupportedOperation(s) => write!(f, "Unsupported operation: {}", s),
        }
    }
}

// Helper function for f32 matrix multiplication
fn matmul_f32_compute(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
    let m = a.shape.get(0).copied().unwrap_or(0);
    let k_a = a.shape.get(1).copied().unwrap_or(0);
    let k_b = b.shape.get(0).copied().unwrap_or(0);
    let n = b.shape.get(1).copied().unwrap_or(0);

    // Note: Rank checks are expected to be done by the public matmul calling this.
    // if a.rank() != 2 || b.rank() != 2 {
    //     return Err(TensorError::InvalidDimension(
    //         "Matmul currently only supports 2D tensors".to_string(),
    //     ));
    // }

    // Note: Compatibility checks are expected to be done by the public matmul calling this.
    // if k_a != k_b {
    //     return Err(TensorError::IncompatibleShapes(format!(
    //         "Incompatible shapes for matmul: A has shape [{}, {}], B has shape [{}, {}]",
    //         m, k_a, k_b, n
    //     )));
    // }

    let mut result_data = vec![0.0f32; m * n];

    if cfg!(target_arch = "x86_64") &&
        is_x86_feature_detected!("avx2") &&
        is_x86_feature_detected!("fma") &&
        m > 0 && k_a > 0 && n > 0
    {
        // Transpose B for the SIMD kernel
        let b_transposed = b.transpose()?; // This transpose is on Tensor<f32>
        unsafe {
            // Pass b_transposed.data to the AVX2 kernel
            matmul_2d_avx2_fma(&a.data, &b_transposed.data, m, k_a, n, &mut result_data);
        }
    } else if m > 0 && k_a > 0 && n > 0 {
        // Scalar path
        for i_idx in 0..m {
            for j_idx in 0..n {
                let mut sum = 0.0f32;
                for k_sidx in 0..k_a {
                    // Accessing data directly, assuming a and b are &Tensor<f32>
                    sum += a.data[i_idx * k_a + k_sidx] * b.data[k_sidx * n + j_idx];
                }
                result_data[i_idx * n + j_idx] = sum;
            }
        }
    }
    // If m, n, or k_a is 0, result_data will be empty or loops won't run, which is correct.
    Tensor::new(result_data, vec![m, n])
}

fn softmax_f32_compute(input: &Tensor<f32>, axis: usize) -> Result<Tensor<f32>, TensorError> {
    // Axis validation is implicitly handled by the check `axis >= input.rank()` before calling.
    // However, to be safe, this helper could also perform it if called from elsewhere.
    // For now, assuming the public `softmax` method does the primary axis check.

    let mut result_data = input.data.clone();
    let axis_size = input.shape[axis]; // input.shape instead of self.shape
    if axis_size == 0 {
        return Ok(Tensor::new(result_data, input.shape.clone())?);
    }
    let outer_dims_product: usize = input.shape[..axis].iter().product();
    let inner_dims_product: usize = input.shape[axis + 1..].iter().product();

    for i in 0..outer_dims_product {
        for j in 0..inner_dims_product {
            let mut current_processing_slice: Vec<f32> = Vec::with_capacity(axis_size);
            for k in 0..axis_size {
                let current_flat_idx = input._flat_index_for_softmax(i, k, j, axis, inner_dims_product)?; // input._ instead of self._
                current_processing_slice.push(input.data[current_flat_idx]); // input.data instead of self.data
            }

            if cfg!(target_arch = "x86_64") &&
                is_x86_feature_detected!("avx2") &&
                is_x86_feature_detected!("fma") &&
                !current_processing_slice.is_empty()
            {
                unsafe {
                    softmax_slice_avx2(&mut current_processing_slice);
                }
            } else if !current_processing_slice.is_empty() {
                let max_val_scalar = current_processing_slice.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let mut sum_exp_values_scalar = 0.0f32;

                for val_ref in current_processing_slice.iter_mut() {
                    *val_ref = (*val_ref - max_val_scalar).exp();
                    sum_exp_values_scalar += *val_ref;
                }

                if sum_exp_values_scalar == 0.0 { sum_exp_values_scalar = 1e-9; } // Avoid division by zero
                let inv_sum_exp_scalar = sum_exp_values_scalar.recip();
                for val_ref in current_processing_slice.iter_mut() {
                    *val_ref *= inv_sum_exp_scalar;
                }
            }

            for k in 0..axis_size {
                let current_flat_idx = input._flat_index_for_softmax(i, k, j, axis, inner_dims_product)?; // input._
                if let Some(val_to_write) = current_processing_slice.get(k) {
                        if let Some(target_loc) = result_data.get_mut(current_flat_idx) {
                        *target_loc = *val_to_write;
                        } else {
                        return Err(TensorError::OutOfBounds(format!("Softmax: Target index {} out of bounds for result_data", current_flat_idx)));
                        }
                } else {
                        return Err(TensorError::OutOfBounds(format!("Softmax: Source index {} out of bounds for current_slice", k)));
                }
            }
        }
    }
    Tensor::new(result_data, input.shape.clone())
}

impl<F: FloatType> Tensor<F> {
    pub fn matmul(a: &Tensor<F>, b: &Tensor<F>) -> Result<Tensor<F>, TensorError> {
        if a.rank() != 2 || b.rank() != 2 {
            return Err(TensorError::InvalidDimension(
                "Matmul currently only supports 2D tensors".to_string(),
            ));
        }

        let k_a = a.shape.get(1).copied().unwrap_or(0);
        let k_b = b.shape.get(0).copied().unwrap_or(0);

        if k_a != k_b {
            return Err(TensorError::IncompatibleShapes(format!(
                "Incompatible shapes for matmul: A has shape {:?}, B has shape {:?}",
                a.shape, b.shape
            )));
        }

        let a_f32 = a.convert::<f32>()?;
        let b_f32 = b.convert::<f32>()?;

        let result_f32 = matmul_f32_compute(&a_f32, &b_f32)?;

        result_f32.convert::<F>()
    }

    pub fn softmax(&self, axis: usize) -> Result<Tensor<F>, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::UnsupportedAxis(format!(
                "Axis {} is out of bounds for tensor with rank {}",
                axis, self.rank()
            )));
        }

        let self_f32 = self.convert::<f32>()?;
        let result_f32 = softmax_f32_compute(&self_f32, axis)?;
        result_f32.convert::<F>()
    }
}

impl std::error::Error for TensorError {}

// FloatType Trait Definition
pub trait FloatType:
    Copy + Clone + Debug + PartialEq + PartialOrd + Send + Sync + 'static + FloatCore + NumAssign
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f32(f: f32) -> Self;
    fn to_f32(&self) -> f32;
}

impl FloatType for f32 {
    const ZERO: Self = 0.0f32;
    const ONE: Self = 1.0f32;

    fn from_f32(val: f32) -> Self {
        val
    }

    fn to_f32(&self) -> f32 {
        *self
    }
}

impl FloatType for f16 {
    const ZERO: Self = f16::from_f32_const(0.0);
    const ONE: Self = f16::from_f32_const(1.0);

    fn from_f32(val: f32) -> Self {
        f16::from_f32(val)
    }

    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}

impl FloatType for bf16 {
    const ZERO: Self = bf16::from_f32_const(0.0);
    const ONE: Self = bf16::from_f32_const(1.0);

    fn from_f32(val: f32) -> Self {
        bf16::from_f32(val)
    }

    fn to_f32(&self) -> f32 {
        self.to_f32()
    }
}

// Tensor Struct Definition
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

// Basic Tensor Creation and Manipulation
impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let num_elements_shape: usize = if shape.is_empty() {
            1 // Scalar case: product of empty shape is 1
        } else if shape.iter().any(|&dim| dim == 0) {
            0 // If any dimension is zero, total elements is zero
        } else {
            shape.iter().product()
        };

        if data.len() != num_elements_shape {
            if !(data.is_empty() && num_elements_shape == 0) {
                 if !(data.len() == 1 && num_elements_shape == 1 && shape.is_empty()){
                    return Err(TensorError::ShapeMismatch(format!(
                        "Data length {} does not match shape product {} (shape: {:?})",
                        data.len(),
                        num_elements_shape,
                        shape
                    )));
                 }
            }
        }
        Ok(Tensor { data, shape })
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() { // Scalar
            return 1;
        }
        if self.shape.iter().any(|&dim| dim == 0) {
            0
        } else {
            self.shape.iter().product()
        }
    }

    fn _flat_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if self.rank() == 0 {
            return if indices.is_empty() { Ok(0) } else {
                Err(TensorError::InvalidDimension(format!(
                    "Expected 0 indices for scalar tensor (rank 0), got {}", indices.len()
                )))};
        }
        if indices.len() != self.rank() {
            return Err(TensorError::InvalidDimension(format!(
                "Expected {} indices for tensor of rank {}, got {}", self.rank(), self.rank(), indices.len()
            )));
        }
        let mut flat_idx = 0;
        let mut multiplier = 1;
        for (i, &dim_idx) in indices.iter().rev().enumerate() {
            let current_dim_shape_idx = self.rank() - 1 - i;
            let dim_size = self.shape[current_dim_shape_idx];
            if dim_idx >= dim_size {
                 return Err(TensorError::OutOfBounds(format!(
                    "Index {} out of bounds for dimension {} with size {}", dim_idx, current_dim_shape_idx, dim_size
                )));
            }
            flat_idx += dim_idx * multiplier;
            multiplier *= dim_size;
        }
        Ok(flat_idx)
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        let flat_idx = self._flat_index(indices)?;
        if self.data.is_empty() {
             return Err(TensorError::OutOfBounds("Attempting to get from a tensor with no data elements".to_string()));
        }
        self.data.get(flat_idx).ok_or_else(||
            TensorError::OutOfBounds(format!("Calculated flat index {} out of bounds for data length {}", flat_idx, self.data.len()))
        )
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError> {
        let flat_idx = self._flat_index(indices)?;
        if self.data.is_empty() {
             return Err(TensorError::OutOfBounds("Attempting to get_mut from a tensor with no data elements".to_string()));
        }
        self.data.get_mut(flat_idx).ok_or_else(|| TensorError::OutOfBounds(format!("Calculated flat index {} out of bounds for data length {} (mut access)", flat_idx, self.data.len())))
    }

    // Moved _flat_index_for_softmax here as it's generic
    fn _flat_index_for_softmax(&self, outer_idx: usize, axis_idx: usize, inner_idx: usize, axis: usize, _inner_dims_product: usize) -> Result<usize, TensorError> {
        let mut md_indices = vec![0; self.rank()];
        let mut current_outer = outer_idx;
        for d_rev_idx in 0..axis {
            let d = axis - 1 - d_rev_idx;
            md_indices[d] = current_outer % self.shape[d];
            current_outer /= self.shape[d];
        }
        md_indices[axis] = axis_idx;
        let mut current_inner = inner_idx;
        for d_rev_idx in 0..(self.rank() - 1 - axis) {
            let d = self.rank() - 1 - d_rev_idx;
            md_indices[d] = current_inner % self.shape[d];
            current_inner /= self.shape[d];
        }
        self._flat_index(&md_indices)
    }
}

impl<T: Clone> Tensor<T> {
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let current_num_elements = self.num_elements();
        let new_num_elements: usize = if new_shape.is_empty() {1} else if new_shape.iter().any(|&d| d==0) {0} else {new_shape.iter().product()};
        if current_num_elements != new_num_elements {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot reshape tensor with {} elements (shape {:?}) into shape {:?} ({} new elements)",
                current_num_elements, self.shape, new_shape, new_num_elements
            )));
        }
        Ok(Tensor { data: self.data.clone(), shape: new_shape })
    }

    pub fn transpose(&self) -> Result<Tensor<T>, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::InvalidDimension(
                "Transpose operation only supports 2D tensors.".to_string(),
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let new_shape = vec![cols, rows];

        if self.num_elements() == 0 {
            return Tensor::new(Vec::new(), new_shape);
        }

        let mut new_data = Vec::with_capacity(self.data.len());
        for j_new_row in 0..cols {
            for i_new_col in 0..rows {
                new_data.push(self.data[i_new_col * cols + j_new_row].clone());
            }
        }
        Tensor::new(new_data, new_shape)
    }
}

impl<F: FloatType> Tensor<F> {
    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let num_elements = if shape.is_empty() {
            1 // Scalar
        } else if shape.iter().any(|&d| d == 0) {
            0 // Zero elements if any dimension is zero
        } else {
            shape.iter().product()
        };
        let data = vec![F::ZERO; num_elements];
        Tensor::new(data, shape)
    }

    pub fn convert<T_OUT: FloatType>(&self) -> Result<Tensor<T_OUT>, TensorError> {
        if self.data.is_empty() {
            // Handle tensors with no data (e.g. shape [2,0,3])
            Tensor::new(Vec::new(), self.shape.clone())
        } else {
            let converted_data: Vec<T_OUT> = self
                .data
                .iter()
                .map(|element| T_OUT::from_f32(element.to_f32()))
                .collect();
            Tensor::new(converted_data, self.shape.clone())
        }
    }

    pub fn scalar_mul(&self, scalar: F) -> Result<Tensor<F>, TensorError> {
        if self.data.is_empty() && self.num_elements() == 0 {
            return Ok(self.clone());
        }
        let mut new_data = self.data.clone();
        for val in new_data.iter_mut() {
            *val *= scalar;
        }
        Tensor::new(new_data, self.shape.clone())
    }

    pub fn concat(tensors: &[&Tensor<F>], axis: usize) -> Result<Tensor<F>, TensorError> {
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
        if total_elements == 0 {
            return Tensor::new(Vec::new(), output_shape);
        }
        let mut output_data = Vec::with_capacity(total_elements);
        let mut first_tensor_strides = vec![0; rank];
        if rank > 0 {
            first_tensor_strides[rank - 1] = 1;
            for d in (0..rank - 1).rev() {
                first_tensor_strides[d] = first_tensor_strides[d + 1] * first_tensor.shape[d + 1];
            }
        }
        let outer_dims_product: usize = first_tensor.shape[..axis].iter().product();
        for outer_idx in 0..outer_dims_product {
            for t_ref in tensors {
                let current_tensor_axis_dim = t_ref.shape[axis];
                let current_tensor_inner_dims_product: usize = t_ref.shape[axis+1..].iter().product();
                for axis_el_idx in 0..current_tensor_axis_dim {
                    let mut current_input_flat_idx = 0;
                    let mut temp_outer_idx = outer_idx;
                    for d_rev_idx in 0..axis { // Corrected d_idx_rev to d_rev_idx
                        let d = axis - 1 - d_rev_idx;
                        current_input_flat_idx += (temp_outer_idx % first_tensor.shape[d]) * first_tensor_strides[d];
                        temp_outer_idx /= first_tensor.shape[d];
                    }
                    current_input_flat_idx += axis_el_idx * current_tensor_inner_dims_product;

                    if t_ref.data.is_empty() && current_tensor_inner_dims_product > 0 {
                         return Err(TensorError::ShapeMismatch(format!("Tensor data is empty but shape {:?} implies non-empty for concat.", t_ref.shape)));
                    }
                    if !t_ref.data.is_empty() {
                        if current_input_flat_idx + current_tensor_inner_dims_product <= t_ref.data.len() {
                             output_data.extend_from_slice(&t_ref.data[current_input_flat_idx .. current_input_flat_idx + current_tensor_inner_dims_product]);
                        } else {
                            return Err(TensorError::OutOfBounds(format!("Concat: Calculated source slice (start: {}, len: {}) for tensor with data len {} is out of bounds. Tensor shape: {:?}, Outer idx: {}, Axis el idx: {}", current_input_flat_idx, current_tensor_inner_dims_product, t_ref.data.len(), t_ref.shape, outer_idx, axis_el_idx)));
                        }
                    } else if current_tensor_inner_dims_product > 0 {
                         return Err(TensorError::ShapeMismatch(format!("Concat: Tensor data is empty but shape {:?} implies non-empty elements to copy.", t_ref.shape)));
                    }
                }
            }
        }
        Tensor::new(output_data, output_shape)
    }
}

// SIMD specific imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Define block sizes for cache-aware matmul
const MATMUL_BLOCK_M: usize = 32;
const MATMUL_BLOCK_K: usize = 64;
const MATMUL_BLOCK_N: usize = 24;

#[cfg(target_arch = "x86_64")]
unsafe fn pack_a_block(
    a_data: &[f32],
    m_start: usize,
    k_start: usize,
    k_dim_orig: usize,
    current_block_m: usize,
    current_block_k: usize,
    packed_a: &mut [f32]
) {
    for val_idx in 0..packed_a.len() {
        *packed_a.get_unchecked_mut(val_idx) = 0.0f32;
    }
    for r_idx_block in 0..current_block_m {
        for k_idx_block in 0..current_block_k {
            let src_row_orig = m_start + r_idx_block;
            let src_col_orig = k_start + k_idx_block;
            let src_flat_idx = src_row_orig * k_dim_orig + src_col_orig;
            let dest_flat_idx = r_idx_block * MATMUL_BLOCK_K + k_idx_block;
            *packed_a.get_unchecked_mut(dest_flat_idx) = *a_data.get_unchecked(src_flat_idx);
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_b_transposed_block(
    b_t_data: &[f32],
    n_start: usize,
    k_start: usize,
    k_dim_orig: usize,
    current_block_n: usize,
    current_block_k: usize,
    packed_b_t: &mut [f32]
) {
    for val_idx in 0..packed_b_t.len() {
        *packed_b_t.get_unchecked_mut(val_idx) = 0.0f32;
    }
    for r_idx_block in 0..current_block_n {
        for k_idx_block in 0..current_block_k {
            let src_row_orig_bt = n_start + r_idx_block;
            let src_col_orig_bt = k_start + k_idx_block;
            let src_flat_idx = src_row_orig_bt * k_dim_orig + src_col_orig_bt;
            let dest_flat_idx = r_idx_block * MATMUL_BLOCK_K + k_idx_block;
            *packed_b_t.get_unchecked_mut(dest_flat_idx) = *b_t_data.get_unchecked(src_flat_idx);
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn matmul_micro_kernel_avx2(
    packed_a: &[f32],
    packed_b_t: &[f32],
    c_data: &mut [f32],
    current_block_m: usize,
    current_block_n: usize,
    current_block_k: usize,
    row_offset_c: usize,
    col_offset_c: usize,
    n_dim_orig_c: usize
) {
    for m_idx in 0..current_block_m {
        for n_idx in 0..current_block_n {
            let a_row_start_ptr = packed_a.as_ptr().add(m_idx * MATMUL_BLOCK_K);
            let b_t_row_start_ptr = packed_b_t.as_ptr().add(n_idx * MATMUL_BLOCK_K);
            let mut sum_vec = _mm256_setzero_ps();
            let mut k_sidx = 0;
            while k_sidx + 7 < current_block_k {
                let a_vec = _mm256_loadu_ps(a_row_start_ptr.add(k_sidx));
                let b_t_vec = _mm256_loadu_ps(b_t_row_start_ptr.add(k_sidx));
                sum_vec = _mm256_fmadd_ps(a_vec, b_t_vec, sum_vec);
                k_sidx += 8;
            }
            let mut dot_product_result = horizontal_sum_m256(sum_vec);
            while k_sidx < current_block_k {
                dot_product_result += (*a_row_start_ptr.add(k_sidx)) * (*b_t_row_start_ptr.add(k_sidx));
                k_sidx += 1;
            }
            let c_flat_idx = (row_offset_c + m_idx) * n_dim_orig_c + (col_offset_c + n_idx);
            *c_data.get_unchecked_mut(c_flat_idx) += dot_product_result;
        }
    }
}

// SIMD Helper functions (Continued from above, or this is where they start if not above)
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn tanhf_approx_avx2(x: __m256) -> __m256 {
    let clamp_val = 3.0;
    let c27_vec = _mm256_set1_ps(27.0);
    let c9_vec = _mm256_set1_ps(9.0);
    let clamp_val_vec = _mm256_set1_ps(clamp_val);
    let neg_clamp_val_vec = _mm256_set1_ps(-clamp_val);
    let x_clamped = _mm256_max_ps(x, neg_clamp_val_vec);
    let x_clamped = _mm256_min_ps(x_clamped, clamp_val_vec);
    let x_sq = _mm256_mul_ps(x_clamped, x_clamped);
    let num = _mm256_mul_ps(x_clamped, _mm256_add_ps(c27_vec, x_sq));
    let den = _mm256_add_ps(c27_vec, _mm256_mul_ps(c9_vec, x_sq));
    _mm256_div_ps(num, den)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_m256(vec: __m256) -> f32 {
    let sum_halves = _mm_add_ps(_mm256_castps256_ps128(vec), _mm256_extractf128_ps(vec, 1));
    let hsum_ps = _mm_hadd_ps(sum_halves, sum_halves);
    let hsum_ps2 = _mm_hadd_ps(hsum_ps, hsum_ps);
    _mm_cvtss_f32(hsum_ps2)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_max_m256(vec: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(vec);
    let vhigh = _mm256_extractf128_ps(vec, 1);
    let vmax128 = _mm_max_ps(vlow, vhigh);
    let vshuf1 = _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 3, 2));
    let vmax_intermediate1 = _mm_max_ps(vmax128, vshuf1);
    let vshuf2 = _mm_shuffle_ps(vmax_intermediate1, vmax_intermediate1, _MM_SHUFFLE(0, 0, 0, 1));
    let vmax_final = _mm_max_ps(vmax_intermediate1, vshuf2);
    _mm_cvtss_f32(vmax_final)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn expf_approx_taylor_avx2(x: __m256) -> __m256 {
    let c1 = _mm256_set1_ps(1.0);
    let c_div2 = _mm256_set1_ps(1.0 / 2.0);
    let c_div3 = _mm256_set1_ps(1.0 / 3.0);
    let c_div4 = _mm256_set1_ps(1.0 / 4.0);
    let c_div5 = _mm256_set1_ps(1.0 / 5.0);
    let mut res = _mm256_fmadd_ps(x, c_div5, c1);
    res = _mm256_fmadd_ps(_mm256_mul_ps(x, c_div4), res, c1);
    res = _mm256_fmadd_ps(_mm256_mul_ps(x, c_div3), res, c1);
    res = _mm256_fmadd_ps(_mm256_mul_ps(x, c_div2), res, c1);
    res = _mm256_fmadd_ps(x, res, c1);
    res = _mm256_max_ps(res, _mm256_setzero_ps());
    res
}

#[cfg(target_arch = "x86_64")]
unsafe fn softmax_slice_avx2(slice_data: &mut [f32]) {
    let lanes = 8;
    let len = slice_data.len();
    if len == 0 { return; }
    let mut i = 0;
    let mut max_val_simd_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        max_val_simd_vec = _mm256_max_ps(max_val_simd_vec, data_vec);
        i += lanes;
    }
    let mut max_val = horizontal_max_m256(max_val_simd_vec);
    for k in i..len {
        if slice_data[k] > max_val { max_val = slice_data[k]; }
    }
    let max_val_bcast_vec = _mm256_set1_ps(max_val);
    let mut exp_values_temp: Vec<f32> = vec![0.0f32; len];
    i = 0;
    let mut sum_exp_vec_acc = _mm256_setzero_ps();
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        let norm_vec = _mm256_sub_ps(data_vec, max_val_bcast_vec);
        let exp_vec = expf_approx_taylor_avx2(norm_vec);
        _mm256_storeu_ps(exp_values_temp.as_mut_ptr().add(i), exp_vec);
        sum_exp_vec_acc = _mm256_add_ps(sum_exp_vec_acc, exp_vec);
        i += lanes;
    }
    let mut total_sum_exp = horizontal_sum_m256(sum_exp_vec_acc);
    for k in i..len {
        let val = *slice_data.get_unchecked(k);
        let norm_val = val - max_val;
        let exp_val = libm::expf(norm_val);
        *exp_values_temp.get_unchecked_mut(k) = exp_val;
        total_sum_exp += exp_val;
    }
    if total_sum_exp == 0.0 { total_sum_exp = 1e-9; }
    let inv_total_sum_exp = total_sum_exp.recip();
    let inv_total_sum_exp_vec = _mm256_set1_ps(inv_total_sum_exp);
    i = 0;
    while i + lanes <= len {
        let exp_vec_loaded = _mm256_loadu_ps(exp_values_temp.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(exp_vec_loaded, inv_total_sum_exp_vec);
        _mm256_storeu_ps(slice_data.as_mut_ptr().add(i), result_vec);
        i += lanes;
    }
    for k in i..len {
        *slice_data.get_unchecked_mut(k) = *exp_values_temp.get_unchecked(k) * inv_total_sum_exp;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn layernorm_slice_avx2(slice_data: &mut [f32], gamma_data: &[f32], beta_data: &[f32], epsilon: f32) {
    let lanes = 8;
    let len = slice_data.len();
    let mut i = 0;
    let mut sum_vec = _mm256_setzero_ps();
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        sum_vec = _mm256_add_ps(sum_vec, data_vec);
        i += lanes;
    }
    let mut total_sum_simd = horizontal_sum_m256(sum_vec);
    for k in i..len { total_sum_simd += slice_data[k]; }
    let mean = total_sum_simd / (len as f32);
    let mean_vec = _mm256_set1_ps(mean);
    i = 0;
    let mut var_sum_vec = _mm256_setzero_ps();
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        let diff_vec = _mm256_sub_ps(data_vec, mean_vec);
        let sq_diff_vec = _mm256_mul_ps(diff_vec, diff_vec);
        var_sum_vec = _mm256_add_ps(var_sum_vec, sq_diff_vec);
        i += lanes;
    }
    let mut total_var_sum_simd = horizontal_sum_m256(var_sum_vec);
    for k in i..len {
        let diff = slice_data[k] - mean;
        total_var_sum_simd += diff * diff;
    }
    let variance = total_var_sum_simd / (len as f32);
    let inv_stddev_val = (variance + epsilon).sqrt().recip();
    let inv_stddev_vec = _mm256_set1_ps(inv_stddev_val);
    i = 0;
    while i + lanes <= len {
        let d_ptr = slice_data.as_mut_ptr().add(i);
        let g_ptr = gamma_data.as_ptr().add(i);
        let b_ptr = beta_data.as_ptr().add(i);
        let data_vec = _mm256_loadu_ps(d_ptr);
        let gamma_vec = _mm256_loadu_ps(g_ptr);
        let beta_vec = _mm256_loadu_ps(b_ptr);
        let normalized_vec = _mm256_mul_ps(_mm256_sub_ps(data_vec, mean_vec), inv_stddev_vec);
        let result_vec = _mm256_fmadd_ps(normalized_vec, gamma_vec, beta_vec);
        _mm256_storeu_ps(d_ptr, result_vec);
        i += lanes;
    }
    for k in i..len {
        let normalized_x = (slice_data[k] - mean) * inv_stddev_val;
        slice_data[k] = normalized_x * gamma_data[k] + beta_data[k];
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn gelu_avx2(data_slice: &mut [f32]) {
    let lanes = 8;
    let mut i = 0;
    let len = data_slice.len();
    let c_half = _mm256_set1_ps(0.5);
    let c_one = _mm256_set1_ps(1.0);
    let c_inv_sqrt_2 = _mm256_set1_ps(1.0 / std::f32::consts::SQRT_2);
    while i + lanes <= len {
        let ptr = data_slice.as_mut_ptr().add(i);
        let x_vec = _mm256_loadu_ps(ptr);
        let v = _mm256_mul_ps(x_vec, c_inv_sqrt_2);
        let tanh_v = tanhf_approx_avx2(v);
        let sum_val = _mm256_add_ps(c_one, tanh_v);
        let mul_val = _mm256_mul_ps(x_vec, sum_val);
        let result_vec = _mm256_mul_ps(c_half, mul_val);
        _mm256_storeu_ps(ptr, result_vec);
        i += lanes;
    }
    for k_base in i..len {
        let x_val = data_slice[k_base];
        let x_f64 = x_val as f64;
        let result_f64 = 0.5 * x_f64 * (1.0 + (x_f64 / std::f64::consts::SQRT_2).tanh());
        data_slice[k_base] = result_f64 as f32;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn matmul_2d_avx2_fma(
    a_data: &[f32],
    b_transposed_data: &[f32], // B is already transposed (shape [n_dim, k_dim])
    m: usize,
    k_dim: usize, // Common dimension K
    n_dim: usize,
    c_data: &mut [f32], // Output C matrix data (shape [m, n_dim])
) {
    // Assumes c_data is already zeroed out by the caller (e.g., Tensor::matmul)

    let mut n_c = 0; // Start of current N-block (iterates over columns of C / rows of B^T)
    while n_c < n_dim {
        let current_block_n = std::cmp::min(MATMUL_BLOCK_N, n_dim - n_c);

        let mut k_c = 0; // Start of current K-block (iterates over the common K dimension)
        while k_c < k_dim {
            let current_block_k = std::cmp::min(MATMUL_BLOCK_K, k_dim - k_c);

            // Allocate buffer for the packed block of B^T.
            // Size is fixed by MATMUL_BLOCK_N and MATMUL_BLOCK_K for the buffer.
            // The actual data packed is current_block_n x current_block_k.
            let mut packed_b_t_buffer = vec![0.0f32; MATMUL_BLOCK_N * MATMUL_BLOCK_K];
            pack_b_transposed_block(
                b_transposed_data,
                n_c,                // n_start (row start in B^T)
                k_c,                // k_start (col start in B^T)
                k_dim,              // k_dim_orig for B^T (stride of B^T, which is its number of columns = original k_dim)
                current_block_n,    // actual rows to pack from B^T
                current_block_k,    // actual columns to pack from B^T
                &mut packed_b_t_buffer,
            );

            let mut m_c = 0; // Start of current M-block (iterates over rows of C / rows of A)
            while m_c < m {
                let current_block_m = std::cmp::min(MATMUL_BLOCK_M, m - m_c);

                // Allocate buffer for the packed block of A.
                // Size is fixed by MATMUL_BLOCK_M and MATMUL_BLOCK_K.
                let mut packed_a_buffer = vec![0.0f32; MATMUL_BLOCK_M * MATMUL_BLOCK_K];
                pack_a_block(
                    a_data,
                    m_c,                // m_start (row start in A)
                    k_c,                // k_start (col start in A)
                    k_dim,              // k_dim_orig for A (stride of A, which is its number of columns = k_dim)
                    current_block_m,    // actual rows to pack from A
                    current_block_k,    // actual columns to pack from A
                    &mut packed_a_buffer,
                );

                matmul_micro_kernel_avx2(
                    &packed_a_buffer,
                    &packed_b_t_buffer,
                    c_data,
                    current_block_m,    // current rows in A_block and C_sub_block
                    current_block_n,    // current columns in C_sub_block (rows in B_T_block)
                    current_block_k,    // current common K dimension for this pass
                    m_c,                // row_offset_c (start row of C_sub_block in main C)
                    n_c,                // col_offset_c (start col of C_sub_block in main C)
                    n_dim,              // n_dim_orig_c (total columns/stride of main C matrix)
                );
                m_c += MATMUL_BLOCK_M;
            } // end m_c loop (rows of A and C)
            k_c += MATMUL_BLOCK_K;
        } // end k_c loop (common K dimension)
        n_c += MATMUL_BLOCK_N;
    } // end n_c loop (columns of C / rows of B^T)
}

impl Tensor<f32> {
    pub fn gelu_simd(&self) -> Result<Tensor<f32>, TensorError> {
        Err(TensorError::UnsupportedOperation("gelu_simd requires portable_simd feature, which is unstable.".to_string()))
    }

    pub fn matmul_simd(&self, _other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        Err(TensorError::UnsupportedOperation("matmul_simd requires portable_simd feature, which is unstable.".to_string()))
    }

    // pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
    //     let m = a.shape.get(0).copied().unwrap_or(0);
    //     let k_a = a.shape.get(1).copied().unwrap_or(0);
    //     let k_b = b.shape.get(0).copied().unwrap_or(0);
    //     let n = b.shape.get(1).copied().unwrap_or(0);

    //     if a.rank() != 2 || b.rank() != 2 {
    //         return Err(TensorError::InvalidDimension(
    //             "Matmul currently only supports 2D tensors".to_string(),
    //         ));
    //     }

    //     if k_a != k_b {
    //         return Err(TensorError::IncompatibleShapes(format!(
    //             "Incompatible shapes for matmul: A has shape [{}, {}], B has shape [{}, {}]",
    //             m, k_a, k_b, n
    //         )));
    //     }

    //     let mut result_data = vec![0.0f32; m * n];

    //     if cfg!(target_arch = "x86_64") &&
    //        is_x86_feature_detected!("avx2") &&
    //        is_x86_feature_detected!("fma") &&
    //        m > 0 && k_a > 0 && n > 0
    //     {
    //         // Transpose B for the SIMD kernel
    //         let b_transposed = b.transpose()?;
    //         unsafe {
    //             // Pass b_transposed.data to the AVX2 kernel
    //             matmul_2d_avx2_fma(&a.data, &b_transposed.data, m, k_a, n, &mut result_data);
    //         }
    //     } else if m > 0 && k_a > 0 && n > 0 {
    //         // Scalar path
    //         for i_idx in 0..m {
    //             for j_idx in 0..n {
    //                 let mut sum = 0.0;
    //                 for k_sidx in 0..k_a {
    //                     sum += a.data[i_idx * k_a + k_sidx] * b.data[k_sidx * n + j_idx];
    //                 }
    //                 result_data[i_idx * n + j_idx] = sum;
    //             }
    //         }
    //     }
    //     // If m, n, or k_a is 0, result_data will be empty or loops won't run, which is correct.
    //     Tensor::new(result_data, vec![m, n])
    // }

    // Removed softmax from impl Tensor<f32>
    // pub fn softmax(&self, axis: usize) -> Result<Tensor<f32>, TensorError> { ... }

    // _flat_index_for_softmax was here, moved to impl<T> Tensor<T>

    pub fn layernorm(&self, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<Tensor<f32>, TensorError> {
        if self.rank() == 0 {
            return Err(TensorError::InvalidDimension("LayerNorm not supported for scalar tensors".to_string()));
        }
        let last_dim_size = *self.shape.last().ok_or_else(|| TensorError::InvalidDimension("Cannot get last dim of empty shape".to_string()))?;
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

        let mut result_data = self.data.clone();
        if last_dim_size == 0 {
            return Tensor::new(result_data, self.shape.clone());
        }
        let num_vectors = result_data.len() / last_dim_size;

        for i in 0..num_vectors {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            let current_mut_slice = &mut result_data[start..end];

            if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    layernorm_slice_avx2(current_mut_slice, &gamma.data, &beta.data, epsilon);
                }
            } else {
                let mean_scalar: f32 = current_mut_slice.iter().sum::<f32>() / (last_dim_size as f32);
                let variance_scalar: f32 = current_mut_slice.iter().map(|&x| (x - mean_scalar).powi(2)).sum::<f32>() / (last_dim_size as f32);
                let inv_stddev_scalar = (variance_scalar + epsilon).sqrt().recip();

                for j in 0..last_dim_size {
                    let normalized_x = (current_mut_slice[j] - mean_scalar) * inv_stddev_scalar;
                    current_mut_slice[j] = normalized_x * gamma.data[j] + beta.data[j];
                }
            }
        }
        Tensor::new(result_data, self.shape.clone())
    }

    pub fn gelu(&self) -> Result<Tensor<f32>, TensorError> {
        let mut result_data = self.data.clone();

        if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2") {
            unsafe {
                gelu_avx2(&mut result_data);
            }
        } else {
            for x_val_ref in result_data.iter_mut() {
                let x = *x_val_ref as f64;
                let gelu_val = 0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).tanh());
                *x_val_ref = gelu_val as f32;
            }
        }
        Tensor::new(result_data, self.shape.clone())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::SQRT_2;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    const FLOAT_TOLERANCE: f32 = 1e-5;

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

    fn create_random_tensor(shape: Vec<usize>, seed: u64) -> Tensor<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let num_elements = if shape.is_empty() {1} else {shape.iter().filter(|&&d| d != 0).product()};
        let data = (0..num_elements).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        Tensor::new(data, shape).unwrap()
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
        assert_eq!(result.err(), Some(TensorError::ShapeMismatch("Data length 3 does not match shape product 4 (shape: [2, 2])".to_string())));
    }

    #[test]
    fn test_tensor_new_empty_shape_non_empty_data() {
        let result = Tensor::new(vec![1.0, 2.0], vec![]);
         assert_eq!(result.err(), Some(TensorError::ShapeMismatch(
            "Data length 2 does not match shape product 1 (shape: [])".to_string()
        )));
    }

    #[test]
    fn test_tensor_new_scalar_empty_shape() {
        let t = Tensor::new(vec![5.0], vec![]).unwrap();
        assert_eq!(t.data, vec![5.0]);
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.rank(), 0);
        assert_eq!(t.num_elements(), 1);
    }

    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<f32> = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(t.data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.shape, vec![2, 3]);
        let scalar_zeros: Tensor<f32> = Tensor::zeros(vec![]).unwrap();
        assert_eq!(scalar_zeros.data, vec![0.0]);
        assert_eq!(scalar_zeros.shape, Vec::<usize>::new());
         let empty_zeros: Tensor<f32> = Tensor::zeros(vec![2,0,3]).unwrap();
        assert_eq!(empty_zeros.data.len(), 0);
        assert_eq!(empty_zeros.shape, vec![2,0,3]);

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
        assert!(t.get(&[0,0]).is_err());
        let scalar_t = Tensor::new(vec![1.0], vec![]).unwrap();
        assert!(scalar_t.get(&[0]).is_err());
    }

    #[test]
    fn test_tensor_get_mut() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        *t.get_mut(&[0, 1]).unwrap() = 5.0;
        assert_eq!(t.get(&[0, 1]), Ok(&5.0));
        let mut scalar_t = Tensor::new(vec![42.0], vec![]).unwrap();
        *scalar_t.get_mut(&[]).unwrap() = 43.0;
        assert_eq!(scalar_t.get(&[]), Ok(&43.0));
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t_reshaped = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(t_reshaped.shape, vec![3, 2]);
        assert_eq!(t_reshaped.data, t.data);
        let t_reshaped_flat = t.reshape(vec![6]).unwrap();
        assert_eq!(t_reshaped_flat.shape, vec![6]);
        let scalar_t = Tensor::new(vec![5.0], vec![]).unwrap();
        let scalar_reshaped = scalar_t.reshape(vec![1,1]).unwrap();
        assert_eq!(scalar_reshaped.shape, vec![1,1]);
        let scalar_reshaped_empty = scalar_t.reshape(vec![]).unwrap();
        assert_eq!(scalar_reshaped_empty.shape, vec![]);
    }

    #[test]
    fn test_tensor_reshape_incompatible() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = t.reshape(vec![1, 3]);
        assert!(result.is_err());
        match result.err().unwrap() {
            TensorError::ShapeMismatch(_) => {}
            _ => panic!("Unexpected error type for reshape incompatibility"),
        }
    }
     #[test]
    fn test_tensor_transpose() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t1_transposed = t1.transpose().unwrap();
        assert_eq!(t1_transposed.shape, vec![3, 2]);
        assert_eq!(t1_transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let t2_transposed = t2.transpose().unwrap();
        assert_eq!(t2_transposed.shape, vec![1, 3]);
        assert_eq!(t2_transposed.data, vec![1.0, 2.0, 3.0]);

        let t3 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let t3_transposed = t3.transpose().unwrap();
        assert_eq!(t3_transposed.shape, vec![3, 1]);
        assert_eq!(t3_transposed.data, vec![1.0, 2.0, 3.0]);

        let t4_empty_data_valid_shape = Tensor::new(Vec::<f32>::new(), vec![0,3]).unwrap();
        let t4_transposed = t4_empty_data_valid_shape.transpose().unwrap();
        assert_eq!(t4_transposed.shape, vec![3,0]);
        assert!(t4_transposed.data.is_empty());

        let t5_empty_data_zero_shape = Tensor::new(Vec::<f32>::new(), vec![0,0]).unwrap();
        let t5_transposed = t5_empty_data_zero_shape.transpose().unwrap();
        assert_eq!(t5_transposed.shape, vec![0,0]);
        assert!(t5_transposed.data.is_empty());
    }

    #[test]
    fn test_transpose_invalid_dim() {
        let t_1d = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(t_1d.transpose().is_err());
        let t_3d = Tensor::new(vec![1.0; 8], vec![2,2,2]).unwrap();
        assert!(t_3d.transpose().is_err());
    }


    #[test]
    fn test_matmul_2x2_2x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let result = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let result = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_incompatible_shapes() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = Tensor::matmul(&a, &b);
        assert!(matches!(result, Err(TensorError::IncompatibleShapes(_))));
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
        let result = t.softmax(0).unwrap();
        let expected = vec![0.09003057, 0.24472847, 0.66524096];
        assert_eq!(result.shape, vec![3]);
        assert_f32_slice_eq(&result.data, &expected, 1e-6);
    }

    #[test]
    fn test_layernorm_simple() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let beta = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]).unwrap();
        let epsilon = 1e-5;
        let result = input.layernorm(&gamma, &beta, epsilon).unwrap();
        let expected_data = vec![
            -1.2247448, 0.0, 1.2247448,
            -1.2247448, 0.0, 1.2247448,
        ];
        assert_eq!(result.shape, input.shape);
        assert_f32_slice_eq(&result.data, &expected_data, 1e-5);
    }

    #[test]
    fn test_gelu() {
        let input = Tensor::new(vec![0.0, 1.0, -1.0, 2.0, -2.0], vec![5]).unwrap();
        let expected_data = vec![
            0.0,
            0.5 * 1.0 * (1.0 + (1.0f64 / std::f64::consts::SQRT_2).tanh()) as f32,
            0.5 * -1.0 * (1.0 + (-1.0f64 / std::f64::consts::SQRT_2).tanh()) as f32,
            0.5 * 2.0 * (1.0 + (2.0f64 / std::f64::consts::SQRT_2).tanh()) as f32,
            0.5 * -2.0 * (1.0 + (-2.0f64 / std::f64::consts::SQRT_2).tanh()) as f32,
        ];
        let result = input.gelu().unwrap();
        assert_eq!(result.shape, input.shape);
        assert_f32_slice_eq(&result.data, &expected_data, FLOAT_TOLERANCE);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_tanhf_approx_avx2() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not detected, skipping test_tanhf_approx_avx2.");
            return;
        }
        unsafe {
            let test_inputs = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 0.1, -0.1];
            let mut results_approx = Vec::with_capacity(test_inputs.len());
            let mut results_precise = Vec::with_capacity(test_inputs.len());
            for &val in &test_inputs { results_precise.push(libm::tanhf(val)); }
            let mut i = 0;
            while i + 8 <= test_inputs.len() {
                let input_slice = &test_inputs[i..i+8];
                let input_vec = _mm256_loadu_ps(input_slice.as_ptr());
                let tanh_vec = tanhf_approx_avx2(input_vec);
                let mut output_slice = [0.0f32; 8];
                _mm256_storeu_ps(output_slice.as_mut_ptr(), tanh_vec);
                results_approx.extend_from_slice(&output_slice);
                i += 8;
            }
            for k in i..test_inputs.len() {
                let input_val = test_inputs[k];
                 let clamp_val = 3.0;
                 let x_clamped = input_val.max(-clamp_val).min(clamp_val);
                 let x_sq = x_clamped * x_clamped;
                 let num = x_clamped * (27.0 + x_sq);
                 let den = 27.0 + 9.0 * x_sq;
                 results_approx.push(num/den);
            }
            for (idx, (approx, precise)) in results_approx.iter().zip(results_precise.iter()).enumerate() {
                let input_val = test_inputs[idx];
                let tolerance = if input_val.abs() <= 3.0 { 0.01 } else { 0.1 };
                assert!((approx - precise).abs() < tolerance, "Input: {}, Approx: {}, Precise: {}, Diff: {}", input_val, approx, precise, (approx-precise).abs());
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_expf_approx_taylor_avx2() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            eprintln!("AVX2 or FMA not detected, skipping test_expf_approx_taylor_avx2.");
            return;
        }
        unsafe {
            let test_inputs = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, -0.1, -0.01, -4.0, -2.5, -1.5, -0.75, -0.25];
            let mut results_approx = Vec::with_capacity(test_inputs.len());
            let mut results_precise = Vec::with_capacity(test_inputs.len());
            for &val in &test_inputs { results_precise.push(libm::expf(val)); }
            let mut i = 0;
            while i + 8 <= test_inputs.len() {
                let input_slice = &test_inputs[i..i+8];
                let input_vec = _mm256_loadu_ps(input_slice.as_ptr());
                let exp_vec = expf_approx_taylor_avx2(input_vec);
                let mut output_slice = [0.0f32; 8];
                _mm256_storeu_ps(output_slice.as_mut_ptr(), exp_vec);
                results_approx.extend_from_slice(&output_slice);
                i += 8;
            }
            for k in i..test_inputs.len() {
                let x = test_inputs[k];
                let mut res = 1.0 + x/5.0;
                res = 1.0 + x/4.0 * res;
                res = 1.0 + x/3.0 * res;
                res = 1.0 + x/2.0 * res;
                res = 1.0 + x * res;
                results_approx.push(res.max(0.0));
            }
            for (idx, (approx, precise)) in results_approx.iter().zip(results_precise.iter()).enumerate() {
                let input_val = test_inputs[idx];
                let tolerance = if input_val > -2.0 { 0.01 } else if input_val > -4.0 { 0.05 } else { 0.2 };
                assert!((approx - precise).abs() < tolerance, "Input: {}, Approx: {}, Precise: {}, Diff: {}", input_val, approx, precise, (approx-precise).abs());
            }
        }
    }
}

[end of rust-native-transformer/tensor_core_optimized/src/lib.rs]

[end of rust-native-transformer/tensor_core_optimized/src/lib.rs]

[end of rust-native-transformer/tensor_core_optimized/src/lib.rs]
