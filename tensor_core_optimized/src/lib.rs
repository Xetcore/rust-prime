use half::{f16, bf16};
use num_traits::{Float, Zero}; // Added Zero
use std::arch::x86_64::*;
use std::ops::{Add, Mul, Div, Sub}; // For potential generic operations

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    DimensionMismatch(String),
    OutOfBounds(String),
    InvalidShape(String),
    UnsupportedType(String),
    ConversionError(String),
    GenericError(String),
}

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    // Stride might be added later for more advanced operations
}

// --- Generic Tensor Operations ---
impl<T: Clone> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        if data.len() != shape.iter().product() {
            if shape.is_empty() && data.len() == 1 { // Allow scalar
                // ok
            } else if shape.len() == 1 && shape[0] == 0 && data.is_empty() { // Allow empty tensor
                // ok
            }
            else {
                return Err(TensorError::DimensionMismatch(format!(
                    "Data length {} does not match product of shape dimensions {:?}",
                    data.len(),
                    shape
                )));
            }
        }
        Ok(Tensor { data, shape })
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() { // scalar
            return 1;
        }
        if self.shape.len() == 1 && self.shape[0] == 0 { // empty
            return 0;
        }
        self.shape.iter().product()
    }

    fn _flat_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if self.shape.is_empty() { // Scalar tensor
            if indices.is_empty() || (indices.len() == 1 && indices[0] == 0) {
                return Ok(0);
            } else {
                return Err(TensorError::OutOfBounds(
                    "Indices for scalar tensor must be empty or [0]".to_string(),
                ));
            }
        }
        if indices.len() != self.rank() {
            return Err(TensorError::InvalidShape(format!(
                "Number of indices {} does not match tensor rank {}",
                indices.len(),
                self.rank()
            )));
        }
        let mut index = 0;
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
            index += dim_idx * multiplier;
            multiplier *= dim_size;
        }
        Ok(index)
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        let flat_index = self._flat_index(indices)?;
        self.data.get(flat_index).ok_or_else(|| {
            TensorError::OutOfBounds("Calculated flat index is out of data bounds".to_string())
        })
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError> {
        let flat_index = self._flat_index(indices)?;
        self.data.get_mut(flat_index).ok_or_else(|| {
            TensorError::OutOfBounds("Calculated flat index is out of data bounds".to_string())
        })
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        let new_num_elements: usize = new_shape.iter().product();
        if new_num_elements != self.num_elements() {
            return Err(TensorError::DimensionMismatch(
                "Total number of elements must remain the same after reshape".to_string(),
            ));
        }
        Ok(Tensor {
            data: self.data.clone(), // Data is cloned
            shape: new_shape,
        })
    }

    pub fn transpose(&self) -> Result<Tensor<T>, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedType(
                "Transpose currently only supports 2D tensors.".to_string(),
            ));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let mut new_data = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                new_data.push(self.data[i * n + j].clone());
            }
        }
        Ok(Tensor {
            data: new_data,
            shape: vec![n, m],
        })
    }
}

impl<T: Clone + Zero> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Tensor<T> {
        let num_elements = shape.iter().product();
        let data = vec![T::zero(); num_elements];
        Tensor { data, shape }
    }
}


// --- f32 Specific Operations (including SIMD) ---
// Constants for AVX2 operations (8 f32s)
const AVX2_F32_COUNT: usize = 8;
// MatMul block sizes (example values, can be tuned)
const MATMUL_BLOCK_M: usize = 64;
const MATMUL_BLOCK_K: usize = 32;
const MATMUL_BLOCK_N: usize = 256;


// Helper for tanh approximation with AVX2
#[target_feature(enable = "avx2")]
unsafe fn tanhf_approx_avx2(v: __m256) -> __m256 {
    let v_sq = _mm256_mul_ps(v, v);
    let p = _mm256_add_ps(_mm256_mul_ps(v_sq, _mm256_set1_ps(0.0872929)), _mm256_set1_ps(-0.288679));
    let p = _mm256_add_ps(_mm256_mul_ps(v_sq, p), _mm256_set1_ps(1.0));
    let tanh_v = _mm256_mul_ps(v,p);
    _mm256_max_ps(_mm256_set1_ps(-1.0), _mm256_min_ps(tanh_v, _mm256_set1_ps(1.0)))
}

// Helper for horizontal sum of an __m256 vector
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn horizontal_sum_m256(vec: __m256) -> f32 {
    let hsum = _mm256_hadd_ps(vec, vec);
    let hsum = _mm256_hadd_ps(hsum, hsum);
    let perm = _mm256_permute2f128_ps(hsum, hsum, 1);
    let sum = _mm256_add_ps(hsum, perm);
    _mm_cvtss_f32(_mm256_castps256_ps128(sum))
}

// Helper for horizontal max of an __m256 vector
#[target_feature(enable = "avx2")]
unsafe fn horizontal_max_m256(vec: __m256) -> f32 {
    let perm1 = _mm256_permute_ps(vec, 0b_01_00_11_10);
    let max1 = _mm256_max_ps(vec, perm1);
    let perm2 = _mm256_permute_ps(max1, 0b_10_11_00_01);
    let max2 = _mm256_max_ps(max1, perm2);
    let low_128 = _mm256_castps256_ps128(max2);
    let high_128 = _mm256_extractf128_ps(max2, 1);
    let max_128 = _mm_max_ps(low_128, high_128);
    let vhigh = _mm256_extractf128_ps(vec, 1); // This part had a slight logic error, simplified
    let vlow = _mm256_castps256_ps128(vec);
    let max_val_128_lanes = _mm_max_ps(vhigh, vlow);
    let temp1 = _mm_shuffle_ps(max_val_128_lanes, max_val_128_lanes, _MM_SHUFFLE(0,0,3,2));
    let max_val_64 = _mm_max_ps(max_val_128_lanes,temp1);
    let temp2 = _mm_shuffle_ps(max_val_64,max_val_64, _MM_SHUFFLE(0,0,0,1));
    let max_val_32 = _mm_max_ps(max_val_64,temp2);
    _mm_cvtss_f32(max_val_32)
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn expf_approx_taylor_avx2(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-10.0));
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let mut res = _mm256_fmadd_ps(x, c5, c4);
    res = _mm256_fmadd_ps(x, res, c3);
    res = _mm256_fmadd_ps(x, res, c2);
    res = _mm256_fmadd_ps(x, res, c1);
    res = _mm256_fmadd_ps(x, res, c0);
    _mm256_max_ps(res, _mm256_set1_ps(0.0))
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn pack_a_block(
    _a_data: &[f32], a_ptr: *const f32, m_start: usize, k_start: usize,
    block_m: usize, block_k: usize, k_dim: usize, packed_a: &mut [f32]
) {
    let mut packed_idx = 0;
    for i in 0..block_m {
        for j in 0..block_k {
            packed_a[packed_idx] = *a_ptr.add((m_start + i) * k_dim + (k_start + j));
            packed_idx += 1;
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn pack_b_transposed_block( // This packs a block of B_transposed (N_orig x K_orig) into (K_block x N_block) column major for microkernel
    _b_transposed_data: &[f32], b_transposed_ptr: *const f32, k_block_start: usize, n_block_start: usize,
    block_k: usize, block_n: usize, k_dim_original_b: usize, packed_b: &mut [f32]
) {
    let mut packed_idx = 0;
    // b_transposed_data is (N_orig x K_orig)
    // We want to pack a sub-block of it: (rows from k_block_start to k_block_start + block_k)
    //                                    (cols from n_block_start to n_block_start + block_n)
    // into packed_b with dimensions (block_k x block_n) but in column-major order for B.
    for n_idx_in_block in 0..block_n { // Iterate columns of the block of B we want to form
        for k_idx_in_block in 0..block_k { // Iterate rows of the block of B we want to form
            // Accessing B_transposed at row (n_block_start + n_idx_in_block), col (k_block_start + k_idx_in_block)
            packed_b[packed_idx] = *b_transposed_ptr.add(
                (n_block_start + n_idx_in_block) * k_dim_original_b + (k_block_start + k_idx_in_block)
            );
            packed_idx += 1;
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_micro_kernel_avx2( // Simplified: processes 8 rows of A, 8 cols of B at a time
    packed_a_ptr: *const f32, // Block of A (M_block x K_block), row-major
    packed_b_ptr: *const f32, // Block of B (K_block x N_block), col-major (transposed from original B's block)
    c_ptr: *mut f32,          // Output C matrix (M x N)
    m_start_in_c: usize, n_start_in_c: usize, // Top-left corner of the C sub-block
    block_k: usize, n_dim_c: usize, // K of the block, N of the overall C matrix
    micro_m: usize, micro_n: usize // Actual dimensions of this micro-kernel pass (e.g. 8x8)
) {
    let mut c_accum = [_mm256_setzero_ps(); 8]; // Max 8 rows of A processed by AVX2 vector at once

    for k_loop in 0..block_k {
        // Load 8 values from packed_b (column k_loop, for micro_n columns)
        // packed_b is (K_block x N_block) column-major.
        // So, k_loop'th row of original B block, now k_loop'th "column group" in packed_b.
        let b_vals = _mm256_loadu_ps(packed_b_ptr.add(k_loop * micro_n)); // Assumes micro_n is 8

        for m_idx in 0..micro_m { // For each of the M rows of A we are processing
            let a_val = *packed_a_ptr.add(m_idx * block_k + k_loop); // A is row-major
            let a_vec = _mm256_set1_ps(a_val);
            c_accum[m_idx] = _mm256_fmadd_ps(a_vec, b_vals, c_accum[m_idx]);
        }
    }

    for m_idx in 0..micro_m {
        let c_elem_ptr = c_ptr.add((m_start_in_c + m_idx) * n_dim_c + n_start_in_c);
        let existing_c_vals = _mm256_loadu_ps(c_elem_ptr);
        _mm256_storeu_ps(c_elem_ptr, _mm256_add_ps(existing_c_vals, c_accum[m_idx]));
    }
}

impl Tensor<f32> {
    #[target_feature(enable = "avx2")]
    unsafe fn gelu_avx2(data_slice: &mut [f32]) {
        let mut i = 0;
        let n = data_slice.len();
        while i + AVX2_F32_COUNT <= n {
            let v = _mm256_loadu_ps(data_slice.as_ptr().add(i));
            let c_sqrt_2_pi = _mm256_set1_ps(std::f32::consts::FRAC_2_SQRT_PI);
            let c_0044715 = _mm256_set1_ps(0.044715);
            let x_cubed = _mm256_mul_ps(v, _mm256_mul_ps(v, v));
            let inner_sum = _mm256_add_ps(v, _mm256_mul_ps(c_0044715, x_cubed));
            let tanh_arg = _mm256_mul_ps(c_sqrt_2_pi, inner_sum);
            let tanh_val = tanhf_approx_avx2(tanh_arg);
            let one = _mm256_set1_ps(1.0);
            let half = _mm256_set1_ps(0.5);
            let intermediate = _mm256_mul_ps(v, _mm256_add_ps(one, tanh_val));
            let result = _mm256_mul_ps(half, intermediate);
            _mm256_storeu_ps(data_slice.as_mut_ptr().add(i), result);
            i += AVX2_F32_COUNT;
        }
        for val in data_slice.iter_mut().skip(i) {
            *val = 0.5 * *val * (1.0 + libm::tanhf(std::f32::consts::FRAC_2_SQRT_PI * (*val + 0.044715 * val.powi(3))));
        }
    }

    pub fn gelu(&mut self) -> Result<(), TensorError> {
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::gelu_avx2(&mut self.data); }
        } else {
            for x in self.data.iter_mut() {
                *x = 0.5 * *x * (1.0 + libm::tanhf(std::f32::consts::FRAC_2_SQRT_PI * (*x + 0.044715 * x.powi(3))));
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn layernorm_slice_avx2(
        slice_data: &mut [f32], gamma_data: &[f32], beta_data: &[f32], epsilon: f32
    ) {
        let n = slice_data.len();
        if n == 0 { return; }
        let mut sum = _mm256_setzero_ps();
        let mut sum_sq = _mm256_setzero_ps();
        let mut i = 0;
        while i + AVX2_F32_COUNT <= n {
            let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
            sum = _mm256_add_ps(sum, data_vec);
            sum_sq = _mm256_fmadd_ps(data_vec, data_vec, sum_sq);
            i += AVX2_F32_COUNT;
        }
        let mut total_sum = horizontal_sum_m256(sum);
        let mut total_sum_sq = horizontal_sum_m256(sum_sq);
        for j in i..n {
            total_sum += slice_data[j];
            total_sum_sq += slice_data[j] * slice_data[j];
        }
        let mean = total_sum / (n as f32);
        let variance = total_sum_sq / (n as f32) - mean * mean;
        let std_dev_inv = 1.0 / (variance + epsilon).sqrt();
        let mean_vec = _mm256_set1_ps(mean);
        let std_dev_inv_vec = _mm256_set1_ps(std_dev_inv);
        i = 0;
        while i + AVX2_F32_COUNT <= n {
            let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
            let gamma_vec = _mm256_loadu_ps(gamma_data.as_ptr().add(i));
            let beta_vec = _mm256_loadu_ps(beta_data.as_ptr().add(i));
            let normalized_vec = _mm256_mul_ps(_mm256_sub_ps(data_vec, mean_vec), std_dev_inv_vec);
            let result_vec = _mm256_fmadd_ps(normalized_vec, gamma_vec, beta_vec);
            _mm256_storeu_ps(slice_data.as_mut_ptr().add(i), result_vec);
            i += AVX2_F32_COUNT;
        }
        for j in i..n {
            let normalized = (slice_data[j] - mean) * std_dev_inv;
            slice_data[j] = normalized * gamma_data[j] + beta_data[j];
        }
    }

    pub fn layernorm(&mut self, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<(), TensorError> {
        if self.rank() == 0 { return Ok(()); }
        let last_dim_size = *self.shape.last().unwrap();
        if gamma.shape != [last_dim_size] || beta.shape != [last_dim_size] {
            return Err(TensorError::DimensionMismatch(
                "Gamma/Beta dimensions must match last dimension of tensor".to_string(),
            ));
        }
        let num_slices = self.data.len() / last_dim_size;
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        for i in 0..num_slices {
            let slice_start = i * last_dim_size;
            let slice_end = (i + 1) * last_dim_size;
            let current_slice = &mut self.data[slice_start..slice_end];
            if use_avx2 {
                unsafe { Self::layernorm_slice_avx2(current_slice, &gamma.data, &beta.data, epsilon); }
            } else {
                let mut sum = 0.0;
                for val in current_slice.iter() { sum += *val; }
                let mean = sum / (last_dim_size as f32);
                let mut sum_sq_diff = 0.0;
                for val in current_slice.iter() { sum_sq_diff += (*val - mean).powi(2); }
                let variance = sum_sq_diff / (last_dim_size as f32);
                let std_dev_inv = 1.0 / (variance + epsilon).sqrt();
                for (j, val) in current_slice.iter_mut().enumerate() {
                    *val = (*val - mean) * std_dev_inv * gamma.data[j] + beta.data[j];
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn matmul_2d_avx2_fma(
        a_data: &[f32], b_transposed_data: &[f32],
        m: usize, k_dim: usize, n_dim: usize,
        c_data: &mut [f32]
    ) {
        let mut packed_a_block = vec![0.0f32; MATMUL_BLOCK_M * MATMUL_BLOCK_K];
        let mut packed_b_block = vec![0.0f32; MATMUL_BLOCK_K * MATMUL_BLOCK_N];

        let a_ptr = a_data.as_ptr();
        let b_transposed_ptr = b_transposed_data.as_ptr();
        let c_ptr = c_data.as_mut_ptr();

        // Initialize C data to zero
        for val in c_data.iter_mut() { *val = 0.0; }


        for m_block_start in (0..m).step_by(MATMUL_BLOCK_M) {
            let current_block_m_a = (m - m_block_start).min(MATMUL_BLOCK_M);
            for k_block_start in (0..k_dim).step_by(MATMUL_BLOCK_K) {
                let current_block_k = (k_dim - k_block_start).min(MATMUL_BLOCK_K);

                pack_a_block(a_data, a_ptr, m_block_start, k_block_start, current_block_m_a, current_block_k, k_dim, &mut packed_a_block);

                for n_block_start in (0..n_dim).step_by(MATMUL_BLOCK_N) {
                    let current_block_n_b = (n_dim - n_block_start).min(MATMUL_BLOCK_N);

                    pack_b_transposed_block(b_transposed_data, b_transposed_ptr, k_block_start, n_block_start, current_block_k, current_block_n_b, k_dim, &mut packed_b_block);

                    // Micro-kernel application
                    let micro_m_unit = 8; // Process 8 rows of A with AVX
                    let micro_n_unit = 8; // Process 8 columns of B with AVX

                    for m_micro_offset in (0..current_block_m_a).step_by(micro_m_unit) {
                        let actual_micro_m = (current_block_m_a - m_micro_offset).min(micro_m_unit);
                        for n_micro_offset in (0..current_block_n_b).step_by(micro_n_unit) {
                            let actual_micro_n = (current_block_n_b - n_micro_offset).min(micro_n_unit);

                            if actual_micro_m == micro_m_unit && actual_micro_n == micro_n_unit { // Full 8x8 micro-kernel
                                matmul_micro_kernel_avx2(
                                    packed_a_block.as_ptr().add(m_micro_offset * current_block_k),
                                    packed_b_block.as_ptr().add(n_micro_offset * current_block_k), // packed_b is K_block x N_block (col-major like)
                                    c_ptr,
                                    m_block_start + m_micro_offset,
                                    n_block_start + n_micro_offset,
                                    current_block_k, n_dim,
                                    actual_micro_m, actual_micro_n
                                );
                            } else { // Scalar fallback for partial micro-blocks (peeling)
                                for m_i in 0..actual_micro_m {
                                    for n_j in 0..actual_micro_n {
                                        let mut acc = 0.0f32;
                                        for k_i in 0..current_block_k {
                                            // A is M_block x K_block (row-major)
                                            let a_val = packed_a_block[(m_micro_offset + m_i) * current_block_k + k_i];
                                            // B is K_block x N_block (col-major like for kernel)
                                            let b_val = packed_b_block[k_i * current_block_n_b + (n_micro_offset + n_j)];
                                            acc += a_val * b_val;
                                        }
                                        let c_val_ptr = c_ptr.add((m_block_start + m_micro_offset + m_i) * n_dim + (n_block_start + n_micro_offset + n_j));
                                        *c_val_ptr += acc;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn matmul(&self, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if self.rank() != 2 || b.rank() != 2 {
            return Err(TensorError::InvalidShape("Matmul only supports 2D tensors.".to_string()));
        }
        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = b.shape[0];
        let n = b.shape[1];
        if k1 != k2 {
            return Err(TensorError::DimensionMismatch(format!(
                "Matrix dimensions incompatible for multiplication: A({},{}) B({},{})", m, k1, k2, n
            )));
        }
        let mut c_data = vec![0.0f32; m * n];
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let b_transposed = b.transpose()?;
            unsafe { Self::matmul_2d_avx2_fma(&self.data, &b_transposed.data, m, k1, n, &mut c_data); }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l_idx in 0..k1 {
                        sum += self.data[i * k1 + l_idx] * b.data[l_idx * n + j];
                    }
                    c_data[i * n + j] = sum;
                }
            }
        }
        Tensor::new(c_data, vec![m, n])
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn softmax_slice_avx2(slice_data: &mut [f32]) {
        let n = slice_data.len();
        if n == 0 { return; }
        let mut max_val_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + AVX2_F32_COUNT <= n {
            let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
            max_val_vec = _mm256_max_ps(max_val_vec, data_vec);
            i += AVX2_F32_COUNT;
        }
        let mut max_val = horizontal_max_m256(max_val_vec);
        for j in i..n { if slice_data[j] > max_val { max_val = slice_data[j]; } }

        let max_val_vec_repeated = _mm256_set1_ps(max_val);
        let mut sum_exp_vec = _mm256_setzero_ps();
        i = 0;
        while i + AVX2_F32_COUNT <= n {
            let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
            let x = _mm256_sub_ps(data_vec, max_val_vec_repeated);
            let mut temp_exp = [0.0f32; AVX2_F32_COUNT];
            let mut temp_x = [0.0f32; AVX2_F32_COUNT];
            _mm256_storeu_ps(temp_x.as_mut_ptr(), x);
            for k_exp in 0..AVX2_F32_COUNT { temp_exp[k_exp] = libm::expf(temp_x[k_exp]); } // Fallback to scalar exp for now
            let exp_vec = _mm256_loadu_ps(temp_exp.as_ptr());
            sum_exp_vec = _mm256_add_ps(sum_exp_vec, exp_vec);
            _mm256_storeu_ps(slice_data.as_mut_ptr().add(i), exp_vec);
            i += AVX2_F32_COUNT;
        }
        let mut sum_exp = horizontal_sum_m256(sum_exp_vec);
        for j in i..n {
            let exp_val = libm::expf(slice_data[j] - max_val);
            slice_data[j] = exp_val;
            sum_exp += exp_val;
        }
        let sum_exp_inv = if sum_exp == 0.0 { 0.0 } else { 1.0 / sum_exp }; // Avoid division by zero
        let sum_exp_inv_vec = _mm256_set1_ps(sum_exp_inv);
        i = 0;
        while i + AVX2_F32_COUNT <= n {
            let exp_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
            let result_vec = _mm256_mul_ps(exp_vec, sum_exp_inv_vec);
            _mm256_storeu_ps(slice_data.as_mut_ptr().add(i), result_vec);
            i += AVX2_F32_COUNT;
        }
        for j in i..n { slice_data[j] *= sum_exp_inv; }
    }

    pub fn softmax(&mut self, axis: Option<usize>) -> Result<(), TensorError> {
        let default_axis = self.rank().saturating_sub(1); // Ensure axis is not negative for rank 0
        let axis_to_apply = axis.unwrap_or(default_axis);

        if self.rank() > 0 && axis_to_apply >= self.rank() {
            return Err(TensorError::InvalidShape("Axis out of bounds for softmax".to_string()));
        }
        if self.rank() == 0 { return Ok(()); }

        if axis_to_apply != self.rank() - 1 {
             return Err(TensorError::UnsupportedType("Softmax on non-last axis requires more complex data manipulation not yet implemented efficiently.".to_string()));
        }

        let last_dim_size = self.shape[axis_to_apply];
        if last_dim_size == 0 { return Ok(()); } // Softmax on empty dimension is a no-op
        let num_slices = self.data.len() / last_dim_size;

        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        for i in 0..num_slices {
            let slice_start = i * last_dim_size;
            let slice_end = (i + 1) * last_dim_size;
            let current_slice = &mut self.data[slice_start..slice_end];
            if use_avx2 {
                unsafe { Self::softmax_slice_avx2(current_slice); }
            } else {
                let max_val = current_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0;
                for val in current_slice.iter_mut() {
                    *val = libm::expf(*val - max_val);
                    sum_exp += *val;
                }
                let inv_sum_exp = if sum_exp == 0.0 { 0.0 } else { 1.0 / sum_exp };
                for val in current_slice.iter_mut() { *val *= inv_sum_exp; }
            }
        }
        Ok(())
    }

    pub fn scalar_mul(&mut self, scalar: f32) -> Result<(), TensorError> {
        for val in self.data.iter_mut() { *val *= scalar; }
        Ok(())
    }

    pub fn concat(&self, other: &Tensor<f32>, axis: usize) -> Result<Tensor<f32>, TensorError> {
        if self.rank() != other.rank() {
            return Err(TensorError::DimensionMismatch("Tensors must have the same rank to concatenate.".to_string()));
        }
        if axis >= self.rank() {
            return Err(TensorError::InvalidShape("Concatenation axis out of bounds.".to_string()));
        }
        for i in 0..self.rank() {
            if i != axis && self.shape[i] != other.shape[i] {
                return Err(TensorError::DimensionMismatch(
                    "Dimensions must match along non-concatenation axes.".to_string()
                ));
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] += other.shape[axis];
        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        let self_axis_len = self.shape[axis];
        let other_axis_len = other.shape[axis];
        let outer_dims_prod: usize = self.shape.iter().take(axis).product();
        let inner_dims_prod: usize = self.shape.iter().skip(axis + 1).product();
        for i in 0..outer_dims_prod {
            for j_self in 0..self_axis_len {
                for k in 0..inner_dims_prod {
                    let self_multi_dim_idx = construct_index(i, j_self, k, axis, &self.shape, inner_dims_prod);
                    new_data.push(self.get(&self_multi_dim_idx)?.clone());
                }
            }
            for j_other in 0..other_axis_len {
                for k in 0..inner_dims_prod {
                     let other_multi_dim_idx = construct_index(i, j_other, k, axis, &other.shape, inner_dims_prod);
                    new_data.push(other.get(&other_multi_dim_idx)?.clone());
                }
            }
        }
        Tensor::new(new_data, new_shape)
    }

    pub fn to_f16_tensor(&self) -> Result<Tensor<f16>, TensorError> {
        let f16_data: Vec<f16> = self.data.iter().map(|x| f16::from_f32(*x)).collect();
        Tensor::new(f16_data, self.shape.clone())
    }

    pub fn to_bf16_tensor(&self) -> Result<Tensor<bf16>, TensorError> {
        let bf16_data: Vec<bf16> = self.data.iter().map(|x| bf16::from_f32(*x)).collect();
        Tensor::new(bf16_data, self.shape.clone())
    }
}

// Helper function for concat indexing
fn construct_index(
    outer_loop_idx: usize, axis_loop_idx: usize, inner_loop_idx: usize,
    axis: usize, original_shape: &[usize], _inner_dims_prod: usize, // _inner_dims_prod currently unused
) -> Vec<usize> {
    let rank = original_shape.len();
    let mut current_multi_dim_idx = vec![0; rank];
    let mut temp_outer = outer_loop_idx;
    for d in (0..axis).rev() {
        current_multi_dim_idx[d] = temp_outer % original_shape[d];
        temp_outer /= original_shape[d];
    }
    current_multi_dim_idx[axis] = axis_loop_idx;
    let mut temp_inner = inner_loop_idx;
    for d in ((axis + 1)..rank).rev() { // Corrected loop range
        if original_shape[d] == 0 { // Avoid division by zero for empty dimensions
            current_multi_dim_idx[d] = 0;
        } else {
            current_multi_dim_idx[d] = temp_inner % original_shape[d];
            temp_inner /= original_shape[d];
        }
    }
    if rank > axis + 1 && original_shape[axis + 1] != 0 { // Handle the first inner dimension correctly
         current_multi_dim_idx[axis+1] = temp_inner % original_shape[axis+1]; // Remainder for the first inner dim
    } else if rank > axis + 1 && original_shape[axis+1] == 0 {
         current_multi_dim_idx[axis+1] = 0;
    }


    current_multi_dim_idx
}

// --- f16 Specific Operations ---
impl Tensor<f16> {
    pub fn zeros(shape: Vec<usize>) -> Tensor<f16> {
        let num_elements = shape.iter().product();
        let data = vec![f16::from_f32(0.0); num_elements];
        Tensor { data, shape }
    }

    pub fn scalar_mul(&mut self, scalar: f16) -> Result<(), TensorError> {
        for val in self.data.iter_mut() { *val = *val * scalar; }
        Ok(())
    }

    pub fn concat(&self, other: &Tensor<f16>, axis: usize) -> Result<Tensor<f16>, TensorError> {
        if self.rank() != other.rank() {
            return Err(TensorError::DimensionMismatch("Tensors must have the same rank to concatenate.".to_string()));
        }
        if axis >= self.rank() {
            return Err(TensorError::InvalidShape("Concatenation axis out of bounds.".to_string()));
        }
        for i in 0..self.rank() {
            if i != axis && self.shape[i] != other.shape[i] {
                return Err(TensorError::DimensionMismatch(
                    "Dimensions must match along non-concatenation axes.".to_string()
                ));
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] += other.shape[axis];
        let new_num_elements: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_num_elements);
        let self_axis_len = self.shape[axis];
        let other_axis_len = other.shape[axis];
        let outer_dims_prod: usize = self.shape.iter().take(axis).product();
        let inner_dims_prod: usize = self.shape.iter().skip(axis + 1).product();
        for i in 0..outer_dims_prod {
            for j_self in 0..self_axis_len {
                for k in 0..inner_dims_prod {
                    let self_multi_dim_idx = construct_index(i, j_self, k, axis, &self.shape, inner_dims_prod);
                    new_data.push(self.get(&self_multi_dim_idx)?.clone());
                }
            }
            for j_other in 0..other_axis_len {
                for k in 0..inner_dims_prod {
                     let other_multi_dim_idx = construct_index(i, j_other, k, axis, &other.shape, inner_dims_prod);
                    new_data.push(other.get(&other_multi_dim_idx)?.clone());
                }
            }
        }
        Tensor::new(new_data, new_shape)
    }

    pub fn to_f32_tensor(&self) -> Result<Tensor<f32>, TensorError> {
        let f32_data: Vec<f32> = self.data.iter().map(|x| x.to_f32()).collect();
        Tensor::new(f32_data, self.shape.clone())
    }

    pub fn gelu(&mut self) -> Result<(), TensorError> {
        let mut f32_tensor = self.to_f32_tensor()?;
        f32_tensor.gelu()?;
        if self.data.len() != f32_tensor.data.len() {
            return Err(TensorError::GenericError("Data length mismatch during f16 gelu conversion".to_string()));
        }
        for (original_val, f32_val) in self.data.iter_mut().zip(f32_tensor.data.iter()) {
            *original_val = f16::from_f32(*f32_val);
        }
        Ok(())
    }

    pub fn layernorm(&mut self, gamma: &Tensor<f16>, beta: &Tensor<f16>, epsilon: f32) -> Result<(), TensorError> {
        let last_dim_size = *self.shape.last().ok_or_else(|| TensorError::InvalidShape("Tensor must have at least one dimension for layernorm".to_string()))?;
        if gamma.shape != [last_dim_size] || beta.shape != [last_dim_size] {
            return Err(TensorError::DimensionMismatch(
                "Gamma/Beta dimensions must match last dimension of tensor for f16 layernorm".to_string(),
            ));
        }
        let mut f32_self = self.to_f32_tensor()?;
        let f32_gamma = gamma.to_f32_tensor()?;
        let f32_beta = beta.to_f32_tensor()?;
        f32_self.layernorm(&f32_gamma, &f32_beta, epsilon)?;
        if self.data.len() != f32_self.data.len() {
            return Err(TensorError::GenericError("Data length mismatch during f16 layernorm conversion".to_string()));
        }
        for (original_val, f32_val) in self.data.iter_mut().zip(f32_self.data.iter()) {
            *original_val = f16::from_f32(*f32_val);
        }
        Ok(())
    }

    pub fn matmul(&self, other: &Tensor<f16>) -> Result<Tensor<f16>, TensorError> {
        let f32_self = self.to_f32_tensor()?;
        let f32_other = other.to_f32_tensor()?;

        let f32_result = f32_self.matmul(&f32_other)?; // Calls f32 SIMD version

        f32_result.to_f16_tensor()
    }

    pub fn softmax(&mut self, axis: Option<usize>) -> Result<(), TensorError> {
        let mut f32_tensor = self.to_f32_tensor()?;
        f32_tensor.softmax(axis)?; // Calls f32 SIMD version

        if self.data.len() != f32_tensor.data.len() {
            return Err(TensorError::GenericError("Data length mismatch during f16 softmax conversion".to_string()));
        }
        for (original_val, f32_val) in self.data.iter_mut().zip(f32_tensor.data.iter()) {
            *original_val = f16::from_f32(*f32_val);
        }
        Ok(())
    }
}

// --- bf16 Specific Operations ---
impl Tensor<bf16> {
    pub fn zeros(shape: Vec<usize>) -> Tensor<bf16> {
        let num_elements = shape.iter().product();
        let data = vec![bf16::from_f32(0.0); num_elements];
        Tensor { data, shape }
    }

    pub fn scalar_mul(&mut self, scalar: bf16) -> Result<(), TensorError> {
        for val in self.data.iter_mut() { *val = *val * scalar; }
        Ok(())
    }

    pub fn concat(&self, other: &Tensor<bf16>, axis: usize) -> Result<Tensor<bf16>, TensorError> {
        if self.rank() != other.rank() {
            return Err(TensorError::DimensionMismatch("Tensors must have the same rank to concatenate.".to_string()));
        }
        if axis >= self.rank() {
            return Err(TensorError::InvalidShape("Concatenation axis out of bounds.".to_string()));
        }
        for i in 0..self.rank() {
            if i != axis && self.shape[i] != other.shape[i] {
                return Err(TensorError::DimensionMismatch(
                    "Dimensions must match along non-concatenation axes.".to_string()
                ));
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] += other.shape[axis];
        let new_num_elements: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_num_elements);
        let self_axis_len = self.shape[axis];
        let other_axis_len = other.shape[axis];
        let outer_dims_prod: usize = self.shape.iter().take(axis).product();
        let inner_dims_prod: usize = self.shape.iter().skip(axis + 1).product();
        for i in 0..outer_dims_prod {
            for j_self in 0..self_axis_len {
                for k in 0..inner_dims_prod {
                    let self_multi_dim_idx = construct_index(i, j_self, k, axis, &self.shape, inner_dims_prod);
                    new_data.push(self.get(&self_multi_dim_idx)?.clone());
                }
            }
            for j_other in 0..other_axis_len {
                for k in 0..inner_dims_prod {
                     let other_multi_dim_idx = construct_index(i, j_other, k, axis, &other.shape, inner_dims_prod);
                    new_data.push(other.get(&other_multi_dim_idx)?.clone());
                }
            }
        }
        Tensor::new(new_data, new_shape)
    }

    pub fn to_f32_tensor(&self) -> Result<Tensor<f32>, TensorError> {
        let f32_data: Vec<f32> = self.data.iter().map(|x| x.to_f32()).collect();
        Tensor::new(f32_data, self.shape.clone())
    }

    pub fn gelu(&mut self) -> Result<(), TensorError> {
        let mut f32_tensor = self.to_f32_tensor()?;
        f32_tensor.gelu()?;
        if self.data.len() != f32_tensor.data.len() {
            return Err(TensorError::GenericError("Data length mismatch during bf16 gelu conversion".to_string()));
        }
        for (original_val, f32_val) in self.data.iter_mut().zip(f32_tensor.data.iter()) {
            *original_val = bf16::from_f32(*f32_val);
        }
        Ok(())
    }

    pub fn layernorm(&mut self, gamma: &Tensor<bf16>, beta: &Tensor<bf16>, epsilon: f32) -> Result<(), TensorError> {
        let last_dim_size = *self.shape.last().ok_or_else(|| TensorError::InvalidShape("Tensor must have at least one dimension for layernorm".to_string()))?;
        if gamma.shape != [last_dim_size] || beta.shape != [last_dim_size] {
            return Err(TensorError::DimensionMismatch(
                "Gamma/Beta dimensions must match last dimension of tensor for bf16 layernorm".to_string(),
            ));
        }
        let mut f32_self = self.to_f32_tensor()?;
        let f32_gamma = gamma.to_f32_tensor()?;
        let f32_beta = beta.to_f32_tensor()?;
        f32_self.layernorm(&f32_gamma, &f32_beta, epsilon)?;
        if self.data.len() != f32_self.data.len() {
            return Err(TensorError::GenericError("Data length mismatch during bf16 layernorm conversion".to_string()));
        }
        for (original_val, f32_val) in self.data.iter_mut().zip(f32_self.data.iter()) {
            *original_val = bf16::from_f32(*f32_val);
        }
        Ok(())
    }

    pub fn matmul(&self, other: &Tensor<bf16>) -> Result<Tensor<bf16>, TensorError> {
        let f32_self = self.to_f32_tensor()?;
        let f32_other = other.to_f32_tensor()?;

        let f32_result = f32_self.matmul(&f32_other)?; // Calls f32 SIMD version

        f32_result.to_bf16_tensor()
    }

    pub fn softmax(&mut self, axis: Option<usize>) -> Result<(), TensorError> {
        let mut f32_tensor = self.to_f32_tensor()?;
        f32_tensor.softmax(axis)?; // Calls f32 SIMD version

        if self.data.len() != f32_tensor.data.len() {
            return Err(TensorError::GenericError("Data length mismatch during bf16 softmax conversion".to_string()));
        }
        for (original_val, f32_val) in self.data.iter_mut().zip(f32_tensor.data.iter()) {
            *original_val = bf16::from_f32(*f32_val);
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

    fn assert_f32_vec_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_relative_eq!(*x, *y, epsilon = tol, max_relative = tol,
                message = "Element at index {} differs: {} vs {}", i, *x, *y);
        }
    }

    #[test]
    fn test_tensor_new_get_reshape() -> Result<(), TensorError> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data.clone(), shape.clone())?;
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(*tensor.get(&[0, 1])?, 2.0);
        assert_eq!(*tensor.get(&[1, 2])?, 6.0);
        let reshaped_tensor = tensor.reshape(vec![3, 2])?;
        assert_eq!(reshaped_tensor.shape, vec![3, 2]);
        assert_eq!(*reshaped_tensor.get(&[0, 1])?, 2.0);
        assert_eq!(*reshaped_tensor.get(&[2, 1])?, 6.0);
        let scalar_tensor = Tensor::new(vec![42.0], vec![])?;
        assert_eq!(scalar_tensor.rank(), 0);
        assert_eq!(scalar_tensor.num_elements(), 1);
        assert_eq!(*scalar_tensor.get(&[])?, 42.0);
        assert_eq!(*scalar_tensor.get(&[0])?, 42.0);
        Ok(())
    }

    #[test]
    fn test_transpose_2d() -> Result<(), TensorError> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let transposed = tensor.transpose()?;
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_f16() -> Result<(), TensorError> {
        let a_f16 = Tensor::new(vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)], vec![2, 2])?;
        let b_f16 = Tensor::new(vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0), f16::from_f32(8.0)], vec![2, 2])?;
        // Expected f32: [[19.0, 22.0], [43.0, 50.0]]
        let expected_f32_data = vec![19.0, 22.0, 43.0, 50.0];

        let c_f16 = a_f16.matmul(&b_f16)?;
        assert_eq!(c_f16.shape, vec![2, 2]);

        for (val_f16, expected_f32) in c_f16.data.iter().zip(expected_f32_data.iter()) {
            assert_relative_eq!(val_f16.to_f32(), *expected_f32, epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }

    #[test]
    fn test_matmul_bf16() -> Result<(), TensorError> {
        let a_bf16 = Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0)], vec![2, 2])?;
        let b_bf16 = Tensor::new(vec![bf16::from_f32(5.0), bf16::from_f32(6.0), bf16::from_f32(7.0), bf16::from_f32(8.0)], vec![2, 2])?;
        let expected_f32_data = vec![19.0, 22.0, 43.0, 50.0];

        let c_bf16 = a_bf16.matmul(&b_bf16)?;
        assert_eq!(c_bf16.shape, vec![2, 2]);

        for (val_bf16, expected_f32) in c_bf16.data.iter().zip(expected_f32_data.iter()) {
            assert_relative_eq!(val_bf16.to_f32(), *expected_f32, epsilon = 1e-1, max_relative = 1e-1); // bf16 has less precision
        }
        Ok(())
    }

    #[test]
    fn test_gelu_f32() -> Result<(), TensorError> {
        let mut tensor = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5])?;
        let expected_values_scalar = tensor.data.iter().map(|x| 0.5 * x * (1.0 + libm::tanhf(std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))))).collect::<Vec<f32>>();
        tensor.gelu()?;
        assert_f32_vec_eq(&tensor.data, &expected_values_scalar, 1e-4);
        Ok(())
    }

    #[test]
    fn test_tanhf_approx_avx2_consistency() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not detected, skipping test_tanhf_approx_avx2_consistency");
            return;
        }
        let inputs = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &x_val in &inputs {
            let mut data = [x_val; AVX2_F32_COUNT];
            unsafe {
                let v = _mm256_loadu_ps(data.as_ptr());
                let result_avx2_vec = tanhf_approx_avx2(v);
                 _mm256_storeu_ps(data.as_mut_ptr(), result_avx2_vec);
            }
            assert_relative_eq!(data[0], libm::tanhf(x_val), epsilon = 0.01, max_relative = 0.01,
                                message="tanh approx failed for x={}", x_val);
        }
    }

    #[test]
    fn test_layernorm_f32() -> Result<(), TensorError> {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0], vec![3])?;
        let beta = Tensor::new(vec![0.0, 0.0, 0.0], vec![3])?;
        let epsilon = 1e-5;
        let mut expected_data = tensor.data.clone();
        for i in 0..2 {
            let slice_start = i * 3;
            let slice_end = (i + 1) * 3;
            let current_slice = &mut expected_data[slice_start..slice_end];
            let mut sum = 0.0;
            for val in current_slice.iter() { sum += *val; }
            let mean = sum / 3.0;
            let mut sum_sq_diff = 0.0;
            for val in current_slice.iter() { sum_sq_diff += (*val - mean).powi(2); }
            let variance = sum_sq_diff / 3.0;
            let std_dev_inv = 1.0 / (variance + epsilon).sqrt();
            for (j, val) in current_slice.iter_mut().enumerate() {
                *val = (*val - mean) * std_dev_inv * gamma.data[j] + beta.data[j];
            }
        }
        tensor.layernorm(&gamma, &beta, epsilon)?;
        assert_f32_vec_eq(&tensor.data, &expected_data, 1e-5);
        Ok(())
    }

    #[test]
    fn test_matmul_f32_simple() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        let expected_c = Tensor::new(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape, expected_c.shape);
        assert_f32_vec_eq(&c.data, &expected_c.data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_matmul_f32_rect() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?;
        let expected_c = Tensor::new(vec![58.0, 64.0, 139.0, 154.0], vec![2, 2])?;
        let c = a.matmul(&b)?;
        assert_eq!(c.shape, expected_c.shape);
        assert_f32_vec_eq(&c.data, &expected_c.data, 1e-6);
        Ok(())
    }

    #[test]
    fn test_softmax_f32_last_axis() -> Result<(), TensorError> {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3])?;
        let expected_data = vec![
            0.09003057, 0.24472847, 0.66524096,
            0.33333333, 0.33333333, 0.33333333,
        ];
        tensor.softmax(None)?;
        assert_f32_vec_eq(&tensor.data, &expected_data, 1e-5);
        Ok(())
    }

    #[test]
    fn test_scalar_mul_f32() -> Result<(), TensorError> {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        tensor.scalar_mul(2.5)?;
        assert_f32_vec_eq(&tensor.data, &[2.5, 5.0, 7.5], 1e-6);
        Ok(())
    }

    #[test]
    fn test_concat_f32_axis0() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0], vec![1,2])?;
        let b = Tensor::new(vec![3.0, 4.0], vec![1,2])?;
        let c = a.concat(&b, 0)?;
        assert_eq!(c.shape, vec![2,2]);
        assert_f32_vec_eq(&c.data, &[1.0, 2.0, 3.0, 4.0], 1e-6);
        Ok(())
    }

    #[test]
    fn test_concat_f32_axis1() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 3.0], vec![2,1])?;
        let b = Tensor::new(vec![2.0, 4.0], vec![2,1])?;
        let c = a.concat(&b, 1)?;
        assert_eq!(c.shape, vec![2,2]);
        assert_f32_vec_eq(&c.data, &[1.0, 2.0, 3.0, 4.0], 1e-6);
        Ok(())
    }

    #[test]
    fn test_zeros_f16() {
        let tensor_f16 = Tensor::<f16>::zeros(vec![2,2]);
        assert_eq!(tensor_f16.shape, vec![2,2]);
        assert_eq!(tensor_f16.data.len(), 4);
        for val in tensor_f16.data { assert_eq!(val, f16::from_f32(0.0)); }
    }

    #[test]
    fn test_zeros_bf16() {
        let tensor_bf16 = Tensor::<bf16>::zeros(vec![2,2]);
        assert_eq!(tensor_bf16.shape, vec![2,2]);
        assert_eq!(tensor_bf16.data.len(), 4);
        for val in tensor_bf16.data { assert_eq!(val, bf16::from_f32(0.0)); }
    }

    #[test]
    fn test_scalar_mul_f16() -> Result<(), TensorError> {
        let mut tensor = Tensor::new(vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)], vec![3])?;
        tensor.scalar_mul(f16::from_f32(2.5))?;
        assert_eq!(tensor.data[0], f16::from_f32(2.5));
        assert_eq!(tensor.data[1], f16::from_f32(5.0));
        assert_eq!(tensor.data[2], f16::from_f32(7.5));
        Ok(())
    }

    #[test]
    fn test_scalar_mul_bf16() -> Result<(), TensorError> {
        let mut tensor = Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)], vec![3])?;
        tensor.scalar_mul(bf16::from_f32(2.5))?;
        assert!((tensor.data[0].to_f32() - 2.5).abs() < bf16::EPSILON.to_f32() * 5.0);
        assert!((tensor.data[1].to_f32() - 5.0).abs() < bf16::EPSILON.to_f32() * 5.0);
        assert!((tensor.data[2].to_f32() - 7.5).abs() < bf16::EPSILON.to_f32() * 5.0);
        Ok(())
    }

    #[test]
    fn test_concat_f16_axis0() -> Result<(), TensorError> {
        let a = Tensor::new(vec![f16::from_f32(1.0), f16::from_f32(2.0)], vec![1,2])?;
        let b = Tensor::new(vec![f16::from_f32(3.0), f16::from_f32(4.0)], vec![1,2])?;
        let c = a.concat(&b, 0)?;
        assert_eq!(c.shape, vec![2,2]);
        let expected_data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        for (actual, expected) in c.data.iter().zip(expected_data.iter()) { assert_eq!(actual, expected); }
        Ok(())
    }

    #[test]
    fn test_concat_f16_axis1() -> Result<(), TensorError> {
        let a = Tensor::new(vec![f16::from_f32(1.0), f16::from_f32(3.0)], vec![2,1])?;
        let b = Tensor::new(vec![f16::from_f32(2.0), f16::from_f32(4.0)], vec![2,1])?;
        let c = a.concat(&b, 1)?;
        assert_eq!(c.shape, vec![2,2]);
        let expected_data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
         for (actual, expected) in c.data.iter().zip(expected_data.iter()) { assert_eq!(actual, expected); }
        Ok(())
    }

    #[test]
    fn test_concat_bf16_axis0() -> Result<(), TensorError> {
        let a = Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)], vec![1,2])?;
        let b = Tensor::new(vec![bf16::from_f32(3.0), bf16::from_f32(4.0)], vec![1,2])?;
        let c = a.concat(&b, 0)?;
        assert_eq!(c.shape, vec![2,2]);
        let expected_f32_data = vec![1.0, 2.0, 3.0, 4.0];
        for (actual_bf16, expected_f32) in c.data.iter().zip(expected_f32_data.iter()) {
             assert!((actual_bf16.to_f32() - expected_f32).abs() < bf16::EPSILON.to_f32() * 5.0);
        }
        Ok(())
    }

    #[test]
    fn test_gelu_f16() -> Result<(), TensorError> {
        let mut tensor_f16 = Tensor::new(
            vec![f16::from_f32(-3.0), f16::from_f32(-1.0), f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(3.0)],
            vec![5]
        )?;
        let expected_f32_values: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0].iter()
            .map(|x| 0.5 * x * (1.0 + libm::tanhf(std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3)))))
            .collect();
        tensor_f16.gelu()?;
        for (i, val_f16) in tensor_f16.data.iter().enumerate() {
            let val_f32 = val_f16.to_f32();
            assert_relative_eq!(val_f32, expected_f32_values[i], epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }

    #[test]
    fn test_gelu_bf16() -> Result<(), TensorError> {
        let mut tensor_bf16 = Tensor::new(
            vec![bf16::from_f32(-3.0), bf16::from_f32(-1.0), bf16::from_f32(0.0), bf16::from_f32(1.0), bf16::from_f32(3.0)],
            vec![5]
        )?;
        let expected_f32_values: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0].iter()
            .map(|x| 0.5 * x * (1.0 + libm::tanhf(std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3)))))
            .collect();
        tensor_bf16.gelu()?;
        for (i, val_bf16) in tensor_bf16.data.iter().enumerate() {
            let val_f32 = val_bf16.to_f32();
            assert_relative_eq!(val_f32, expected_f32_values[i], epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }

    #[test]
    fn test_layernorm_f16() -> Result<(), TensorError> {
        let mut tensor_f16 = Tensor::new(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
            vec![2, 3]
        )?;
        let gamma_f16 = Tensor::new(vec![f16::from_f32(1.0), f16::from_f32(1.5), f16::from_f32(0.5)], vec![3])?;
        let beta_f16 = Tensor::new(vec![f16::from_f32(0.0), f16::from_f32(0.1), f16::from_f32(-0.1)], vec![3])?;
        let epsilon = 1e-5;
        let mut expected_f32_tensor = tensor_f16.to_f32_tensor()?;
        let expected_gamma_f32 = gamma_f16.to_f32_tensor()?;
        let expected_beta_f32 = beta_f16.to_f32_tensor()?;
        expected_f32_tensor.layernorm(&expected_gamma_f32, &expected_beta_f32, epsilon)?;
        tensor_f16.layernorm(&gamma_f16, &beta_f16, epsilon)?;
        for (val_f16, expected_f32) in tensor_f16.data.iter().zip(expected_f32_tensor.data.iter()) {
            assert_relative_eq!(val_f16.to_f32(), *expected_f32, epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }

    #[test]
    fn test_layernorm_bf16() -> Result<(), TensorError> {
        let mut tensor_bf16 = Tensor::new(
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0), bf16::from_f32(5.0), bf16::from_f32(6.0)],
            vec![2, 3]
        )?;
        let gamma_bf16 = Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(1.5), bf16::from_f32(0.5)], vec![3])?;
        let beta_bf16 = Tensor::new(vec![bf16::from_f32(0.0), bf16::from_f32(0.1), bf16::from_f32(-0.1)], vec![3])?;
        let epsilon = 1e-5;
        let mut expected_f32_tensor = tensor_bf16.to_f32_tensor()?;
        let expected_gamma_f32 = gamma_bf16.to_f32_tensor()?;
        let expected_beta_f32 = beta_bf16.to_f32_tensor()?;
        expected_f32_tensor.layernorm(&expected_gamma_f32, &expected_beta_f32, epsilon)?;
        tensor_bf16.layernorm(&gamma_bf16, &beta_bf16, epsilon)?;
        for (val_bf16, expected_f32) in tensor_bf16.data.iter().zip(expected_f32_tensor.data.iter()) {
            assert_relative_eq!(val_bf16.to_f32(), *expected_f32, epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_f16() -> Result<(), TensorError> {
        let mut tensor_f16 = Tensor::new(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(1.0), f16::from_f32(1.0), f16::from_f32(1.0)],
            vec![2, 3]
        )?;
        // Expected f32 data (from test_softmax_f32_last_axis)
        let expected_f32_data = vec![
            0.09003057, 0.24472847, 0.66524096,
            0.33333333, 0.33333333, 0.33333333,
        ];

        tensor_f16.softmax(None)?; // Axis = last axis

        for (val_f16, expected_f32) in tensor_f16.data.iter().zip(expected_f32_data.iter()) {
            assert_relative_eq!(val_f16.to_f32(), *expected_f32, epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_bf16() -> Result<(), TensorError> {
         let mut tensor_bf16 = Tensor::new(
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(1.0), bf16::from_f32(1.0), bf16::from_f32(1.0)],
            vec![2, 3]
        )?;
        let expected_f32_data = vec![
            0.09003057, 0.24472847, 0.66524096,
            0.33333333, 0.33333333, 0.33333333,
        ];

        tensor_bf16.softmax(None)?;

        for (val_bf16, expected_f32) in tensor_bf16.data.iter().zip(expected_f32_data.iter()) {
            assert_relative_eq!(val_bf16.to_f32(), *expected_f32, epsilon = 1e-2, max_relative = 1e-2);
        }
        Ok(())
    }


     #[test]
    fn test_matmul_avx2_fma_consistency() -> Result<(), TensorError> {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("AVX2/FMA not detected, skipping test_matmul_avx2_fma_consistency");
            return Ok(());
        }
        let mut rng = rand::thread_rng();
        let m = MATMUL_BLOCK_M;
        let k = MATMUL_BLOCK_K;
        let n = MATMUL_BLOCK_N;
        let a_data: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let tensor_a = Tensor::new(a_data.clone(), vec![m,k])?;
        let tensor_b = Tensor::new(b_data.clone(), vec![k,n])?;
        let mut c_scalar_data = vec![0.0f32; m * n];
        for i in 0..m {
            for j_c in 0..n { // Renamed j to j_c to avoid conflict
                let mut sum = 0.0;
                for l_idx in 0..k {
                    sum += tensor_a.data[i * k + l_idx] * tensor_b.data[l_idx * n + j_c];
                }
                c_scalar_data[i * n + j_c] = sum;
            }
        }
        let mut c_simd_data = vec![0.0f32; m * n];
        let b_transposed = tensor_b.transpose()?;
        unsafe {
             Tensor::<f32>::matmul_2d_avx2_fma(&tensor_a.data, &b_transposed.data, m, k, n, &mut c_simd_data);
        }
        assert_f32_vec_eq(&c_simd_data, &c_scalar_data, 1e-3);
        Ok(())
    }
}
