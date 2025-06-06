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
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Helper function for approximate tanh using AVX2
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn tanhf_approx_avx2(x: __m256) -> __m256 {
    let clamp_val = 3.0; // Clamping range for better approximation accuracy
    let c27_vec = _mm256_set1_ps(27.0);
    let c9_vec = _mm256_set1_ps(9.0);
    let clamp_val_vec = _mm256_set1_ps(clamp_val);
    let neg_clamp_val_vec = _mm256_set1_ps(-clamp_val);

    // Clamp input x to [-clamp_val, clamp_val]
    let x_clamped = _mm256_max_ps(x, neg_clamp_val_vec);
    let x_clamped = _mm256_min_ps(x_clamped, clamp_val_vec);

    // Polynomial approximation: x * (27 + x^2) / (27 + 9 * x^2)
    let x_sq = _mm256_mul_ps(x_clamped, x_clamped);
    let num = _mm256_mul_ps(x_clamped, _mm256_add_ps(c27_vec, x_sq));
    let den = _mm256_add_ps(c27_vec, _mm256_mul_ps(c9_vec, x_sq));

    // Handle cases where den is zero to avoid division by zero (e.g., return sign(x) or clamp result)
    // For this approximation, at x=0, num=0, den=27, result is 0.
    // If x is such that 27 + 9*x^2 = 0, this would be an issue, but 9x^2 is always >=0, so den is always >= 27.
    let tanh_approx = _mm256_div_ps(num, den);
    tanh_approx
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_m256(vec: __m256) -> f32 {
    // Sum upper and lower 128-bit lanes
    let sum_halves = _mm_add_ps(_mm256_castps256_ps128(vec), _mm256_extractf128_ps(vec, 1));
    // Horizontal sum of the lower 128-bit lane (which now contains sums of corresponding elements)
    let hsum_ps = _mm_hadd_ps(sum_halves, sum_halves); // hsum_ps[0] = sum_halves[0] + sum_halves[1], hsum_ps[1] = sum_halves[2] + sum_halves[3]
                                                      // hsum_ps[2] = sum_halves[0] + sum_halves[1], hsum_ps[3] = sum_halves[2] + sum_halves[3] (due to hadd behavior)
    // Second hadd to sum the results further
    let hsum_ps2 = _mm_hadd_ps(hsum_ps, hsum_ps);      // hsum_ps2[0] = hsum_ps[0] + hsum_ps[1] which is (v[0]+v[4]+v[1]+v[5]) + (v[2]+v[6]+v[3]+v[7])
                                                      // This is effectively sum of all 8 original floats.
    _mm_cvtss_f32(hsum_ps2) // Extract the first element
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_max_m256(vec: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(vec);          // Lower 128 bits
    let vhigh = _mm256_extractf128_ps(vec, 1);     // Upper 128 bits
    let vmax128 = _mm_max_ps(vlow, vhigh);          // Max of lower and upper

    // Reduce __m128 horizontally
    let vshuf1 = _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 3, 2)); // [v3, v2, v3, v2]
    let vmax_intermediate1 = _mm_max_ps(vmax128, vshuf1);                   // [max(v0,v3), max(v1,v2), max(v2,v3), max(v3,v2)]

    let vshuf2 = _mm_shuffle_ps(vmax_intermediate1, vmax_intermediate1, _MM_SHUFFLE(0, 0, 0, 1)); // [max(v1,v2), _, _, _] (effectively)
    let vmax_final = _mm_max_ps(vmax_intermediate1, vshuf2);               // Contains the max in the first element

    _mm_cvtss_f32(vmax_final)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn expf_approx_taylor_avx2(x: __m256) -> __m256 {
    // Taylor series approximation for exp(x) around 0:
    // exp(x) approx 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5!
    // This is equivalent to: 1 + x(1 + x/2(1 + x/3(1 + x/4(1 + x/5))))
    // Clamping input x to a negative range (e.g. <=0) is important for stability of this poly.
    // Max subtraction in softmax should ensure this. For general exp, other methods are needed.

    let c1 = _mm256_set1_ps(1.0);
    // Coefficients for x/2, x/3, x/4, x/5 (actually 1/2, 1/3, 1/4, 1/5)
    let c_div2 = _mm256_set1_ps(1.0 / 2.0);
    let c_div3 = _mm256_set1_ps(1.0 / 3.0);
    let c_div4 = _mm256_set1_ps(1.0 / 4.0);
    let c_div5 = _mm256_set1_ps(1.0 / 5.0);

    // Term: (1 + x/5)
    let mut res = _mm256_fmadd_ps(x, c_div5, c1);
    // Term: (1 + x/4 * prev)
    res = _mm256_fmadd_ps(_mm256_mul_ps(x, c_div4), res, c1);
    // Term: (1 + x/3 * prev)
    res = _mm256_fmadd_ps(_mm256_mul_ps(x, c_div3), res, c1);
    // Term: (1 + x/2 * prev)
    res = _mm256_fmadd_ps(_mm256_mul_ps(x, c_div2), res, c1);
    // Term: 1 + x * prev
    res = _mm256_fmadd_ps(x, res, c1);

    // Clamp results to avoid excessively large values if x was not negative enough
    // For softmax, x should be <= 0, so exp(x) should be <= 1.
    // However, approximation errors might occur. A max_exp_val can be used.
    // let max_exp_val = _mm256_set1_ps(3.4028235e38_f32); // FLT_MAX basically
    // res = _mm256_min_ps(res, max_exp_val);
    // Ensure non-negative (exp(x) is always positive)
    res = _mm256_max_ps(res, _mm256_setzero_ps());
    res
}


#[cfg(target_arch = "x86_64")]
unsafe fn softmax_slice_avx2(slice_data: &mut [f32]) {
    let lanes = 8;
    let len = slice_data.len();
    if len == 0 { return; }
    let mut i = 0;

    // 1. Max Value Calculation
    let mut max_val_simd_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        max_val_simd_vec = _mm256_max_ps(max_val_simd_vec, data_vec);
        i += lanes;
    }
    let mut max_val = horizontal_max_m256(max_val_simd_vec);
    // Scalar remainder for max value
    for k in i..len {
        if slice_data[k] > max_val {
            max_val = slice_data[k];
        }
    }
    let max_val_bcast_vec = _mm256_set1_ps(max_val);

    // 2. Subtract Max, Exp, and Sum Exp Values
    // A temporary buffer is needed because we iterate twice: once for exp sum, once for division.
    // Alternatively, one could do it in one pass if exp values are stored.
    let mut exp_values_temp: Vec<f32> = vec![0.0f32; len]; // Consider pre-allocating outside if called often

    i = 0;
    let mut sum_exp_vec_acc = _mm256_setzero_ps();
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        let norm_vec = _mm256_sub_ps(data_vec, max_val_bcast_vec); // x - max_val
        let exp_vec = expf_approx_taylor_avx2(norm_vec);
        _mm256_storeu_ps(exp_values_temp.as_mut_ptr().add(i), exp_vec);
        sum_exp_vec_acc = _mm256_add_ps(sum_exp_vec_acc, exp_vec);
        i += lanes;
    }
    let mut total_sum_exp = horizontal_sum_m256(sum_exp_vec_acc);
    // Scalar remainder for exp sum
    for k in i..len {
        let val = *slice_data.get_unchecked(k);
        let norm_val = val - max_val;
        // Using precise expf for scalar part for better accuracy, taylor approx might be less accurate here
        let exp_val = libm::expf(norm_val); // Or use scalar version of taylor approx if consistency is key over precision
        *exp_values_temp.get_unchecked_mut(k) = exp_val;
        total_sum_exp += exp_val;
    }

    // Handle sum_exp being zero or very small to prevent division by zero or NaNs/Infs
    if total_sum_exp == 0.0 { total_sum_exp = 1e-9; } // Avoid division by zero, distribute uniformly (almost)

    // 3. Divide by Sum
    let inv_total_sum_exp = total_sum_exp.recip(); // 1.0 / total_sum_exp
    let inv_total_sum_exp_vec = _mm256_set1_ps(inv_total_sum_exp);

    i = 0;
    while i + lanes <= len {
        let exp_vec_loaded = _mm256_loadu_ps(exp_values_temp.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(exp_vec_loaded, inv_total_sum_exp_vec);
        _mm256_storeu_ps(slice_data.as_mut_ptr().add(i), result_vec); // Store back into original slice
        i += lanes;
    }
    // Scalar remainder for division
    for k in i..len {
        *slice_data.get_unchecked_mut(k) = *exp_values_temp.get_unchecked(k) * inv_total_sum_exp;
    }
}


#[cfg(target_arch = "x86_64")]
unsafe fn layernorm_slice_avx2(slice_data: &mut [f32], gamma_data: &[f32], beta_data: &[f32], epsilon: f32) {
    let lanes = 8;
    let len = slice_data.len();
    let mut i = 0;

    // Mean Calculation
    let mut sum_vec = _mm256_setzero_ps();
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        sum_vec = _mm256_add_ps(sum_vec, data_vec);
        i += lanes;
    }
    let mut total_sum_simd = horizontal_sum_m256(sum_vec);
    // Scalar remainder for sum
    for k in i..len {
        total_sum_simd += slice_data[k];
    }
    let mean = total_sum_simd / (len as f32);
    let mean_vec = _mm256_set1_ps(mean);

    // Variance Calculation
    i = 0; // Reset index for next pass
    let mut var_sum_vec = _mm256_setzero_ps();
    while i + lanes <= len {
        let data_vec = _mm256_loadu_ps(slice_data.as_ptr().add(i));
        let diff_vec = _mm256_sub_ps(data_vec, mean_vec);
        let sq_diff_vec = _mm256_mul_ps(diff_vec, diff_vec);
        var_sum_vec = _mm256_add_ps(var_sum_vec, sq_diff_vec);
        i += lanes;
    }
    let mut total_var_sum_simd = horizontal_sum_m256(var_sum_vec);
    // Scalar remainder for variance sum
    for k in i..len {
        let diff = slice_data[k] - mean;
        total_var_sum_simd += diff * diff;
    }
    let variance = total_var_sum_simd / (len as f32);

    // Normalization, Scaling, Shifting
    let inv_stddev_val = (variance + epsilon).sqrt().recip(); // 1.0 / (variance + epsilon).sqrt()
    let inv_stddev_vec = _mm256_set1_ps(inv_stddev_val);

    i = 0; // Reset index for the final pass
    while i + lanes <= len {
        let d_ptr = slice_data.as_mut_ptr().add(i); // Use as_mut_ptr for store
        let g_ptr = gamma_data.as_ptr().add(i);
        let b_ptr = beta_data.as_ptr().add(i);

        let data_vec = _mm256_loadu_ps(d_ptr); // Load from slice_data (which will be mutated)
        let gamma_vec = _mm256_loadu_ps(g_ptr);
        let beta_vec = _mm256_loadu_ps(b_ptr);

        let normalized_vec = _mm256_mul_ps(_mm256_sub_ps(data_vec, mean_vec), inv_stddev_vec);

        // Assuming FMA is available (checked at dispatch)
        let result_vec = _mm256_fmadd_ps(normalized_vec, gamma_vec, beta_vec);
        // If FMA not available: _mm256_add_ps(_mm256_mul_ps(normalized_vec, gamma_vec), beta_vec);

        _mm256_storeu_ps(d_ptr, result_vec);
        i += lanes;
    }
    // Scalar remainder for normalization
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

    // Handle scalar remainder
    for k_base in i..len {
        let x_val = data_slice[k_base];
        let x_f64 = x_val as f64; // Match original scalar precision for tanh
        let result_f64 = 0.5 * x_f64 * (1.0 + (x_f64 / std::f64::consts::SQRT_2).tanh());
        data_slice[k_base] = result_f64 as f32;
    }
}

impl Tensor<f32> {
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
        let inner_dims_product: usize = first_tensor.shape[axis + 1..].iter().product();


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
                    current_input_flat_idx += axis_el_idx * (if axis < rank -1 {t_ref.shape[axis+1..].iter().product::<usize>()} else {1});
                     if axis < rank -1 { // This is actually wrong above, should be stride for axis
                        let mut stride_for_axis_in_t_ref = 1;
                        for d_idx in (axis + 1)..rank {
                           stride_for_axis_in_t_ref *= t_ref.shape[d_idx];
                        }
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

    /*
    pub fn matmul_simd(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        // 1. Shape checks (self is A, other is B)
        if self.rank() != 2 || other.rank() != 2 {
            return Err(TensorError::InvalidDimension(
                "matmul_simd currently only supports 2D tensors".to_string(),
            ));
        }

        let m = self.shape[0]; // Rows of A
        let k_a = self.shape[1]; // Cols of A / common dimension
        let k_b = other.shape[0]; // Rows of B / common dimension
        let n = other.shape[1]; // Cols of B

        if k_a != k_b {
            return Err(TensorError::IncompatibleShapes(format!(
                "Incompatible shapes for matmul_simd: A has shape [{}, {}], B has shape [{}, {}]",
                m, k_a, k_b, n
            )));
        }
        let common_k = k_a; // K

        // 3. Create result tensor `output_data: Vec<f32>` initialized to zeros, shape `[M, N]`
        let mut output_data = vec![0.0f32; m * n];

        // 4. Loop i from 0 to M-1 (rows of A / output)
        for i in 0..m {
            // 5. Loop j from 0 to N-1 (cols of B / output)
            for j in 0..n {
                // 6. Initialize `dot_product_sum = 0.0f32;`
                let mut dot_product_sum = 0.0f32;
                
                let mut k_idx = 0;
                // 7. Loop k_base from 0 to K-1, step 8 (SIMD part for dot product)
                while k_idx + 7 < common_k {
                    // 8. Load `a_vec = f32x8::from_slice(&self.data[i*K + k_base .. i*K + k_base + 8]);`
                    // Offset for row i in A: i * common_k
                    let a_vec = f32x8::from_slice(&self.data[i * common_k + k_idx .. i * common_k + k_idx + 8]);

                    // 9. Manually construct `b_col_elements: [f32; 8]` by picking `other.data[(k_base+offset)*N + j]`
                    // This gathers elements from column j of B
                    let b_col_elements: [f32; 8] = [
                        other.data[(k_idx + 0) * n + j],
                        other.data[(k_idx + 1) * n + j],
                        other.data[(k_idx + 2) * n + j],
                        other.data[(k_idx + 3) * n + j],
                        other.data[(k_idx + 4) * n + j],
                        other.data[(k_idx + 5) * n + j],
                        other.data[(k_idx + 6) * n + j],
                        other.data[(k_idx + 7) * n + j],
                    ];
                    // 10. Load `b_vec = f32x8::from_array(b_col_elements);`
                    let b_vec = f32x8::from_array(b_col_elements);
                    
                    // 11. `dot_product_sum += (a_vec * b_vec).reduce_sum();`
                    dot_product_sum += (a_vec * b_vec).reduce_sum();
                    
                    k_idx += 8;
                }

                // 12. Handle scalar remainder for k if K % 8 != 0
                // 13. Loop k_scalar from (K - K % 8) to K-1 (or current k_idx to K-1)
                while k_idx < common_k {
                    // 14. `dot_product_sum += self.data[i*K + k_scalar] * other.data[k_scalar*N + j];`
                    dot_product_sum += self.data[i * common_k + k_idx] * other.data[k_idx * n + j];
                    k_idx += 1;
                }
                
                // 15. `output_data[i*N + j] = dot_product_sum;`
                output_data[i * n + j] = dot_product_sum;
            }
        }

        // 16. Return Ok(Tensor::new(output_data, vec![M, N])?)
        Tensor::new(output_data, vec![m, n])
    }
    */
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
#[cfg(target_arch = "x86_64")]
unsafe fn matmul_2d_avx2_fma(
    a_data: &[f32],
    b_data: &[f32],
    m: usize,
    k_dim: usize,
    n_dim: usize,
    c_data: &mut [f32]
) {
    let lanes = 8; // AVX2 processes 8 f32s at a time

    for i in 0..m { // Iterate over rows of A (and C)
        for j in 0..n_dim { // Iterate over columns of B (and C)
            let mut sum_val = 0.0f32;
            let mut k_sidx = 0;

            if k_dim >= lanes {
                let mut sum_simd_acc = _mm256_setzero_ps();
                while k_sidx + lanes <= k_dim {
                    let a_ptr = a_data.as_ptr().add(i * k_dim + k_sidx);
                    let a_vec = _mm256_loadu_ps(a_ptr);

                    // Gather elements for b_vec from column j of B
                    // This is the performance bottleneck due to strided memory access.
                    // _mm256_set_ps takes arguments in reverse order for memory layout.
                    let b_val7 = *b_data.get_unchecked((k_sidx + 7) * n_dim + j);
                    let b_val6 = *b_data.get_unchecked((k_sidx + 6) * n_dim + j);
                    let b_val5 = *b_data.get_unchecked((k_sidx + 5) * n_dim + j);
                    let b_val4 = *b_data.get_unchecked((k_sidx + 4) * n_dim + j);
                    let b_val3 = *b_data.get_unchecked((k_sidx + 3) * n_dim + j);
                    let b_val2 = *b_data.get_unchecked((k_sidx + 2) * n_dim + j);
                    let b_val1 = *b_data.get_unchecked((k_sidx + 1) * n_dim + j);
                    let b_val0 = *b_data.get_unchecked(k_sidx * n_dim + j);
                    let b_vec = _mm256_set_ps(b_val7, b_val6, b_val5, b_val4, b_val3, b_val2, b_val1, b_val0);

                    sum_simd_acc = _mm256_fmadd_ps(a_vec, b_vec, sum_simd_acc);
                    k_sidx += lanes;
                }
                sum_val += horizontal_sum_m256(sum_simd_acc);
            }

            // Scalar loop for remaining elements in k_dim
            for k_rem_idx in k_sidx..k_dim {
                sum_val += *a_data.get_unchecked(i * k_dim + k_rem_idx) * *b_data.get_unchecked(k_rem_idx * n_dim + j);
            }
            *c_data.get_unchecked_mut(i * n_dim + j) = sum_val;
        }
    }
}

impl Tensor<f32> {
    pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let m = a.shape.get(0).copied().unwrap_or(0);
        let k_a = a.shape.get(1).copied().unwrap_or(0);
        let k_b = b.shape.get(0).copied().unwrap_or(0);
        let n = b.shape.get(1).copied().unwrap_or(0);

        if a.rank() != 2 || b.rank() != 2 {
            return Err(TensorError::InvalidDimension(
                "Matmul currently only supports 2D tensors".to_string(),
            ));
        }

        if k_a != k_b {
            return Err(TensorError::IncompatibleShapes(format!(
                "Incompatible shapes for matmul: A has shape [{}, {}], B has shape [{}, {}]",
                m, k_a, k_b, n
            )));
        }

        let mut result_data = vec![0.0f32; m * n];

        if cfg!(target_arch = "x86_64") &&
           is_x86_feature_detected!("avx2") &&
           is_x86_feature_detected!("fma") &&
           m > 0 && k_a > 0 && n > 0 // Ensure dimensions are not zero
        {
            // SAFETY: We've checked for AVX2 and FMA support.
            // Input slices a.data, b.data and mutable slice result_data are valid.
            // Dimensions m, k_a, n are passed to ensure bounds are respected within the function.
            unsafe {
                matmul_2d_avx2_fma(&a.data, &b.data, m, k_a, n, &mut result_data);
            }
        } else if m > 0 && k_a > 0 && n > 0 { // Fallback to scalar if SIMD not available or dims are zero
            for i_idx in 0..m {
                for j_idx in 0..n {
                    let mut sum = 0.0;
                    for k_sidx in 0..k_a { // k_a is common_k
                        // Using direct data access assuming row-major layout
                        sum += a.data[i_idx * k_a + k_sidx] * b.data[k_sidx * n + j_idx];
                    }
                    result_data[i_idx * n + j_idx] = sum;
                }
            }
        } else {
            // If any dimension is zero, the result is typically an empty tensor (or tensor with zero elements in some dims)
            // The result_data is already initialized as vec![0.0; m*n], so if m*n is 0, it's empty.
            // If m, k_a or n is 0, m*n will be 0, leading to an empty result_data if matmul conditions not met.
            // This path ensures we don't try to execute loops with zero dimensions if SIMD path is skipped due to zero dims.
        }
        Tensor::new(result_data, vec![m, n])
    }

    pub fn softmax(&self, axis: usize) -> Result<Tensor<f32>, TensorError> {
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

        let mut result_data = self.data.clone(); // Output tensor data
        let axis_size = self.shape[axis];
        if axis_size == 0 { // Avoid division by zero or issues with empty slices
            return Ok(Tensor::new(result_data, self.shape.clone())?);
        }
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
                    current_slice.push(self.data[current_flat_idx]); // Read from original self.data
                }

                // Process the extracted slice
                if cfg!(target_arch = "x86_64") &&
                   is_x86_feature_detected!("avx2") &&
                   is_x86_feature_detected!("fma") && // FMA is used in expf_approx_taylor_avx2
                   current_slice.len() > 0
                {
                    // SAFETY: AVX2 & FMA checked. current_slice is a valid mutable slice.
                    unsafe {
                        softmax_slice_avx2(&mut current_slice);
                    }
                } else if !current_slice.is_empty() {
                    // Original scalar logic for current_slice
                    let max_val_scalar = current_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_exp_values_scalar = 0.0f32;

                    for val_ref in current_slice.iter_mut() {
                        *val_ref = (*val_ref - max_val_scalar).exp(); // exp in place
                        sum_exp_values_scalar += *val_ref;
                    }

                    if sum_exp_values_scalar == 0.0 { sum_exp_values_scalar = 1e-9; } // Avoid division by zero
                    let inv_sum_exp_scalar = sum_exp_values_scalar.recip();
                    for val_ref in current_slice.iter_mut() {
                        *val_ref *= inv_sum_exp_scalar; // divide in place
                    }
                }

                // Copy processed slice back into result_data
                for k in 0..axis_size {
                    let current_flat_idx = self._flat_index_for_softmax(i, k, j, axis, inner_dims_product)?;
                    if let Some(val_to_write) = current_slice.get(k) {
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
        Tensor::new(result_data, self.shape.clone())
    }

    // Helper for softmax indexing (no change needed here)
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

        let mut result_data = self.data.clone(); // Clone data to modify in place
        let num_vectors = result_data.len() / last_dim_size;

        for i in 0..num_vectors {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            let current_mut_slice = &mut result_data[start..end];

            if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: We've checked for AVX2 and FMA.
                // layernorm_slice_avx2 operates on mutable slices, ensuring memory safety
                // as long as the slices are valid and non-overlapping for reads/writes.
                // gamma.data and beta.data are read-only here. current_mut_slice is mutated.
                unsafe {
                    layernorm_slice_avx2(current_mut_slice, &gamma.data, &beta.data, epsilon);
                }
            } else {
                // Original scalar logic for the slice
                let mean_scalar: f32 = current_mut_slice.iter().sum::<f32>() / (last_dim_size as f32);
                let variance_scalar: f32 = current_mut_slice.iter().map(|&x| (x - mean_scalar).powi(2)).sum::<f32>() / (last_dim_size as f32);
                let inv_stddev_scalar = (variance_scalar + epsilon).sqrt().recip(); // Using recip() for 1.0/sqrt()

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
            // SAFETY: We've checked that AVX2 is supported by the CPU.
            // The gelu_avx2 function operates on a mutable slice, ensuring memory safety
            // as long as the slice itself is valid, which result_data is.
            unsafe {
                gelu_avx2(&mut result_data);
            }
        } else {
            // Fallback to scalar implementation
            for x_val_ref in result_data.iter_mut() {
                let x = *x_val_ref as f64; // Use f64 for tanh precision, matching original scalar path
                let gelu_val = 0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).tanh());
                *x_val_ref = gelu_val as f32;
            }
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

    // The matmul_simd tests are removed as matmul_simd itself was removed.
    // If matmul_simd is re-added, its tests should be too.

    // Tests for gelu_avx2 (via public gelu method)
    // We can't directly test gelu_simd_correctness anymore as gelu_simd was removed.
    // Instead, the existing test_gelu will now cover both AVX2 and scalar paths
    // depending on the CPU features of the test environment.
    // For more rigorous testing, one might need to use #[cfg(target_feature = "avx2")]
    // on specific test functions or use runtime checks to force one path or another if possible.

    // A new test specifically for the AVX2 approximation of tanh, if desired:
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_tanhf_approx_avx2() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not detected, skipping test_tanhf_approx_avx2.");
            return;
        }
        unsafe {
            // Test values including some around the clamp range and zero
            let test_inputs = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 0.1, -0.1];
            let mut results_approx = Vec::with_capacity(test_inputs.len());
            let mut results_precise = Vec::with_capacity(test_inputs.len());

            for &val in &test_inputs {
                results_precise.push(libm::tanhf(val)); // Using libm::tanhf for a more precise reference
            }

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
            // Remainder
            for k in i..test_inputs.len() {
                let input_val = test_inputs[k];
                // Scalar version of the same approximation for direct comparison for remainder
                 let clamp_val = 3.0;
                 let x_clamped = input_val.max(-clamp_val).min(clamp_val);
                 let x_sq = x_clamped * x_clamped;
                 let num = x_clamped * (27.0 + x_sq);
                 let den = 27.0 + 9.0 * x_sq;
                 results_approx.push(num/den);
            }


            for (idx, (approx, precise)) in results_approx.iter().zip(results_precise.iter()).enumerate() {
                // Polynomial approximation has its limits, especially outside [-3, 3]
                // Let's check values within [-3,3] with a tighter tolerance
                // and values outside with a looser one, or accept they diverge.
                let input_val = test_inputs[idx];
                let tolerance = if input_val.abs() <= 3.0 { 0.01 } else { 0.1 }; // Looser for outside clamp_val used in approx
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
            // Test inputs (typically x <= 0 for softmax after max subtraction)
            // The Taylor approx is more accurate for x close to 0.
            let test_inputs = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, -0.1, -0.01, -4.0, -2.5, -1.5, -0.75, -0.25];
            let mut results_approx = Vec::with_capacity(test_inputs.len());
            let mut results_precise = Vec::with_capacity(test_inputs.len());

            for &val in &test_inputs {
                results_precise.push(libm::expf(val)); // Precise reference
            }

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
            // Remainder using scalar version of the same Taylor approximation for direct comparison
            for k in i..test_inputs.len() {
                let x = test_inputs[k];
                let mut res = 1.0 + x/5.0;
                res = 1.0 + x/4.0 * res;
                res = 1.0 + x/3.0 * res;
                res = 1.0 + x/2.0 * res;
                res = 1.0 + x * res;
                results_approx.push(res.max(0.0)); // Ensure non-negative like SIMD version
            }

            for (idx, (approx, precise)) in results_approx.iter().zip(results_precise.iter()).enumerate() {
                let input_val = test_inputs[idx];
                // Taylor series is most accurate near 0. For large negative numbers, it will deviate.
                // For x = -5, exp(-5) is ~0.0067. Approx gives ~0.016.
                let tolerance = if input_val > -2.0 { 0.01 } else if input_val > -4.0 { 0.05 } else { 0.2 };
                assert!((approx - precise).abs() < tolerance, "Input: {}, Approx: {}, Precise: {}, Diff: {}", input_val, approx, precise, (approx-precise).abs());
            }
        }
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
