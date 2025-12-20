/*
 * VecAdd: Element-wise vector addition (C = A + B)
 * Inputs:
 * A: __global float* - First input vector
 * B: __global float* - Second input vector
 * Output:
 * C: __global float* - Output vector where C[i] = A[i] + B[i]
 */
__kernel void VecAdd(__global const float* A,
                     __global const float* B,
                     __global float* C)
{
    // Get the unique global index for this work-item
    int gid = get_global_id(0);

    // Perform the element-wise addition
    C[gid] = A[gid] + B[gid];
}

/*
 * VecSub: Element-wise vector subtraction (C = A - B)
 * Inputs:
 * A: __global float* - First input vector
 * B: __global float* - Second input vector
 * Output:
 * C: __global float* - Output vector where C[i] = A[i] - B[i]
 */
__kernel void VecSub(__global const float* A,
                     __global const float* B,
                     __global float* C)
{
    // Get the unique global index for this work-item
    int gid = get_global_id(0);

    // Perform the element-wise subtraction
    C[gid] = A[gid] - B[gid];
}

/*
 * VecElementMul: Element-wise vector multiplication (C = A * B)
 * Inputs:
 * A: __global float* - First input vector
 * B: __global float* - Second input vector
 * Output:
 * C: __global float* - Output vector where C[i] = A[i] * B[i]
 */
__kernel void VecElementMul(__global const float* A,
                            __global const float* B,
                            __global float* C)
{
    int gid = get_global_id(0);

    // Perform the element-wise multiplication
    C[gid] = A[gid] * B[gid];
}

/*
 * VecElementDiv: Element-wise vector division (C = A / B)
 * Inputs:
 * A: __global float* - First input vector
 * B: __global float* - Second input vector
 * Output:
 * C: __global float* - Output vector where C[i] = A[i] / B[i]
 */
__kernel void VecDiv(__global const float* A,
                            __global const float* B,
                            __global float* C)
{
    int gid = get_global_id(0);

    // Perform the element-wise division
    C[gid] = A[gid] / B[gid];
}

/*
 * VecPow: Element-wise power operation (C = A ^ B or C = base ^ exponent)
 * Supports both element-wise and scalar base broadcasting:
 * - If base buffer has same length as work size: C[i] = base[i] ^ exponent[i]
 * - If base buffer has length 1: C[i] = base[0] ^ exponent[i] (broadcasting)
 * Inputs:
 * base: __global float* - Vector of bases (length N or 1)
 * exponent: __global float* - Vector of exponents (length N)
 * C: __global float* - Output vector
 * base_len_f: float - Length of base buffer as float (for broadcast detection)
 */
__kernel void VecPow(__global const float* base,
                     __global const float* exponent,
                     __global float* C,
                     float base_len_f)
{
    int gid = get_global_id(0);
    int base_len = (int)base_len_f;
    
    // Broadcast if base has only one element
    int base_idx = (base_len == 1) ? 0 : gid;
    
    // Calculate base[base_idx] raised to the power of exponent[gid]
    C[gid] = pow(base[base_idx], exponent[gid]);
}

/*
 * VecLog: Element-wise logarithm operation (B = log_base(A))
 * Inputs:
 * A: __global float* - Input vector (values to take the logarithm of)
 * base: float - Base of logarithm (scalar)
 * Output:
 * B: __global float* - Output vector where B[i] = log_base(A[i])
 */
__kernel void VecLog(__global const float* A,
                     __global float* C,
                     float base)
{
    int gid = get_global_id(0);
    C[gid] = log(A[gid]) / log(base);
}

/*
 * VecSin: Element-wise sine (C = sin(A))
 * Inputs:
 * A: __global float* - Input vector (angles in radians)
 * Output:
 * C: __global float* - Output vector where C[i] = sin(A[i])
 */
__kernel void VecSin(__global const float* A,
                     __global float* C)
{
    int gid = get_global_id(0);

    // Calculate the sine
    C[gid] = sin(A[gid]);
}

/*
 * VecCos: Element-wise cosine (C = cos(A))
 * Inputs:
 * A: __global float* - Input vector (angles in radians)
 * Output:
 * C: __global float* - Output vector where C[i] = cos(A[i])
 */
__kernel void VecCos(__global const float* A,
                     __global float* C)
{
    int gid = get_global_id(0);

    // Calculate the cosine
    C[gid] = cos(A[gid]);
}

/*
 * VecTan: Element-wise tangent (C = tan(A))
 * Inputs:
 * A: __global float* - Input vector (angles in radians)
 * Output:
 * C: __global float* - Output vector where C[i] = tan(A[i])
 */
__kernel void VecTan(__global const float* A,
                     __global float* C)
{
    int gid = get_global_id(0);

    // Calculate the tangent
    C[gid] = tan(A[gid]);
}

/*
 * VecAbs: Element-wise absolute value (C = |A|)
 * Inputs:
 * A: __global float* - Input vector
 * Output:
 * C: __global float* - Output vector where C[i] = fabs(A[i])
 */
__kernel void VecAbs(__global const float* A,
                     __global float* C)
{
    int gid = get_global_id(0);

    // Calculate the absolute value
    C[gid] = fabs(A[gid]); // Use fabs for float absolute value
}

/*
 * VecTanh: Element-wise hyperbolic tangent (C = tanh(A))
 * Inputs:
 * A: __global float* - Input vector
 * Output:
 * C: __global float* - Output vector where C[i] = tanh(A[i])
 */
__kernel void VecTanh(__global const float* A,
                      __global float* C)
{
    int gid = get_global_id(0);

    // Calculate the hyperbolic tangent
    C[gid] = tanh(A[gid]);
}

/*
 * VecLeakyReLU: Element-wise Leaky Rectified Linear Unit
 * C = A          if A > 0
 * C = alpha * A  if A <= 0
 * Inputs:
 * A: __global float* - Input vector
 * alpha: float       - The 'leak' factor (a small positive constant, e.g., 0.01f)
 * Output:
 * C: __global float* - Output vector
 */
__kernel void VecLeakyReLU(__global const float* A,
                           __global float* C,
                           float alpha)
{
    int gid = get_global_id(0);
    float val = A[gid];
    C[gid] = (val > 0.0f) ? val : alpha * val;
}

/*
 * VecSum: Reduction sum along an axis.
 *
 * Treats input as shape [pre_axis, axis_len, post_axis]
 * and outputs shape [pre_axis, post_axis].
 */
__kernel void VecSum(__global const float* input,
                        __global float* output,
                        float pre_axis_f,
                        float axis_len_f,
                        float post_axis_f)
{
    int pre_axis = (int)pre_axis_f;
    int axis_len = (int)axis_len_f;
    int post_axis = (int)post_axis_f;

    int id = get_global_id(0);
    if (id >= pre_axis * post_axis) return;

    int pre_idx = id / post_axis;
    int post_idx = id - pre_idx * post_axis;

    float sum = 0.0f;
    for (int k = 0; k < axis_len; k++) {
        int input_idx = (pre_idx * axis_len + k) * post_axis + post_idx;
        sum += input[input_idx];
    }

    output[id] = sum;
}

/*
 * VecMax: Reduction max along an axis.
 *
 * Treats input as shape [pre_axis, axis_len, post_axis]
 * and outputs shape [pre_axis, post_axis].
 */
__kernel void VecMax(__global const float* input,
                        __global float* output,
                        float pre_axis_f,
                        float axis_len_f,
                        float post_axis_f)
{
    int pre_axis = (int)pre_axis_f;
    int axis_len = (int)axis_len_f;
    int post_axis = (int)post_axis_f;

    int id = get_global_id(0);
    if (id >= pre_axis * post_axis) return;

    int pre_idx = id / post_axis;
    int post_idx = id - pre_idx * post_axis;

    int input_idx0 = (pre_idx * axis_len + 0) * post_axis + post_idx;
    float best = input[input_idx0];
    for (int k = 1; k < axis_len; k++) {
        int input_idx = (pre_idx * axis_len + k) * post_axis + post_idx;
        float v = input[input_idx];
        best = (v > best) ? v : best;
    }
    output[id] = best;
}

/*
 * VecMin: Reduction min along an axis.
 *
 * Treats input as shape [pre_axis, axis_len, post_axis]
 * and outputs shape [pre_axis, post_axis].
 */
__kernel void VecMin(__global const float* input,
                        __global float* output,
                        float pre_axis_f,
                        float axis_len_f,
                        float post_axis_f)
{
    int pre_axis = (int)pre_axis_f;
    int axis_len = (int)axis_len_f;
    int post_axis = (int)post_axis_f;

    int id = get_global_id(0);
    if (id >= pre_axis * post_axis) return;

    int pre_idx = id / post_axis;
    int post_idx = id - pre_idx * post_axis;

    int input_idx0 = (pre_idx * axis_len + 0) * post_axis + post_idx;
    float best = input[input_idx0];
    for (int k = 1; k < axis_len; k++) {
        int input_idx = (pre_idx * axis_len + k) * post_axis + post_idx;
        float v = input[input_idx];
        best = (v < best) ? v : best;
    }
    output[id] = best;
}

/*
 * VecExpandTemplate: Broadcast/expand a tensor to a target shape.
 *
 * This kernel is used as a TEMPLATE by the Python frontend, which rewrites:
 *  - the kernel function name (to keep it unique per graph kernel), and
 *  - the RANK define (so the unrolled loop bound is a compile-time constant).
 *
 * Signature is intentionally simple: src + shape/stride metadata + out.
 * Metadata buffers are float arrays but are cast to int inside the kernel.
 */
#define RANK 1
__kernel void VecExpandTemplate(
    __global const float* src,
    __global const float* src_shape,
    __global const float* src_strides,
    __global const float* tgt_shape,
    __global const float* tgt_strides,
    __global float* out
) {
    int gid = (int)get_global_id(0);
    int src_idx = 0;
    for (int d = 0; d < RANK; d++) {
        int tstride = (int)tgt_strides[d];
        int tshape = (int)tgt_shape[d];
        int coord = (gid / tstride) % tshape;
        int sshape = (int)src_shape[d];
        int scol = (sshape == 1) ? 0 : coord;
        int sstride = (int)src_strides[d];
        src_idx += scol * sstride;
    }
    out[gid] = src[src_idx];
}


/*
 * MatMul: Batched Matrix Multiplication (C = A @ B)
 * Inputs:
 * A: __global float* - Matrix A (Batch x M x K)
 * B: __global float* - Matrix B (Batch x K x N)
 * M_buf: __global float* - M dimension (scalar)
 * N_buf: __global float* - N dimension (scalar)
 * K_buf: __global float* - K dimension (scalar)
 * Output:
 * C: __global float* - Matrix C (Batch x M x N)
 */
__kernel void MatMul(__global const float* A,
                     __global const float* B,
                     __global const float* M_buf,
                     __global const float* N_buf,
                     __global const float* K_buf,
                     __global float* C)
{
    int M = (int)M_buf[0];
    int N = (int)N_buf[0];
    int K = (int)K_buf[0];

    int gid = get_global_id(0);
    
    int batch_size = M * N;
    int batch_idx = gid / batch_size;
    int rem = gid % batch_size;
    
    int row = rem / N;
    int col = rem % N;

    int a_offset = batch_idx * M * K;
    int b_offset = batch_idx * K * N;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[a_offset + row * K + k] * B[b_offset + k * N + col];
    }
    C[gid] = sum;
}

