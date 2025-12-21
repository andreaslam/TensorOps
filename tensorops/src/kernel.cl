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
                     __global float* C,
                     float a_len_f,
                     float b_len_f)
{
    // Get the unique global index for this work-item
    int gid = get_global_id(0);

    int a_len = (int)a_len_f;
    int b_len = (int)b_len_f;
    int a_idx = (a_len == 1) ? 0 : gid;
    int b_idx = (b_len == 1) ? 0 : gid;

    // Perform the element-wise addition
    C[gid] = A[a_idx] + B[b_idx];
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
                     __global float* C,
                     float a_len_f,
                     float b_len_f)
{
    // Get the unique global index for this work-item
    int gid = get_global_id(0);

    int a_len = (int)a_len_f;
    int b_len = (int)b_len_f;
    int a_idx = (a_len == 1) ? 0 : gid;
    int b_idx = (b_len == 1) ? 0 : gid;

    // Perform the element-wise subtraction
    C[gid] = A[a_idx] - B[b_idx];
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
                            __global float* C,
                            float a_len_f,
                            float b_len_f)
{
    int gid = get_global_id(0);

    int a_len = (int)a_len_f;
    int b_len = (int)b_len_f;
    int a_idx = (a_len == 1) ? 0 : gid;
    int b_idx = (b_len == 1) ? 0 : gid;

    // Perform the element-wise multiplication
    C[gid] = A[a_idx] * B[b_idx];
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
                            __global float* C,
                            float a_len_f,
                            float b_len_f)
{
    int gid = get_global_id(0);

    int a_len = (int)a_len_f;
    int b_len = (int)b_len_f;
    int a_idx = (a_len == 1) ? 0 : gid;
    int b_idx = (b_len == 1) ? 0 : gid;

    // Perform the element-wise division
    C[gid] = A[a_idx] / B[b_idx];
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
 * VecPermuteTemplate: Stride-based permute/transpose copy.
 *
 * This kernel copies from a source tensor into a contiguous output tensor
 * given:
 * - tgt_shape / tgt_strides: describe the output indexing (row-major)
 * - src_strides: strides to use when mapping output coordinates back into the
 *   source buffer.
 *
 * For a permute with dims, Python precomputes:
 *   src_strides[i] = parent_strides[dims[i]]
 * and sets tgt_shape to the permuted shape.
 *
 * Inputs:
 * src:        __global float* - source buffer
 * src_shape:  __global float* - (unused, kept for signature symmetry)
 * src_strides:__global float* - strides mapping output coords -> source index
 * tgt_shape:  __global float* - target shape
 * tgt_strides:__global float* - target row-major strides
 * out:        __global float* - output buffer
 */
#define RANK 1
__kernel void VecPermuteTemplate(
    __global const float* src,
    __global const float* src_shape,
    __global const float* src_strides,
    __global const float* tgt_shape,
    __global const float* tgt_strides,
    __global float* out
) {
    int gid = get_global_id(0);

    int tgt_size = 1;
    for (int i = 0; i < RANK; i++) {
        tgt_size *= (int)tgt_shape[i];
    }
    if (gid >= tgt_size) return;

    int src_idx = 0;
    for (int dim = 0; dim < RANK; dim++) {
        int stride = (int)tgt_strides[dim];
        int tdim = (int)tgt_shape[dim];
        int coord = (gid / stride) % tdim;
        src_idx += coord * (int)src_strides[dim];
    }

    out[gid] = src[src_idx];
}

/*
 * TiledMatMul_16x16: Tiled Matrix Multiplication with 2D NDRange
 *
 * Execution model:
 *   - 2D global NDRange: (M, N) where each workitem computes one output element C[row, col]
 *   - 2D local NDRange (workgroup): 16x16
 *   - Cooperative tiling: tiles of 16x16 from A and B are loaded into local memory
 *   - Inner K-loop over tile blocks of size 16
 *   - Barriers synchronize before loading next tile
 *
 * Epilogue fusion: Users can provide epilogue_src that operates on 'acc' register.
 *   The epilogue code replaces EPILOGUE_PLACEHOLDER in this kernel.
 *   Example: float result = sin(acc) + bias[col]; C[row, col] = result;
 *
 * Inputs:
 * A: __global float* - Matrix A (M x K)
 * B: __global float* - Matrix B (K x N)
 * M_buf: __global float* - M dimension (scalar)
 * N_buf: __global float* - N dimension (scalar)
 * K_buf: __global float* - K dimension (scalar)
 * Output:
 * C: __global float* - Matrix C (M x N)
 */
#define TILE_SIZE 16

__kernel void TiledMatMul_16x16(
    __global const float* A,
    __global const float* B,
    __global const float* M_buf,
    __global const float* N_buf,
    __global const float* K_buf,
    __local float* A_tile,
    __local float* B_tile,
    __global float* C
)
{
    int M = (int)M_buf[0];
    int N = (int)N_buf[0];
    int K = (int)K_buf[0];

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int global_row = get_group_id(0) * TILE_SIZE + local_row;
    int global_col = get_group_id(1) * TILE_SIZE + local_col;

    float acc = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        if (global_row < M && (tile_k + local_col) < K) {
            A_tile[local_row * TILE_SIZE + local_col] = A[global_row * K + tile_k + local_col];
        } else {
            A_tile[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        if ((tile_k + local_row) < K && global_col < N) {
            B_tile[local_row * TILE_SIZE + local_col] = B[(tile_k + local_row) * N + global_col];
        } else {
            B_tile[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_tile[local_row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        EPILOGUE_PLACEHOLDER
        C[global_row * N + global_col] = acc;
    }
}

