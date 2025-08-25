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
 * VecPow: Element-wise power operation (C = A ^ B or C = A ^ exponent)
 * Version 1: Element-wise exponent (C[i] = A[i] ^ B[i])
 * Inputs:
 * base: __global float* - Vector of bases
 * exponent: __global float* - Vector of exponents
 * Output:
 * C: __global float* - Output vector where C[i] = pow(base[i], exponent[i])
 */
__kernel void VecPow(__global const float* base,
                     __global const float* exponent,
                     __global float* C)
{
    int gid = get_global_id(0);

    // Calculate base[gid] raised to the power of exponent[gid]
    C[gid] = pow(base[gid], exponent[gid]);
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
                     __global float* B,
                     float base)
{
    int gid = get_global_id(0);
    B[gid] = log(A[gid]) / log(base);
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
