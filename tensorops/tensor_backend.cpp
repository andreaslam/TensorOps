#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

// Kernel for unary operations
template <typename Op>
__global__ void unary_operation_kernel(const float* input, float* result, int size, Op operation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = operation(input[idx]);
    }
}

// Kernel for binary operations
template <typename Op>
__global__ void binary_operation_kernel(const float* a, const float* b, float* result, int size, Op operation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = operation(a[idx], b[idx]);
    }
}

// Generic function for unary operations
template <typename Op>
void run_vector_operation_unary(const std::vector<float>& input,
                                std::vector<float>& result,
                                Op operation) {
    if (input.size() != result.size()) {
        throw std::invalid_argument("Input and result vectors must have the same size.");
    }

    int size = input.size();
    float *d_input, *d_result;

    hipMalloc(&d_input, size * sizeof(float));
    hipMalloc(&d_result, size * sizeof(float));

    hipMemcpy(d_input, input.data(), size * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(unary_operation_kernel<Op>, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_input, d_result, size, operation);

    hipMemcpy(result.data(), d_result, size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_result);
}

// Generic function for binary operations
template <typename Op>
void run_vector_operation_binary(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 std::vector<float>& result,
                                 Op operation) {
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    int size = a.size();
    float *d_a, *d_b, *d_result;

    hipMalloc(&d_a, size * sizeof(float));
    hipMalloc(&d_b, size * sizeof(float));
    hipMalloc(&d_result, size * sizeof(float));

    hipMemcpy(d_a, a.data(), size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), size * sizeof(float), hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(binary_operation_kernel<Op>, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_a, d_b, d_result, size, operation);

    hipMemcpy(result.data(), d_result, size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_result);
}

void run_vector_exp(const std::vector<float>& input, std::vector<float>& result) {
    run_vector_operation_unary(input, result, [] __device__ (float x) { return pow(2.718281828459045, x); });
}

// Implementation of vector operations
void run_vector_cos(const std::vector<float>& input, std::vector<float>& result) {
    run_vector_operation_unary(input, result, [] __device__ (float x) { return cosf(x); });
}

void run_vector_sin(const std::vector<float>& input, std::vector<float>& result) {
    run_vector_operation_unary(input, result, [] __device__ (float x) { return sinf(x); });
}

void run_vector_tanh(const std::vector<float>& input, std::vector<float>& result) {
    run_vector_operation_unary(input, result, [] __device__ (float x) { return tanhf(x); });
}

void run_vector_relu(const std::vector<float>& input, std::vector<float>& result) {
    run_vector_operation_unary(input, result, [] __device__ (float x) { return x > 0 ? x : 0; });
}

void run_vector_leakyrelu(const std::vector<float>& input,
                          std::vector<float>& result,
                          float alpha) {
    run_vector_operation_unary(input, result,
                               [alpha] __device__ (float x) { return x > 0 ? x : alpha * x; });
}

void run_vector_add(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result) {
    run_vector_operation_binary(a, b, result,
                                [] __device__ (float x, float y) { return x + y; });
}

void run_vector_sub(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result) {
    run_vector_operation_binary(a, b, result,
                                [] __device__ (float x, float y) { return x - y; });
}

void run_vector_element_mul(const std::vector<float>& a,
                            const std::vector<float>& b,
                            std::vector<float>& result) {
    run_vector_operation_binary(a, b, result,
                                [] __device__ (float x, float y) { return x * y; });
}

void run_vector_div(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result) {
    run_vector_operation_binary(a, b, result,
                                [] __device__ (float x, float y) { return x / y; });
}

void run_vector_pow(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result) {
    run_vector_operation_binary(a, b, result,
                                [] __device__ (float x, float y) { return pow(x, y); });
}

// Kernel for GEMM
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K,
                            float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void run_gemm(const std::vector<float>& A,
              const std::vector<float>& B,
              std::vector<float>& C,
              int M, int N, int K,
              float alpha = 1.0f,
              float beta = 0.0f) {
    
    // Validate matrix dimensions
    if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    hipMalloc(&d_A, A.size() * sizeof(float));
    hipMalloc(&d_B, B.size() * sizeof(float));
    hipMalloc(&d_C, C.size() * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_A, A.data(), A.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), B.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_C, C.data(), C.size() * sizeof(float), hipMemcpyHostToDevice);

    // Configure kernel launch dimensions
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch GEMM kernel
    hipLaunchKernelGGL(gemm_kernel, blocksPerGrid, threadsPerBlock, 0, 0,
                       d_A, d_B, d_C, M, N, K, alpha, beta);

    // Copy result back to host
    hipMemcpy(C.data(), d_C, C.size() * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

