#ifndef TENSOR_BACKEND
#define TENSOR_BACKEND

#include <vector>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>


void run_vector_add(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result);

void run_vector_sub(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result);

void run_vector_element_mul(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result);

void run_vector_div(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result);

void run_vector_pow(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& result);

void run_gemm(const std::vector<float>& A, const std::vector<float>& B, 
              std::vector<float>& C, int M, int N, int K,
              float alpha = 1.0f, float beta = 0.0f);

void run_vector_cos(const std::vector<float>& input, std::vector<float>& result);
void run_vector_sin(const std::vector<float>& input, std::vector<float>& result);
void run_vector_tanh(const std::vector<float>& input, std::vector<float>& result);
void run_vector_relu(const std::vector<float>& input, std::vector<float>& result);
void run_vector_exp(const std::vector<float>& input, std::vector<float>& result);
void run_vector_leakyrelu(const std::vector<float> &input,
                          std::vector<float> &result, float alpha = 0.01f);

#endif // TENSOR_BACKEND
