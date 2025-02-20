#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor_backend.hpp"
#include <iostream>

namespace py = pybind11;

// Recursive helper function to flatten a nested list
void flatten(const py::list &nested_list, std::vector<py::object> &flat_list) {
    for (auto &item : nested_list) {
        if (py::isinstance<py::list>(item)) {
            flatten(item.cast<py::list>(), flat_list); // Recursively flatten
        } else {
            flat_list.push_back(py::reinterpret_borrow<py::object>(item));
        }
    }
}

// Exposed function for flattening a list
py::list flatten_list(const py::list &nested_list) {
    std::vector<py::object> flat_vector;
    flatten(nested_list, flat_vector);
    return py::cast(flat_vector); // Convert vector to Python list
}

// Recursive function to compute the shape of a nested vector
std::vector<size_t> get_shape(const py::object& obj) {
    if (py::isinstance<py::list>(obj)) {
        auto list = obj.cast<py::list>();
        std::vector<size_t> shape = {list.size()};
        if (!list.empty()) {
            auto sub_shape = get_shape(list[0]);
            shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
        }
        return shape;
    }
    return {}; // Base case: not a list
}

PYBIND11_MODULE(hip_cpu_bindings, m) {
    m.doc() = "Python bindings for HIP CPU-based vector operations and GEMM";

    // Bind vector elementwise operations (add, subtract, multiply, divide)
    auto bind_vector_op = [&m](const char *name, auto func, const char *doc) {
        m.def(name, [func](const std::vector<float> &a,
                           const std::vector<float> &b) {
            if (a.size() != b.size()) {
                throw std::invalid_argument(
                    "Vector size mismatch:\n"
                    "  Size of 'a': " + std::to_string(a.size()) + "\n"
                    "  Size of 'b': " + std::to_string(b.size()) + "\n"
                    "Both vectors must have the same size.");
            }

            // Prepare output vector
            std::vector<float> output_values(a.size(), 0.0f);

            // Perform the operation
            func(a, b, output_values);

            return output_values;
        }, doc, py::arg("a"), py::arg("b"));
    };

    bind_vector_op("run_vector_add", run_vector_add,
                   "Run vector addition using HIP-CPU and return the result");
    bind_vector_op("run_vector_sub", run_vector_sub,
                   "Run vector subtraction using HIP-CPU and return the result");
    bind_vector_op("run_vector_element_mul", run_vector_element_mul,
                   "Run vector elementwise multiplication using HIP-CPU and return the result");
    bind_vector_op("run_vector_div", run_vector_div,
                   "Run vector division using HIP-CPU and return the result");

    bind_vector_op("run_vector_pow", run_vector_pow,
                   "Run elementwise power operation on input and return the result");


    // Bind elementwise mathematical operations (cosine, sine, tanh, relu, leakyrelu)
    auto bind_elementwise_op = [&m](const char *name, auto func, const char *doc) {
        m.def(name, [func](const std::vector<float> &input) {
            // Prepare output vector
            std::vector<float> output_values(input.size(), 0.0f);

            // Perform the operation
            func(input, output_values);

            return output_values;
        }, doc, py::arg("input"));
    };

    bind_elementwise_op("run_vector_cos", run_vector_cos,
                        "Run elementwise cosine operation on input and return the result");
    bind_elementwise_op("run_vector_sin", run_vector_sin,
                        "Run elementwise sine operation on input and return the result");
    bind_elementwise_op("run_vector_tanh", run_vector_tanh,
                        "Run elementwise tanh operation on input and return the result");
    bind_elementwise_op("run_vector_relu", run_vector_relu,
                        "Run elementwise ReLU operation on input and return the result");

    bind_elementwise_op("run_vector_exp", run_vector_exp,
                        "Run elementwise exponential operation on input and return the result");

    // Expose flatten_list to Python
    m.def("flatten_list", &flatten_list, "Flatten an n-dimensional list");

    m.def("get_shape", &get_shape, "Get the shape of a nested list");

    m.def("run_vector_leakyrelu", [](const std::vector<float> &input,
                                     float alpha = 0.01f) {
        // Prepare output vector
        std::vector<float> output_values(input.size(), 0.0f);

        // Perform Leaky ReLU operation
        run_vector_leakyrelu(input, output_values, alpha);

        return output_values;
    }, py::arg("input"), py::arg("alpha") = 0.01f,
       R"pbdoc(
           Run elementwise Leaky ReLU operation on input and return the result.
           The default value of alpha is set to 0.01.
       )pbdoc");

    m.def("run_gemm", [](const std::vector<float> &A,
                         const std::vector<float> &B,
                         int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f) {
        // Validate dimensions
        if (A.size() != M * K || B.size() != K * N) {
            throw std::invalid_argument(
                "Matrix dimension mismatch:\n"
                "  Expected sizes:\n"
                "    A: M*K (" + std::to_string(M) + "*" + std::to_string(K) + ") = " + std::to_string(M * K) + "\n"
                "    B: K*N (" + std::to_string(K) + "*" + std::to_string(N) + ") = " + std::to_string(K * N) + "\n"
                "  Actual sizes:\n"
                "    A: " + std::to_string(A.size()) + "\n"
                "    B: " + std::to_string(B.size()) + "\n");
        }

        // Prepare a temporary output vector for computation
        std::vector<float> C_values(M * N, 0.0f);

        // Perform GEMM operation
        run_gemm(A, B, C_values, M, N, K, alpha, beta);

        return C_values;
    }, 
       "Run GEMM using HIP-CPU and return the resulting matrix",
       py::arg("A"), py::arg("B"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
