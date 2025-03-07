cmake_minimum_required(VERSION 3.14)
project(tensorops_hip_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# OpenMP setup
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -lomp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(OpenMP_INCLUDE_DIRS "/opt/homebrew/opt/libomp/include")
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
    include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include")
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else()
    find_package(OpenMP REQUIRED)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

if(NOT WIN32)
    if(Python3_INCLUDE_DIRS MATCHES "/mnt/c/Users")
        set(Python3_INCLUDE_DIRS "/usr/include/python3.11")
    endif()
endif()

# GPU detection via nvidia-smi
find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
if(NVIDIA_SMI_EXECUTABLE)
    message(STATUS "GPU detected: using HIP (GPU) library.")
    # For GPU, assume the header is in repos/HIP (so that <hip/hip_runtime.h> resolves to repos/HIP/hip/hip_runtime.h)
    set(HIP_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../repos/HIP/include")
    set(HIP_COMPILE_DEFS __HIP_GPU_RT__ __HIP_PLATFORM_GPU__)
else()
    message(STATUS "No GPU detected: using HIP-CPU library.")
    # For HIP-CPU, use the include directory that worked previously.
    set(HIP_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../repos/HIP-CPU/include")
    set(HIP_COMPILE_DEFS __HIP_CPU_RT__ __HIP_PLATFORM_CPU__)
endif()

add_library(tensorops_hip_bindings MODULE 
    tensorops/bindings.cpp 
    tensorops/tensor_backend.cpp
)

# Use the selected HIP directory
target_include_directories(tensorops_hip_bindings PRIVATE 
    ${HIP_ROOT_DIR}
    ${OpenMP_INCLUDE_DIRS}
)

if(NOT WIN32)
    target_include_directories(tensorops_hip_bindings PRIVATE ${Python3_INCLUDE_DIRS})
endif()

target_link_libraries(tensorops_hip_bindings PRIVATE 
    pybind11::module
)

if(WIN32)
    set_target_properties(tensorops_hip_bindings PROPERTIES 
        PREFIX "" 
        SUFFIX ".pyd"
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    )
    target_compile_options(tensorops_hip_bindings PRIVATE 
        /openmp:experimental 
        /bigobj
    )
else()
    set_target_properties(tensorops_hip_bindings PROPERTIES 
        PREFIX "" 
        SUFFIX ".so"
    )
endif()

# Define HIP-specific macros based on GPU detection
target_compile_definitions(tensorops_hip_bindings PRIVATE ${HIP_COMPILE_DEFS})

# Enable OpenMP for parallel execution
if(MSVC)
    target_compile_options(tensorops_hip_bindings PRIVATE /openmp:experimental)
else()
    target_compile_options(tensorops_hip_bindings PRIVATE -fopenmp)
endif()
