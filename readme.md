<div align="center">

# TensorOps
#### A Work-In-Progress Autograd Library

<img src="https://img.shields.io/badge/Powered%20by-Python-yellow" alt="Powered by Python">
<img src="https://img.shields.io/badge/Powered%20by-C++-blue" alt="Powered by C++">
<img src="https://badgen.net/github/commits/andreaslam/TensorOps/main" alt="Total commits">

</div>

## Setting up TensorOps

Firstly, to use this repo, use git clone to make the Repository available locally:

```
git clone https://github.com/andreaslam/TensorOps.git
```

Then, in a terminal, enter:

```
pip install .
```

## Setting up HIP Backend (Work in Progress)

Currently, development is done using the HIP-CPU library, which is the drop-in replacement for the HIP backend. All headers in the code would refer to HIP-CPU and not HIP.

Git clone the [HIP-CPU repo](https://github.com/ROCm/HIP-CPU) onto the project root and create build folder

```
git clone https://github.com/ROCm-Developer-Tools/HIP-CPU.git
cd HIP-CPU
mkdir build
cd build
cmake ..
cmake --build . --config Release

python setup.py build_ext --inplace --verbose
pip install .
```

## Getting started with TensorOps

There are some examples available in the [examples folder](https://github.com/andreaslam/TensorOps/tree/main/examples)

Most examples implemented in the examples folder, will have a corresponding [PyTorch](https://github.com/pytorch/pytorch) implementation for comparison and juxtaposition.

## TensorOps Features

### Node (Deprecating)
- Forward pass
- Backward pass
- Node weight and gradient tracking (enable/disable)
- Arithmetic operations (BIDMAS, negation, exponentiation, modulo, [several Python reverse operations](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types))
- Non-linear activation functions (sin, cos, tanh, ReLU, sigmoid, ramp)
- Lazy evaluation

### Tensor (Work In Progress)
- Forward pass (Work In Progress)
- Backward pass (Work In Progress)
- Node weight and gradient tracking (enable/disable)
- Arithmetic operations (BIDMAS, negation, exponentiation, modulo, [several Python reverse operations](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types))
- Non-linear activation functions (sin, cos, tanh, ReLU, sigmoid, ramp)
- Lazy evaluation
- HIP Backend (compatible with CUDA, CPU and AMD GPU backends)

### Model (New Version In Progress)
- Mix and match activation functions
- Configurable layer sizes
- Customisable loss functions
- Customisable forward passes and general-purpose neural network abstractions

## Loss functions (New Version In Progress)
- Mean Absolute Error
- Mean Square Error

### Optimisers (New Version In Progress)
- Adam
- AdamW
- Stochastic Gradient Descent (SGD)

### Utility features (New Version In Progress)
- Function graphing and plotting
- Colour-coded plotter for Directed Acyclic Graphs (DAGs)
