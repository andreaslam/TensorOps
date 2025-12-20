<div align="center">

# TensorOps
#### A Work-In-Progress Autograd Library

<img src="https://img.shields.io/badge/Powered%20by-Python-yellow" alt="Powered by Python">
<img src="https://img.shields.io/badge/Powered%20by-Rust-red" alt="Powered by Rust">
<img src="https://badgen.net/github/commits/andreaslam/TensorOps/main" alt="Total commits">

</div>

## Setting up TensorOps

Firstly, to use this repo, use git clone to make the Repository available locally:

```
git clone https://github.com/andreaslam/TensorOps.git
```

At the root of the repository, run:

```
pip install -e .
```

### Building the Backend

To compile the Rust OpenCL backend, `maturin` is needed. If not installed already, run:

```
pip install maturin
```

Then run:
```
cd tensorops
maturin develop --release
```

The backend should be installed into 


Support for non-OpenCL backend is still in progress. PRs welcome.

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
- Weight and gradient tracking (enable/disable)
- Arithmetic operations (BIDMAS, negation, exponentiation, modulo, [several Python reverse operations](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types))
- Non-linear activation functions (sin, cos, tanh, ReLU, sigmoid, ramp)
- Lazy evaluation
- OpenCL Backend
- Partial graph execution
- Kernel fusion

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
