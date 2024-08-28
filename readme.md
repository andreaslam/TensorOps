<div align="center">

# TensorOps
#### A Work-In-Progress Autograd Library

<img src="https://img.shields.io/badge/Powered%20by-Python-306998" alt="Powered by Python">
<img src="https://badgen.net/github/commits/andreaslam/TensorOps/main" alt="Total commits">

</div>

## Setting up TensorOps

Firstly, to use this repo, use git clone to make the Repository available locally:

```
git clone https://github.com/andreaslam/TensorOps
```

Then, in a terminal, enter:

```
pip install -e .
```

## Getting started with TensorOps

There are some examples available in the [examples folder](https://github.com/andreaslam/TensorOps/tree/main/examples)

Most examples implemented in the examples folder, will have a corresponding [PyTorch](https://github.com/pytorch/pytorch) implementation for comparison and juxtaposition.

## TensorOps Features

### Node
- Forward pass
- Backward pass
- Node weight and gradient tracking (enable/disable)
- Arithmetic operations (BIDMAS, negation, exponentiation)
- Non-linear activation functions (sin, cos, tanh, ReLU, sigmoid)

### Model
- Customisable neural networks
- Customisable methods

## Loss functions
- Mean Absolute Error
- Mean Square Error

### Optimisers
- Adam
- AdamW
- Stochastic Gradient Descent (SGD)

### Utility features
- Function graphing and plotting
- Colour-coded plotter for Directed Acyclic Graphs (DAGs)
