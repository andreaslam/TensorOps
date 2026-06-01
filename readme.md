<div align="center">

# TensorOps
#### A work-in-progress autograd and tensor library

<img src="https://img.shields.io/badge/Powered%20by-Python-yellow" alt="Powered by Python">
<img src="https://img.shields.io/badge/Powered%20by-Rust-red" alt="Powered by Rust">
<img src="https://badgen.net/github/commits/andreaslam/TensorOps/main" alt="Total commits">

</div>

TensorOps is a Python-first autograd and tensor library with a Rust/OpenCL backend.
On macOS, an MLX runtime is available as an alternative backend.

## Repo layout

- tensorops/ - Python API, MLX runtime, utils, and maturin extension wrapper
- tensorops/src/ - Rust backend and OpenCL kernels
- examples/ - TensorOps examples, PyTorch comparisons, and MLX demos
- tensorops/notes/ - design docs and API references (start with INDEX.md)
- test_*.py and tensorops/utils/test_engine.py - quick sanity tests
- precompile_kernels.py, dockerfile, entrypoint.sh - build and tooling scripts

## Install

Clone the repo:

```
git clone https://github.com/andreaslam/TensorOps.git
```

Create a virtual environment and install:

```
python -m venv .venv
. .venv/Scripts/activate
pip install -U pip
pip install .
```

### Rust/OpenCL backend (Windows/Linux)

The Rust extension is required on non-macOS platforms.
Install Rust and OpenCL drivers, then build the backend:

```
cd tensorops
maturin develop --release
```

### macOS (MLX backend)

Install the MLX extra and select the MLX runtime:

```
pip install .[mac]
```

Then either set the environment variable or select the device explicitly:

```
TENSOROPS_BACKEND=mlx
```

Or in code:

```
from tensorops.device import TensorOpsDevice
TensorOpsDevice.APPLE
```

### Optional extras

PyTorch comparison examples:

```
pip install .[pytorch]
```

### Kernel cache (optional)

Precompile kernel binaries for faster startup:

```
python precompile_kernels.py
```

## Quick start

```python
from tensorops.tensor import Tensor, TensorContext

with TensorContext() as ctx:
	x = Tensor([[1.0, 2.0]], requires_grad=True)
	w = Tensor([[0.5], [1.0]], requires_grad=True, weight=True)
	y = x @ w
	loss = (y - Tensor([[1.5]], requires_grad=False)) ** 2
	ctx.forward()
	ctx.backward()

print("loss:", loss.tolist())
print("dL/dw:", w.grads.tolist())
```

You can also materialise a single value without an explicit context:

```python
from tensorops.tensor import Tensor

out = (Tensor([1.0, 2.0, 3.0]) + 1).tanh().compute()
print(out.tolist())
```

## Examples

- examples/tensorops/tensordemo.py - tensor ops and graph execution
- examples/tensorops/mnist.py - MNIST classifier with SequentialModel
- examples/pytorch/ - PyTorch equivalents for comparison
- examples/mlx_examples.py - MLX backend demos for macOS

## Features (implemented)

### Tensor and graph execution
- Lazy graph building with TensorContext forward/backward
- Elementwise ops: add, sub, mul, div, pow
- Reductions: sum, max, min
- Linear algebra: matmul with batch support
- Shape ops: reshape, expand, permute, squeeze, unsqueeze
- Activations: tanh, sin, cos, relu, leaky_relu, sigmoid, softplus, softmax
- Other ops: log/log2/log10, exp, argmax, detach

### Backends
- Rust/OpenCL backend via tensorops_backend
- MLX runtime on macOS (TENSOROPS_BACKEND=mlx or TensorOpsDevice.APPLE)
- Kernel fusion for elementwise ops and matmul epilogues in the Rust backend

### Models and training
- Model, Layer, SequentialModel, FullyConnectedNetwork, SimpleSequentialModel
- Losses: L1 (MAE), MSE, BCE, CrossEntropy
- Optimisers: Adam, AdamW, SGD (with weight decay and gradient clipping)

### Utilities
- Graph visualisation (visualise_graph)
- Plotting helper (PlotterUtil)
- Tensor conversions: tolist(), numpy(), head()

## Documentation

Design notes and API references live in tensorops/notes. Start at INDEX.md.

## Status notes

- Node-based API exists but is deprecated and incomplete; use Tensor and TensorContext.
- ONNX import/export is available via tensorops/utils/onnx_exporter.py.
- This is a work in progress and APIs may change.
