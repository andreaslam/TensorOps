#!/usr/bin/env python3
"""
Example: Using TensorOps with MLX backend on macOS

This example demonstrates:
1. Creating tensors with MLX device affinity
2. Building a computation graph
3. Automatic runtime selection (MLX on macOS)
4. Kernel mapping from TensorOps to MLX operations
"""

import platform
import sys

# Check if we're on macOS
if platform.system().lower() != "darwin":
    print("This example requires macOS. Exiting.")
    sys.exit(1)

try:
    import mlx.core as mx
except ImportError:
    print("MLX is not installed. Install it with: pip install mlx")
    sys.exit(1)

import numpy as np

from tensorops.device import TensorOpsDevice
from tensorops.tensor import Tensor, TensorContext


def example_basic_ops():
    """Basic arithmetic on MLX device."""
    print("\n" + "=" * 50)
    print("Example 1: Basic Operations")
    print("=" * 50)

    # Create tensors on APPLE device (MLX)
    a = Tensor([1.0, 2.0, 3.0], device=TensorOpsDevice.APPLE)
    b = Tensor([4.0, 5.0, 6.0], device=TensorOpsDevice.APPLE)

    # Build computation graph
    c = a + b
    d = c * 2.0

    # Execute on MLX
    with TensorContext(device=TensorOpsDevice.APPLE) as ctx:
        ctx.add_op(d)
        ctx.forward()

    result = d.tolist()
    expected = [10.0, 14.0, 18.0]

    print(f"a:        {a.tolist()}")
    print(f"b:        {b.tolist()}")
    print(f"c = a+b:  {c.tolist()}")
    print(f"d = c*2:  {result}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(r - e) < 1e-5 for r, e in zip(result, expected))}")


def example_activations():
    """Activation functions on MLX device."""
    print("\n" + "=" * 50)
    print("Example 2: Activation Functions")
    print("=" * 50)

    x = Tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=TensorOpsDevice.APPLE)

    # Different activations
    y_tanh = x.tanh()
    y_relu = x.relu()
    y_sigmoid = x.sigmoid()

    with TensorContext(device=TensorOpsDevice.APPLE) as ctx:
        ctx.add_op(y_tanh)
        ctx.add_op(y_relu)
        ctx.add_op(y_sigmoid)
        ctx.forward()

    print(f"Input:    {x.tolist()}")
    print(f"tanh(x):  {y_tanh.tolist()}")
    print(f"relu(x):  {y_relu.tolist()}")
    print(f"sigmoid:  {y_sigmoid.tolist()}")


def example_matmul():
    """Matrix multiplication with MLX."""
    print("\n" + "=" * 50)
    print("Example 3: Matrix Multiplication")
    print("=" * 50)

    # Create 2x3 and 3x2 matrices
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=TensorOpsDevice.APPLE)
    a.shape = (2, 3)

    b = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], device=TensorOpsDevice.APPLE)
    b.shape = (3, 2)

    # Matrix multiplication
    c = a @ b

    with TensorContext(device=TensorOpsDevice.APPLE) as ctx:
        ctx.add_op(c)
        ctx.forward()

    result = c.tolist(shaped=True)
    # Manual computation:
    # [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
    # [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
    expected = [[58.0, 64.0], [139.0, 154.0]]

    print(f"A shape: {a.shape}")
    print(f"A:\n{a.numpy().reshape(a.shape)}")
    print(f"\nB shape: {b.shape}")
    print(f"B:\n{b.numpy().reshape(b.shape)}")
    print("\nC = A @ B:")
    print(f"{np.array(result)}")
    print("\nExpected:")
    print(f"{np.array(expected)}")


def example_chained_ops():
    """Chained operations on MLX."""
    print("\n" + "=" * 50)
    print("Example 4: Chained Operations")
    print("=" * 50)

    x = Tensor([0.0, 1.0, 2.0], device=TensorOpsDevice.APPLE)

    # Build complex expression: (x + 2) * exp(x) / (x + 1)
    one = Tensor(1.0, device=TensorOpsDevice.APPLE)
    two = Tensor(2.0, device=TensorOpsDevice.APPLE)

    numerator = (x + two) * x.exp()
    denominator = x + one

    # Handle division by adding small epsilon for safety
    eps = Tensor(1e-7, device=TensorOpsDevice.APPLE)
    denominator = denominator + eps

    y = numerator / denominator

    with TensorContext(device=TensorOpsDevice.APPLE) as ctx:
        ctx.add_op(y)
        ctx.forward()

    result = y.tolist()
    print(f"x:           {x.tolist()}")
    print(f"(x+2)*exp(x)/(x+1): {result}")

    # Verify a few values manually
    x_vals = [0.0, 1.0, 2.0]
    manual = [(val + 2) * np.exp(val) / (val + 1) for val in x_vals]
    print(f"Manual calc: {manual}")


def example_device_prop():
    """Verify device propagation through graph."""
    print("\n" + "=" * 50)
    print("Example 5: Device Propagation")
    print("=" * 50)

    # Create tensors on APPLE device
    a = Tensor([1.0, 2.0], device=TensorOpsDevice.APPLE)
    b = Tensor([3.0, 4.0], device=TensorOpsDevice.APPLE)

    # Operations should inherit device from operands
    c = a + b
    d = c.relu()
    e = d * 2.0

    print(f"a.device: {a.device}")
    print(f"b.device: {b.device}")
    print(f"c.device: {c.device}")
    print(f"d.device: {d.device}")
    print(f"e.device: {e.device}")

    # Execute context should use device
    with TensorContext(device=TensorOpsDevice.APPLE) as ctx:
        ctx.add_op(e)
        print(f"Context device: {ctx.device}")
        ctx.forward()

    print(f"Result: {e.tolist()}")


def example_runtime_selection():
    """Show automatic runtime selection."""
    print("\n" + "=" * 50)
    print("Example 6: Runtime Selection")
    print("=" * 50)

    from tensorops import get_runtime_for_device

    # Get runtimes for different devices
    apple_runtime = get_runtime_for_device(TensorOpsDevice.APPLE)
    opencl_runtime = get_runtime_for_device(TensorOpsDevice.OPENCL)

    print(f"APPLE device runtime: {type(apple_runtime).__name__}")
    print(f"OPENCL device runtime: {type(opencl_runtime).__name__}")

    # Create tensors and let framework choose runtime
    a = Tensor([1.0, 2.0, 3.0], device=TensorOpsDevice.APPLE)
    b = a + 1.0

    with TensorContext(device=TensorOpsDevice.APPLE) as ctx:
        ctx.add_op(b)
        print(f"Executing on: {get_runtime_for_device(ctx.device).__class__.__name__}")
        ctx.forward()

    print(f"Result: {b.tolist()}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TensorOps MLX Kernel Mapping Examples")
    print("=" * 60)
    print(f"Platform: {platform.system()}")
    print(f"MLX installed: {True}")

    try:
        example_basic_ops()
        example_activations()
        example_matmul()
        example_chained_ops()
        example_device_prop()
        example_runtime_selection()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
