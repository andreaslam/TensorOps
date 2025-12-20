from __future__ import annotations

import math
import os
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul, xor
from typing import Optional, cast

import tensorops_backend

from tensorops import rt

# TODO
# impl nn essentials, eg softplus, argmax
# docstrings/


class Tensor(ABC):
    def __init__(
        self,
        values,
        enable_fusion: bool = True,
        requires_grad: bool = True,
        is_op: bool = False,
        weight: bool = False,
        grad_tensor: bool = False,
        device=None,
    ) -> None:
        from .device import TensorOpsDevice

        self.weight = weight
        self.grad_tensor = grad_tensor
        self.device = device or TensorOpsDevice.OPENCL  # Default to OpenCL
        has_value = values is not None
        assert xor(has_value, is_op), (
            f"Must be either a valued Tensor or an OP, got {values}"
        )
        self.is_op = is_op
        self.enable_fusion = enable_fusion
        # user decides whether the tensor can be fused or not, for example, if the user wants to see the value of a tensor, since the value of the tensor might not be guaranteed to be present due to kernel fusion not returning intermediate results. this guarantees that the tensor value is returned
        self.available_during_exec = False
        self.memview = None
        self._direct_input = None
        self._shape = None  # Initialize _shape early to avoid AttributeError
        if values is not None:
            if isinstance(values, (float, int)):
                self._values = [values]
            elif isinstance(values, list):
                self._values = values
            elif isinstance(values, (bytearray, memoryview, bytes)):
                self._values = values
            else:
                raise ValueError("Invalid subtype inside list!")
        else:
            assert isinstance(self, OP), (
                "Tensor must either have value or is an instance of an OP class!"
            )
            # OP can init with no values or grad but needs shape
            self._values = None
            self._shape = None

        self.requires_grad = requires_grad

        if self._values is not None:
            if isinstance(self._values, (bytearray, memoryview, bytes)):
                self.memview = self._values
                self._flat = memoryview(self.memview).cast("f")
            else:
                self.memview, inferred_shape = tensorops_backend.tensor_from_list(
                    self._values
                )
                self._shape = (
                    tuple(inferred_shape) if inferred_shape is not None else None
                )
                self._flat = memoryview(self.memview).cast("f")
        else:
            self._flat = None
        if self.weight:
            self.requires_grad = True
        self.capacity = reduce(mul, self._shape, 1) if self._shape is not None else None
        # Gradients are allocated lazily to reduce peak memory usage.
        self.grads = None

        # Lazy result distribution (set on OPs by TensorContext.distribute_results).
        self._pending_kernel_result = None
        self._pending_kernel_op_index = None
        self._pending_ctx = None

    def _get_direct_input(self):
        """Return a cached `tensorops_backend.DirectInput` for this tensor.

        Creating DirectInput copies the whole buffer into a Rust Vec<f32>.
        In tight recompute loops where kernels are rebuilt each iteration,
        caching avoids repeatedly copying the same large inputs.
        """

        if self._direct_input is not None:
            return self._direct_input

        data = self._flat if self._flat is not None else self._values
        if data is None:
            raise ValueError("Cannot create DirectInput for tensor with no host data")

        self._direct_input = tensorops_backend.DirectInput(data)
        return self._direct_input

    @property
    def flat(self):
        if self.is_op and self._pending_kernel_result is not None:
            # Trigger lazy materialisation on first access
            _ = self.values
        return self._flat

    @flat.setter
    def flat(self, value):
        self._flat = value

    @property
    def values(self):
        # Lazy distribution: populate this op if it has pending backend results.
        if self.is_op and self._pending_kernel_result is not None:
            ctx = self._pending_ctx or TensorContext.current_context
            kernel_result = self._pending_kernel_result
            op_idx = self._pending_kernel_op_index

            if op_idx is None:
                raise RuntimeError("Pending kernel op index is missing")

            if ctx is None:
                raise RuntimeError(
                    "Cannot materialise pending results without an active TensorContext"
                )

            cached = ctx._kernel_val_cache.get(id(kernel_result))
            if cached is None:
                cached = kernel_result.val
                ctx._kernel_val_cache[id(kernel_result)] = cached

            result = cached[op_idx]

            if isinstance(result, (bytearray, memoryview, bytes)):
                self.memview = result
                self._flat = memoryview(result).cast("f")
                self._values = result
            else:
                self._flat = result
                self.memview = None
                self._values = result

            self._pending_kernel_result = None
            self._pending_kernel_op_index = None
            self._pending_ctx = None
        return self._values

    @values.setter
    def values(self, new_value):
        # Any value assignment invalidates cached DirectInput.
        self._direct_input = None
        if new_value is None:
            self._values = None
            self._flat = None
            self.memview = None
            return

        # Handle binary data directly
        if isinstance(new_value, (bytearray, memoryview, bytes)):
            self.memview = new_value
            self._flat = memoryview(self.memview).cast("f")
            self._values = new_value
            if self.shape is None:
                self.shape = (len(self._flat),)
            else:
                expected = reduce(mul, self.shape, 1)
                if len(self._flat) != expected:
                    raise ValueError(
                        f"Value length {len(self._flat)} does not match tensor shape {self.shape} (expected {expected})"
                    )
            return

        # Fast path: backend outputs are already flat lists. If we already know
        # the semantic shape, avoid calling tensor_from_list() (which is expensive
        # for large buffers) just to re-infer a 1D shape.
        actual_count = None
        if self.shape is not None and isinstance(new_value, list):
            expected = reduce(mul, self.shape, 1)
            if new_value and isinstance(new_value[0], list):
                # For nested lists (e.g. batched data), validate total element count.
                actual_count = sum(len(row) for row in new_value)
            else:
                actual_count = len(new_value)

            if expected != actual_count:
                raise ValueError(
                    f"Value length {actual_count} does not match tensor shape {self.shape} (expected {expected})"
                )

            # Fast path for already-flat lists.
            if not new_value or not isinstance(new_value[0], list):
                self._flat = new_value
                self.memview = None
                self._values = new_value
                return

        memview, inferred_shape = tensorops_backend.tensor_from_list(new_value)

        # If this tensor already has a semantic shape (e.g. Sum / MatMul outputs,
        # reshape results, etc.), do NOT overwrite it with the backend-inferred
        # 1D shape coming from a flat list.
        if self.shape is None:
            self.shape = inferred_shape
        else:
            expected = reduce(mul, self.shape, 1)
            if actual_count is None:
                if (
                    isinstance(new_value, list)
                    and new_value
                    and isinstance(new_value[0], list)
                ):
                    actual_count = sum(len(row) for row in new_value)
                else:
                    actual_count = (
                        len(new_value) if isinstance(new_value, list) else expected
                    )
            if expected != actual_count:
                raise ValueError(
                    f"Value length {actual_count} does not match tensor shape {self.shape} (expected {expected})"
                )

        self._flat = memoryview(memview).cast("f")
        self.memview = memview
        self._values = new_value

    def item(self):
        if self.capacity != 1:
            raise ValueError(
                "only one element tensors can be converted to Python scalars"
            )
        data = self.flat if self.flat is not None else self.values
        if data is None:
            raise ValueError("Tensor has no values")
        if isinstance(data, (bytearray, memoryview, bytes)):
            mv = memoryview(data)
            if mv.format == "f":
                return mv[0]
            return mv.cast("f")[0]
        return data[0]

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        if new_shape and self.shape:
            _ = _check_shape(self.shape, new_shape)
        self._shape = tuple(new_shape) if new_shape is not None else None
        # Capacity must always match the shape's total element count.
        # Keeping a stale capacity causes subtle bugs (e.g. batched tensors
        # reporting the wrong flattened length during forward/backward).
        self.capacity = reduce(mul, self._shape, 1) if self._shape is not None else None

    def _alloc_zero_grads(self) -> Tensor:
        cap = self.capacity
        if cap is None:
            if self.shape is not None:
                cap = reduce(mul, self.shape, 1)
            elif self.values is not None:
                cap = len(self.values)
            else:
                raise ValueError("Cannot allocate grads for tensor with unknown size")

        g = Tensor(
            [0.0] * int(cap),
            requires_grad=False,
            grad_tensor=True,
            device=self.device,
        )
        if self.shape is not None:
            g.shape = self.shape
        return g

    def add_grad(self, grad: Tensor) -> None:
        if grad is None:
            return
        if self.grads is None:
            self.grads = grad
        else:
            self.grads = Add(self.grads, grad)

    def reshape(self, shape) -> ShapeOP:
        # support -1 reshaping (unknown)
        shape = (shape,) if isinstance(shape, int) else shape
        assert (count := shape.count(-1)) <= 1, (
            f"cannot reshape tensor to shape {shape}"
        )
        assert len(self) % abs(reduce(mul, shape)) == 0, f"invalid shape {shape}"
        dim_size = len(self) // abs(reduce(mul, shape))
        if count == 1:
            modified_shape = list(shape)
            modified_shape[modified_shape.index(-1)] = dim_size
            shape = tuple(modified_shape)
        return ShapeOP(self, shape)

    def sum(self, axis=None):
        if axis is None:
            # Flatten and sum
            return Sum(self.reshape((-1,)), 0)
        return Sum(self, axis)

    def max(self, axis: Optional[int] = None, keepdims: bool = False):
        data = self.flat if self.flat else self.values

        if not data:
            raise ValueError("Cannot compute max of empty tensor")

        result_data, result_shape = tensorops_backend.tensor_max(
            data, list(self.shape), axis
        )

        # If axis is None, return scalar
        if axis is None:
            return memoryview(result_data).cast("f")[0]

        # Handle keepdims
        if keepdims:
            # Insert dimension of size 1 at the reduced axis
            axis_idx = axis if axis >= 0 else len(self.shape) + axis
            result_shape = list(result_shape)
            result_shape.insert(axis_idx, 1)
            result_shape = tuple(result_shape)
        else:
            result_shape = tuple(result_shape)

        # Return as Tensor
        result = Tensor(result_data, requires_grad=False, device=self.device)
        result.shape = result_shape
        return result

    def min(self, axis: Optional[int] = None, keepdims: bool = False):
        data = self.flat if self.flat else self.values

        if not data:
            raise ValueError("Cannot compute min of empty tensor")

        result_data, result_shape = tensorops_backend.tensor_min(
            data, list(self.shape), axis
        )

        # If axis is None, return scalar
        if axis is None:
            return memoryview(result_data).cast("f")[0]

        # Handle keepdims
        if keepdims:
            # Insert dimension of size 1 at the reduced axis
            axis_idx = axis if axis >= 0 else len(self.shape) + axis
            result_shape = list(result_shape)
            result_shape.insert(axis_idx, 1)
            result_shape = tuple(result_shape)
        else:
            result_shape = tuple(result_shape)

        # Return as Tensor
        result = Tensor(result_data, requires_grad=False, device=self.device)
        result.shape = result_shape
        return result

    def expand(self, shape):
        # Broadcast/expand along singleton dimensions.
        # Fast-path uses Rust backend when available.
        current_shape = self.shape
        if current_shape is None:
            raise ValueError("Cannot expand a tensor with unknown shape")
        shape = tuple(shape)
        if len(shape) != len(current_shape):
            raise ValueError(
                "Expand requires same number of dimensions (use reshape/unsqueeze first)"
            )

        for s_curr, s_new in zip(current_shape, shape):
            if s_curr != 1 and s_curr != s_new:
                raise ValueError(f"Cannot expand {current_shape} to {shape}")

        if shape == current_shape:
            return self

        # Lazy path: if we're building a graph (or don't have values yet), defer
        # expansion until execution.
        if TensorContext.current_context is not None or (
            self.values is None and self.flat is None
        ):
            return ExpandOP(self, shape)

        data = self.flat if self.flat is not None else self.values
        if data is None:
            raise ValueError("Cannot expand tensor without values")

        # Fast path: run OpenCL expand kernel directly when not building a graph.
        # This avoids the CPU tensor_expand helper (and its host-side loops).
        try:
            template = tensorops_backend.get_kernel_source_by_name("VecExpandTemplate")
        except Exception:
            template = None

        if template:
            # Build row-major strides
            src_shape = list(current_shape)
            tgt_shape = list(shape)

            src_strides = [1] * len(src_shape)
            stride = 1
            for i in range(len(src_shape) - 1, -1, -1):
                src_strides[i] = stride
                stride *= int(src_shape[i])

            tgt_strides = [1] * len(tgt_shape)
            stride = 1
            for i in range(len(tgt_shape) - 1, -1, -1):
                tgt_strides[i] = stride
                stride *= int(tgt_shape[i])

            rank = len(tgt_shape)
            kernel_name = f"VecExpand_r{rank}"

            import re

            custom_src = re.sub(
                r"#define\s+RANK\s+\d+",
                f"#define RANK {rank}",
                template,
                count=1,
            )
            custom_src = re.sub(
                r"__kernel\s+void\s+VecExpandTemplate",
                f"__kernel void {kernel_name}",
                custom_src,
                count=1,
            )

            # Prefer passing a buffer-like object into DirectInput to hit the fast
            # buffer-protocol path in the Rust extension.
            src_buf = data
            if isinstance(src_buf, (bytearray, bytes)):
                src_buf = memoryview(src_buf).cast("f")

            k = tensorops_backend.KernelTensorOps(
                kernel_type=tensorops_backend.KernelType.custom(kernel_name),
                kernel_id=0,
                num_output_bufs=1,
                custom_kernel_src=custom_src,
                inputs=[
                    tensorops_backend.DirectInput(src_buf),
                    tensorops_backend.DirectInput([float(x) for x in src_shape]),
                    tensorops_backend.DirectInput([float(x) for x in src_strides]),
                    tensorops_backend.DirectInput([float(x) for x in tgt_shape]),
                    tensorops_backend.DirectInput([float(x) for x in tgt_strides]),
                ],
                scalar_inputs=None,
                work_size_override=int(reduce(mul, tgt_shape, 1)),
            )
            res = rt.execute_graph([k])
            out_bytes = res[0].val[0]
            out = Tensor(
                out_bytes, requires_grad=self.requires_grad, device=self.device
            )
            out.shape = tuple(tgt_shape)
            return out

        # Fallback Python implementation
        new_capacity = reduce(mul, shape, 1)
        new_values = [0.0] * new_capacity

        src_strides = [1] * len(current_shape)
        stride = 1
        for i in range(len(current_shape) - 1, -1, -1):
            src_strides[i] = stride
            stride *= current_shape[i]

        tgt_strides = [1] * len(shape)
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            tgt_strides[i] = stride
            stride *= shape[i]

        # Iterate over all elements in target
        # Note: This is slow for large tensors
        for i in range(new_capacity):
            src_flat_idx = 0
            for dim in range(len(shape)):
                coord = (i // tgt_strides[dim]) % shape[dim]
                src_coord = 0 if current_shape[dim] == 1 else coord
                src_flat_idx += src_coord * src_strides[dim]
            new_values[i] = data[src_flat_idx]

        return Tensor(
            new_values, requires_grad=self.requires_grad, device=self.device
        ).reshape(shape)

    def permute(self, dims):
        assert len(dims) == len(self.shape), "Permute dims must match tensor dims"
        return PermuteOP(self, dims)

    def save(self, path: str):
        """
        Saves a `tensorops.tensor.Tensor` to a `.pkl` file given a binary file `open()` handle.

        Passing an instance of the file handle would allow for repeated insertion and saving `tensor.tensor.Tensor` to a `.pkl` file

        Args:
            path (str): file path to save the pickle instance
        """

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """
        Loads a single `tensorops.tensor.Tensor()` or a generator of `tensorops.tensor.Tensor()`.

        Args:
            path (str): The file path from which to load the node(s).

        Returns:
            Tensor: The loaded tensor
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def flatten(self) -> None:
        self.shape = (self.capacity,)

    def _to_tensor_on_device(self, value, *, requires_grad: bool = False) -> Tensor:
        return (
            value
            if isinstance(value, Tensor)
            else Tensor(value, requires_grad=requires_grad, device=self.device)
        )

    def __add__(self, other) -> Add:
        return Add(self, self._to_tensor_on_device(other, requires_grad=False))

    def __radd__(self, other) -> Add:
        return Add(self._to_tensor_on_device(other, requires_grad=False), self)

    def __sub__(self, other) -> Sub:
        return Sub(self, self._to_tensor_on_device(other, requires_grad=False))

    def __rsub__(self, other) -> Sub:
        return Sub(self._to_tensor_on_device(other, requires_grad=False), self)

    def __mul__(self, other) -> ElementMul:
        return ElementMul(self, self._to_tensor_on_device(other, requires_grad=False))

    def __rmul__(self, other) -> ElementMul:
        return ElementMul(self._to_tensor_on_device(other, requires_grad=False), self)

    def __neg__(self) -> ElementMul:
        return ElementMul(self, self._to_tensor_on_device(-1, requires_grad=False))

    def __truediv__(self, other) -> Div:
        return Div(self, self._to_tensor_on_device(other, requires_grad=False))

    def __rtruediv__(self, other) -> Div:
        return Div(self._to_tensor_on_device(other, requires_grad=False), self)

    def __matmul__(self, other) -> Tensor:
        other = self._to_tensor_on_device(other)

        shape_a = self.shape
        shape_b = other.shape

        assert len(shape_a) >= 2 and len(shape_b) >= 2

        M = shape_a[-2]
        K = shape_a[-1]
        K2 = shape_b[-2]
        N = shape_b[-1]

        assert K == K2, f"MatMul shape mismatch: {shape_a} vs {shape_b}"

        # Handle batch broadcasting
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]

        # Simple batch broadcasting check (naive)
        if batch_a != batch_b:
            # If one is empty, use the other
            if not batch_a:
                batch_dims = batch_b
            elif not batch_b:
                batch_dims = batch_a
            else:
                # Assume they match for now or raise error
                assert batch_a == batch_b, "Batch dims must match for now"
                batch_dims = batch_a
        else:
            batch_dims = batch_a

        # Broadcast A
        if len(batch_a) < len(batch_dims):
            # Prepend 1s
            diff = len(batch_dims) - len(batch_a)
            new_shape = (1,) * diff + self.shape
            a_reshaped = self.reshape(new_shape)
            a_expanded = a_reshaped.expand(tuple(list(batch_dims) + [M, K]))
        elif batch_a != batch_dims:
            # Same rank but different sizes (e.g. (1, M, K) vs (10, M, K))
            a_expanded = self.expand(tuple(list(batch_dims) + [M, K]))
        else:
            a_expanded = self

        # Broadcast B
        if len(batch_b) < len(batch_dims):
            diff = len(batch_dims) - len(batch_b)
            new_shape = (1,) * diff + other.shape
            b_reshaped = other.reshape(new_shape)
            b_expanded = b_reshaped.expand(tuple(list(batch_dims) + [K, N]))
        elif batch_b != batch_dims:
            b_expanded = other.expand(tuple(list(batch_dims) + [K, N]))
        else:
            b_expanded = other

        return MatMul(a_expanded, b_expanded)

    def __rmatmul__(self, other) -> Tensor:
        other = self._to_tensor_on_device(other)
        return other.__matmul__(self)

    def __pow__(self, other) -> Pow:
        return Pow(self, self._to_tensor_on_device(other, requires_grad=False))

    def __rpow__(self, other) -> Pow:
        return Pow(self._to_tensor_on_device(other, requires_grad=False), self)

    def exp(self) -> Pow:
        return Pow(self._to_tensor_on_device(math.e, requires_grad=False), self)

    def log(self, base: float | Tensor = math.e):
        return GenericLog(
            base
            if isinstance(base, Tensor)
            else self._to_tensor_on_device(base, requires_grad=False),
            self,
        )

    def detach(self):
        return StopGrad(self)

    def log10(self):
        return GenericLog(self._to_tensor_on_device(10, requires_grad=False), self)

    def log2(self):
        return GenericLog(self._to_tensor_on_device(2, requires_grad=False), self)

    def sin(self) -> Sin:
        return Sin(self)

    def cos(self) -> Cos:
        return Cos(self)

    def tanh(self) -> Tanh:
        return Tanh(self)

    def relu(self) -> LeakyReLU:
        return LeakyReLU(self, 0.0)

    def leaky_relu(self, leaky_grad: float | Tensor | list = 0.01) -> LeakyReLU:
        return LeakyReLU(self, leaky_grad)

    def sigmoid(self):
        return 1 / (1 + math.e ** (-self))

    def softmax(self, axis: int = -1):
        """Numerically stable softmax along `axis`.

        Uses the standard max-shift trick for stability and preserves shape.
        """
        shape = self.shape
        assert shape is not None and len(shape) > 0, "Softmax requires known shape"
        if axis < 0:
            axis += len(shape)

        # Row-wise max for stability (reduce along axis)
        max_vals = Max(self, axis=axis).detach()
        # Reshape reduced tensor to insert singleton dim at `axis`, then expand
        reshaped = max_vals.reshape(
            tuple(list(shape[:axis]) + [1] + list(shape[axis + 1 :]))
        )
        max_expanded = reshaped.expand(shape)

        shifted = self - max_expanded
        exp_vals = shifted.exp()
        sum_exp = exp_vals.sum(axis=axis)
        sum_reshaped = sum_exp.reshape(
            tuple(list(shape[:axis]) + [1] + list(shape[axis + 1 :]))
        )
        sum_expanded = sum_reshaped.expand(shape)
        return exp_vals / sum_expanded

    def ramp(self):
        """Ramp (alias for ReLU)."""
        return self.relu()

    def softplus(self):
        """Softplus activation: log(1 + exp(x))."""
        return (self.exp() + Tensor(1.0, requires_grad=False, device=self.device)).log()

    def argmax(self, axis: Optional[int] = None, keepdims: bool = False):
        """Argmax along axis. Returns Python int when axis is None; otherwise a Tensor of indices."""
        data = self.flat if self.flat else self.values
        if not data:
            raise ValueError("Cannot compute argmax of empty tensor")

        if axis is None:
            # Global argmax index
            if isinstance(data, (bytearray, memoryview, bytes)):
                mv = self._flat
                if mv is None:
                    mv = memoryview(data)
                    mv = mv.cast("f") if mv.format != "f" else mv
                max_idx = max(range(len(mv)), key=lambda i: mv[i])
                return int(max_idx)
            max_idx = max(range(len(data)), key=lambda i: data[i])
            return int(max_idx)

        shape = self.shape
        if shape is None:
            raise ValueError("Argmax requires known shape")

        import numpy as np

        if isinstance(data, (bytearray, memoryview, bytes)):
            mv = self._flat
            if mv is None:
                mv = memoryview(data)
                mv = mv.cast("f") if mv.format != "f" else mv
            arr = np.frombuffer(cast(memoryview, mv), dtype=np.float32).reshape(shape)
        else:
            arr = np.asarray(data, dtype=np.float32).reshape(shape)

        idx = np.argmax(arr, axis=axis)
        if keepdims:
            idx = np.expand_dims(idx, axis=axis)
        idx_list = idx.tolist()
        out = Tensor(idx_list, requires_grad=False, device=self.device)
        out.shape = tuple(idx.shape)
        return out

    def to(self, device):
        """Transfer tensor to a different device.

        Currently a no-op for single-device execution.
        In multi-device setups, this would trigger data movement.
        """
        self.device = device
        return self

    def seed_grad(self, seed: int) -> None:
        # Seed gradients without forcing value materialisation.
        cap = self.capacity
        if cap is None:
            vals = self.values
            if not vals:
                raise ValueError(
                    f"Cannot seed gradient, the tensor must not be empty! {self}"
                )
            cap = len(vals)

        target_shape = tuple(self.shape) if self.shape is not None else (int(cap),)
        cap = int(cap)

        # Reuse existing grad tensor if it's already the right constant.
        if (
            self.grads is not None
            and getattr(self.grads, "capacity", None) == cap
            and getattr(self.grads, "shape", None) == target_shape
            and getattr(self.grads, "_seed_value", None) == seed
        ):
            return

        # Context-level cache to reuse large constant seed buffers across backward() calls.
        ctx = TensorContext.current_context
        cache = getattr(ctx, "_seed_grad_cache", None) if ctx is not None else None
        cache_key = (seed, target_shape)
        if cache is not None and cache_key in cache:
            self.grads = cache[cache_key]
            return

        # Fast path: allocate the constant buffer in Rust (no Python list, no recursive flatten).
        memview, _ = tensorops_backend.tensor_full(float(seed), list(target_shape))
        grad_t = Tensor(
            memview,
            requires_grad=False,
            grad_tensor=True,
            device=self.device,
        )
        grad_t.shape = target_shape
        grad_t._seed_value = seed
        self.grads = grad_t
        if cache is not None:
            cache[cache_key] = grad_t

    def squeeze(self):
        assert self.shape[0] == 1, f"Cannot squeeze tensor shaped {self.shape}"
        modify_shape = list(self.shape)
        modify_shape.remove(0)
        self.shape = tuple(modify_shape)

    def unsqueeze(self, dim):
        assert dim >= -1 and dim <= len(self.shape), (
            f"Cannot unsqueeze tensor shaped {self.shape} at dim {dim}, expected values from [-1,{len(self.shape)}]"
        )
        modify_shape = list(self.shape)
        modify_shape.insert(dim, 1)
        self.shape = tuple(modify_shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, values={self.values}, requires_grad={self.requires_grad}, weight={self.weight})"

    def __len__(self) -> int:
        # Avoid forcing lazy materialisation.
        if self._flat is not None:
            return len(self._flat)
        if self.capacity is not None:
            return self.capacity
        if self._values is not None:
            return len(self._values)
        return 0

    def __list__(self) -> list:
        return self.values if self.values else []

    def __getitem__(self, idx):
        idx = (idx,) if isinstance(idx, int) else idx
        assert len(idx) == len(self.shape), (
            "Index dimensions must match tensor dimensions"
        )

        flat_idx = 0
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            assert idx[i] < self.shape[i], (
                f"Index {idx[i]} exceeds tensor dimension {self.shape[i]}"
            )
            flat_idx += idx[i] * stride
            stride *= self.shape[i]

        return self.flat[flat_idx]

    def __iter__(self):
        return iter(self.values if self.values else [])

    def max_value(self):
        return max(self.flat if self.flat else self.values)

    def min_value(self):
        return min(self.flat if self.flat else self.values)

    # Ergonomic accessors
    def tolist(self, shaped: bool = True) -> list:
        """Return tensor contents as a Python list.

        - When `shaped=True` and `self.shape` is known, returns a nested list
          matching the tensor's shape.
        - When `shaped=False` or shape is unknown, returns a flat list.
        - Works transparently whether the underlying storage is a Python list
          or a bytes-like buffer (memoryview/bytearray/bytes).
        """
        # materialise pending backend results if needed
        data = self.flat if self.flat is not None else self.values
        if data is None:
            return []

        # Handle bytes-like efficiently
        if isinstance(data, (bytearray, memoryview, bytes)):
            mv = self._flat
            if mv is None:
                mv = memoryview(data)
                # Ensure float view for element access
                mv = mv.cast("f") if mv.format != "f" else mv

            if shaped and self.shape is not None:
                try:
                    import numpy as np  # local import to avoid hard dependency at module import

                    arr = np.frombuffer(cast(memoryview, mv), dtype=np.float32)
                    shaped_arr = arr.reshape(self.shape)
                    return cast(list, shaped_arr.tolist())
                except Exception:
                    # Fallback: flat list if numpy reshape fails/unavailable
                    pass
            return list(mv)

        # Python list path
        if shaped and self.shape is not None:
            try:
                import numpy as np

                arr = np.asarray(data, dtype=np.float32)
                return arr.reshape(self.shape).tolist()
            except Exception:
                pass
        return list(data)

    def numpy(self, copy: bool = False):
        """Return a NumPy array view of the tensor.

        - When `copy=False` and the tensor is backed by a bytes-like buffer,
          returns a zero-copy `numpy.frombuffer` view (fast, memory-efficient).
        - Otherwise returns a regular `numpy.array` copy.
        - Reshapes to `self.shape` when available.
        """
        data = self.flat if self.flat is not None else self.values
        if data is None:
            raise ValueError("Tensor has no values")

        import numpy as np

        if isinstance(data, (bytearray, memoryview, bytes)) and not copy:
            mv = self._flat
            if mv is None:
                mv = memoryview(data)
                mv = mv.cast("f") if mv.format != "f" else mv
            arr = np.frombuffer(cast(memoryview, mv), dtype=np.float32)
        else:
            if isinstance(data, (bytearray, memoryview, bytes)):
                # Ensure conversion to list for consistent dtype/shape handling
                base = list(memoryview(data).cast("f"))
            else:
                base = data
            arr = np.array(base, dtype=np.float32)

        if self.shape is not None:
            try:
                arr = arr.reshape(self.shape)
            except Exception:
                # If reshape fails, return flat array as a safe fallback
                pass
        return arr

    def head(self, n: int = 10) -> list:
        """Return the first `n` elements as a simple Python list.

        Useful for quick inspection without printing large buffers.
        Returns a flat list regardless of tensor shape.
        """
        data = self.flat if self.flat is not None else self.values
        if data is None:
            return []
        if isinstance(data, (bytearray, memoryview, bytes)):
            mv = self._flat
            if mv is None:
                mv = memoryview(data)
                mv = mv.cast("f") if mv.format != "f" else mv
            return list(mv[:n])
        return list(data[:n])

    def compute(self):
        """materialise this tensor's values without requiring an explicit TensorContext.

        Returns self for convenient chaining (e.g., `tensor.compute().tolist()`).
        """
        # Already materialised
        if self._values is not None or self._flat is not None:
            return self

        # Leaf tensor with no op
        if not getattr(self, "is_op", False):
            return self

        # If lazy backend results are attached, accessing `values` will populate buffers
        if getattr(self, "_pending_kernel_result", None) is not None:
            _ = self.values
            return self

        # Try current context if this op belongs to it
        ctx = TensorContext.current_context
        if ctx is not None:
            try:
                if self in ctx.ops:
                    ctx.forward(recompute=False)
                    _ = self.values
                    return self
            except Exception:
                pass

        # Build minimal temporary context: collect upstream ops and execute
        tmp_ctx = TensorContext(device=self.device)
        visited_ops = set()
        visited_leaves = set()

        def _collect_upstream(t):
            if not isinstance(t, OP):
                if t not in visited_leaves:
                    visited_leaves.add(t)
                    tmp_ctx.add_operands(t)
                return
            if t in visited_ops:
                return
            for p in getattr(t, "parents", []) or []:
                _collect_upstream(p)
            visited_ops.add(t)
            tmp_ctx.add_op(t)

        _collect_upstream(self)
        tmp_ctx.forward(recompute=False)
        _ = self.values
        return self


class Repeat(Tensor):
    def __init__(self, val, shape, device=None) -> None:
        super().__init__(val, device=device)
        self.shape = shape

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, values={self.values})"


# Helper Functions for Tensors
def repeat(val, shape, device=None):
    return Repeat(val, shape, device=device)


def zeros(shape, device=None):
    cap = reduce(mul, shape, 1)
    t = Tensor([0.0] * cap, requires_grad=False, device=device)
    t.shape = shape
    return t


def ones(shape, device=None):
    cap = reduce(mul, shape, 1)
    t = Tensor([1.0] * cap, requires_grad=False, device=device)
    t.shape = shape
    return t


def eye(shape, device=None):
    assert len(shape) == 2 or len(shape) == 1, "shape must be 2D or 1D"
    if len(shape) == 1:
        n = shape[0]
        rows, cols = n, n
    else:
        rows, cols = shape

    cap = rows * cols
    flat = [1.0 if (i % cols) == (i // cols) else 0.0 for i in range(cap)]
    t = Tensor(flat, requires_grad=False, device=device)
    t.reshape((rows, cols))
    return t


class OP(Tensor):
    def __init__(self, operands, requires_grad, weight) -> None:
        self.parents = [operands]
        self.parent_data_tensors = all(parent.values for parent in operands)

        device_candidates = {
            getattr(parent, "device", None) for parent in operands if parent is not None
        }
        device_candidates.discard(None)
        if len(device_candidates) > 1:
            device_list = ", ".join(sorted(str(d) for d in device_candidates))
            raise ValueError(f"{type(self).__name__} cannot mix devices: {device_list}")
        parent_device = next(iter(device_candidates), None)

        def _unwrap_shapeop(t: Tensor) -> Tensor:
            # ShapeOP is a view/metadata op; treat it as transparent for fusion decisions.
            while type(t).__name__ == "ShapeOP":
                t = getattr(t, "tensor1", t)
            return t

        self.fusable_op = all(
            _unwrap_shapeop(parent).enable_fusion for parent in operands
        )  # whether the kernel would exclude the op from being fused
        self.available_during_exec = False
        super().__init__(
            None,
            requires_grad=requires_grad,
            weight=weight,
            is_op=True,
            device=parent_device,
        )

        self.scalar_operands = []

        if TensorContext.current_context is not None:
            TensorContext.current_context.add_op(self)
            if operands not in TensorContext.current_context.operands:
                TensorContext.current_context.add_operands(operands)

    @abstractmethod
    def get_grad(self) -> None: ...

    def __repr__(self) -> str:
        display = f"requires_grad={self.requires_grad}, weight={self.weight}, self.shape={self.shape}"
        return f"{type(self).__name__}({display})"


class ShapeOP(OP):
    def __init__(self, tensor1, new_shape) -> None:
        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.parents = [tensor1]
        self.shape = new_shape
        self.tensor1 = tensor1
        # Shape/view ops must never be fused into arithmetic kernels.
        # They do not have a kernel snippet and are handled as cheap views.
        self.fusable_op = False
        # Don't copy values/flat here - let properties handle it dynamically
        self.capacity = self.tensor1.capacity

    @property
    def values(self):
        # Dynamically reference parent to support lazy loading
        return self.tensor1.values

    @values.setter
    def values(self, new_value):
        # ShapeOP doesn't own data, just redirect to parent
        self.tensor1.values = new_value

    @property
    def flat(self):
        # Dynamically reference parent to support lazy loading
        return self.tensor1.flat

    @flat.setter
    def flat(self, value):
        # ShapeOP doesn't own data, just redirect to parent
        self.tensor1.flat = value

    def get_grad(self) -> None:
        if self.requires_grad and self.tensor1.requires_grad:
            self.tensor1.add_grad(self.grads)


class StopGrad(OP):
    def __init__(self, tensor1) -> None:
        # Blocks gradient propagation; forwards values from parent.
        super().__init__([tensor1], False, False)
        self.parents = [tensor1]
        self.tensor1 = tensor1
        self.shape = tensor1.shape
        self.fusable_op = False
        self.capacity = tensor1.capacity

    @property
    def values(self):
        return self.tensor1.values

    @values.setter
    def values(self, new_value):
        self.tensor1.values = new_value

    @property
    def flat(self):
        return self.tensor1.flat

    @flat.setter
    def flat(self, value):
        self.tensor1.flat = value

    def get_grad(self) -> None:
        # Intentionally no-op: stop gradient.
        return


class ExpandOP(OP):
    def __init__(self, tensor1: Tensor, new_shape) -> None:
        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.parents = [tensor1]
        self.tensor1 = tensor1
        self.src_shape = tensor1.shape
        self.shape = tuple(new_shape)
        # Expand is executed as a dedicated custom OpenCL kernel; keep it out of fusion.
        self.fusable_op = False

    def get_grad(self) -> None:
        if not (self.requires_grad and self.tensor1.requires_grad):
            return

        src_shape = self.tensor1.shape
        tgt_shape = self.shape
        if src_shape is None or tgt_shape is None:
            raise ValueError("Expand backward requires known shapes")
        if len(src_shape) != len(tgt_shape):
            raise ValueError(
                f"Expand backward rank mismatch: {src_shape} vs {tgt_shape}"
            )

        reduced = self.grads
        # Reduce broadcasted axes (where src dim == 1 and tgt dim > 1)
        for axis in range(len(tgt_shape) - 1, -1, -1):
            if src_shape[axis] == 1 and tgt_shape[axis] > 1:
                reduced = reduced.sum(axis=axis)
                # Sum drops the axis; re-insert as size-1 to match src rank.
                new_shape = list(reduced.shape)
                new_shape.insert(axis, 1)
                reduced = reduced.reshape(tuple(new_shape))

        self.tensor1.add_grad(reduced)


class ReduceOP(OP):
    def __init__(self, tensor1, axis) -> None:
        self.tensor1 = tensor1

        shape = tensor1.shape
        if axis < 0:
            axis += len(shape)
        assert 0 <= axis < len(shape), f"Axis {axis} out of bounds for shape {shape}"

        # Store the normalised (non-negative) axis. Backward relies on this for
        # correct gradient reshaping/expansion.
        self.axis = axis

        self.axis_len = shape[axis]
        self.pre_axis = reduce(mul, shape[:axis], 1)
        self.post_axis = reduce(mul, shape[axis + 1 :], 1)

        new_shape = list(shape)
        new_shape.pop(axis)

        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.parents = [tensor1]

        # Reduce ops are not elementwise and cannot be fused by the current Fusor.
        self.fusable_op = False

        self.shape = tuple(new_shape)
        self.capacity = reduce(mul, self.shape, 1)
        self.grads = None

        # Scalar operands for the backend reduce kernels
        self.scalar_operands = [
            Tensor([float(self.pre_axis)], requires_grad=False, device=self.device),
            Tensor([float(self.axis_len)], requires_grad=False, device=self.device),
            Tensor([float(self.post_axis)], requires_grad=False, device=self.device),
        ]

    def _expanded_output_grads_to_input(self):
        grad_shape = list(self.shape)
        grad_shape.insert(self.axis, 1)
        reshaped_grads = self.grads.reshape(tuple(grad_shape))
        return reshaped_grads.expand(self.tensor1.shape)


class Sum(ReduceOP):
    def __init__(self, tensor1, axis) -> None:
        super().__init__(tensor1, axis)

    def get_grad(self) -> None:
        # Gradient of sum is broadcasting the grad to the input shape
        if self.requires_grad and self.tensor1.requires_grad:
            expanded_grads = self._expanded_output_grads_to_input()
            self.tensor1.add_grad(expanded_grads)


class Max(ReduceOP):
    def __init__(self, tensor1, axis) -> None:
        super().__init__(tensor1, axis)

    def get_grad(self) -> None:
        if not (self.requires_grad and self.tensor1.requires_grad):
            return

        if self.tensor1.flat is None or self.values is None:
            raise ValueError("Max backward requires forward values")

        expanded_grads = self._expanded_output_grads_to_input()

        # Build a mask where input equals the reduced max value.
        import numpy as np

        x = np.array(self.tensor1.flat).reshape(self.tensor1.shape)
        m = np.array(self.values).reshape(self.shape)

        # Expand m to match x's rank for comparison
        m_expanded = np.expand_dims(m, self.axis)
        mask_arr = (x == m_expanded).astype(np.float32)

        mask = Tensor(
            mask_arr.flatten().tolist(), requires_grad=False, device=self.device
        ).reshape(self.tensor1.shape)
        self.tensor1.add_grad(expanded_grads * mask)


class Min(ReduceOP):
    def __init__(self, tensor1, axis) -> None:
        super().__init__(tensor1, axis)

    def get_grad(self) -> None:
        if not (self.requires_grad and self.tensor1.requires_grad):
            return

        if self.tensor1.flat is None or self.values is None:
            raise ValueError("Min backward requires forward values")

        expanded_grads = self._expanded_output_grads_to_input()

        import numpy as np

        x = np.array(self.tensor1.flat).reshape(self.tensor1.shape)
        m = np.array(self.values).reshape(self.shape)

        m_expanded = np.expand_dims(m, self.axis)
        mask_arr = (x == m_expanded).astype(np.float32)

        mask = Tensor(
            mask_arr.flatten().tolist(), requires_grad=False, device=self.device
        ).reshape(self.tensor1.shape)
        self.tensor1.add_grad(expanded_grads * mask)


class BinaryOP(OP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(
            [tensor1, tensor2],
            False if not tensor1.requires_grad and not tensor2.requires_grad else True,
            False,
        )
        self.num_parents = 2
        self.tensor1 = tensor1
        self.tensor2 = tensor2

        self.parents = [self.tensor1, self.tensor2]

        original_shape1 = self.tensor1.shape
        original_shape2 = self.tensor2.shape
        t1_len = len(self.tensor1)
        t2_len = len(self.tensor2)
        self.broadcast = (t1_len != t2_len) and xor(t1_len == 1, t2_len == 1)
        assert (t1_len == t2_len) or (self.broadcast), (
            f"Tensor lengths must match! Got {t1_len} and {t2_len}"
        )

        # The current Fusor does not support scalar-broadcast indexing for external
        # inputs (len-1 tensors). Fusing such ops would generate `v_in[gid]` reads
        # and can go out-of-bounds, producing NaNs. Keep scalar-broadcast binary ops
        # as standalone predefined kernels (runtime passes lens as scalar args).
        if self.broadcast:
            self.fusable_op = False

        if original_shape1 != original_shape2:
            self.shape = self.tensor1.shape
        else:
            self.shape = original_shape1

        if self.broadcast:
            # Don't broadcast scalar tensors in place - this was causing issues where
            # scalar constants like Tensor(math.e) were being modified.
            # The backend will handle broadcasting during execution.
            shape_copy = max(self.tensor1, self.tensor2, key=lambda x: len(x))
            self.shape = shape_copy.shape

        self.grads = None


class Add(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        if self.tensor1.requires_grad:
            self.tensor1.add_grad(self.grads)
        if self.tensor2.requires_grad:
            self.tensor2.add_grad(self.grads)


class Sub(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        if self.tensor1.requires_grad:
            self.tensor1.add_grad(self.grads)
        if self.tensor2.requires_grad:
            self.tensor2.add_grad(self.grads * -1)


class ElementMul(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        if self.tensor1.requires_grad:
            self.tensor1.add_grad(self.grads * self.tensor2)
        if self.tensor2.requires_grad:
            self.tensor2.add_grad(self.grads * self.tensor1)


class Div(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        if self.tensor1.requires_grad:
            self.tensor1.add_grad(self.grads * 1 / self.tensor2)
        if self.tensor2.requires_grad:
            self.tensor2.add_grad(self.grads * -self.tensor1 / (self.tensor2**2))


class Pow(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)
        # Pow requires special kernel handling (e.g. scalar base broadcasting via
        # base_len) which the generic Fusor does not model.
        self.fusable_op = False

    def get_grad(self):
        if self.tensor1.requires_grad:
            self.tensor1.add_grad(
                self.grads * self.tensor2 * (self.tensor1 ** (self.tensor2 - 1))
            )
        if self.tensor2.requires_grad:
            self.tensor2.add_grad(
                self.grads * ((self.tensor1**self.tensor2) * self.tensor1.log())
            )


class GenericLog(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        # tensor1 is base (scalar or single-element tensor), tensor2 is input
        # Check before calling super().__init__ because BinaryOP will broadcast tensors
        assert len(tensor1) == 1, (
            f"Log base must be a scalar (single-element tensor), got length {len(tensor1)}"
        )
        super().__init__(tensor1, tensor2)
        self.base_value = self.tensor1
        self.scalar_operands = [self.base_value]
        # VecLog takes a scalar base argument; keep it out of fusion.
        self.fusable_op = False

    def get_grad(self):
        if self.tensor1.requires_grad:
            self.tensor1.add_grad(
                self.grads
                * (-(self.tensor2.log() / (self.tensor1 * ((self.tensor1.log()) ** 2))))
            )
        if self.tensor2.requires_grad:
            self.tensor2.add_grad(
                self.grads * (1 / (self.tensor2 * self.tensor1.log()))
            )


class UnaryOP(OP):
    def __init__(self, tensor1) -> None:
        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.num_parents = 1
        self.tensor1 = tensor1
        self.shape = self.tensor1.shape
        self.parents = [self.tensor1]

        self.grads = None


class Cos(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads * -self.tensor1.sin())


class PermuteOP(OP):
    def __init__(self, tensor1, dims) -> None:
        super().__init__([tensor1], tensor1.requires_grad, False)
        self.parents = [tensor1]
        self.tensor1 = tensor1
        self.dims = dims
        self.shape = tuple([tensor1.shape[d] for d in dims])
        self.fusable_op = False

    def get_grad(self) -> None:
        if self.requires_grad:
            # Inverse permutation
            inv_dims = [0] * len(self.dims)
            for i, d in enumerate(self.dims):
                inv_dims[d] = i
            self.tensor1.add_grad(self.grads.permute(tuple(inv_dims)))


class Sin(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads * self.tensor1.cos())


class Tanh(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads * (1 - (self.tensor1.tanh() ** 2)))


class LeakyReLU(OP):
    def __init__(self, tensor1, leaky_grad: float | Tensor | list = 0.01) -> None:
        alpha = (
            leaky_grad
            if isinstance(leaky_grad, Tensor)
            else Tensor(leaky_grad, requires_grad=False, device=tensor1.device)
        )
        alpha.requires_grad = False

        super().__init__(
            [tensor1, alpha], True if tensor1.requires_grad else False, False
        )
        self.num_parents = 2
        self.tensor1 = tensor1
        self.tensor2 = alpha
        self.parents = [self.tensor1, self.tensor2]
        self.shape = self.tensor1.shape
        self.grads = None

        assert len(self.tensor2) == 1, (
            "Leaky gradient must be a scalar (single-element tensor)"
        )
        self.leaky_grad = self.tensor2
        self.scalar_operands = [self.leaky_grad]

    def get_grad(self) -> None:
        if not (self.requires_grad and self.tensor1.requires_grad):
            return

        flat_vals = self.tensor1.flat
        # Some execution paths may leave `_flat` empty while the op still has
        # pending results; force materialisation if we expect data.
        if (flat_vals is None or flat_vals == []) and self.tensor1.capacity:
            _ = self.tensor1.values
            flat_vals = self.tensor1.flat

        alpha = (
            self.leaky_grad.flat[0]
            if self.leaky_grad.flat is not None
            else float(self.leaky_grad.values[0])
        )

        # If the pre-activation input was fused into an epilogue (e.g. MatMul+Bias+LeakyReLU),
        # `self.tensor1` may have no standalone buffer to materialise.
        # For alpha > 0, the sign of the LeakyReLU output matches the sign of the input,
        # so we can compute the gradient mask from the output instead.
        if flat_vals is None or flat_vals == []:
            if alpha <= 0.0:
                raise ValueError(
                    "LeakyReLU backward requires forward input values when alpha <= 0"
                )
            out_flat = self.flat
            if (out_flat is None or out_flat == []) and self.capacity:
                _ = self.values
                out_flat = self.flat
            if out_flat is None or out_flat == []:
                raise ValueError("LeakyReLU backward requires forward values")
            flat_vals = out_flat

        # d/dx leaky_relu(x) = 1 if x>0 else alpha
        scale = [1.0 if v > 0.0 else alpha for v in flat_vals]
        scale_t = Tensor(scale, requires_grad=False, device=self.device).reshape(
            self.tensor1.shape
        )
        self.tensor1.add_grad(self.grads * scale_t)


class MatMul(OP):
    def __init__(self, tensor1, tensor2) -> None:
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        super().__init__([tensor1, tensor2], requires_grad, False)
        self.fusable_op = False
        self.parents = [tensor1, tensor2]
        self.tensor1 = tensor1
        self.tensor2 = tensor2

        shape_a = tensor1.shape
        shape_b = tensor2.shape

        M = shape_a[-2]
        N = shape_b[-1]

        # Output shape: batch dims + [M, N]
        self.shape = tuple(list(shape_a[:-2]) + [M, N])

    def get_grad(self) -> None:
        if not self.requires_grad:
            return

        # C = A @ B
        # dA = dC @ B.T
        # dB = A.T @ dC

        if self.tensor1.requires_grad:
            # Permute B: (..., K, N) -> (..., N, K)
            perm_b = list(range(len(self.tensor2.shape)))
            perm_b[-1], perm_b[-2] = perm_b[-2], perm_b[-1]
            b_T = self.tensor2.permute(perm_b)

            grad_a = self.grads @ b_T
            self.tensor1.add_grad(grad_a)

        if self.tensor2.requires_grad:
            # Permute A: (..., M, K) -> (..., K, M)
            perm_a = list(range(len(self.tensor1.shape)))
            perm_a[-1], perm_a[-2] = perm_a[-2], perm_a[-1]
            a_T = self.tensor1.permute(perm_a)

            grad_b = a_T @ self.grads
            self.tensor2.add_grad(grad_b)


def relu(x) -> LeakyReLU:
    return LeakyReLU(x, 0.0)


def leaky_relu(x, leaky_grad=0.01) -> LeakyReLU:
    return LeakyReLU(x, leaky_grad=leaky_grad)


def _check_shape(target_shape, new_shape):
    new_shape = (new_shape,) if isinstance(new_shape, int) else new_shape
    assert (count := new_shape.count(-1)) <= 1, (
        f"cannot reshape tensor to shape {new_shape}"
    )
    flat_size = reduce(mul, target_shape)
    assert flat_size % abs(flat_size) == 0, f"invalid shape {new_shape}"
    return count


class TensorContext:
    """
    `tensorops.TensorContext` manages the operational context for `Tensor`s during computation.
    """

    current_context = None

    def __init__(self, device=None) -> None:
        from .device import TensorOpsDevice

        self.ops = []
        self.operands = []
        self.device = device or TensorOpsDevice.OPENCL  # Default to OpenCL

        self.kernel_lookup = {}  # maps op object to index of the kernel it belongs to
        self.kernel_dependencies = {}
        self.kernel_inputs = {}

        self.kernels_objs = []
        self.kernels = []
        self.kernel_number = 0
        self.locked_kernels = []
        self.all_custom_instructions = []
        self.completed_kernels = []

        # Track the portion of the graph that has already been finalised/executed.
        # This enables calling forward() multiple times on the same context without
        # re-emitting kernels with duplicate IDs.
        self._forward_op_cursor = 0
        self._forward_kernel_cursor = 0

        # Lazy result distribution cache: map KernelResult identity -> Python lists.
        # This avoids repeated expensive Rust->Python conversions for fused kernels.
        self._kernel_val_cache = {}

        # Cache constant seed grad tensors (e.g. ones_like(output)).
        # Keyed by (seed, shape_tuple).
        self._seed_grad_cache = {}

        # Kernel output index lookup built during finalise().
        self._kernel_output_index = []

        # Used by execute_ops() to limit LogicalInputSource wiring.
        self._exec_lim_kernels = 0

        # Used by __enter__/__exit__
        self.prev_context = None

    def reset(self, *, keep_weights: bool = False) -> None:
        """Clear the accumulated forward graph and execution state.

        This is useful when reusing a single TensorContext in a loop where new
        tensors/ops are created each iteration. Without resetting, `self.ops`
        grows unbounded and CPU RAM usage will climb.

        If `keep_weights=True`, ops with `weight=True` are preserved.
        """

        if keep_weights:
            self.ops = [op for op in self.ops if getattr(op, "weight", False)]
        else:
            self.ops = []
        self.operands = []

        # Reset caches (safe default; forward/backward tight loops typically use
        # reset_execution_state(), not reset()).
        self._kernel_val_cache = {}
        self._seed_grad_cache = {}

        self.kernel_lookup = {}
        self.kernel_dependencies = {}
        self.kernel_inputs = {}

        self.kernels_objs = []
        self.kernels = []
        self.kernel_number = 0
        self.locked_kernels = []
        self.all_custom_instructions = []
        self.completed_kernels = []

        self._forward_op_cursor = 0
        self._forward_kernel_cursor = 0
        self._kernel_val_cache = {}
        self._kernel_output_index = []
        self._exec_lim_kernels = 0

    def reset_execution_state(self) -> None:
        """Clear only execution/kernel state while keeping `self.ops` intact.

        This is used to re-run an already-built graph with updated input tensor
        values, without appending new ops (so memory stays bounded).
        """

        self.kernel_lookup = {}
        self.kernel_dependencies = {}
        self.kernel_inputs = {}

        self.kernels_objs = []
        self.kernels = []
        self.kernel_number = 0
        self.locked_kernels = []
        self.all_custom_instructions = []
        self.completed_kernels = []

        self._forward_op_cursor = 0
        self._forward_kernel_cursor = 0
        self._kernel_val_cache = {}
        self._kernel_output_index = []
        self._exec_lim_kernels = 0

    def zero_grads(self) -> None:
        """Clear stored gradients on all tensors/ops in this context.

        This is the equivalent of a typical framework's `zero_grad()`.
        It is important when calling `backward()` repeatedly in a loop;
        otherwise `.grads` can retain references to ops created during a
        previous backward pass (which have since been removed from `self.ops`).
        """

        seen: set[Tensor] = set()

        # Clear grads on ops and any parent tensors they reference.
        for op in self.ops:
            if isinstance(op, Tensor) and op not in seen:
                op.grads = None
                seen.add(op)
            if not isinstance(op, OP):
                continue
            for parent in getattr(op, "parents", []) or []:
                if isinstance(parent, Tensor) and parent not in seen:
                    parent.grads = None
                    seen.add(parent)

        # Also clear grads on tracked operands (covers leaf tensors like weights/inputs).
        for t in self.operands:
            if isinstance(t, Tensor) and t not in seen:
                t.grads = None
                seen.add(t)

    def __enter__(self):
        self.prev_context = TensorContext.current_context
        TensorContext.current_context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        TensorContext.current_context = self.prev_context

    def add_op(self, op) -> None:
        """
        Creates a node to be added to the computational graph stored in `tensorops.TensorContext.context`
        Args:
            op (tensorops.OP): The tensor operation to be added to the computational graph
        """
        self.ops.append(op)

    def weights_enabled(self):
        return [op for op in self.ops if op.weight]

    def grads_enabled(self):
        return [op for op in self.ops if op.grad_tensor]

    def get_flatten(self):
        return self.operands

    def add_operands(self, operand) -> None:
        self.operands.extend(operand if isinstance(operand, list) else [operand])

    def __repr__(self) -> str:
        return str(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def rewrite(self):
        pass

    def finalise(self, lim=0, lim_kernels=0):
        from tensorops.backend import AnchoredKernel, Fusor, Kernel

        def _unwrap_shapeop_parent(p):
            # ShapeOP is metadata-only; treat it as transparent for dependency tracking.
            while isinstance(p, ShapeOP) or type(p).__name__ == "StopGrad":
                p = getattr(p, "tensor1", p)
            return p

        for op in self.ops[lim:]:
            # ShapeOP/StopGrad are view/pass-through ops. They have no kernel and should not
            # create a "locked" fusion boundary. Map to underlying producer kernel and skip.
            if isinstance(op, ShapeOP) or type(op).__name__ == "StopGrad":
                parent = getattr(op, "tensor1", None)
                parent = _unwrap_shapeop_parent(parent)
                if isinstance(parent, OP) and parent in self.kernel_lookup:
                    self.kernel_lookup[op] = self.kernel_lookup[parent]
                op.available_during_exec = True
                continue

            op_parents = []
            for p in op.parents:
                if not isinstance(p, OP):
                    continue
                unwrapped = _unwrap_shapeop_parent(p)
                if isinstance(unwrapped, OP):
                    op_parents.append(unwrapped)

            parent_kernel_indices = set()
            parents_found_in_kernels = True
            for p in op_parents:
                if p in self.kernel_lookup:
                    parent_kernel_id = self.kernel_lookup[p]
                    # When executing a subset of kernels (e.g., backward()),
                    # parents produced by kernels outside this submission must be
                    # treated as external (host-materialised) dependencies.
                    if parent_kernel_id < lim_kernels:
                        parents_found_in_kernels = False
                        break
                    parent_kernel_indices.add(parent_kernel_id)
                else:
                    print(
                        f"  Warning: Parent {p} not found in kernel_lookup. Treating as external dependency."
                    )
                    parents_found_in_kernels = False
                    break

            # Check for MatMul epilogue fusion
            # We can only fuse into the latest kernel to avoid dependency issues (cannot depend on future kernels)
            is_matmul_epilogue = False
            target_kernel_idx = None

            if parent_kernel_indices:
                max_kernel_idx = max(parent_kernel_indices)
                if max_kernel_idx in self.locked_kernels:
                    kops = self.kernels[max_kernel_idx]
                    if type(kops[0]).__name__ == "MatMul":
                        op_name = type(op).__name__
                        if False and op_name in (
                            "Add",
                            "Sub",
                            "ElementMul",
                            "Div",
                            "Sin",
                            "Cos",
                            "Tanh",
                            "LeakyReLU",
                            "Pow",
                            "GenericLog",
                        ):
                            target_kernel_idx = max_kernel_idx
                            is_matmul_epilogue = True

            if is_matmul_epilogue:
                self.kernels[target_kernel_idx].append(op)
                self.kernel_lookup[op] = target_kernel_idx
                op.available_during_exec = True
                continue

            # 1. new kernel: If no OP parents OR parents belong to >1 kernel OR a parent wasn't tracked OR operation explicitly disables fusion
            if (
                not op_parents
                or not parents_found_in_kernels
                or len(parent_kernel_indices) > 1
                or not op.fusable_op
                or any(parent in self.completed_kernels for parent in op_parents)
            ):
                check = [
                    (
                        True
                        if p in (self.completed_kernels)
                        and (len(self.completed_kernels) != 0)
                        else False
                    )
                    for p in op_parents
                ]
                self.kernel_dependencies[self.kernel_number] = (
                    None
                    if (op.parent_data_tensors) or (all(check) and len(check) != 0)
                    else list(parent_kernel_indices)
                )

                self.kernel_inputs[self.kernel_number] = (
                    [parent.values for parent in op.parents]
                    if op.parent_data_tensors
                    else None
                )
                self.kernels.append([op])
                self.kernel_lookup[op] = self.kernel_number
                op.available_during_exec = (
                    True  # mark op's output as available after this new kernel runs
                )
                if not op.fusable_op:
                    self.locked_kernels.append(self.kernel_number)
                self.kernel_number += 1

            # 2. fuse Kernel: If all OP parents belong to the *same* single kernel.
            elif len(parent_kernel_indices) == 1:
                target_kernel_idx = parent_kernel_indices.pop()
                if target_kernel_idx not in self.locked_kernels:
                    self.kernels[target_kernel_idx].append(op)
                    self.kernel_lookup[op] = target_kernel_idx
                    op.available_during_exec = True
                else:
                    self.kernels.append([op])
                    self.kernel_lookup[op] = self.kernel_number
                    op.available_during_exec = True
                    self.kernel_number += 1

            else:
                self.kernels.append([op])
                self.kernel_lookup[op] = self.kernel_number
                op.available_during_exec = True
                self.kernel_number += 1

        # Build lookup maps from op -> output buffer index *as produced by the backend*.
        #
        # - Fused elementwise kernels (Fusor) emit one output buffer per op.
        # - Most single-op kernels emit a single output buffer (index 0).
        # - Anchored MatMul+epilogue kernels emit only the final op's output
        #   (index 0). Intermediate ops in that kernel do not have standalone
        #   outputs and must not be referenced via LogicalInputSource.
        self._kernel_output_index = []
        for kid, kops in enumerate(self.kernels):
            anchored = AnchoredKernel.try_build_from_ops(kops, kid)
            if anchored is not None:
                self._kernel_output_index.append({kops[-1]: 0})
            elif len(kops) > 1:
                self._kernel_output_index.append(
                    {op: idx for idx, op in enumerate(kops)}
                )
            else:
                self._kernel_output_index.append({kops[0]: 0})

        # build custom instructions for fused kernels
        for i, k in zip(
            range(lim_kernels, len(self.kernels)), self.kernels[lim_kernels:]
        ):
            custom_instruction = None
            kernel_name = None

            # Try anchored kernel pattern first (MatMul + epilogues)
            anchored = AnchoredKernel.try_build_from_ops(k, i)
            if anchored is not None:
                # MatMul with epilogue fusion
                kernel_result = anchored.convert_kernel(self)
                self.kernels_objs.append(kernel_result)
                continue

            # Otherwise, try regular fusion
            if len(k) > 1:
                fusor = Fusor(k)
                custom_instruction, kernel_name = fusor.build_kernel()
                self.all_custom_instructions.append(custom_instruction)

            kernel = Kernel(
                [op for op in k],
                custom_instruction,
                i,
            )
            self.kernels_objs.append(kernel.convert_kernel(kernel_name, self))

    def forward(self, recompute: bool = True):
        """Execute the forward graph.

        - When new ops have been added since the last call, only that new tail is
          finalised/executed (prevents duplicate kernel IDs).
        - When no new ops were added and `recompute=True`, the already-finalised
          kernels are re-executed. This enables tight loops that update input
          tensor `.values` without rebuilding the whole graph.

        Note: execution does not require being inside a `with TensorContext():`
        block; the context is temporarily set as current during execution.
        """

        prev_ctx = TensorContext.current_context
        TensorContext.current_context = self
        try:
            if (
                recompute
                and self._forward_op_cursor == len(self.ops)
                and self._forward_kernel_cursor == len(self.kernels_objs)
                and len(self.ops) > 0
            ):
                # Inputs for kernels are captured at kernel-conversion time
                # (DirectInput). If tensor values change, we must rebuild the
                # kernel objects to rebind inputs.
                self.reset_execution_state()
                self.execute_ops(0, 0)
                self.completed_kernels = [op for k in self.kernels for op in k]
                self._forward_op_cursor = len(self.ops)
                self._forward_kernel_cursor = len(self.kernels_objs)
                return

            # Execute only the new portion of the graph since the last forward().
            self.execute_ops(self._forward_op_cursor, self._forward_kernel_cursor)
            self.completed_kernels = [op for k in self.kernels for op in k]
            self._forward_op_cursor = len(self.ops)
            self._forward_kernel_cursor = len(self.kernels_objs)
        finally:
            TensorContext.current_context = prev_ctx

    def backward(self, *, accumulate: bool = False):
        prev_ctx = TensorContext.current_context
        TensorContext.current_context = self
        try:
            if not accumulate:
                self.zero_grads()

            debug_nonfinite = os.getenv("TENSOROPS_DEBUG_NONFINITE_GRADS") not in (
                None,
                "",
                "0",
            )

            def _tensor_has_nonfinite(t: Tensor) -> bool:
                src = t.flat if getattr(t, "flat", None) is not None else t.values
                if src is None:
                    return False
                if isinstance(src, memoryview):
                    if src.format != "f":
                        src = src.cast("f")
                    for v in src:
                        if math.isnan(v) or math.isinf(v):
                            return True
                    return False
                if isinstance(src, (bytearray, bytes)):
                    for v in memoryview(src).cast("f"):
                        if math.isnan(v) or math.isinf(v):
                            return True
                    return False
                for v in src:
                    if math.isnan(v) or math.isinf(v):
                        return True
                return False

            # Seed gradients for true graph outputs only.
            #
            # Important: when fusion is enabled, a single kernel can contain many ops.
            # Seeding *all* ops in a sink kernel (no downstream kernel deps) is wrong
            # and corrupts gradients. Instead, seed only ops that are not consumed by
            # any other op in the current (forward) graph.
            forward_ops = list(self.ops)
            consumed: set[OP] = set()
            for op in forward_ops:
                if not isinstance(op, OP):
                    continue
                for parent in getattr(op, "parents", []) or []:
                    if isinstance(parent, OP):
                        consumed.add(parent)

            output_ops = [
                op for op in forward_ops if isinstance(op, OP) and op not in consumed
            ]
            for op in output_ops:
                op.seed_grad(1)

            backward_graph_start = len(
                self.ops
            )  # use self.ops because this is where the ops accumulate during gradient graph building
            backward_operands_start = len(self.operands)
            backward_kernel_start = len(
                self.kernels_objs
            )  # kernel index before backward
            custom_instr_start = len(self.all_custom_instructions)
            for op in self.ops[::-1]:
                if not getattr(op, "requires_grad", False):
                    continue
                if getattr(op, "grads", None) is None:
                    continue
                op.get_grad()

                if debug_nonfinite and getattr(op, "grads", None) is not None:
                    if _tensor_has_nonfinite(op.grads):
                        raise ValueError(
                            f"Non-finite grads produced by {type(op).__name__} during backward()"
                        )

                if debug_nonfinite:
                    for parent in getattr(op, "parents", []) or []:
                        if not isinstance(parent, Tensor):
                            continue
                        if getattr(parent, "grads", None) is None:
                            continue
                        if _tensor_has_nonfinite(parent.grads):
                            raise ValueError(
                                f"Non-finite grads reached parent {type(parent).__name__} from {type(op).__name__}"
                            )

            # Backward is executed as a separate graph submission (execute_ops with lim_kernels).
            # That means backward kernels cannot reference forward kernel outputs via
            # LogicalInputSource (those kernels won't be re-executed). Materialise only the
            # forward tensors that are actually referenced by the backward graph.
            def _unwrap_shapeop_parent(p):
                while type(p).__name__ == "ShapeOP":
                    p = getattr(p, "tensor1", p)
                return p

            needed_forward_tensors: set[OP] = set()
            for bop in self.ops[backward_graph_start:]:
                if not isinstance(bop, OP):
                    continue
                for parent in getattr(bop, "parents", []) or []:
                    parent = _unwrap_shapeop_parent(parent)
                    if (
                        isinstance(parent, OP)
                        and getattr(parent, "_pending_kernel_result", None) is not None
                    ):
                        needed_forward_tensors.add(parent)

            for t in needed_forward_tensors:
                _ = t.values

            # Execute backward-only kernels.
            self.execute_ops(backward_graph_start, backward_kernel_start)

            if debug_nonfinite:
                # After backward kernels have executed, grads should be materialised.
                # Scan leaf tensors (operands) and any op parents for NaN/Inf.
                candidates: set[Tensor] = set()
                for t in self.operands:
                    if isinstance(t, Tensor):
                        candidates.add(t)
                for op2 in self.ops:
                    if not isinstance(op2, OP):
                        continue
                    for p in getattr(op2, "parents", []) or []:
                        if isinstance(p, Tensor):
                            candidates.add(p)

                for t in candidates:
                    g = getattr(t, "grads", None)
                    if g is None:
                        continue
                    # Force lazy materialisation if applicable.
                    if (
                        getattr(g, "is_op", False)
                        and getattr(g, "_pending_kernel_result", None) is not None
                    ):
                        _ = g.values
                    if _tensor_has_nonfinite(g):
                        raise ValueError(
                            f"Non-finite grads found after backward execution (tensor_shape={getattr(t, 'shape', None)})"
                        )

            # Restore state to pre-backward so repeated forward/backward cycles
            # don't accumulate stale kernels/lookups.
            backward_ops = self.ops[backward_graph_start:]
            self.ops = self.ops[:backward_graph_start]
            self.operands = self.operands[:backward_operands_start]
            self.kernels_objs = self.kernels_objs[:backward_kernel_start]
            self.kernels = self.kernels[:backward_kernel_start]
            self.kernel_number = backward_kernel_start

            # Remove kernel bookkeeping created during backward.
            for op in backward_ops:
                self.kernel_lookup.pop(op, None)
            for kid in list(self.kernel_dependencies.keys()):
                if kid >= backward_kernel_start:
                    self.kernel_dependencies.pop(kid, None)
            for kid in list(self.kernel_inputs.keys()):
                if kid >= backward_kernel_start:
                    self.kernel_inputs.pop(kid, None)

            self.locked_kernels = [
                k for k in self.locked_kernels if k < backward_kernel_start
            ]
            self.all_custom_instructions = self.all_custom_instructions[
                :custom_instr_start
            ]
            self._kernel_output_index = self._kernel_output_index[
                :backward_kernel_start
            ]
            # Backward submission results should not pollute forward cache.
            self._kernel_val_cache = {}

            # Keep forward cursors consistent with the restored (pre-backward) graph.
            self._forward_op_cursor = len(self.ops)
            self._forward_kernel_cursor = len(self.kernels_objs)
        finally:
            TensorContext.current_context = prev_ctx

    def distribute_results(self, execution_results, lim_kernels=0):
        res_iter = iter(execution_results)
        for i, kernel in enumerate(self.kernels[lim_kernels:]):
            k_obj = self.kernels_objs[lim_kernels + i]
            if k_obj is None:
                # ShapeOP and ExpandOP need immediate execution (they're cheap view ops)
                for op in kernel:
                    if type(op).__name__ == "ShapeOP":
                        # ShapeOP is a view; it forwards values/flat to its parent.
                        pass
                    elif type(op).__name__ == "ExpandOP":
                        parent = op.tensor1
                        if parent.values is None and parent.flat is None:
                            raise ValueError(
                                "ExpandOP execution requires parent values"
                            )
                        src_shape = list(parent.shape)
                        tgt_shape = list(op.shape)
                        data = parent.flat if parent.flat is not None else parent.values
                        if hasattr(tensorops_backend, "tensor_expand"):
                            result = tensorops_backend.tensor_expand(  # type: ignore[attr-defined]
                                list(data), src_shape, tgt_shape
                            )
                            # Fast path: bypass setter
                            op.memview = result
                            op._flat = memoryview(result).cast("f")
                            op._values = result
                        else:
                            # Python fallback if backend helper isn't available
                            new_capacity = reduce(mul, tgt_shape, 1)
                            new_values = [0.0] * new_capacity
                            src_strides = [1] * len(src_shape)
                            stride = 1
                            for j in range(len(src_shape) - 1, -1, -1):
                                src_strides[j] = stride
                                stride *= src_shape[j]
                            tgt_strides = [1] * len(tgt_shape)
                            stride = 1
                            for j in range(len(tgt_shape) - 1, -1, -1):
                                tgt_strides[j] = stride
                                stride *= tgt_shape[j]
                            for out_idx in range(new_capacity):
                                src_flat_idx = 0
                                for dim in range(len(tgt_shape)):
                                    coord = (out_idx // tgt_strides[dim]) % tgt_shape[
                                        dim
                                    ]
                                    src_coord = 0 if src_shape[dim] == 1 else coord
                                    src_flat_idx += src_coord * src_strides[dim]
                                new_values[out_idx] = data[src_flat_idx]
                            # Fast path: bypass setter
                            op._flat = new_values
                            op.memview = None
                            op._values = new_values
                continue

            kernel_result = next(res_iter)

            num_outputs = len(kernel_result.val)
            num_ops = len(kernel)

            if num_outputs == num_ops:
                for j, op in enumerate(kernel):
                    op._pending_kernel_result = kernel_result
                    op._pending_kernel_op_index = j
                    op._pending_ctx = self
            elif num_outputs == 1:
                # Assign to the last op (common for fusion)
                op = kernel[-1]
                op._pending_kernel_result = kernel_result
                op._pending_kernel_op_index = 0
                op._pending_ctx = self
            else:
                # Fallback: assign to last N ops
                for j in range(num_outputs):
                    op_idx = num_ops - num_outputs + j
                    op = kernel[op_idx]
                    op._pending_kernel_result = kernel_result
                    op._pending_kernel_op_index = j
                    op._pending_ctx = self

    def execute_ops(self, lim=0, lim_kernels=0):
        from . import get_runtime_for_device

        self.rewrite()
        # Let backend.convert_kernel know which kernel IDs will be executed in
        # this batch so it can avoid creating LogicalInputSource deps to kernels
        # outside the submitted set (e.g., forward kernels during backward()).
        self._exec_lim_kernels = lim_kernels
        try:
            self.finalise(lim, lim_kernels)
            valid_kernels = [
                k for k in self.kernels_objs[lim_kernels:] if k is not None
            ]
            # Select runtime based on context device
            runtime = get_runtime_for_device(self.device)
            res = runtime.execute_graph(valid_kernels)
            # order of kernels returned from rt is the same as the order given to rt
            self.distribute_results(res, lim_kernels)
        finally:
            self._exec_lim_kernels = 0
