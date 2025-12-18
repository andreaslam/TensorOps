from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul, xor
from typing import Optional

import tensorops_backend

from tensorops import rt

# TODO
# impl nn essentials, eg softmax, softplus, sigmoid, argmax
# docstrings


class Tensor(ABC):
    def __init__(
        self,
        values,
        enable_fusion: bool = True,
        requires_grad: bool = True,
        is_op: bool = False,
        weight: bool = False,
        grad_tensor: bool = False,
    ) -> None:
        self.weight = weight
        self.grad_tensor = grad_tensor
        has_value = values is not None
        assert xor(has_value, is_op), (
            f"Must be either a valued Tensor or an OP, got {values}"
        )
        self.is_op = is_op
        self.enable_fusion = enable_fusion
        # user decides whether the tensor can be fused or not, for example, if the user wants to see the value of a tensor, since the value of the tensor might not be guaranteed to be present due to kernel fusion not returning intermediate results. this guarantees that the tensor value is returned
        self.available_during_exec = False
        self.memview = None
        if values is not None:
            if isinstance(values, (float, int)):
                self._values = [values]
            elif isinstance(values, list):
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
            self.memview, self._shape = tensorops_backend.tensor_from_list(self._values)
            self._flat = list(self.memview)
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

    @property
    def flat(self):
        if self.is_op and self._pending_kernel_result is not None:
            # Trigger lazy materialization on first access
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
                    "Cannot materialize pending results without an active TensorContext"
                )

            cached = ctx._kernel_val_cache.get(id(kernel_result))
            if cached is None:
                cached = kernel_result.val
                ctx._kernel_val_cache[id(kernel_result)] = cached

            result = cached[op_idx]

            self._flat = result
            self.memview = None
            self._values = result

            self._pending_kernel_result = None
            self._pending_kernel_op_index = None
            self._pending_ctx = None
        return self._values

    @values.setter
    def values(self, new_value):
        if new_value is None:
            self._values = None
            self._flat = None
            self.memview = None
            return

        # Fast path: backend outputs are already flat lists. If we already know
        # the semantic shape, avoid calling tensor_from_list() (which is expensive
        # for large buffers) just to re-infer a 1D shape.
        if self.shape is not None and isinstance(new_value, list):
            expected = reduce(mul, self.shape, 1)
            if expected != len(new_value):
                raise ValueError(
                    f"Value length {len(new_value)} does not match tensor shape {self.shape} (expected {expected})"
                )
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
            if expected != len(new_value):
                raise ValueError(
                    f"Value length {len(new_value)} does not match tensor shape {self.shape} (expected {expected})"
                )

        self._flat = list(memview)
        self.memview = memview
        self._values = new_value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        if new_shape and self.shape:
            _ = _check_shape(self.shape, new_shape)
        self._shape = new_shape
        self.capacity = self.capacity if self.capacity else reduce(mul, new_shape, 1)

    def _alloc_zero_grads(self) -> Tensor:
        cap = self.capacity
        if cap is None:
            if self.shape is not None:
                cap = reduce(mul, self.shape, 1)
            elif self.values is not None:
                cap = len(self.values)
            else:
                raise ValueError("Cannot allocate grads for tensor with unknown size")

        g = Tensor([0.0] * int(cap), requires_grad=False, grad_tensor=True)
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
            return result_data[0]

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
        result = Tensor(result_data, requires_grad=False)
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
            return result_data[0]

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
        result = Tensor(result_data, requires_grad=False)
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
            raise ValueError("Cannot expand a tensor without values")

        # Rust backend path
        if hasattr(tensorops_backend, "tensor_expand"):
            new_values = tensorops_backend.tensor_expand(  # type: ignore[attr-defined]
                list(data), list(current_shape), list(shape)
            )
            return Tensor(new_values, requires_grad=self.requires_grad).reshape(shape)

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

        return Tensor(new_values, requires_grad=self.requires_grad).reshape(shape)

    def permute(self, dims):
        assert len(dims) == len(self.shape), "Permute dims must match tensor dims"
        new_shape = tuple([self.shape[d] for d in dims])
        new_capacity = self.capacity
        new_values = [0.0] * new_capacity

        src_strides = [1] * len(self.shape)
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            src_strides[i] = stride
            stride *= self.shape[i]

        tgt_strides = [1] * len(new_shape)
        stride = 1
        for i in range(len(new_shape) - 1, -1, -1):
            tgt_strides[i] = stride
            stride *= new_shape[i]

        if self.values:
            for i in range(new_capacity):
                idx = i
                src_flat_idx = 0
                for dim_idx in range(len(new_shape)):
                    coord = (idx // tgt_strides[dim_idx]) % new_shape[dim_idx]
                    src_dim = dims[dim_idx]
                    src_flat_idx += coord * src_strides[src_dim]

                new_values[i] = self.flat[src_flat_idx]

        return Tensor(new_values).reshape(new_shape)

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

    def __add__(self, other) -> Add:
        return Add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __radd__(self, other) -> Add:
        return Add(other if isinstance(other, Tensor) else Tensor(other), self)

    def __sub__(self, other) -> Sub:
        return Sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rsub__(self, other) -> Sub:
        return Sub(other if isinstance(other, Tensor) else Tensor(other), self)

    def __mul__(self, other) -> ElementMul:
        return ElementMul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rmul__(self, other) -> ElementMul:
        return ElementMul(other if isinstance(other, Tensor) else Tensor(other), self)

    def __neg__(self) -> ElementMul:
        return ElementMul(self, Tensor(-1, requires_grad=False))

    def __truediv__(self, other) -> Div:
        return Div(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rtruediv__(self, other) -> Div:
        return Div(other if isinstance(other, Tensor) else Tensor(other), self)

    def __matmul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)

        shape_a = self.shape
        shape_b = other.shape

        assert len(shape_a) >= 2 and len(shape_b) >= 2

        M = shape_a[-2]
        K = shape_a[-1]
        K2 = shape_b[-2]
        N = shape_b[-1]

        assert K == K2, f"MatMul shape mismatch: {shape_a} vs {shape_b}"

        # Reshape A to (..., M, 1, K)
        new_shape_a = list(shape_a[:-2]) + [M, 1, K]
        a_reshaped = self.reshape(tuple(new_shape_a))

        # Permute B to (..., N, K)
        # B is (..., K, N). Permute last two dims.
        perm_b = list(range(len(shape_b)))
        perm_b[-1], perm_b[-2] = perm_b[-2], perm_b[-1]
        b_permuted = other.permute(perm_b)

        # Reshape B to (..., 1, N, K)
        new_shape_b = list(shape_b[:-2]) + [1, N, K]
        b_reshaped = b_permuted.reshape(tuple(new_shape_b))

        # Expand A to (..., M, N, K)
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

        target_shape = list(batch_dims) + [M, N, K]

        a_expanded = a_reshaped.expand(tuple(target_shape))
        b_expanded = b_reshaped.expand(tuple(target_shape))

        # Element-wise Mul
        prod = a_expanded * b_expanded

        # Sum over last axis (K)
        res = prod.sum(axis=-1)

        return res

    def __rmatmul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__matmul__(self)

    def __pow__(self, other) -> Pow:
        return Pow(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rpow__(self, other) -> Pow:
        return Pow(other if isinstance(other, Tensor) else Tensor(other), self)

    def exp(self) -> Pow:
        return Pow(Tensor(math.e, requires_grad=False), self)

    def log(self, base: float | Tensor = math.e):
        return GenericLog(base if isinstance(base, Tensor) else Tensor(base), self)

    def log10(self):
        return GenericLog(Tensor(10, requires_grad=False), self)

    def log2(self):
        return GenericLog(Tensor(2, requires_grad=False), self)

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

    def seed_grad(self, seed: int) -> None:
        # Seed gradients without forcing value materialization.
        cap = self.capacity
        if cap is None:
            vals = self.values
            if not vals:
                raise ValueError(
                    f"Cannot seed gradient, the tensor must not be empty! {self}"
                )
            cap = len(vals)
        self.grads = Tensor([seed] * cap, requires_grad=False, grad_tensor=True)
        if self.shape is not None:
            self.grads.shape = self.shape

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
        # Avoid forcing lazy materialization.
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


class Repeat(Tensor):
    def __init__(self, val, shape) -> None:
        super().__init__(val)
        self.shape = shape

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, values={self.values})"


# Helper Functions for Tensors
def repeat(val, shape):
    return Repeat(val, shape)


def zeros(shape):
    return Repeat(0.0, shape)


def ones(shape):
    return Repeat(1.0, shape)


def eye(shape):
    assert len(shape) == 2 or len(shape) == 1, "shape must be 2D or 1D"
    if len(shape) == 1:
        n = shape[0]
        rows, cols = n, n
    else:
        rows, cols = shape

    cap = rows * cols
    flat = [1.0 if (i % cols) == (i // cols) else 0.0 for i in range(cap)]
    t = Tensor(flat, requires_grad=False)
    t.reshape((rows, cols))
    return t


class OP(Tensor):
    def __init__(self, operands, requires_grad, weight) -> None:
        self.parents = [operands]
        self.parent_data_tensors = all(parent.values for parent in operands)

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
        self.axis = axis

        shape = tensor1.shape
        if axis < 0:
            axis += len(shape)
        assert 0 <= axis < len(shape), f"Axis {axis} out of bounds for shape {shape}"

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
            Tensor([float(self.pre_axis)], requires_grad=False),
            Tensor([float(self.axis_len)], requires_grad=False),
            Tensor([float(self.post_axis)], requires_grad=False),
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
        mask_flat = [0.0] * self.tensor1.capacity
        x = self.tensor1.flat
        # Output is laid out as [pre_axis, post_axis]
        for pre in range(self.pre_axis):
            for post in range(self.post_axis):
                base = (pre * self.axis_len) * self.post_axis + post
                best = x[base]
                for k in range(1, self.axis_len):
                    v = x[base + k * self.post_axis]
                    if v > best:
                        best = v
                for k in range(self.axis_len):
                    idx = base + k * self.post_axis
                    if x[idx] == best:
                        mask_flat[idx] = 1.0

        mask = Tensor(mask_flat, requires_grad=False).reshape(self.tensor1.shape)
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

        # Build a mask where input equals the reduced min value.
        mask_flat = [0.0] * self.tensor1.capacity
        x = self.tensor1.flat
        for pre in range(self.pre_axis):
            for post in range(self.post_axis):
                base = (pre * self.axis_len) * self.post_axis + post
                best = x[base]
                for k in range(1, self.axis_len):
                    v = x[base + k * self.post_axis]
                    if v < best:
                        best = v
                for k in range(self.axis_len):
                    idx = base + k * self.post_axis
                    if x[idx] == best:
                        mask_flat[idx] = 1.0

        mask = Tensor(mask_flat, requires_grad=False).reshape(self.tensor1.shape)
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

        if original_shape1 != original_shape2:
            self.shape = self.tensor1.shape
        else:
            self.shape = original_shape1

        if self.broadcast:
            shape_copy = max(self.tensor1, self.tensor2, key=lambda x: len(x))
            broadcasted = min(self.tensor1, self.tensor2, key=lambda x: len(x))

            # Broadcast a scalar (len==1) to match the other operand without
            # going through the `values` setter (which preserves existing shape).
            if broadcasted.values is not None and len(broadcasted) == 1:
                target_len = max(t1_len, t2_len)
                broadcasted._values = broadcasted.values * target_len
                broadcasted.memview, _ = tensorops_backend.tensor_from_list(
                    broadcasted._values
                )
                broadcasted.flat = list(broadcasted.memview)
                broadcasted._shape = shape_copy.shape
                broadcasted.capacity = reduce(mul, broadcasted._shape, 1)
            self.shape = shape_copy.shape

        self.grads = None


class Add(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads)
        self.tensor2.add_grad(self.grads)


class Sub(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads)
        self.tensor2.add_grad(self.grads * -1)


class ElementMul(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads * self.tensor2)
        self.tensor2.add_grad(self.grads * self.tensor1)


class Div(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.add_grad(self.grads * 1 / self.tensor2)
        self.tensor2.add_grad(self.grads * -self.tensor1 / (self.tensor2**2))


class Pow(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self):
        self.tensor1.add_grad(
            self.grads * self.tensor2 * (self.tensor1 ** (self.tensor2 - 1))
        )
        self.tensor2.add_grad(
            self.grads * ((self.tensor1**self.tensor2) * self.tensor1.log())
        )


class GenericLog(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)
        # tensor1 is base (scalar or single-element tensor), tensor2 is input
        assert len(self.tensor1) == 1, (
            "Log base must be a scalar (single-element tensor)"
        )
        self.base_value = self.tensor1
        self.scalar_operands = [self.base_value]

    def get_grad(self):
        self.tensor1.add_grad(
            self.grads
            * (-(self.tensor2.log() / (self.tensor1 * ((self.tensor1.log()) ** 2))))
        )
        self.tensor2.add_grad(self.grads * (1 / (self.tensor2 * self.tensor1.log())))


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
            else Tensor(leaky_grad, requires_grad=False)
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

        if self.tensor1.flat is None:
            raise ValueError("LeakyReLU backward requires forward input values")

        alpha = (
            self.leaky_grad.flat[0]
            if self.leaky_grad.flat is not None
            else float(self.leaky_grad.values[0])
        )

        # d/dx leaky_relu(x) = 1 if x>0 else alpha
        scale = [1.0 if v > 0.0 else alpha for v in self.tensor1.flat]
        scale_t = Tensor(scale, requires_grad=False).reshape(self.tensor1.shape)
        self.tensor1.add_grad(self.grads * scale_t)


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

    def __init__(self) -> None:
        self.ops = []
        self.operands = []

        self.kernel_lookup = {}  # maps op object to index of the kernel it belongs to
        self.kernel_dependencies = {}
        self.kernel_inputs = {}

        self.kernels_objs = []
        self.kernels = []
        self.kernel_number = 0
        self.locked_kernels = []
        self.all_custom_instructions = []
        self.completed_kernels = []

        # Lazy result distribution cache: map KernelResult identity -> Python lists.
        # This avoids repeated expensive Rust->Python conversions for fused kernels.
        self._kernel_val_cache = {}

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
        from tensorops.backend import Fusor, Kernel

        def _unwrap_shapeop_parent(p):
            # ShapeOP is metadata-only; treat it as transparent for dependency tracking.
            while type(p).__name__ == "ShapeOP":
                p = getattr(p, "tensor1", p)
            return p

        for op in self.ops[lim:]:
            # ShapeOP is a view op (reshape). It has no kernel and should not
            # create a "locked" fusion boundary. Map it to its underlying
            # producer kernel (if any) and skip kernel creation.
            if type(op).__name__ == "ShapeOP":
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
                    parent_kernel_indices.add(self.kernel_lookup[p])
                else:
                    print(
                        f"  Warning: Parent {p} not found in kernel_lookup. Treating as external dependency."
                    )
                    parents_found_in_kernels = False
                    break

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

        # Build fast lookup maps for producing-kernel output indices.
        # This avoids repeated O(n) list.index() scans when wiring LogicalInputSource
        # dependencies inside Kernel.convert_kernel().
        self._kernel_output_index = [
            {op: idx for idx, op in enumerate(kernel_ops)}
            for kernel_ops in self.kernels
        ]

        # build custom instructions for fused kernels
        for i, k in zip(
            (
                range(lim_kernels, len(self.kernels))
                if lim_kernels
                else range(len(self.kernels))
            ),
            self.kernels[lim_kernels:],
        ):
            custom_instruction = None
            kernel_name = None
            if len(k) > 1:
                fusor = Fusor(k)
                custom_instruction, kernel_name = fusor.build_kernel()
                self.all_custom_instructions.append(custom_instruction)
            kernel = Kernel(
                [op for op in k],
                custom_instruction,
                i,
            )
            self.kernels_objs.append(kernel.convert_kernel(kernel_name))

    def forward(self):
        self.execute_ops()
        self.completed_kernels = [op for k in self.kernels for op in k]

    def backward(self):
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
        backward_kernel_start = len(self.kernels_objs)  # kernel index before backward
        for op in filter(lambda o: o.requires_grad, self.ops[::-1]):
            op.get_grad()

        # Backward is executed as a separate graph submission (execute_ops with lim_kernels).
        # That means backward kernels cannot reference forward kernel outputs via
        # LogicalInputSource (those kernels won't be re-executed). Materialize only the
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
        self.execute_ops(backward_graph_start, backward_kernel_start)

        self.ops = self.ops[:backward_graph_start]  # reset graph to pre-backward
        self.kernels_objs = self.kernels_objs[
            :backward_kernel_start
        ]  # reset graph to pre-backward

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
                            op._flat = result
                            op.memview = None
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
            for j, op in enumerate(kernel):
                # LAZY: store the KernelResult without calling .val.
                # Each op remembers which output index it needs.
                op._pending_kernel_result = kernel_result
                op._pending_kernel_op_index = j
                op._pending_ctx = self

    def execute_ops(self, lim=0, lim_kernels=0):
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
            res = rt.execute_graph(valid_kernels)
            # order of kernels returned from rt is the same as the order given to rt
            self.distribute_results(res, lim_kernels)
        finally:
            self._exec_lim_kernels = 0
