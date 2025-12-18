from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from operator import mul, xor

import matplotlib.pyplot as plt
import networkx as nx
import tensorops_backend

from tensorops import rt

# TODO
# impl nn essentials, eg softmax, softplus, sigmoid, max, min, argmax, sum, ones, zeros, repeats
# docstrings


class Tensor(ABC):
    def __init__(
        self,
        values,
        enable_fusion: bool = False,
        requires_grad: bool = True,
        is_op: bool = False,
        weight: bool = False,
        grad_tensor: bool = False,
    ) -> None:
        self.weight = weight
        self.grad_tensor = grad_tensor
        assert xor(bool(values), is_op), (
            f"Must be either a valued Tensor or an OP, got {values}"
        )
        self.is_op = is_op
        self.enable_fusion = enable_fusion
        # user decides whether the tensor can be fused or not, for example, if the user wants to see the value of a tensor, since the value of the tensor might not be guaranteed to be present due to kernel fusion not returning intermediate results. this guarantees that the tensor value is returned
        self.available_during_exec = False
        self.memview = None
        if values:
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

        if self.values:
            self.memview, self._shape = tensorops_backend.tensor_from_list(self._values)
            self.flat = list(self.memview)
        else:
            self.flat = None
        if self.weight:
            self.requires_grad = True
        self.capacity = reduce(mul, self._shape) if self._shape else None
        if not self.grad_tensor:
            if self.capacity is not None:
                self.grads = Tensor(
                    [0.0] * self.capacity, requires_grad=False, grad_tensor=True
                )
            else:
                self.grads = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_value):
        if new_value is None:
            self._values = None
            self.flat = None
            self.memview = None
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

        self.flat = list(memview)
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
        self.capacity = self.capacity if self.capacity else reduce(mul, new_shape)

        # If this tensor was created lazily (e.g. an OP before execution) it may
        # not have had capacity available during __init__, so grads could be None.
        # Once shape/capacity exists, ensure grads are allocated for tensors that
        # participate in autograd.
        if (
            not getattr(self, "grad_tensor", False)
            and getattr(self, "requires_grad", False)
            and self.capacity is not None
            and getattr(self, "grads", None) is None
        ):
            self.grads = Tensor(
                [0.0] * self.capacity, requires_grad=False, grad_tensor=True
            )

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

    def expand(self, shape):
        # Naive implementation using list repetition
        current_shape = self.shape
        if len(shape) != len(current_shape):
            raise ValueError(
                "Expand requires same number of dimensions (use reshape/unsqueeze first)"
            )

        for s_curr, s_new in zip(current_shape, shape):
            if s_curr != 1 and s_curr != s_new:
                raise ValueError(f"Cannot expand {current_shape} to {shape}")

        if shape == current_shape:
            return self

        new_capacity = reduce(mul, shape, 1)
        new_values = [0.0] * new_capacity

        src_strides = [1] * len(self.shape)
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            src_strides[i] = stride
            stride *= self.shape[i]

        tgt_strides = [1] * len(shape)
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            tgt_strides[i] = stride
            stride *= shape[i]

        # Iterate over all elements in target
        # Note: This is slow for large tensors
        if self.values:
            for i in range(new_capacity):
                idx = i
                src_flat_idx = 0
                for dim in range(len(shape)):
                    coord = (idx // tgt_strides[dim]) % shape[dim]
                    if self.shape[dim] == 1:
                        src_coord = 0
                    else:
                        src_coord = coord
                    src_flat_idx += src_coord * src_strides[dim]

                new_values[i] = self.flat[src_flat_idx]

        return Tensor(new_values).reshape(shape)

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
        return LeakyReLU(self, [0.0])

    def leaky_relu(self, leaky_grad: float | Tensor | list = 0.01) -> LeakyReLU:
        return LeakyReLU(self, leaky_grad)

    def seed_grad(self, seed: int) -> None:
        assert self.values, (
            f"Cannot seed gradient, the valued tensor must not be empty! {self}"
        )
        self.grads = Tensor([seed] * len(self.values), requires_grad=False)

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
        return len(self.flat) if self.values else self.capacity

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

    def max(self):
        return max(self.flat if self.flat else self.values)

    def min(self):
        return max(self.flat if self.flat else self.values)


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
    t = Tensor(
        [
            1.0 if i % (shape[0]) == (i // shape[0]) else 0.0
            for i in range(self.capacity if len(shape) == 2 else shape[0] ** 2)
        ]
    )
    t.reshape(shape if len(shape) == 2 else (shape[0], shape[0]))
    return t


class OP(Tensor):
    def __init__(self, operands, requires_grad, weight) -> None:
        self.parents = [operands]
        self.parent_data_tensors = all(parent.values for parent in operands)
        self.fusable_op = all(
            parent.enable_fusion for parent in operands
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
        self.values = self.tensor1.values
        self.flat = self.tensor1.flat
        self.capacity = self.tensor1.capacity

    def get_grad(self) -> None:
        if self.requires_grad and self.tensor1.requires_grad:
            self.tensor1.grads = Add(self.tensor1.grads, self.grads)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_value):
        if new_value is None:
            self._values = None
            self.flat = None
            self.memview = None
            return

        memview, shape = tensorops_backend.tensor_from_list(new_value)
        self.flat = list(memview)
        self.memview = memview
        self._values = new_value


class Sum(OP):
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

        self.shape = tuple(new_shape)
        self.capacity = reduce(mul, self.shape, 1)
        self.grads = Tensor([0.0] * self.capacity, requires_grad=False)

        # Scalar operands for the kernel
        self.scalar_operands = [
            Tensor([float(self.pre_axis)], requires_grad=False),
            Tensor([float(self.axis_len)], requires_grad=False),
            Tensor([float(self.post_axis)], requires_grad=False),
        ]

    def get_grad(self) -> None:
        # Gradient of sum is broadcasting the grad to the input shape
        # We need to reshape grad to insert the axis back, then expand
        if self.requires_grad and self.tensor1.requires_grad:
            # Reshape grads to (..., 1, ...)
            grad_shape = list(self.shape)
            grad_shape.insert(self.axis, 1)
            reshaped_grads = self.grads.reshape(tuple(grad_shape))
            # Expand to input shape
            expanded_grads = reshaped_grads.expand(self.tensor1.shape)
            self.tensor1.grads += expanded_grads


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

        self.grads = Tensor([0.0] * reduce(mul, self.shape), requires_grad=False)


class Add(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads
        self.tensor2.grads += self.grads


class Sub(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads
        self.tensor2.grads += self.grads * -1


class ElementMul(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads * self.tensor2
        self.tensor2.grads += self.grads * self.tensor1


class Div(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads * 1 / self.tensor2
        self.tensor2.grads += self.grads * -self.tensor1 / (self.tensor2**2)


class MatMul(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

        tensor1_mk = self.tensor1.shape
        output_ndims = len(self.tensor1.shape)
        tensor2_mk = self.tensor2.shape

        assert (
            tensor1_mk[1] == tensor2_mk[0]
            and len(tensor1_mk) == 2
            and len(tensor2_mk) == 2
        ), f"Incorrect shape for MatMul, got {tensor1_mk} and {tensor2_mk}"
        self.m = tensor1_mk[0]
        self.n = tensor2_mk[1]
        self.k = tensor1_mk[1]
        self.tensor1.flat = self.tensor1.flatten()
        self.tensor2.flat = self.tensor2.flatten()

        self.shape = tuple(([1] * (output_ndims - 2)) + [self.m, self.n])
        self.grads = Tensor([0.0] * self.capacity, requires_grad=False)

    def get_grad(self):
        raise NotImplementedError


class Pow(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def get_grad(self):
        self.tensor1.grads += (
            self.grads * self.tensor2 * (self.tensor1 ** (self.tensor2 - 1))
        )
        self.tensor2.grads += self.grads * (
            (self.tensor1**self.tensor2) * self.tensor1.log()
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
        self.tensor1.grads += self.grads * (
            -(self.tensor2.log() / (self.tensor1 * ((self.tensor1.log()) ** 2)))
        )
        self.tensor2.grads += self.grads * (1 / (self.tensor2 * self.tensor1.log()))


class UnaryOP(OP):
    def __init__(self, tensor1) -> None:
        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.num_parents = 1
        self.tensor1 = tensor1
        self.shape = self.tensor1.shape
        self.parents = [self.tensor1]

        self.grads = Tensor([0.0] * self.capacity, requires_grad=False)


class Cos(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads * -self.tensor1.sin()


class Sin(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads * self.tensor1.cos()


class Tanh(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def get_grad(self) -> None:
        self.tensor1.grads += self.grads * (1 - (self.tensor1.tanh() ** 2))


class LeakyReLU(BinaryOP):
    def __init__(self, tensor1, leaky_grad: float | Tensor | list = 0.01) -> None:
        super().__init__(tensor1, Tensor(leaky_grad, requires_grad=False))
        assert len(self.tensor2) == 1, (
            "Leaky gradient must be a scalar (single-element tensor)"
        )
        self.leaky_grad = self.tensor2
        self.scalar_operands = [self.leaky_grad]

    def get_grad(self) -> None:
        raise NotImplementedError


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

        for op in self.ops[lim:]:
            op_parents = [p for p in op.parents if isinstance(p, OP)]

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
        # find all kernels that do not have dependencies associated with it
        deps = set(
            chain.from_iterable(v for v in self.kernel_dependencies.values() if v)
        )
        no_deps = set(range(self.kernel_number)).difference(deps)
        for i, k in enumerate(self.kernels):
            if i in no_deps:
                for op in k:
                    op.seed_grad(1.0)
        backward_graph_start = len(
            self.ops
        )  # use self.ops because this is where the ops accumulate during gradient graph building
        backward_kernel_start = len(self.kernels_objs)  # kernel index before backward
        for op in filter(lambda o: o.requires_grad, self.ops[::-1]):
            op.get_grad()
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
                for op in kernel:
                    if type(op).__name__ == "ShapeOP":
                        op.values = op.tensor1.values
                continue

            kernel_result = next(res_iter)
            for op_res, op in zip(kernel_result.val, kernel):
                op.values = op_res

    def execute_ops(self, lim=0, lim_kernels=0):
        self.rewrite()
        self.finalise(lim, lim_kernels)
        valid_kernels = [k for k in self.kernels_objs[lim_kernels:] if k is not None]
        res = rt.execute_graph(valid_kernels)
        # order of kernels returned from rt is the same as the order given to rt
        self.distribute_results(res, lim_kernels)


def visualise_graph(
    initial_nodes, save_img=True, img_path="graph.png", display=True
) -> None:
    """
    Visualizes an operator graph starting from a list of final (output) nodes.

    Args:
    -----
    initial_nodes (Union[list[Tensor], Tensor]): A list of Tensor/OP objects that are the final nodes of the graph to visualize. The graph is built by traversing backwards.
    save_img (bool): Whether to save the graph image to a file.
    img_path (str): Path to save the image.
    display (bool): Whether to display the graph using matplotlib.
    """
    G = nx.DiGraph()
    labels = {}

    all_nodes_map = {}

    if not initial_nodes:
        queue = []
    elif not isinstance(initial_nodes, list):
        queue = [initial_nodes]
    else:
        queue = list(initial_nodes)

    visited_ids = set()
    while queue:
        current_node = queue.pop(0)
        current_id = id(current_node)

        if current_id in visited_ids:
            continue
        visited_ids.add(current_id)
        all_nodes_map[current_id] = current_node

        if hasattr(current_node, "parents") and current_node.parents is not None:
            for parent_node in current_node.parents:
                if id(parent_node) not in all_nodes_map:
                    all_nodes_map[id(parent_node)] = parent_node
                if id(parent_node) not in visited_ids:
                    queue.append(parent_node)

    if not all_nodes_map:
        if display or save_img:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)
            if save_img:
                plt.savefig(img_path, bbox_inches="tight")
            if display:
                plt.show()
            plt.close()
        return

    for node_id, node_obj in all_nodes_map.items():
        G.add_node(node_id)
        label_text = type(node_obj).__name__
        if hasattr(node_obj, "shape") and node_obj.shape is not None:
            label_text += f"\nshape={node_obj.shape}"
        labels[node_id] = label_text

    for node_id, node_obj in all_nodes_map.items():
        if hasattr(node_obj, "parents") and node_obj.parents is not None:
            for parent_node in node_obj.parents:
                parent_id = id(parent_node)
                if parent_id in all_nodes_map:
                    G.add_edge(parent_id, node_id)

    if not G.nodes:
        if display or save_img:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)
            if save_img:
                plt.savefig(img_path, bbox_inches="tight")
            if display:
                plt.show()
            plt.close()
        return

    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        try:
            pos = nx.kamada_kawai_layout(G)
        except nx.NetworkXException:
            pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for node_id_in_graph in G.nodes():
        node_obj = all_nodes_map[node_id_in_graph]

        color = "#C1E1C1"
        if hasattr(node_obj, "weight") and node_obj.weight:
            color = "#FFB6C1"
        elif hasattr(node_obj, "requires_grad") and node_obj.requires_grad:
            color = "#00B4D9"
        node_colors.append(color)

    fig_width = max(10, G.number_of_nodes() * 0.8 if G.number_of_nodes() > 0 else 10)
    fig_height = max(8, G.number_of_nodes() * 0.6 if G.number_of_nodes() > 0 else 8)
    plt.figure(figsize=(fig_width, fig_height))

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=2500,
        node_color=node_colors,
        font_size=9,
        font_weight="normal",
        arrowsize=15,
        width=1.5,
    )

    if save_img:
        plt.savefig(img_path, bbox_inches="tight", dpi=150)
    if display:
        plt.show()
    plt.close()
