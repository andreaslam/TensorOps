from __future__ import annotations
from typing import Any
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul, xor
import hip_cpu_bindings

# TODO
# impl back pass (from node)
# impl forward backend, graph optim and rewrite, cannonicalisation
# impl backward backend, kernel allocation
# impl nn essentials, eg softmax, softplus, sigmoid, max, min, argmax, sum, ones, zeros, repeats


class Tensor(ABC):
    def __init__(
        self,
        values,
        requires_grad: bool = True,
        is_op: bool = False,
        weight: bool = False,
    ) -> None:
        self.weight = weight
        assert xor(bool(values), is_op), "Must be either a valued Tensor or an OP"
        self.is_op = is_op

        if values:
            if isinstance(values, (float, int)):
                self.values = [values]
            elif isinstance(values, list):
                self.values = values
            else:
                raise ValueError("Invalid subtype inside list!")
        else:
            assert isinstance(
                self, OP
            ), "Tensor must either have value or is an instance of an OP class!"
            # OP can init with no values or grad but needs shape
            self.values = None

        self.requires_grad = requires_grad

        if self.values:
            self.shape = tuple(hip_cpu_bindings.get_shape(self.values))
            if len(self.shape) == 1:
                self.flat = self.values
            else:
                self.flat = None
        else:
            self.flat = None
            self.shape = None
        if self.weight:
            self.requires_grad = True

        self.grads = None

    def reshape(self, shape) -> ShapeOP:
        # support -1 reshaping (unknown)
        shape = (shape,) if isinstance(shape, int) else shape
        assert (
            count := shape.count(-1)
        ) <= 1, f"cannot reshape tensor to shape {shape}"
        assert len(self) % abs(reduce(mul, shape)) == 0, f"invalid shape {shape}"
        dim_size = len(self) // abs(reduce(mul, shape))
        if count == 1:
            modified_shape = list(shape)
            modified_shape[modified_shape.index(-1)] = dim_size
            shape = tuple(modified_shape)
        return ShapeOP(self, shape)
        

    def flatten(self) -> None:
        self.shape = (reduce(mul, self.shape),)

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
        return ElementMul(self, -1)

    def __truediv__(self, other) -> Div:
        return Div(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rtruediv__(self, other) -> Div:
        return Div(other if isinstance(other, Tensor) else Tensor(other), self)

    def __matmul__(self, other) -> MatMul:
        return MatMul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rmatmul__(self, other) -> MatMul:
        return MatMul(other if isinstance(other, Tensor) else Tensor(other), self)

    def __pow__(self, other) -> Pow:
        return Pow(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rpow__(self, other) -> Pow:
        return Pow(other if isinstance(other, Tensor) else Tensor(other), self)

    def exp(self) -> Exp:
        return Exp(self)

    def sin(self) -> Sin:
        return Sin(self)

    def cos(self) -> Cos:
        return Cos(self)

    def tanh(self) -> Tanh:
        return Tanh(self)

    def relu(self) -> ReLU:
        return ReLU(self)

    def leaky_relu(self) -> LeakyReLU:
        return LeakyReLU(self)

    def seed_grad(self, seed: int) -> None:
        assert self.values, "Cannot seed gradient, the valued tensor must not be empty!"
        self.grads = [seed] * len(self.values)

    def squeeze(self):
        assert self.shape[0] == 1, f"Cannot squeeze tensor shaped {self.shape}"
        modify_shape = list(self.shape)
        modify_shape.remove(0)
        self.shape = tuple(modify_shape)

    def unsqueeze(self, dim):
        assert dim >= -1 and dim <= len(
            self.shape
        ), f"Cannot unsqueeze tensor shaped {self.shape} at dim {dim}, expected values from [-1,{len(self.shape)}]"
        modify_shape = list(self.shape)
        modify_shape.insert(dim, 1)
        self.shape = tuple(modify_shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, values={self.values}, requires_grad={self.requires_grad}, weight={self.weight})"

    def __len__(self) -> int:
        return len(self.values) if self.flat else reduce(mul, self.shape)

    def __list__(self) -> list:
        return self.values if self.values else []

    def __getitem__(self, idx):
        return self.values[idx]

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
            for i in range(reduce(mul, shape) if len(shape) == 2 else shape[0] ** 2)
        ]
    )
    t.reshape(shape if len(shape) == 2 else (shape[0], shape[0]))
    return t


class OP(Tensor):
    def __init__(self, operands, requires_grad, weight) -> None:
        self.parents = [operands]
        self.fuse = False
        needs_grad = [x.requires_grad for x in operands]
        fuse_condition = (
            needs_grad == [False, False] or needs_grad == [False]
        ) and all([x.values for x in operands])
        super().__init__(
            None,
            requires_grad=requires_grad if not fuse_condition else False,
            weight=weight,
            is_op=True,
        )
        if fuse_condition and TensorContext.current_context is not None:
            self.fuse = True
            print("fuse now")
        else:
            if TensorContext.current_context is not None:
                TensorContext.current_context.add_op(self)
                if operands not in TensorContext.current_context.operands:
                    TensorContext.current_context.add_operands(operands)

    @abstractmethod
    def compute(self, reshape=False) -> None:
        ...

    # @abstractmethod
    # def get_grad(self):
    #     ...

    def __repr__(self) -> str:
        display = (
            f"values={self.values}, requires_grad={self.requires_grad}, weight={self.weight}, self.shape={self.shape}"
            if self.values
            else f"operands={self.parents}, self.fuse={self.fuse}, requires_grad={self.requires_grad}, self.weight={self.weight}, self.shape={self.shape}"
        )
        return f"{type(self).__name__}({display})"

class ShapeOP(OP):
    def __init__(self, tensor1, new_shape) -> None:
        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.shape = new_shape
        self.tensor1 = tensor1

    def compute(self, reshape=False) -> None:
        self.values = self.tensor1.values


class BinaryOP(OP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(
            [tensor1, tensor2],
            False if not tensor1.requires_grad and tensor2.requires_grad else True,
            False,
        )
        self.tensor1 = tensor1
        self.tensor2 = tensor2

        self.parents = [self.tensor1, self.tensor2]

        original_shape1 = self.tensor1.shape
        original_shape2 = self.tensor2.shape
        t1_len = len(self.tensor1)
        t2_len = len(self.tensor2)
        self.broadcast = (t1_len != t2_len) and xor(t1_len == 1, t2_len == 1)
        assert (t1_len == t2_len) or (
            self.broadcast
        ), f"Tensor lengths must match! Got {t1_len} and {t2_len}"

        if original_shape1 != original_shape2:
            self.shape = self.tensor1.shape
        else:
            self.shape = original_shape1

        if self.broadcast:
            shape_copy = max(self.tensor1, self.tensor2, key=lambda x: len(x))
            broadcasted = min(self.tensor1, self.tensor2, key=lambda x: len(x))
            broadcasted.values *= max(t1_len, t2_len)
            self.shape = shape_copy.shape
        if self.fuse:
            self.compute()


class Add(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        if not self.tensor2.flat:
            self.tensor2.flat = hip_cpu_bindings.flatten_list(self.tensor2.values)
            self.tensor2.flattened = True
        self.values = hip_cpu_bindings.run_vector_add(
            self.tensor1.flat, self.tensor2.flat
        )
        if reshape:
            self.reshape(self.shape)


class Sub(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        if not self.tensor2.flat:
            self.tensor2.flat = hip_cpu_bindings.flatten_list(self.tensor2.values)
            self.tensor2.flattened = True
        self.values = hip_cpu_bindings.run_vector_sub(
            self.tensor1.flat, self.tensor2.flat
        )
        if reshape:
            self.reshape(self.shape)


class ElementMul(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        if not self.tensor2.flat:
            self.tensor2.flat = hip_cpu_bindings.flatten_list(self.tensor2.values)
            self.tensor2.flattened = True
        self.values = hip_cpu_bindings.run_vector_element_mul(
            self.tensor1.flat, self.tensor2.flat
        )
        if reshape:
            self.reshape(self.shape)


class Div(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        if not self.tensor2.flat:
            self.tensor2.flat = hip_cpu_bindings.flatten_list(self.tensor2.values)
            self.tensor2.flattened = True
        self.values = hip_cpu_bindings.run_vector_div(
            self.tensor1.flat, self.tensor2.flat
        )
        if reshape:
            self.reshape(self.shape)


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

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        if not self.tensor2.flat:
            self.tensor2.flat = hip_cpu_bindings.flatten_list(self.tensor2.values)
            self.tensor2.flattened = True
        self.values = hip_cpu_bindings.run_gemm(
            self.tensor1.flat,
            self.tensor2.flat,
            self.m,
            self.n,
            self.k,
            1.0,
            0.0,
        )
        if reshape:
            self.reshape(self.shape)


class Pow(BinaryOP):
    def __init__(self, tensor1, tensor2) -> None:
        super().__init__(tensor1, tensor2)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        if not self.tensor2.flat:
            self.tensor2.flat = hip_cpu_bindings.flatten_list(self.tensor2.values)
            self.tensor2.flattened = True
        self.values = hip_cpu_bindings.run_vector_pow(
            self.tensor1.flat, self.tensor2.flat
        )
        if reshape:
            self.reshape(self.shape)


class UnaryOP(OP):
    def __init__(self, tensor1) -> None:
        super().__init__([tensor1], True if tensor1.requires_grad else False, False)
        self.tensor1 = tensor1
        self.shape = self.tensor1.shape
        self.parents = [self.tensor1]
        if self.fuse:
            self.compute()


class Cos(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        self.values = hip_cpu_bindings.run_vector_cos(self.tensor1.flat)
        if reshape:
            self.reshape(self.shape)


class Exp(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        self.values = hip_cpu_bindings.run_vector_exp(self.tensor1.flat)
        if reshape:
            self.reshape(self.shape)


class Sin(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        self.values = hip_cpu_bindings.run_vector_sin(self.tensor1.flat)
        if reshape:
            self.reshape(self.shape)


class Tanh(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        self.values = hip_cpu_bindings.run_vector_tanh(self.tensor1.flat)
        if reshape:
            self.reshape(self.shape)


class ReLU(UnaryOP):
    def __init__(
        self,
        tensor1,
    ) -> None:
        super().__init__(tensor1)

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        self.values = hip_cpu_bindings.run_vector_relu(self.tensor1.flat)
        if reshape:
            self.reshape(self.shape)


class LeakyReLU(UnaryOP):
    def __init__(self, tensor1, leaky_grad=0.01) -> None:
        super().__init__(tensor1)
        self.leaky_grad = leaky_grad

    def compute(self, reshape=False) -> None:
        if not self.tensor1.flat:
            self.tensor1.flat = hip_cpu_bindings.flatten_list(self.tensor1.values)
            self.tensor1.flattened = True
        self.values = hip_cpu_bindings.run_vector_leakyrelu(
            self.tensor1.flat, self.leaky_grad
        )
        if reshape:
            self.reshape(self.shape)


def relu(x) -> ReLU:
    return ReLU(x)


def leaky_relu(x, leaky_grad=0.01) -> LeakyReLU:
    return LeakyReLU(x, leaky_grad=leaky_grad)


def forward(ops):
    for op in ops[:-1]:
        op.compute()
    ops[-1].compute(True)


def backward(ops):
    ops[-1].seed_grad(1)
    for op in ops[::-1]:
        if op.requires_grad:
            op.get_grad()


class TensorContext:
    """
    `tensorops.TensorContext` manages the operational context for `Tensor`s during computation.
    """

    current_context = None

    def __init__(self) -> None:
        self.ops = []
        self.operands = []

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


def visualise_graph(nodes, save_img=True, img_path="graph.png", display=True) -> None:
    G = nx.DiGraph()
    labels = {}
    for node in nodes:
        node_id = id(node)
        node_label = f"{type(node).__name__}"
        labels[node_id] = node_label
        G.add_node(node_id)
        for parent in node.parents:
            parent_id = id(parent)
            G.add_edge(parent_id, node_id)

    pos = nx.planar_layout(G)
    colourmap = [
        "#FFB6C1" if node.weight else "#00B4D9" if node.requires_grad else "#C1E1C1"
        for (_, node) in zip(G, nodes)
    ]
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=800,
        node_color=colourmap,
        font_size=6,
    )
    if save_img:
        plt.savefig(img_path)
    if display:
        plt.show()


class Optim:
    def __init__(
        self, lr=1e-3, maximise: bool = False, weight_decay: float = 0.0
    ) -> None:
        self.lr = lr
        self.maximise = maximise
        self.weight_decay = weight_decay

    def step(self) -> None:
        pass

    def save(self, path: str) -> None:
        """
        Saves the optimiser to a `.pkl` file.

        Args:
            path (str): The file path where the optimiser should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> Any:
        """
        Loads an optimiser from a `.pkl` file.

        Args:
            path (str): The file path from which to load the optimiser.

        Returns:
            Optim: The loaded optimiser.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class Adam(Optim):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        maximise: bool = False,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0.0 for param in parameters}
        self.v = {param: 0.0 for param in parameters}
        self.eps = eps
        self.amsgrad = amsgrad
        self.v_hat_max = {param: 0.0 for param in parameters}
        hip_cpu_bindings.init_adam()

    def step(self) -> None:
        hip_cpu_bindings.run_adam()


class AdamW(Optim):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        maximise: bool = False,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0.0 for param in parameters}
        self.v = {param: 0.0 for param in parameters}
        self.eps = eps
        self.amsgrad = amsgrad
        self.v_hat_max = {param: 0.0 for param in parameters}
        hip_cpu_bindings.init_adamw()

    def step(self) -> None:
        hip_cpu_bindings.run_adamw()


class SGD(Optim):
    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        maximise: bool = False,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: int = 0,
        momentum: int = 0,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.dampening = dampening
        self.nesterov = nesterov
        self.momentum = momentum
        self.b_t = {param: 0 for param in parameters}
        hip_cpu_bindings.init_sgd()

    def step(self) -> None:
        hip_cpu_bindings.run_sgd()
