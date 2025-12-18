from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from operator import mul, xor
import math

import matplotlib.pyplot as plt
import networkx as nx


# TODO
# impl nn essentials, eg softmax, softplus, sigmoid, max, min, argmax, sum, ones, zeros, repeats
# docstrings


def py_flatten(lst):
    shape = []

    # Helper to determine shape recursively
    def get_shape(x):
        if isinstance(x, list):
            if x:
                return [len(x)] + get_shape(x[0])
            else:
                return [0]
        else:
            return []

    # Helper to flatten recursively
    def flatten(x):
        if isinstance(x, list):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    shape = get_shape(lst)
    flat_list = list(flatten(lst))
    return flat_list, shape

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
        self._shape = None
        self._values = None
        self.grad_tensor = grad_tensor
        assert xor(
            bool(values), is_op
        ), f"Must be either a valued Tensor or an OP, got {values}"
        self.is_op = is_op
        self.enable_fusion = enable_fusion
        # user decides whether the tensor can be fused or not, for example, if the user wants to see the value of a tensor, since the value of the tensor might not be guaranteed to be present due to kernel fusion not returning intermediate results. this guarantees that the tensor value is returned
        self.available_during_exec = False
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

        self.requires_grad = requires_grad

        if self.values:
            self.flat, self.shape = py_flatten(self._values)
        else:
            self.flat = None
        if self.weight:
            self.requires_grad = True
        self.capacity = reduce(mul, self._shape) if self._shape else None
        if not self.grad_tensor and self.capacity:
            self.grads = Tensor(
                [0.0] * self.capacity, requires_grad=False, grad_tensor=True
            )

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_value):
        self.flat, self.shape = py_flatten(self._values)
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

    def reshape(self, shape):
        # support -1 reshaping (unknown)
        count = _check_shape(tuple(self.shape), shape := shape)
        dim_size = len(self) // abs(self.capacity)
        if count == 1:
            modified_shape = list(shape)
            modified_shape[modified_shape.index(-1)] = dim_size
            shape = tuple(modified_shape)
        return self

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

    def __matmul__(self, other) -> MatMul:
        return MatMul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rmatmul__(self, other) -> MatMul:
        return MatMul(other if isinstance(other, Tensor) else Tensor(other), self)

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
        assert (
            self.values
        ), f"Cannot seed gradient, the valued tensor must not be empty! {self}"
        self.grads = Tensor([seed] * len(self.values), requires_grad=False)

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
        return len(self.flat) if self.values else self.capacity

    def __list__(self) -> list:
        return self.values if self.values else []

    def __getitem__(self, idx):
        idx = (idx,) if isinstance(idx, int) else idx
        assert len(idx) == len(
            self.shape
        ), "Index dimensions must match tensor dimensions"

        flat_idx = 0
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            assert (
                idx[i] < self.shape[i]
            ), f"Index {idx[i]} exceeds tensor dimension {self.shape[i]}"
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
    def get_grad(self) -> None:
        ...

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
        assert (
            len(self.tensor1) == 1
        ), "Log base must be a scalar (single-element tensor)"
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
        print("tanh", self.grads, self.grads.values)
        self.tensor1.grads += self.grads * (1 - (self.tensor1.tanh() ** 2))


class LeakyReLU(BinaryOP):
    def __init__(self, tensor1, leaky_grad: float | Tensor | list = 0.01) -> None:
        super().__init__(tensor1, Tensor(leaky_grad, requires_grad=False))
        assert (
            len(self.tensor2) == 1
        ), "Leaky gradient must be a scalar (single-element tensor)"
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
    assert (
        count := new_shape.count(-1)
    ) <= 1, f"cannot reshape tensor to shape {new_shape}"
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
                print(f"  Decision: Start New Kernel {self.kernel_number}")
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
                    print(f"  Decision: Fuse into Kernel {target_kernel_idx}")
                    self.kernels[target_kernel_idx].append(op)
                    self.kernel_lookup[
                        op
                    ] = target_kernel_idx 
                    op.available_during_exec = (
                        True 
                    )
                else:
                    print(
                        f"  Decision: Start New Kernel {self.kernel_number} because parent does not enable fusion"
                    )
                    self.kernels.append([op])
                    self.kernel_lookup[op] = self.kernel_number
                    op.available_during_exec = True
                    self.kernel_number += 1

            else:
                print(f"  Decision: Fallback - Start New Kernel {self.kernel_number}")
                self.kernels.append([op])
                self.kernel_lookup[op] = self.kernel_number
                op.available_during_exec = True
                self.kernel_number += 1

        print("\n--- Kernel Construction Complete ---")
        print(f"Total Kernels: {len(self.kernels)}")

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
            print(f"Kernel {i}: {[str(op) for op in k]}")
            if len(k) > 1:
                print(f"  Kernel {i} is fused ({len(k)} ops). Building instructions...")
                fusor = Fusor(k)
                custom_instruction, kernel_name = fusor.build_kernel()
                self.all_custom_instructions.append(custom_instruction)
                print(f"  Kernel {i} instructions built.")
            kernel = Kernel(
                [op for op in k],
                custom_instruction,
                i,
            )
            self.kernels_objs.append(kernel.convert_kernel(kernel_name))
            print(custom_instruction)

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

    def distribute_results(self, execution_results, lim=None):
        for kernel_result, kernel in zip(execution_results, self.kernels[lim:]):
            for op_res, op in zip(kernel_result.val, kernel):
                op.values = op_res

    def execute_ops(self, lim=0, lim_kernels=0):
        self.rewrite()
        self.finalise(lim, lim_kernels)
        res = rt.execute_graph(self.kernels_objs[lim_kernels:])
        # order of kernels returned from rt is the same as the order given to rt
        self.distribute_results(res, lim)


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
