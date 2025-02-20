from abc import ABC
from functools import reduce
from operator import mul

# import hip_cpu_bindings
from tensorops.node import Node, cos, leaky_relu, relu, sin, tanh


class Tensor(ABC):
    def __init__(
        self, values, requires_grad: bool = True, weight: bool = False
    ) -> None:
        super().__init__()
        self.values = values
        self.shape = None
        self.tensor1 = None
        # values could be a Node, list[Node], float, int but not another Tensor
        if isinstance(
            self.values, (float, int)
        ):  # wrap float/int values in a Node first
            self.values = [
                Node(self.values, requires_grad=requires_grad, weight=weight)
            ]
        elif isinstance(self.values, Node):
            self.values = [self.values]
        elif isinstance(self.values, list):
            self.shape = get_shape(self.values)
            flat = self.flatten()
            if all(
                isinstance(val, (int, float, Node)) for val in flat
            ):  # allowed datatype
                self.values = [
                    (
                        Node(val, requires_grad=requires_grad, weight=weight)
                        if isinstance(val, (int, float))
                        else val
                    )
                    for val in flat
                ]
                self.reshape(self.shape)
            else:
                raise ValueError("Invalid subtype inside list!")
        else:
            raise ValueError(
                f"Tensor must contain either a Node, list[Node], float, int, got {type(self.values).__name__}"
            )
        self.shape = get_shape(self.values) if not self.shape else self.shape
        self.children = []
        self.requires_grad = requires_grad
        self.weight = weight
        self.grad = lambda: [
            value.grad for value in self.flatten()
        ]  # THIS SHOWS FLATTENED VIEW OF ALL GRADIENTS, THIS MIGHT NOT ACTUALLY REFLECT THE TENSOR'S ACTUAL ORIENTATION OF GRADIENTS

    def compute(self):
        pass

    def __getitem__(self, item):
        return self.values

    def flatten(self):
        def recursive_flatten(lst):
            flat_list = []
            for item in lst:
                if isinstance(item, list):
                    flat_list.extend(recursive_flatten(item))
                else:
                    flat_list.append(item)
            get_shape(flat_list)
            return flat_list

        return recursive_flatten(self.values)

    def reshape(self, shape):
        """
        Reshape the flattened list into the specified n-dimensional shape.

        Args:
            shape (tuple or list): The target shape as a tuple or list (e.g., (m, n)).

        Raises:
            ValueError: If the total number of elements does not match the product of dimensions.

        Returns:
            None: Updates `self.values` and `self.shape` in-place.
        """
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list")

        # Calculate total elements in the target shape
        total_elements_target = reduce(mul, shape, 1)

        # Check if reshaping is possible
        if len(self.flatten()) != total_elements_target:
            raise ValueError(
                f"Cannot reshape list of size {len(self.flatten())} into shape {shape}"
            )

        def recursive_reshape(flat_list, target_shape):
            """Recursively reshape a flat list into nested lists matching target shape."""
            if len(target_shape) == 1:
                return flat_list[: target_shape[0]]

            sublist_size = reduce(mul, target_shape[1:], 1)
            reshaped_list = []

            for i in range(target_shape[0]):
                start_idx = i * sublist_size
                end_idx = start_idx + sublist_size
                reshaped_list.append(
                    recursive_reshape(flat_list[start_idx:end_idx], target_shape[1:])
                )

            return reshaped_list

        # Perform the reshape operation
        self.values = recursive_reshape(self.flatten(), shape)
        self.shape = get_shape(self.values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, values={self.values})"

    def __len__(self) -> int:
        return len(self.flatten())

    def seed_grad(self, seed: int):  # TODO parallelise
        for f in self.flatten():
            f.seed_grad(seed)

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return ElementMul(self, other)

    def __neg__(self):
        return ElementMul(self, Node(-1))

    def sin(self):
        return Sin(self)

    def cos(self):
        return Cos(self)

    def tanh(self):
        return Tanh(self)

    def relu(self):
        return ReLU(self)

    def leaky_relu(self):
        return LeakyReLU(self)

    def __truediv__(self, other):
        return Div(self, other)

    def __matmul__(self, other):
        return MatMul(self, other)


def repeat(val, shape):
    t = Tensor([val] * reduce(mul, list(shape), 1))
    t.reshape(shape)
    return t


def zeros(shape):
    return repeat(0.0, shape)


def ones(shape):
    return repeat(1.0, shape)


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


class Add(Tensor):  
    def __init__(
        self, tensor1, tensor2, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        original_shape1 = self.tensor1.shape
        original_shape2 = self.tensor2.shape
        self.tensor1_flat = tensor1.flatten()
        self.tensor2_flat = tensor2.flatten()

        assert len(self.tensor1) == len(self.tensor2), "Tensor lengths must match!"
        assert len(self.tensor1_flat) == len(self.tensor2_flat)
        if original_shape1 != original_shape2:
            self.output_shape = self.tensor1.shape
        else:
            self.output_shape = original_shape1

        super().__init__(
            [x + y for x, y in zip(self.tensor1_flat, self.tensor2_flat)],
            requires_grad,
            weight,
        )
        self.parents = [tensor1, tensor2]

    def compute(self):
        # self.values = [
        #     x + y for x, y in zip(self.tensor1_flat, self.tensor2_flat)
        # ]  # TODO check if this line is needed
        hip_cpu_bindings.run_vector_add(
            self.tensor1_flat, self.tensor2_flat, self.values
        )
        self.reshape(self.output_shape)


class Sub(Tensor):  
    def __init__(
        self, tensor1, tensor2, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        original_shape1 = self.tensor1.shape
        original_shape2 = self.tensor2.shape
        self.tensor1_flat = tensor1.flatten()
        self.tensor2_flat = tensor2.flatten()

        assert len(self.tensor1) == len(self.tensor2), "Tensor lengths must match!"
        assert len(self.tensor1_flat) == len(self.tensor2_flat)
        if original_shape1 != original_shape2:
            self.output_shape = self.tensor1.shape
        else:
            self.output_shape = original_shape1

        super().__init__(
            [x - y for x, y in zip(self.tensor1_flat, self.tensor2_flat)],
            requires_grad,
            weight,
        )
        self.parents = [tensor1, tensor2]

    def compute(self):
        # self.values = [
        #     x - y for x, y in zip(self.tensor1_flat, self.tensor2_flat)
        # ]  # TODO check if this line is needed
        hip_cpu_bindings.run_vector_sub(
            self.tensor1_flat, self.tensor2_flat, self.values
        )
        self.reshape(self.output_shape)


class ElementMul(Tensor):  
    def __init__(
        self, tensor1, tensor2, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        original_shape1 = self.tensor1.shape
        original_shape2 = self.tensor2.shape
        self.tensor1_flat = tensor1.flatten()
        self.tensor2_flat = tensor2.flatten()

        assert len(self.tensor1) == len(self.tensor2), "Tensor lengths must match!"
        assert len(self.tensor1_flat) == len(self.tensor2_flat)
        if original_shape1 != original_shape2:
            self.output_shape = self.tensor1.shape
        else:
            self.output_shape = original_shape1

        super().__init__(
            [x * y for x, y in zip(self.tensor1_flat, self.tensor2_flat)],
            requires_grad,
            weight,
        )
        self.parents = [tensor1, tensor2]

    def compute(self):
        # self.values = [
        #     x * y for x, y in zip(self.tensor1_flat, self.tensor2_flat)
        # ]  # TODO check if this line is needed
        hip_cpu_bindings.run_vector_element_mul(
            self.tensor1_flat, self.tensor2_flat, self.values
        )
        self.reshape(self.output_shape)


class Div(Tensor):  
    def __init__(
        self, tensor1, tensor2, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        original_shape1 = self.tensor1.shape
        original_shape2 = self.tensor2.shape
        self.tensor1_flat = tensor1.flatten()
        self.tensor2_flat = tensor2.flatten()
        assert len(self.tensor1) == len(self.tensor2), "Tensor lengths must match!"
        assert len(self.tensor1_flat) == len(self.tensor2_flat)

        if original_shape1 != original_shape2:
            self.output_shape = self.tensor1.shape
        else:
            self.output_shape = original_shape1

        super().__init__(
            [x / y for x, y in zip(self.tensor1_flat, self.tensor2_flat)],
            requires_grad,
            weight,
        )
        self.parents = [tensor1, tensor2]

    def compute(self):
        # self.values = [
        #     x / y for x, y in zip(self.tensor1_flat, self.tensor2_flat)
        # ]  # TODO check if this line is needed
        hip_cpu_bindings.run_vector_div(
            self.tensor1_flat, self.tensor2_flat, self.values
        )
        self.reshape(self.output_shape)


class Cos(Tensor):  
    def __init__(
        self, tensor1, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.output_shape = self.tensor1.shape
        self.tensor1_flat = tensor1.flatten()
        super().__init__(
            [cos(x) for x in self.tensor1_flat],
            requires_grad,
            weight,
        )
        self.parents = [tensor1]

    def compute(self):
        # self.values = [cos(x) for x in self.tensor1_flat]
        hip_cpu_bindings.run_vector_cos(self.tensor1_flat, self.values)
        self.reshape(self.output_shape)


class Sin(Tensor):  
    def __init__(
        self, tensor1, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.output_shape = self.tensor1.shape
        self.tensor1_flat = tensor1.flatten()

        super().__init__(
            [sin(x) for x in self.tensor1_flat],
            requires_grad,
            weight,
        )
        self.parents = [tensor1]

    def compute(self):
        # self.values = [sin(x) for x in self.tensor1_flat]
        hip_cpu_bindings.run_vector_sin(self.tensor1_flat, self.values)
        self.reshape(self.output_shape)


class Tanh(Tensor):  
    def __init__(
        self, tensor1, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.output_shape = self.tensor1.shape
        self.tensor1_flat = tensor1.flatten()

        super().__init__(
            [tanh(x) for x in self.tensor1_flat],
            requires_grad,
            weight,
        )
        self.parents = [tensor1]

    def compute(self):
        # self.values = [tanh(x) for x in self.tensor1_flat]
        hip_cpu_bindings.run_vector_tanh(self.tensor1_flat, self.values)
        self.reshape(self.output_shape)


class ReLU(Tensor):  
    def __init__(
        self, tensor1, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.output_shape = self.tensor1.shape
        self.tensor1_flat = tensor1.flatten()

        super().__init__(
            [relu(x) for x in self.tensor1_flat],
            requires_grad,
            weight,
        )
        self.parents = [tensor1]

    def compute(self):
        # self.values = [relu(x) for x in self.tensor1_flat]
        hip_cpu_bindings.run_vector_relu(self.tensor1_flat, self.values)
        self.reshape(self.output_shape)


class LeakyReLU(Tensor):  
    def __init__(
        self, tensor1, requires_grad: bool = True, weight: bool = False
    ) -> None:
        # validate inputs
        self.tensor1 = tensor1
        self.output_shape = self.tensor1.shape
        self.tensor1_flat = tensor1.flatten()

        super().__init__(
            [leaky_relu(x) for x in self.tensor1_flat],
            requires_grad,
            weight,
        )
        self.parents = [tensor1]

    def compute(self):
        # self.values = ([leaky_relu(x) for x in self.tensor1_flat])
        hip_cpu_bindings.run_vector_leakyrelu(self.tensor1_flat, self.values)
        self.reshape(self.output_shape)


class MatMul(Tensor):
    def __init__(
        self, tensor1, tensor2, requires_grad: bool = True, weight: bool = False
    ) -> None:
        self.tensor1 = tensor1
        self.tensor2 = tensor2

        # validate matrix shapes based on previous inputs, as self.tensor1 for tensors that have not been computed before yet is flattened
        # yet those matrices are still able to be multiplied
        # need to check for those
        # however there are some matrices that do not have a prior self.tensor1 (such as user-created `Tensor` literals)
        # those "real" arrays created by users cannot be mistaken for a valid "flattened" array and if they don't match valid sizes, they would fail the assertion below

        if self.tensor1.tensor1:
            tensor1_mk = list(get_shape(self.tensor1.tensor1.values)[-2:])
            output_ndims = len(self.tensor1.tensor1.shape)
        else:
            tensor1_mk = list(get_shape(self.tensor1.values)[-2:])
            output_ndims = len(self.tensor1.shape)
        if self.tensor2.tensor1:
            tensor2_mk = list(get_shape(self.tensor2.tensor1.values)[-2:])
        else:
            tensor2_mk = list(get_shape(self.tensor2.values)[-2:])

        assert (
            tensor1_mk[1] == tensor2_mk[0]
            and len(tensor1_mk) == 2
            and len(tensor2_mk) == 2
        )
        self.m = tensor1_mk[0]
        self.n = tensor2_mk[1]
        self.k = tensor1_mk[1]
        self.tensor1_flat = self.tensor1.flatten()
        self.tensor2_flat = self.tensor2.flatten()
        assert len(self.tensor1_flat) == len(self.tensor2_flat)
        self.output_shape = tuple(([1] * (output_ndims - 2)) + [self.m, self.n])
        super().__init__(
            self.create_matmul_result(
                self.tensor1_flat, self.tensor2_flat, self.m, self.n, self.k
            ),
            requires_grad,
            weight,
        )
        self.parents = [tensor1, tensor2]

    def create_matmul_result(self, tensor1, tensor2, m, n, k):
        matmul_output = [[None for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                result = Node(0.0)
                for k_idx in range(k):
                    a_idx = i * k + k_idx
                    b_idx = k_idx * n + j
                    product = tensor1[a_idx] * tensor2[b_idx]
                    result = result + product
                matmul_output[i][j] = result
        return matmul_output

    def compute(self):
        # self.values = self.create_matmul_result(
        #     self.tensor1_flat, self.tensor2_flat, self.m, self.n, self.k
        # )
        hip_cpu_bindings.run_gemm(
            self.tensor1_flat,
            self.tensor2_flat,
            self.flatten(),
            self.m,
            self.n,
            self.k,
            1.0,
            0.0,
        )
        self.reshape(self.output_shape)


def get_shape(value):
    if isinstance(value, list):
        if len(value) == 0:
            return (0,)
        sub_shapes = [get_shape(sub) for sub in value]
        if all(shape == sub_shapes[0] for shape in sub_shapes):
            return (len(value),) + sub_shapes[0]
        else:
            raise ValueError("Irregular shapes are not supported")
    return ()
