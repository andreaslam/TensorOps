# This code is to be used as comparison with examples/pytorch/tensordemo.py
import time
import cProfile
import random
import tensorops
from tensorops.tensor import Tensor, TensorContext


def main():
    random.seed(42)
    n = 5
    iters = 10000
    with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as nc:
        # Build graph once
        a_base = Tensor(
            [random.random() for _ in range(n * n)],
            requires_grad=True,
            device=tensorops.device.TensorOpsDevice.APPLE,
        )
        b_base = Tensor(
            [random.random() for _ in range(n * n)],
            requires_grad=True,
            device=tensorops.device.TensorOpsDevice.APPLE,
        )
        a = a_base.reshape((n, n))
        b = b_base.reshape((n, n))
        _ = a @ b

        for _ in range(iters):
            a_base.values = [random.random() for _ in range(n * n)]
            b_base.values = [random.random() for _ in range(n * n)]
            start = time.time()
            nc.forward(recompute=True)
            nc.backward()
            print(time.time() - start)


if __name__ == "__main__":
    cProfile.run("main()")


# # # An assortment of tensor functionalities demonstrated using tensorops.
# # # This code is to be used as comparison with examples/pytorch/tensordemo.py

# import cProfile
# import random
# import time
# from tensorops.tensor import Tensor, TensorContext


# def main():
#     random.seed(42)
#     n = 2
#     with TensorContext() as nc:
#         a = Tensor([[random.random() for _ in range(n * n)]], requires_grad=False)
#         a_cos = a.cos()
#         b = Tensor([[random.random() for _ in range(n * n)]], requires_grad=False)
#         x = Tensor([random.random() for _ in range(n * n)], requires_grad=False)
#         y = Tensor(
#             [[random.random() for _ in range(n)] for _ in range(n)], requires_grad=False
#         )
#         c = a_cos + b
#         d = a_cos - x
#         e = d * c
#         f = a_cos / d
#         f = f.reshape((n, n))
#         g = f @ y
#         h = g.tanh()
#         h.reshape((n**2))
#         i = h + e
#         start = time.time()
#         nc.forward()
#         nc.backward()
#         end = time.time()
#         print(f"took {end - start}s")


# if __name__ == "__main__":
#     cProfile.run("main()")

# import cProfile
# import random

# import tensorops
# from tensorops.tensor import Tensor


# def main():
#     random.seed(42)
#     n = 500
#     a = Tensor(
#         [[random.random() for _ in range(n * n)]],
#         requires_grad=False,
#         device=tensorops.device.TensorOpsDevice.APPLE,
#     )
#     a_cos = a.cos()
#     b = Tensor(
#         [[random.random() for _ in range(n * n)]],
#         requires_grad=False,
#         device=tensorops.device.TensorOpsDevice.APPLE,
#     )
#     x = Tensor(
#         [random.random() for _ in range(n * n)],
#         requires_grad=False,
#         device=tensorops.device.TensorOpsDevice.APPLE,
#     )
#     y = Tensor(
#         [[random.random() for _ in range(n)] for _ in range(n)],
#         requires_grad=False,
#         device=tensorops.device.TensorOpsDevice.APPLE,
#     )
#     c = a_cos + b
#     d = a_cos - x
#     e = d * c
#     f = a_cos / d
#     f = f.reshape((n, n))
#     g = f @ y
#     h = g.tanh()
#     h.reshape((n**2))
#     i = h + e
#     i.compute()


# if __name__ == "__main__":
#     cProfile.run("main()")
#     cProfile.run("main()")
