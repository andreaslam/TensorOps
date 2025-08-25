# An assortment of tensor functionalities demonstrated using tensorops.
# This code is to be used as comparison with examples/pytorch/tensordemo.py

import cProfile
import random
import time
from tensorops.tensor import Tensor, TensorContext, visualise_graph


def main():
    random.seed(42)
    n = 4
    with TensorContext() as nc:
        start = time.time()
        a = Tensor([[random.random() for _ in range(n * n)]], requires_grad=True)
        a_cos = a.cos()
        # print(a_cos)
        b = Tensor([[random.random() for _ in range(n * n)]], requires_grad=False)
        x = Tensor([random.random() for _ in range(n * n)], requires_grad=False)
        y = Tensor(
            [[random.random() for _ in range(n)] for _ in range(n)], requires_grad=False
        )
        c = a_cos + b
        d = a_cos - x
        e = d * c
        f = a_cos / d
        f = f.reshape((n, n))
        # g = f @ y
        h = f.tanh()
        h.reshape((n**2))
        i = h + e
        nc.forward()
        print(i.grads.shape)
        print(h.grads.shape)
        print(e.grads.shape)
        print(f.grads.shape)
        visualise_graph(nc.ops, display=False)
        nc.backward()
        end = time.time()
        print(f"took {end - start}s")


if __name__ == "__main__":
    main()
