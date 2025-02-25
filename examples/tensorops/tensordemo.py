# An assortment of tensor functionalities demonstrated using tensorops.
# This code is to be used as comparison with examples/pytorch/tensordemo.py

import cProfile
import random
import time
from tensorops.tensor import Tensor, TensorContext, forward


def main():
    random.seed(42)
    n = 2000
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
        g = f @ y
        h = g.tanh()
        h.reshape((n**2))
        i = h + e
        forward(nc.ops)
        i.seed_grad(1)
        # visualise_graph(nc.ops, display=False)
        end = time.time()
        print(f"took {end - start}s")


if __name__ == "__main__":
    cProfile.run("main()")
