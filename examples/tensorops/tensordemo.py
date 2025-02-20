# An assortment of tensor functionalities demonstrated using tensorops.
# This code is to be used as comparison with examples/pytorch/tensordemo.py

import cProfile
import random
import time

from tensorops.node import NodeContext, backward
from tensorops.tensor import Tensor


def main():
    random.seed(42)
    n = 2000
    with NodeContext() as nc:
        start = time.time()
        a = Tensor([[random.random() for _ in range(n * n)]], requires_grad=True)
        a_cos = a.cos()
        a_cos.compute()
        b = Tensor([[random.random() for _ in range(n * n)]], requires_grad=False)
        c = a_cos + b
        c.compute()
        d = a_cos - Tensor([random.random() for _ in range(n * n)], requires_grad=False)
        d.compute()
        e = d * c
        e.compute()
        f = a_cos / d
        f.compute()
        f.reshape([n, n])
        g = f @ Tensor(
            [[random.random() for _ in range(n)] for _ in range(n)], requires_grad=False
        )
        g.compute()
        h = g.tanh()
        h.compute()
        h.reshape([n**2])
        i = h + e
        i.compute()
        i.reshape([1, n**2])
        i.seed_grad(1)
        backward(nc.nodes)
        end = time.time()
        print(f"took {end - start}s")


if __name__ == "__main__":
    cProfile.run("main()")
