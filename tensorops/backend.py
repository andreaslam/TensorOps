# An assortment of tensor functionalities demonstrated using tensorops.
# This code is to be used as comparison with examples/pytorch/tensordemo.py

import cProfile
import random
import time
from tensorops.newtensor import Tensor, TensorContext, forward, visualise_graph


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
        # print(b)
        c = a_cos + b
        # print(c)
        d = a_cos - x
        # print(d)
        e = d * c
        # print(e)
        f = a_cos / d
        f = f.reshape((n, n))
        g = f @ y
        # print(g)
        h = g.tanh()
        # print(h)
        h.reshape((n**2))
        i = h + e
        forward(nc.ops)
        i.seed_grad(1)
        # print(*nc.ops, sep="\n")
        # print(i)
        # print(i.grads)
        # visualise_graph(nc.ops, display=False)
        end = time.time()
        print(f"took {end - start}s")


if __name__ == "__main__":
    cProfile.run("main()")
