# An assortment of tensor functionalities demonstrated using pytorch.
# This code is to be used as comparison with examples/tensorops/tensordemo.py

import cProfile
import random
import time

import torch


def main():
    random.seed(42)
    n = 2000
    start = time.time()
    a = torch.tensor(
        [[random.random() for _ in range(n * n)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    a_cos = torch.cos(a)
    # print(a_cos)
    b = torch.tensor([[random.random() for _ in range(n * n)]], dtype=torch.float64)
    c = a_cos + b
    # print(c)
    d = a_cos - torch.tensor(
        [[random.random() for _ in range(n * n)]], dtype=torch.float64
    )
    # print(d)
    e = d * c
    # print(e)
    f = a_cos / d
    # print(f)
    f = f.reshape([n, n])
    g = torch.matmul(
        f,
        torch.tensor(
            [[random.random() for _ in range(n)] for _ in range(n)], dtype=torch.float64
        ),
    )
    # print(g)
    h = torch.tanh(g)
    # print(h)
    h = h.reshape(n**2)
    i = h + e
    i = i.reshape([1, n**2])

    # print(i)
    print(f"took {time.time() - start}s")
    # i.backward(torch.tensor([[1.0] * (n**2)]))


if __name__ == "__main__":
    cProfile.run("main()")
