import cProfile
import random
import time
import torch


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda")
    n = 5000

    start = time.time()

    # a: shape (1, n*n), requires grad
    a = torch.tensor(
        [[random.random() for _ in range(n * n)]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    a_cos = torch.cos(a)

    # b: shape (1, n*n), no grad
    b = torch.tensor(
        [[random.random() for _ in range(n * n)]],
        device=device,
        dtype=torch.float32,
        requires_grad=False,
    )

    # x: shape (n*n,), no grad
    x = torch.tensor(
        [random.random() for _ in range(n * n)],
        device=device,
        dtype=torch.float32,
        requires_grad=False,
    )

    # y: shape (n, n), no grad
    y = torch.tensor(
        [[random.random() for _ in range(n)] for _ in range(n)],
        device=device,
        dtype=torch.float32,
        requires_grad=False,
    )

    c = a_cos + b  # broadcast-safe
    d = a_cos - x  # broadcast (1, n*n) - (n*n,)
    e = d * c
    f = a_cos / d

    f = f.reshape((n, n))
    g = f @ y
    h = torch.tanh(g)

    h = h.reshape((n * n,))
    i = h + e  # broadcast (n*n,) + (1, n*n)

    # Backward pass
    i.sum().backward()

    end = time.time()
    print(f"took {end - start}s")


if __name__ == "__main__":
    cProfile.run("main()")
