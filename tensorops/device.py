# test code adapted from https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
import torch
from tensorops.tensor import Tensor, TensorContext


def test_sanity_check():
    with TensorContext() as context:
        x = Tensor([-4.0], requires_grad=True, weight=True)
        z = 2 * x + 2 + x
        q = z + z * x
        h = z * z
        y = h + q + q * x
        context.forward()
        context.backward()
        print("tensorops x (val, grad)", x.values, x.grads.values)
        print("tensorops z (val, grad)", z.values, z.grads.values)
        print("tensorops q (val, grad)", q.values, q.grads.values)
        print("tensorops h (val, grad)", h.values, h.grads.values)
        print("tensorops y (val, grad)", y.values, y.grads.values)
        x_tensorops, y_tensorops = x, y
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    z.retain_grad()
    q = z + z * x
    q.retain_grad()
    h = z * z
    h.retain_grad()
    y = h + q + q * x
    y.retain_grad()
    y.backward()
    print("torch x (val, grad)", x, x.grad.item())
    print("torch z (val, grad)", z, z.grad.item())
    print("torch q (val, grad)", q, q.grad.item())
    print("torch h (val, grad)", h, h.grad.item())
    print("torch y (val, grad)", y, y.grad.item())
    x_pytorch, y_pytorch = x, y
    # forward pass went well
    assert y_tensorops.values[0] == y_pytorch.data.item()
    # backward pass went well
    assert x_tensorops.grads.values[0] == x_pytorch.grad.item()


def test_more_ops():
    with TensorContext() as context:
        a = Tensor([-4.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d += d * 2 + (b + a)
        d += 3 * d + (b - a)
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        context.forward()
        context.backward()
        print("tensorops a (val, grad)", a.values, a.grads.values)
        print("tensorops b (val, grad)", b.values, b.grads.values)
        print("tensorops c (val, grad)", c.values, c.grads.values)
        print("tensorops d (val, grad)", d.values, d.grads.values)
        print("tensorops e (val, grad)", e.values, e.grads.values)
        print("tensorops f (val, grad)", f.values, f.grads.values)
        print("tensorops g (val, grad)", g.values, g.grads.values)
        a_tensorops, b_tensorops, g_tensorops = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    c.retain_grad()
    d = d + d * 2 + (b + a)
    d = d + 3 * d + (b - a)
    d.retain_grad()
    e = c - d
    e.retain_grad()
    f = e**2
    f.retain_grad()
    g = f / 2.0
    g = g + 10.0 / f
    g.retain_grad()
    g.backward()
    print("torch a (val, grad)", a, a.grad.item())
    print("torch b (val, grad)", b, b.grad.item())
    print("torch c (val, grad)", c, c.grad.item())
    print("torch d (val, grad)", d, d.grad.item())
    print("torch e (val, grad)", e, e.grad.item())
    print("torch f (val, grad)", f, f.grad.item())
    print("torch g (val, grad)", g, g.grad.item())

    a_pytorch, b_pytorch, g_pytorch = a, b, g

    tolerance = 1e-6
    # forward pass went well
    print(g_tensorops.values[0], g_pytorch.data.item())
    assert abs(g_tensorops.values[0] - g_pytorch.data.item()) < tolerance
    # backward pass went well
    assert abs(a_tensorops.grads.values[0] - a_pytorch.grad.item()) < tolerance
    assert abs(b_tensorops.grads.values[0] - b_pytorch.grad.item()) < tolerance


def main():
    test_sanity_check()
    test_more_ops()


if __name__ == "__main__":
    main()
