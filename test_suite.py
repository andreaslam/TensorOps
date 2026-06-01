import math

from tensorops.loss import BCELoss, CrossEntropyLoss, MSELoss
from tensorops.tensor import Tensor, TensorContext
from tensorops.utils.models import SequentialModel

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class SkipTest(Exception):
    pass


def require_numpy():
    if not NUMPY_AVAILABLE:
        raise SkipTest("numpy not installed")


def require_torch():
    if not TORCH_AVAILABLE:
        raise SkipTest("torch not installed")


def _as_list(value):
    if isinstance(value, Tensor):
        return value.tolist()
    return value


def _flatten(value):
    value = _as_list(value)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-5):
    actual = _as_list(actual)
    expected = _as_list(expected)
    if NUMPY_AVAILABLE:
        a = np.asarray(actual, dtype=np.float64)
        b = np.asarray(expected, dtype=np.float64)
        if a.shape != b.shape:
            raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            diff = np.max(np.abs(a - b))
            raise AssertionError(f"values differ (max abs diff={diff})")
        return

    a = _flatten(actual)
    b = _flatten(expected)
    if len(a) != len(b):
        raise AssertionError(f"length mismatch: {len(a)} vs {len(b)}")
    for idx, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > (atol + rtol * abs(y)):
            raise AssertionError(f"values differ at {idx}: {x} vs {y}")


def _softmax_np(values, axis=-1):
    require_numpy()
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    exp_arr = np.exp(arr)
    return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)


class TinyLinearModel(SequentialModel):
    def __init__(self, loss_criterion, *, batch_size, in_features, out_features):
        super().__init__(loss_criterion, seed=123, batch_size=batch_size)
        with self.context:
            self.add_layer(in_features, out_features, activation_function=None)
            self.loss = self.loss_criterion(
                self.model_output_layer.layer_output, self.targets
            )

    def forward(self, model_inputs: Tensor) -> Tensor:  # type: ignore[override]
        with self.context:
            for layer in self.model_layers:
                layer.forward(model_inputs)
                model_inputs = layer.layer_output
            return model_inputs


def test_tensor_basic_ops_numpy():
    require_numpy()
    a_vals = [[1.0, 2.0], [3.0, 4.0]]
    b_vals = [[5.0, 6.0], [7.0, 8.0]]
    with TensorContext() as tc:
        a = Tensor(a_vals, requires_grad=True)
        b = Tensor(b_vals, requires_grad=True)
        add = a + b
        sub = a - b
        mul = a * b
        div = a / b
        tc.forward()

    np_a = np.array(a_vals, dtype=np.float64)
    np_b = np.array(b_vals, dtype=np.float64)
    assert_allclose(add, np_a + np_b)
    assert_allclose(sub, np_a - np_b)
    assert_allclose(mul, np_a * np_b)
    assert_allclose(div, np_a / np_b)


def test_tensor_matmul_numpy():
    require_numpy()
    a_vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b_vals = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    with TensorContext() as tc:
        a = Tensor(a_vals, requires_grad=True)
        b = Tensor(b_vals, requires_grad=True)
        c = a @ b
        tc.forward()

    expected = np.matmul(
        np.array(a_vals, dtype=np.float64), np.array(b_vals, dtype=np.float64)
    )
    assert_allclose(c, expected)


def test_tensor_permute_numpy():
    require_numpy()
    vals = [[1.0, 2.0], [3.0, 4.0]]
    with TensorContext() as tc:
        t = Tensor(vals, requires_grad=False)
        p = t.permute([1, 0])
        tc.forward()

    expected = np.transpose(np.array(vals, dtype=np.float64), (1, 0))
    assert_allclose(p, expected)


def test_tensor_reshape_expand_numpy():
    require_numpy()
    a_vals = [[1.0, 2.0], [3.0, 4.0]]
    b_vals = [[1.0], [2.0]]
    with TensorContext() as tc:
        a = Tensor(a_vals, requires_grad=False)
        r = a.reshape((4,))
        b = Tensor(b_vals, requires_grad=False)
        e = b.expand((2, 3))
        tc.forward()

    assert_allclose(r, np.reshape(np.array(a_vals, dtype=np.float64), (4,)))
    expected = np.broadcast_to(np.array(b_vals, dtype=np.float64), (2, 3))
    assert_allclose(e, expected)


def test_tensor_sum_numpy():
    require_numpy()
    vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with TensorContext() as tc:
        t = Tensor(vals, requires_grad=True)
        sum0 = t.sum(axis=0)
        sum1 = t.sum(axis=1)
        tc.forward()

    arr = np.array(vals, dtype=np.float64)
    assert_allclose(sum0, arr.sum(axis=0))
    assert_allclose(sum1, arr.sum(axis=1))


def test_tensor_softmax_numpy():
    require_numpy()
    vals = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    with TensorContext() as tc:
        t = Tensor(vals, requires_grad=True)
        s = t.softmax(axis=1)
        tc.forward()

    expected = _softmax_np(vals, axis=1)
    assert_allclose(s, expected, rtol=1e-5, atol=1e-5)


def test_tensor_log_numpy():
    require_numpy()
    vals = [1.0, 2.0, 4.0]
    base = 10.0
    with TensorContext() as tc:
        t = Tensor(vals, requires_grad=True)
        l = t.log(base)
        tc.forward()

    expected = np.log(np.array(vals, dtype=np.float64)) / math.log(base)
    assert_allclose(l, expected, rtol=1e-6, atol=1e-6)


def test_mse_loss_numpy():
    require_numpy()
    actual_vals = [[1.0, 2.0], [3.0, 4.0]]
    target_vals = [[1.5, 1.0], [2.5, 5.0]]
    with TensorContext() as tc:
        actual = Tensor(actual_vals, requires_grad=True)
        target = Tensor(target_vals, requires_grad=False)
        loss = MSELoss()(actual, target)
        tc.forward()

    expected = (
        np.array(target_vals, dtype=np.float64)
        - np.array(actual_vals, dtype=np.float64)
    ) ** 2
    assert_allclose(loss, expected)


def test_bce_loss_numpy():
    require_numpy()
    actual_vals = [[0.2, 0.8], [0.9, 0.1]]
    target_vals = [[0.0, 1.0], [1.0, 0.0]]
    with TensorContext() as tc:
        actual = Tensor(actual_vals, requires_grad=True)
        target = Tensor(target_vals, requires_grad=False)
        loss = BCELoss()(actual, target)
        tc.forward()

    eps = 1e-15
    actual_np = np.array(actual_vals, dtype=np.float64)
    target_np = np.array(target_vals, dtype=np.float64)
    expected = -(
        target_np * np.log(actual_np + eps)
        + (1 - target_np) * np.log(1 - actual_np + eps)
    )
    assert_allclose(loss, expected, rtol=1e-5, atol=1e-5)


def test_cross_entropy_torch():
    require_torch()
    logits = [[0.1, 0.2, 0.7], [1.1, 0.2, -0.1]]
    targets = [2, 0]
    with TensorContext() as tc:
        t_logits = Tensor(logits, requires_grad=True)
        t_targets = Tensor(targets, requires_grad=False)
        loss = CrossEntropyLoss()(t_logits, t_targets)
        tc.forward()
        loss_val = loss.item()
        tc.backward()
        grads = t_logits.grads.tolist()

    torch_logits = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
    torch_targets = torch.tensor(targets, dtype=torch.long)
    loss_t = torch.nn.CrossEntropyLoss()(torch_logits, torch_targets)
    loss_t.backward()

    assert_allclose(loss_val, loss_t.item(), rtol=1e-4, atol=1e-4)
    assert_allclose(grads, torch_logits.grad.tolist(), rtol=1e-4, atol=1e-4)


def test_tensor_autograd_sanity_torch():
    require_torch()
    with TensorContext() as tc:
        x = Tensor([-4.0], requires_grad=True, weight=True)
        z = 2 * x + 2 + x
        q = z + z * x
        h = z * z
        y = h + q + q * x
        tc.forward()
        tc.backward()
        x_tensorops, y_tensorops = x, y

    x_t = torch.tensor([-4.0], dtype=torch.float32, requires_grad=True)
    z_t = 2 * x_t + 2 + x_t
    q_t = z_t + z_t * x_t
    h_t = z_t * z_t
    y_t = h_t + q_t + q_t * x_t
    y_t.backward()

    assert_allclose(y_tensorops.tolist()[0], y_t.item(), rtol=1e-5, atol=1e-5)
    assert_allclose(
        x_tensorops.grads.tolist()[0], x_t.grad.item(), rtol=1e-5, atol=1e-5
    )


def test_tensor_matmul_grad_torch():
    require_torch()
    with TensorContext() as tc:
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
        c = a @ b
        tc.forward()
        tc.backward()
        ac, bc = a.grads.tolist(), b.grads.tolist()
        v = c.tolist()

    a_t = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
    )
    b_t = torch.tensor(
        [[2.0, 0.0], [1.0, 2.0]], dtype=torch.float32, requires_grad=True
    )
    c_t = a_t @ b_t
    c_t.sum().backward()
    ac_t, bc_t = a_t.grad.tolist(), b_t.grad.tolist()
    v_t = c_t.tolist()

    assert_allclose(v, v_t)
    assert_allclose(ac, ac_t, rtol=1e-5, atol=1e-5)
    assert_allclose(bc, bc_t, rtol=1e-5, atol=1e-5)


def test_tensor_log_grad_torch():
    require_torch()
    with TensorContext() as tc:
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a.log()
        tc.forward()
        tc.backward()
        g = a.grads.tolist()
        v = b.tolist()

    a_t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    b_t = a_t.log()
    b_t.sum().backward()
    g_t = a_t.grad.tolist()
    v_t = b_t.tolist()

    assert_allclose(v, v_t, rtol=1e-5, atol=1e-5)
    assert_allclose(g, g_t, rtol=1e-5, atol=1e-5)


def test_tensor_softmax_grad_torch():
    require_torch()
    with TensorContext() as tc:
        a = Tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], requires_grad=True)
        b = a.softmax(axis=1)
        tc.forward()
        tc.backward()
        g = a.grads.tolist()
        v = b.tolist()

    a_t = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float32, requires_grad=True
    )
    b_t = torch.softmax(a_t, dim=1)
    b_t.sum().backward()
    g_t = a_t.grad.tolist()
    v_t = b_t.tolist()

    assert_allclose(v, v_t, rtol=1e-5, atol=1e-5)
    assert_allclose(g, g_t, rtol=1e-5, atol=1e-5)


def test_model_linear_forward_numpy():
    require_numpy()
    model = TinyLinearModel(MSELoss(), batch_size=2, in_features=3, out_features=2)
    layer = model.model_layers[0]
    weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    bias = [[0.01, -0.02]]
    layer.output_weights.values = weights
    layer.output_bias.values = bias

    inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = Tensor(inputs, requires_grad=False)
    out = model(x, execute=False)
    model.context.forward(recompute=True)

    expected = np.dot(
        np.array(inputs, dtype=np.float64), np.array(weights, dtype=np.float64)
    ) + np.array(bias, dtype=np.float64)
    assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_model_linear_forward_torch():
    require_torch()
    model = TinyLinearModel(MSELoss(), batch_size=2, in_features=3, out_features=2)
    layer = model.model_layers[0]
    weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    bias = [[0.01, -0.02]]
    layer.output_weights.values = weights
    layer.output_bias.values = bias

    inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = Tensor(inputs, requires_grad=False)
    out = model(x, execute=False)
    model.context.forward(recompute=True)

    layer_t = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        layer_t.weight.copy_(torch.tensor(weights, dtype=torch.float32).t())
        layer_t.bias.copy_(torch.tensor(bias[0], dtype=torch.float32))
    out_t = layer_t(torch.tensor(inputs, dtype=torch.float32))

    assert_allclose(out, out_t.tolist(), rtol=1e-5, atol=1e-5)


def run_all_tests():
    tests = [
        ("tensor_basic_ops_numpy", test_tensor_basic_ops_numpy),
        ("tensor_matmul_numpy", test_tensor_matmul_numpy),
        ("tensor_permute_numpy", test_tensor_permute_numpy),
        ("tensor_reshape_expand_numpy", test_tensor_reshape_expand_numpy),
        ("tensor_sum_numpy", test_tensor_sum_numpy),
        ("tensor_softmax_numpy", test_tensor_softmax_numpy),
        ("tensor_log_numpy", test_tensor_log_numpy),
        ("mse_loss_numpy", test_mse_loss_numpy),
        ("bce_loss_numpy", test_bce_loss_numpy),
        ("cross_entropy_torch", test_cross_entropy_torch),
        ("tensor_autograd_sanity_torch", test_tensor_autograd_sanity_torch),
        ("tensor_matmul_grad_torch", test_tensor_matmul_grad_torch),
        ("tensor_log_grad_torch", test_tensor_log_grad_torch),
        ("tensor_softmax_grad_torch", test_tensor_softmax_grad_torch),
        ("model_linear_forward_numpy", test_model_linear_forward_numpy),
        ("model_linear_forward_torch", test_model_linear_forward_torch),
    ]

    failures = []
    skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"== {name} ok ==")
        except SkipTest as exc:
            skipped += 1
            print(f"== {name} skipped: {exc} ==")
        except Exception as exc:
            failures.append((name, exc))
            print(f"== {name} failed: {exc} ==")

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise AssertionError(f"Failed tests: {names}")

    print(f"All tests passed. Skipped: {skipped}")


if __name__ == "__main__":
    run_all_tests()
