import tensorops
from tensorops.tensor import Tensor, TensorContext
from tensorops.loss import CrossEntropyLoss
with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    logits = Tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    target = Tensor([[0.0, 0.0, 1.0]])
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    tc.forward()
    print("Loss:", loss.item())
    tc.backward()
    print("Logits grad:", logits.grads.values)
