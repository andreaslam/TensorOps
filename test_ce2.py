import tensorops
from tensorops.tensor import Tensor, TensorContext
from tensorops.loss import CrossEntropyLoss
with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    logits = Tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    target = Tensor([[0.0, 0.0, 1.0]])
    probs = logits.softmax(axis=1)
    log_probs = probs.log()
    target_x_log = (target * log_probs)
    sum_val = target_x_log.sum(axis=1)
    tc.forward()
    print("probs:", probs.values)
    print("log probs:", log_probs.values)
    print("target_x_log:", target_x_log.values)
    print("sum_val:", sum_val.values)
    print("-1 tensor:", Tensor(-1.0).values)
