import tensorops
from tensorops.tensor import Tensor, TensorContext
with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
    b_T = b.permute([1, 0])
    tc.forward()
    print("b_T:", b_T.values)
