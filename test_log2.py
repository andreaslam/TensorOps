import tensorops
from tensorops.tensor import Tensor, TensorContext
with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    t = Tensor(10.0)
    l = t.log(100.0)
    tc.forward()
    print("log100(10):", l.item())
