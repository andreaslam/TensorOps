import tensorops
from tensorops.tensor import Tensor, TensorContext
with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    t = Tensor(100.0)
    l = t.log(10.0)
    tc.forward()
    print("log10(100):", l.item())
