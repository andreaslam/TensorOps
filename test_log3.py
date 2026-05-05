import tensorops
from tensorops.tensor import Tensor, TensorContext
with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    t = Tensor(10.0)
    l = t.log(100.0)
    tc.finalise()
    for k in tc.kernels:
        print("Kernel:", k)
        if hasattr(k, 'inputs'):
            print("Inputs:", len(k.inputs))
            print("Scalars:", getattr(k, 'scalars', []))
