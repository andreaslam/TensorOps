from tensorops.utils.tensorutils import visualise_graph
from tensorops.tensor import Tensor, TensorContext
import tensorops

with TensorContext(device=tensorops.device.TensorOpsDevice.APPLE) as tc:
    a = Tensor(1)
    b = Tensor(2)
    c = a * b
    d = (c + Tensor(3)).tanh()
    d.compute()
    tc.backward()
    visualise_graph(tc.ops, save_img=False, display=True)
