from tensorops.tensorpy import Tensor, TensorContext
with TensorContext() as tc:
    a = Tensor(1)
    a += 1
    tc.forward(a)
print(a)