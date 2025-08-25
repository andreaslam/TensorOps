from tinygrad import Tensor

a = Tensor.randn(1024, 512)
b = Tensor.randn(512, 1024)
aa = a.reshape([1024, 1, 512]).expand([1024, 1024, 512])
bb = b.permute([1, 0]).reshape([1, 1024, 512]).expand([1024, 1024, 512])
cc = (aa * bb).sum(axis=2)  # matmul result
print("a@b", (a @ b).numpy())
print("cc", (cc).numpy())
