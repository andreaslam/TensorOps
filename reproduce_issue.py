
import numpy as np

import tensorops.tensor as t
from tensorops.tensor import TensorContext


def test_sub_logical_direct():
    with TensorContext() as ctx:
        # A is logical (result of an op)
        t1 = t.Tensor([10.0, 20.0])
        t2 = t.Tensor([1.0, 2.0])
        a = t1 + t2 # a is [11.0, 22.0], logical because it's an op result
        
        # B is direct
        b = t.Tensor([5.0, 5.0])
        
        # C = A - B
        # Expected: [11-5, 22-5] = [6.0, 17.0]
        c = a - b
        
        ctx.forward()
        
        print(f"A: {a.values}")
        print(f"B: {b.values}")
        print(f"C: {c.values}")
        
        expected = [6.0, 17.0]
        np.testing.assert_allclose(c.values, expected, atol=1e-5)
        print("Test Passed!")

if __name__ == "__main__":
    try:
        test_sub_logical_direct()
    except Exception as e:
        print(f"Test Failed: {e}")
