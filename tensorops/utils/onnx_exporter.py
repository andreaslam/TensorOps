import onnx
from tensorops.newtensor import TensorContext
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)
from onnx import TensorProto, load
from onnx.checker import check_model
from tensorops.newtensor import *

operation_mapping = {
    "Add": "Add",
    "Sub": "Sub",
    "ElementMul": "Mul",
    "Div": "Div",
}

class OnnxExporter:
    def __init__(self, ops: list):
        self.ops = ops
        self.id_mapping = {op: id(op) for op in self.ops}
        self.inputs = []
        self.nodes = []
        self.outputs = []

    def define_op(self, op):
        ids = []
        for tensor in op.parents:
            self.inputs.append(make_tensor_value_info(op_id:=f"{type(tensor).__name__}_{str(id(tensor))}", TensorProto.FLOAT if tensor.values else TensorProto.UNDEFINED, tensor.shape if tensor.shape else [None]))
            ids.append(op_id)

        self.nodes.append(
            make_node(
                operation_mapping[type(op).__name__],
                ids,
                [str(id(op))],
            )
        )

        self.outputs.append(
            make_tensor_value_info(
                f"{type(op).__name__}_{id(op)}",
                TensorProto.FLOAT if op.values else TensorProto.UNDEFINED,
                op.shape if op.shape else [None],
            )
        )

    def export_all(self):
        for op in self.ops:
            self.define_op(op)

        graph = make_graph(
            self.nodes,
            "graph",
            self.inputs,
            self.outputs,
        )

        model = make_model(graph)


        check_model(model)

        print(model)
        onnx.save(model, "model.onnx")


with TensorContext() as nc:
    a = Tensor([1,2])
    b = Tensor([3,4])
    c = a * b
    d = c - Tensor([5,6])
    forward(nc.ops)

exp = OnnxExporter(nc.ops)
exp.export_all()

from onnx import numpy_helper

# Load the ONNX model
model = onnx.load("model.onnx")

# Iterate through initializers (weights)
for initializer in model.graph.initializer:
    # Convert tensor to numpy array
    weight_array = numpy_helper.to_array(initializer)

    # Print weight name and its values
    print(f"Weight Name: {initializer.name}")
    print(f"Weight Values:\n{weight_array}\n")
