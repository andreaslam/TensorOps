from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
	import onnx
	from onnx import TensorProto, helper, numpy_helper
except ImportError:  # pragma: no cover - guarded by _require_onnx
	onnx = None
	TensorProto = None
	helper = None
	numpy_helper = None

from tensorops.model import Model
from tensorops.tensor import (
	OP,
	Abs,
	Add,
	Cos,
	Div,
	ElementMul,
	ExpandOP,
	GenericLog,
	LeakyReLU,
	MatMul,
	Max,
	Min,
	PermuteOP,
	Pow,
	ShapeOP,
	Sin,
	StopGrad,
	Sub,
	Sum,
	Tanh,
	Tensor,
	TensorContext,
	zeros,
)


def _require_onnx() -> None:
	if onnx is None:
		raise ImportError(
			"onnx is required for import/export. Install with `pip install onnx`."
		)


def _as_list(value: Any) -> list:
	if value is None:
		return []
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]


def _infer_shape_from_values(values: Any) -> tuple | None:
	if values is None:
		return None
	if not isinstance(values, list):
		return ()
	if not values:
		return (0,)
	if isinstance(values[0], list):
		inner = _infer_shape_from_values(values[0]) or ()
		return (len(values),) + inner
	return (len(values),)


def _tensor_shape(tensor: Tensor) -> tuple | None:
	if getattr(tensor, "shape", None) is not None:
		return tuple(tensor.shape)
	return _infer_shape_from_values(tensor.values)


def _tensor_flat_list(tensor: Tensor) -> list[float]:
	data = tensor.tolist(shaped=False)
	return [float(x) for x in data]


def _extract_scalar(tensor: Tensor) -> float | None:
	data = tensor.tolist(shaped=False)
	if not data:
		return None
	if len(data) != 1:
		return None
	return float(data[0])


def _collect_graph(outputs: Sequence[Tensor]) -> tuple[list[OP], list[Tensor]]:
	visited: set[int] = set()
	ops: list[OP] = []
	leaves: list[Tensor] = []

	def visit(node: Tensor) -> None:
		node_id = id(node)
		if node_id in visited:
			return
		visited.add(node_id)
		if isinstance(node, OP):
			for parent in getattr(node, "parents", []) or []:
				if isinstance(parent, Tensor):
					visit(parent)
			ops.append(node)
		else:
			leaves.append(node)

	for out in outputs:
		visit(out)
	return ops, leaves


def _normalize_inputs(inputs: Any | None) -> list[Tensor] | None:
	if inputs is None:
		return None
	if isinstance(inputs, Tensor):
		return [inputs]
	return [t for t in inputs]


def _resolve_model_outputs(model: Model, inputs: Any | None) -> list[Tensor]:
	if getattr(model, "model_output_layer", None) is not None:
		output = getattr(model.model_output_layer, "layer_output", None)
		if output is not None:
			return _as_list(output)
	if hasattr(model, "output_tensor"):
		output = getattr(model, "output_tensor")
		if output is not None:
			return _as_list(output)
	if inputs is not None:
		return _as_list(model(inputs))
	raise ValueError(
		"Unable to determine model outputs. Provide `inputs` or pass explicit outputs."
	)


def _infer_inputs(
	model: Model | None,
	leaves: Sequence[Tensor],
	explicit_inputs: list[Tensor] | None,
) -> list[Tensor]:
	if explicit_inputs is not None:
		return explicit_inputs

	if model is not None and getattr(model, "model_input_layer", None) is not None:
		layer_inputs = getattr(model.model_input_layer, "layer_input_tensors", None)
		if layer_inputs is not None:
			return _as_list(layer_inputs)

	if model is not None and hasattr(model, "input_tensor"):
		input_tensor = getattr(model, "input_tensor")
		if input_tensor is not None:
			return _as_list(input_tensor)

	candidates = [t for t in leaves if not getattr(t, "weight", False)]
	if len(candidates) == 1:
		return candidates
	non_scalar = [t for t in candidates if (len(t) if t is not None else 0) > 1]
	if len(non_scalar) == 1:
		return non_scalar

	raise ValueError(
		"Unable to infer graph inputs. Pass `inputs` explicitly to export_onnx()."
	)


@dataclass
class _NameGenerator:
	used: dict[str, int]

	def new(self, prefix: str) -> str:
		idx = self.used.get(prefix, 0)
		self.used[prefix] = idx + 1
		return f"{prefix}_{idx}"


def export_onnx(
	model_or_outputs: Model | Tensor | Sequence[Tensor],
	file_path: str | None = None,
	*,
	inputs: Tensor | Sequence[Tensor] | None = None,
	input_names: Sequence[str] | None = None,
	output_names: Sequence[str] | None = None,
	opset_version: int = 17,
	model_name: str = "TensorOpsModel",
) -> "onnx.ModelProto":
	_require_onnx()

	model = model_or_outputs if isinstance(model_or_outputs, Model) else None
	resolved_inputs = _normalize_inputs(inputs)

	if model is not None:
		outputs = _resolve_model_outputs(model, inputs)
	else:
		outputs = _as_list(model_or_outputs)

	ops, leaves = _collect_graph(outputs)
	input_tensors = _infer_inputs(model, leaves, resolved_inputs)

	if input_names is not None and len(input_names) != len(input_tensors):
		raise ValueError("input_names length must match number of inputs")

	if output_names is not None and len(output_names) != len(outputs):
		raise ValueError("output_names length must match number of outputs")

	name_gen = _NameGenerator(used={})
	name_map: dict[int, str] = {}

	if output_names is not None:
		for idx, out in enumerate(outputs):
			name_map[id(out)] = output_names[idx]

	for idx, tensor in enumerate(input_tensors):
		name = (
			input_names[idx]
			if input_names is not None
			else name_gen.new("input")
		)
		name_map[id(tensor)] = name

	initializers: list[onnx.TensorProto] = []
	initializer_names: set[str] = set()

	def add_initializer(name: str, values: list, shape: Sequence[int], dtype) -> str:
		if name in initializer_names:
			return name
		initializer_names.add(name)
		tensor_proto = helper.make_tensor(name, dtype, list(shape), values)
		initializers.append(tensor_proto)
		return name

	input_set = {id(t) for t in input_tensors}
	for tensor in leaves:
		if id(tensor) in input_set:
			continue
		if id(tensor) in name_map:
			continue
		name = name_gen.new("param" if getattr(tensor, "weight", False) else "const")
		shape = _tensor_shape(tensor)
		if shape is None:
			raise ValueError("Cannot export tensor with unknown shape")
		flat = _tensor_flat_list(tensor)
		add_initializer(name, flat, shape, TensorProto.FLOAT)
		name_map[id(tensor)] = name

	nodes: list[onnx.NodeProto] = []

	def get_name(t: Tensor) -> str:
		if id(t) not in name_map:
			name_map[id(t)] = name_gen.new("val")
		return name_map[id(t)]

	def add_shape_initializer(prefix: str, values: Sequence[int]) -> str:
		name = name_gen.new(prefix)
		add_initializer(name, [int(v) for v in values], [len(values)], TensorProto.INT64)
		return name

	def add_scalar_initializer(prefix: str, value: float) -> str:
		name = name_gen.new(prefix)
		add_initializer(name, [float(value)], [1], TensorProto.FLOAT)
		return name

	for op in ops:
		out_name = get_name(op)
		parents = getattr(op, "parents", []) or []
		in_names = [get_name(p) for p in parents if isinstance(p, Tensor)]

		if isinstance(op, Add):
			nodes.append(helper.make_node("Add", in_names, [out_name]))
		elif isinstance(op, Sub):
			nodes.append(helper.make_node("Sub", in_names, [out_name]))
		elif isinstance(op, ElementMul):
			nodes.append(helper.make_node("Mul", in_names, [out_name]))
		elif isinstance(op, Div):
			nodes.append(helper.make_node("Div", in_names, [out_name]))
		elif isinstance(op, Pow):
			nodes.append(helper.make_node("Pow", in_names, [out_name]))
		elif isinstance(op, MatMul):
			nodes.append(helper.make_node("MatMul", in_names, [out_name]))
		elif isinstance(op, Sum):
			axis = int(op.axis)
			if opset_version >= 13:
				axes_name = add_shape_initializer("axes", [axis])
				nodes.append(
					helper.make_node("ReduceSum", [in_names[0], axes_name], [out_name], keepdims=0)
				)
			else:
				nodes.append(
					helper.make_node("ReduceSum", in_names, [out_name], axes=[axis], keepdims=0)
				)
		elif isinstance(op, Max):
			axis = int(op.axis)
			if opset_version >= 13:
				axes_name = add_shape_initializer("axes", [axis])
				nodes.append(
					helper.make_node("ReduceMax", [in_names[0], axes_name], [out_name], keepdims=0)
				)
			else:
				nodes.append(
					helper.make_node("ReduceMax", in_names, [out_name], axes=[axis], keepdims=0)
				)
		elif isinstance(op, Min):
			axis = int(op.axis)
			if opset_version >= 13:
				axes_name = add_shape_initializer("axes", [axis])
				nodes.append(
					helper.make_node("ReduceMin", [in_names[0], axes_name], [out_name], keepdims=0)
				)
			else:
				nodes.append(
					helper.make_node("ReduceMin", in_names, [out_name], axes=[axis], keepdims=0)
				)
		elif isinstance(op, ShapeOP):
			shape_name = add_shape_initializer("shape", list(op.shape))
			nodes.append(helper.make_node("Reshape", [in_names[0], shape_name], [out_name]))
		elif isinstance(op, ExpandOP):
			shape_name = add_shape_initializer("shape", list(op.shape))
			nodes.append(helper.make_node("Expand", [in_names[0], shape_name], [out_name]))
		elif isinstance(op, PermuteOP):
			nodes.append(
				helper.make_node("Transpose", [in_names[0]], [out_name], perm=list(op.dims))
			)
		elif isinstance(op, LeakyReLU):
			alpha = _extract_scalar(op.leaky_grad)
			if alpha is None:
				raise ValueError("LeakyReLU alpha must be a scalar to export")
			nodes.append(
				helper.make_node("LeakyRelu", [in_names[0]], [out_name], alpha=float(alpha))
			)
		elif isinstance(op, Tanh):
			nodes.append(helper.make_node("Tanh", [in_names[0]], [out_name]))
		elif isinstance(op, Sin):
			nodes.append(helper.make_node("Sin", [in_names[0]], [out_name]))
		elif isinstance(op, Cos):
			nodes.append(helper.make_node("Cos", [in_names[0]], [out_name]))
		elif isinstance(op, Abs):
			nodes.append(helper.make_node("Abs", [in_names[0]], [out_name]))
		elif isinstance(op, GenericLog):
			base = _extract_scalar(op.tensor1)
			if base is None:
				raise ValueError("GenericLog base must be a scalar to export")
			if abs(base - math.e) < 1e-6:
				nodes.append(helper.make_node("Log", [in_names[1]], [out_name]))
			else:
				log_name = name_gen.new("log")
				nodes.append(helper.make_node("Log", [in_names[1]], [log_name]))
				denom = add_scalar_initializer("log_base", math.log(base))
				nodes.append(helper.make_node("Div", [log_name, denom], [out_name]))
		elif isinstance(op, StopGrad):
			nodes.append(helper.make_node("Identity", [in_names[0]], [out_name]))
		else:
			raise NotImplementedError(
				f"ONNX export does not support op: {type(op).__name__}"
			)

	graph_inputs: list[onnx.ValueInfoProto] = []
	for tensor in input_tensors:
		name = get_name(tensor)
		shape = _tensor_shape(tensor)
		graph_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

	graph_outputs: list[onnx.ValueInfoProto] = []
	for idx, out in enumerate(outputs):
		name = get_name(out)
		shape = _tensor_shape(out)
		graph_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

	graph = helper.make_graph(nodes, model_name, graph_inputs, graph_outputs, initializer=initializers)
	model_proto = helper.make_model(
		graph, opset_imports=[helper.make_opsetid("", int(opset_version))], producer_name="TensorOps"
	)

	if file_path is not None:
		onnx.save(model_proto, file_path)

	return model_proto


class ImportedOnnxModel:
	def __init__(
		self,
		context,
		inputs: list[Tensor],
		outputs: list[Tensor],
		input_names: list[str],
		output_names: list[str],
	) -> None:
		self.context = context
		self.inputs = inputs
		self.outputs = outputs
		self.input_names = input_names
		self.output_names = output_names

	def __call__(self, inputs: Any, *, execute: bool = True):
		if isinstance(inputs, Mapping):
			for name, value in inputs.items():
				if name not in self.input_names:
					raise ValueError(f"Unknown input name: {name}")
				idx = self.input_names.index(name)
				_assign_tensor_value(self.inputs[idx], value)
		elif isinstance(inputs, (list, tuple)):
			if len(inputs) != len(self.inputs):
				raise ValueError("Input count does not match model inputs")
			for tensor, value in zip(self.inputs, inputs):
				_assign_tensor_value(tensor, value)
		else:
			if len(self.inputs) != 1:
				raise ValueError("Expected a single input value")
			_assign_tensor_value(self.inputs[0], inputs)

		if execute:
			self.context.forward(recompute=True)

		if len(self.outputs) == 1:
			return self.outputs[0]
		return self.outputs


def _assign_tensor_value(target: Tensor, value: Any) -> None:
	if isinstance(value, Tensor):
		src = value.values if value.values is not None else value.flat
		target.values = src
		return
	if hasattr(value, "tolist"):
		value = value.tolist()
	target.values = value


def import_onnx(
	model_or_path: str | "onnx.ModelProto",
	*,
	input_shapes: Mapping[str, Sequence[int]] | None = None,
	device=None,
) -> ImportedOnnxModel:
	_require_onnx()
	input_shapes = input_shapes or {}

	model = onnx.load(model_or_path) if isinstance(model_or_path, str) else model_or_path
	graph = model.graph

	initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

	with TensorContext(device=device) as context:
		tensor_map: dict[str, Tensor] = {}

		input_tensors: list[Tensor] = []
		input_names: list[str] = []

		for inp in graph.input:
			name = inp.name
			if name in initializers:
				arr = initializers[name]
				t = Tensor(arr.tolist(), requires_grad=False, device=device)
				t.shape = tuple(arr.shape)
				tensor_map[name] = t
				continue
			shape = input_shapes.get(name)
			if shape is None:
				dims = []
				for dim in inp.type.tensor_type.shape.dim:
					if dim.dim_value:
						dims.append(int(dim.dim_value))
					else:
						dims.append(1)
				shape = dims
			t = zeros(tuple(shape), device=device)
			tensor_map[name] = t
			input_tensors.append(t)
			input_names.append(name)

		for name, arr in initializers.items():
			if name in tensor_map:
				continue
			t = Tensor(arr.tolist(), requires_grad=False, device=device)
			t.shape = tuple(arr.shape)
			tensor_map[name] = t

		for node in graph.node:
			op_type = node.op_type
			inputs = [tensor_map[name] for name in node.input if name]
			outputs = list(node.output)

			def get_attr(attr_name, default=None):
				for attr in node.attribute:
					if attr.name == attr_name:
						return helper.get_attribute_value(attr)
				return default

			if op_type == "Add":
				out = inputs[0] + inputs[1]
			elif op_type == "Sub":
				out = inputs[0] - inputs[1]
			elif op_type == "Mul":
				out = inputs[0] * inputs[1]
			elif op_type == "Div":
				out = inputs[0] / inputs[1]
			elif op_type == "Pow":
				out = inputs[0] ** inputs[1]
			elif op_type == "MatMul":
				out = inputs[0] @ inputs[1]
			elif op_type == "Gemm":
				alpha = float(get_attr("alpha", 1.0))
				beta = float(get_attr("beta", 1.0))
				trans_a = int(get_attr("transA", 0))
				trans_b = int(get_attr("transB", 0))
				a = inputs[0]
				b = inputs[1]
				if trans_a:
					a = a.permute([1, 0])
				if trans_b:
					b = b.permute([1, 0])
				out = a @ b
				if abs(alpha - 1.0) > 1e-6:
					out = out * Tensor(alpha, requires_grad=False, device=device)
				if len(inputs) > 2:
					c = inputs[2]
					if abs(beta - 1.0) > 1e-6:
						c = c * Tensor(beta, requires_grad=False, device=device)
					out = out + c
			elif op_type == "Relu":
				out = inputs[0].relu()
			elif op_type == "LeakyRelu":
				alpha = float(get_attr("alpha", 0.01))
				out = inputs[0].leaky_relu(alpha)
			elif op_type == "Tanh":
				out = inputs[0].tanh()
			elif op_type == "Sigmoid":
				out = inputs[0].sigmoid()
			elif op_type == "Softmax":
				axis = int(get_attr("axis", 1))
				if axis < 0 and inputs[0].shape is not None:
					axis += len(inputs[0].shape)
				out = inputs[0].softmax(axis=axis)
			elif op_type == "Exp":
				out = inputs[0].exp()
			elif op_type == "Log":
				out = inputs[0].log()
			elif op_type == "Abs":
				out = Abs(inputs[0])
			elif op_type in ("ReduceSum", "ReduceMax", "ReduceMin"):
				axes = get_attr("axes")
				if axes is None and len(inputs) > 1:
					axes = _tensor_flat_list(inputs[1])
				if axes is None:
					axes = list(range(len(inputs[0].shape)))
				if inputs[0].shape is None:
					raise ValueError("Reduce ops require known shape")
				rank = len(inputs[0].shape)
				axes = [int(a) + rank if int(a) < 0 else int(a) for a in axes]
				out = inputs[0]
				for axis in sorted([int(a) for a in axes], reverse=True):
					if op_type == "ReduceSum":
						out = out.sum(axis=axis)
					elif op_type == "ReduceMax":
						out = Max(out, axis=axis)
					else:
						out = Min(out, axis=axis)
			elif op_type == "Transpose":
				perm = get_attr("perm")
				if perm is None:
					perm = list(range(len(inputs[0].shape) - 1, -1, -1))
				out = inputs[0].permute(tuple(perm))
			elif op_type == "Reshape":
				shape_vals = _tensor_flat_list(inputs[1])
				out = inputs[0].reshape(tuple(int(v) for v in shape_vals))
			elif op_type == "Expand":
				shape_vals = _tensor_flat_list(inputs[1])
				out = inputs[0].expand(tuple(int(v) for v in shape_vals))
			elif op_type == "Squeeze":
				axes = get_attr("axes")
				if axes is None and len(inputs) > 1:
					axes = _tensor_flat_list(inputs[1])
				if axes is None:
					if inputs[0].shape is None:
						raise ValueError("Squeeze requires known shape")
					axes = [i for i, d in enumerate(inputs[0].shape) if d == 1]
				if inputs[0].shape is None:
					raise ValueError("Squeeze requires known shape")
				rank = len(inputs[0].shape)
				axes = [int(a) + rank if int(a) < 0 else int(a) for a in axes]
				shape = list(inputs[0].shape)
				for axis in sorted([int(a) for a in axes], reverse=True):
					shape.pop(axis)
				out = inputs[0].reshape(tuple(shape))
			elif op_type == "Unsqueeze":
				axes = get_attr("axes")
				if axes is None and len(inputs) > 1:
					axes = _tensor_flat_list(inputs[1])
				if axes is None:
					raise ValueError("Unsqueeze requires axes")
				if inputs[0].shape is None:
					raise ValueError("Unsqueeze requires known shape")
				rank = len(inputs[0].shape)
				axes = [int(a) + rank + 1 if int(a) < 0 else int(a) for a in axes]
				shape = list(inputs[0].shape)
				for axis in sorted([int(a) for a in axes]):
					shape.insert(axis, 1)
				out = inputs[0].reshape(tuple(shape))
			elif op_type == "Identity":
				out = inputs[0]
			elif op_type == "Constant":
				value = get_attr("value")
				if value is None:
					raise NotImplementedError("Constant without value attribute")
				arr = numpy_helper.to_array(value)
				out = Tensor(arr.tolist(), requires_grad=False, device=device)
				out.shape = tuple(arr.shape)
			else:
				raise NotImplementedError(f"ONNX import does not support op: {op_type}")

			if len(outputs) != 1:
				raise NotImplementedError(
					f"ONNX import only supports single-output nodes, got {len(outputs)}"
				)
			tensor_map[outputs[0]] = out

		output_tensors: list[Tensor] = []
		output_names: list[str] = []
		for out in graph.output:
			output_names.append(out.name)
			output_tensors.append(tensor_map[out.name])

	return ImportedOnnxModel(context, input_tensors, output_tensors, input_names, output_names)
