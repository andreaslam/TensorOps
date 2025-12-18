import re
import uuid
from enum import Enum
from typing import Dict, List, Tuple, Union

import tensorops_backend

from tensorops.tensor import *


class KernelOP(Enum):
    VecAdd = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecAdd
    )
    VecSub = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecSub
    )
    VecElementMul = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecElementMul
    )
    VecDiv = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecDiv
    )
    VecPow = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecPow
    )
    VecSin = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecSin
    )
    VecCos = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecCos
    )
    VecTan = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecTan
    )
    VecLog = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecLog
    )
    VecAbs = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecAbs
    )
    VecTanh = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecTanh
    )
    VecLeakyReLU = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecLeakyReLU
    )
    VecSum = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecSum
    )
    VecMax = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecMax
    )
    VecMin = tensorops_backend.KernelType.predefined(
        tensorops_backend.PredefinedKernel.VecMin
    )


class PredefinedKernel(Enum):
    VecAdd = tensorops_backend.PredefinedKernel.VecAdd
    VecSub = tensorops_backend.PredefinedKernel.VecSub
    VecElementMul = tensorops_backend.PredefinedKernel.VecElementMul
    VecDiv = tensorops_backend.PredefinedKernel.VecDiv
    VecPow = tensorops_backend.PredefinedKernel.VecPow
    VecSin = tensorops_backend.PredefinedKernel.VecSin
    VecCos = tensorops_backend.PredefinedKernel.VecCos
    VecTan = tensorops_backend.PredefinedKernel.VecTan
    VecLog = tensorops_backend.PredefinedKernel.VecLog
    VecAbs = tensorops_backend.PredefinedKernel.VecAbs
    VecTanh = tensorops_backend.PredefinedKernel.VecTanh
    VecLeakyReLU = tensorops_backend.PredefinedKernel.VecLeakyReLU
    VecSum = tensorops_backend.PredefinedKernel.VecSum
    VecMax = tensorops_backend.PredefinedKernel.VecMax
    VecMin = tensorops_backend.PredefinedKernel.VecMin


def prepare_kernel_snippets():
    """
    Extracts the kernel snippet (the right-hand side string) for each predefined kernel.
    Keys of the returned dict are members of the user-defined PredefinedKernel Enum.
    """
    matches = {}
    pattern_re = re.compile(r"^\s*C\[gid\](.*)\s*$")
    # Iterate through the USER-DEFINED Enum
    for k in PredefinedKernel:
        try:
            # Get the backend kernel value associated with the user enum member
            backend_kernel_val = k.value
            kernel = tensorops_backend.get_predefined_kernel_source(backend_kernel_val)
            found_match = False
            for line in kernel.splitlines():
                stripped_line = line.strip("    ")
                if match := pattern_re.match(stripped_line):
                    # Key: User enum member (k), Value: Extracted snippet
                    matches[k] = match.group(1).strip()
                    found_match = True
                    break
            if not found_match:
                potential_snippet_lines = [
                    line.strip("    ")
                    for line in kernel.splitlines()
                    if ("val" in line or "alpha" in line)
                    and "=" in line
                    and not line.strip().startswith("//")
                    and "C[gid]" not in line  # Avoid C[gid] lines here
                ]
                if potential_snippet_lines:
                    # Find the part after the '=' sign
                    snippet_part = potential_snippet_lines[0].split("=", 1)[-1].strip()
                    matches[k] = snippet_part
        except Exception as e:
            print(f"Warning: Could not process kernel source for {k.name}. Error: {e}")

    # Ensure snippets only contain the RHS after '=' and strip trailing ';'
    for key, val in matches.items():
        if "=" in val:
            matches[key] = val.split("=", 1)[-1].strip().rstrip(";")
        else:
            matches[key] = val.strip().rstrip(";")

    return matches


# global dictionary holding the kernel snippet patterns
pattern = prepare_kernel_snippets()

op_map = {
    Add: PredefinedKernel.VecAdd,
    Sub: PredefinedKernel.VecSub,
    ElementMul: PredefinedKernel.VecElementMul,
    Div: PredefinedKernel.VecDiv,
    Sin: PredefinedKernel.VecSin,
    Cos: PredefinedKernel.VecCos,
    Tanh: PredefinedKernel.VecTanh,
    LeakyReLU: PredefinedKernel.VecLeakyReLU,
    Pow: PredefinedKernel.VecPow,
    GenericLog: PredefinedKernel.VecLog,
    Sum: PredefinedKernel.VecSum,
    Max: PredefinedKernel.VecMax,
    Min: PredefinedKernel.VecMin,
}


class Kernel:
    def __init__(
        self, op_list: List[OP], custom_src_str: Union[str, None], kernel_id_val: int
    ):
        self.op: List[OP] = (
            op_list  # List of OP objects in this kernel (1 for predefined, multiple for fused)
        )
        self.src: Union[str, None] = (
            custom_src_str  # Custom source from Fusor, or None for predefined
        )

        # self.deps is not directly used for KernelTensorOps creation anymore in the new way
        # self.inputs (direct_initial_inputs) is used for direct_inputs field

        self.kernel_id: int = kernel_id_val

        # This lambda will be called from TensorContext.finalise

    def convert_kernel(
        self, fusor_kernel_name_if_custom: Union[str, None]
    ) -> tensorops_backend.KernelTensorOps:
        if type(self.op[0]).__name__ in ("ShapeOP",):
            return None

        if type(self.op[0]).__name__ == "ExpandOP":
            expand_op = self.op[0]
            parent = expand_op.tensor1

            def _describe_tensor(t) -> str:
                # Avoid calling __repr__ / .values (can materialize huge buffers)
                try:
                    shape = getattr(t, "shape", None)
                except Exception:
                    shape = None
                return (
                    f"{type(t).__name__}(shape={shape}, is_op={getattr(t, 'is_op', False)}, "
                    f"has__values={getattr(t, '_values', None) is not None}, "
                    f"has__flat={getattr(t, '_flat', None) is not None}, "
                    f"has_pending={getattr(t, '_pending_kernel_result', None) is not None})"
                )

            src_shape = list(getattr(expand_op, "src_shape", None) or parent.shape)
            tgt_shape = list(expand_op.shape)
            if src_shape is None or tgt_shape is None:
                raise ValueError("ExpandOP requires known shapes")
            if len(src_shape) != len(tgt_shape):
                raise ValueError(
                    f"ExpandOP rank mismatch: src={src_shape} tgt={tgt_shape}"
                )

            def _strides(shape):
                strides = [1] * len(shape)
                stride = 1
                for i in range(len(shape) - 1, -1, -1):
                    strides[i] = stride
                    stride *= int(shape[i])
                return strides

            src_strides = _strides(src_shape)
            tgt_strides = _strides(tgt_shape)
            tgt_size = reduce(mul, tgt_shape, 1)

            kernel_name = f"VecExpand_{self.kernel_id}"
            rank = len(tgt_shape)
            template = tensorops_backend.get_kernel_source_by_name("VecExpandTemplate")

            custom_src_for_rust = re.sub(
                r"#define\s+RANK\s+\d+",
                f"#define RANK {rank}",
                template,
                count=1,
            )
            custom_src_for_rust = re.sub(
                r"__kernel\s+void\s+VecExpandTemplate",
                f"__kernel void {kernel_name}",
                custom_src_for_rust,
                count=1,
            )

            kernel_type_obj = tensorops_backend.KernelType.custom(kernel_name)

            py_inputs = []

            # Parent tensor: device dependency (LogicalInputSource) or host (DirectInput)
            # Prefer LogicalInputSource when the parent is produced by a kernel in this graph.
            actual_parent = parent
            while type(actual_parent).__name__ == "ShapeOP":
                actual_parent = getattr(actual_parent, "tensor1", actual_parent)

            source_kernel_id = TensorContext.current_context.kernel_lookup.get(
                actual_parent
            )
            exec_lim_kernels = getattr(
                TensorContext.current_context, "_exec_lim_kernels", 0
            )
            if source_kernel_id is not None and source_kernel_id >= exec_lim_kernels:
                ctx = TensorContext.current_context
                source_output_idx = None
                output_index_maps = getattr(ctx, "_kernel_output_index", None)
                if output_index_maps is not None and source_kernel_id < len(
                    output_index_maps
                ):
                    source_output_idx = output_index_maps[source_kernel_id].get(
                        actual_parent
                    )
                if source_output_idx is None:
                    source_kernel_op_list = ctx.kernels[source_kernel_id]
                    try:
                        source_output_idx = source_kernel_op_list.index(actual_parent)
                    except ValueError:
                        raise Exception(
                            f"ExpandOP parent not found in producing kernel op list (kernel {source_kernel_id}): {_describe_tensor(actual_parent)}"
                        )
                py_inputs.append(
                    tensorops_backend.LogicalInputSource(
                        source_kernel_id, source_output_idx
                    )
                )
            elif (
                getattr(actual_parent, "_values", None) is not None
                or getattr(actual_parent, "_flat", None) is not None
            ):
                # Use host data only if it is already present; do not trigger lazy materialization.
                host_flat = getattr(actual_parent, "_flat", None)
                host_values = getattr(actual_parent, "_values", None)
                py_inputs.append(
                    tensorops_backend.DirectInput(
                        host_flat if host_flat is not None else host_values
                    )
                )
            else:
                raise Exception(
                    f"ExpandOP parent not found in kernel_lookup (kernel {self.kernel_id}) and has no host values: {_describe_tensor(actual_parent)}"
                )

            # Small metadata buffers (floats; cast to int in kernel)
            py_inputs.append(
                tensorops_backend.DirectInput([float(x) for x in src_shape])
            )
            py_inputs.append(
                tensorops_backend.DirectInput([float(x) for x in src_strides])
            )
            py_inputs.append(
                tensorops_backend.DirectInput([float(x) for x in tgt_shape])
            )
            py_inputs.append(
                tensorops_backend.DirectInput([float(x) for x in tgt_strides])
            )
            return tensorops_backend.KernelTensorOps(
                kernel_type=kernel_type_obj,
                kernel_id=self.kernel_id,
                num_output_bufs=1,
                custom_kernel_src=custom_src_for_rust,
                inputs=py_inputs,
                scalar_inputs=None,
                work_size_override=int(tgt_size),
            )

        is_predefined = len(self.op) == 1
        kernel_type_obj = (
            tensorops_backend.KernelType.predefined(op_map[type(self.op[0])].value)
            if is_predefined
            else tensorops_backend.KernelType.custom(
                fusor_kernel_name_if_custom or self.src or f"fused_{self.kernel_id}"
            )
        )
        custom_src_for_rust = self.src if not is_predefined else None

        py_inputs = []
        scalars = None
        input_tensors_for_kernel_object = []

        if is_predefined:
            if scalars := self.op[0].scalar_operands:
                if type(self.op[0]).__name__ == "GenericLog":
                    assert len(self.op[0].parents) == 2, (
                        "GenericLog expects two parents (base, input)"
                    )
                for scalar in scalars:
                    assert len(scalar.values) == 1, "Scalar must be a single value"
                input_tensors = set(self.op[0].parents).difference(set(scalars))
                for input_tensor in input_tensors:
                    input_tensors_for_kernel_object.append(input_tensor)
                input_tensors_for_kernel_object.extend(scalars)
            else:
                # For other predefined ops, include all parents
                input_tensors_for_kernel_object.extend(self.op[0].parents)
        else:
            # Fused kernel handling (unchanged)
            seen_internal_op_ids = {id(op_in_fuse) for op_in_fuse in self.op}
            temp_external_inputs_map = {}
            for op_in_fuse in self.op:
                for parent_tensor_obj in op_in_fuse.parents:
                    if id(parent_tensor_obj) not in seen_internal_op_ids:
                        if id(parent_tensor_obj) not in temp_external_inputs_map:
                            temp_external_inputs_map[id(parent_tensor_obj)] = (
                                parent_tensor_obj
                            )
            sorted_external_input_items = sorted(temp_external_inputs_map.items())
            input_tensors_for_kernel_object.extend(
                [tensor for _, tensor in sorted_external_input_items]
            )

        for input_tensor in input_tensors_for_kernel_object:
            # Prefer LogicalInputSource when possible, even if the tensor currently
            # has `.values` populated (to avoid copying large buffers back into Rust).
            actual_op_for_lookup = input_tensor
            while type(actual_op_for_lookup).__name__ == "ShapeOP":
                actual_op_for_lookup = getattr(
                    actual_op_for_lookup, "tensor1", actual_op_for_lookup
                )

            source_kernel_id = TensorContext.current_context.kernel_lookup.get(
                actual_op_for_lookup
            )
            exec_lim_kernels = getattr(
                TensorContext.current_context, "_exec_lim_kernels", 0
            )
            if source_kernel_id is not None and source_kernel_id >= exec_lim_kernels:
                ctx = TensorContext.current_context
                source_output_idx = None
                output_index_maps = getattr(ctx, "_kernel_output_index", None)
                if output_index_maps is not None and source_kernel_id < len(
                    output_index_maps
                ):
                    source_output_idx = output_index_maps[source_kernel_id].get(
                        actual_op_for_lookup
                    )
                if source_output_idx is None:
                    source_kernel_op_list = ctx.kernels[source_kernel_id]
                    try:
                        source_output_idx = source_kernel_op_list.index(
                            actual_op_for_lookup
                        )
                    except ValueError:
                        print(
                            f"Error details: actual_op_for_lookup={actual_op_for_lookup}, type={type(actual_op_for_lookup)}, source_kernel_id={source_kernel_id}, source_kernel_op_list={[str(o) for o in source_kernel_op_list]}"
                        )
                        raise Exception(
                            f"Parent OP {actual_op_for_lookup} (logical input to kernel {self.kernel_id}) "
                            f"not found in its own producing kernel's op list (kernel {source_kernel_id})."
                        )
                py_inputs.append(
                    tensorops_backend.LogicalInputSource(
                        source_kernel_id, source_output_idx
                    )
                )
            elif (
                getattr(actual_op_for_lookup, "_values", None) is not None
                or getattr(actual_op_for_lookup, "_flat", None) is not None
            ):
                # Use host data only if it is already present; do not trigger lazy materialization.
                host_flat = getattr(actual_op_for_lookup, "_flat", None)
                host_values = getattr(actual_op_for_lookup, "_values", None)
                py_inputs.append(
                    tensorops_backend.DirectInput(
                        host_flat if host_flat is not None else host_values
                    )
                )
            else:
                raise Exception(
                    f"Logical input {actual_op_for_lookup} (kernel {self.kernel_id}) not found in kernel_lookup and has no host values."
                )
        return tensorops_backend.KernelTensorOps(
            kernel_type=kernel_type_obj,
            kernel_id=self.kernel_id,
            num_output_bufs=len(self.op),
            custom_kernel_src=custom_src_for_rust,
            inputs=py_inputs if py_inputs else None,
            scalar_inputs=scalars if scalars else None,
        )

    def __repr__(self) -> str:
        return f"Kernel(kernel_id={self.kernel_id}, num_ops={len(self.op)}, type={self.op})"


DEBUG_FUSION_KERNEL_SRC = False


class Fusor:
    def __init__(self, fuse_ops: List[OP]) -> None:
        assert all(isinstance(op, OP) for op in fuse_ops), (
            "All items in fuse_ops must be OP instances"
        )
        self.fuse_ops = fuse_ops
        self.kernel_name = "custom_fused_" + uuid.uuid4().hex[:10]
        self.kernel_instructions: str | None = None

    def build_kernel(self) -> Tuple[str, str]:
        """Build the fused OpenCL kernel source.

        Note: We still emit one global output buffer per op so downstream kernels
        (including backward graph ops) can reference intermediates. The key perf
        win here is computing intermediates into local temps and writing them once
        (so later ops reuse registers instead of re-reading from global memory).
        """
        if self.kernel_instructions:
            return self.kernel_instructions, self.kernel_name

        op_output_buffer_map: Dict[int, str] = {
            id(op): f"output_buf_{i}" for i, op in enumerate(self.fuse_ops)
        }

        # External inputs are any parent tensors not produced inside this fused kernel.
        # Keep the same ordering strategy as Kernel.convert_kernel (sorted by id).
        external_input_ids: List[int] = []
        external_seen: set[int] = set()
        for op in self.fuse_ops:
            for parent in op.parents:
                pid = id(parent)
                if pid in op_output_buffer_map:
                    continue
                if pid not in external_seen:
                    external_seen.add(pid)
                    external_input_ids.append(pid)
        external_input_ids.sort()

        input_vars_map: Dict[int, str] = {
            pid: f"v_in_{i}" for i, pid in enumerate(external_input_ids)
        }

        op_temp_var_map: Dict[int, str] = {}
        fused_body: List[str] = []

        for i, op_instance in enumerate(self.fuse_ops):
            op_type = type(op_instance)
            kernel_enum_val = op_map.get(op_type)
            snippet_rhs = pattern.get(kernel_enum_val)
            if not snippet_rhs:
                raise NotImplementedError(
                    f"Operation {op_type.__name__} is not supported in fusion or its snippet is missing."
                )

            expr = snippet_rhs

            # Parent value expressions in parent order.
            parent_exprs: List[str] = []
            for parent in op_instance.parents:
                pid = id(parent)
                if pid in op_output_buffer_map:
                    temp = op_temp_var_map.get(pid)
                    if temp is not None:
                        parent_exprs.append(temp)
                    else:
                        parent_exprs.append(f"{op_output_buffer_map[pid]}[gid]")
                else:
                    parent_exprs.append(f"{input_vars_map[pid]}[gid]")

            # Substitute placeholders like A[gid], B[gid] according to parent order.
            raw_placeholders = re.findall(r"\b([a-zA-Z_]+)\[gid\]", expr)
            placeholders: List[str] = []
            for ph in raw_placeholders:
                if ph not in placeholders:
                    placeholders.append(ph)

            for idx, ph in enumerate(placeholders):
                if idx >= len(parent_exprs):
                    raise ValueError(
                        f"Not enough inputs for placeholders in snippet for {op_type.__name__}"
                    )
                expr = re.sub(
                    r"\b" + re.escape(ph) + r"\[gid\]", parent_exprs[idx], expr
                )

            # Some unary snippets use 'val' rather than A[gid].
            if re.search(r"\bval\b", expr) and parent_exprs:
                expr = re.sub(r"\bval\b", parent_exprs[0], expr)

            # GenericLog snippet uses scalar token 'base'. In fusion we pass base as a 1-element buffer.
            if isinstance(op_instance, GenericLog):
                base_parent = op_instance.parents[0]
                base_id = id(base_parent)
                if base_id in op_output_buffer_map:
                    base_expr = op_temp_var_map.get(
                        base_id, f"{op_output_buffer_map[base_id]}[gid]"
                    )
                else:
                    base_expr = f"{input_vars_map[base_id]}[0]"
                expr = re.sub(r"\bbase\b", base_expr, expr)

            # LeakyReLU snippet uses scalar token 'alpha'. In fusion we pass alpha as a 1-element buffer.
            if isinstance(op_instance, LeakyReLU):
                if len(op_instance.parents) < 2:
                    raise ValueError("LeakyReLU expects alpha as a second parent")
                alpha_parent = op_instance.parents[1]
                alpha_id = id(alpha_parent)
                if alpha_id in op_output_buffer_map:
                    alpha_expr = op_temp_var_map.get(
                        alpha_id, f"{op_output_buffer_map[alpha_id]}[gid]"
                    )
                else:
                    alpha_expr = f"{input_vars_map[alpha_id]}[0]"
                expr = re.sub(r"\balpha\b", alpha_expr, expr)

            out_buf = op_output_buffer_map[id(op_instance)]
            temp_name = f"t{i}"
            fused_body.append(f"    float {temp_name} = {expr};")
            fused_body.append(f"    {out_buf}[gid] = {temp_name};")
            op_temp_var_map[id(op_instance)] = temp_name

        # Kernel signature: external input buffers, then one output buffer per op.
        kernel_declaration = f"__kernel void {self.kernel_name}(\n"
        kernel_args: List[str] = []
        for pid in external_input_ids:
            kernel_args.append(f"    __global const float* {input_vars_map[pid]}")
        for op in self.fuse_ops:
            kernel_args.append(f"    __global float* {op_output_buffer_map[id(op)]}")

        self.kernel_instructions = (
            kernel_declaration
            + ",\n".join(kernel_args)
            + "\n) {\n    int gid = get_global_id(0);\n"
            + "\n".join(fused_body)
            + "\n}\n"
        )
        if DEBUG_FUSION_KERNEL_SRC:
            print(self.kernel_instructions)
        return self.kernel_instructions, self.kernel_name
