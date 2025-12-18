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
        if type(self.op[0]).__name__ == "ShapeOP":
            return None

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
            if input_tensor.values is not None:
                py_inputs.append(tensorops_backend.DirectInput(input_tensor.flat))
            else:
                actual_op_for_lookup = input_tensor
                source_kernel_id = TensorContext.current_context.kernel_lookup.get(
                    actual_op_for_lookup
                )
                if source_kernel_id is None:
                    raise Exception(
                        f"Logical input {actual_op_for_lookup} (kernel {self.kernel_id}) not found in kernel_lookup."
                    )
                source_kernel_op_list = TensorContext.current_context.kernels[
                    source_kernel_id
                ]
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
                lis = tensorops_backend.LogicalInputSource(
                    source_kernel_id, source_output_idx
                )
                py_inputs.append(lis)
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


class Fusor:
    def __init__(self, fuse_ops: List[OP]) -> None:
        assert all(isinstance(op, OP) for op in fuse_ops), (
            "All items in fuse_ops must be OP instances"
        )
        self.fuse_ops = fuse_ops  # List of OP objects to be fused
        self.kernel_name = "custom_fused_" + uuid.uuid4().hex[:10]
        self.kernel_instructions: str | None = None

    def build_kernel(self) -> Tuple[str, str]:
        """
        Builds the complete, fused OpenCL kernel source code.

        Each operation in the fusion group writes its result to a dedicated output buffer.
        These buffers can then be read by subsequent operations within the same kernel execution.
        The kernel signature lists input buffers first, then all output buffers.
        """
        if self.kernel_instructions:
            return self.kernel_instructions, self.kernel_name

        # input_vars_map: Maps external tensor IDs to their kernel buffer names (e.g., v_xyz)
        input_vars_map: Dict[int, str] = {}
        # op_output_buffer_map: Maps internal op IDs to their dedicated output buffer names (e.g., output_buf_0)
        op_output_buffer_map: Dict[int, str] = {
            id(op): f"output_buf_{i}" for i, op in enumerate(self.fuse_ops)
        }

        fused_instructions_body: List[str] = []

        for i, op_instance in enumerate(self.fuse_ops):
            op_type = type(op_instance)
            kernel_enum_val = op_map.get(op_type)
            snippet_rhs = pattern.get(kernel_enum_val)

            if not snippet_rhs:
                raise NotImplementedError(
                    f"Operation {op_type.__name__} is not supported in fusion or its snippet is missing."
                )

            current_op_snippet = snippet_rhs

            # Resolve input variables for the current op_instance
            # These are the actual kernel variable names (v_... for external, output_buf_... for internal)
            resolved_input_kernel_vars: List[str] = []
            for parent_tensor_obj in op_instance.parents:
                parent_id = id(parent_tensor_obj)

                if parent_id in op_output_buffer_map:
                    # Input is from a preceding op_instance in the same fused kernel
                    var_name = op_output_buffer_map[parent_id]
                elif parent_id in input_vars_map:
                    # Input is an external tensor, already mapped to a kernel input variable
                    var_name = input_vars_map[parent_id]
                else:
                    # This is a new external tensor input for the kernel
                    var_name = f"v_in_{uuid.uuid4().hex[:8]}"
                    input_vars_map[parent_id] = var_name

                resolved_input_kernel_vars.append(var_name)

            # Substitute placeholders like a[gid], b[gid] in the snippet
            # Placeholders in snippets are assumed to be 'a', 'b', etc., matching parent order
            placeholders_in_snippet = re.findall(
                r"\b([a-zA-Z_]+)\[gid\]", current_op_snippet
            )
            for placeholder_idx, placeholder_base_name in enumerate(
                placeholders_in_snippet
            ):
                if placeholder_idx < len(resolved_input_kernel_vars):
                    replacement_kernel_var = resolved_input_kernel_vars[placeholder_idx]
                    # Replace specific placeholder e.g., "a[gid]" with "v_in_xxxx[gid]" or "output_buf_y[gid]"
                    current_op_snippet = re.sub(
                        r"\b" + re.escape(placeholder_base_name) + r"\[gid\]",
                        f"{replacement_kernel_var}[gid]",
                        current_op_snippet,
                        count=1,  # Replace one by one to maintain order
                    )
                else:
                    raise ValueError(
                        f"Not enough input variables for placeholders in snippet for {op_type.__name__}"
                    )

            # Special handling for 'val' (often used in unary op snippets, refers to the first parent)
            if re.search(r"\bval\b", current_op_snippet) and resolved_input_kernel_vars:
                first_input_var = resolved_input_kernel_vars[0]
                # 'val' in snippet implies access to the buffer element at current gid
                current_op_snippet = re.sub(
                    r"\bval\b", f"{first_input_var}[gid]", current_op_snippet
                )

            # Special handling for LeakyReLU's alpha parameter
            # Assumes alpha is the second parent and passed as a 1-element buffer.
            if isinstance(op_instance, LeakyReLU):
                if len(resolved_input_kernel_vars) > 1:
                    alpha_buffer_kernel_name = resolved_input_kernel_vars[
                        1
                    ]  # Kernel name for the buffer holding alpha
                    # Snippet uses 'alpha'; replace with access to the first (and only) element of that buffer
                    current_op_snippet = re.sub(
                        r"\balpha\b",
                        f"{alpha_buffer_kernel_name}[0]",
                        current_op_snippet,
                    )
                else:
                    raise ValueError(
                        "LeakyReLU expects a second parent for its alpha value in fusion."
                    )

            # The result of this op_instance is written to its dedicated output buffer
            current_op_output_buffer_name = op_output_buffer_map[id(op_instance)]
            instruction_line = (
                f"    {current_op_output_buffer_name}[gid] = {current_op_snippet};"
            )
            fused_instructions_body.append(instruction_line)

        # Assemble the full kernel source string
        kernel_declaration = f"__kernel void {self.kernel_name}(\n"
        kernel_argument_declarations: List[str] = []

        # Ensure stable order for input buffer arguments from input_vars_map
        # Sort by the original tensor ID (the key in input_vars_map)

        sorted_external_input_vars = sorted(input_vars_map.items())
        for _, var_name in sorted_external_input_vars:
            kernel_argument_declarations.append(f"    __global const float* {var_name}")

        # Output buffer arguments (one for each op in the fusion, order based on self.fuse_ops)
        sorted_output_buffer_names = [
            op_output_buffer_map[id(op)] for op in self.fuse_ops
        ]
        for name in sorted_output_buffer_names:
            kernel_argument_declarations.append(f"    __global float* {name}")

        gid_setup = "\n) {\n    int gid = get_global_id(0);\n"
        kernel_body_str = "\n".join(fused_instructions_body)
        kernel_end = "\n}\n"

        self.kernel_instructions = (
            kernel_declaration
            + ",\n".join(kernel_argument_declarations)
            + gid_setup
            + kernel_body_str
            + kernel_end
        )
        return self.kernel_instructions, self.kernel_name
