import hashlib
import re
from enum import Enum
from functools import reduce
from operator import mul
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
    MatMul = tensorops_backend.PredefinedKernel.MatMul


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

    # Some predefined kernels (VecAdd/VecSub/VecElementMul/VecDiv) may include
    # scalar-broadcast indexing helpers (e.g. a_idx/b_idx) which are not valid
    # inside fused kernels (the fusor injects only the RHS expression).
    # Override these to the canonical gid-indexed expressions for fusion.
    matches[PredefinedKernel.VecAdd] = "A[gid] + B[gid]"
    matches[PredefinedKernel.VecSub] = "A[gid] - B[gid]"
    matches[PredefinedKernel.VecElementMul] = "A[gid] * B[gid]"
    matches[PredefinedKernel.VecDiv] = "A[gid] / B[gid]"

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

# Epilogue-fusible ops (elementwise, shape-preserving)
EPILOGUE_FUSIBLE_OPS = {
    Add,
    Sub,
    ElementMul,
    Div,
    Sin,
    Cos,
    Tanh,
    LeakyReLU,
    Pow,
    GenericLog,
}


class AnchoredKernel:
    """
    A kernel with a tiling anchor (e.g., MatMul) and optional epilogue ops.

    The anchor defines:
    - The iteration structure (2D NDRange for MatMul)
    - Memory layout (tiles in local memory)
    - Loop nests (K-loop, tile loads/stores)

    Epilogue ops are elementwise and fuse inside the anchor kernel:
    - They operate on the accumulator register before writeback
    - They do not require additional global reads
    - They must be shape-preserving

    Example:
        C = A @ B          # MatMul anchor
        D = C + bias       # Add epilogue (fused)
        E = relu(D)        # LeakyReLU epilogue (fused)
        Output: E (one kernel, zero intermediate writes)
    """

    def __init__(
        self,
        anchor_op: "MatMul",
        epilogue_ops: List["OP"],
        kernel_id_val: int,
        tile_size: int = 16,
    ):
        assert type(anchor_op).__name__ == "MatMul", "Only MatMul anchors supported"
        self.anchor = anchor_op
        self.epilogue_ops = epilogue_ops  # List of elementwise OPs to fuse
        self.kernel_id = kernel_id_val
        self.tile_size = tile_size

    @staticmethod
    def try_build_from_ops(
        ops: List["OP"], kernel_id_val: int, tile_size: int = 16
    ) -> Union["AnchoredKernel", None]:
        """
        Try to detect an anchored pattern in a list of sequential ops.
        Returns AnchoredKernel if found, None otherwise.

        Pattern:
        - ops[0] is MatMul
        - ops[1:] are all epilogue-fusible (elementwise, shape-preserving)
        - No new external dependencies introduced by epilogue ops
        """
        if not ops or type(ops[0]).__name__ != "MatMul":
            return None

        anchor = ops[0]
        epilogues = ops[1:] if len(ops) > 1 else []

        # Validate all epilogues are fusible
        for epi in epilogues:
            op_type = type(epi)
            if op_type not in EPILOGUE_FUSIBLE_OPS:
                return None  # Not an epilogue-fusible op

            # Epilogue must have only one parent (the previous op in sequence or anchor)
            # OR it can be a binary op with a side input.
            # Special-case LeakyReLU: it may carry `alpha` as a second scalar parent.
            if len(epi.parents) != 1:
                if (
                    op_type in (Add, Sub, ElementMul, Div, Pow)
                    and len(epi.parents) == 2
                ):
                    pass  # Allowed
                elif op_type is LeakyReLU and len(epi.parents) == 2:
                    # Allow scalar alpha (len 1) as the second parent.
                    alpha_parent = epi.parents[1]
                    while type(alpha_parent).__name__ in ("ShapeOP", "StopGrad"):
                        alpha_parent = getattr(alpha_parent, "tensor1", alpha_parent)
                    alpha_cap = getattr(alpha_parent, "capacity", None)
                    alpha_shape = getattr(alpha_parent, "shape", None)
                    if alpha_cap == 1 or alpha_shape in ((), (1,)):
                        pass
                    else:
                        return None
                else:
                    return None  # Multiple dependencies = new external input

        ak = AnchoredKernel(anchor, epilogues, kernel_id_val, tile_size)
        return ak

    def generate_epilogue_code(self, M: int, N: int) -> Tuple[str, List["Tensor"]]:
        """
        Generate OpenCL code that fuses epilogue ops into the accumulator register.
        Returns a tuple: (code_string, list_of_side_input_tensors).
        """
        if not self.epilogue_ops:
            return "", []

        code_lines = []
        side_inputs = []
        acc_var = "acc"  # Tiled matmul accumulates into 'acc'

        prev_op = self.anchor

        for i, op in enumerate(self.epilogue_ops):
            op_type = type(op)
            side_input = None
            side_input_name = f"side_input_{i}"
            acc_is_first = True

            # LeakyReLU may have a scalar alpha parent; treat it as a scalar (either embedded
            # constant or a 1-element side input) and do not treat it as a binary epilogue.
            leaky_alpha_expr = None
            if op_type is LeakyReLU and len(op.parents) == 2:
                alpha_parent = op.parents[1]
                while type(alpha_parent).__name__ in ("ShapeOP", "StopGrad"):
                    alpha_parent = getattr(alpha_parent, "tensor1", alpha_parent)

                # Prefer embedding as a constant if we can cheaply read it.
                try:
                    if getattr(alpha_parent, "capacity", None) == 1:
                        alpha_val = float(alpha_parent.item())
                        leaky_alpha_expr = f"{alpha_val}f"
                except Exception:
                    leaky_alpha_expr = None

                # Fallback: pass alpha as a scalar side input.
                if leaky_alpha_expr is None:
                    side_inputs.append(alpha_parent)
                    leaky_alpha_expr = f"side_input_{len(side_inputs) - 1}[0]"

            if len(op.parents) == 2:
                if op.parents[0] is prev_op:
                    side_input = op.parents[1]
                    acc_is_first = True
                elif op.parents[1] is prev_op:
                    side_input = op.parents[0]
                    acc_is_first = False
                else:
                    # Should not happen if try_build_from_ops is correct
                    pass

            idx_expr = "global_col"  # Default
            if side_input:
                # Unwrap ShapeOPs (e.g. Expand) to get the underlying data tensor
                # This allows passing the smaller tensor (e.g. bias vector) to the kernel
                # instead of the expanded one, enabling efficient broadcasting.
                while type(side_input).__name__ in ("ShapeOP", "StopGrad"):
                    side_input = getattr(side_input, "tensor1", side_input)

                side_inputs.append(side_input)

                # Determine index expression based on shape
                shape = side_input.shape
                # Simple broadcasting logic for 2D (M, N) output
                if shape == (M, N):
                    idx_expr = f"(global_row * {N} + global_col)"
                elif shape == (1, N) or shape == (N,):
                    idx_expr = "global_col"
                elif shape == (M, 1):
                    idx_expr = "global_row"
                elif shape == (1, 1) or shape == (1,):
                    idx_expr = "0"
                else:
                    # Fallback to flat index if shapes match flat size?
                    # Or assume full broadcast if not singleton?
                    # For safety, default to full index if size matches
                    if side_input.capacity == M * N:
                        idx_expr = f"(global_row * {N} + global_col)"
                    else:
                        # Hope for the best (likely (N,))
                        idx_expr = "global_col"

            side_expr = f"{side_input_name}[{idx_expr}]" if side_input else ""

            # Use pattern substitution
            if op_type in op_map:
                kernel_enum = op_map[op_type]
                if kernel_enum in pattern:
                    snippet = pattern[kernel_enum]
                    current_code = snippet

                    # Replace operands
                    if side_input:
                        if acc_is_first:
                            current_code = current_code.replace("A[gid]", acc_var)
                            current_code = current_code.replace("B[gid]", side_expr)
                        else:
                            current_code = current_code.replace("A[gid]", side_expr)
                            current_code = current_code.replace("B[gid]", acc_var)
                    else:
                        # Unary
                        current_code = current_code.replace("A[gid]", acc_var)

                    # Some unary snippets use 'val' rather than A[gid] (e.g. LeakyReLU).
                    current_code = re.sub(r"\bval\b", acc_var, current_code)

                    # Handle LeakyReLU alpha (constant or scalar side input)
                    if op_type is LeakyReLU:
                        if leaky_alpha_expr is not None:
                            current_code = current_code.replace(
                                "alpha", leaky_alpha_expr
                            )
                        else:
                            alpha = getattr(op, "alpha", 0.01)
                            current_code = current_code.replace("alpha", f"{alpha}f")

                    code_lines.append(f"{acc_var} = {current_code};")
                else:
                    # Fallback for ops not in pattern (should not happen for supported ops)
                    print(f"Warning: No pattern found for {op_type}")
            else:
                print(f"Warning: Op {op_type} not in op_map")

            prev_op = op

        return "\n        ".join(code_lines), side_inputs

    def convert_kernel(
        self,
        ctx: "TensorContext",
    ) -> tensorops_backend.KernelTensorOps:
        """
        Convert to a tiled MatMul with fused epilogue.
        """
        anchor = self.anchor
        M = anchor.tensor1.shape[0]
        N = anchor.tensor2.shape[1]
        K = anchor.tensor1.shape[1]

        # Tile size
        TILE_SIZE = self.tile_size
        local_mem_size = 2 * TILE_SIZE * TILE_SIZE * 4  # 2 tiles * 4 bytes per float

        # Epilogue code
        epilogue_code, side_inputs = self.generate_epilogue_code(M, N)
        if epilogue_code:
            epilogue_code = "// Epilogue fusion\n        " + epilogue_code
        else:
            epilogue_code = ""

        # Get kernel template and replace placeholder
        template = tensorops_backend.get_kernel_source_by_name("TiledMatMul_16x16")

        # Avoid replacing the comment
        template = template.replace("// EPILOGUE_PLACEHOLDER", "// PLACEHOLDER_COMMENT")

        # Replace TILE_SIZE definition
        # Assuming template has #define TILE_SIZE 16
        custom_src = template.replace(
            "#define TILE_SIZE 16", f"#define TILE_SIZE {TILE_SIZE}"
        )

        # Replace kernel name.
        # Must be globally unique within a compiled OpenCL program; otherwise we can
        # hit "conflicting types" errors when a predefined MatMul kernel and an
        # anchored custom MatMul (tile=16) both define TiledMatMul_16x16.
        kernel_name = f"TiledMatMul_{TILE_SIZE}x{TILE_SIZE}_k{self.kernel_id}"
        custom_src = custom_src.replace("TiledMatMul_16x16", kernel_name)

        custom_src = custom_src.replace("EPILOGUE_PLACEHOLDER", epilogue_code)

        # Update signature if side inputs exist
        if side_inputs:
            new_args = []
            for i in range(len(side_inputs)):
                new_args.append(f"__global const float* side_input_{i}")

            # Insert before A_tile to match Rust backend argument order (Inputs -> Local -> Outputs)
            custom_src = custom_src.replace(
                "__local float* A_tile", ", ".join(new_args) + ", __local float* A_tile"
            )

        # Use custom kernel name
        kernel_type_obj = tensorops_backend.KernelType.custom(kernel_name)

        # Build input list: A, B, then any epilogue deps
        py_inputs = []

        # A and B tensors
        actual_t1 = anchor.tensor1
        while type(actual_t1).__name__ in ("ShapeOP", "StopGrad"):
            actual_t1 = actual_t1.tensor1

        actual_t2 = anchor.tensor2
        while type(actual_t2).__name__ in ("ShapeOP", "StopGrad"):
            actual_t2 = actual_t2.tensor1

        source_kernel_id_t1 = ctx.kernel_lookup.get(actual_t1)
        source_kernel_id_t2 = ctx.kernel_lookup.get(actual_t2)
        exec_lim = getattr(ctx, "_exec_lim_kernels", 0)

        if (
            source_kernel_id_t1 is not None
            and source_kernel_id_t1 >= exec_lim
            and source_kernel_id_t1 != self.kernel_id
        ):
            py_inputs.append(
                tensorops_backend.LogicalInputSource(source_kernel_id_t1, 0)
            )
        elif (
            getattr(actual_t1, "_values", None) is not None
            or getattr(actual_t1, "_flat", None) is not None
        ):
            py_inputs.append(actual_t1._get_direct_input())
        else:
            raise ValueError("Cannot resolve input tensor for MatMul anchor")

        if (
            source_kernel_id_t2 is not None
            and source_kernel_id_t2 >= exec_lim
            and source_kernel_id_t2 != self.kernel_id
        ):
            py_inputs.append(
                tensorops_backend.LogicalInputSource(source_kernel_id_t2, 0)
            )
        elif (
            getattr(actual_t2, "_values", None) is not None
            or getattr(actual_t2, "_flat", None) is not None
        ):
            py_inputs.append(actual_t2._get_direct_input())
        else:
            raise ValueError("Cannot resolve input tensor for MatMul anchor")

        # Dimensions (M, N, K)
        py_inputs.append(tensorops_backend.DirectInput([float(M)]))
        py_inputs.append(tensorops_backend.DirectInput([float(N)]))
        py_inputs.append(tensorops_backend.DirectInput([float(K)]))

        # Add side inputs
        for inp in side_inputs:
            source_kernel_id = ctx.kernel_lookup.get(inp)
            if (
                source_kernel_id is not None
                and source_kernel_id >= exec_lim
                and source_kernel_id != self.kernel_id
            ):
                py_inputs.append(
                    tensorops_backend.LogicalInputSource(source_kernel_id, 0)
                )
            elif (
                getattr(inp, "_values", None) is not None
                or getattr(inp, "_flat", None) is not None
            ):
                py_inputs.append(inp._get_direct_input())
            else:
                raise ValueError("Cannot resolve side input tensor for MatMul epilogue")

        # Create KernelTensorOps with 2D workgroup
        M_blocks = (M + TILE_SIZE - 1) // TILE_SIZE
        N_blocks = (N + TILE_SIZE - 1) // TILE_SIZE
        global_x = M_blocks * TILE_SIZE
        global_y = N_blocks * TILE_SIZE

        kernel_obj = tensorops_backend.KernelTensorOps(
            kernel_type=kernel_type_obj,
            kernel_id=self.kernel_id,
            num_output_bufs=1,
            custom_kernel_src=custom_src,
            inputs=py_inputs,
            scalar_inputs=None,
            # IMPORTANT: runtime otherwise infers output size from the largest input buffer.
            # For MatMul, that would be len(A) which is wrong; output is M*N.
            work_size_override=int(M * N),
            global_work_size_2d=(global_x, global_y),
            local_work_size=(TILE_SIZE, TILE_SIZE),
        )

        return kernel_obj


class Kernel:
    def __init__(
        self, op_list: List[OP], custom_src_str: Union[str, None], kernel_id_val: int
    ):
        self.op: List[
            OP
        ] = op_list  # List of OP objects in this kernel (1 for predefined, multiple for fused)
        self.src: Union[
            str, None
        ] = custom_src_str  # Custom source from Fusor, or None for predefined

        # self.deps is not directly used for KernelTensorOps creation anymore in the new way
        # self.inputs (direct_initial_inputs) is used for direct_inputs field

        self.kernel_id: int = kernel_id_val

        # This lambda will be called from TensorContext.finalise

    def convert_kernel(
        self,
        fusor_kernel_name_if_custom: Union[str, None],
        ctx: "TensorContext",
    ) -> tensorops_backend.KernelTensorOps:
        if type(self.op[0]).__name__ in ("ShapeOP",):
            return None

        if type(self.op[0]).__name__ == "PermuteOP":
            permute_op = self.op[0]
            parent = permute_op.tensor1
            dims = permute_op.dims

            src_shape = parent.shape
            tgt_shape = permute_op.shape

            def _strides(shape):
                strides = [1] * len(shape)
                stride = 1
                for i in range(len(shape) - 1, -1, -1):
                    strides[i] = stride
                    stride *= int(shape[i])
                return strides

            parent_strides = _strides(src_shape)
            # Permute parent strides to match target order
            src_strides = [parent_strides[d] for d in dims]

            # Use tgt_shape as src_shape to avoid broadcast logic in kernel
            kernel_src_shape = tgt_shape

            tgt_strides = _strides(tgt_shape)
            tgt_size = reduce(mul, tgt_shape, 1)

            rank = len(tgt_shape)
            kernel_name = f"VecPermute_r{rank}"
            template = tensorops_backend.get_kernel_source_by_name("VecPermuteTemplate")

            custom_src_for_rust = re.sub(
                r"#define\s+RANK\s+\d+",
                f"#define RANK {rank}",
                template,
                count=1,
            )
            custom_src_for_rust = re.sub(
                r"__kernel\s+void\s+VecPermuteTemplate",
                f"__kernel void {kernel_name}",
                custom_src_for_rust,
                count=1,
            )

            kernel_type_obj = tensorops_backend.KernelType.custom(kernel_name)

            py_inputs = []

            # Parent tensor logic
            actual_parent = parent
            while type(actual_parent).__name__ in ("ShapeOP", "StopGrad"):
                actual_parent = getattr(actual_parent, "tensor1", actual_parent)

            source_kernel_id = ctx.kernel_lookup.get(actual_parent)
            exec_lim_kernels = getattr(ctx, "_exec_lim_kernels", 0)

            if (
                source_kernel_id is not None
                and source_kernel_id >= exec_lim_kernels
                and source_kernel_id != self.kernel_id
            ):
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
                        raise Exception("PermuteOP parent not found")
                py_inputs.append(
                    tensorops_backend.LogicalInputSource(
                        source_kernel_id, source_output_idx
                    )
                )
            elif (
                getattr(actual_parent, "_values", None) is not None
                or getattr(actual_parent, "_flat", None) is not None
            ):
                host_flat = getattr(actual_parent, "_flat", None)
                host_values = getattr(actual_parent, "_values", None)
                py_inputs.append(
                    tensorops_backend.DirectInput(
                        host_flat if host_flat is not None else host_values
                    )
                )

            # Metadata buffers
            py_inputs.append(
                tensorops_backend.DirectInput([float(x) for x in kernel_src_shape])
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

        if type(self.op[0]).__name__ == "ExpandOP":
            expand_op = self.op[0]
            parent = expand_op.tensor1

            def _describe_tensor(t) -> str:
                # Avoid calling __repr__ / .values (can materialise huge buffers)
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

            rank = len(tgt_shape)
            kernel_name = f"VecExpand_r{rank}"
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
            while type(actual_parent).__name__ in ("ShapeOP", "StopGrad"):
                actual_parent = getattr(actual_parent, "tensor1", actual_parent)

            source_kernel_id = ctx.kernel_lookup.get(actual_parent)
            exec_lim_kernels = getattr(ctx, "_exec_lim_kernels", 0)
            # Skip internal dependencies within the same fused kernel (not applicable for ExpandOP)
            # Create LogicalInputSource for external kernel dependencies
            if (
                source_kernel_id is not None
                and source_kernel_id >= exec_lim_kernels
                and source_kernel_id != self.kernel_id
            ):
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
                # ExpandOP must always receive its data input as the first kernel arg.
                # If we can't reference it via LogicalInputSource and it's not already
                # host-resident, force materialization and pass it as DirectInput.
                try:
                    _ = actual_parent.values
                    py_inputs.append(actual_parent._get_direct_input())
                except Exception as e:
                    raise Exception(
                        f"ExpandOP parent could not be resolved for kernel input: {_describe_tensor(actual_parent)}; materialization failed: {e}"
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

        if type(self.op[0]).__name__ == "MatMul":
            matmul_op = self.op[0]
            # Assume parents are [A, B]
            A = matmul_op.parents[0]
            B = matmul_op.parents[1]

            shape_a = A.shape
            shape_b = B.shape

            M = shape_a[-2]
            K = shape_a[-1]
            N = shape_b[-1]

            # Output size
            tgt_shape = list(shape_a[:-2]) + [M, N]
            tgt_size = reduce(mul, tgt_shape, 1)

            kernel_type_obj = tensorops_backend.KernelType.predefined(
                tensorops_backend.PredefinedKernel.MatMul
            )

            py_inputs = []

            # Add A and B as inputs
            for parent in [A, B]:
                actual_parent = parent
                while type(actual_parent).__name__ in ("ShapeOP", "StopGrad"):
                    actual_parent = getattr(actual_parent, "tensor1", actual_parent)

                source_kernel_id = ctx.kernel_lookup.get(actual_parent)
                exec_lim_kernels = getattr(ctx, "_exec_lim_kernels", 0)

                if (
                    source_kernel_id is not None
                    and source_kernel_id >= exec_lim_kernels
                    and source_kernel_id != self.kernel_id
                ):
                    source_output_idx = None
                    output_index_maps = getattr(ctx, "_kernel_output_index", None)
                    if output_index_maps is not None and source_kernel_id < len(
                        output_index_maps
                    ):
                        source_output_idx = output_index_maps[source_kernel_id].get(
                            actual_parent
                        )
                    if source_output_idx is None:
                        raise Exception(
                            f"MatMul parent {actual_parent} is not available as an output of kernel {source_kernel_id} (likely fused)."
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
                    host_flat = getattr(actual_parent, "_flat", None)
                    host_values = getattr(actual_parent, "_values", None)
                    py_inputs.append(
                        tensorops_backend.DirectInput(
                            host_flat if host_flat is not None else host_values
                        )
                    )
                else:
                    raise Exception(f"MatMul input {actual_parent} not found")

            # Add M, N, K as scalar inputs (DirectInput buffers of size 1)
            py_inputs.append(tensorops_backend.DirectInput([float(M)]))
            py_inputs.append(tensorops_backend.DirectInput([float(N)]))
            py_inputs.append(tensorops_backend.DirectInput([float(K)]))

            # Launch as a 2D tiled kernel (same convention as AnchoredKernel).
            TILE_SIZE = 16
            M_blocks = (M + TILE_SIZE - 1) // TILE_SIZE
            N_blocks = (N + TILE_SIZE - 1) // TILE_SIZE
            global_x = int(M_blocks * TILE_SIZE)
            global_y = int(N_blocks * TILE_SIZE)

            return tensorops_backend.KernelTensorOps(
                kernel_type=kernel_type_obj,
                kernel_id=self.kernel_id,
                num_output_bufs=1,
                custom_kernel_src=None,  # Predefined
                inputs=py_inputs,
                scalar_inputs=None,
                work_size_override=int(tgt_size),
                global_work_size_2d=(global_x, global_y),
                local_work_size=(TILE_SIZE, TILE_SIZE),
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

        # Optional scalar inputs attached to the KernelTensorOps (used for simple predefined ops)
        scalar_values = None
        # Reduce-op metadata to be passed as DirectInput scalars (pre, axis_len, post)
        reduce_scalar_direct_inputs: List[float] | None = None
        reduce_work_size: int | None = None

        if is_predefined:
            op_name = type(self.op[0]).__name__
            # Special handling for reductions: Sum/Max/Min require (pre, axis_len, post)
            if op_name in ("Sum", "Max", "Min"):
                # Data tensor is the single parent
                input_tensors_for_kernel_object.append(self.op[0].parents[0])
                pre = float(getattr(self.op[0], "pre_axis"))
                axis_len = float(getattr(self.op[0], "axis_len"))
                post = float(getattr(self.op[0], "post_axis"))
                reduce_scalar_direct_inputs = [pre, axis_len, post]
                reduce_work_size = int(pre * post)
            elif scalars := self.op[0].scalar_operands:
                # Generic handling for predefined ops with scalar operands (e.g., GenericLog, LeakyReLU)
                if op_name == "GenericLog":
                    assert (
                        len(self.op[0].parents) == 2
                    ), "GenericLog expects two parents (base, input)"
                for scalar in scalars:
                    assert len(scalar.values) == 1, "Scalar must be a single value"

                scalar_values = []
                for parent in self.op[0].parents:
                    if parent in scalars:
                        scalar_values.append([float(parent.values[0])])
                    else:
                        input_tensors_for_kernel_object.append(parent)
            else:
                # For other predefined ops, include all parents
                input_tensors_for_kernel_object.extend(self.op[0].parents)
        else:
            # Fused kernel handling: collect external inputs, unwrapping ShapeOPs
            seen_internal_op_ids = {id(op_in_fuse) for op_in_fuse in self.op}
            seen_external_ids = set()

            for op_in_fuse in self.op:
                for parent_tensor_obj in op_in_fuse.parents:
                    # Unwrap ShapeOP/StopGrad to get the actual producer
                    actual_parent = parent_tensor_obj
                    while type(actual_parent).__name__ in ("ShapeOP", "StopGrad"):
                        actual_parent = getattr(actual_parent, "tensor1", actual_parent)
                    # Check if the actual producer is internal to this fused kernel
                    if id(actual_parent) not in seen_internal_op_ids:
                        pid = id(parent_tensor_obj)
                        if pid not in seen_external_ids:
                            seen_external_ids.add(pid)
                            input_tensors_for_kernel_object.append(parent_tensor_obj)

        output_index_maps = getattr(ctx, "_kernel_output_index", None)
        exec_lim_kernels = getattr(ctx, "_exec_lim_kernels", 0)

        for input_tensor in input_tensors_for_kernel_object:
            # Prefer LogicalInputSource when possible, even if the tensor currently
            # has `.values` populated (to avoid copying large buffers back into Rust).
            actual_op_for_lookup = input_tensor
            while type(actual_op_for_lookup).__name__ in ("ShapeOP", "StopGrad"):
                actual_op_for_lookup = getattr(
                    actual_op_for_lookup, "tensor1", actual_op_for_lookup
                )

            source_kernel_id = ctx.kernel_lookup.get(actual_op_for_lookup)
            # Skip internal dependencies within the same fused kernel
            # (these are handled by the fusion code itself, not as kernel inputs)
            if source_kernel_id is not None and source_kernel_id == self.kernel_id:
                continue
            # Create LogicalInputSource for external kernel dependencies
            if source_kernel_id is not None and source_kernel_id >= exec_lim_kernels:
                source_output_idx = None
                if output_index_maps is not None and source_kernel_id < len(
                    output_index_maps
                ):
                    source_output_idx = output_index_maps[source_kernel_id].get(
                        actual_op_for_lookup
                    )
                if source_output_idx is None:
                    raise Exception(
                        f"Parent OP {actual_op_for_lookup} is not available as an output of kernel {source_kernel_id} (likely fused)."
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
                py_inputs.append(actual_op_for_lookup._get_direct_input())
            else:
                # As a fallback (common in backward passes), force materialization
                # of the producer's host values and pass them as DirectInput.
                try:
                    _ = actual_op_for_lookup.values
                    py_inputs.append(actual_op_for_lookup._get_direct_input())
                except Exception as e:
                    raise Exception(
                        f"Logical input {actual_op_for_lookup} (kernel {self.kernel_id}) not found in kernel_lookup and has no host values; materialization failed: {e}"
                    )
        # Attach reduce-op scalar DirectInputs at the end (so resolved_inputs len>=4)
        if reduce_scalar_direct_inputs is not None:
            py_inputs.append(
                tensorops_backend.DirectInput([reduce_scalar_direct_inputs[0]])
            )
            py_inputs.append(
                tensorops_backend.DirectInput([reduce_scalar_direct_inputs[1]])
            )
            py_inputs.append(
                tensorops_backend.DirectInput([reduce_scalar_direct_inputs[2]])
            )

        res = tensorops_backend.KernelTensorOps(
            kernel_type=kernel_type_obj,
            kernel_id=self.kernel_id,
            num_output_bufs=len(self.op),
            custom_kernel_src=custom_src_for_rust,
            inputs=py_inputs if py_inputs else None,
            scalar_inputs=scalar_values,
            work_size_override=reduce_work_size,
        )

        return res

    def __repr__(self) -> str:
        return f"Kernel(kernel_id={self.kernel_id}, num_ops={len(self.op)}, type={self.op})"


DEBUG_FUSION_KERNEL_SRC = False


class Fusor:
    # Class-level cache: maps fusion signature -> (kernel_source, kernel_name)
    _kernel_cache: Dict[str, Tuple[str, str]] = {}

    def __init__(self, fuse_ops: List[OP]) -> None:
        assert all(
            isinstance(op, OP) for op in fuse_ops
        ), "All items in fuse_ops must be OP instances"
        self.fuse_ops = fuse_ops
        self.kernel_name: str | None = None
        self.kernel_instructions: str | None = None

    def build_kernel(self) -> Tuple[str, str]:
        """Build the fused OpenCL kernel source.

        Note: We still emit one global output buffer per op so downstream kernels
        (including backward graph ops) can reference intermediates. The key perf
        win here is computing intermediates into local temps and writing them once
        (so later ops reuse registers instead of re-reading from global memory).
        """
        if self.kernel_instructions:
            assert self.kernel_name is not None
            return self.kernel_instructions, self.kernel_name

        # Generate a stable signature for this fusion pattern to check cache.
        # Use op types and parent count to identify the fusion structure.
        fusion_signature = "|".join(
            f"{type(op).__name__}:{len(op.parents)}" for op in self.fuse_ops
        )

        # Check class-level Python cache first (fastest)
        if fusion_signature in Fusor._kernel_cache:
            cached_src, cached_name = Fusor._kernel_cache[fusion_signature]
            self.kernel_instructions = cached_src
            self.kernel_name = cached_name
            return cached_src, cached_name

        op_output_buffer_map: Dict[int, str] = {
            id(op): f"output_buf_{i}" for i, op in enumerate(self.fuse_ops)
        }

        # External inputs are any parent tensors not produced inside this fused kernel.
        # Keep the same ordering strategy as Kernel.convert_kernel (sorted by id).
        # Unwrap ShapeOPs to check if the underlying producer is internal.
        # Determine external inputs in a stable order.
        # Using insertion order (first-seen in graph traversal) avoids depending on
        # Python object ids, which vary between runs and destroy cache hit rates.
        external_inputs: List[OP] = []
        external_seen: set[int] = set()
        for op in self.fuse_ops:
            for parent in op.parents:
                # Unwrap ShapeOP/StopGrad to check if the actual producer is internal
                actual_parent = parent
                while type(actual_parent).__name__ in ("ShapeOP", "StopGrad"):
                    actual_parent = getattr(actual_parent, "tensor1", actual_parent)
                if id(actual_parent) in op_output_buffer_map:
                    continue
                pid = id(parent)
                if pid not in external_seen:
                    external_seen.add(pid)
                    external_inputs.append(parent)

        input_vars_map: Dict[int, str] = {
            id(p): f"v_in_{i}" for i, p in enumerate(external_inputs)
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
                # Unwrap ShapeOP to get the actual producer
                actual_parent = parent
                # Treat view/pass-through ops as transparent when resolving producers.
                # StopGrad is used by CrossEntropyLoss (detach max logits). If we don't
                # unwrap it here, fusion can incorrectly treat an internal producer as an
                # external input, generating wrong reads and NaNs.
                while type(actual_parent).__name__ in ("ShapeOP", "StopGrad"):
                    actual_parent = getattr(actual_parent, "tensor1", actual_parent)
                actual_pid = id(actual_parent)

                if actual_pid in op_output_buffer_map:
                    temp = op_temp_var_map.get(actual_pid)
                    if temp is not None:
                        parent_exprs.append(temp)
                    else:
                        parent_exprs.append(f"{op_output_buffer_map[actual_pid]}[gid]")
                else:
                    # Decide scalar vs vector by Python-side length for external inputs
                    try:
                        is_scalar = len(parent) == 1
                    except Exception:
                        is_scalar = False
                    idx = "[0]" if is_scalar else "[gid]"
                    parent_exprs.append(f"{input_vars_map[pid]}{idx}")

            # Substitute placeholders like A[gid], B[gid] according to parent order.
            raw_placeholders = re.findall(r"\b([a-zA-Z_]+)\[gid\]", expr)
            placeholders: List[str] = []
            for ph in raw_placeholders:
                if ph not in placeholders:
                    placeholders.append(ph)

            # Special handling for GenericLog: A[gid] should map to the second parent (input), not the first (base)
            current_parent_exprs = parent_exprs
            if isinstance(op_instance, GenericLog):
                current_parent_exprs = parent_exprs[1:]

            for idx, ph in enumerate(placeholders):
                if idx >= len(current_parent_exprs):
                    raise ValueError(
                        f"Not enough inputs for placeholders in snippet for {op_type.__name__}"
                    )
                expr = re.sub(
                    r"\b" + re.escape(ph) + r"\[gid\]", current_parent_exprs[idx], expr
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

            # Pow fusion special-case: handle scalar broadcasting for base/exponent
            # Build explicit pow(base_expr, exp_expr) using [0] for scalar inputs and [gid] otherwise.
            if op_type.__name__ == "Pow":
                if len(op_instance.parents) != 2:
                    raise ValueError("Pow expects two parents (base, exponent)")

                def _parent_expr(parent_op):
                    # Unwrap ShapeOP
                    actual = parent_op
                    while type(actual).__name__ in ("ShapeOP", "StopGrad"):
                        actual = getattr(actual, "tensor1", actual)
                    actual_id = id(actual)
                    # Internal producer -> temp or output_buf[gid]
                    if actual_id in op_output_buffer_map:
                        tmp = op_temp_var_map.get(actual_id)
                        return (
                            tmp
                            if tmp is not None
                            else f"{op_output_buffer_map[actual_id]}[gid]",
                            False,
                        )
                    # External input -> v_in_n[idx]
                    vid = id(parent_op)
                    # Decide scalar vs vector by Python-side length
                    try:
                        is_scalar = len(parent_op) == 1
                    except Exception:
                        is_scalar = False
                    idx = "[0]" if is_scalar else "[gid]"
                    return f"{input_vars_map[vid]}{idx}", is_scalar

                base_expr, _ = _parent_expr(op_instance.parents[0])
                exp_expr, _ = _parent_expr(op_instance.parents[1])
                expr = f"pow({base_expr}, {exp_expr})"

            out_buf = op_output_buffer_map[id(op_instance)]
            temp_name = f"t{i}"
            fused_body.append(f"    float {temp_name} = {expr};")
            fused_body.append(f"    {out_buf}[gid] = {temp_name};")
            op_temp_var_map[id(op_instance)] = temp_name

        # Kernel signature: external input buffers, then one output buffer per op.
        kernel_declaration = "__kernel void KERNEL_NAME(\n"
        kernel_args: List[str] = []
        for p in external_inputs:
            pid = id(p)
            kernel_args.append(f"    __global const float* {input_vars_map[pid]}")
        for op in self.fuse_ops:
            kernel_args.append(f"    __global float* {op_output_buffer_map[id(op)]}")

        kernel_src_with_placeholder = (
            kernel_declaration
            + ",\n".join(kernel_args)
            + "\n) {\n    int gid = get_global_id(0);\n"
            + "\n".join(fused_body)
            + "\n}\n"
        )

        # Deterministic kernel name for stable Rust program_cache hits.
        # Hash the kernel source with a placeholder name so the name doesn't
        # self-influence the hash.
        h = hashlib.sha1(kernel_src_with_placeholder.encode("utf-8")).hexdigest()[:16]
        self.kernel_name = f"custom_fused_{h}"
        self.kernel_instructions = kernel_src_with_placeholder.replace(
            "KERNEL_NAME", self.kernel_name
        )

        # Cache the result for future fusion of the same pattern
        Fusor._kernel_cache[fusion_signature] = (
            self.kernel_instructions,
            self.kernel_name,
        )

        # if DEBUG_FUSION_KERNEL_SRC:
        # print(self.kernel_instructions)
        return self.kernel_instructions, self.kernel_name
