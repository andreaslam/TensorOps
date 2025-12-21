import os
import platform
import time
from typing import Any, Dict, List, Tuple

try:
    import mlx.core as mx  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "MLX package not installed. Please install MLX before using this backend."
    ) from e
import numpy as np


# Debug configuration via environment variables (mirrors backend.py)
def _env_flag(name: str) -> bool:
    v = os.getenv(name)
    if v is None:
        return False
    v = v.strip().lower()
    return v not in ("", "0", "false", "no", "off")


DEBUG_ALL = _env_flag("TENSOROPS_DEBUG") or _env_flag("TENSOROPS_DEBUG_MLX")
DEBUG_KERNEL_SRC = DEBUG_ALL or _env_flag("TENSOROPS_DEBUG_KERNEL_SRC")
DEBUG_KERNEL_INPUTS = DEBUG_ALL or _env_flag("TENSOROPS_DEBUG_INPUTS")
DEBUG_TIMINGS = DEBUG_ALL or _env_flag("TENSOROPS_DEBUG_TIMINGS")


def _dbg_print(tag: str, msg: str) -> None:
    # Print if either DEBUG_ALL is set or the specific tag's debug flag is set
    if (
        DEBUG_ALL
        or (tag == "timings" and DEBUG_TIMINGS)
        or (tag == "inputs" and DEBUG_KERNEL_INPUTS)
    ):
        print(f"[TensorOps:mlx:{tag}] {msg}")


class MLXRuntime:
    """MLX backend runtime adapter for macOS.

    This module provides a device-aware runtime for macOS using MLX (MLX framework).
    Maps TensorOps kernels to MLX operations and handles graph execution.
    Enable via environment variable: TENSOROPS_BACKEND=mlx
    """

    def __init__(self) -> None:
        if platform.system().lower() != "darwin":
            raise RuntimeError("MLX backend is only supported on macOS (Darwin).")

        self.mx = None  # Lazy import
        self._kernel_outputs: Dict[
            int, List[Any]
        ] = {}  # Cache kernel outputs for dependencies

    @staticmethod
    def _is_mx_array(val: Any) -> bool:
        """Best-effort check for an MLX array without importing numpy.

        MLX arrays live under the `mlx.core` module with class name `array`.
        Avoid converting them back into MLX via `mx.array(...)` to keep the
        computation graph intact and prevent premature evaluation.
        """

        return hasattr(val, "__class__") and getattr(
            val.__class__, "__module__", ""
        ).startswith("mlx")

    def _import_mlx(self):
        """Lazy import of MLX to avoid startup overhead."""
        if self.mx is None:
            import mlx.core as mx

            self.mx = mx
        return self.mx

    def _resolve_input(self, inp, outputs_cache: Dict[int, List[Any]]):
        """Resolve a kernel input (DirectInput or LogicalInputSource) to an MLX array."""
        mx = self._import_mlx()
        import tensorops_backend

        # Fast-path: already an MLX array (preserve graph node and laziness)
        if self._is_mx_array(inp):
            return inp

        # Check if it's a LogicalInputSource (reference to another kernel's output)
        if isinstance(inp, tensorops_backend.LogicalInputSource):
            # LogicalInputSource: fetch from cache
            src_id = inp.source_kernel_id
            src_idx = getattr(inp, "source_output_index", 0)

            # By wave-based execution design, dependency should already be complete
            if src_id not in outputs_cache or len(outputs_cache[src_id]) <= src_idx:
                raise RuntimeError(
                    f"Dependency {src_id} output {src_idx} not ready. "
                    f"Cache: {src_id in outputs_cache}, "
                    f"available: {len(outputs_cache.get(src_id, []))}"
                )

            output = outputs_cache[src_id][src_idx]
            if output is None:
                # Upstream kernel failed to build; propagating None into mx.array
                # causes TypeErrors. Raise here to let execute_graph mark this
                # kernel as failed and continue.
                raise RuntimeError(f"Upstream kernel {src_id} output {src_idx} is None")
            if self._is_mx_array(output):
                return output
            # Convert to MLX if needed
            if hasattr(output, "__array__"):
                return mx.array(output, dtype=mx.float32)
            elif hasattr(output, "tolist"):
                return mx.array(output.tolist(), dtype=mx.float32)
            else:
                return mx.array(output, dtype=mx.float32)

        # DirectInput: extract data directly
        if isinstance(inp, tensorops_backend.DirectInput):
            data = inp.data
            if self._is_mx_array(data):
                return data
            if isinstance(data, (list, np.ndarray)):
                return mx.array(data, dtype=mx.float32)
            elif isinstance(data, (bytes, bytearray, memoryview)):
                # Convert bytes to float32 array
                mv = memoryview(data)
                return mx.array(np.frombuffer(mv, dtype=np.float32), dtype=mx.float32)
            else:
                return mx.array(data, dtype=mx.float32)

        # Fallback: try to convert directly
        return mx.array(inp, dtype=mx.float32)

    def _map_kernel_to_mlx(
        self, kernel, outputs_cache: Dict[int, List[Any]]
    ) -> Tuple[Any, int]:
        """Map a TensorOps kernel to an MLX operation and execute it.

        Returns:
            tuple: (result_mlx_array, num_outputs) where num_outputs indicates
                   how many output buffers this kernel produces.
        """
        t_start = time.perf_counter() if DEBUG_TIMINGS else None

        kernel_id = getattr(kernel, "kernel_id", 0)
        kernel_type = getattr(kernel, "kernel_type", None)
        inputs = getattr(kernel, "inputs", [])
        scalar_inputs = getattr(kernel, "scalar_inputs", [])
        num_outputs = getattr(kernel, "num_output_bufs", 1)

        # Resolve input tensors
        resolved_inputs = []
        if inputs:
            for inp in inputs:
                resolved_inputs.append(self._resolve_input(inp, outputs_cache))

        if DEBUG_KERNEL_INPUTS:
            _dbg_print(
                "inputs",
                f"kernel#{kernel_id} kernel_type={kernel_type} inputs={len(resolved_inputs)} scalar_inputs={len(scalar_inputs)}",
            )

        # Map kernel type to MLX op
        if kernel_type is None:
            raise RuntimeError(f"Kernel {kernel_id} has no kernel_type")

        # Extract kernel type name (handle both enum and string representations)
        # The kernel_type could be KernelType.Custom or KernelType.Predefined(...)
        kernel_type_str = str(kernel_type).lower()

        # For debugging kernel type
        # print(f"Kernel {kernel_id}: type_str={kernel_type_str}, inputs={len(resolved_inputs)}")

        mx = self._import_mlx()

        # Helper: best-effort broadcast for binary ops when one input is 1D and
        # matches the other's trailing dimension.
        def _binary_op(op_name: str):
            x = resolved_inputs[0]
            y = resolved_inputs[1] if len(resolved_inputs) > 1 else None
            if y is None:
                return x
            sx = getattr(x, "shape", None)
            sy = getattr(y, "shape", None)
            if sx is not None and sy is not None and sx != sy:
                try:
                    # If y is 1D of length matching x's last dim, reshape to (1, ..., last)
                    if len(sy) == 1 and len(sx) >= 1 and sy[0] == sx[-1]:
                        y = (
                            mx.reshape(y, (1,) + (sx[-1],))
                            if len(sx) == 2
                            else mx.reshape(y, (sx[-1],))
                        )
                        y = mx.broadcast_to(y, sx)
                    # If x is 1D vector matching y's last dim
                    elif len(sx) == 1 and len(sy) >= 1 and sx[0] == sy[-1]:
                        x = (
                            mx.reshape(x, (1,) + (sy[-1],))
                            if len(sy) == 2
                            else mx.reshape(x, (sy[-1],))
                        )
                        x = mx.broadcast_to(x, sy)
                    # If one side is 1D whose length equals the total elements of the other
                    else:
                        try:
                            prod_sy = int(np.prod(sy))
                            prod_sx = int(np.prod(sx))
                            if len(sx) == 1 and sx[0] == prod_sy:
                                x = mx.reshape(x, sy)
                            elif len(sy) == 1 and sy[0] == prod_sx:
                                y = mx.reshape(y, sx)
                        except Exception:
                            pass
                except Exception:
                    pass
            if op_name == "add":
                return x + y
            if op_name == "sub":
                return x - y
            if op_name == "mul":
                return x * y
            if op_name == "div":
                return x / y
            if op_name == "pow":
                return x**y
            raise RuntimeError(f"Unknown binary op {op_name}")

        # Handle basic arithmetic operations (MLX ops mirror NumPy semantics)
        if "vecadd" in kernel_type_str or "add" in kernel_type_str:
            result = (
                _binary_op("add")
                if len(resolved_inputs) >= 2
                else (resolved_inputs[0] if resolved_inputs else mx.array([0.0]))
            )

        elif "vecsub" in kernel_type_str or "sub" in kernel_type_str:
            result = (
                _binary_op("sub")
                if len(resolved_inputs) >= 2
                else (-resolved_inputs[0] if resolved_inputs else mx.array([0.0]))
            )

        elif (
            "vecelementmul" in kernel_type_str
            or (
                "mul" in kernel_type_str
                and "matmul" not in kernel_type_str
                and "tiledmatmul" not in kernel_type_str
            )
            or (
                "element" in kernel_type_str
                and "matmul" not in kernel_type_str
                and "tiledmatmul" not in kernel_type_str
            )
        ):
            result = (
                _binary_op("mul")
                if len(resolved_inputs) >= 2
                else (resolved_inputs[0] if resolved_inputs else mx.array([1.0]))
            )

        elif "vecdiv" in kernel_type_str or (
            "div" in kernel_type_str and "vecdiv" not in kernel_type_str
        ):
            result = (
                _binary_op("div")
                if len(resolved_inputs) >= 2
                else (resolved_inputs[0] if resolved_inputs else mx.array([1.0]))
            )

        elif "vecpow" in kernel_type_str or "pow" in kernel_type_str:
            result = (
                _binary_op("pow")
                if len(resolved_inputs) >= 2
                else (resolved_inputs[0] if resolved_inputs else mx.array([1.0]))
            )

        # Activation functions
        elif "vecsin" in kernel_type_str or (
            "sin" in kernel_type_str and "vecsin" not in kernel_type_str
        ):
            result = mx.sin(resolved_inputs[0])

        elif "veccos" in kernel_type_str or (
            "cos" in kernel_type_str and "veccos" not in kernel_type_str
        ):
            result = mx.cos(resolved_inputs[0])

        elif "vectanh" in kernel_type_str or (
            "tanh" in kernel_type_str and "vectanh" not in kernel_type_str
        ):
            result = mx.tanh(resolved_inputs[0])

        elif "veclog" in kernel_type_str or (
            "log" in kernel_type_str and "veclog" not in kernel_type_str
        ):
            # VecLog with base support
            if scalar_inputs and len(scalar_inputs) > 0:
                base = float(scalar_inputs[0])
                result = mx.log(resolved_inputs[-1]) / mx.log(mx.array(base))
            else:
                result = mx.log(resolved_inputs[0])

        elif "vecleakyrelu" in kernel_type_str or (
            "leaky" in kernel_type_str and "relu" in kernel_type_str
        ):
            # LeakyReLU with alpha parameter
            alpha = 0.01
            if scalar_inputs and len(scalar_inputs) > 0:
                alpha = float(scalar_inputs[0])
            x = resolved_inputs[0]
            result = mx.where(x > 0, x, alpha * x)

        # Reductions
        elif "vecsum" in kernel_type_str or (
            "sum" in kernel_type_str and "vecsum" not in kernel_type_str
        ):
            x = resolved_inputs[0]
            if scalar_inputs and len(scalar_inputs) >= 3:
                # Sum reduction: pre_axis, axis_len, post_axis
                pre = int(scalar_inputs[0])
                axis_len = int(scalar_inputs[1])
                post = int(scalar_inputs[2])
                # Reshape for reduction along axis
                try:
                    x_reshaped = mx.reshape(x, (pre, axis_len, post))
                    result = mx.sum(x_reshaped, axis=1)
                except Exception:
                    result = mx.sum(x)
            else:
                result = mx.sum(x)

        elif "vecmax" in kernel_type_str or (
            "max" in kernel_type_str and "vecmax" not in kernel_type_str
        ):
            x = resolved_inputs[0]
            if scalar_inputs and len(scalar_inputs) >= 3:
                pre = int(scalar_inputs[0])
                axis_len = int(scalar_inputs[1])
                post = int(scalar_inputs[2])
                try:
                    x_reshaped = mx.reshape(x, (pre, axis_len, post))
                    result = mx.max(x_reshaped, axis=1)
                except Exception:
                    result = mx.max(x)
            else:
                result = mx.max(x)

        elif "vecmin" in kernel_type_str or (
            "min" in kernel_type_str and "vecmin" not in kernel_type_str
        ):
            x = resolved_inputs[0]
            if scalar_inputs and len(scalar_inputs) >= 3:
                pre = int(scalar_inputs[0])
                axis_len = int(scalar_inputs[1])
                post = int(scalar_inputs[2])
                try:
                    x_reshaped = mx.reshape(x, (pre, axis_len, post))
                    result = mx.min(x_reshaped, axis=1)
                except Exception:
                    result = mx.min(x)
            else:
                result = mx.min(x)

        # Expand (VecExpand_r{rank} custom kernel)
        elif "vecexpand" in kernel_type_str or "expand_r" in kernel_type_str:
            # Inputs: [parent, src_shape, src_strides, tgt_shape, tgt_strides]
            # We can implement via reshape + broadcast_to in MLX
            try:
                parent = resolved_inputs[0]
                src_shape = tuple(int(x) for x in np.array(resolved_inputs[1]).tolist())
                tgt_shape = tuple(int(x) for x in np.array(resolved_inputs[3]).tolist())
                parent_reshaped = mx.reshape(parent, src_shape)
                result = mx.broadcast_to(parent_reshaped, tgt_shape)
            except Exception:
                # Fallback: return parent as-is to avoid crashing
                result = resolved_inputs[0]

        # Permute (VecPermute_r{rank} custom kernel)
        elif "vecpermute" in kernel_type_str or "permute_r" in kernel_type_str:
            # Inputs: [parent, kernel_src_shape, src_strides, tgt_shape, tgt_strides]
            # Implement generic reindex using numpy on host, then wrap in MLX.
            try:
                parent = resolved_inputs[0]
                tgt_shape = tuple(int(x) for x in np.array(resolved_inputs[3]).tolist())
                # Materialise parent on host
                arr = np.array(parent, dtype=np.float32)
                # Infer source shape by matching total size
                if arr.ndim == 1:
                    # Attempt simple 2D transpose detection
                    if len(tgt_shape) == 2:
                        M, N = tgt_shape
                        src = arr.reshape((N, M))
                        result_np = src.T
                    else:
                        result_np = arr.reshape(tgt_shape)
                else:
                    # If already multi-d, just reshape to target and trust
                    result_np = np.reshape(arr, tgt_shape)
                result = mx.array(result_np, dtype=mx.float32)
            except Exception:
                result = resolved_inputs[0]

        # MatMul (including with epilogue)
        elif "matmul" in kernel_type_str or "tiledmatmul" in kernel_type_str:
            a = resolved_inputs[0]
            b = resolved_inputs[1]

            # Extract (M, N, K) from DirectInput scalars: inputs[2], [3], [4]
            try:
                M = int(np.array(resolved_inputs[2]).tolist()[0])
                N = int(np.array(resolved_inputs[3]).tolist()[0])
                K = int(np.array(resolved_inputs[4]).tolist()[0])
                a = mx.reshape(a, (M, K))
                b = mx.reshape(b, (K, N))
            except Exception:
                pass

            result = mx.matmul(a, b)

            # Handle epilogue ops (fused into MatMul by backend)
            # Parse epilogue from inputs if present (A, B, M, N, K, then epilogue side inputs)
            if len(resolved_inputs) > 5:
                try:
                    # Apply any side inputs (bias, etc.)
                    for side_input in resolved_inputs[5:]:
                        # Ensure side_input has compatible shape
                        try:
                            si = side_input
                            if getattr(si, "shape", None) == (N,):
                                si = mx.reshape(si, (1, N))
                                si = mx.broadcast_to(si, (M, N))
                            elif getattr(si, "shape", None) == (1, N):
                                si = mx.broadcast_to(si, (M, N))
                            elif getattr(si, "shape", None) == (M, 1):
                                si = mx.broadcast_to(si, (M, N))
                            result = result + si
                        except Exception:
                            # Best-effort add without reshape
                            result = result + side_input
                except (IndexError, ValueError):
                    # Fallback: skip epilogue
                    pass

        # Custom kernels
        elif "custom" in kernel_type_str:
            # Handle common customs above; otherwise pass-through first input.
            if resolved_inputs:
                result = resolved_inputs[0]
            else:
                result = mx.zeros((1,), dtype=mx.float32)

        else:
            # Unsupported kernel type - return input as-is
            print(
                f"Warning: Unsupported kernel type {kernel_type_str}, returning input"
            )
            if resolved_inputs:
                result = resolved_inputs[0]
            else:
                result = mx.zeros((1,), dtype=mx.float32)

        if DEBUG_TIMINGS and t_start is not None:
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            _dbg_print("timings", f"kernel#{kernel_id} mapped in {t_elapsed_ms:.3f} ms")

        return result, num_outputs

    def _get_kernel_dependencies(self, kernel) -> set:
        """Extract kernel IDs that this kernel depends on."""
        import tensorops_backend

        deps = set()
        inputs = getattr(kernel, "inputs", [])
        for inp in inputs:
            if isinstance(inp, tensorops_backend.LogicalInputSource):
                deps.add(inp.source_kernel_id)
        return deps

    def execute_graph(self, kernels):
        """Execute the entire kernel set as a single MLX evaluation.

        Instead of running each kernel eagerly, build the MLX computation graph for
        all kernels first, then trigger one `mx.eval` over every output. This keeps
        data on-device and lets MLX fuse/optimise across kernel boundaries.
        """
        t_graph_start = time.perf_counter() if DEBUG_TIMINGS else None

        if not kernels:
            return []

        mx = self._import_mlx()
        kernel_by_id = {
            getattr(k, "kernel_id", i): (k, i) for i, k in enumerate(kernels)
        }
        outputs_cache: Dict[int, List[Any]] = {}
        results_order: List[Any] = [None] * len(kernels)

        # Build the lazy MLX graph first.
        eval_plan: List[Tuple[int, int, int, Any]] = []
        eval_targets: List[Any] = []

        for kid in sorted(kernel_by_id):
            kernel, idx = kernel_by_id[kid]
            num_outputs = getattr(kernel, "num_output_bufs", 1)
            try:
                result, num_outputs = self._map_kernel_to_mlx(kernel, outputs_cache)
                outputs_cache[kid] = [result] * num_outputs
                eval_plan.append((kid, idx, num_outputs, result))
                eval_targets.append(result)
            except Exception as e:
                print(
                    f"Warning: Failed to build kernel {getattr(kernel, 'kernel_id', '?')}: {e}"
                )
                import traceback

                traceback.print_exc()
                outputs_cache[kid] = [None] * num_outputs
                mock_result = type(
                    "KernelResult",
                    (),
                    {
                        "val": [bytearray(4096) for _ in range(num_outputs)],
                        "kernel_id": kid,
                    },
                )()
                results_order[idx] = mock_result

        # Trigger a single MLX evaluation across all outputs.
        t_eval_start = time.perf_counter() if DEBUG_TIMINGS else None
        try:
            if eval_targets:
                mx.eval(*eval_targets)
            if DEBUG_TIMINGS and t_eval_start is not None:
                t_eval_ms = (time.perf_counter() - t_eval_start) * 1000.0
                _dbg_print("timings", f"graph eval completed in {t_eval_ms:.3f} ms")
        except Exception as e:
            print(f"Warning: MLX eval failed for fused graph: {e}")
            import traceback

            traceback.print_exc()
            # Fallback: mark all unevaluated kernels as failed.
            for kid, idx, num_outputs, _ in eval_plan:
                if results_order[idx] is not None:
                    continue
                outputs_cache[kid] = [None] * num_outputs
                mock_result = type(
                    "KernelResult",
                    (),
                    {
                        "val": [bytearray(4096) for _ in range(num_outputs)],
                        "kernel_id": kid,
                    },
                )()
                results_order[idx] = mock_result
            return results_order

        # Materialise outputs to host bytearrays once after the fused eval.
        for kid, idx, num_outputs, result in eval_plan:
            if results_order[idx] is not None:
                continue
            try:
                result_np = np.array(result, dtype=np.float32)
                result_bytes = result_np.tobytes()
                mock_result = type(
                    "KernelResult",
                    (),
                    {
                        "val": [bytearray(result_bytes) for _ in range(num_outputs)],
                        "kernel_id": kid,
                    },
                )()
            except Exception:
                mock_result = type(
                    "KernelResult",
                    (),
                    {
                        "val": [bytearray(4096) for _ in range(num_outputs)],
                        "kernel_id": kid,
                    },
                )()
            results_order[idx] = mock_result

        if DEBUG_TIMINGS and t_graph_start is not None:
            t_total_ms = (time.perf_counter() - t_graph_start) * 1000.0
            _dbg_print("timings", f"execute_graph total: {t_total_ms:.3f} ms")

        return results_order

    def execute_ops_graph(self, ops, ctx):
        """Execute a list of OPs directly using MLX's graph building."""
        t_start = time.perf_counter() if DEBUG_TIMINGS else None
        if not ops:
            return []
        mx = self._import_mlx()

        op_results: Dict[int, Any] = {}
        results_list: List[Any] = []

        for op_idx, op in enumerate(ops):
            op_type = type(op).__name__
            current_result = None

            # 1. Handle Pass-through/View Ops within the current batch
            if op_type in ("ShapeOP", "StopGrad", "ExpandOP"):
                parent = getattr(op, "tensor1", None)
                if parent is not None and id(parent) in op_results:
                    prev_res = op_results[id(parent)]
                    if op_type == "ShapeOP":
                        current_result = mx.reshape(prev_res, op.shape)
                    elif op_type == "ExpandOP":
                        current_result = mx.broadcast_to(prev_res, op.shape)
                    else:  # StopGrad
                        current_result = prev_res

            # 2. Resolve Inputs for Compute Ops or external View Ops
            if current_result is None:
                parents = getattr(op, "parents", [])
                parent_inputs = []
                for p in parents:
                    if p is None:
                        continue

                    if id(p) in op_results:
                        # Parent is in this execution batch
                        parent_inputs.append(op_results[id(p)])
                    else:
                        # Parent is external (Leaf or from a previous forward/backward pass)
                        # Use .values to ensure we get data even if it's a view op
                        try:
                            val = p.values  # This triggers lazy-load if needed
                            if val is None:
                                raise ValueError(f"Tensor {id(p)} has no values")

                            # Convert to MLX
                            arr = mx.array(val, dtype=mx.float32)

                            # Ensure shape matches (crucial for ShapeOP parents)
                            if p.shape and arr.shape != tuple(p.shape):
                                if arr.size == np.prod(p.shape):
                                    arr = mx.reshape(arr, p.shape)
                                else:
                                    arr = mx.broadcast_to(arr, p.shape)
                            parent_inputs.append(arr)
                        except Exception as e:
                            if DEBUG_KERNEL_INPUTS:
                                _dbg_print(
                                    "inputs",
                                    f"Failed to resolve parent {type(p).__name__}: {e}",
                                )
                            parent_inputs.append(
                                mx.zeros(p.shape or (1,), dtype=mx.float32)
                            )

                current_result = self._execute_op(op, parent_inputs, op_idx)

            # 3. Finalize result for this op
            if current_result is None:
                current_result = mx.zeros(op.shape or (1,), dtype=mx.float32)

            op_results[id(op)] = current_result
            results_list.append(current_result)

        # Trigger MLX Graph Evaluation
        if results_list:
            mx.eval(*results_list)

        # Wrap for TensorOps compatibility
        kernel_results = []
        for i, res in enumerate(results_list):
            # We convert to numpy then bytes to satisfy the existing Tensor.py infrastructure
            res_np = np.array(res, copy=False).astype(np.float32)
            kr = type(
                "KernelResult",
                (),
                {"val": [bytearray(res_np.tobytes())], "kernel_id": i},
            )()
            kernel_results.append(kr)

        if DEBUG_TIMINGS and t_start:
            _dbg_print(
                "timings",
                f"execute_ops_graph took {(time.perf_counter() - t_start) * 1000:.2f}ms",
            )

        return kernel_results

    def _execute_op(self, op, parent_inputs, op_idx):
        """Execute a single op given its parent inputs.

        Maps Python OP types (Add, Sin, MatMul, etc.) to MLX operations.
        """
        op_type = type(op).__name__
        mx = self._import_mlx()

        def _reshape_for_broadcast(x, y):
            """Helper to reshape mismatched binary operands."""
            try:
                sx = getattr(x, "shape", None)
                sy = getattr(y, "shape", None)
                if sx is None or sy is None or sx == sy:
                    return x, y
                # If one is flat and the other is shaped, reshape the flat one
                if len(sx) == 1 and len(sy) > 1:
                    if sx[0] == int(np.prod(sy)):
                        return mx.reshape(x, sy), y
                if len(sy) == 1 and len(sx) > 1:
                    if sy[0] == int(np.prod(sx)):
                        return x, mx.reshape(y, sx)
                # Otherwise try direct broadcast (MLX will handle it)
                return x, y
            except Exception:
                return x, y

        # Unary ops
        if op_type == "Sin" and parent_inputs:
            return mx.sin(parent_inputs[0])
        elif op_type == "Cos" and parent_inputs:
            return mx.cos(parent_inputs[0])
        elif op_type == "Tanh" and parent_inputs:
            return mx.tanh(parent_inputs[0])

        # Binary ops
        elif op_type == "Add" and len(parent_inputs) >= 2:
            x, y = _reshape_for_broadcast(parent_inputs[0], parent_inputs[1])
            return x + y
        elif op_type == "Sub" and len(parent_inputs) >= 2:
            x, y = _reshape_for_broadcast(parent_inputs[0], parent_inputs[1])
            return x - y
        elif op_type == "ElementMul" and len(parent_inputs) >= 2:
            x, y = _reshape_for_broadcast(parent_inputs[0], parent_inputs[1])
            return x * y
        elif op_type == "Div" and len(parent_inputs) >= 2:
            x, y = _reshape_for_broadcast(parent_inputs[0], parent_inputs[1])
            return x / y
        elif op_type == "Pow" and len(parent_inputs) >= 2:
            x, y = _reshape_for_broadcast(parent_inputs[0], parent_inputs[1])
            return x**y

        # MatMul
        elif op_type == "MatMul" and len(parent_inputs) >= 2:
            a = parent_inputs[0]
            b = parent_inputs[1]
            # Reshape to 2D if needed
            a_shape = getattr(op, "parents", [None])[0].shape if op.parents else None
            b_shape = (
                getattr(op, "parents", [None])[1].shape if len(op.parents) > 1 else None
            )
            if a_shape and len(a_shape) == 2:
                a = mx.reshape(a, a_shape)
            if b_shape and len(b_shape) == 2:
                b = mx.reshape(b, b_shape)
            return mx.matmul(a, b)

        # Activation functions
        elif op_type == "LeakyReLU" and parent_inputs:
            alpha = getattr(op, "leaky_grad", 0.01)
            if isinstance(alpha, (list, tuple)):
                alpha = alpha[0] if alpha else 0.01
            x = parent_inputs[0]
            return mx.where(x > 0, x, alpha * x)

        # Reductions
        elif op_type == "Sum" and parent_inputs:
            axis = getattr(op, "axis", None)
            return mx.sum(parent_inputs[0], axis=axis)

        elif op_type == "Max" and parent_inputs:
            axis = getattr(op, "axis", None)
            return mx.max(parent_inputs[0], axis=axis)

        elif op_type == "Min" and parent_inputs:
            axis = getattr(op, "axis", None)
            return mx.min(parent_inputs[0], axis=axis)

        # Fallback for unsupported ops
        else:
            if DEBUG_KERNEL_INPUTS:
                _dbg_print(
                    "execute_ops_graph",
                    f"op#{op_idx} ({op_type}): unsupported, passing through first input",
                )
            return (
                parent_inputs[0] if parent_inputs else mx.zeros((1,), dtype=mx.float32)
            )
