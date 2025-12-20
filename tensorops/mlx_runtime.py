"""MLX backend runtime adapter for macOS.

This module provides a device-aware runtime for macOS using MLX (MLX framework).
Maps TensorOps kernels to MLX operations and handles graph execution.
Enable via environment variable: TENSOROPS_BACKEND=mlx
"""

import platform
from typing import Any, Dict, List, Tuple

try:
    import mlx.core as mx  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "MLX package not installed. Please install MLX before using this backend."
    ) from e
import numpy as np

class MLXRuntime:
    def __init__(self) -> None:
        if platform.system().lower() != "darwin":
            raise RuntimeError("MLX backend is only supported on macOS (Darwin).")

        self.mx = None  # Lazy import
        self._kernel_outputs: Dict[
            int, List[Any]
        ] = {}  # Cache kernel outputs for dependencies

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

        # Check if it's a LogicalInputSource (reference to another kernel's output)
        if isinstance(inp, tensorops_backend.LogicalInputSource):
            # LogicalInputSource: fetch from cache
            src_id = inp.source_kernel_id
            src_idx = getattr(inp, "source_output_index", 0)
            if src_id in outputs_cache:
                output = outputs_cache[src_id][src_idx]
                # Convert to MLX if needed
                if hasattr(output, "__array__"):
                    return mx.array(output, dtype=mx.float32)
                elif hasattr(output, "tolist"):
                    return mx.array(output.tolist(), dtype=mx.float32)
                else:
                    return mx.array(output, dtype=mx.float32)
            else:
                raise RuntimeError(f"Missing kernel output for dependency: {src_id}")

        # DirectInput: extract data directly
        if isinstance(inp, tensorops_backend.DirectInput):
            data = inp.data
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

        # Map kernel type to MLX op
        if kernel_type is None:
            raise RuntimeError(f"Kernel {kernel_id} has no kernel_type")

        # Extract kernel type name (handle both enum and string representations)
        # The kernel_type could be KernelType.Custom or KernelType.Predefined(...)
        kernel_type_str = str(kernel_type).lower()

        # For debugging kernel type
        # print(f"Kernel {kernel_id}: type_str={kernel_type_str}, inputs={len(resolved_inputs)}")

        mx = self._import_mlx()

        # Handle basic arithmetic operations (MLX ops mirror NumPy semantics)
        if "vecadd" in kernel_type_str or "add" in kernel_type_str:
            if len(resolved_inputs) >= 2:
                result = resolved_inputs[0] + resolved_inputs[1]
            else:
                result = resolved_inputs[0] if resolved_inputs else mx.array([0.0])

        elif "vecsub" in kernel_type_str or "sub" in kernel_type_str:
            if len(resolved_inputs) >= 2:
                result = resolved_inputs[0] - resolved_inputs[1]
            else:
                result = -resolved_inputs[0] if resolved_inputs else mx.array([0.0])

        elif (
            "vecelementmul" in kernel_type_str
            or "mul" in kernel_type_str
            or "element" in kernel_type_str
        ):
            if len(resolved_inputs) >= 2:
                result = resolved_inputs[0] * resolved_inputs[1]
            else:
                result = resolved_inputs[0] if resolved_inputs else mx.array([1.0])

        elif "vecdiv" in kernel_type_str or (
            "div" in kernel_type_str and "vecdiv" not in kernel_type_str
        ):
            if len(resolved_inputs) >= 2:
                result = resolved_inputs[0] / resolved_inputs[1]
            else:
                result = resolved_inputs[0] if resolved_inputs else mx.array([1.0])

        elif "vecpow" in kernel_type_str or "pow" in kernel_type_str:
            if len(resolved_inputs) >= 2:
                result = resolved_inputs[0] ** resolved_inputs[1]
            else:
                result = resolved_inputs[0] if resolved_inputs else mx.array([1.0])

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

        # MatMul (including with epilogue)
        elif "matmul" in kernel_type_str or "tiledmatmul" in kernel_type_str:
            a = resolved_inputs[0]
            b = resolved_inputs[1]
            result = mx.matmul(a, b)

            # Handle epilogue ops (fused into MatMul by backend)
            # Parse epilogue from inputs if present (A, B, M, N, K, then epilogue side inputs)
            if len(resolved_inputs) > 3:
                try:
                    # Apply any side inputs (bias, etc.)
                    for side_input in resolved_inputs[5:]:
                        # Simple broadcasting add for bias
                        if side_input.shape[-1] == result.shape[-1]:
                            result = result + side_input
                        elif side_input.shape[-1] == result.shape[-2]:
                            result = result + mx.expand_dims(side_input, axis=-2)
                except (IndexError, ValueError):
                    # Fallback: skip epilogue
                    pass

        # Custom kernels - just return input as-is
        elif "custom" in kernel_type_str:
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

        return result, num_outputs

    def execute_graph(self, kernels):
        """Execute graph using MLX.

        Receives a list of KernelTensorOps objects and maps them to MLX operations.
        Returns results in the same format as Rust backend (KernelResult-like objects).
        """
        if not kernels:
            return []

        outputs_cache: Dict[int, List[Any]] = {}
        results = []

        for kernel in kernels:
            try:
                kernel_id = getattr(kernel, "kernel_id", 0)

                # Execute kernel and get result
                result, num_outputs = self._map_kernel_to_mlx(kernel, outputs_cache)

                # Ensure result is concrete and convert to bytes via NumPy view
                result_np = np.array(result, dtype=np.float32)
                result_bytes = result_np.tobytes()

                # Store in cache for dependent kernels
                outputs_cache[kernel_id] = [result]

                # Create result object matching KernelResult interface
                mock_result = type(
                    "KernelResult",
                    (),
                    {
                        "val": [bytearray(result_bytes)],
                        "kernel_id": kernel_id,
                    },
                )()
                results.append(mock_result)

            except Exception as e:
                # Fallback: return zeros with warning
                print(
                    f"Warning: Failed to execute kernel {getattr(kernel, 'kernel_id', '?')}: {e}"
                )
                import traceback

                traceback.print_exc()

                num_outputs = getattr(kernel, "num_output_bufs", 1)
                mock_result = type(
                    "KernelResult",
                    (),
                    {
                        "val": [bytearray(4096) for _ in range(num_outputs)],
                        "kernel_id": getattr(kernel, "kernel_id", 0),
                    },
                )()
                results.append(mock_result)

        return results
