import os
import platform
from typing import Any, Union

from .device import TensorOpsDevice

# Try to import tensorops_backend (Rust/OpenCL backend)
# Not available on macOS - only MLX is available there
tensorops_backend: Any = None
TENSOROPS_BACKEND_AVAILABLE = False

try:
    import tensorops_backend as _tb

    tensorops_backend = _tb
    TENSOROPS_BACKEND_AVAILABLE = True
except ImportError:
    TENSOROPS_BACKEND_AVAILABLE = False

# Backend selection: default to Rust extension; allow MLX via env var
_backend = os.getenv("TENSOROPS_BACKEND", "").lower()
rt: Union[Any, None] = None

if _backend == "mlx" or platform.system().lower() == "darwin":
    try:
        from .mlx_runtime import MLXRuntime

        rt = MLXRuntime()
    except Exception as e:
        print(f"Warning: Failed to initialize MLX runtime: {e}")
        # Fall back to Rust backend if available
        if TENSOROPS_BACKEND_AVAILABLE and tensorops_backend is not None:
            rt = tensorops_backend.Runtime()
        else:
            raise RuntimeError(
                "MLX runtime failed and tensorops_backend not available. "
                "On macOS, install MLX. On other platforms, build tensorops_backend."
            )
else:
    # Use Rust backend (OpenCL/CPU)
    if not TENSOROPS_BACKEND_AVAILABLE or tensorops_backend is None:
        is_macos = platform.system().lower() == "darwin"
        if is_macos:
            raise ImportError(
                "tensorops_backend is not available on macOS. "
                "Use TensorOpsDevice.APPLE with MLX backend instead: "
                "set TENSOROPS_BACKEND=mlx environment variable or create tensors with device=TensorOpsDevice.APPLE"
            )
        else:
            raise ImportError(
                "tensorops_backend not found. Please build the Rust extension or install from package."
            )
    rt = tensorops_backend.Runtime()


def get_runtime_for_device(device: TensorOpsDevice):
    """Get or create the appropriate runtime for a given device."""
    if device == TensorOpsDevice.APPLE:
        try:
            from .mlx_runtime import MLXRuntime

            return MLXRuntime()
        except Exception as e:
            if TENSOROPS_BACKEND_AVAILABLE and tensorops_backend is not None:
                print(
                    f"Warning: MLX runtime failed ({e}), falling back to Rust backend"
                )
                return tensorops_backend.Runtime()
            else:
                raise RuntimeError(
                    f"MLX runtime unavailable ({e}) and tensorops_backend not available on this platform"
                )
    elif device == TensorOpsDevice.OPENCL or device == TensorOpsDevice.CPU:
        if not TENSOROPS_BACKEND_AVAILABLE or tensorops_backend is None:
            raise ImportError(
                f"tensorops_backend required for device {device} but not available. "
                "Please build the Rust extension or use TensorOpsDevice.APPLE with MLX."
            )
        return tensorops_backend.Runtime()
    else:
        raise ValueError(f"Unknown device: {device}")
