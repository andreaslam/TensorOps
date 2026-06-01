import os
import platform
import sys
from pathlib import Path
from typing import Any, Union

from .device import TensorOpsDevice

# Try to import tensorops_backend (Rust/OpenCL backend)
# Not available on macOS - only MLX is available there
tensorops_backend: Any = None
TENSOROPS_BACKEND_AVAILABLE = False


def _maybe_add_repo_venv_site_packages() -> None:
    if os.getenv("VIRTUAL_ENV"):
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".venv" / "Lib" / "site-packages"
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            break


try:
    import importlib

    _maybe_add_repo_venv_site_packages()
    _tb = importlib.import_module("tensorops.tensorops_backend")
    if not hasattr(_tb.Runtime, "execute_graph_with_updates"):
        try:
            _tb_alt = importlib.import_module("tensorops_backend")
            if hasattr(_tb_alt.Runtime, "execute_graph_with_updates"):
                _tb = _tb_alt
        except ImportError:
            pass

    sys.modules["tensorops.tensorops_backend"] = _tb
    tensorops_backend = _tb
    TENSOROPS_BACKEND_AVAILABLE = True
except ImportError:
    TENSOROPS_BACKEND_AVAILABLE = False

_runtime_cache: dict[TensorOpsDevice, Any] = {}


def _create_runtime_for_device(device: TensorOpsDevice):
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
            raise RuntimeError(
                f"MLX runtime unavailable ({e}) and tensorops_backend not available on this platform"
            )

    if device == TensorOpsDevice.OPENCL or device == TensorOpsDevice.CPU:
        if not TENSOROPS_BACKEND_AVAILABLE or tensorops_backend is None:
            is_macos = platform.system().lower() == "darwin"
            if is_macos:
                raise ImportError(
                    "tensorops_backend is not available on macOS. "
                    "Use TensorOpsDevice.APPLE with MLX backend instead: "
                    "set TENSOROPS_BACKEND=mlx environment variable or create tensors with device=TensorOpsDevice.APPLE"
                )
            raise ImportError(
                "tensorops_backend required for OpenCL/CPU but not available. "
                "Please build the Rust extension or install from package."
            )
        return tensorops_backend.Runtime()

    raise ValueError(f"Unknown device: {device}")


def get_runtime_for_device(device: TensorOpsDevice):
    """Get or create the appropriate runtime for a given device."""
    cached = _runtime_cache.get(device)
    if cached is not None:
        return cached

    runtime = _create_runtime_for_device(device)
    _runtime_cache[device] = runtime
    return runtime


# Backend selection: default to Rust extension; allow MLX via env var
_backend = os.getenv("TENSOROPS_BACKEND", "").lower()
rt: Union[Any, None] = None

if _backend == "mlx" or platform.system().lower() == "darwin":
    rt = get_runtime_for_device(TensorOpsDevice.APPLE)
else:
    rt = get_runtime_for_device(TensorOpsDevice.OPENCL)
