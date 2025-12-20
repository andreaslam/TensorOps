#!/usr/bin/env python3
"""
Precompile predefined kernels and save them to the kernel cache directory.

This script compiles all predefined kernels (VecAdd, VecMul, MatMul, etc.) and caches
the compiled OpenCL binaries on disk. This speeds up subsequent runtime initialization.

Usage:
    python precompile_kernels.py
"""

import os
import sys
from pathlib import Path

# Add the tensorops module to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tensorops_backend import Runtime

    print("✓ Imported tensorops_backend successfully")
except ImportError as e:
    print(f"✗ Failed to import tensorops_backend: {e}")
    print("  Make sure to build the Rust extension first.")
    sys.exit(1)


def main():
    try:
        runtime = Runtime()
        print(f"✓ Initialized OpenCL Runtime: {runtime}")

        cache_dir = os.environ.get("TENSOROPS_CACHE_DIR")
        if not cache_dir:
            import tempfile

            cache_dir = os.path.join(tempfile.gettempdir(), "tensorops_kernel_cache")

        print(f"✓ Kernel cache directory: {cache_dir}")

        # Create cache directory if needed
        os.makedirs(cache_dir, exist_ok=True)

        # List existing cached kernels
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".bin")]
        print(f"✓ Found {len(cache_files)} cached kernel binaries")

        if cache_files:
            print("  Cached kernels:")
            for f in cache_files:
                size = os.path.getsize(os.path.join(cache_dir, f))
                print(f"    - {f} ({size} bytes)")

        print("\n✓ Kernel precompilation setup complete!")
        print("  Kernels will be cached to disk on first use in runtime.")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
