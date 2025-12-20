import enum


class TensorOpsDevice(enum.Enum):
    CPU = "CPU"
    OPENCL = "OpenCL"
    APPLE = "Apple"  # Apple Silicon, uses mlx


class Device:
    def __init__(self, device: TensorOpsDevice) -> None:
        self.device_name = device
