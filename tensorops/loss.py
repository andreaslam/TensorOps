from abc import abstractmethod, ABC

from tensorops.tensor import Tensor


class Loss(ABC):
    """
    `tensorops.Loss` is the abstract base class that handles cost function computation.
    """

    def __init__(self) -> None:
        self.loss_value = None

    @abstractmethod
    def loss(self, actual, target) -> Tensor:
        ...

    def __call__(self, actual, target) -> Tensor:
        return self.loss(actual, target)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(value={round(self.loss_value.value, 4) if self.loss_value else None})"


class L1Loss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, actual, target) -> Tensor:
        """
        Calculate the L1 loss between the actual and target values.

        If the inputs are lists, it computes the average absolute difference element-wise.
        If the inputs are floats, it computes the absolute difference.

        Args:
            actual (Union[float, List[tensorops.tensor.Tensor]]): The actual output value(s).
            target (Union[float, List[tensorops.tensor.Tensor]]): The target output value(s).

        Returns:
            tensorops.tensor.Tensor(): The computed L1 loss value.
        """
        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(
                target
            ), "Actual and target lists must have the same length."
            total_loss = Tensor(0.0, requires_grad=False)
            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss += abs(actual_datapoint - target_datapoint)
            self.loss_value = total_loss / Tensor(len(actual), requires_grad=False)
            return self.loss_value
        else:
            assert isinstance(actual, Tensor) and isinstance(
                target, Tensor
            ), f"Values passed into {type(self).__name__}.loss() must be instances of tensorops.Tensor"
            self.loss_value = abs(actual - target)
            return self.loss_value


class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, actual, target) -> Tensor:
        """
        Calculate the MSE loss between the actual and target values. Works with both single values and lists of `tensorops.tensor.Tensor`.

        Args:
            actual (Union[float, List[tensorops.tensor.Tensor]]): The actual output value(s).
            target (Union[float, List[tensorops.tensor.Tensor]]): The target output value(s).

        Returns:
            tensorops.tensor.Tensor(): The computed MSE loss value.
        """
        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(
                target
            ), "Actual and target lists must have the same length."

            total_loss = Tensor(0.0, requires_grad=False, weight=False)
            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss = total_loss + ((actual_datapoint - target_datapoint) ** 2)

            self.loss_value = total_loss / Tensor(
                len(actual), requires_grad=False, weight=False
            )
            return self.loss_value
        else:
            assert isinstance(actual, Tensor) and isinstance(
                target, Tensor
            ), f"Values passed into {type(self).__name__}.loss() must be instances of tensorops.Tensor"
            self.loss_value = (target - actual) ** 2
            return self.loss_value
