from abc import abstractmethod, ABC

from tensorops.node import Node


class Loss(ABC):
    """
    `tensorops.Loss` is the abstract base class that handles cost function computation.
    """

    def __init__(self):
        ...

    @abstractmethod
    def loss(self, actual, target):
        ...

    def __call__(self, actual, target):
        return self.loss(actual, target)


class L1Loss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        """
        Calculate the L1 loss between the actual and target values.

        If the inputs are lists, it computes the average absolute difference element-wise.
        If the inputs are floats, it computes the absolute difference.

        Args:
            actual (Union[float, List[float]]): The actual output value(s).
            target (Union[float, List[float]]): The target output value(s).

        Returns:
            float: The computed L1 loss value.
        """
        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(
                target
            ), "Actual and target lists must have the same length."
            total_loss = sum(abs(a - t) for a, t in zip(actual, target))
            return total_loss / len(actual)
        else:
            return abs(actual - target)


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        """
        Calculate the MSE loss between the actual and target values. Works with both single values and lists of `tensorops.node.Node`.

        Args:
            actual (Union[float, List[tensorops.node.Node]]): The actual output value(s).
            target (Union[float, List[tensorops.node.Node]]): The target output value(s).

        Returns:
            float: The computed MSE loss value.
        """

        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(
                target
            ), "Actual and target lists must have the same length."

            total_loss = Node(0.0, requires_grad=False, weight=False)

            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss = total_loss + (
                    (actual_datapoint - target_datapoint)
                    ** Node(2, requires_grad=False, weight=False)
                )

            return total_loss / Node(len(actual), requires_grad=False)
        else:
            return (target - actual) ** 2
