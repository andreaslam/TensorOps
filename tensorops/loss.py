from abc import abstractmethod, ABC

from tensorops.node import Node


class Loss(ABC):
    """
    `tensorops.Loss` is the abstract base class that handles cost function computation.
    """

    def __init__(self) -> None:
        ...

    @abstractmethod
    def loss(self, actual, target) -> Node:
        ...

    def __call__(self, actual, target) -> Node:
        return self.loss(actual, target)


class L1Loss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, actual, target) -> Node:
        """
        Calculate the L1 loss between the actual and target values.

        If the inputs are lists, it computes the average absolute difference element-wise.
        If the inputs are floats, it computes the absolute difference.

        Args:
            actual (Union[float, List[tensorops.node.Node]]): The actual output value(s).
            target (Union[float, List[tensorops.node.Node]]): The target output value(s).

        Returns:
            tensorops.node.Node(): The computed L1 loss value.
        """
        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(
                target
            ), "Actual and target lists must have the same length."
            total_loss = Node(0.0, requires_grad=False)
            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss += abs(actual_datapoint - target_datapoint)

            return total_loss / Node(len(actual), requires_grad=False)
        else:
            return abs(actual - target)


class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, actual, target) -> Node:
        """
        Calculate the MSE loss between the actual and target values. Works with both single values and lists of `tensorops.node.Node`.

        Args:
            actual (Union[float, List[tensorops.node.Node]]): The actual output value(s).
            target (Union[float, List[tensorops.node.Node]]): The target output value(s).

        Returns:
            tensorops.node.Node(): The computed MSE loss value.
        """
        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(
                target
            ), "Actual and target lists must have the same length."

            total_loss = Node(0.0, requires_grad=False, weight=False)
            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss = total_loss + ((actual_datapoint - target_datapoint) ** 2)

            loss = total_loss / Node(len(actual), requires_grad=False, weight=False)
            return loss
        else:
            return (target - actual) ** 2
