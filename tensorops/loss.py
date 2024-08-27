from abc import abstractmethod
from typing import Any


class Loss:
    def __init__(self):
        pass

    @abstractmethod
    def loss(self, actual, target):
        pass

    def __call__(self, actual, target):
        return self.loss(actual, target)


class L1Loss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):  # takes Node, not float
        result = abs(actual - target)
        return result


class MSELoss(Loss):  # L2 loss
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):  # takes Node, not float
        result = (target - actual) ** 2
        return result
