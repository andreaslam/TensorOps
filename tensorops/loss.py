from abc import abstractmethod


class Loss:
    def __init__(self):
        ...

    @abstractmethod
    def loss(self, actual, target):
        ...

    def __call__(self, actual, target):
        return self.loss(actual, target)


class L1Loss(Loss): # MAE loss
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        result = abs(actual - target)
        return result


class MSELoss(Loss):  # L2 loss
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        result = (target - actual) ** 2
        return result
