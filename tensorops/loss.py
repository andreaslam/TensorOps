from abc import abstractmethod, ABC


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
        result = abs(actual - target)
        return result


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        result = (target - actual) ** 2
        return result
