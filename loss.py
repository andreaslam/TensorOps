class Loss:
    def __init__(self):
        pass

    def loss(self):
        pass

    def backward(self, context):
        context.nodes[-1].seed_grad(
            1
        )  # seed the gradient of the last node (output node) as 1
        for node in reversed(context.nodes):
            if node.requires_grad:
                node.get_grad()


class L1Loss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):  # takes float, not Node
        result = abs(actual - target)
        return result


class MSELoss(Loss):  # L2 loss
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):  # takes Node, not float
        result = (target - actual) ** 2
        return result
