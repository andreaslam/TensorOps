import random

from tensorops.loss import CrossEntropyLoss
from tensorops.tensor import LeakyReLU, Tensor
from tensorops.utils.models import SequentialModel


class MNISTModelLocal(SequentialModel):
    def __init__(
        self,
        num_hidden_layers,
        num_hidden_nodes,
        loss_criterion,
        seed=None,
        batch_size=256,
    ):
        super().__init__(loss_criterion, seed, batch_size=batch_size)
        self.activation_function = LeakyReLU
        self.num_hidden_layers = num_hidden_layers
        with self.context:
            self.add_layer(784, num_hidden_nodes, self.activation_function)
            for _ in range(self.num_hidden_layers):
                self.add_layer(
                    num_hidden_nodes, num_hidden_nodes, self.activation_function
                )
            self.add_layer(num_hidden_nodes, 10, None)
            self.loss = self.loss_criterion(
                self.targets, self.model_output_layer.layer_output
            )

    def forward(self, model_inputs: Tensor) -> Tensor:
        with self.context:
            for layer in self.model_layers:
                layer.forward(model_inputs)
                model_inputs = layer.layer_output
            return model_inputs


# Create random inputs and labels for one batch
BATCH = 256
X = Tensor(
    [[random.random() for _ in range(784)] for _ in range(BATCH)], requires_grad=False
)
Y = Tensor([[0.0] * 10 for _ in range(BATCH)], requires_grad=False)
for i in range(BATCH):
    Y.values[i][random.randint(0, 9)] = 1.0

model = MNISTModelLocal(2, 256, CrossEntropyLoss(), seed=42, batch_size=BATCH)
optim = SGD(model.get_weights(), lr=1e-4)
model.train()

# Quick convergence check on a fixed batch
STEPS = 50
for step in range(1, STEPS + 1):
    logits = model(X, execute=True)
    loss = model.calculate_loss(logits, Y)
    model.backward()
    if step == 1:
        for i, w in enumerate(model.get_weights()):
            g = w.grads
            print(
                f"Weight {i} shape={w.shape}, grad_shape={getattr(g, 'shape', None)}, grad_len={len(g.values) if getattr(g, 'values', None) is not None else None}"
            )
    optim.step()
    if step % 10 == 0:
        print(f"Step {step} loss: {loss.item()}")
