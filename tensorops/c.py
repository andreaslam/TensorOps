import random

from tensorops.loss import Loss, MSELoss
from tensorops.model import Model
from tensorops.optim import AdamW
from tensorops.tensor import Tanh, Tensor

HIDDEN_SIZE = 5
NUM_LAYERS = 1


class SimpleModel(Model):
    def __init__(self, loss_criterion: Loss, seed: int | None = None) -> None:
        super().__init__(loss_criterion, seed)
        with self.context:
            for _ in range(NUM_LAYERS):
                self.add_layer(HIDDEN_SIZE, HIDDEN_SIZE, Tanh)
            # Set up loss calculation in the graph
            self.loss = self.loss_criterion(
                self.model_output_layer.layer_output, self.targets
            )

    def forward(self, model_inputs):
        # Only update input values, don't execute the graph
        for layer in self.model_layers:
            layer.forward(model_inputs)
            model_inputs = layer.layer_output
        return self.model_layers[-1].layer_output

    def calculate_loss(self, output: Tensor, target: Tensor) -> None | Tensor:
        # Update target values and return the loss tensor
        self.targets.values = target.values
        return self.loss


model = SimpleModel(MSELoss(), 42)
with model.context:
    output = model(Tensor([random.random() for _ in range(HIDDEN_SIZE)]))
    # Execute the forward pass
    model.context.forward()
    model.calculate_loss(output, Tensor([random.random() for _ in range(HIDDEN_SIZE)]))
    model.backward()

# Create optimizer with model.weights directly
optim = AdamW(model.weights)

print("Before optimizer step:")
for i, w in enumerate(model.weights):
    print(f"Weight {i}: {w.values[:5]}")

optim.step()

print("\nAfter optimizer step:")
for i, w in enumerate(model.weights):
    print(f"Weight {i}: {w.values[:5]}")
