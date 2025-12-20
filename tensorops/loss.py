from abc import ABC, abstractmethod

from tensorops.tensor import Tensor, ones, zeros


class Loss(ABC):
    """
    `tensorops.Loss` is the abstract base class that handles cost function computation.
    """

    def __init__(self) -> None:
        self.loss_value = None

    @abstractmethod
    def loss(self, actual, target) -> Tensor: ...

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
            assert len(actual) == len(target), (
                "Actual and target lists must have the same length."
            )
            total_loss = Tensor(0.0, requires_grad=False)
            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss += abs(actual_datapoint - target_datapoint)
            self.loss_value = total_loss / Tensor(len(actual), requires_grad=False)
            return self.loss_value
        else:
            assert isinstance(actual, Tensor) and isinstance(target, Tensor), (
                f"Values passed into {type(self).__name__}.loss() must be instances of tensorops.Tensor"
            )
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
            assert len(actual) == len(target), (
                "Actual and target lists must have the same length."
            )

            total_loss = Tensor(0.0, requires_grad=False, weight=False)
            for actual_datapoint, target_datapoint in zip(actual, target):
                total_loss = total_loss + ((actual_datapoint - target_datapoint) ** 2)

            self.loss_value = total_loss / Tensor(
                len(actual), requires_grad=False, weight=False
            )
            return self.loss_value
        else:
            assert isinstance(actual, Tensor) and isinstance(target, Tensor), (
                f"Values passed into {type(self).__name__}.loss() must be instances of tensorops.Tensor"
            )
            self.loss_value = (target - actual) ** 2
            return self.loss_value


class BCELoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, actual, target) -> Tensor:
        """
        Calculate the Binary Cross Entropy loss between the actual and target values.

        Args:
            actual (Union[float, List[tensorops.tensor.Tensor]]): The actual output value(s) (probabilities).
            target (Union[float, List[tensorops.tensor.Tensor]]): The target output value(s) (0 or 1).

        Returns:
            tensorops.tensor.Tensor(): The computed BCE loss value.
        """
        if isinstance(actual, list) and isinstance(target, list):
            assert len(actual) == len(target), (
                "Actual and target lists must have the same length."
            )

            total_loss = Tensor(0.0, requires_grad=False, weight=False)
            eps = 1e-15
            one = Tensor(1.0, requires_grad=False)

            for actual_datapoint, target_datapoint in zip(actual, target):
                # Avoid log(0)
                actual_safe = actual_datapoint + eps
                one_minus_actual_safe = (one - actual_datapoint) + eps

                term1 = target_datapoint * actual_safe.log()
                term2 = (one - target_datapoint) * one_minus_actual_safe.log()
                total_loss = total_loss + (term1 + term2)

            self.loss_value = (total_loss * Tensor(-1.0, requires_grad=False)) / Tensor(
                len(actual), requires_grad=False, weight=False
            )
            return self.loss_value
        else:
            assert isinstance(actual, Tensor) and isinstance(target, Tensor), (
                f"Values passed into {type(self).__name__}.loss() must be instances of tensorops.Tensor"
            )

            eps = 1e-15
            one_target = ones(target.shape)
            one_actual = ones(actual.shape)

            actual_safe = actual + eps
            one_minus_actual_safe = (one_actual - actual) + eps

            term1 = target * actual_safe.log()
            term2 = (one_target - target) * one_minus_actual_safe.log()

            neg_one = zeros(target.shape) - ones(target.shape)
            self.loss_value = (term1 + term2) * neg_one
            return self.loss_value


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute the multi-class cross entropy loss from unnormalised logits.

        Args:
            logits: Tensor of shape (batch, num_classes) containing raw scores.
            target: Tensor of either the same shape (one-hot/soft labels) or
                shape (batch,) containing integer class indices.

        Returns:
            Tensor: Scalar mean cross-entropy over the batch.
        """
        assert isinstance(logits, Tensor) and isinstance(target, Tensor), (
            f"Values passed into {type(self).__name__}.loss() must be instances of tensorops.Tensor"
        )
        assert logits.shape is not None and len(logits.shape) >= 2, (
            "CrossEntropyLoss expects logits with shape (batch, num_classes)"
        )

        batch_size, num_classes = logits.shape[0], logits.shape[-1]

        # Build one-hot targets if necessary so we can stay in Tensor land.
        if target.shape == logits.shape:
            # Already one-hot encoded (or soft labels)
            target_one_hot = target
        else:
            # Need to convert class indices to one-hot.
            # During graph construction (model init), target.values may be a placeholder
            # with zeros. In that case, we build a placeholder one-hot tensor that will
            # be filled with real values during forward pass via .values assignment.
            raw = target.values

            # If target is a placeholder (all zeros or uninitialized), create one-hot placeholder
            if raw is None or (
                isinstance(raw, list)
                and all(
                    v == 0.0 or (isinstance(v, list) and all(x == 0.0 for x in v))
                    for v in raw
                )
            ):
                # Create placeholder one-hot tensor with correct shape
                target_one_hot = Tensor(
                    [[0.0] * num_classes for _ in range(batch_size)],
                    requires_grad=False,
                )
            else:
                # Real data: convert class indices to one-hot
                assert len(raw) == batch_size, (
                    f"Target length {len(raw)} does not match batch size {batch_size}"
                )

                one_hot_rows: list[list[float]] = []
                for lbl in raw:
                    idx = int(lbl if not isinstance(lbl, list) else lbl[0])
                    assert 0 <= idx < num_classes, (
                        f"Class index {idx} out of range for {num_classes} classes"
                    )
                    row = [0.0] * num_classes
                    row[idx] = 1.0
                    one_hot_rows.append(row)

                target_one_hot = Tensor(one_hot_rows, requires_grad=False)
                target_one_hot.shape = (batch_size, num_classes)

        # Use the built-in stable softmax and take log to obtain log-probabilities
        probs = logits.softmax(axis=1)
        log_probs = probs.log()

        per_sample = (target_one_hot * log_probs).sum(axis=1) * Tensor(
            -1.0, requires_grad=False
        )
        self.loss_value = per_sample.sum(axis=0) / Tensor(
            float(batch_size), requires_grad=False
        )
        return self.loss_value
