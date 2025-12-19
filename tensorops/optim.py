from __future__ import annotations

import math
import pickle
from typing import Any

from tensorops.tensor import Tensor


class Optim:
    def __init__(
        self, lr=1e-3, maximise: bool = False, weight_decay: float = 0.0
    ) -> None:
        self.lr = lr
        self.maximise = maximise
        self.weight_decay = weight_decay

    def step(self) -> None:
        pass

    def save(self, path: str) -> None:
        """
        Saves the optimiser to a `.pkl` file.

        Args:
            path (str): The file path where the optimiser should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> Any:
        """
        Loads an optimiser from a `.pkl` file.

        Args:
            path (str): The file path from which to load the optimiser.

        Returns:
            Optim: The loaded optimiser.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class Adam(Optim):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        maximise: bool = False,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrads: bool = False,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0.0 for param in parameters}
        self.v = {param: 0.0 for param in parameters}
        self.eps = eps
        self.amsgrads = amsgrads
        self.v_hat_max = {param: 0.0 for param in parameters}

    def step(self) -> None:
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            g_t = -param.grads if self.maximise else param.grads

            if self.weight_decay != 0.0:
                g_t += self.weight_decay * param.values

            self.m[param] = self.betas[0] * self.m[param] + (1 - self.betas[0]) * g_t
            self.v[param] = self.betas[1] * self.v[param] + (1 - self.betas[1]) * (
                g_t**2
            )

            m_hat_t = self.m[param] / (1 - self.betas[0] ** self.t)
            v_hat_t = self.v[param] / (1 - self.betas[1] ** self.t)

            if self.amsgrads:
                self.v_hat_max[param] = max(self.v_hat_max[param], v_hat_t)
                param.values -= (
                    self.lr * m_hat_t / (math.sqrt(self.v_hat_max[param]) + self.eps)
                )
            else:
                param.values -= self.lr * m_hat_t / (math.sqrt(v_hat_t) + self.eps)


class AdamW(Optim):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        maximise: bool = False,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrads: bool = False,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0.0 for param in parameters}
        self.v = {param: 0.0 for param in parameters}
        self.eps = eps
        self.amsgrads = amsgrads
        self.v_hat_max = {param: 0.0 for param in parameters}

    def step(self) -> None:
        self.t += 1
        filtered = list(filter(lambda p: p.requires_grad, self.parameters))
        for param in filtered:
            # Get gradient values (not the Tensor operation)
            g_t = param.grads.values if param.grads is not None else None
            if g_t is None:
                continue

            if self.maximise:
                g_t = [-g for g in g_t]

            if self.weight_decay != 0.0:
                param.values = [v - self.weight_decay * v for v in param.values]

            # Update biased first moment estimate
            if isinstance(self.m[param], float):
                self.m[param] = [
                    self.betas[0] * 0 + (1 - self.betas[0]) * g for g in g_t
                ]
            else:
                self.m[param] = [
                    self.betas[0] * m + (1 - self.betas[0]) * g
                    for m, g in zip(self.m[param], g_t)
                ]

            # Update biased second raw moment estimate
            if isinstance(self.v[param], float):
                self.v[param] = [
                    self.betas[1] * 0 + (1 - self.betas[1]) * (g**2) for g in g_t
                ]
            else:
                self.v[param] = [
                    self.betas[1] * v + (1 - self.betas[1]) * (g**2)
                    for v, g in zip(self.v[param], g_t)
                ]

            # Compute bias-corrected first moment estimate
            m_hat_t = [m / (1 - self.betas[0] ** self.t) for m in self.m[param]]
            # Compute bias-corrected second raw moment estimate
            v_hat_t = [v / (1 - self.betas[1] ** self.t) for v in self.v[param]]

            if self.amsgrads:
                self.v_hat_max[param] = (
                    [max(v_max, v) for v_max, v in zip(self.v_hat_max[param], v_hat_t)]
                    if not isinstance(self.v_hat_max[param], float)
                    else v_hat_t
                )
                param.values = [
                    v - self.lr * m / (math.sqrt(v_max) + self.eps)
                    for v, m, v_max in zip(param.values, m_hat_t, self.v_hat_max[param])
                ]
            else:
                param.values = [
                    v - self.lr * m / (math.sqrt(vt) + self.eps)
                    for v, m, vt in zip(param.values, m_hat_t, v_hat_t)
                ]


class SGD(Optim):
    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        maximise: bool = False,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: int = 0,
        momentum: int = 0,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.dampening = dampening
        self.nesterov = nesterov
        self.momentum = momentum
        self.b_t = {param: 0 for param in parameters}

    def step(self) -> None:
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            g_t = param.grads
            if self.weight_decay != 0.0:
                g_t += self.weight_decay * param.values

            if self.momentum != 0:
                if self.t > 1:
                    self.b_t[param] = (
                        self.momentum * self.b_t[param] + (1 - self.dampening) * g_t
                    )
                else:
                    self.b_t[param] = g_t

                if self.nesterov:
                    g_t += self.momentum * self.b_t[param]
                else:
                    g_t = self.b_t[param]

            if self.maximise:
                param.values += self.lr * g_t
            else:
                param.values -= self.lr * g_t
                param.values -= self.lr * g_t
