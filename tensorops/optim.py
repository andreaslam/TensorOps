from __future__ import annotations
import math
import pickle
from typing import Any

from tensorops.node import Node


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
        parameters: list[Node],
        lr: float = 1e-3,
        maximise: bool = False,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0.0 for param in parameters}
        self.v = {param: 0.0 for param in parameters}
        self.eps = eps
        self.amsgrad = amsgrad
        self.v_hat_max = {param: 0.0 for param in parameters}

    def step(self) -> None:
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            g_t = -param.grad if self.maximise else param.grad

            if self.weight_decay != 0.0:
                g_t += self.weight_decay * param.value

            self.m[param] = self.betas[0] * self.m[param] + (1 - self.betas[0]) * g_t
            self.v[param] = self.betas[1] * self.v[param] + (1 - self.betas[1]) * (
                g_t**2
            )

            m_hat_t = self.m[param] / (1 - self.betas[0] ** self.t)
            v_hat_t = self.v[param] / (1 - self.betas[1] ** self.t)

            if self.amsgrad:
                self.v_hat_max[param] = max(self.v_hat_max[param], v_hat_t)
                param.value -= (
                    self.lr * m_hat_t / (math.sqrt(self.v_hat_max[param]) + self.eps)
                )
            else:
                param.value -= self.lr * m_hat_t / (math.sqrt(v_hat_t) + self.eps)


class AdamW(Optim):
    def __init__(
        self,
        parameters: list[Node],
        lr: float = 1e-3,
        maximise: bool = False,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0.0 for param in parameters}
        self.v = {param: 0.0 for param in parameters}
        self.eps = eps
        self.amsgrad = amsgrad
        self.v_hat_max = {param: 0.0 for param in parameters}

    def step(self) -> None:
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            g_t = -param.grad if self.maximise else param.grad

            if self.weight_decay != 0.0:
                param.value -= self.weight_decay * param.value

            self.m[param] = self.betas[0] * self.m[param] + (1 - self.betas[0]) * g_t
            self.v[param] = self.betas[1] * self.v[param] + (1 - self.betas[1]) * (
                g_t**2
            )

            m_hat_t = self.m[param] / (1 - self.betas[0] ** self.t)
            v_hat_t = self.v[param] / (1 - self.betas[1] ** self.t)

            if self.amsgrad:
                self.v_hat_max[param] = max(self.v_hat_max[param], v_hat_t)
                param.value -= (
                    self.lr * m_hat_t / (math.sqrt(self.v_hat_max[param]) + self.eps)
                )
            else:
                param.value -= self.lr * m_hat_t / (math.sqrt(v_hat_t) + self.eps)


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
            g_t = param.grad
            if self.weight_decay != 0.0:
                g_t += self.weight_decay * param.value

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
                param.value += self.lr * g_t
            else:
                param.value -= self.lr * g_t
