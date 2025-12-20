from __future__ import annotations

import math
import pickle
from typing import Any

from tensorops.tensor import Tensor


def _numeric_grad_for_param(param: Tensor) -> list[float] | None:
    if param.grads is None:
        return None
    # Prefer flat memory view for correct element count; fallback to values.
    if getattr(param.grads, "flat", None) is not None:
        g_src = param.grads.flat
        g_list = list(g_src)
    else:
        g_src = param.grads.values
        if g_src is None:
            return None
        # If bytes-like, cast to float view
        if isinstance(g_src, (bytearray, memoryview, bytes)):
            g_list = list(memoryview(g_src).cast("f"))
        else:
            g_list = list(g_src)

    p_vals = param.values
    if len(g_list) != len(p_vals):
        if len(g_list) == 1:
            return [g_list[0]] * len(p_vals)
        raise ValueError(
            f"Gradient length {len(g_list)} does not match param length {len(p_vals)}"
        )

    return g_list


class Optim:
    def __init__(
        self,
        lr=1e-3,
        maximise: bool = False,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        grad_clip_value: float | None = None,
    ) -> None:
        self.lr = lr
        self.maximise = maximise
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value

    def _apply_clipping(self, g: list[float]) -> list[float]:
        # Clip by global L2 norm if configured
        if self.grad_clip_norm is not None:
            norm2 = sum(gi * gi for gi in g)
            if norm2 > 0:
                norm = math.sqrt(norm2)
                if norm > self.grad_clip_norm:
                    scale = self.grad_clip_norm / norm
                    g = [gi * scale for gi in g]
        # Elementwise clamp if configured
        if self.grad_clip_value is not None:
            c = self.grad_clip_value
            g = [max(-c, min(c, gi)) for gi in g]
        return g

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
        self.m: dict[Tensor, list[float] | None] = {param: None for param in parameters}
        self.v: dict[Tensor, list[float] | None] = {param: None for param in parameters}
        self.eps = eps
        self.amsgrads = amsgrads
        self.v_hat_max: dict[Tensor, list[float] | None] = {
            param: None for param in parameters
        }

    def step(self) -> None:
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            g_t = _numeric_grad_for_param(param)
            if g_t is None:
                continue

            p_vals = param.values

            if self.maximise:
                g_t = [-g for g in g_t]

            if self.weight_decay != 0.0:
                g_t = [g + self.weight_decay * v for g, v in zip(g_t, p_vals)]

            g_t = self._apply_clipping(g_t)

            # Update biased first moment estimate
            if self.m[param] is None:
                self.m[param] = [(1 - self.betas[0]) * g for g in g_t]
            else:
                self.m[param] = [
                    self.betas[0] * m + (1 - self.betas[0]) * g
                    for m, g in zip(self.m[param], g_t)  # pyright: ignore[reportArgumentType]
                ]

            # Update biased second raw moment estimate
            if self.v[param] is None:
                self.v[param] = [(1 - self.betas[1]) * (g**2) for g in g_t]
            else:
                self.v[param] = [
                    self.betas[1] * v + (1 - self.betas[1]) * (g**2)
                    for v, g in zip(self.v[param], g_t)  # pyright: ignore[reportArgumentType]
                ]

            # Bias-corrected estimates
            m_list = self.m[param] or []
            v_list = self.v[param] or []
            m_hat_t = [m / (1 - self.betas[0] ** self.t) for m in m_list]
            v_hat_t = [v / (1 - self.betas[1] ** self.t) for v in v_list]

            if self.amsgrads:
                if self.v_hat_max[param] is None:
                    self.v_hat_max[param] = list(v_hat_t)
                else:
                    self.v_hat_max[param] = [
                        max(v_max, v)
                        for v_max, v in zip(self.v_hat_max[param], v_hat_t)  # pyright: ignore[reportArgumentType]
                    ]
                denom = [
                    math.sqrt(v_max) + self.eps
                    for v_max in (self.v_hat_max[param] or [])
                ]
            else:
                denom = [math.sqrt(v) + self.eps for v in v_hat_t]

            # Use flat view to fetch and update parameters
            p_vals_floats = (
                list(param.flat)
                if getattr(param, "flat", None) is not None
                else list(p_vals)
            )
            updated = [
                p - self.lr * m / d for p, m, d in zip(p_vals_floats, m_hat_t, denom)
            ]
            param.values = updated


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
        self.m: dict[Tensor, list[float] | None] = {param: None for param in parameters}
        self.v: dict[Tensor, list[float] | None] = {param: None for param in parameters}
        self.eps = eps
        self.amsgrads = amsgrads
        self.v_hat_max: dict[Tensor, list[float] | None] = {
            param: None for param in parameters
        }

    def step(self) -> None:
        self.t += 1
        filtered = list(filter(lambda p: p.requires_grad, self.parameters))
        for param in filtered:
            g_t = _numeric_grad_for_param(param)
            if g_t is None:
                continue

            # Use flat view for reliable float values
            p_vals = (
                list(param.flat)
                if getattr(param, "flat", None) is not None
                else list(param.values)
            )

            if self.maximise:
                g_t = [-g for g in g_t]

            # Apply gradient clipping
            g_t = self._apply_clipping(g_t)

            # Update biased first moment estimate
            if self.m[param] is None:
                self.m[param] = [(1 - self.betas[0]) * g for g in g_t]
            else:
                self.m[param] = [
                    self.betas[0] * m + (1 - self.betas[0]) * g
                    for m, g in zip(self.m[param], g_t)
                ]

            # Update biased second raw moment estimate
            if self.v[param] is None:
                self.v[param] = [(1 - self.betas[1]) * (g**2) for g in g_t]
            else:
                self.v[param] = [
                    self.betas[1] * v + (1 - self.betas[1]) * (g**2)
                    for v, g in zip(self.v[param], g_t)
                ]

            # Compute bias-corrected moment estimates
            m_hat_t = [m / (1 - self.betas[0] ** self.t) for m in self.m[param]]
            v_hat_t = [v / (1 - self.betas[1] ** self.t) for v in self.v[param]]

            if self.amsgrads:
                if self.v_hat_max[param] is None:
                    self.v_hat_max[param] = list(v_hat_t)
                else:
                    self.v_hat_max[param] = [
                        max(v_max, v)
                        for v_max, v in zip(self.v_hat_max[param], v_hat_t)
                    ]
                denom = [
                    math.sqrt(v_max) + self.eps
                    for v_max in (self.v_hat_max[param] or [])
                ]
            else:
                denom = [math.sqrt(vt) + self.eps for vt in v_hat_t]

            # Decoupled weight decay
            updated = [
                v - self.lr * m / d - (self.lr * self.weight_decay * v)
                for v, m, d in zip(p_vals, m_hat_t, denom)
            ]
            param.values = updated


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
            g_t = _numeric_grad_for_param(param)
            if g_t is None:
                continue

            p_vals = param.values

            if self.weight_decay != 0.0:
                g_t = [g + self.weight_decay * v for g, v in zip(g_t, p_vals)]

            if self.momentum != 0:
                if self.t > 1 and not isinstance(self.b_t[param], int):
                    self.b_t[param] = [
                        self.momentum * b + (1 - self.dampening) * g
                        for b, g in zip(self.b_t[param], g_t)
                    ]
                else:
                    self.b_t[param] = list(g_t)

                if self.nesterov:
                    g_t = [g + self.momentum * b for g, b in zip(g_t, self.b_t[param])]
                else:
                    g_t = self.b_t[param]

            # Apply gradient clipping
            g_t = self._apply_clipping(g_t)

            if self.maximise:
                param.values = [v + self.lr * g for v, g in zip(p_vals, g_t)]
            else:
                param.values = [v - self.lr * g for v, g in zip(p_vals, g_t)]
