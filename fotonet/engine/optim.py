"""Optimizer factory and MuSGD implementation for FOTO-NET training."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def split_optimizer_params(model):
    """Return no-decay, decay, and bias parameter groups without dropping direct Parameters."""
    g_bnw, g_w, g_b = [], [], []
    seen = set()
    norm_types = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.SyncBatchNorm)

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if name == "bias":
                g_b.append(param)
            elif isinstance(module, norm_types) or param.ndim <= 1:
                g_bnw.append(param)
            else:
                g_w.append(param)

    return g_bnw, g_w, g_b


def _zeropower_via_newton_schulz(update, steps=5, eps=1e-7):
    """Muon-style Newton-Schulz orthogonalization for matrix-shaped updates."""
    original_shape = update.shape
    matrix = update.float().reshape(update.shape[0], -1)
    if matrix.numel() == 0:
        return update

    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.t()

    matrix = matrix / matrix.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = matrix @ matrix.t()
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix.reshape(original_shape).to(dtype=update.dtype)


class MuSGD(torch.optim.Optimizer):
    """SGD momentum with Muon-style orthogonalized matrix updates.

    Weight decay is deliberately decoupled from the orthogonalized gradient.
    Adding ``weight_decay * parameter`` to the update before Newton--Schulz
    normalization erases decay magnitude and turns regularization into an
    arbitrary rotation of the update.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.937,
        weight_decay=0.0,
        nesterov=True,
        ns_steps=5,
    ):
        if not math.isfinite(float(lr)) or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not math.isfinite(float(momentum)) or momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not math.isfinite(float(weight_decay)) or weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            use_muon=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            use_muon = bool(group.get("use_muon", True))

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MuSGD does not support sparse gradients")

                # Decoupled weight decay must happen outside zero-power
                # normalization. This also makes a zero-gradient step behave
                # exactly like ordinary multiplicative decay.
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                update = grad.detach().float()

                state = self.state[param]
                if momentum != 0.0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(update).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(update)
                    update = update.add(buf, alpha=momentum) if nesterov else buf

                if use_muon and update.ndim >= 2 and param.ndim >= 2:
                    update = _zeropower_via_newton_schulz(update, steps=ns_steps)

                param.add_(update.to(dtype=param.dtype), alpha=-lr)

        return loss


def build_optimizer(
    model,
    name="sgd",
    lr=1e-3,
    momentum=0.937,
    weight_decay=5e-4,
    nesterov=True,
    muon_ns_steps=5,
):
    """Build optimizer with FOTO-NET's bias/norm/decay grouping."""
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError(f"lr must be finite and positive, got {lr!r}")
    if not math.isfinite(float(momentum)) or float(momentum) < 0.0:
        raise ValueError(f"momentum must be finite and non-negative, got {momentum!r}")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be finite and non-negative, got {weight_decay!r}")
    g_bnw, g_w, g_b = split_optimizer_params(model)
    optimizer_name = str(name or "sgd").lower().replace("-", "_")
    use_nesterov = bool(nesterov) and float(momentum) > 0.0

    def groups_for(optimizer_kind):
        if optimizer_kind == "musgd":
            groups = [
                {"params": g_bnw, "weight_decay": 0.0, "use_muon": False},
                {"params": g_w, "weight_decay": weight_decay, "use_muon": True},
                {"params": g_b, "weight_decay": 0.0, "use_muon": False},
            ]
        else:
            groups = [
                {"params": g_bnw, "weight_decay": 0.0},
                {"params": g_w, "weight_decay": weight_decay},
                {"params": g_b, "weight_decay": 0.0},
            ]
        groups = [group for group in groups if group["params"]]
        if not groups:
            raise ValueError("Cannot build an optimizer for a model with no trainable parameters")
        return groups

    if optimizer_name in {"sgd", "nesterov_sgd", "auto"}:
        return torch.optim.SGD(
            groups_for("sgd"),
            lr=lr,
            momentum=momentum,
            nesterov=use_nesterov,
        )

    if optimizer_name in {"adamw", "adam_w"}:
        return torch.optim.AdamW(groups_for("adamw"), lr=lr)

    if optimizer_name in {"musgd", "mu_sgd", "muon_sgd"}:
        return MuSGD(
            groups_for("musgd"),
            lr=lr,
            momentum=momentum,
            weight_decay=0.0,
            nesterov=use_nesterov,
            ns_steps=muon_ns_steps,
        )

    raise ValueError("optimizer must be 'sgd', 'adamw', or 'musgd', got %r" % name)
