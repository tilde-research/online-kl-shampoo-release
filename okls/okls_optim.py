"""
Online KL-Shampoo: torch.optim.Optimizer wrapper.

Supports both 2D (m, n) and 3D (N, m, n) parameters.  3D params are
treated as N independent (m, n) matrices that share hyperparameters
but maintain separate preconditioners

Usage:
    # Matrix parameters only (2D or 3D).  Use a separate optimizer for
    # embeddings, heads, norms, biases, etc.
    matrix_params = [p for p in model.parameters() if p.ndim in (2, 3)]
    optimizer = OnlineKLShampoo(matrix_params, lr=0.09434)

    for step in range(total_steps):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # if using an LR scheduler
"""

import torch
from torch.optim import Optimizer

from .okls_step import okls_step


class OnlineKLShampoo(Optimizer):
    r"""Online KL-Shampoo optimizer for matrix (2D) and batched-matrix (3D) parameters.

    Implements the Online KL-Shampoo algorithm from [website] with:
      - ScaledCANS coupled Newton-Schulz for S^{-1/2}
      - AdamC weight decay correction (wd_eff = wd · lr · lr/lr_peak)
      - muP shape scaling
      - Nesterov momentum with exact variance correction

    Based on the KL-Shampoo framework from arXiv:2509.03378.

    Supports:
      - 2D parameters (m, n): single matrix with (m, m) and (n, n) preconditioners.
      - 3D parameters (N, m, n): N independent matrices sharing hyperparameters.
        but with separate per-matrix preconditioners.

    Non-matrix parameters (embeddings, biases, norms) should use a separate
    optimizer (e.g. AdamW).

    Args:
        params: Iterable of 2D or 3D parameters to optimize.
        lr: Learning rate (also used as lr_peak for AdamC).
        beta1: Momentum EMA coefficient (default: 0.9684).
        beta2: Preconditioner EMA coefficient (default: 0.9482).
        eps: Epsilon for numerical stability (default: 1e-9).
        weight_decay: Decoupled weight decay coefficient (default: 0.0303).

    Note:
        When using an LR scheduler, lr_peak is automatically set to the
        initial ``lr`` value.  The scheduler modifies ``group['lr']`` while
        ``group['lr_peak']`` stays fixed, giving the correct AdamC correction.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.09434,
        beta1: float = 0.9684,
        beta2: float = 0.9482,
        eps: float = 1e-9,
        weight_decay: float = 0.0303,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            lr_peak=lr,  # fixed for AdamC; not modified by LR schedulers
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Validate all params are 2D or 3D
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim not in (2, 3):
                    raise ValueError(
                        f"OnlineKLShampoo only supports 2D or 3D parameters, "
                        f"got shape {tuple(p.shape)}.  Use a separate optimizer "
                        f"for non-matrix parameters."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_peak = group["lr_peak"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                is_2d = p.ndim == 2
                state = self.state[p]

                # ── Lazy state initialization ──
                if len(state) == 0:
                    # Normalize to 3D: (N, m, n)
                    if is_2d:
                        batch, m, n = 1, p.shape[0], p.shape[1]
                    else:
                        batch, m, n = p.shape

                    state["step"] = 0
                    state["momentum"] = torch.zeros(batch, m, n, dtype=torch.float32, device=p.device)
                    state["S_a"] = torch.zeros(batch, m, m, dtype=torch.float32, device=p.device)
                    state["S_b"] = torch.zeros(batch, n, n, dtype=torch.float32, device=p.device)
                    state["P_a"] = torch.zeros(batch, m, m, dtype=torch.float32, device=p.device)
                    state["P_b"] = torch.zeros(batch, n, n, dtype=torch.float32, device=p.device)

                # Cast to fp32 and ensure 3D
                grad_fp32 = grad.float()
                param_fp32 = p.data.float()
                if is_2d:
                    grad_fp32 = grad_fp32.unsqueeze(0)
                    param_fp32 = param_fp32.unsqueeze(0)

                new_param = okls_step(
                    param=param_fp32,
                    grad=grad_fp32,
                    momentum=state["momentum"],
                    S_a=state["S_a"],
                    S_b=state["S_b"],
                    P_a=state["P_a"],
                    P_b=state["P_b"],
                    step=state["step"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    lr=lr,
                    weight_decay=wd,
                    lr_peak=lr_peak,
                )

                # Squeeze back if original was 2D
                if is_2d:
                    new_param = new_param.squeeze(0)

                p.data.copy_(new_param.to(p.dtype))
                state["step"] += 1

        return loss
