"""
Online KL-Shampoo: core step math.

Hardcoded choices:
  - ScaledCANS coupled Newton-Schulz for S^{-1/2}
  - AdamC weight decay (lr²/lr_peak)
  - muP shape scaling
  - Nesterov momentum with variance correction

All functions expect batched 3D tensors: (N, m, n) for params/grads,
(N, m, m) for S_a/P_a, (N, n, n) for S_b/P_b.  The optimizer wrapper
handles unsqueezing 2D parameters to (1, m, n).
"""

import math
import torch
from torch import Tensor

from scaled_cans_coupled_ns import scaled_cans_coupled_ns


def init_preconditioners(
    grad: Tensor,
    S_a: Tensor,
    S_b: Tensor,
    P_a: Tensor,
    P_b: Tensor,
    eps: float,
) -> None:
    """Warm-start covariance factors S and preconditioners P from the first gradient.

    Args:
        grad:  (N, m, n)
        S_a:   (N, m, m)  mutated in-place.
        S_b:   (N, n, n)  mutated in-place.
        P_a:   (N, m, m)  mutated in-place.
        P_b:   (N, n, n)  mutated in-place.
        eps:   stability constant.
    """
    N, m, n = grad.shape
    frob_sq = grad.square().sum(dim=(-2, -1))                        # (N,)

    # ── S_a = sqrt(m / (n · ||G||²)) · GGᵀ  +  (||S_a||_F/√m + ε)I ──
    GGt = torch.bmm(grad, grad.mT)                                   # (N, m, m)
    scale_a = torch.sqrt(m / (n * frob_sq + eps)).view(N, 1, 1)
    S_a.copy_(scale_a * GGt)
    S_a.copy_(0.5 * (S_a + S_a.mT))
    frob_Sa = S_a.square().sum(dim=(-2, -1)).sqrt()                   # (N,)
    k_a = frob_Sa / math.sqrt(m)                                     # (N,)
    S_a.diagonal(dim1=-2, dim2=-1).add_(k_a.unsqueeze(-1) + eps)

    # ── S_b = sqrt(n / (m · ||G||²)) · GᵀG  +  (||S_b||_F/√n + ε)I ──
    GtG = torch.bmm(grad.mT, grad)                                   # (N, n, n)
    scale_b = torch.sqrt(n / (m * frob_sq + eps)).view(N, 1, 1)
    S_b.copy_(scale_b * GtG)
    S_b.copy_(0.5 * (S_b + S_b.mT))
    frob_Sb = S_b.square().sum(dim=(-2, -1)).sqrt()                   # (N,)
    k_b = frob_Sb / math.sqrt(n)
    S_b.diagonal(dim1=-2, dim2=-1).add_(k_b.unsqueeze(-1) + eps)

    # ── P = S^{-1/2} via ScaledCANS ──
    P_a.copy_(scaled_cans_coupled_ns(S_a))
    P_b.copy_(scaled_cans_coupled_ns(S_b))


def okls_step(
    param: Tensor,
    grad: Tensor,
    momentum: Tensor,
    S_a: Tensor,
    S_b: Tensor,
    P_a: Tensor,
    P_b: Tensor,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    weight_decay: float,
    lr_peak: float,
) -> Tensor:
    """One step of Online KL-Shampoo (batched).

    Args:
        param:        (N, m, n)  current parameter, fp32.
        grad:         (N, m, n)  gradient, fp32.
        momentum:     (N, m, n)  EMA momentum buffer (mutated in-place).
        S_a:          (N, m, m)  left covariance factor (mutated in-place).
        S_b:          (N, n, n)  right covariance factor (mutated in-place).
        P_a:          (N, m, m)  left preconditioner (mutated in-place).
        P_b:          (N, n, n)  right preconditioner (mutated in-place).
        step:         current step index (0-based).
        beta1:        momentum EMA coefficient.
        beta2:        preconditioner EMA coefficient.
        eps:          epsilon for numerical stability.
        lr:           current (scheduled) learning rate.
        weight_decay: decoupled weight decay coefficient.
        lr_peak:      peak learning rate for AdamC correction.

    Returns:
        New parameter tensor (N, m, n), fp32.
    """
    _, m, n = grad.shape

    # ── Warm start at step 0 ──
    if step == 0:
        init_preconditioners(grad, S_a, S_b, P_a, P_b, eps)

    # ── 1. Nesterov Momentum ──
    momentum.lerp_(grad, 1 - beta1)
    N_t = torch.lerp(grad, momentum, beta1)

    # ── 2. Preconditioner EMA ──
    GPb = torch.bmm(grad, P_b)                                      # (N, m, n)
    PaG = torch.bmm(P_a, grad)                                      # (N, m, n)

    S_a.baddbmm_(GPb, GPb.mT, beta=beta2, alpha=(1 - beta2) / n)
    S_a.copy_(0.5 * (S_a + S_a.mT))
    S_a.diagonal(dim1=-2, dim2=-1).add_(eps)

    S_b.baddbmm_(PaG.mT, PaG, beta=beta2, alpha=(1 - beta2) / m)
    S_b.copy_(0.5 * (S_b + S_b.mT))
    S_b.diagonal(dim1=-2, dim2=-1).add_(eps)

    # ── 3. Fast Matrix Roots ──
    P_a.copy_(scaled_cans_coupled_ns(S_a))
    P_b.copy_(scaled_cans_coupled_ns(S_b))

    # ── 4. Gradient Whitening ──
    U = torch.bmm(torch.bmm(P_a, N_t), P_b)

    # ── 5. Decoupled Weight Decay (AdamC) and Update ──
    v_nesterov = ((1 - beta1) / (1 + beta1)) * (1 + 2 * beta1 - 2 * beta1**3)
    c_momentum = v_nesterov**-0.5
    s_shape = (m / n) ** 0.5 / (m**0.5 + n**0.5)                     # muP

    wd_coeff = weight_decay * lr * (lr / lr_peak)                     # AdamC
    new_param = param * (1 - wd_coeff) - lr * c_momentum * s_shape * U

    return new_param
