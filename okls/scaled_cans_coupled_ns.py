"""
Pure-torch 10-step coupled Newton-Schulz  S^{-1/2}  via CANS polynomials.
"""

import torch
import math


# ──────────── CANS polynomial coefficients (arXiv:2506.10935) ────────────
CANS_COEFFS = [
    (5.182503604966906, -5.178098480082684),
    (2.586120737395915, -0.6479542005271643),
    (2.567364126726186, -0.6454968804392178),
    (2.520560084348265, -0.6393528082067044),
    (2.410759275435182, -0.6248683598710716),
    (2.1883348130094173, -0.5952022073798908),
    (1.8595760874873613, -0.5504490972723968),
    (1.589020160467417, -0.5126569802066718),
    (1.5051653981684994, -0.5007377068751799),
    (1.5, -0.5),
]

# ──────────────── Per-step safe-scale ceilings (eigs in [0, 1]) ─────────────
_M = 16384.0
_Z_MAX = [
    1.0, 5.183, 13.403, 34.409, 86.731, 209.09,
    457.55, 850.85, 1352.0, 2035.0, 3052.5,
]
_Y_MAX = [1.0, 1.297, 1.726, 1.473, 1.545, 1.415, 1.273, 1.148, 1.0, 1.0, 1.0]
_P_MAX = [1.0, 3.98, 3.95, 3.88, 3.71, 3.32, 2.60, 1.73, 1.15, 1.008, 1.0]
_S_Z = [_M / z for z in _Z_MAX]
_S_Y = [_M / y for y in _Y_MAX]
_S_P = [_M / p for p in _P_MAX]

# ── Step-0 precomputed constants ──
_ALPHA_Y_0 = CANS_COEFFS[0][1] * _S_Y[1] / (_S_Y[0] ** 2)
_BETA_Y_0 = CANS_COEFFS[0][0] * _S_Y[1] / _S_Y[0]
_Z_SCALE_0 = CANS_COEFFS[0][1] * _S_Z[1] / _S_Y[0]
_Z_DIAG_ADD_0 = CANS_COEFFS[0][0] * _S_Z[1]

# ── Steps 1–9 precomputed constants ──
_ALPHA_P = []
_ALPHA_Y = []
_BETA_Y = []
_ALPHA_Z = []
_BETA_Z = []
for _k in range(1, 10):
    _a, _b = CANS_COEFFS[_k]
    _ALPHA_P.append(_S_P[_k] / (_S_Z[_k] * _S_Y[_k]))
    _ALPHA_Y.append(_b * _S_Y[_k + 1] / (_S_Y[_k] * _S_P[_k]))
    _BETA_Y.append(_a * _S_Y[_k + 1] / _S_Y[_k])
    _ALPHA_Z.append(_b * _S_Z[_k + 1] / (_S_Z[_k] * _S_P[_k]))
    _BETA_Z.append(_a * _S_Z[_k + 1] / _S_Z[_k])

# Absorb _S_Z[10] into last-step coefficients so runtime only needs 1/√L
_ALPHA_Z[8] /= _S_Z[10]
_BETA_Z[8] /= _S_Z[10]


def _estimate_max_eigenvalue(A: torch.Tensor) -> torch.Tensor:
    """Strict upper bound via min(Wolkowicz-Styan, Minc-Sainte-Marie). Cost: O(n²)."""
    n = A.size(-1)
    diag = A.diagonal(dim1=-2, dim2=-1)
    m = diag.sum(dim=-1) / n
    sq_norm = torch.sum(A**2, dim=(-2, -1))
    s_sq = torch.clamp((sq_norm / n) - (m**2), min=0.0)
    ws_bound = m + torch.sqrt(s_sq) * math.sqrt(n - 1)

    abs_A = torch.abs(A)
    d = torch.sum(abs_A, dim=-1)
    d_clamped = torch.clamp(d, min=1e-12)
    y = torch.einsum("...ij,...j->...i", abs_A, d_clamped)
    minc_bound = torch.max(y / d_clamped, dim=-1).values

    return torch.minimum(ws_bound, minc_bound)


def scaled_cans_coupled_ns(
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Computes S^{-1/2} via 10-step coupled Newton-Schulz with CANS polynomials.

    FP16 GEMM inputs with FP32 accumulation via torch.baddbmm's out_dtype
    parameter.  Per-step magnitude scaling prevents FP16 overflow.

    Args:
        S: (B, d, d) symmetric positive-definite matrix.
    Returns:
        W: (B, d, d) approximate S^{-1/2} in S.dtype.
    """
    B, d, _ = S.shape
    orig_dtype = S.dtype
    S = S.float()

    # ── Normalize: eigenvalues of Y₀ ∈ (0, 1] ──
    L = _estimate_max_eigenvalue(S) * 1.01
    Y = S * (_S_Y[0] / L.view(B, 1, 1))

    # ── Step 0: Z₁ = s_z₁·(a₀I + b₀Y₀),  Y₁ = s_y₁·(a₀Y₀ + b₀Y₀²) ──
    Z = Y.mul(_Z_SCALE_0)
    Z.diagonal(dim1=-2, dim2=-1).add_(_Z_DIAG_ADD_0)
    Y_half = Y.half()
    Y = torch.baddbmm(Y, Y_half, Y_half, torch.float32, beta=_BETA_Y_0, alpha=_ALPHA_Y_0)

    # ── Steps 1–8 (full coupled update) ──
    for k in range(8):
        Y_half = Y.half()
        Z_half = Z.half()
        P_half = torch.baddbmm(Z_half, Z_half, Y_half, beta=0.0, alpha=_ALPHA_P[k])
        Y, Z = (
            torch.baddbmm(Y, Y_half, P_half, torch.float32, beta=_BETA_Y[k], alpha=_ALPHA_Y[k]),
            torch.baddbmm(Z, P_half, Z_half, torch.float32, beta=_BETA_Z[k], alpha=_ALPHA_Z[k]),
        )

    # ── Step 9: Z only, with final 1/(s_{z,10}·√L) fused into coefficients ──
    Y_half = Y.half()
    Z_half = Z.half()
    P_half = torch.baddbmm(Z_half, Z_half, Y_half, beta=0.0, alpha=_ALPHA_P[8])
    inv_scale = torch.rsqrt(L).view(B, 1, 1)
    W = torch.baddbmm(Z, P_half, Z_half, torch.float32, beta=_BETA_Z[8], alpha=_ALPHA_Z[8])
    W.mul_(inv_scale)
    W = (W + W.mT) / 2.0
    return W.to(orig_dtype)
