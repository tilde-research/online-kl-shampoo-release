<p align="center">
  <img src="assets/logo.png" width="400" alt="Tilde Research">
</p>

# Online KL Shampoo

Online KL Shampoo (OKLS) is a zero-staleness, Kronecker-factored optimizer that
approximates full-matrix AdaGrad at language-model scale. It updates the
KL-optimal left and right covariance factors, computes fresh inverse-square-root
preconditioners, and applies the whitened update within the same training step.

The implementation combines:

- **KL-optimal Kronecker preconditioning** across both matrix axes.
- **Scaled CANS Coupled Newton–Schulz**, a 10-step inverse-root method using 27
  FP16 GEMMs with FP32 accumulation.
- **Zero-staleness updates**: covariance factors and their inverse roots are
  produced and consumed in the same step.
- **muP shape scaling**, **Nesterov momentum** with variance correction, and **AdamC**
  decoupled weight decay.

In our scaling experiments, OKLS achieves **1.45× Muon's parameter efficiency**
while maintaining roughly **98% of its training throughput**.

See the blog for the derivation, system design, and experiments:
https://blog.tilderesearch.com/blog/online-kl-shampoo

## How it works

For a matrix parameter with gradient $G_t \in \mathbb{R}^{m \times n}$, OKLS
maintains left and right covariance factors $S_{a,t}$ and $S_{b,t}$. Using the
previous preconditioners $P_{a,t-1}$ and $P_{b,t-1}$, it updates

```math
S_{a,t}
=
\beta_2 S_{a,t-1}
+
\frac{1-\beta_2}{n}
(G_t P_{b,t-1})(G_t P_{b,t-1})^\top,
```

```math
S_{b,t}
=
\beta_2 S_{b,t-1}
+
\frac{1-\beta_2}{m}
(P_{a,t-1}G_t)^\top(P_{a,t-1}G_t).
```

Fresh inverse roots are then computed immediately:

```math
P_{a,t} = S_{a,t}^{-1/2},
\qquad
P_{b,t} = S_{b,t}^{-1/2}.
```

The Nesterov momentum $N_t$ is whitened from both sides,

```math
U_t = P_{a,t} N_t P_{b,t},
```

and scaled using the matrix-shape-dependent muP multiplier

```math
c_{\mathrm{shape}}
=
\frac{\sqrt{m/n}}{\sqrt{m}+\sqrt{n}}.
```

The inverse roots are evaluated with Scaled CANS Coupled Newton–Schulz. The
iteration uses Chebyshev-optimized coefficients and deterministic per-step
scales to keep FP16 GEMM inputs within range while retaining FP32 accumulation
and persistent state.

## Requirements

- A CUDA-capable GPU.
- A recent CUDA-enabled PyTorch build with FP16 `torch.baddbmm` and FP32
  `out_dtype` support.

This repository currently provides the optimizer as a compact Python package
without a build configuration. Run examples from the repository root, or add
the repository root to `PYTHONPATH`.

## Usage

`OnlineKLShampoo` is a standard `torch.optim.Optimizer`. Pass only matrix
parameters that should receive OKLS updates, and use a separate optimizer for
embeddings, output heads, norms, biases, and other parameters.

```python
import torch

from okls import OnlineKLShampoo

# Choose the matrix weights that should use OKLS explicitly.
okls_params = []
for block in model.layers:
    okls_params.extend([
        block.attn.q_proj.weight,
        block.attn.k_proj.weight,
        block.attn.v_proj.weight,
        block.attn.o_proj.weight,
        block.mlp.gate_proj.weight,
        block.mlp.up_proj.weight,
        block.mlp.down_proj.weight,
    ])

# Manage all remaining parameters separately.
okls_param_ids = {id(p) for p in okls_params}
other_params = [p for p in model.parameters() if id(p) not in okls_param_ids]

okls = OnlineKLShampoo(
    okls_params,
    lr=0.01,
    beta1=0.95,
    beta2=0.95,
    eps=1e-12,
    weight_decay=0.0,
)
adamw = torch.optim.AdamW(other_params, lr=3e-4)

for input_ids, labels in dataloader:
    loss = model(input_ids, labels=labels).loss
    loss.backward()

    okls.step()
    adamw.step()

    okls.zero_grad()
    adamw.zero_grad()
```

When using a learning-rate scheduler, construct it normally from `okls`:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    okls,
    T_max=total_steps,
)

# After okls.step():
scheduler.step()
```

The initial learning rate is retained internally as `lr_peak` for the AdamC
weight-decay correction, while the scheduler updates the current `lr`.

### Batched matrices

The optimizer accepts both:

- **2D parameters** with shape `(m, n)`.
- **3D parameters** with shape `(N, m, n)`.

A 3D parameter is treated as a batch of `N` independent matrices. Each matrix
receives its own `(m, m)` and `(n, n)` covariance factors and preconditioners,
while all matrices share the optimizer hyperparameters.

## Hyperparameters

| Argument | Default | Description |
| --- | ---: | --- |
| `lr` | required | Current learning rate and fixed initial `lr_peak` used by AdamC. |
| `beta1` | `0.95` | Nesterov momentum EMA coefficient. |
| `beta2` | `0.95` | Kronecker-factor EMA coefficient. |
| `eps` | `1e-12` | Stability term added during factor initialization and updates. |
| `weight_decay` | `0.0` | Decoupled AdamC weight-decay coefficient. |

The implementation stores momentum, covariance factors, and preconditioners in
FP32. For a matrix of shape `(m, n)`, the persistent state contains one
`(m, n)` momentum matrix and two copies each of the `(m, m)` and `(n, n)`
factor shapes.

## Code structure

```text
okls/
├── __init__.py                    # public OnlineKLShampoo export
├── okls_optim.py                  # torch.optim.Optimizer wrapper and state
├── okls_step.py                   # factor, momentum, muP, and AdamC updates
└── scaled_cans_coupled_ns.py      # scaled 10-step inverse-root iteration
```

This is the compact, pure-PyTorch optimizer implementation. It does not include
the distributed optimizer-state offload and fused production training system
described in the blog post.

## References

- [Online KL Shampoo](https://blog.tilderesearch.com/blog/online-kl-shampoo)
- [Understanding and Improving Shampoo and SOAP via Kullback–Leibler Minimization](https://arxiv.org/abs/2509.03378)
- [Accelerating Newton–Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials](https://arxiv.org/abs/2506.10935)

## Citation

```bibtex
@misc{zhang2026onlineklshampoo,
  title  = {Online KL Shampoo},
  author = {Zhang, Ashley and Keigwin, Ben and Pai, Dhruv and Dewulf, Alec},
  year   = {2026},
  url    = {https://blog.tilderesearch.com/blog/online-kl-shampoo}
}
```
