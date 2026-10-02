"""Kernel and feature in-context learners on single-index targets.

Small JAX slice of arXiv 2610.01712: degree-1 and degree-2 feature maps,
the two attention reductions, ridge fits for the attention matrices, a
binned second-stage readout, and the infinite-pretraining error formulas.

Not in this slice: the replica saddle point, Figure 2, and the N=12288
pretraining run. The degree-2 map keeps the trace-zero redundancy, so its
width is one larger than D_2.

Equation (20) is an infinite-pretraining limit. A finite prompt count N,
including the short CPU train with N=32, is not that limit, so fitted
predictions must not be compared to equation (20).
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Hermite coefficients of sigma_star, with E[sigma_star(g)^2] = 1.
# sigma_star(z) = c_1 * He_1(z) / sqrt(1!) + c_2 * He_2(z) / sqrt(2!).
C1 = math.sqrt(3.0) / 2.0
C2 = 0.5
KERNEL_DEGREES = (1, 2)


def D1(d):
    """Dimension of the degree-1 spherical harmonics: D_{1,d} = d."""
    return int(d)


def D2(d):
    """Dimension of trace-free degree-2 harmonics: D_{2,d} = (d-1)(d+2)/2."""
    return (int(d) - 1) * (int(d) + 2) // 2


def h2_width(d):
    """Width of h_2. One larger than D_2: the zero-trace constraint is kept."""
    return int(d) * (int(d) + 1) // 2


def teacher_coefficients(k_max=2):
    """c_1 = sqrt(3)/2, c_2 = 1/2, and c_k = 0 for k > 2.

    Index 0 is the constant mode, which is zero for this teacher.
    """
    coeffs = [0.0] * (int(k_max) + 1)
    if k_max >= 1:
        coeffs[1] = C1
    if k_max >= 2:
        coeffs[2] = C2
    return coeffs


def residual_variance(coeffs, K, noise_var=0.0):
    """Variance outside the kernel basis K, plus label-noise variance.

    For the teacher and K = {1, 2} this is 0 when noise_var is 0.
    """
    basis = set(K)
    tail = 0.0
    for k in range(1, len(coeffs)):
        if k not in basis:
            tail += float(coeffs[k]) ** 2
    return float(noise_var) + tail


def sigma_star(z):
    """Single-index teacher: sqrt(3)/2 * z + sqrt(2)/4 * (z^2 - 1)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return (jnp.sqrt(3.0) / 2.0) * z + (jnp.sqrt(2.0) / 4.0) * (z**2 - 1.0)


def sample_sphere(key, batch_shape, d):
    """Uniform on the sphere of radius sqrt(d): normal vector, then rescale."""
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)
    raw = jax.random.normal(key, batch_shape + (d,), dtype=jnp.float64)
    norm = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    return raw * (jnp.sqrt(d) / norm)


def single_index_labels(a, x, noise_std=0.0, key=None):
    """y = sigma_star(a^T x / sqrt(d)) + noise.

    noise_std is the standard deviation of additive Gaussian noise.
    """
    d = a.shape[-1]
    z = (x @ a) / jnp.sqrt(d)
    y = sigma_star(z)
    if noise_std and key is not None:
        y = y + float(noise_std) * jax.random.normal(key, y.shape, dtype=jnp.float64)
    return y


def make_prompts(key, n, L, d, noise_std=0.0, linear=False):
    """Draw one shared task vector and n prompts of context length L.

    Returns a (d,), X_ctx (n, L, d), y_ctx (n, L), x_query (n, d), y_query (n,).
    linear=True uses the noiseless teacher sigma_star(z) = z.
    """
    k_a, k_ctx, k_q, k_n1, k_n2 = jax.random.split(key, 5)
    a = sample_sphere(k_a, (), d)
    X_ctx = sample_sphere(k_ctx, (n, L), d)
    x_query = sample_sphere(k_q, (n,), d)
    if linear:
        y_ctx = (X_ctx @ a) / jnp.sqrt(d)
        y_query = (x_query @ a) / jnp.sqrt(d)
        if noise_std:
            y_ctx = y_ctx + float(noise_std) * jax.random.normal(
                k_n1, y_ctx.shape, dtype=jnp.float64
            )
            y_query = y_query + float(noise_std) * jax.random.normal(
                k_n2, y_query.shape, dtype=jnp.float64
            )
    else:
        y_ctx = single_index_labels(a, X_ctx, noise_std, k_n1)
        y_query = single_index_labels(a, x_query, noise_std, k_n2)
    return a, X_ctx, y_ctx, x_query, y_query


def h1(x):
    """Degree-1 feature map. Width d."""
    return jnp.asarray(x, dtype=jnp.float64)


def h2(x):
    """Degree-2 feature map with the trace-zero redundancy kept.

    S = x x^T - I. Upper triangle, diagonal as-is, off-diagonal times sqrt(2),
    then the whole vector times s = (D_2 / 2) ** 0.25 / sqrt(d).

    Width is d*(d+1)/2 = D_2 + 1. On the sphere of radius sqrt(d),
    h_2(a)^T h_2(x) = sqrt(D_2 / 2) * ((a^T x)^2 / d - 1).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    d = x.shape[-1]
    gram = x[..., :, None] * x[..., None, :]
    S = gram - jnp.eye(d, dtype=jnp.float64)
    iu, ju = jnp.triu_indices(d)
    vals = S[..., iu, ju]
    scale = jnp.where(iu == ju, 1.0, jnp.sqrt(2.0))
    vals = vals * scale
    D2_f = (d - 1) * (d + 2) / 2.0
    s = (D2_f / 2.0) ** 0.25 / jnp.sqrt(d)
    return vals * s


def feature_map(x):
    """h = concat(h_1, h_2), the K = {1, 2} kernel feature map."""
    return jnp.concatenate([h1(x), h2(x)], axis=-1)


def kernel_attention(x_query, X_ctx, y_ctx):
    """Equation (6): H_K = (1/L) * h(x_query) (sum_l y_l h(x_l))^T.

    x_query is (d,), X_ctx is (L, d), y_ctx is (L,). Returns (D_h, D_h).
    """
    h_q = feature_map(x_query)
    h_c = feature_map(X_ctx)
    L = y_ctx.shape[0]
    weighted = jnp.sum(y_ctx[:, None] * h_c, axis=0)
    return (h_q[:, None] * weighted[None, :]) / L


def kernel_prediction_trace(W, x_query, X_ctx, y_ctx):
    """y_hat = trace(W H_K^T) for one prompt."""
    H = kernel_attention(x_query, X_ctx, y_ctx)
    return jnp.trace(W @ H.T)


def kernel_prediction_sum(W, x_query, X_ctx, y_ctx):
    """(1/L) * sum_l y_l * h(x_query)^T W h(x_l)."""
    h_q = feature_map(x_query)
    h_c = feature_map(X_ctx)
    L = y_ctx.shape[0]
    # h_l^T W^T h_q = h_q^T W h_l
    dots = h_c @ (W.T @ h_q)
    return jnp.sum(y_ctx * dots) / L


def feature_attention(x_query, X_ctx, y_ctx):
    """H_F = (1/L) * x_query (sum_l y_l x_l)^T from equation (9)."""
    L = y_ctx.shape[0]
    weighted = jnp.sum(y_ctx[:, None] * X_ctx, axis=0)
    return (x_query[:, None] * weighted[None, :]) / L


def feature_preactivation(W, x_query, X_ctx, y_ctx):
    """u = trace(W H_F^T). Theorem 1 uses sigma_0(v) = v on this scalar."""
    H = feature_attention(x_query, X_ctx, y_ctx)
    return jnp.trace(W @ H.T)


def feature_preactivation_sum(W, x_query, X_ctx, y_ctx):
    """u = (1/L) * sum_l y_l * x_query^T W x_l."""
    L = y_ctx.shape[0]
    dots = X_ctx @ (W.T @ x_query)
    return jnp.sum(y_ctx * dots) / L


def _ridge_solve(H, y, lam, N0):
    """Closed-form ridge on vectorized square attention matrices.

    Minimizes sum_n (trace(W H_n^T) - y_n)^2 + lam * N0 * ||W||_F^2.
    Vectorization is row-major, so the Frobenius inner product is the
    Euclidean inner product of the vectors. H is (N, p, p).
    """
    H = jnp.asarray(H, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    n_prompts, p, _ = H.shape
    Phi = H.reshape((n_prompts, p * p), order="C")
    penalty = float(lam) * float(N0)
    eye = jnp.eye(p * p, dtype=jnp.float64)
    A = Phi.T @ Phi + penalty * eye
    b = Phi.T @ y
    w = jnp.linalg.solve(A, b)
    return w.reshape((p, p), order="C")


def fit_kernel_ridge(x_query, X_ctx, y_ctx, y_query, lam, N0):
    """Equation (7): ridge on the D_h by D_h kernel attention matrix.

    x_query (N, d), X_ctx (N, L, d), y_ctx (N, L), y_query (N,).
    """
    H = jax.vmap(kernel_attention)(x_query, X_ctx, y_ctx)
    return _ridge_solve(H, y_query, lam, N0)


def fit_feature_ridge(x_query, X_ctx, y_ctx, y_query, lam, N0):
    """First stage of the feature learner. sigma_0(v) = v, same ridge penalty.

    Predicts y_query from u = trace(W H_F^T). Returns the d by d matrix W.
    """
    H = jax.vmap(feature_attention)(x_query, X_ctx, y_ctx)
    return _ridge_solve(H, y_query, lam, N0)


def predict_kernel(W, x_query, X_ctx, y_ctx):
    """Batch kernel predictions via the trace formula. Shape (N,)."""
    return jax.vmap(lambda xq, X, y: kernel_prediction_trace(W, xq, X, y))(
        x_query, X_ctx, y_ctx
    )


def predict_kernel_sum(W, x_query, X_ctx, y_ctx):
    """Batch kernel predictions via the explicit context sum. Shape (N,)."""
    return jax.vmap(lambda xq, X, y: kernel_prediction_sum(W, xq, X, y))(
        x_query, X_ctx, y_ctx
    )


def predict_feature_u(W, x_query, X_ctx, y_ctx):
    """Batch first-stage preactivations u. Shape (N,)."""
    return jax.vmap(lambda xq, X, y: feature_preactivation(W, xq, X, y))(
        x_query, X_ctx, y_ctx
    )


def binned_conditional_mean(u, y, n_bins=8):
    """Equal-width binned estimate of E[y | u].

    Returns edges (n_bins+1,), means (n_bins,), counts (n_bins,).
    Empty bins have a NaN mean. The second stage asks for at least 8 bins.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    u_np = np.asarray(u, dtype=np.float64).reshape(-1)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    if u_np.shape[0] != y_np.shape[0]:
        raise ValueError("u and y must have the same length")
    lo = float(u_np.min())
    hi = float(u_np.max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("u must be finite")
    if hi == lo:
        edges = np.linspace(lo - 0.5, hi + 0.5, n_bins + 1)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    # Rightmost edge is exclusive in digitize; nudge it so the max is kept.
    edges = edges.copy()
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.digitize(u_np, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    means = np.empty(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            means[b] = np.nan
        else:
            means[b] = float(y_np[mask].mean())
    return edges, means, counts


def fit_binned_readout(W, x_query, X_ctx, y_ctx, y_query, n_bins=8):
    """Second stage: freeze W, estimate sigma(u) on a fresh batch by binning."""
    if n_bins < 8:
        raise ValueError("second stage uses at least 8 bins")
    u = predict_feature_u(W, x_query, X_ctx, y_ctx)
    return binned_conditional_mean(u, y_query, n_bins=n_bins)


def eta(c, L, D):
    """eta(c, L, D) = c^2 * L / (D + c^2 * L)."""
    c2 = float(c) ** 2
    return c2 * float(L) / (float(D) + c2 * float(L))


def kernel_error(components, nu_out, L, L_tilde):
    """Equation (20).

    components is a list of (c_k, D_k). Returns
    nu_out + sum_k [c_k^2 * (1 - eta)^2 + (D_k / L_tilde) * eta^2]
    with eta = eta(c_k, L, D_k).

    This is the infinite-pretraining limit. Do not compare it to a finite-N
    fit such as the short CPU train (N=32).
    """
    err = float(nu_out)
    for c_k, D_k in components:
        e = eta(c_k, L, D_k)
        err += (float(c_k) ** 2) * (1.0 - e) ** 2 + (float(D_k) / float(L_tilde)) * (e**2)
    return err


def _as_coeff_list(coeffs, k_max):
    """Pad or truncate so index k holds c_k for k = 0..k_max."""
    values = [float(c) for c in coeffs]
    cutoff = int(k_max)
    if len(values) < cutoff + 1:
        values = values + [0.0] * (cutoff + 1 - len(values))
    else:
        values = values[: cutoff + 1]
    return values, cutoff


def feature_error(coeffs, alpha, alpha_tilde, k_max=8):
    """Equation (21).

    coeffs[k] = c_k (coeffs[0] is the constant mode). The sums run through
    K_max, default 8, so a longer coefficient list can be passed by raising
    k_max. Degrees above the cutoff are dropped; missing degrees are zero.

    rho_alpha = c_1 / sqrt(c_1^2 + 1/alpha)
    r_alpha = rho_alpha^2
    s = rho_alpha / rho_alpha_tilde

    E = 1 - sum_{k>=1} c_k^2 * r_alpha_tilde^k
        + sum_ell [sum_m c_{ell+2m} * rho_alpha^{ell+2m}
                   * sqrt((ell+2m)! / ell!) * s^ell / m!
                   * ((s^2 - 1)/2)^m
                   - c_ell * rho_alpha_tilde^ell]^2

    ell runs from 0 through the cutoff. The written sum does not start at 1,
    and the degree-0 mode is the constant term of the scaled Hermite expansion.
    """
    values, cutoff = _as_coeff_list(coeffs, k_max)
    c1 = values[1]
    rho = c1 / math.sqrt(c1 * c1 + 1.0 / float(alpha))
    rho_t = c1 / math.sqrt(c1 * c1 + 1.0 / float(alpha_tilde))
    r_t = rho_t * rho_t
    s = rho / rho_t
    half = 0.5 * (s * s - 1.0)

    signal = 0.0
    for k in range(1, cutoff + 1):
        signal += values[k] ** 2 * (r_t**k)

    bias = 0.0
    for ell in range(0, cutoff + 1):
        acc = 0.0
        max_m = (cutoff - ell) // 2
        for m in range(0, max_m + 1):
            n = ell + 2 * m
            c_n = values[n]
            if c_n == 0.0:
                continue
            log_sqrt = 0.5 * (math.lgamma(n + 1) - math.lgamma(ell + 1))
            acc += (
                c_n
                * (rho**n)
                * math.exp(log_sqrt)
                * (s**ell)
                / math.factorial(m)
                * (half**m)
            )
        diff = acc - values[ell] * (rho_t**ell)
        bias += diff * diff
    return 1.0 - signal + bias


def feature_limit(s, alpha, coeffs):
    """Corollary (22): limit of the feature error as d -> infinity.

    L = L_tilde = alpha * d^s. The limit is 1 if 0 < s < 1,
    1 - sum_k c_k^2 r_alpha^k if s = 1, and 0 if s > 1.
    """
    s = float(s)
    if s > 1.0:
        return 0.0
    if s < 1.0:
        return 1.0
    values = [float(c) for c in coeffs]
    c1 = values[1]
    rho = c1 / math.sqrt(c1 * c1 + 1.0 / float(alpha))
    r = rho * rho
    captured = 0.0
    for k in range(1, len(values)):
        captured += values[k] ** 2 * (r**k)
    return 1.0 - captured


def kernel_limit(s, alpha, K, coeffs, nu_out=0.0):
    """Corollary (23) for the kernel learner.

    nu_out + sum_{k in K, k>s} c_k^2
    + (if s is an integer in K: c_s^2 / (1 + alpha * s! * c_s^2)).
    """
    s = float(s)
    alpha = float(alpha)
    err = float(nu_out)
    s_int = int(round(s))
    s_is_int = abs(s - s_int) < 1e-9
    basis = set(int(k) for k in K)
    for k in sorted(basis):
        c_k = float(coeffs[k])
        if k > s:
            err += c_k**2
        elif s_is_int and k == s_int:
            err += c_k**2 / (1.0 + alpha * math.factorial(s_int) * c_k**2)
    return err
