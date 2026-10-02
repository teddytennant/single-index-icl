"""Tests for the single-index kernel and feature in-context learners."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import single_index_icl as si


def test_dimensions_d32():
    assert si.D1(32) == 32
    assert si.D2(32) == 527
    assert si.D1(32) + si.D2(32) == 559
    width = si.h2(jnp.zeros(32)).shape[-1]
    assert width == 32 * 33 // 2
    assert width == si.D2(32) + 1
    assert si.h2_width(32) == si.D2(32) + 1


def test_teacher_matches_hermite_and_is_normalized():
    coeffs = si.teacher_coefficients(4)
    assert coeffs[1] == pytest.approx(math.sqrt(3) / 2)
    assert coeffs[2] == pytest.approx(0.5)
    assert coeffs[3] == 0.0 and coeffs[4] == 0.0
    assert coeffs[1] ** 2 + coeffs[2] ** 2 == pytest.approx(1.0)
    z = np.linspace(-1.5, 1.7, 9)
    hermite = coeffs[1] * z + (coeffs[2] / math.sqrt(2.0)) * (z**2 - 1.0)
    assert np.allclose(np.asarray(si.sigma_star(z)), hermite)
    assert si.residual_variance(coeffs, K={1, 2}) == pytest.approx(0.0)


def _explicit_sphere_pair(d):
    # Angle chosen so (a·x)^2 / d - 1 = 0.8 * d - 1 is order-1 or larger.
    a = np.zeros(d, dtype=np.float64)
    x = np.zeros(d, dtype=np.float64)
    a[0] = math.sqrt(d)
    x[0] = math.sqrt(d * 0.8)
    x[1] = math.sqrt(d * 0.2)
    return jnp.asarray(a), jnp.asarray(x)


def test_h1_inner_product():
    a, x = _explicit_sphere_pair(6)
    got = si.h1(a) @ si.h1(x) / jnp.sqrt(6.0)
    expected = a @ x / jnp.sqrt(6.0)
    assert float(got) == pytest.approx(float(expected))
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    a_r = si.sample_sphere(k1, (), 9)
    x_r = si.sample_sphere(k2, (), 9)
    assert float(jnp.linalg.norm(a_r)) == pytest.approx(math.sqrt(9.0))
    got_r = si.h1(a_r) @ si.h1(x_r) / jnp.sqrt(9.0)
    expected_r = a_r @ x_r / jnp.sqrt(9.0)
    assert float(got_r) == pytest.approx(float(expected_r))


def test_h2_inner_product_identity():
    for d in (2, 5, 8, 17):
        a, x = _explicit_sphere_pair(d)
        lhs = float(si.h2(a) @ si.h2(x))
        rhs = math.sqrt(si.D2(d) / 2.0) * ((float(a @ x) ** 2) / d - 1.0)
        assert abs(lhs - rhs) <= 1e-6 * abs(rhs)
    key = jax.random.PRNGKey(2)
    for d in (3, 7, 12):
        rhs = 0.0
        for _ in range(8):
            k1, k2, key = jax.random.split(key, 3)
            a = si.sample_sphere(k1, (), d)
            x = si.sample_sphere(k2, (), d)
            lhs = float(si.h2(a) @ si.h2(x))
            rhs = math.sqrt(si.D2(d) / 2.0) * ((float(a @ x) ** 2) / d - 1.0)
            if abs(rhs) > 1e-2:
                break
        assert abs(rhs) > 1e-2
        assert abs(lhs - rhs) <= 1e-6 * abs(rhs)


def test_kernel_attention_trace_matches_sum():
    d, L = 3, 4
    key = jax.random.PRNGKey(3)
    a, X_ctx, y_ctx, x_q, y_q = si.make_prompts(key, n=1, L=L, d=d, noise_std=0.0)
    W = jnp.arange(1, si.feature_map(x_q[0]).shape[0] ** 2 + 1, dtype=jnp.float64)
    Dh = si.feature_map(x_q[0]).shape[0]
    W = W.reshape(Dh, Dh)
    trace_y = si.kernel_prediction_trace(W, x_q[0], X_ctx[0], y_ctx[0])
    sum_y = si.kernel_prediction_sum(W, x_q[0], X_ctx[0], y_ctx[0])
    assert float(trace_y) == pytest.approx(float(sum_y), rel=1e-10, abs=1e-10)


def test_feature_attention_trace_matches_sum():
    d, L = 4, 5
    key = jax.random.PRNGKey(4)
    _, X_ctx, y_ctx, x_q, _ = si.make_prompts(key, n=2, L=L, d=d)
    W = jnp.arange(d * d, dtype=jnp.float64).reshape(d, d)
    W = W + 0.3 * jnp.eye(d)
    for n in range(2):
        trace_u = si.feature_preactivation(W, x_q[n], X_ctx[n], y_ctx[n])
        sum_u = si.feature_preactivation_sum(W, x_q[n], X_ctx[n], y_ctx[n])
        assert float(trace_u) == pytest.approx(float(sum_u), rel=1e-10, abs=1e-10)


def _hand_normal_equation(H, y, lam, N0):
    """Hand-built ridge normal equation. Not a library ridge call."""
    Phi = np.asarray(H, dtype=np.float64).reshape(H.shape[0], -1, order="C")
    y = np.asarray(y, dtype=np.float64)
    p = Phi.shape[1]
    A = Phi.T @ Phi + float(lam) * float(N0) * np.eye(p)
    b = Phi.T @ y
    w = np.linalg.solve(A, b)
    # The solution of the normal equation must satisfy it.
    assert np.allclose(A @ w, b, rtol=1e-10, atol=1e-10)
    return w, A, b


def test_kernel_ridge_matches_hand_built_normal_equation():
    d, L, N = 2, 3, 4
    lam = 0.1
    N0 = 2.0
    key = jax.random.PRNGKey(5)
    _, X_ctx, y_ctx, x_q, y_q = si.make_prompts(key, n=N, L=L, d=d, noise_std=0.02)
    H = []
    for n in range(N):
        h_q = np.asarray(si.feature_map(x_q[n]))
        h_c = np.asarray(si.feature_map(X_ctx[n]))
        weighted = (np.asarray(y_ctx[n])[:, None] * h_c).sum(axis=0)
        H.append(np.outer(h_q, weighted) / L)
    H = np.stack(H)
    w, _, _ = _hand_normal_equation(H, y_q, lam, N0)
    W = si.fit_kernel_ridge(x_q, X_ctx, y_ctx, y_q, lam, N0)
    assert np.allclose(np.asarray(W).reshape(-1, order="C"), w, rtol=1e-8, atol=1e-8)
    # A different N0 must change the hand equation and still match.
    w_other, _, _ = _hand_normal_equation(H, y_q, lam, N0=7.0)
    W_other = si.fit_kernel_ridge(x_q, X_ctx, y_ctx, y_q, lam, 7.0)
    assert np.allclose(
        np.asarray(W_other).reshape(-1, order="C"), w_other, rtol=1e-8, atol=1e-8
    )
    assert not np.allclose(w, w_other)


def test_eta_and_kernel_error_hand_expanded():
    # c=0.5, L=8, D=4, L_tilde=10, nu_out=0.1
    # eta = (0.25*8) / (4 + 0.25*8) = 2/6 = 1/3
    # err = 0.1 + 0.25*(2/3)^2 + (4/10)*(1/3)^2
    #     = 1/10 + 1/9 + 0.4/9 = 1/10 + 7/45 = 23/90
    assert si.eta(0.5, 8, 4) == pytest.approx(1.0 / 3.0)
    got = si.kernel_error([(0.5, 4.0)], nu_out=0.1, L=8, L_tilde=10)
    assert got == pytest.approx(23.0 / 90.0)
    # Two copies of the same mode: the sum is not a replacement.
    two = si.kernel_error([(0.5, 4.0), (0.5, 4.0)], nu_out=0.1, L=8, L_tilde=10)
    assert two == pytest.approx(0.1 + 2.0 * (7.0 / 45.0))


def test_feature_error_hand_expanded():
    # c1=1, c2=1, alpha=1, alpha_tilde=3, cutoff high enough that c_k=0 for k>2.
    # rho_a = 1/sqrt(2), rho_t = sqrt(3)/2, r_t = 3/4, s^2 = 2/3
    # signal = 3/4 + (3/4)^2 = 21/16
    # ell=0, m=1: c2 * rho_a^2 * sqrt(2) * ((s^2-1)/2) = -sqrt(2)/12
    # ell=1, m=0: rho_a * s - rho_t = -1/(2*sqrt(3))
    # ell=2, m=0: rho_a^2 * s^2 - rho_t^2 = -5/12
    # bias = 1/72 + 1/12 + 25/144 = 13/48
    # E = 1 - 21/16 + 13/48 = -1/24
    coeffs = [0.0, 1.0, 1.0]
    got = si.feature_error(coeffs, alpha=1.0, alpha_tilde=3.0, k_max=6)
    assert got == pytest.approx(-1.0 / 24.0, rel=1e-12, abs=1e-12)
    # Equal alpha: bias vanishes, E = 1 - sum c_k^2 r^k.
    c1 = 1.0
    alpha = 1.0
    rho = c1 / math.sqrt(c1**2 + 1.0 / alpha)
    r = rho**2
    direct = 1.0 - (1.0 * r + 1.0 * r**2)
    assert si.feature_error(coeffs, alpha, alpha, k_max=4) == pytest.approx(direct)


def test_feature_error_reads_longer_coefficient_list():
    base = [0.0, 1.0, 0.5]
    longer = [0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.25]
    e_base = si.feature_error(base, alpha=2.0, alpha_tilde=2.0, k_max=6)
    e_long = si.feature_error(longer, alpha=2.0, alpha_tilde=2.0, k_max=6)
    e_cut = si.feature_error(longer, alpha=2.0, alpha_tilde=2.0, k_max=4)
    assert e_long != pytest.approx(e_base)
    assert e_cut == pytest.approx(si.feature_error(base, 2.0, 2.0, k_max=4))


def test_feature_limit_three_cases():
    coeffs = si.teacher_coefficients(2)
    alpha = 2.0
    assert si.feature_limit(0.5, alpha, coeffs) == 1.0
    assert si.feature_limit(1.5, alpha, coeffs) == 0.0
    # s=1: r = c1^2 / (c1^2 + 1/alpha) = (3/4) / (5/4) = 3/5
    # sum = (3/4)*(3/5) + (1/4)*(9/25) = 27/50
    # limit = 1 - 27/50 = 23/50
    c1 = math.sqrt(3.0) / 2.0
    r = (c1**2) / (c1**2 + 1.0 / alpha)
    direct = 1.0 - (c1**2 * r + (0.5**2) * r**2)
    assert direct == pytest.approx(23.0 / 50.0)
    assert si.feature_limit(1, alpha, coeffs) == pytest.approx(direct)
    assert si.feature_limit(1.0, alpha, coeffs) == pytest.approx(23.0 / 50.0)


def test_kernel_limit_s1_and_s2():
    coeffs = si.teacher_coefficients(2)
    alpha = 2.0
    K = {1, 2}
    nu = 0.0
    # s=1: c2^2 + c1^2 / (1 + alpha * 1! * c1^2) = 1/4 + 3/10 = 11/20
    assert si.kernel_limit(1, alpha, K, coeffs, nu_out=nu) == pytest.approx(11.0 / 20.0)
    # s=2: c2^2 / (1 + alpha * 2! * c2^2) = (1/4) / 2 = 1/8
    assert si.kernel_limit(2, alpha, K, coeffs, nu_out=nu) == pytest.approx(1.0 / 8.0)
    # Nonzero nu_out is added once, and a non-integer s drops the critical term.
    assert si.kernel_limit(1.5, alpha, K, coeffs, nu_out=0.2) == pytest.approx(0.2 + 0.25)


def test_binned_readout_linear_teacher_is_monotone():
    # Noiseless linear teacher, W scaled so u = y exactly. The optimal
    # readout on that well-scaled u is the identity, hence monotone.
    d = 4
    L = 5
    a = np.zeros(d, dtype=np.float64)
    a[0] = math.sqrt(d)
    ctx = []
    for j in range(L):
        v = np.zeros(d, dtype=np.float64)
        v[0] = 1.0 + 0.15 * j
        v[1 + (j % (d - 1))] = 0.7
        ctx.append(v * math.sqrt(d) / np.linalg.norm(v))
    X_ctx = jnp.asarray(np.stack(ctx))
    y_ctx = jnp.asarray(X_ctx @ a / math.sqrt(d))
    zs = np.linspace(-1.3, 1.3, 400)
    queries = []
    for z in zs:
        x = np.zeros(d, dtype=np.float64)
        x[0] = z
        x[1] = math.sqrt(d - z * z)
        queries.append(x)
    x_q = jnp.asarray(np.stack(queries))
    y_q = jnp.asarray(x_q @ a / math.sqrt(d))
    c = L / float(jnp.sum((X_ctx @ a) ** 2))
    W = c * jnp.outer(jnp.asarray(a), jnp.asarray(a))
    # Equation (9) on one query, trace versus the sum.
    u0 = si.feature_preactivation(W, x_q[0], X_ctx, y_ctx)
    u0_sum = si.feature_preactivation_sum(W, x_q[0], X_ctx, y_ctx)
    assert float(u0) == pytest.approx(float(u0_sum), rel=1e-10, abs=1e-10)
    X_rep = jnp.repeat(X_ctx[None, :, :], x_q.shape[0], axis=0)
    y_rep = jnp.repeat(y_ctx[None, :], x_q.shape[0], axis=0)
    u = si.predict_feature_u(W, x_q, X_rep, y_rep)
    assert np.allclose(np.asarray(u), np.asarray(y_q), rtol=1e-8, atol=1e-8)
    edges, means, counts = si.fit_binned_readout(
        W, x_q, X_rep, y_rep, y_q, n_bins=8
    )
    assert len(means) >= 8
    assert counts.min() > 0
    assert np.all(np.diff(means) > 0)


def test_short_cpu_train_predictions_are_finite():
    # Equation (20) is the infinite-pretraining limit. N=32 is not that
    # limit, so these predictions are not expected to match equation (20).
    d, L, N, N0 = 4, 6, 32, 8
    lam = 0.1
    noise = 0.01
    key = jax.random.PRNGKey(11)
    _, X_ctx, y_ctx, x_q, y_q = si.make_prompts(
        key, n=N, L=L, d=d, noise_std=noise, linear=False
    )
    W_k = si.fit_kernel_ridge(x_q, X_ctx, y_ctx, y_q, lam, N0)
    W_f = si.fit_feature_ridge(x_q, X_ctx, y_ctx, y_q, lam, N0)
    y_hat = si.predict_kernel(W_k, x_q, X_ctx, y_ctx)
    y_sum = si.predict_kernel_sum(W_k, x_q, X_ctx, y_ctx)
    u = si.predict_feature_u(W_f, x_q, X_ctx, y_ctx)
    assert np.all(np.isfinite(np.asarray(y_hat)))
    assert np.all(np.isfinite(np.asarray(u)))
    assert np.allclose(np.asarray(y_hat), np.asarray(y_sum), rtol=1e-8, atol=1e-8)
    # Independent trace formula, not the sum form.
    def one_trace(xq, X, y):
        H = si.kernel_attention(xq, X, y)
        return jnp.trace(W_k @ H.T)

    y_trace = jax.vmap(one_trace)(x_q, X_ctx, y_ctx)
    assert np.allclose(np.asarray(y_hat), np.asarray(y_trace), rtol=1e-8, atol=1e-8)
