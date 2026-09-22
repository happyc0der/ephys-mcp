"""Low-dimensional latent factors underlying population activity.

Two models on square-root-transformed spike counts, per trial:

* PCA: orthogonal directions of maximal variance, no noise model.
* GPFA (Yu et al. 2009): factor analysis whose factors evolve as smooth Gaussian
  processes, fitted by EM. Gives denoised single-trial trajectories and a
  timescale per factor. Written from the paper's equations.

Both return trajectories with shape (n_trials, n_bins, n_factors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

MAX_FACTORS = 20
EM_ITERS = 60
EM_TOL = 1e-4


@dataclass
class LatentFit:
    method: str
    trajectories: np.ndarray  # (n_trials, n_bins, n_factors)
    loading: np.ndarray  # (n_units, n_factors)
    variance_explained: np.ndarray  # PCA: fraction of total variance; GPFA: fraction of the shared (signal) variance
    timescales_s: np.ndarray | None  # GPFA only
    log_likelihood: list[float] | None  # GPFA only, per EM iteration
    mean: np.ndarray  # (n_units,)


def preprocess(counts: np.ndarray) -> np.ndarray:
    """sqrt stabilises Poisson variance so high-rate units do not dominate."""
    return np.sqrt(np.asarray(counts, dtype=float))


def fit_pca(counts: np.ndarray, n_factors: int) -> LatentFit:
    """counts: (n_trials, n_bins, n_units)."""
    y = preprocess(counts)
    n_trials, n_bins, n_units = y.shape
    flat = y.reshape(-1, n_units)
    mean = flat.mean(axis=0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    var = s**2 / (len(flat) - 1)
    loading = vt[:n_factors].T
    traj = ((flat - mean) @ loading).reshape(n_trials, n_bins, n_factors)
    return LatentFit("pca", traj, loading, var[:n_factors] / var.sum(), None, None, mean)


def _logdet_spd(a: np.ndarray) -> float:
    """log det of a symmetric positive-definite matrix via Cholesky; LU-based slogdet underflows here."""
    return 2.0 * float(np.log(np.diag(np.linalg.cholesky(a))).sum())


def _rbf(n_bins: int, bin_s: float, tau: float, noise: float = 1e-3) -> np.ndarray:
    t = np.arange(n_bins) * bin_s
    d = (t[:, None] - t[None, :]) ** 2
    return (1 - noise) * np.exp(-d / (2 * tau**2)) + noise * np.eye(n_bins)


def _estep_trial(y_c: np.ndarray, C: np.ndarray, R_diag: np.ndarray, K_inv: list, K_logdet: float, p: int, T: int):
    """Posterior mean (T, p), posterior covariance (pT, pT) and marginal log-likelihood of one trial.

    Latents are stacked factor-major (x_1[0..T), x_2[0..T), ...). The posterior precision is
    blockdiag(K_k^-1) + kron(C^T R^-1 C, I_T); by the Woodbury identity the marginal
    log-likelihood needs only that same matrix, never the (Tn x Tn) observation covariance.
    """
    R_inv = 1.0 / R_diag
    CtRC = (C.T * R_inv) @ C
    prec = np.kron(CtRC, np.eye(T))
    for k in range(p):
        prec[k * T : (k + 1) * T, k * T : (k + 1) * T] += K_inv[k]
    cov = np.linalg.inv(prec)
    b = ((y_c * R_inv) @ C).T.reshape(-1)  # C~^T R~^-1 y, factor-major
    mean = cov @ b
    n = C.shape[0]
    quad = float((y_c**2 * R_inv).sum() - b @ mean)
    logdet = T * float(np.log(R_diag).sum()) + K_logdet + _logdet_spd(prec)
    ll = -0.5 * (quad + logdet + T * n * np.log(2 * np.pi))
    return mean.reshape(p, T).T, cov, ll


def fit_gpfa(counts: np.ndarray, n_factors: int, bin_s: float, iters: int = EM_ITERS) -> LatentFit:
    """counts: (n_trials, n_bins, n_units). EM over C, d, R (diagonal) and per-factor timescales."""
    y = preprocess(counts)
    n_trials, T, n = y.shape
    p = n_factors
    flat = y.reshape(-1, n)
    d = flat.mean(axis=0)
    y_c = y - d

    # Initialise with factor analysis via PCA of the covariance (a standard, adequate start).
    cov_y = np.cov(flat.T)
    evals, evecs = np.linalg.eigh(cov_y)
    order = np.argsort(evals)[::-1][:p]
    C = evecs[:, order] * np.sqrt(np.maximum(evals[order], 1e-6))
    R = np.diag(np.maximum(np.diag(cov_y) - (C**2).sum(axis=1), 1e-3))
    tau = np.full(p, min(0.1, T * bin_s / 2))  # seconds

    ll_hist: list[float] = []
    N = n_trials * T
    Y = y.reshape(-1, n)
    S_yy_diag = (Y**2).sum(axis=0)
    for it in range(iters):
        K = [_rbf(T, bin_s, tau[k]) for k in range(p)]
        K_inv = [np.linalg.inv(Kk) for Kk in K]
        K_logdet = float(sum(_logdet_spd(Kk) for Kk in K))
        R_diag = np.diag(R)

        # E-step: posterior moments of the latents, trial by trial
        Ex = np.empty((n_trials, T, p))
        sum_xx = np.zeros((p, p))  # sum over trials and time of E[x_t x_t^T]
        sum_xx_time = [np.zeros((T, T)) for _ in range(p)]  # per factor, sum over trials of E[x_k x_k^T]
        ll = 0.0
        for i in range(n_trials):
            m, cov, ll_i = _estep_trial(y_c[i], C, R_diag, K_inv, K_logdet, p, T)
            Ex[i], ll = m, ll + ll_i
            for t in range(T):
                idx = np.arange(p) * T + t
                sum_xx += cov[np.ix_(idx, idx)] + np.outer(m[t], m[t])
            for k in range(p):
                blk = slice(k * T, (k + 1) * T)
                sum_xx_time[k] += cov[blk, blk] + np.outer(m[:, k], m[:, k])
        ll_hist.append(float(ll))
        if it > 2 and abs(ll_hist[-1] - ll_hist[-2]) < EM_TOL * abs(ll_hist[-2]):
            break

        # M-step for [C d] and diagonal R: the factor-analysis closed forms with expected statistics
        Ex_flat = Ex.reshape(-1, p)
        G = np.zeros((p + 1, p + 1))
        G[:p, :p], G[p, p] = sum_xx, N
        G[:p, p] = G[p, :p] = Ex_flat.sum(axis=0)
        S_xy = np.concatenate([Ex_flat, np.ones((N, 1))], axis=1).T @ Y  # (p+1, n)
        Cd = np.linalg.solve(G, S_xy).T  # (n, p+1)
        C, d = Cd[:, :p], Cd[:, p]
        y_c = y - d
        R = np.diag(np.maximum((S_yy_diag - (Cd * S_xy.T).sum(axis=1)) / N, 1e-4))

        # M-step for timescales: one scalar optimisation per factor
        for k in range(p):
            S = sum_xx_time[k]

            def nll(log_tau, S=S):
                Kk = _rbf(T, bin_s, float(np.exp(log_tau[0])))
                return 0.5 * (n_trials * _logdet_spd(Kk) + np.trace(np.linalg.solve(Kk, S)))

            # A timescale longer than the window is unresolvable and makes K numerically singular.
            bounds = [(np.log(bin_s), np.log(T * bin_s))]
            res = minimize(nll, [np.log(tau[k])], method="L-BFGS-B", bounds=bounds)
            tau[k] = float(np.exp(res.x[0]))

    # Final E-step and orthonormalised factors ordered by explained shared variance (as in the paper).
    K = [_rbf(T, bin_s, tau[k]) for k in range(p)]
    K_inv = [np.linalg.inv(Kk) for Kk in K]
    K_logdet = float(sum(_logdet_spd(Kk) for Kk in K))
    Ex = np.stack([_estep_trial(y_c[i], C, np.diag(R), K_inv, K_logdet, p, T)[0] for i in range(n_trials)])
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    traj = Ex @ Vt.T * s  # orthonormalised latents, ordered by variance
    shared = s**2  # variance carried by each orthonormalised factor; noise R is excluded by construction
    order_tau = tau @ np.abs(Vt.T)  # timescales mapped to the orthonormal factors, approximately
    return LatentFit("gpfa", traj, U, shared / shared.sum(), order_tau, ll_hist, d)


def summarize(fit: LatentFit, bin_s: float) -> dict:
    ve = fit.variance_explained
    out = {
        "method": fit.method,
        "n_factors": int(fit.trajectories.shape[2]),
        "n_trials": int(fit.trajectories.shape[0]),
        "n_bins_per_trial": int(fit.trajectories.shape[1]),
        "bin_s": bin_s,
        "variance_explained_per_factor": [round(float(v), 3) for v in ve],
        "cumulative_variance_explained": [round(float(v), 3) for v in np.cumsum(ve)],
        "variance_basis": "fraction of total variance"
        if fit.method == "pca"
        else "fraction of shared variance (spiking noise excluded, so factors sum to 1)",
    }
    if fit.timescales_s is not None:
        out["timescale_s_per_factor"] = [round(float(v), 3) for v in fit.timescales_s]
    if fit.log_likelihood:
        out["em_iterations"] = len(fit.log_likelihood)
        out["log_likelihood_change"] = round(float(fit.log_likelihood[-1] - fit.log_likelihood[0]), 1)
    return out
