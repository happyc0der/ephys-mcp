"""Linear decoders from binned spike counts to a continuous behavioural signal.

Both decoders are causal: the estimate at bin t uses counts up to bin t only.
Hyperparameters left as None are chosen on the tail of the training data, so
the caller's test split is never touched.
"""

from __future__ import annotations

import numpy as np

VALIDATION_FRACTION = 0.2
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
LEADS = (0, 1, 2, 3)  # bins by which neural activity leads behaviour


def r2_score(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def _select(make, candidates, counts: np.ndarray, y: np.ndarray):
    """Best candidate by R² on the tail of the training data."""
    cut = int(len(y) * (1 - VALIDATION_FRACTION))
    scores = [r2_score(y[cut:], make(c)._fit(counts[:cut], y[:cut]).predict(counts[cut:])).mean() for c in candidates]
    return candidates[int(np.argmax(scores))]


class RidgeDecoder:
    """Ridge regression on a short history of spike counts."""

    kind = "ridge"

    def __init__(self, n_lags: int = 10, alpha: float | None = None):
        self.n_lags, self.alpha = n_lags, alpha

    @property
    def params(self) -> dict:
        return {"history_bins": self.n_lags, "alpha": self.alpha}

    def _design(self, counts: np.ndarray) -> np.ndarray:
        lagged = [np.roll(counts, k, axis=0) for k in range(self.n_lags)]
        X = np.concatenate(lagged, axis=1)
        X[: self.n_lags] = 0.0
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    def _fit(self, counts: np.ndarray, y: np.ndarray) -> RidgeDecoder:
        X = self._design(counts)
        reg = self.alpha * np.eye(X.shape[1])
        reg[-1, -1] = 0.0
        self.W = np.linalg.solve(X.T @ X + reg, X.T @ y)
        return self

    def fit(self, counts: np.ndarray, y: np.ndarray) -> RidgeDecoder:
        if self.alpha is None:
            self.alpha = _select(lambda a: RidgeDecoder(self.n_lags, a), ALPHAS, counts, y)
        return self._fit(counts, y)

    def predict(self, counts: np.ndarray) -> np.ndarray:
        return self._design(counts) @ self.W


class KalmanDecoder:
    """Kalman filter (Wu et al. 2006): x_t = A x_{t-1} + w, z_t = H x_t + q.

    The state holds the target at the current and previous bin, which gives the
    smooth second-order dynamics real movements have. `lead_bins` models motor
    cortex firing ahead of the movement it drives.
    """

    kind = "kalman"

    def __init__(self, lead_bins: int | None = None):
        self.lead_bins = lead_bins

    @property
    def params(self) -> dict:
        return {"lead_bins": self.lead_bins, "state": "target at t and t-1"}

    def _fit(self, counts: np.ndarray, y: np.ndarray) -> KalmanDecoder:
        k = self.lead_bins
        self.d = y.shape[1]
        if k:
            counts, y = counts[:-k], y[k:]
        counts, y = counts[1:], np.hstack([y[1:], y[:-1]])
        self.z_mean, self.x_mean = counts.mean(axis=0), y.mean(axis=0)
        Z, X = counts - self.z_mean, y - self.x_mean
        X0, X1 = X[:-1], X[1:]
        self.A = np.linalg.lstsq(X0, X1, rcond=None)[0].T
        self.W = np.cov((X1 - X0 @ self.A.T).T) + 1e-9 * np.eye(X.shape[1])
        self.H = np.linalg.lstsq(X, Z, rcond=None)[0].T
        self.Q = np.cov((Z - X @ self.H.T).T) + 1e-6 * np.eye(Z.shape[1])
        return self

    def fit(self, counts: np.ndarray, y: np.ndarray) -> KalmanDecoder:
        if self.lead_bins is None:
            self.lead_bins = _select(KalmanDecoder, LEADS, counts, y)
        return self._fit(counts, y)

    def predict(self, counts: np.ndarray) -> np.ndarray:
        Z = counts - self.z_mean
        n = self.A.shape[0]
        x, P = np.zeros(n), np.eye(n)
        HtQinv = self.H.T @ np.linalg.inv(self.Q)
        HtQinvH = HtQinv @ self.H
        out = np.empty((Z.shape[0], n))
        for t in range(Z.shape[0]):
            x = self.A @ x
            P = self.A @ P @ self.A.T + self.W
            P = np.linalg.inv(np.linalg.inv(P) + HtQinvH)  # information form: small inverse only
            x = x + P @ HtQinv @ (Z[t] - self.H @ x)
            out[t] = x
        est = (out + self.x_mean)[:, : self.d]
        k = self.lead_bins
        if k:  # counts at bin t describe behaviour at t + k; realign to the input bins
            est = np.vstack([np.repeat(est[:1], k, axis=0), est[:-k]])
        return est


DECODERS = {"ridge": RidgeDecoder, "kalman": KalmanDecoder}
