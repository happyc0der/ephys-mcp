"""Linear decoders from binned spike counts to a continuous behavioural signal.

Both decoders are causal: the estimate at bin t uses counts up to bin t only.
Hyperparameters left as None are chosen by blocked cross-validation inside the
training data (contiguous folds, so autocorrelation cannot leak), and the
caller's test split is never touched.
"""

from __future__ import annotations

import itertools

import numpy as np

CV_FOLDS = 5
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
LEADS = (0, 1, 2, 3)  # bins by which neural activity leads behaviour
HISTORY_S = 0.5  # how far back the ridge decoder looks, whatever the bin size


def r2_score(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def _select(make, candidates, counts: np.ndarray, y: np.ndarray):
    """Best candidate by mean R² over contiguous folds of the training data."""
    edges = np.linspace(0, len(y), CV_FOLDS + 1).astype(int)
    scores = []
    for c in candidates:
        fold_scores = []
        for a, b in itertools.pairwise(edges):
            train = np.r_[0:a, b : len(y)]
            model = make(c)._fit(counts[train], y[train])
            fold_scores.append(r2_score(y[a:b], model.predict(counts[a:b])).mean())
        scores.append(np.mean(fold_scores))
    return candidates[int(np.argmax(scores))]


class RidgeDecoder:
    """Ridge regression on a short history of spike counts."""

    kind = "ridge"

    def __init__(self, n_lags: int | None = None, alpha: float | None = None, bin_s: float = 0.05):
        self.n_lags = n_lags if n_lags is not None else max(1, round(HISTORY_S / bin_s))
        self.alpha, self.bin_s = alpha, bin_s

    @property
    def params(self) -> dict:
        return {"history_bins": self.n_lags, "history_s": round(self.n_lags * self.bin_s, 3), "alpha": self.alpha}

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
            self.alpha = _select(lambda a: RidgeDecoder(self.n_lags, a, self.bin_s), ALPHAS, counts, y)
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

    def __init__(self, lead_bins: int | None = None, bin_s: float = 0.05):
        self.lead_bins, self.bin_s = lead_bins, bin_s

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
            self.lead_bins = _select(lambda k: KalmanDecoder(k, self.bin_s), LEADS, counts, y)
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
