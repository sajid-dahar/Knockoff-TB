"""Knockoff generation (Python port of Knockoff-ML.R by Wang et al., npj Digital Medicine 2025)."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def sparse_cor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    cmeans = x.mean(axis=0)
    cov = (x.T @ x - n * np.outer(cmeans, cmeans)) / max(n - 1, 1)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    outer = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        cor = np.where(outer > 0, cov / outer, 0.0)
    cor = np.nan_to_num(cor, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    return cov, cor


def sparse_cov_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    cx, cy = x.mean(axis=0), y.mean(axis=0)
    cov = (x.T @ y - n * np.outer(cx, cy)) / max(n - 1, 1)
    return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)


def _princomp_cov(covmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(covmat)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def create_mk(x: np.ndarray, m: int = 5, corr_max: float = 0.75) -> np.ndarray:
    """Generate M knockoff replicates using SCIT (Gimenez & Zou 2019)."""
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    cov_x, cor_x = sparse_cor(x)
    cor_x = np.clip(cor_x, -1, 1)

    if p > 1:
        dist = 1.0 - np.abs(cor_x)
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform(dist, checks=False), method="single")
        clusters = fcluster(z, t=1.0 - corr_max, criterion="distance")
    else:
        clusters = np.ones(p, dtype=int)

    x_k = np.zeros((n, p, m), dtype=float)
    index_exist: list[int] = []

    for cluster_id in np.unique(clusters):
        cluster_idx = np.where(clusters == cluster_id)[0]
        cluster_fitted = np.full((n, len(cluster_idx)), np.nan)
        cluster_residuals = np.full((n, len(cluster_idx)), np.nan)

        for i in cluster_idx:
            temp = np.abs(cor_x[i, :].copy())
            temp[clusters == cluster_id] = 0.0
            order = np.argsort(temp)[::-1]
            n_idx = min(
                len(order),
                int(np.sum(temp > 0.001)),
                int(np.floor(n ** (1.0 / 3.0))),
            )
            index = [j for j in order[:n_idx] if j != i]

            y = x[:, i]
            if not index:
                fitted = np.full(n, y.mean())
            else:
                x_pred = x[:, index]
                temp_xy = np.concatenate(
                    [[y.mean()], (x_pred.T @ y) / n - x_pred.mean(axis=0) * y.mean()]
                )

                x_exist_blocks: list[np.ndarray] = []
                for j in range(m):
                    overlap = [k for k in index if k in index_exist]
                    if overlap:
                        x_exist_blocks.append(x_k[:, overlap, j])
                if x_exist_blocks:
                    x_exist = np.column_stack(x_exist_blocks)
                    temp_xy = np.concatenate(
                        [
                            temp_xy,
                            (x_exist.T @ y) / n - x_exist.mean(axis=0) * y.mean(),
                        ]
                    )
                    temp_cov_cross = sparse_cov_cross(x_pred, x_exist)
                    cov_exist, _ = sparse_cor(x_exist)
                    temp_xx = np.block(
                        [
                            [cov_x[np.ix_(index, index)], temp_cov_cross],
                            [temp_cov_cross.T, cov_exist],
                        ]
                    )
                else:
                    temp_xx = cov_x[np.ix_(index, index)]

                temp_xx = np.pad(temp_xx, ((1, 0), (1, 0)), constant_values=0.0)
                temp_xx[0, 0] = 1.0

                evals, loadings = _princomp_cov(temp_xx)
                cump = np.cumsum(evals) / max(np.sum(evals), 1e-12)
                n_pc = int(np.searchsorted(cump, 0.999) + 1)
                pca_idx = [k for k in range(n_pc) if evals[k] > 0]

                v = loadings[:, pca_idx]
                s = evals[pca_idx]
                temp_inv = v @ np.diag(1.0 / s) @ v.T
                beta = temp_inv @ temp_xy

                fitted = np.full(n, beta[0])
                offset = 1
                b_orig = beta[offset : offset + len(index)]
                fitted += x_pred @ b_orig - x_pred.mean(axis=0) @ b_orig
                offset += len(index)

                if index and x_exist_blocks:
                    for j in range(m):
                        overlap = [k for k in index if k in index_exist]
                        if overlap:
                            temp_x = x_k[:, overlap, j]
                            b = beta[offset : offset + len(overlap)]
                            fitted += temp_x @ b - temp_x.mean(axis=0) @ b
                            offset += len(overlap)

            residuals = y - fitted
            pos = int(np.where(cluster_idx == i)[0][0])
            cluster_fitted[:, pos] = fitted
            cluster_residuals[:, pos] = residuals
            index_exist.append(i)

        sample_idx = np.random.randint(0, n, size=m)
        for j in range(m):
            x_k[:, cluster_idx, j] = cluster_fitted + cluster_residuals[sample_idx[j], :][:, None]

    return x_k


def _scale_continuous(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for j in range(x.shape[1]):
        if len(np.unique(x[:, j])) > 2:
            mu, sd = x[:, j].mean(), x[:, j].std()
            if sd > 0:
                x[:, j] = (x[:, j] - mu) / sd
    return x


def leverage_subsample(x: np.ndarray, seed: int = 2025) -> np.ndarray:
    """Leverage-based subsample indices (Knockoff-ML / SLEV-style)."""
    from sklearn.utils.extmath import randomized_svd

    x = np.asarray(x, dtype=float)
    n, p = x.shape
    n_al = int(min(np.floor(10 * (n ** (1.0 / 3.0)) * np.log(max(n, 2))), n))
    k = max(1, int(np.floor(np.sqrt(p * np.log(max(p, 2))))))
    rng = np.random.default_rng(seed)
    n_comp = min(k, min(n, p) - 1)
    if n_comp < 1:
        return np.arange(n)
    u, _, _ = randomized_svd(x, n_components=n_comp, random_state=seed)
    h1 = np.sum(u**2, axis=1)
    prob = 0.5 * (h1 / h1.sum()) + 0.5 * (1.0 / n)
    idx = rng.choice(n, size=min(n_al, n), replace=False, p=prob)
    return np.sort(idx)


def generate_knockoff(
    x: np.ndarray,
    m: int = 5,
    corr_max: float = 0.75,
    scaled: bool = False,
    seed: int = 2025,
    subsample: bool = True,
) -> dict:
    """Generate knockoffs and optional leverage subsample for SHAP."""
    np.random.seed(seed)
    x = np.asarray(x, dtype=float)
    if not scaled:
        x = _scale_continuous(x)

    x_mk = create_mk(x, m=m, corr_max=corr_max)
    out = {"X": x, "X_MK": x_mk}
    out["Index"] = leverage_subsample(x, seed=seed) if subsample else np.arange(x.shape[0])
    return out
