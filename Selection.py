"""Knockoff filter for FDR-controlled variable selection (Knockoff-ML)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _which_max_alt(row: np.ndarray) -> int:
    mx = row.max()
    idx = np.where(row == mx)[0]
    return int(idx[1]) if len(idx) > 1 else int(idx[0])


def mk_statistic(t0: np.ndarray, tk: np.ndarray, method: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Compute knockoff statistics (kappa, tau) for each feature."""
    t0 = np.asarray(t0, dtype=float).reshape(-1)
    tk = np.asarray(tk, dtype=float)
    if tk.ndim == 1:
        tk = tk.reshape(1, -1)
    combined = np.column_stack([t0, tk.T])
    combined = np.nan_to_num(combined, nan=0.0)

    kappa = np.array([_which_max_alt(combined[i]) for i in range(combined.shape[0])])
    if method == "max":
        tau = combined.max(axis=1) - np.sort(combined, axis=1)[:, -2]
    else:
        tau = []
        for row in combined:
            mx = row.max()
            rest = row[row < mx]
            med = np.median(rest) if len(rest) else 0.0
            tau.append(mx - med)
        tau = np.array(tau)
    return kappa, tau


def mk_threshold_by_stat(kappa: np.ndarray, tau: np.ndarray, m: int, fdr: float = 0.1) -> float:
    order = np.argsort(tau)[::-1]
    c0 = kappa[order] == 0
    temp0 = 0
    ratios = []
    for i in range(len(order)):
        temp0 += int(c0[i])
        temp1 = (i + 1) - temp0
        ratios.append((1.0 / m + temp1 / m) / max(1, temp0))
    ok = np.where(np.array(ratios) <= fdr)[0]
    return float(tau[order[ok[-1]]]) if len(ok) else float("inf")


def mk_q_by_stat(kappa: np.ndarray, tau: np.ndarray, m: int) -> np.ndarray:
    order = np.argsort(tau)[::-1]
    c0 = kappa[order] == 0
    temp0 = 0
    ratios = []
    for i in range(len(order)):
        temp0 += int(c0[i])
        temp1 = (i + 1) - temp0
        ratios.append((1.0 / m + temp1 / m) / max(1, temp0))
    ratios = np.array(ratios)
    q = np.ones(len(tau))
    positive = tau[order] > 0
    if not np.any(positive):
        return q
    index_bound = max(np.where(positive)[0])
    for i in range(len(order)):
        temp_index = slice(i, min(len(order), index_bound + 1))
        q[order[i]] = min(ratios[temp_index]) * c0[i] + (1 - c0[i])
    return q


def calculate_w_kappatau(t0: np.ndarray, tk: np.ndarray, m: int = 5) -> dict:
    """Compute knockoff W statistic and q-values."""
    t0 = np.asarray(t0, dtype=float).reshape(-1)
    tk = np.asarray(tk, dtype=float)
    if tk.ndim == 1:
        tk = tk.reshape(1, -1)
    t2_med = np.median(tk, axis=0)
    t2_max = np.max(tk, axis=0)
    w = (t0 - t2_med) * (t0 >= t2_max)
    kappa, tau = mk_statistic(t0, tk, method="median")
    q = mk_q_by_stat(kappa, tau, m=m)
    return {"w": w, "w_raw": t0 - t2_med, "kappa": kappa, "tau": tau, "q": q}


def select_features(
    feature_names: list[str],
    t0: np.ndarray,
    tk: np.ndarray,
    m: int = 5,
    fdr: float = 0.1,
) -> pd.DataFrame:
    """Apply knockoff filter at target FDR and return selection table."""
    stats = calculate_w_kappatau(t0, tk, m=m)
    thr = mk_threshold_by_stat(stats["kappa"], stats["tau"], m=m, fdr=fdr)
    selected = stats["w"] >= thr if np.isfinite(thr) else np.zeros_like(stats["w"], dtype=bool)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_original": t0,
            "importance_knockoff_median": np.median(tk, axis=0) if tk.ndim > 1 else tk,
            "w_statistic": stats["w"],
            "q_value": stats["q"],
            "selected": selected,
            "fdr_threshold": fdr,
        }
    ).sort_values("w_statistic", ascending=False)


def consensus_features(selections: list[pd.DataFrame], threshold: int = 3) -> list[str]:
    """Union rule: feature selected by >= threshold models (paper uses 3/5)."""
    if not selections:
        return []
    counts: dict[str, int] = {}
    for sel in selections:
        for feat in sel.loc[sel["selected"], "feature"]:
            counts[feat] = counts.get(feat, 0) + 1
    return sorted([f for f, c in counts.items() if c >= threshold])
