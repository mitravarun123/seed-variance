"""Frequency profiles and profile-similarity measures for Phase 3."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analysis.fourier import concentration, dominant_frequencies, frequency_energy, grid_frequency_energy


def energy_profile(representations: np.ndarray, p: int) -> np.ndarray:
    """Compute normalized one-dimensional frequency energy over a p-cycle."""
    profile = frequency_energy(np.asarray(representations), exclude_dc=True)
    total = profile.sum()
    return profile / total if total else profile


def joint_energy_profile(representations: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return joint (a,b) energy and a symmetric marginal participation profile."""
    joint = grid_frequency_energy(representations, p)
    marginal = (joint.sum(axis=0) + joint.sum(axis=1)) / 2.0
    total = marginal.sum()
    if total:
        marginal = marginal / total
    return joint, marginal


def profile_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """Compare complete frequency profiles without treating hidden units as aligned."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError('Frequency profiles must have the same shape')
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denom) if denom else None
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {'cosine': cosine, 'pearson': None, 'spearman': None}
    return {
        'cosine': cosine,
        'pearson': float(pearsonr(a, b).statistic),
        'spearman': float(spearmanr(a, b).statistic),
    }


def frequency_rows(profile: np.ndarray, seed: int, stage: str, representation: str, top_k: int = 10) -> list[dict]:
    """Expand a profile into long-form machine-readable frequency rows."""
    profile = np.asarray(profile, dtype=np.float64)
    dom = set(dominant_frequencies(profile, top_k))
    concentration_stats = concentration(profile, top_k)
    return [
        {
            'seed': seed, 'stage': stage, 'representation': representation,
            'frequency': k, 'normalized_energy': float(profile[k]),
            'dominant_top_k': k in dom,
            'top_k_energy_fraction': concentration_stats['top_k_fraction'],
            'normalized_entropy': concentration_stats['normalized_entropy'],
        }
        for k in range(len(profile))
    ]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return float(len(a & b) / len(a | b)) if a | b else 1.0
