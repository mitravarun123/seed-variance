"""Representation and functional-head comparisons across independent seeds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis.representation_analysis import linear_cka


def cka_layer_matrix(representations_a: dict[str, np.ndarray], representations_b: dict[str, np.ndarray]) -> list[dict]:
    """Compute all available layer-to-layer CKA values for aligned examples."""
    keys_a = sorted(k for k in representations_a if k.startswith('L') and 'resid' in k)
    keys_b = sorted(k for k in representations_b if k.startswith('L') and 'resid' in k)
    rows = []
    for a in keys_a:
        for b in keys_b:
            rows.append({'layer_a': a, 'layer_b': b, 'linear_cka': linear_cka(representations_a[a], representations_b[b])})
    return rows


def match_heads(features_a: np.ndarray, features_b: np.ndarray, minimum_similarity: float = 0.25) -> list[dict]:
    """Hungarian assignment of functional descriptors, with weak matches flagged."""
    if not len(features_a) or not len(features_b):
        return []
    similarities = np.clip(features_a @ features_b.T, -1.0, 1.0)
    row_idx, col_idx = linear_sum_assignment(-similarities)
    return [
        {'head_index_a': int(a), 'head_index_b': int(b),
         'functional_similarity': float(similarities[a, b]),
         'accepted_match': bool(similarities[a, b] >= minimum_similarity)}
        for a, b in zip(row_idx, col_idx)
    ]


def pairwise_matrix(vectors: dict[int, np.ndarray], metric) -> tuple[list[int], np.ndarray]:
    seeds = sorted(vectors)
    matrix = np.full((len(seeds), len(seeds)), np.nan)
    for i, seed_a in enumerate(seeds):
        for j, seed_b in enumerate(seeds):
            matrix[i, j] = metric(vectors[seed_a], vectors[seed_b])
    return seeds, matrix
