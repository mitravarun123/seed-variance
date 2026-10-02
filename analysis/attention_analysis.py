"""Descriptive functional signatures for attention heads."""
from __future__ import annotations

import numpy as np


def head_signatures(cache: dict[str, np.ndarray], seed: int, stage: str) -> list[dict]:
    """Summarize query-to-token attention and head output norms.

    The sequence is [a, +, b, =]. These are behavioral descriptors, not
    functional labels or causal claims.
    """
    rows = []
    layers = sorted({int(key.split('.')[0][1:]) for key in cache if key.endswith('.pattern')})
    for layer in layers:
        patterns = np.asarray(cache[f'L{layer}.pattern'])  # N,H,4 at prediction position
        head_outputs = np.asarray(cache[f'L{layer}.heads'])  # N,H,D at prediction position
        entropy = -(patterns * np.log(np.clip(patterns, 1e-12, 1.0))).sum(axis=-1)
        for head in range(patterns.shape[1]):
            rows.append({
                'seed': seed, 'checkpoint': stage, 'layer': layer, 'head': head,
                'attention_entropy': float(entropy[:, head].mean()),
                'attention_to_operand_a': float(patterns[:, head, 0].mean()),
                'attention_to_operator': float(patterns[:, head, 1].mean()),
                'attention_to_operand_b': float(patterns[:, head, 2].mean()),
                'attention_to_equals': float(patterns[:, head, 3].mean()),
                'mean_output_norm': float(np.linalg.norm(head_outputs[:, head], axis=-1).mean()),
            })
    return rows


def signature_matrix(rows: list[dict], seed: int) -> np.ndarray:
    """Return comparable head feature rows in stable layer/head order."""
    selected = sorted((r for r in rows if r['seed'] == seed), key=lambda r: (r['layer'], r['head']))
    features = ['attention_entropy', 'attention_to_operand_a', 'attention_to_operator',
                'attention_to_operand_b', 'attention_to_equals', 'mean_output_norm',
                'ablation_delta_logit_margin']
    x = np.asarray([[row.get(key, 0.0) for key in features] for row in selected], dtype=np.float64)
    if x.size:
        scale = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(scale > 1e-12, scale, 1.0)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(norms > 1e-12, norms, 1.0)
    return x
