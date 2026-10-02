"""Deterministic one-factor-at-a-time Phase 4 condition planning."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from training.config import config_hash, validate

FACTORS = {
    'train_fraction': ('data', 'train_fraction'), 'prime': ('data', 'prime'),
    'width': ('model', 'd_model'), 'depth': ('model', 'n_layers'),
    'heads': ('model', 'n_heads'), 'weight_decay': ('training', 'weight_decay'),
    'learning_rate': ('training', 'learning_rate'), 'operation': ('data', 'operation'),
    'data_seed': ('data', 'data_seed'), 'data_split': ('data', 'data_seed'),
    'control': ('data', 'control'),
}
PRESETS = {
    'core': ['train_fraction', 'prime', 'width'],
    'architecture': ['depth', 'heads'],
    'optimization': ['weight_decay', 'learning_rate'],
    'task': ['operation'],
    'data': ['data_split'],
    'all': [factor for factor in FACTORS if factor != 'data_seed'],
}


def _label(value):
    if isinstance(value, float):
        return f'{value:g}'.replace('.', 'p')
    return re.sub(r'[^A-Za-z0-9]+', '', str(value))


def conditions(config, group='core', factor_value=None):
    """Return stable condition rows; the validated baseline appears once per preset."""
    if group in PRESETS:
        groups = PRESETS[group]
    elif group in FACTORS:
        groups = [group]
    else:
        raise ValueError(f'Unknown Phase-4 preset or factor: {group}')
    baseline = deepcopy(config)
    validate(baseline)
    baseline_hash = config_hash(baseline)
    baseline_id = f'baseline_{baseline_hash[:12]}'
    rows = []
    seen = set()
    for factor in groups:
        spec_key = 'data_seed' if factor == 'data_split' else factor
        spec = config['phase4']['experiments'].get(spec_key, [])
        if isinstance(spec, dict):
            if not spec.get('enabled', True):
                continue
            values = spec.get('values', [])
        else:
            values = spec
        section, key = FACTORS[factor]
        for value in values:
            if factor_value is not None and str(value) != str(factor_value):
                continue
            candidate = deepcopy(baseline)
            candidate[section][key] = value
            try:
                validate(candidate)
            except ValueError as exc:
                raise ValueError(f'Invalid {factor}={value}: {exc}') from exc
            resolved_hash = config_hash(candidate)
            if resolved_hash == baseline_hash:
                continue  # one shared baseline is trained once, not once per factor
            key_seen = (factor, resolved_hash)
            if key_seen in seen:
                continue
            seen.add(key_seen)
            condition_id = f'{factor}_{_label(value)}_{resolved_hash[:10]}'
            rows.append({'condition_id': condition_id, 'baseline_condition_id': baseline_id,
                         'baseline_config_hash': baseline_hash, 'config_hash': resolved_hash,
                         'factor': factor, 'factor_value': value, 'config': candidate})
    include_baseline = factor_value is None
    if factor_value is not None and group in FACTORS:
        section, key = FACTORS[group]
        include_baseline = str(factor_value) == str(baseline[section][key])
    if include_baseline:
        if group == 'all' or group in PRESETS or group in FACTORS:
            rows.insert(0, {'condition_id': baseline_id, 'baseline_condition_id': baseline_id,
                            'baseline_config_hash': baseline_hash, 'config_hash': baseline_hash,
                            'factor': 'baseline', 'factor_value': None, 'config': baseline})
    return rows
