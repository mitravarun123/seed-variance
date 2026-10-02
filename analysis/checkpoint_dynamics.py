"""Checkpoint inventory and stage labeling for Phase 3."""
from pathlib import Path

STAGES = ('initial', 'memorization', 'grokking', 'final')


def checkpoint_inventory(root, seeds):
    root = Path(root)
    rows = []
    for seed in seeds:
        for stage in STAGES:
            path = root / f'seed_{seed}' / f'{stage}.pt'
            rows.append({'seed': seed, 'stage': stage, 'checkpoint_path': str(path), 'available': path.is_file()})
    return rows


def stage_status(stage, inventory_row, summary):
    if not inventory_row['available']:
        if stage == 'grokking' and summary and summary.get('status') == 'completed' and summary.get('right_censored', False):
            return 'not_observed_within_training_horizon'
        if summary and summary.get('status') == 'interrupted':
            return 'source_run_interrupted'
        return 'checkpoint_missing'
    return 'available'
