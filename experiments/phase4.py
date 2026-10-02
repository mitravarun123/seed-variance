"""Sequential, resumable robustness runs isolated from Phases 1–3."""
from __future__ import annotations

import csv
import json
import math
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.conditions import conditions
from training.config import config_hash, validate
from training.trainer import Trainer
from training.utils import cleanup, device, json_save


def _read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def _runtime_per_run(config):
    """Use observed Phase-2 wall time per step when available; never invent timing."""
    roots = [Path('results/multi_seed'), *Path('results').glob('*/phase2')]
    candidates = sorted({path for root in roots for path in root.glob('seed_*/summary.json')})
    rates = []
    for path in candidates:
        summary = _read_json(path)
        if summary and summary.get('total_training_steps', 0) and summary.get('total_training_time_seconds'):
            rates.append(summary['total_training_time_seconds'] / summary['total_training_steps'])
    if not rates:
        return None
    return float(np.median(rates)) * config['training']['max_steps']


def _storage_per_run(config):
    roots = [Path('checkpoints/multi_seed'), Path('checkpoints/phase2'), *Path('checkpoints').glob('*/phase2')]
    sizes = []
    for root in roots:
        if root.exists():
            for seed_dir in root.glob('seed_*'):
                try:
                    sizes.append(sum(path.stat().st_size for path in seed_dir.rglob('*.pt')))
                except OSError:
                    pass
    if sizes:
        return float(np.median(sizes))
    # Approximate optimizer + model bytes multiplied by the configured checkpoint
    # count. This is a planning estimate; real serialized sizes are read above
    # when Phase 2 measurements exist.
    from models.tiny_transformer import TinyTransformer
    model = TinyTransformer(config['data']['prime'], config['model'])
    parameters = sum(p.numel() for p in model.parameters())
    bytes_per_checkpoint = parameters * 12  # fp32 weights plus two Adam moments
    training = config['training']
    periodic = math.ceil(training['max_steps'] / training['checkpoint_every'])
    milestones = sum(step <= training['max_steps'] for step in training.get('milestones', []))
    return float(bytes_per_checkpoint * (periodic + milestones + 4))


def plan(config, preset='core', factor_value=None, seeds=None):
    validate(config)
    rows = conditions(config, preset, factor_value)
    selected_seeds = list(config['phase4']['model_seeds'] if seeds is None else seeds)
    run_count = len(rows) * len(selected_seeds)
    per_run = _runtime_per_run(config)
    storage = _storage_per_run(config)
    print(f'PHASE 4 DRY RUN | preset={preset} | conditions={len(rows)} | seeds/condition={len(selected_seeds)} | training runs={run_count}')
    print(f'Baseline: p={config["data"]["prime"]}, operation={config["data"]["operation"]}, train_fraction={config["data"]["train_fraction"]}, width={config["model"]["d_model"]}, depth={config["model"]["n_layers"]}, heads={config["model"]["n_heads"]}, lr={config["training"]["learning_rate"]}, weight_decay={config["training"]["weight_decay"]}')
    if per_run is None:
        print('Estimated runtime: unavailable (no completed Phase-2 timing measurements found).')
    else:
        seconds = per_run * run_count
        print(f'Estimated runtime: {seconds / 3600:.2f} hours total (median observed Phase-2 seconds/step × configured max_steps).')
    print(f'Checkpoint storage: approximately {storage * run_count / (1024**3):.3f} GiB (observed Phase-2 checkpoint size when available; otherwise Adam-state size × configured retained-checkpoint count).')
    for row in rows:
        print(f'  {row["condition_id"]}: {row["factor"]}={row["factor_value"]}; config={row["config_hash"][:12]}')
    return rows, selected_seeds, {'training_runs': run_count, 'runtime_seconds': per_run * run_count if per_run is not None else None,
                                  'checkpoint_bytes': storage * run_count}


def _find_resume(checkpoint_dir):
    latest = checkpoint_dir / 'latest.pt'
    return latest if latest.is_file() else None


def _run_phase3_tier(config, condition_root, checkpoint_root, seeds, tier):
    """Run only the requested analysis tier into its isolated Phase-4 subdirectory."""
    analysis_config = json.loads(json.dumps(config))
    analysis_root = condition_root / 'analysis' if tier == 1 else condition_root / 'analysis' / f'tier{tier}'
    analysis_config['phase3'] = {
        'checkpoint_root': str(checkpoint_root.resolve()),
        'source_results_root': str((condition_root / 'runs').resolve()),
        'output_root': str(analysis_root.resolve()),
        'seeds': list(seeds), 'checkpoints': ['initial', 'final'],
        'batch_size': 256, 'representation_examples': 512,
        'patching_examples': 128, 'ablation_examples': 128,
        'top_k': [3, 5, 10], 'tier': tier, 'minimum_head_match_similarity': 0.25,
        'activation_patching': tier >= 3,
    }
    from experiments.phase3 import run_phase3
    mode = 'robustness' if tier == 1 else 'ablation' if tier == 2 else 'all'
    return run_phase3(analysis_config, analysis=mode, seeds=list(seeds), checkpoint='all')


def _csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def aggregate(root):
    """Write descriptive condition/factor tables; keep censored runs explicit."""
    root = Path(root); records = []
    for path in sorted(root.glob('*/conditions/*/runs/seed_*/summary.json')):
        summary = _read_json(path)
        if not summary:
            continue
        metadata = _read_json(path.parents[2] / 'condition.json') or {}
        records.append({**{k: metadata.get(k) for k in ('condition_id', 'baseline_condition_id', 'factor', 'factor_value')},
                        **summary})
    aggregate_dir = root / 'aggregate'; aggregate_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        json_save({'status': 'no_completed_runs', 'message': 'No Phase-4 run summaries were found; no scientific metrics were computed.'}, aggregate_dir / 'paper_summary.json')
        _csv(aggregate_dir / 'robustness_summary.csv', [])
        _csv(aggregate_dir / 'factor_effects.csv', [])
        _csv(aggregate_dir / 'condition_fourier_summary.csv', [])
        return {'status': 'no_completed_runs', 'runs': 0}
    frame = pd.DataFrame(records)
    frame.to_csv(aggregate_dir / 'run_metrics.csv', index=False)
    summaries = []
    for condition_id, group in frame.groupby('condition_id', dropna=False):
        completed = group[group['status'] == 'completed']
        grokked = completed[completed['grokking_detected'].astype(bool)]
        row = {'condition_id': condition_id, 'factor': group['factor'].iloc[0], 'factor_value': group['factor_value'].iloc[0],
               'seeds_requested': int(len(group)), 'completed': int(len(completed)),
               'grokked': int(grokked.shape[0]), 'right_censored': int(completed.get('right_censored', pd.Series(dtype=bool)).fillna(False).sum()),
               'grokking_rate': float(len(grokked) / len(completed)) if len(completed) else None,
               'median_grokking_step_observed': float(grokked['grokking_step'].median()) if len(grokked) else None,
               'median_final_test_accuracy': float(completed['final_test_accuracy'].median()) if len(completed) else None,
               'fourier_similarity_mean': None, 'within_condition_cka_mean': None,
               'head_ablation_max_abs_delta_loss': None, 'mlp_ablation_max_abs_delta_loss': None}
        analysis = root / str(group['factor'].iloc[0]) / 'conditions' / str(condition_id) / 'analysis'
        fourier = analysis / 'tables' / 'fourier_similarity.csv'
        cka = analysis / 'tables' / 'circuit_similarity.csv'
        if fourier.is_file():
            values = pd.read_csv(fourier).get('cosine')
            if values is not None and values.notna().any(): row['fourier_similarity_mean'] = float(values.mean())
        if cka.is_file():
            values = pd.read_csv(cka).get('final_residual_cka')
            if values is not None and values.notna().any(): row['within_condition_cka_mean'] = float(values.mean())
        tier2_tables = analysis / 'tier2' / 'tables'
        for filename, key in [('head_ablation.csv', 'head_ablation_max_abs_delta_loss'),
                              ('mlp_ablation.csv', 'mlp_ablation_max_abs_delta_loss')]:
            path = tier2_tables / filename
            if path.is_file():
                values = pd.read_csv(path).get('delta_loss')
                if values is not None and values.notna().any(): row[key] = float(values.abs().max())
        summaries.append(row)
    _csv(aggregate_dir / 'robustness_summary.csv', summaries)
    baseline_rows = [r for r in summaries if r['factor'] == 'baseline']
    baseline = baseline_rows[0] if baseline_rows else None
    effects = []
    for row in summaries:
        if row['factor'] == 'baseline': continue
        effect = {'condition_id': row['condition_id'], 'factor': row['factor'], 'factor_value': row['factor_value'],
                  'baseline_condition_id': baseline['condition_id'] if baseline else None}
        for key, out in [('median_grokking_step_observed', 'delta_median_grokking_step'),
                         ('fourier_similarity_mean', 'delta_fourier_similarity'),
                         ('within_condition_cka_mean', 'delta_within_condition_cka')]:
            a = row.get(key); b = baseline.get(key) if baseline else None
            effect[out] = a - b if a is not None and b is not None else None
        effects.append(effect)
    _csv(aggregate_dir / 'factor_effects.csv', effects)
    baseline_id = baseline['condition_id'] if baseline else None
    profile_by_condition = {}
    fourier_rows = []
    for summary_row in summaries:
        condition_root = root / str(summary_row['factor']) / 'conditions' / str(summary_row['condition_id'])
        profile_path = condition_root / 'analysis' / 'tables' / 'fourier_profiles.csv'
        if not profile_path.is_file():
            continue
        profile_frame = pd.read_csv(profile_path)
        selected = profile_frame[(profile_frame['stage'] == 'final') & (profile_frame['representation'] == 'embedding')]
        profile_by_condition[summary_row['condition_id']] = {}
        for seed, seed_frame in selected.groupby('seed'):
            seed_frame = seed_frame.sort_values('frequency')
            energies = seed_frame['normalized_energy'].to_numpy(dtype=float)
            frequency_ids = seed_frame['frequency'].to_numpy(dtype=int)
            non_dc = frequency_ids != 0
            usable = energies[non_dc]
            total = float(usable.sum())
            normalized = usable / total if total else usable
            order = np.argsort(normalized)[::-1]
            top_count = min(5, len(order))
            dominant = frequency_ids[non_dc][order[:top_count]].tolist()
            nz = normalized[normalized > 0]
            entropy = float(-(nz * np.log(nz)).sum() / np.log(len(normalized))) if len(normalized) > 1 and len(nz) else None
            concentration = float(normalized[order[:top_count]].sum()) if len(order) else None
            profile_by_condition[summary_row['condition_id']][int(seed)] = energies
            fourier_rows.append({'condition_id': summary_row['condition_id'], 'factor': summary_row['factor'],
                                 'factor_value': summary_row['factor_value'], 'model_seed': int(seed),
                                 'dominant_frequencies': json.dumps(dominant), 'top_5_energy_fraction': concentration,
                                 'normalized_entropy': entropy, 'fourier_similarity_to_baseline': None})
    baseline_profiles = profile_by_condition.get(baseline_id, {}) if baseline_id else {}
    for item in fourier_rows:
        reference = baseline_profiles.get(item['model_seed'])
        current = profile_by_condition.get(item['condition_id'], {}).get(item['model_seed'])
        if reference is not None and current is not None and len(reference) == len(current):
            denominator = np.linalg.norm(reference) * np.linalg.norm(current)
            item['fourier_similarity_to_baseline'] = float(np.dot(reference, current) / denominator) if denominator else None
    _csv(aggregate_dir / 'condition_fourier_summary.csv', fourier_rows)
    result = {'status': 'aggregated', 'runs': len(records), 'conditions': len(summaries),
              'baseline_available': baseline is not None,
              'note': 'Descriptive summaries only; grokking steps summarize observed grokked seeds, while right-censored counts remain separate.'}
    json_save(result, aggregate_dir / 'paper_summary.json')
    return result


def run_phase4(config, preset='core', factor_value=None, seeds=None, dry_run=False,
               confirm_large_run=False, force=False, max_steps=None, run_analysis=True,
               resume_checkpoint=None):
    if max_steps is not None:
        config['training']['max_steps'] = max_steps
    validate(config)
    rows, selected_seeds, estimate = plan(config, preset, factor_value, seeds)
    if resume_checkpoint is not None:
        if force:
            raise ValueError('--resume and --force cannot be combined.')
        if len(rows) != 1 or len(selected_seeds) != 1:
            raise ValueError('An explicit Phase 4 --resume requires exactly one condition and one model seed.')
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f'Phase 4 resume checkpoint not found: {resume_checkpoint}')
    if dry_run:
        return {'status': 'dry_run', **estimate}
    tier = int(config['phase4'].get('analysis_tier', 1))
    if tier not in (0, 1, 2, 3):
        raise ValueError('phase4.analysis_tier must be 0, 1, 2, or 3.')
    threshold = int(config['phase4'].get('max_runs_without_confirmation', 10))
    if estimate['training_runs'] > threshold and not confirm_large_run:
        raise ValueError(f"This plan requests {estimate['training_runs']} training runs, above the safety limit {threshold}. Review the dry-run and pass --confirm-large-run to start it.")
    root = Path(config['experiment']['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    failures = root / 'aggregate' / 'failed_runs.csv'
    failure_rows = []
    for row in rows:
        factor_dir = root / row['factor'] / 'conditions' / row['condition_id']
        result_root = factor_dir / 'runs'
        checkpoint_root = Path(config['experiment']['checkpoint_root']) / row['factor'] / 'conditions' / row['condition_id']
        metadata_path = factor_dir / 'condition.json'
        proposed = {k: row[k] for k in ('condition_id', 'baseline_condition_id', 'baseline_config_hash', 'config_hash', 'factor', 'factor_value')}
        proposed['config'] = row['config']
        if metadata_path.exists():
            previous = _read_json(metadata_path)
            if previous.get('config_hash') != row['config_hash']:
                raise ValueError(f'Condition metadata mismatch at {metadata_path}; refusing to mix runs.')
        else:
            json_save(proposed, metadata_path)
        for seed in selected_seeds:
            result_dir = result_root / f'seed_{seed}'; checkpoint_dir = checkpoint_root / f'seed_{seed}'
            summary_path = result_dir / 'summary.json'; latest = _find_resume(checkpoint_dir)
            if resume_checkpoint is not None:
                latest = resume_checkpoint
            existing = _read_json(summary_path)
            if force:
                shutil.rmtree(result_dir, ignore_errors=True); shutil.rmtree(checkpoint_dir, ignore_errors=True)
                shutil.rmtree(factor_dir / 'analysis', ignore_errors=True)
                existing = None; latest = None
            elif existing and existing.get('status') == 'completed':
                print(f'Skipping completed {row["condition_id"]} seed {seed}.', flush=True)
                continue
            try:
                run_config = json.loads(json.dumps(row['config']))
                run_config['experiment']['output_root'] = str(result_root)
                run_config['experiment']['checkpoint_root'] = str(checkpoint_root)
                run_config['experiment']['name'] = f'phase4_{row["condition_id"]}'
                validate(run_config)
                print(f'PHASE 4 | condition={row["condition_id"]} | factor={row["factor"]}={row["factor_value"]} | model_seed={seed} | device={device()} | sequential', flush=True)
                trainer = Trainer(run_config, seed, result_dir, checkpoint_dir)
                summary = trainer.run(resume=latest)
                summary.update({'condition_id': row['condition_id'], 'baseline_condition_id': row['baseline_condition_id'],
                                'factor': row['factor'], 'factor_value': row['factor_value'], 'baseline_config_hash': row['baseline_config_hash']})
                json_save(summary, summary_path)
                cleanup()
            except Exception as exc:
                failure_rows.append({'condition_id': row['condition_id'], 'factor': row['factor'], 'factor_value': row['factor_value'],
                                     'model_seed': seed, 'error_type': type(exc).__name__, 'error': str(exc),
                                     'traceback': traceback.format_exc()})
                _csv(failures, failure_rows)
                print(f'FAILED {row["condition_id"]} seed {seed}: {type(exc).__name__}: {exc}', flush=True)
                cleanup()
    if run_analysis and tier >= 1:
        tier2_ids = set(config['phase4'].get('tier2_condition_ids', []))
        tier3_ids = set(config['phase4'].get('tier3_condition_ids', []))
        for row in rows:
            factor_dir = root / row['factor'] / 'conditions' / row['condition_id']
            done = [_read_json(factor_dir / 'runs' / f'seed_{seed}' / 'summary.json') for seed in selected_seeds]
            if len(selected_seeds) > 1 and all(item and item.get('status') == 'completed' for item in done):
                checkpoint_root = Path(config['experiment']['checkpoint_root']) / row['factor'] / 'conditions' / row['condition_id']
                selected_tiers = [1]
                if tier >= 2 and row['condition_id'] in tier2_ids: selected_tiers.append(2)
                if tier >= 3 and row['condition_id'] in tier3_ids: selected_tiers.append(3)
                for selected_tier in selected_tiers:
                    try:
                        _run_phase3_tier(row['config'], factor_dir, checkpoint_root, selected_seeds, selected_tier)
                    except Exception as exc:
                        failure_rows.append({'condition_id': row['condition_id'], 'factor': row['factor'], 'factor_value': row['factor_value'],
                                             'model_seed': '', 'error_type': type(exc).__name__, 'error': f'Tier {selected_tier} analysis: {exc}',
                                             'traceback': traceback.format_exc()})
                        _csv(failures, failure_rows)
    result = aggregate(root)
    result['failed_runs'] = len(failure_rows)
    return result
