"""Checkpoint-only Fourier and mechanistic analyses for Phase 3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from analysis.ablation import patching, run_ablations
from analysis.activation_extractor import extract
from analysis.attention_analysis import head_signatures, signature_matrix
from analysis.checkpoint_dynamics import STAGES, checkpoint_inventory, stage_status
from analysis.cross_seed_similarity import cka_layer_matrix, match_heads
from analysis.fourier import concentration, dominant_frequencies, frequency_energy
from analysis.frequency_analysis import energy_profile, frequency_rows, joint_energy_profile, jaccard, profile_similarity
from analysis.plots import heatmap, plt, save
from analysis.representation_analysis import linear_cka
from data.modular_arithmetic import array_hash, make_dataset
from models.tiny_transformer import TinyTransformer
from training.config import config_hash
from training.metrics import evaluate
from training.utils import cleanup, device, environment, json_save, parameter_hash

ANALYSIS_VERSION = 1


def _write_csv(rows, path, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _save_profile_figure(seed, rows, out):
    selected = [row for row in rows if row['seed'] == seed and row['representation'] in ('embedding', 'output')]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for representation in ('embedding', 'output'):
        for stage in STAGES:
            values = [r for r in selected if r['representation'] == representation and r['stage'] == stage]
            if values:
                ax.plot([r['frequency'] for r in values], [r['normalized_energy'] for r in values], label=f'{representation}: {stage}')
    ax.set(xlabel='Fourier frequency', ylabel='Normalized energy', title=f'Seed {seed}: number embeddings and output vectors')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
    save(fig, out / f'fourier_seed_{seed}.png')


def _load_model_checkpoint(path: Path, config: dict, seed: int, expected_split_hash: str, target_device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = checkpoint.get('config') or {}
    if saved_config and saved_config.get('data', {}).get('prime', config['data']['prime']) != config['data']['prime']:
        raise ValueError(f'Prime mismatch in checkpoint: {path}')
    if saved_config and saved_config.get('model', config['model']) != config['model']:
        raise ValueError(f'Model architecture mismatch in checkpoint: {path}')
    if checkpoint.get('model_seed', checkpoint.get('seed', seed)) != seed:
        raise ValueError(f'Model seed mismatch in checkpoint: {path}')
    if checkpoint.get('dataset_split_hash') not in (None, expected_split_hash):
        raise ValueError(f'Dataset split hash mismatch in checkpoint: {path}')
    model = TinyTransformer(config['data']['prime'], config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(target_device).eval()
    return model, checkpoint, parameter_hash(model)


def _cached_representations(model, inputs, keys, target_device, batch_size, cache_path, meta_path, metadata):
    if cache_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()) == metadata:
                with np.load(cache_path, allow_pickle=False) as archive:
                    return {key: archive[key] for key in keys}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    values = extract(model, inputs, keys, target_device, batch_size=batch_size, last_position=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **values)
    json_save(metadata, meta_path)
    return values


def _plot_dynamics(fourier_rows, output):
    stages = list(STAGES)
    representations = ['embedding', 'output', 'final_residual']
    fig, axes = plt.subplots(1, len(representations), figsize=(14, 4))
    for ax, representation in zip(axes, representations):
        for stage in stages:
            rows = [r for r in fourier_rows if r['representation'] == representation and r['stage'] == stage]
            if not rows:
                continue
            grouped = pd.DataFrame(rows).groupby('frequency').normalized_energy.mean()
            ax.plot(grouped.index, grouped.values, label=stage)
        ax.set(title=representation.replace('_', ' '), xlabel='Frequency', ylabel='Mean normalized energy')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(output / 'fourier_training_dynamics.png', dpi=220, bbox_inches='tight')
    fig.savefig(output / 'fourier_energy_by_stage.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def _write_similarity_outputs(profile_vectors, reps_by_seed_stage, head_rows, ablation_by_seed, summary_by_seed, out, top_ks, minimum_head_match_similarity=0.25, include_head_matching=True):
    similarity_dir = out / 'similarity'; figure_dir = out / 'figures'; table_dir = out / 'tables'
    similarity_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(profile_vectors)
    kinds = sorted({kind for values in profile_vectors.values() for kind in values})
    profile_rows = []
    profile_matrices = {}
    for kind in kinds:
        available = {s: profile_vectors[s][kind] for s in seeds if kind in profile_vectors[s]}
        pair_rows = []
        ss = sorted(available)
        matrices = {name: np.full((len(ss), len(ss)), np.nan) for name in ('cosine', 'pearson', 'spearman')}
        for i, a in enumerate(ss):
            for j, b in enumerate(ss):
                sims = profile_similarity(available[a], available[b])
                for name, value in sims.items(): matrices[name][i, j] = np.nan if value is None else value
                if i < j:
                    for k in top_ks:
                        da = dominant_frequencies(available[a], k); db = dominant_frequencies(available[b], k)
                        pair_rows.append({'seed_a': a, 'seed_b': b, 'representation': kind, 'top_k': k, 'jaccard': jaccard(da, db)})
        for name, matrix in matrices.items():
            profile_matrices[(kind, name)] = (ss, matrix)
            pd.DataFrame(matrix, index=ss, columns=ss).to_csv(similarity_dir / f'fourier_{name}_similarity{("" if kind == "embedding" else "_"+kind)}.csv', index_label='seed')
        if kind == 'embedding':
            for i, a in enumerate(ss):
                for j, b in enumerate(ss):
                    if i < j:
                        sims = profile_similarity(available[a], available[b])
                        profile_rows.append({'seed_a': a, 'seed_b': b, 'representation': kind, **sims})
            _write_csv(pair_rows, similarity_dir / 'frequency_jaccard.csv')
            if ss:
                for k in top_ks:
                    jmat = np.eye(len(ss))
                    for i, a in enumerate(ss):
                        for j in range(i+1, len(ss)):
                            b = ss[j]
                            jmat[i,j] = jmat[j,i] = jaccard(dominant_frequencies(available[a],k), dominant_frequencies(available[b],k))
                    pd.DataFrame(jmat, index=ss, columns=ss).to_csv(similarity_dir / f'frequency_jaccard_top_{k}.csv', index_label='seed')
                    if k == top_ks[0]:
                        pd.DataFrame(jmat, index=ss, columns=ss).to_csv(similarity_dir / 'frequency_jaccard_matrix.csv', index_label='seed')
            if 'cosine' in matrices:
                _write_csv([{'seed': s, **{str(t): float(matrices['cosine'][i, j]) for j, t in enumerate(ss)}} for i, s in enumerate(ss)], similarity_dir / 'fourier_cosine_similarity_matrix.csv')
    _write_csv(profile_rows, table_dir / 'fourier_similarity.csv')

    final_reps = {s: reps_by_seed_stage.get((s, 'final'), {}) for s in seeds}
    resid_keys = sorted(set.intersection(*(set(r) for r in final_reps.values()))) if final_reps and all(final_reps.values()) else []
    cka_rows = []; final_resid = {}; first_cross_layer = None
    for stage_name in ('initial', 'final'):
        for a_i, seed_a in enumerate(seeds):
            ra = reps_by_seed_stage.get((seed_a, stage_name), {})
            if stage_name == 'final' and 'final_residual' in ra: final_resid[seed_a] = ra['final_residual']
            for seed_b in seeds[a_i + 1:]:
                rb = reps_by_seed_stage.get((seed_b, stage_name), {})
                if ra and rb:
                    for row in cka_layer_matrix(ra, rb):
                        cka_rows.append({'seed_a': seed_a, 'seed_b': seed_b, 'checkpoint': stage_name, **row})
                        if stage_name == 'final' and first_cross_layer is None:
                            first_cross_layer = (seed_a, seed_b, row)
    _write_csv(cka_rows, table_dir / 'cka_similarity.csv')
    if first_cross_layer is not None:
        seed_a, seed_b, _ = first_cross_layer
        selected = [r for r in cka_rows if r['seed_a']==seed_a and r['seed_b']==seed_b and r['checkpoint']=='final']
        layer_a = sorted({r['layer_a'] for r in selected}); layer_b = sorted({r['layer_b'] for r in selected})
        matrix = np.array([[np.nan if (value := next((r['linear_cka'] for r in selected if r['layer_a']==a and r['layer_b']==b), None)) is None else float(value)
                           for b in layer_b] for a in layer_a], dtype=np.float64)
        heatmap(matrix, figure_dir/f'cross_layer_cka_seed_{seed_a}_seed_{seed_b}.png',
                f'Cross-layer residual CKA: seed {seed_a} vs {seed_b}', [k.replace('L','') for k in layer_b])
    cka_matrix = np.full((len(seeds), len(seeds)), np.nan)
    initial_cka_matrix = np.full((len(seeds), len(seeds)), np.nan)
    for i, seed_a in enumerate(seeds):
        if seed_a in final_resid: cka_matrix[i, i] = linear_cka(final_resid[seed_a], final_resid[seed_a])
        initial_a = reps_by_seed_stage.get((seed_a, 'initial'), {}).get('final_residual')
        if initial_a is not None: initial_cka_matrix[i, i] = linear_cka(initial_a, initial_a)
        for j in range(i + 1, len(seeds)):
            seed_b = seeds[j]
            if seed_a in final_resid and seed_b in final_resid:
                cka_matrix[i, j] = cka_matrix[j, i] = linear_cka(final_resid[seed_a], final_resid[seed_b])
            initial_b = reps_by_seed_stage.get((seed_b, 'initial'), {}).get('final_residual')
            if initial_a is not None and initial_b is not None:
                initial_cka_matrix[i, j] = initial_cka_matrix[j, i] = linear_cka(initial_a, initial_b)
    if len(seeds):
        pd.DataFrame(cka_matrix, index=seeds, columns=seeds).to_csv(similarity_dir / 'cka_similarity_matrix.csv', index_label='seed')
        heatmap(cka_matrix, figure_dir / 'cka_similarity_heatmap.png', 'Final residual linear CKA', [str(s) for s in seeds])
        if np.isfinite(initial_cka_matrix).any():
            pd.DataFrame(initial_cka_matrix, index=seeds, columns=seeds).to_csv(similarity_dir / 'cka_initial_similarity_matrix.csv', index_label='seed')
            heatmap(initial_cka_matrix, figure_dir / 'cka_initial_control_heatmap.png', 'Initial residual CKA control', [str(s) for s in seeds])

    matching_rows = []
    seed_head_rows = {s: [r for r in head_rows if r['seed'] == s and r['checkpoint'] == 'final'] for s in seeds}
    head_features = {s: signature_matrix(seed_head_rows[s], s) for s in seeds}
    head_indices = {s: [(r['layer'], r['head']) for r in sorted(seed_head_rows[s], key=lambda x: (x['layer'], x['head']))] for s in seeds}
    pair_head_scores = {}
    if include_head_matching:
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                matches = match_heads(head_features.get(seed_a, np.empty((0, 0))), head_features.get(seed_b, np.empty((0, 0))), minimum_head_match_similarity)
                for match in matches:
                    aidx = head_indices[seed_a][match['head_index_a']]; bidx = head_indices[seed_b][match['head_index_b']]
                    row = {'seed_a': seed_a, 'layer_a': aidx[0], 'head_a': aidx[1], 'seed_b': seed_b,
                           'layer_b': bidx[0], 'head_b': bidx[1], **match}
                    matching_rows.append(row)
                score = float(np.mean([m['functional_similarity'] for m in matches])) if matches else np.nan
                pair_head_scores[(seed_a, seed_b)] = score
                if matches:
                    _write_csv([{'layer_a': r['layer_a'], 'head_a': r['head_a'], 'layer_b': r['layer_b'],
                                 'head_b': r['head_b'], 'functional_similarity': r['functional_similarity'],
                                 'accepted_match': r['accepted_match']} for r in matching_rows if r['seed_a'] == seed_a and r['seed_b'] == seed_b],
                               similarity_dir / f'head_matching_seed_{seed_a}_seed_{seed_b}.csv')
    _write_csv(matching_rows, table_dir / 'head_matching.csv')
    if matching_rows:
        fig, ax = plt.subplots(figsize=(8, 4));
        ax.bar(np.arange(len(matching_rows)), [r['functional_similarity'] for r in matching_rows])
        ax.axhline(minimum_head_match_similarity, linestyle='--', label='minimum accepted match')
        ax.set(xlabel='Matched head pair', ylabel='Functional descriptor cosine', title='Hungarian matching; head indices are not identities')
        ax.legend(); save(fig, figure_dir / 'head_matching.png')

    circuit_rows = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            f_sims = profile_similarity(profile_vectors[a]['embedding'], profile_vectors[b]['embedding']) if 'embedding' in profile_vectors[a] and 'embedding' in profile_vectors[b] else {}
            cka = linear_cka(final_resid[a], final_resid[b]) if a in final_resid and b in final_resid else None
            beh_a = summary_by_seed.get(a, {}).get('final_test_accuracy'); beh_b = summary_by_seed.get(b, {}).get('final_test_accuracy')
            aa = ablation_by_seed.get(a, []); ab = ablation_by_seed.get(b, [])
            heads_a = np.sort([abs(r['delta_logit_margin']) for r in aa if r['kind'] == 'head'])
            heads_b = np.sort([abs(r['delta_logit_margin']) for r in ab if r['kind'] == 'head'])
            causal_similarity = float(np.dot(heads_a, heads_b) / (np.linalg.norm(heads_a) * np.linalg.norm(heads_b))) if len(heads_a) == len(heads_b) and len(heads_a) and np.linalg.norm(heads_a)*np.linalg.norm(heads_b)>0 else None
            circuit_rows.append({'seed_a': a, 'seed_b': b,
                'behavioral_test_accuracy_absolute_difference': abs(beh_a-beh_b) if beh_a is not None and beh_b is not None else None,
                'fourier_embedding_cosine': f_sims.get('cosine'), 'final_residual_cka': cka,
                'functional_head_match_mean': pair_head_scores.get((a,b)), 'causal_head_effect_distribution_cosine': causal_similarity})
    _write_csv(circuit_rows, table_dir / 'circuit_similarity.csv')
    _write_csv(head_rows, table_dir / 'head_signatures.csv')

    if profile_matrices.get(('embedding', 'cosine')):
        ss, matrix = profile_matrices[('embedding', 'cosine')]
        heatmap(matrix, figure_dir / 'fourier_similarity_heatmap.png', 'Embedding Fourier profile cosine similarity', [str(s) for s in ss])
    if profile_rows:
        sims = pd.DataFrame(profile_rows)
        matrix = np.eye(len(seeds))
        index = {s: i for i, s in enumerate(seeds)}
        for row in profile_rows:
            if row['seed_a'] in index and row['seed_b'] in index and row.get('cosine') is not None:
                matrix[index[row['seed_a']], index[row['seed_b']]] = matrix[index[row['seed_b']], index[row['seed_a']]] = row['cosine']
        heatmap(matrix, figure_dir / 'fourier_similarity_heatmap.png', 'Embedding Fourier profile cosine similarity', [str(s) for s in seeds])

    if circuit_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(circuit_rows)); width = .36
        ax.bar(x-width/2, [row['fourier_embedding_cosine'] or 0 for row in circuit_rows], width, label='Fourier profile cosine')
        ax.bar(x+width/2, [row['final_residual_cka'] or 0 for row in circuit_rows], width, label='Residual CKA')
        ax.set(xlabel='Seed pair', ylabel='Similarity', title='Behavior and mechanism metrics remain separate')
        ax.set_xticks(x, [f"{r['seed_a']}–{r['seed_b']}" for r in circuit_rows]); ax.legend(); save(fig, figure_dir / 'behavior_vs_mechanism.png')


def run_phase3(config: dict, analysis: str = 'all', seeds: list[int] | None = None, checkpoint: str = 'all') -> dict:
    """Analyze existing checkpoints only; this function never trains or edits them."""
    if analysis not in {'all', 'fourier', 'activations', 'ablation', 'cross-seed', 'robustness'}:
        raise ValueError(f'Unknown Phase-3 analysis mode: {analysis}')
    p3 = config['phase3']; seeds = list(seeds if seeds is not None else p3['seeds'])
    if checkpoint != 'all':
        if checkpoint not in STAGES: raise ValueError(f'Unknown checkpoint stage: {checkpoint}')
        stages = [checkpoint]
    else:
        stages = [stage for stage in p3['checkpoints'] if stage in STAGES]
    checkpoint_root = Path(p3['checkpoint_root']); source_results = Path(p3['source_results_root'])
    output = Path(p3['output_root']); figure_dir = output / 'figures'; table_dir = output / 'tables'
    fourier_dir = output / 'fourier'; cache_dir = output / 'cache'
    for path in (figure_dir, table_dir, fourier_dir, cache_dir, output/'activations', output/'circuits', output/'similarity'):
        path.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config['data']); target_device = device(); inventory = checkpoint_inventory(checkpoint_root, seeds)
    _write_csv(inventory, table_dir / 'checkpoint_inventory.csv')
    manifest = {
        'analysis_version': ANALYSIS_VERSION, 'analysis_mode': analysis, 'created_at_epoch_seconds': time.time(),
        'environment': environment(), 'device': str(target_device), 'source_checkpoint_root': str(checkpoint_root.resolve()),
        'source_results_root': str(source_results.resolve()), 'dataset_split_hash': dataset.split_hash,
        'train_split_hash': dataset.train_hash, 'test_split_hash': dataset.test_hash,
        'config_hash': config_hash(config), 'seeds_requested': seeds, 'stages_requested': stages,
        'checkpoint_files_are_read_only': True,
    }
    json_save(manifest, output/'analysis_manifest.json')
    available = [row for row in inventory if row['available'] and row['stage'] in stages]
    if not available:
        result = {'status': 'no_checkpoints', 'seeds_requested': seeds,
                  'checkpoint_root': str(checkpoint_root.resolve()),
                  'message': 'No requested Phase-2 checkpoints were found. No models were trained.'}
        json_save(result, output/'analysis_summary.json')
        print(result['message'] + f" Expected files under {checkpoint_root}/seed_<seed>/.", flush=True)
        return result

    full_inputs = dataset.inputs
    rng = np.random.default_rng(config['data']['data_seed'] + 773)
    n_repr = min(p3['representation_examples'], len(full_inputs))
    rep_indices = np.sort(rng.choice(len(full_inputs), size=n_repr, replace=False))
    rep_inputs = full_inputs[rep_indices]
    rep_hash = array_hash(rep_indices)
    test_inputs, test_targets = dataset.subset('test')
    max_ablation = p3['ablation_examples']
    if max_ablation is not None and len(test_inputs) > max_ablation:
        idx = np.sort(np.random.default_rng(config['data']['data_seed'] + 991).choice(len(test_inputs), max_ablation, replace=False))
        ablation_inputs, ablation_targets = test_inputs[idx], test_targets[idx]
    else:
        ablation_inputs, ablation_targets = test_inputs, test_targets
    if len(test_inputs) > p3['patching_examples']:
        idx = np.sort(np.random.default_rng(config['data']['data_seed'] + 1991).choice(len(test_inputs), p3['patching_examples'], replace=False))
        patch_inputs, patch_targets = test_inputs[idx], test_targets[idx]
    else:
        patch_inputs, patch_targets = test_inputs, test_targets

    fourier_rows = []; summary_by_seed = {}; profiles = {}; head_rows = []; ablation_by_seed = {}
    reps_by_seed_stage = {}; dynamics_rows = []; stage_reps = {}
    modes = {'fourier'} if analysis == 'fourier' else {'activations'} if analysis == 'activations' else {'ablation'} if analysis == 'ablation' else {'fourier','activations','cross-seed','ablation'}
    if analysis == 'cross-seed': modes = {'fourier','activations','cross-seed'}
    if analysis == 'robustness': modes = {'fourier','activations'}
    layer_count = config['model']['n_layers']
    fourier_keys = ['final_residual']
    for layer in range(layer_count):
        fourier_keys.extend([f'L{layer}.resid_pre', f'L{layer}.resid_mid', f'L{layer}.resid_post',
            f'L{layer}.attn_out', f'L{layer}.mlp_hidden', f'L{layer}.heads',
            f'L{layer}.q', f'L{layer}.k', f'L{layer}.v'])
    head_keys = [key for layer in range(layer_count) for key in (f'L{layer}.pattern', f'L{layer}.heads', f'L{layer}.q', f'L{layer}.k', f'L{layer}.v')]
    cka_keys = [key for layer in range(layer_count) for key in (f'L{layer}.resid_pre', f'L{layer}.resid_mid', f'L{layer}.resid_post')] + ['final_residual']

    for seed in seeds:
        profiles[seed] = {}; result_summary = source_results / f'seed_{seed}' / 'summary.json'
        if result_summary.exists(): summary_by_seed[seed] = json.loads(result_summary.read_text())
        for stage in stages:
            path = checkpoint_root / f'seed_{seed}' / f'{stage}.pt'
            stage_row = next(row for row in inventory if row['seed'] == seed and row['stage'] == stage)
            status = stage_status(stage, stage_row, summary_by_seed.get(seed))
            dynamics_rows.append({'seed': seed, 'stage': stage, 'status': status,
                'checkpoint_path': str(path), 'grokking_detected_in_source_summary': summary_by_seed.get(seed, {}).get('grokking_detected')})
            if not path.is_file():
                continue
            model, checkpoint_payload, model_hash = _load_model_checkpoint(path, config, seed, dataset.split_hash, target_device)
            prefix = f'seed_{seed}_{stage}'
            if 'fourier' in modes:
                embedding = model.embedding.weight[:config['data']['prime']].detach().cpu().numpy()
                # One row per output class, so the Fourier axis is the class in Z_p.
                output_vectors = model.unembedding.weight.detach().cpu().numpy()
                direct = {'embedding': energy_profile(embedding, config['data']['prime']),
                          'output': energy_profile(output_vectors, config['data']['prime'])}
                for name, profile in direct.items():
                    fourier_rows.extend(frequency_rows(profile, seed, stage, name, max(p3['top_k'])))
                    if stage == 'final': profiles[seed][name] = profile
                activations = extract(model, full_inputs, fourier_keys, target_device, p3['batch_size'], last_position=True)
                raw = {}
                for key, value in activations.items():
                    joint, marginal = joint_energy_profile(value, config['data']['prime'])
                    name = key.replace('.', '_')
                    raw[name+'_joint'] = joint; raw[name+'_marginal'] = marginal
                    fourier_rows.extend(frequency_rows(marginal, seed, stage, name, max(p3['top_k'])))
                    if key == 'final_residual' and stage == 'final': profiles[seed]['final_residual'] = marginal
                np.savez_compressed(fourier_dir / f'{prefix}_fourier.npz', **raw, **{k: v for k,v in direct.items()})
                del activations, raw
            if 'activations' in modes or 'cross-seed' in modes or 'ablation' in modes:
                sign_cache = extract(model, rep_inputs, head_keys, target_device, p3['batch_size'], last_position=True)
                current_head_rows = head_signatures(sign_cache, seed, stage)
                head_rows.extend(current_head_rows)
                if stage in ('initial', 'final') and ('cross-seed' in modes or 'activations' in modes):
                    cka_input_hash = rep_hash
                    meta = {'analysis_version': ANALYSIS_VERSION, 'seed': seed, 'stage': stage,
                            'dataset_split_hash': dataset.split_hash, 'representation_example_hash': cka_input_hash,
                            'model_hash': model_hash, 'keys': cka_keys, 'config_hash': config_hash(config)}
                    cache_path = cache_dir / f'representations_seed_{seed}_{stage}.npz'
                    meta_path = cache_dir / f'representations_seed_{seed}_{stage}.json'
                    reps = _cached_representations(model, rep_inputs, cka_keys, target_device, p3['batch_size'], cache_path, meta_path, meta)
                    reps_by_seed_stage[(seed, stage)] = reps
                if 'activations' in modes and stage == 'final':
                    np.savez_compressed(cache_dir / f'head_activations_seed_{seed}_{stage}.npz', **sign_cache)
                del sign_cache
            if 'ablation' in modes and stage == 'final':
                baseline, rows = run_ablations(model, ablation_inputs, ablation_targets, target_device, p3['batch_size'])
                for row in rows:
                    row['seed'] = seed; row['checkpoint'] = stage
                    row['candidate_by_ablation_only'] = False
                ranked = sorted(rows, key=lambda row: row['delta_logit_margin'], reverse=True)
                n_candidates = min(2, len(ranked))
                for row in ranked[:n_candidates]: row['candidate_by_ablation_only'] = True
                ablation_by_seed[seed] = rows
                head_rows_by_id = {(r['layer'], r['head']): r for r in rows if r['kind'] == 'head'}
                # Add causal effects to the descriptive head signatures, without assigning semantic labels.
                for hrow in head_rows:
                    if hrow['seed'] == seed and hrow['checkpoint'] == stage and (hrow['layer'], hrow['head']) in head_rows_by_id:
                        hrow.update({f"ablation_{k}": v for k,v in head_rows_by_id[(hrow['layer'],hrow['head'])].items() if k.startswith('delta_')})
                head_ablation_path = table_dir / 'head_ablation.csv'
                prior_heads = pd.read_csv(head_ablation_path).to_dict('records') if head_ablation_path.exists() else []
                _write_csv(prior_heads + [r for r in rows if r['kind']=='head'], head_ablation_path)
                mlp_path = table_dir / 'mlp_ablation.csv'
                prior_mlp = pd.read_csv(mlp_path).to_dict('records') if mlp_path.exists() else []
                _write_csv(prior_mlp + [r for r in rows if r['kind']=='mlp'], mlp_path)
                if p3.get('activation_patching', True):
                    patch_keys = ['embedding'] + [f'L{l}.{part}' for l in range(layer_count) for part in ('resid_mid','attn_out','mlp_out','resid_post')] + ['final_residual']
                    patch_rows = patching(model, patch_inputs, patch_targets, target_device, patch_keys)
                    for row in patch_rows: row.update({'seed':seed,'checkpoint':stage})
                    patch_path = table_dir / 'activation_patching.csv'
                    prior_patch = pd.read_csv(patch_path).to_dict('records') if patch_path.exists() else []
                    _write_csv(prior_patch + patch_rows, patch_path)
                    # One targeted interaction check: pair the highest-margin head
                    # candidate with the highest-margin MLP candidate, if both exist.
                    top_head = next((r for r in ranked if r['kind']=='head'), None)
                    top_mlp = next((r for r in ranked if r['kind']=='mlp'), None)
                    if top_head and top_mlp:
                        from analysis.ablation import zero, zero_head
                        interventions = {
                            f"L{top_head['layer']}.heads": zero_head(top_head['head']),
                            f"L{top_mlp['layer']}.mlp_out": zero,
                        }
                        pair_eval = evaluate(model, ablation_inputs, ablation_targets, target_device, p3['batch_size'], interventions)
                        pair_row = {'seed':seed, 'component_a':f"L{top_head['layer']}H{top_head['head']}",
                            'component_b':f"MLP{top_mlp['layer']}", 'accuracy':pair_eval['accuracy'],
                            'loss':pair_eval['loss'], 'correct_logit':pair_eval['correct_logit'], 'logit_margin':pair_eval['logit_margin'],
                            'delta_accuracy':baseline['accuracy']-pair_eval['accuracy'],
                            'delta_loss':baseline['loss']-pair_eval['loss'],
                            'delta_logit_margin':baseline['logit_margin']-pair_eval['logit_margin'],
                            'selection_rule':'top single-component mean logit-margin effect'}
                        pair_file=table_dir/'combinatorial_ablation.csv'
                        old=pd.read_csv(pair_file).to_dict('records') if pair_file.exists() else []
                        _write_csv(old+[pair_row],pair_file)
                # Descriptive diagram labels the measured components without
                # assigning an algorithmic interpretation.
                fig, ax = plt.subplots(figsize=(7, 5)); ax.axis('off')
                labels=['Number-token embeddings']
                labels += [f"Measured high-effect component: L{r['layer']}H{r['head']}" if r['kind']=='head' else f"Measured high-effect component: MLP layer {r['layer']}" for r in ranked[:n_candidates]]
                labels += ['Residual stream', 'Output projection', '(a+b) mod p task']
                y_positions=np.linspace(.9,.1,len(labels))
                for i,(label,ypos) in enumerate(zip(labels,y_positions)):
                    ax.text(.5,ypos,label,ha='center',va='center',bbox={'boxstyle':'round,pad=.4'})
                    if i<len(labels)-1: ax.annotate('',xy=(.5,y_positions[i+1]+.035),xytext=(.5,ypos-.035),arrowprops={'arrowstyle':'->'})
                ax.set_title(f'Seed {seed}: measured component map; functional interpretation pending')
                save(fig,figure_dir/f'circuit_map_seed_{seed}.png')
                fig, ax = plt.subplots(figsize=(8, 4)); ordered=ranked
                ax.bar([f"{r['kind']} L{r['layer']} H{r['head']}" if r['kind']=='head' else f"MLP L{r['layer']}" for r in ordered], [r['delta_logit_margin'] for r in ordered])
                ax.set(ylabel='Baseline margin − ablated margin', title=f'Seed {seed}: single-component causal ablation effects')
                ax.tick_params(axis='x',rotation=45); save(fig,figure_dir/f'component_importance_seed_{seed}.png')
                signature = {
                    'seed': seed, 'stage': stage, 'checkpoint_path': str(path), 'model_hash': model_hash,
                    'source_grokking_detected': summary_by_seed.get(seed, {}).get('grokking_detected'),
                    'candidate_components_by_single_ablation_delta_loss_only': [
                        {'kind':r['kind'],'layer':r['layer'],'head':r['head'],'delta_loss':r['delta_loss']} for r in ranked[:n_candidates]],
                    'interpretation_status': 'descriptive; ablation alone is not a complete circuit claim',
                }
                json_save(signature, output/'circuits'/f'circuit_signature_seed_{seed}.json')
            # Cross-stage activations are reduced to Fourier summaries and released immediately.
            del model
            cleanup()

    _write_csv(dynamics_rows, table_dir / 'checkpoint_dynamics.csv')
    if fourier_rows:
        _write_csv(fourier_rows, table_dir / 'fourier_profiles.csv')
        # Requested compact long-form table, one row per seed and frequency.
        summary_rows = []
        for seed in profiles:
            for freq in range(config['data']['prime']//2 + 1):
                summary_rows.append({'seed':seed,'frequency':freq,
                    'embedding_energy': profiles[seed]['embedding'][freq] if 'embedding' in profiles[seed] else None,
                    'residual_energy': profiles[seed]['final_residual'][freq] if 'final_residual' in profiles[seed] else None,
                    'output_energy': profiles[seed]['output'][freq] if 'output' in profiles[seed] else None})
        _write_csv(summary_rows, table_dir / 'seed_frequency_summary.csv')
        overlap_rows=[]
        for seed in profiles:
            for k in p3['top_k']:
                if 'embedding' in profiles[seed] and 'output' in profiles[seed]:
                    overlap_rows.append({'seed':seed,'top_k':k,
                        'embedding_output_dominant_frequency_jaccard':jaccard(
                            dominant_frequencies(profiles[seed]['embedding'],k),
                            dominant_frequencies(profiles[seed]['output'],k))})
        _write_csv(overlap_rows,table_dir/'embedding_output_frequency_overlap.csv')
        _plot_dynamics(fourier_rows, figure_dir)
        for seed in profiles: _save_profile_figure(seed, fourier_rows, figure_dir)
        final_frequency_seeds = [s for s in sorted(profiles) if 'embedding' in profiles[s]]
        if final_frequency_seeds:
            heat = np.vstack([profiles[s]['embedding'] for s in final_frequency_seeds])
            heatmap(heat, figure_dir/'seed_frequency_heatmap.png', 'Final embedding Fourier energy', [str(s) for s in final_frequency_seeds])
            fig, ax = plt.subplots(figsize=(8,4))
            for seed in final_frequency_seeds:
                ax.plot(np.arange(len(profiles[seed]['embedding'])),profiles[seed]['embedding'],label=f'seed {seed}',alpha=.75)
            ax.set(xlabel='Fourier frequency',ylabel='Normalized energy',title='Number-embedding spectra across seeds')
            ax.legend(fontsize=7); ax.grid(alpha=.2); save(fig,figure_dir/'fourier_spectra_across_seeds.png')
            # A layer profile figure directly answers where pairwise Fourier
            # energy is expressed in residual stream, attention, and MLP states.
            residual_rows=[r for r in fourier_rows if r['stage']=='final' and ('resid_' in r['representation'] or r['representation']=='final_residual')]
            if residual_rows:
                agg=pd.DataFrame(residual_rows).groupby('representation').normalized_energy.sum().sort_index()
                fig,ax=plt.subplots(figsize=(9,4)); ax.bar(agg.index,agg.values)
                ax.set(xlabel='Captured representation',ylabel='Normalized non-DC Fourier energy',title='Residual and layer output Fourier energy')
                ax.tick_params(axis='x',rotation=40); save(fig,figure_dir/'residual_fourier_by_layer.png')
    if head_rows:
        _write_csv(head_rows, table_dir / 'head_signatures.csv')
    random_control_rows=[]
    for seed, rows in ablation_by_seed.items():
        head_effects=[r for r in rows if r['kind']=='head']
        if len(head_effects)>1:
            rng=np.random.default_rng(config['data']['data_seed']+seed+5041)
            chosen=rng.choice(len(head_effects),size=min(2,len(head_effects)),replace=False)
            top=sorted(head_effects,key=lambda r:r['delta_logit_margin'],reverse=True)[:len(chosen)]
            for ix in chosen:
                row=head_effects[int(ix)]
                random_control_rows.append({'seed':seed,'layer':row['layer'],'head':row['head'],
                    'random_delta_logit_margin':row['delta_logit_margin'],
                    'top_component_mean_delta_logit_margin':float(np.mean([r['delta_logit_margin'] for r in top])),
                    'sampling_seed':config['data']['data_seed']+seed+5041,
                    'note':'randomly selected from measured head ablations; descriptive control'})
    if random_control_rows: _write_csv(random_control_rows,table_dir/'random_ablation_control.csv')
    if 'cross-seed' in modes or 'activations' in modes or analysis == 'all':
        _write_similarity_outputs(profiles, reps_by_seed_stage, head_rows, ablation_by_seed, summary_by_seed, output, p3['top_k'], p3['minimum_head_match_similarity'], include_head_matching=analysis != 'robustness')
    if not (table_dir/'head_ablation.csv').exists(): _write_csv([], table_dir/'head_ablation.csv')
    if not (table_dir/'mlp_ablation.csv').exists(): _write_csv([], table_dir/'mlp_ablation.csv')

    summary = {
        'status': 'completed', 'analysis_mode': analysis, 'seeds_requested': seeds,
        'seeds_with_any_checkpoint': sorted({row['seed'] for row in available}),
        'stages_requested': stages, 'stages_available': [row for row in dynamics_rows if row['status']=='available'],
        'no_grokking_checkpoint_is_not_interpreted_as_a_negative_causal_result': True,
        'dataset_split_hash': dataset.split_hash, 'runtime_seconds': time.time()-manifest['created_at_epoch_seconds'],
        'outputs': str(output.resolve()), 'device': str(target_device),
    }
    json_save(summary, output/'analysis_summary.json')
    print(f"Phase 3 analysis complete: {output.resolve()}", flush=True)
    return summary
