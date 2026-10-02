"""Numerical and two-seed smoke checks for Phase 3, using temporary fixtures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis.ablation import zero_head
from analysis.activation_extractor import extract
from analysis.fourier import build_fourier_basis, fourier_transform_matrix, grid_frequency_energy
from analysis.representation_analysis import linear_cka
from data.modular_arithmetic import make_dataset
from experiments.phase3 import run_phase3
from models.tiny_transformer import TinyTransformer
from training.config import load_config
from training.utils import parameter_hash


def tiny_config(root: Path):
    config = load_config('configs/phase3.yaml')
    config['data'].update({'prime': 5, 'train_fraction': .4, 'data_seed': 42})
    config['model'].update({'d_model': 8, 'n_heads': 2, 'n_layers': 1, 'd_mlp': 16, 'dropout': 0.0})
    config['phase3'].update({
        'checkpoint_root': str(root/'checkpoints'),
        'source_results_root': str(root/'source_results'),
        'output_root': str(root/'results'),
        'seeds': [0, 1],
        'checkpoints': ['initial', 'memorization', 'grokking', 'final'],
        'batch_size': 8,
        'representation_examples': 8,
        'ablation_examples': 8,
        'patching_examples': 4,
        'top_k': [1, 2],
    })
    return config


class FourierTests(unittest.TestCase):
    def test_real_basis_is_orthonormal(self):
        basis = build_fourier_basis(113)
        self.assertEqual(basis.shape, (113, 113))
        np.testing.assert_allclose(basis.T @ basis, np.eye(113), atol=1e-10)

    def test_projection_and_pair_frequency_dimensions(self):
        x = np.random.default_rng(12).normal(size=(25, 3))
        self.assertEqual(fourier_transform_matrix(np.ones((5, 3))).shape, (5, 3))
        joint = grid_frequency_energy(x, 5)
        self.assertEqual(joint.shape, (3, 3))
        self.assertAlmostEqual(float(joint.sum()), 1.0)
        basis=build_fourier_basis(5); tensor=x.reshape(5,5,3)
        coeff=np.einsum('ak,abl,bj->kjl',basis,tensor,basis,optimize=True)
        grouped=np.zeros((3,3)); bins=[[0],[1,2],[3,4]]
        for i,ii in enumerate(bins):
            for j,jj in enumerate(bins): grouped[i,j]=(coeff[np.ix_(ii,jj)]**2).sum()
        grouped[0,0]=0; grouped/=grouped.sum()
        np.testing.assert_allclose(joint,grouped,atol=1e-12)


class RepresentationTests(unittest.TestCase):
    def test_linear_cka_identity_symmetry_and_rotation_invariance(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(80, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        y = x @ q
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=10)
        self.assertAlmostEqual(linear_cka(x, y), linear_cka(y, x), places=10)
        self.assertAlmostEqual(linear_cka(x, y), 1.0, places=10)

    def test_activation_extraction_shapes(self):
        cfg = {'d_model': 8, 'n_heads': 2, 'n_layers': 1, 'd_mlp': 16, 'dropout': 0.0}
        model = TinyTransformer(5, cfg)
        dataset = make_dataset({'prime':5,'operation':'addition','train_fraction':.4,'data_seed':42,'control':'none'})
        values = extract(model, dataset.inputs, ['L0.q','L0.k','L0.v','L0.pattern','L0.heads','L0.mlp_hidden'], torch.device('cpu'), batch_size=7)
        self.assertEqual(values['L0.q'].shape, (25, 8))
        self.assertEqual(values['L0.k'].shape, (25, 8))
        self.assertEqual(values['L0.v'].shape, (25, 8))
        self.assertEqual(values['L0.pattern'].shape, (25, 2, 4))
        self.assertEqual(values['L0.heads'].shape, (25, 8))
        self.assertEqual(values['L0.mlp_hidden'].shape, (25, 16))

    def test_head_ablation_only_zeros_selected_head(self):
        values = torch.randn(3, 2, 4, 5)
        ablated = zero_head(1)(values)
        self.assertTrue(torch.equal(ablated[:, 0], values[:, 0]))
        self.assertTrue(torch.equal(ablated[:, 1], torch.zeros_like(values[:, 1])))

    def test_dataset_is_deterministic_and_split_consistent(self):
        cfg={'prime':5,'operation':'addition','train_fraction':.4,'data_seed':42,'control':'none'}
        a=make_dataset(cfg); b=make_dataset(cfg)
        self.assertEqual(a.split_hash,b.split_hash)
        self.assertEqual(a.train_hash,b.train_hash)
        self.assertEqual(a.test_hash,b.test_hash)
        self.assertEqual(len(a.inputs),25)
        self.assertFalse(set(a.train_indices.tolist()) & set(a.test_indices.tolist()))


class CheckpointSmokeTests(unittest.TestCase):
    def test_two_seed_analysis_reads_checkpoints_and_writes_outputs(self):
        with tempfile.TemporaryDirectory(prefix='phase3-smoke-') as directory:
            root = Path(directory)
            cfg = tiny_config(root)
            ds = make_dataset(cfg['data'])
            for seed in (0, 1):
                torch.manual_seed(seed)
                model = TinyTransformer(5, cfg['model'])
                seed_dir = root/'checkpoints'/f'seed_{seed}'
                seed_dir.mkdir(parents=True)
                payload = {
                    'model_state_dict': model.state_dict(),
                    'model_seed': seed,
                    'dataset_split_hash': ds.split_hash,
                    'config': cfg,
                    'initial_model_hash': parameter_hash(model),
                }
                torch.save(payload, seed_dir/'initial.pt')
                torch.save(payload, seed_dir/'final.pt')
                summary_dir = root/'source_results'/f'seed_{seed}'
                summary_dir.mkdir(parents=True)
                (summary_dir/'summary.json').write_text(
                    '{"status":"completed","right_censored":true,"grokking_detected":false,'
                    '"total_training_steps":2,"final_test_accuracy":0.1}'
                )
            result = run_phase3(cfg, analysis='all', seeds=[0,1])
            self.assertEqual(result['status'], 'completed')
            out = root/'results'
            for name in ('fourier_profiles.csv','seed_frequency_summary.csv','head_signatures.csv',
                         'head_ablation.csv','mlp_ablation.csv','activation_patching.csv',
                         'cka_similarity.csv','head_matching.csv','circuit_similarity.csv',
                         'embedding_output_frequency_overlap.csv','combinatorial_ablation.csv'):
                self.assertTrue((out/'tables'/name).is_file(), name)
            self.assertTrue((out/'figures'/'fourier_training_dynamics.png').is_file())
            self.assertTrue((out/'figures'/'cka_similarity_heatmap.png').is_file())
            self.assertTrue((out/'circuits'/'circuit_signature_seed_0.json').is_file())
            dynamics = (out/'tables'/'checkpoint_dynamics.csv').read_text()
            self.assertIn('not_observed_within_training_horizon', dynamics)


if __name__ == '__main__':
    unittest.main()
