import tempfile
import unittest
from pathlib import Path

from experiments.conditions import conditions
from experiments.phase4 import aggregate, plan
from training.config import load_config


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/phase4.yaml')
        self.config['training']['max_steps'] = 2
        self.config['phase4']['model_seeds'] = [0, 1]

    def test_conditions_change_one_factor_and_have_stable_ids(self):
        first = conditions(self.config, 'core')
        second = conditions(self.config, 'core')
        self.assertEqual([row['condition_id'] for row in first], [row['condition_id'] for row in second])
        self.assertEqual(first[0]['factor'], 'baseline')
        for row in first[1:]:
            changes = []
            for section in ('data', 'model', 'training'):
                for key, value in self.config[section].items():
                    if row['config'][section].get(key) != value:
                        changes.append((section, key))
            self.assertEqual(changes, [__import__('experiments.conditions', fromlist=['FACTORS']).FACTORS[row['factor']]])

    def test_core_plan_deduplicates_baseline_values(self):
        rows, seeds, estimate = plan(self.config, 'core')
        self.assertEqual(rows[0]['factor'], 'baseline')
        self.assertEqual(len(seeds), 2)
        self.assertEqual(estimate['training_runs'], len(rows) * 2)
        self.assertEqual(len({row['condition_id'] for row in rows}), len(rows))

    def test_factor_value_selects_only_that_factor_value(self):
        rows = conditions(self.config, 'operation', 'subtraction')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['factor_value'], 'subtraction')
        baseline = conditions(self.config, 'operation', 'addition')
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]['factor'], 'baseline')

    def test_data_split_preset_uses_requested_output_factor(self):
        rows = conditions(self.config, 'data')
        self.assertEqual(rows[0]['factor'], 'baseline')
        self.assertTrue(all(row['factor'] == 'data_split' for row in rows[1:]))

    def test_empty_aggregate_does_not_fabricate_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            result = aggregate(Path(temp))
            self.assertEqual(result['status'], 'no_completed_runs')
            self.assertEqual(result['runs'], 0)


if __name__ == '__main__':
    unittest.main()
