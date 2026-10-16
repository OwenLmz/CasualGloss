import ast
import hashlib
import json
from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_release import canonical_ast_digest


class PaperScopeTests(unittest.TestCase):
    def test_no_broad_archive_tree(self):
        self.assertFalse((ROOT / 'archive').exists())
        self.assertFalse((ROOT / 'tools').exists())

    def test_ablation_builder_does_not_generate_discarded_baseline(self):
        source = (ROOT / 'ablations/prepare_experiment_manifests.py').read_text(encoding='utf-8')
        self.assertNotIn('learned_baseline/', source)
        self.assertNotIn('v1-validation-actions', source)

    def test_recovery_builder_does_not_generate_screen_experiment(self):
        source = (ROOT / 'data_preparation/recovery/prepare_v2_3_manifests.py').read_text(encoding='utf-8')
        self.assertNotIn('train_actions_screen', source)

    def test_speech_launcher_requires_explicit_paper_inputs(self):
        source = (ROOT / 'speech/frontend/run_seamlessstreaming_eval.sh').read_text(encoding='utf-8')
        self.assertNotIn('pilot_100', source)
        for name in ('TASK_ROOT', 'PYTHON_ENV', 'INPUT_TSV', 'OUTPUT_DIR', 'END_INDEX'):
            self.assertIn('${' + name + ':?', source)
        self.assertIn('--source-segment-size 320', source)

    def test_no_mixed_historical_aggregate(self):
        self.assertFalse((ROOT / 'method/causalgloss/aggregate_v2_3_multiseed.py').exists())

    def test_key_algorithm_functions_unchanged(self):
        manifest = json.loads((ROOT / 'provenance/algorithm_function_manifest.json').read_text(encoding='utf-8'))
        for item in manifest['functions']:
            with self.subTest(function=item['function'], file=item['repository_path']):
                tree = ast.parse((ROOT / item['repository_path']).read_text(encoding='utf-8-sig'))
                fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == item['function'])
                value = canonical_ast_digest(fn)
                self.assertEqual(value, item['ast_sha256'])

    def test_no_unreported_clerc_lexical_model(self):
        data = json.loads((ROOT / 'provenance/artifact_manifest.json').read_text(encoding='utf-8'))
        self.assertNotIn('lexical_clerc', [g['name'] for g in data['selected_model_groups']])
        self.assertIn('lexical_speech', [g['name'] for g in data['selected_model_groups']])


if __name__ == '__main__':
    unittest.main()
