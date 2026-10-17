import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


class PaperResultSourceTests(unittest.TestCase):
    def test_original_paper_results_are_separate_from_execution_reports(self):
        result = json.loads((ROOT / 'results/paper_results.json').read_text(encoding='utf-8'))
        self.assertEqual(result['result_kind'], 'original_paper_aggregates')
        self.assertEqual(set(result), {
            'result_kind', 'paper_sha256', 'source', 'locked500', 'clerc', 'ablations', 'speech',
        })
        self.assertNotIn('raw_token_and_id_pairing', result['locked500'])
        for family in ('ablations', 'speech'):
            self.assertNotIn('ci_note', result[family])

    def test_paper_display_values_and_anchor_are_preserved(self):
        result = json.loads((ROOT / 'results/paper_results.json').read_text(encoding='utf-8'))
        expected = json.loads((ROOT / 'results/expected_paper_results.json').read_text(encoding='utf-8'))
        self.assertEqual(result['paper_sha256'], expected['paper_sha256'])
        for system, metrics in expected['table1'].items():
            for metric, (mean, sd) in metrics.items():
                with self.subTest(system=system, metric=metric):
                    digits = 2 if metric == 'bleu4' else 4
                    actual = result['locked500']['table1'][system][metric]
                    self.assertEqual(round(actual['mean'], digits), mean)
                    self.assertEqual(round(actual['sample_sd'], digits), sd)

    def test_documentation_local_links_resolve(self):
        documents = [ROOT / 'README.md', ROOT / 'THIRD_PARTY.md']
        for folder in ('docs', 'requirements', 'figures', 'protocols', 'speech/corpus'):
            documents.extend((ROOT / folder).rglob('*.md'))
        for path in documents:
            text = path.read_text(encoding='utf-8')
            for target in re.findall(r'\]\(([^\s)]+)\)', text):
                target = unquote(target.split('#', 1)[0])
                if not target or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                    continue
                with self.subTest(document=path.relative_to(ROOT), target=target):
                    self.assertTrue((path.parent / target).exists())


if __name__ == '__main__':
    unittest.main()
