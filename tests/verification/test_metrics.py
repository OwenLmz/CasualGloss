"""Small synthetic fixtures, not paper data or evidence for model quality."""
import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[2] / "scripts/verification/verify_paper_results.py"
spec = importlib.util.spec_from_file_location("verify_paper_results", MODULE)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class MetricsTest(unittest.TestCase):
    def test_multiset_overlap(self):
        self.assertAlmostEqual(v.overlap(["A","A","B"],["A","B","B"]), 2/3)
        self.assertEqual(v.overlap([],[]),0)

    def test_laal_stops_at_first_full_source(self):
        self.assertAlmostEqual(v.laal([1,2,4,4],4,4),4/3,places=7)
        self.assertAlmostEqual(v.laal([1,2,4,4],4,8),11/6,places=7)
        self.assertIsNone(v.laal([],4,4))
        self.assertIsNone(v.laal([1],0,4))

    def test_content_normalization(self):
        self.assertEqual(v.normalize_content(["WATER_2?","QUESTION","water_2"]),["WATER"])
        self.assertEqual(v.normalize_content(["A","B","A","B"]),["A","B"])
        self.assertEqual(v.normalize_content(["A","A","B"]),["A","A","B"])

    def test_bleu_exact_and_short_no_smoothing(self):
        row={"target_pseudo_asl_gloss_tokens":["A","B","C","D"],
             "predicted_pseudo_asl_gloss_tokens":["A","B","C","D"]}
        self.assertEqual(v.corpus_bleu([row]),100)
        row["predicted_pseudo_asl_gloss_tokens"]=["A","B","C"]
        self.assertEqual(v.corpus_bleu([row]),0)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): v.index([{"id":"x"},{"id":"x"}],2)

    def test_sample_not_population_sd(self):
        result=v.aggregate([{"x":1},{"x":2},{"x":3}])
        self.assertEqual(result["x"]["mean"],2)
        self.assertEqual(result["x"]["sample_sd"],1)

if __name__=="__main__": unittest.main()
