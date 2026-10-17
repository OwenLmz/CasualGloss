"""Small synthetic runner fixtures for complete fresh-suite materialization."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_checkpoint_materialize", ROOT / "scripts/reproduction/materialize_checkpoint_outputs.py")
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CheckpointMaterializeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper_checkpoint_materialize_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.roots = [self.base / "gpu0", self.base / "gpu1"]
        self.data = self.base / "bundle"
        self.output = self.base / "fresh_verifier_inputs"
        self.registry = self.base / "registry.json"
        self.entries = [{"id": "test_a", "family": "family_a", "expected_count": 2,
                         "verification_local_relative": "locked500_from_local/a.jsonl",
                         "historical_output_relative": "data/old/a.jsonl"},
                        {"id": "test_b", "family": "family_b", "expected_count": 2,
                         "verification_local_relative": "learned/b.jsonl",
                         "historical_output_relative": "data/old/b.jsonl"}]
        self.reference_alias = "full_v4_train/eval/manifests/epee_sources_401.jsonl"
        self.reference_relative = materializer.DATA_PREFIX + "/" + self.reference_alias
        self.indices, self.receipts, self.results = [], [], []
        for root, entry in zip(self.roots, self.entries):
            result = root / entry["id"] / "outputs/sentence_rollouts.jsonl"
            result.parent.mkdir(parents=True)
            payload = (json.dumps({"id": 1, "fresh": entry["id"]}) + "\n" + json.dumps({"id": 2}) + "\n").encode()
            result.write_bytes(payload)
            index = root / "checkpoint_output_index_synthetic.json"
            self.write(index, {"results_are_new_inference": True, "rollout_count": 1,
                               "outputs": [{k: entry[k] for k in ("id", "family", "verification_local_relative", "historical_output_relative")} | {"result": str(result)}]})
            receipt = result.parent.parent / "receipt.json"
            self.write(receipt, {"id": entry["id"], "status": "completed", "returncode": 0,
                                 "result_is_fresh_inference_not_archived_copy": True,
                                 "members": [{"id": entry["id"], "result": str(result), "status": "completed",
                                              "record_count": 2, "expected_count": 2, "sha256": sha(payload)}]})
            self.indices.append(index); self.receipts.append(receipt); self.results.append(result)
        reference = self.data / self.reference_relative
        reference.parent.mkdir(parents=True)
        payload = b'{"id":1,"reference":"one"}\n{"id":2,"reference":"two"}\n'
        reference.write_bytes(payload)
        self.write(self.registry, {"entries": [{"relative_to_remote_root": self.reference_relative, "type": "file",
                                                "size_bytes": len(payload), "sha256": sha(payload)}]})
        self.patchers = [patch.object(materializer, "EXPECTED_ROLLOUT_COUNT", 2),
                         patch.object(materializer, "EXPECTED_REFERENCE_COUNT", 2),
                         patch.object(materializer.matrix_api, "load_matrix", return_value={"experiments": self.entries}),
                         patch.object(materializer.targets_api, "target_paths", return_value=[
                             ("old/a.jsonl", self.entries[0]["verification_local_relative"]),
                             ("old/b.jsonl", self.entries[1]["verification_local_relative"]),
                             (self.reference_alias, self.reference_alias)])]
        for patcher in self.patchers:
            patcher.start(); self.addCleanup(patcher.stop)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def run_fixture(self, **kwargs):
        return materializer.materialize(kwargs.pop("roots", self.roots), self.data,
                                        kwargs.pop("output", self.output), registry_path=self.registry, **kwargs)

    def test_complete_suite_copies_bytes_no_links_or_forged_historical_hashes(self):
        before = [r.stat().st_nlink for r in self.results]
        receipt = self.run_fixture()
        self.assertEqual(receipt["input_files"], 3)
        self.assertEqual(receipt["fresh_rollout_files"], 2)
        self.assertFalse(receipt["historical_prediction_hashes_compared"])
        self.assertFalse(receipt["numeric_verification_performed"])
        for entry, source in zip(self.entries, self.results):
            copied = self.output / entry["verification_local_relative"]
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.stat().st_nlink, 1)
        self.assertEqual([r.stat().st_nlink for r in self.results], before)
        self.assertTrue((self.output / materializer.RECEIPT_NAME).is_file())
        self.assertFalse((self.output / "verification_input_manifest.json").exists())
        self.assertEqual((self.output / self.reference_alias).read_bytes(), (self.data / self.reference_relative).read_bytes())

    def test_single_parent_checkpoint_root_scans_indices_recursively(self):
        external = tempfile.TemporaryDirectory(prefix="paper_checkpoint_output_")
        self.addCleanup(external.cleanup)
        receipt = self.run_fixture(roots=[self.base], output=Path(external.name) / "fresh")
        self.assertEqual(receipt["fresh_rollout_files"], 2)

    def test_missing_index_id_rejected_before_any_write(self):
        self.indices[1].unlink()
        with self.assertRaisesRegex(ValueError, "No checkpoint output index|Incomplete"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_extra_and_duplicate_index_ids_rejected(self):
        index = self.read(self.indices[0])
        for identifier in ("test_a", "unreported_experiment"):
            row = dict(index["outputs"][0], id=identifier)
            self.write(self.indices[0], {**index, "rollout_count": 2, "outputs": index["outputs"] + [row]})
            with self.assertRaisesRegex(ValueError, "Duplicate|Unexpected"):
                self.run_fixture()
            self.assertFalse(self.output.exists())

    def test_index_alias_and_metadata_must_match_matrix(self):
        for key, value in (("verification_local_relative", "wrong/alias.jsonl"), ("family", "wrong_family"), ("historical_output_relative", "wrong/historical.jsonl")):
            index = self.read(self.indices[0])
            original = index["outputs"][0][key]
            index["outputs"][0][key] = value
            self.write(self.indices[0], index)
            with self.assertRaisesRegex(ValueError, "differs from matrix"):
                self.run_fixture()
            index["outputs"][0][key] = original
            self.write(self.indices[0], index)

    def test_outside_root_result_rejected(self):
        index = self.read(self.indices[0])
        index["outputs"][0]["result"] = str(self.results[1])
        self.write(self.indices[0], index)
        with self.assertRaisesRegex(ValueError, "outside its declared"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_nonfresh_or_incomplete_receipts_rejected(self):
        original = self.read(self.receipts[0])
        for field, value in (("status", "failed"), ("returncode", 1), ("returncode", False), ("result_is_fresh_inference_not_archived_copy", False)):
            self.write(self.receipts[0], {**original, field: value})
            with self.assertRaisesRegex(ValueError, "Failed/incomplete/non-fresh"):
                self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_duplicate_or_extra_receipt_members_rejected(self):
        original = self.read(self.receipts[0])
        for identifier in ("test_a", "unknown"):
            extra = {**original["members"][0], "id": identifier}
            self.write(self.receipts[0], {**original, "members": original["members"] + [extra]})
            with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
                self.run_fixture()

    def test_missing_receipt_rejected(self):
        self.receipts[0].unlink()
        with self.assertRaisesRegex(ValueError, "No completed runner receipts"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_changed_result_hash_rejected_even_if_row_count_same(self):
        self.results[0].write_bytes(b'{"id":8}\n{"id":9}\n')
        with self.assertRaisesRegex(ValueError, "count/SHA"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_wrong_receipted_count_rejected(self):
        receipt = self.read(self.receipts[0])
        receipt["members"][0]["record_count"] = 1
        self.write(self.receipts[0], receipt)
        with self.assertRaisesRegex(ValueError, "wrong-size"):
            self.run_fixture()

    def test_index_receipt_paths_must_correspond(self):
        duplicate = self.results[0].with_name("different.jsonl")
        duplicate.write_bytes(self.results[0].read_bytes())
        index = self.read(self.indices[0])
        index["outputs"][0]["result"] = str(duplicate)
        self.write(self.indices[0], index)
        with self.assertRaisesRegex(ValueError, "paths differ"):
            self.run_fixture()

    def test_reference_hash_and_unique_registry_identity_required(self):
        original = self.read(self.registry)
        changed = self.read(self.registry)
        changed["entries"][0]["sha256"] = "0" * 64
        self.write(self.registry, changed)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            self.run_fixture()
        self.write(self.registry, {"entries": original["entries"] * 2})
        with self.assertRaisesRegex(ValueError, "unique artifact"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_protected_overlaps_rejected(self):
        for output in (self.data / "new", self.roots[0] / "new", self.base, ROOT / "new"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlaps protected"):
                self.run_fixture(output=output)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "NEW output"):
            self.run_fixture()

    def test_separate_models_root_overlap_rejected(self):
        model = self.base / "model_cache"
        with self.assertRaisesRegex(ValueError, "overlaps protected"):
            self.run_fixture(output=model / "new", models_root=model)

    def test_overlapping_or_duplicate_checkpoint_roots_rejected(self):
        for roots in ([self.roots[0], self.roots[0]], [self.base, self.roots[0]]):
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                self.run_fixture(roots=roots)

    def test_unsafe_alias_rejected_before_output_creation(self):
        self.entries[0]["verification_local_relative"] = "../escape.jsonl"
        # Alias contract mismatch is also a correct early rejection.
        with self.assertRaises(ValueError):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_symlink_source_or_destination_rejected(self):
        link = self.base / "root_link"
        try:
            link.symlink_to(self.roots[0], target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation unavailable to this Windows account")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.run_fixture(roots=[link, self.roots[1]])

    def test_relative_recorded_output_path_rejected(self):
        index = self.read(self.indices[0])
        index["outputs"][0]["result"] = "relative/output.jsonl"
        self.write(self.indices[0], index)
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            self.run_fixture()

    def test_cli_structured_failure(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = materializer.main(["--checkpoint-root", str(self.base / "missing"), "--data-root", str(self.data), "--output-dir", str(self.output)])
        self.assertEqual(result, 1)
        self.assertFalse(json.loads(stderr.getvalue())["passed"])


if __name__ == "__main__":
    unittest.main()
