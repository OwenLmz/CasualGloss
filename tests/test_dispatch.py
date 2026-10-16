"""No-GPU tests for isolated entrypoint dispatch and source-path safety."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import paper
from check_release import source_expectation


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.root = (Path(__file__).resolve().parent / "synthetic_dispatch_fixture").resolve()
        self.root_patch = patch.object(paper, "ROOT", self.root)
        self.root_patch.start()
        # The dispatcher only asks whether an entry exists. Mock this bounded
        # check instead of creating a fake research tree or requiring temp I/O.
        self.exists_patch = patch.object(Path, "is_file", return_value=True)
        self.exists_patch.start()

    def tearDown(self):
        self.exists_patch.stop()
        self.root_patch.stop()

    def test_arguments_are_forwarded_without_rewriting(self):
        cmd = paper.command("rollout", ["--", "--config", "path with spaces.json", "--split", "test"])
        self.assertEqual(cmd[2:], ["--config", "path with spaces.json", "--split", "test"])

    def test_policy_supports_real_ddp(self):
        self.assertIn("torch.distributed.run", paper.command("train-policy", [], nproc=2))

    def test_distributed_trainers_use_torchrun_for_one_gpu(self):
        for entry, spec in paper.ENTRIES.items():
            if spec.distributed:
                with self.subTest(entry=entry):
                    cmd = paper.command(entry, ["--", "--config", "unchanged.json"])
                    self.assertEqual(cmd[:5], [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"])
                    self.assertEqual(cmd[5:], [str(self.root / spec.path), "--config", "unchanged.json"])

    def test_non_distributed_entries_remain_direct(self):
        for entry, spec in paper.ENTRIES.items():
            if not spec.distributed:
                with self.subTest(entry=entry):
                    self.assertEqual(paper.command(entry, []), [sys.executable, str(self.root / spec.path)])

    def test_transllama_does_not_fake_ddp(self):
        with self.assertRaises(ValueError):
            paper.command("train-learned", [], nproc=2)

    def test_ablation_import_path_cannot_pick_method_common(self):
        with patch.dict(os.environ, {"PYTHONPATH": "unmanaged/path"}):
            paths = paper.environment("train-ablation")["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(paths, [str(self.root / "ablations")])
        self.assertNotIn(str(self.root / "method/causalgloss"), paths)

    def test_clerc_import_paths_are_precise(self):
        action = paper.environment("eval-epee-action")["PYTHONPATH"].split(os.pathsep)
        learned = paper.environment("eval-epee-learned")["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(action, [str(self.root / "evaluation/clerc"), str(self.root / "method/causalgloss")])
        self.assertEqual(learned, [str(self.root / "evaluation/clerc"), str(self.root / "baselines/transllama")])

    def test_nonpositive_process_count_rejected(self):
        with self.assertRaises(ValueError):
            paper.command("train-policy", [], nproc=0)

    def test_manifest_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            source_expectation({"repository_path": "../secret", "sha256": "0" * 64, "size_bytes": 0})

    def test_adapted_source_uses_repository_checksum(self):
        entry = {"repository_path": "method/train.py", "sha256": "a" * 64, "size_bytes": 10,
                 "repository_sha256": "b" * 64, "repository_size_bytes": 11}
        self.assertEqual(source_expectation(entry), ("method/train.py", "b" * 64, 11))


if __name__ == "__main__":
    unittest.main()
