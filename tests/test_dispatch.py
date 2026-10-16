import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_dispatch", ROOT / "scripts/paper.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DispatchTests(unittest.TestCase):
    def test_all_entries_present(self):
        for key, path in module.ENTRIES.items():
            with self.subTest(entry=key):
                self.assertTrue(path.is_file(), str(path))

    def test_arguments_passed_unchanged(self):
        args = ["--", "--adapter-dir", "some path/adapter_final", "--stage", "fresh-check"]
        cmd = module.command("rollout", args)
        self.assertEqual(cmd[-4:], args[1:])
        self.assertEqual(cmd[2:], args[1:])

    def test_distributed_prefix(self):
        cmd = module.command("train-policy", ["--help"], 2)
        self.assertIn("--nproc_per_node=2", cmd)
        self.assertEqual(cmd[-1], "--help")

    def test_no_multigpu_evaluator(self):
        with self.assertRaises(ValueError):
            module.command("rollout", [], 2)

    def test_no_fake_ddp_learned_baseline(self):
        with self.assertRaises(ValueError):
            module.command("train-learned", [], 2)

    def test_original_bytecode_disabled(self):
        self.assertEqual(module.environment()["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
