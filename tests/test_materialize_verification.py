import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_verification_inputs", ROOT / "scripts/reproduction/materialize_verification_inputs.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaterializeVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "installed_SN_Research"
        self.source.mkdir()
        self.targets = MODULE.target_paths()
        self.assertEqual(len(self.targets), 83)
        self.entries = []
        for index, (remote, alias) in enumerate(self.targets):
            relative = MODULE.DATA_PREFIX + "/" + remote
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately include BOM, CRLF, and non-JSON bytes. The wrapper
            # must preserve raw bytes rather than parse or reserialize inputs.
            payload = b"\xef\xbb\xbf" + str(index).encode("ascii") + b"\r\n"
            path.write_bytes(payload)
            self.entries.append({"remote_path": MODULE.REMOTE_ROOT + "/" + remote,
                                 "relative_to_remote_root": relative, "type": "file",
                                 "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
        self.manifest = self.base / "registry.json"
        self.write_manifest()
        self.output = self.base / "new_verifier_inputs"

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"entries": self.entries}), encoding="utf-8")

    def run_materialize(self, **kwargs):
        return MODULE.materialize(self.source, self.output, self.manifest, **kwargs)

    def test_exact_83_allowlist_copy_and_aliases_no_fake_remote_manifest(self):
        (self.source / "unrelated.jsonl").write_bytes(b"do not copy")
        receipt = self.run_materialize()
        expected = {alias for _, alias in self.targets} | {"materialization_receipt.json"}
        actual = {path.relative_to(self.output).as_posix() for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(actual, expected)
        self.assertEqual(receipt["input_files"], 83)
        self.assertFalse(receipt["remote_verification_performed"])
        self.assertEqual(receipt["mode"], "copy")
        for source, alias in self.targets:
            self.assertEqual((self.source / MODULE.DATA_PREFIX / source).read_bytes(),
                             (self.output / alias).read_bytes())
        self.assertFalse((self.output / "verification_input_manifest.json").exists())

    def test_refuses_existing_output_even_if_empty(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_materialize()

    def test_rejects_modified_source_before_creating_output(self):
        (self.source / self.entries[-1]["relative_to_remote_root"]).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.run_materialize()
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_or_unhashed_registry_entry(self):
        self.entries.append(dict(self.entries[0]))
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.run_materialize()
        self.entries.pop()
        self.entries[0]["sha256"] = None
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.run_materialize()
        self.assertFalse(self.output.exists())

    def test_rejects_escaping_allowlist_and_repository_destination(self):
        with patch.object(MODULE, "target_paths", return_value=[("../outside", "escaped.jsonl")]):
            with self.assertRaises(ValueError):
                self.run_materialize()
        with self.assertRaises(ValueError):
            MODULE.materialize(self.source, ROOT / "must_not_create_inputs", self.manifest)
        self.assertFalse((ROOT / "must_not_create_inputs").exists())

    def test_rejects_source_and_destination_overlap(self):
        with self.assertRaises(ValueError):
            MODULE.materialize(self.source, self.source / "inputs", self.manifest)

    def test_rejects_source_symlink_escape(self):
        victim = self.source / self.entries[0]["relative_to_remote_root"]
        external = self.base / "outside_source.jsonl"
        external.write_bytes(victim.read_bytes())
        victim.unlink()
        try:
            victim.symlink_to(external)
        except OSError:
            self.skipTest("Platform does not allow unprivileged symlinks")
        with self.assertRaises(ValueError):
            self.run_materialize()
        self.assertFalse(self.output.exists())

    def test_hardlinks_are_rejected_without_altering_source(self):
        with self.assertRaises(ValueError):
            self.run_materialize(mode="hardlink")
        self.assertFalse(self.output.exists())

    def test_copies_do_not_change_installed_bundle_link_counts(self):
        source, alias = self.targets[0]
        original = self.source / MODULE.DATA_PREFIX / source
        original_links = original.stat().st_nlink
        self.run_materialize()
        self.assertEqual(original_links, original.stat().st_nlink)
        self.assertNotEqual(original.stat().st_ino, (self.output / alias).stat().st_ino)


if __name__ == "__main__":
    unittest.main()
