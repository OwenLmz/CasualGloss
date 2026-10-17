import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/reproduction"))
SPEC = importlib.util.spec_from_file_location("validate_data_builders", ROOT / "scripts/reproduction/validate_data_builders.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DataBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source, self.data = self.base / "clean_source", self.base / "installed_bundle"
        self.source.mkdir()
        self.data.mkdir()
        self.output = self.base / "new_validation"
        self.artifacts, self.sources = [], []
        for relative in MODULE.required_assets(MODULE.CHECKS):
            payload = b'{"id":"fixture"}\n'
            self.write(self.data / relative, payload)
            self.artifacts.append({"relative_to_remote_root": relative, "type": "file",
                "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
        for relative in MODULE.PRODUCERS.values():
            payload = b'# immutable producer fixture\n'
            self.write(self.source / relative, payload)
            self.sources.append({"repository_path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                                 "size_bytes": len(payload)})
        self.write_json(self.source / "provenance/artifact_manifest.json", {"entries": self.artifacts})
        self.write_json(self.source / "provenance/source_manifest.json", {"files": self.sources})

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def write_json(self, path, obj):
        self.write(path, (json.dumps(obj) + "\n").encode())

    def set_asset_json(self, relative, obj):
        path = self.data / relative
        self.write_json(path, obj)
        for entry in self.artifacts:
            if entry["relative_to_remote_root"] == relative:
                entry["sha256"] = MODULE.sha256(path)
                entry["size_bytes"] = path.stat().st_size
        self.write_json(self.source / "provenance/artifact_manifest.json", {"entries": self.artifacts})

    def preflight(self, checks=("recovery",), output=None):
        return MODULE.preflight(self.data, output or self.output, checks, Path(sys.executable),
                                Path(sys.executable), source=self.source)

    def test_recovery_only_needs_no_audio_catalog_or_duplicate_augmentation(self):
        plan = self.preflight()
        self.assertEqual(plan["checks"], ("recovery",))
        self.assertIsNone(plan["audio"])
        self.assertEqual(len(plan["inputs"]), 9)
        self.assertFalse(self.output.exists())
        self.assertNotIn(MODULE.RECOVERY + "/" + MODULE.AUGMENTATION["relative_path"], plan["inputs"])

    def test_missing_input_rejected_before_output_creation(self):
        (self.data / next(iter(MODULE.RECOVERY_INPUTS.values()))).unlink()
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.preflight()
        self.assertFalse(self.output.exists())

    def test_modified_input_rejected_before_execution(self):
        (self.data / next(iter(MODULE.RECOVERY_INPUTS.values()))).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 or size mismatch"):
            self.preflight()
        self.assertFalse(self.output.exists())

    def test_modified_producer_rejected_before_output_creation(self):
        (self.source / MODULE.PRODUCERS["recovery"]).write_bytes(b"modified source")
        with self.assertRaisesRegex(ValueError, "SHA-256 or size mismatch"):
            self.preflight()
        self.assertFalse(self.output.exists())

    def test_missing_registry_entry_rejected(self):
        missing = next(iter(MODULE.RECOVERY_INPUTS.values()))
        self.write_json(self.source / "provenance/artifact_manifest.json",
                        {"entries": [item for item in self.artifacts if item["relative_to_remote_root"] != missing]})
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.preflight()
        self.assertFalse(self.output.exists())

    def test_source_and_entire_data_tree_overlap_rejected(self):
        for location in (self.source, self.source / "new", self.data, self.data / "new", self.base):
            with self.subTest(location=location):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self.preflight(output=location)

    def test_existing_output_rejected_even_if_empty(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.preflight()

    def test_relative_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalized absolute"):
            self.preflight(output=Path("some/new_output"))

    def test_waveform_check_requires_release_audio_metadata(self):
        with self.assertRaises(FileNotFoundError):
            self.preflight(checks=("waveform-qc",))
        self.assertFalse(self.output.exists())

    def test_audio_path_escape_rejected(self):
        raw = self.data / MODULE.CORPUS_PATHS["raw"]
        self.write_json(raw, {"tts_audio_16k_mono": "../escape.wav"})
        with (mock.patch.object(MODULE, "load_catalog", return_value={"manifest": {"sha256": "a" * 64}}),
              mock.patch.object(MODULE, "read_metadata", return_value={"file_manifest_utf8": "{}"}),
              mock.patch.object(MODULE, "parse_file_manifest", return_value={})):
            with self.assertRaisesRegex(ValueError, "Unsafe relative"):
                MODULE.verify_audio(self.data, self.source, raw)

    def test_audio_hash_is_checked_not_only_directory_presence(self):
        relative = MODULE.SPEECH + "/audio/normalized/example.wav"
        self.write(self.data / relative, b"modified waveform")
        raw = self.data / MODULE.CORPUS_PATHS["raw"]
        self.write_json(raw, {"tts_audio_16k_mono": "audio/normalized/example.wav"})
        entry = {"group": "audio", "sha256": hashlib.sha256(b"correct waveform").hexdigest(),
                 "size_bytes": len(b"correct waveform")}
        with (mock.patch.object(MODULE, "load_catalog", return_value={"manifest": {"sha256": "a" * 64}}),
              mock.patch.object(MODULE, "read_metadata", return_value={"file_manifest_utf8": "{}"}),
              mock.patch.object(MODULE, "parse_file_manifest", return_value={relative: entry})):
            with self.assertRaisesRegex(ValueError, "SHA-256 or size mismatch"):
                MODULE.verify_audio(self.data, self.source, raw)

    def test_stable_field_comparison_ignores_only_named_fields(self):
        left = {"created_at_utc": "now", "issues": [{"id": "a"}], "ok": True}
        right = {"created_at_utc": "then", "issues": [{"id": "a"}], "ok": True}
        self.assertEqual([], MODULE.differing_fields(left, right, {"created_at_utc"}))
        right["issues"] = [{"id": "b"}]
        self.assertEqual(["issues"], MODULE.differing_fields(left, right, {"created_at_utc"}))
        self.assertEqual(["missing"], MODULE.differing_fields({"missing": None}, {}))

    def test_producer_failure_is_preserved_as_failed_receipt(self):
        plan = self.preflight()
        with mock.patch.object(MODULE.subprocess, "run", return_value=mock.Mock(returncode=7)) as process:
            report = MODULE.validate(plan, Path(sys.executable), Path(sys.executable))
        self.assertFalse(report["passed"])
        self.assertEqual(report["producer_executions"][0]["returncode"], 7)
        self.assertTrue((self.output / "cpu_validation_report.json").is_file())
        environment = process.call_args.kwargs["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")

    def test_source_changed_after_preflight_is_rechecked(self):
        plan = self.preflight()
        (self.source / MODULE.PRODUCERS["recovery"]).write_bytes(b"changed after planning")
        with mock.patch.object(MODULE.subprocess, "run") as process:
            report = MODULE.validate(plan, Path(sys.executable), Path(sys.executable))
        self.assertFalse(report["passed"])
        process.assert_not_called()

    def test_successful_recovery_compares_six_outputs_without_original_augmentation(self):
        plan = self.preflight()
        expected_counts = {key: 1 for key in MODULE.RECOVERY_COUNTS}
        payload = b'{"rebuilt":"augmentation"}\n'
        augmentation = {"relative_path": MODULE.AUGMENTATION["relative_path"], "row_count": 1,
                        "sha256": hashlib.sha256(payload).hexdigest()}

        def produce(command, **kwargs):
            output = Path(command[command.index("--output-root") + 1])
            for relative in MODULE.RECOVERY_OUTPUTS:
                self.write(output / relative, (self.data / MODULE.RECOVERY / relative).read_bytes())
            self.write(output / augmentation["relative_path"], payload)
            inputs = {name.replace("-", "_"): plan["inputs"][relative]["sha256"]
                      for name, relative in MODULE.RECOVERY_INPUTS.items()}
            self.write_json(output / "reports/manifest_build_report.json",
                            {**expected_counts, "sha256_inputs": inputs})
            return mock.Mock(returncode=0)

        with (mock.patch.object(MODULE, "RECOVERY_COUNTS", expected_counts),
              mock.patch.object(MODULE, "AUGMENTATION", augmentation),
              mock.patch.object(MODULE.subprocess, "run", side_effect=produce)):
            report = MODULE.validate(plan, Path(sys.executable), Path(sys.executable))
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(len(report["checks"]["recovery_manifests"]["files"]), 6)
        self.assertTrue(report["checks"]["recovery_manifests"]["files"][-1]["reference_payload_not_required"])
        self.assertFalse((self.data / MODULE.RECOVERY / augmentation["relative_path"]).exists())

    def test_independent_finalizer_uses_frozen_qc_and_stable_summary(self):
        self.set_asset_json(MODULE.CORPUS_PATHS["qc"],
                            {"record_count": 1, "expected_count": 1, "issues": []})
        summary = {"record_count": 1, "created_at_utc": "historical", "clean_config_sha256": "old",
                   "exclusions_sha256": "old", "stable_field": {"same": True}}
        self.set_asset_json(MODULE.CORPUS_PATHS["summary"], summary)
        with mock.patch.object(MODULE, "verify_audio", return_value={"normalized_waveforms_hash_verified": 1}):
            plan = self.preflight(checks=("finalizer",))

        def produce(command, **kwargs):
            self.assertEqual(command[command.index("--raw-audio-validation") + 1],
                             str(self.data / MODULE.CORPUS_PATHS["qc"]))
            target = Path(command[command.index("--output-dir") + 1])
            self.write(target / "manifests/full.cleaned.jsonl", (self.data / MODULE.CORPUS_PATHS["clean"]).read_bytes())
            self.write_json(target / "reports/corpus_summary.cleaned.json",
                            {**summary, "created_at_utc": "fresh", "clean_config_sha256": "new", "exclusions_sha256": "new"})
            return mock.Mock(returncode=0)

        with mock.patch.object(MODULE.subprocess, "run", side_effect=produce):
            report = MODULE.validate(plan, Path(sys.executable), Path(sys.executable))
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["checks"]["cleaned_corpus"]["waveform_qc_source"], "frozen_registered_report")
        self.assertNotIn("waveform_qc", report["checks"])

    def test_independent_waveform_qc_ignores_only_clock_timestamp(self):
        frozen_qc = {"record_count": 1, "expected_count": 1, "issues": [],
                     "status_counts": {"valid": 1}, "created_at_utc": "historical", "ok": True}
        self.set_asset_json(MODULE.CORPUS_PATHS["qc"], frozen_qc)
        with mock.patch.object(MODULE, "verify_audio", return_value={"normalized_waveforms_hash_verified": 1}):
            plan = self.preflight(checks=("waveform-qc",))

        def produce(command, **kwargs):
            self.write_json(Path(command[command.index("--report") + 1]), {**frozen_qc, "created_at_utc": "fresh"})
            return mock.Mock(returncode=0)

        with mock.patch.object(MODULE.subprocess, "run", side_effect=produce):
            report = MODULE.validate(plan, Path(sys.executable), Path(sys.executable))
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["checks"]["waveform_qc"]["mismatched_fields"], [])
        self.assertNotIn("cleaned_corpus", report["checks"])

    def test_original_producers_have_registered_source_fingerprints(self):
        registry = json.loads((ROOT / "provenance/source_manifest.json").read_text(encoding="utf-8"))
        lookup = {item["repository_path"]: item for item in registry["files"]}
        for relative in MODULE.PRODUCERS.values():
            item = lookup[relative]
            self.assertEqual(MODULE.sha256(ROOT / relative), item.get("repository_sha256", item["sha256"]))


if __name__ == "__main__":
    unittest.main()
