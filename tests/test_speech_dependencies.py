import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_corpus_inputs", ROOT / "scripts/reproduction/validate_corpus_inputs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    config = {"seed": 10, "tts": {"speakers": ["voice"], "model_revision": "tts"},
              "translation": {"model_revision": "translation"}}
    excluded = {"id": "excluded", "split": "train", "tts_duration_seconds": 55.0,
                "tts_speaker": "voice"}
    kept = {"id": "kept", "split": "test", "tts_duration_seconds": 2.0,
            "tts_speaker": "voice", "tts_seed": 10, "tts_model_revision": "tts",
            "source_text_zh_model_revision": "translation", "source_text_zh_status": "completed",
            "tts_status": "completed", "source_text_zh": "repaired",
            "source_text_zh_model_output_before_repair": "before",
            "source_text_zh_repair_reason": "reason", "tts_audio_original": "audio/original.wav",
            "tts_audio_16k_mono": "audio/mono.wav"}
    raw = [excluded, kept]
    clean = [kept]
    qc = {"issues": [{"id": "excluded", "issue": "duration_out_of_range"}],
          "record_count": 2, "expected_count": 2, "status_counts": {"flagged": 1}}
    summary = {"raw_record_count": 2, "excluded_record_count": 1, "record_count": 1,
               "unique_id_count": 1, "split_counts": {"test": 1}, "speaker_counts": {"voice": 1},
               "translation_status_counts": {"completed": 1}, "tts_status_counts": {"completed": 1},
               "total_duration_seconds": 2.0, "total_duration_hours": round(2 / 3600, 6)}
    repairs = {"kept": {"translation": "repaired", "reason": "reason"}}
    repair_report = {"repairs": [{"id": "kept", "before": "before", "after": "repaired", "reason": "reason"}],
                     "applied_repair_count": 1, "ok": True}
    return [[json.dumps(row) for row in raw], raw, [json.dumps(row) for row in clean], clean,
            qc, summary, repairs, repair_report, [kept], config]


class SpeechDependencyTests(unittest.TestCase):
    def test_only_one_original_waveform_qc_source(self):
        path = ROOT / "speech/frontend/validate_audio_manifest.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "622058e1845585c5d94c8000ed6f66515d9dcb3aa77a99f3b8e138387f1dabc5")
        self.assertFalse((ROOT / "speech/corpus/validate_audio_manifest.py").exists())

    def test_original_repairs_are_preserved_byte_for_byte(self):
        path = ROOT / "configs/speech/translation_repairs.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "62098538ef445b480b42449c53eca21455dc09d92891eff3fffc0ea92107a2f8")

    def test_locked500_configuration_is_final_only(self):
        cfg = json.loads((ROOT / "configs/speech/locked500_frontend.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["seed"], 20260726)
        self.assertEqual(cfg["tts"]["speakers"], ["Vivian", "Serena", "Uncle_Fu", "Dylan"])
        self.assertEqual(cfg["seamlessstreaming"]["source_segment_size_ms"], 320)
        for obsolete in ("simulstreaming", "cascade_streaming", "selected_frontend", "pilot_count", "finetune2"):
            self.assertNotIn(obsolete, cfg)

    def test_clean_transform_accepts_actual_computation(self):
        result = MODULE.validate(*fixture())
        self.assertTrue(result["passed"], result["errors"])
        self.assertFalse(result["waveform_samples_revalidated"])
        self.assertFalse(result["computed"]["audio_reference_existence_checked"])

    def test_rejects_cleaned_qc_in_place_of_raw_qc(self):
        args = fixture()
        args[4] = {"issues": [], "record_count": 2, "expected_count": 2, "status_counts": {"flagged": 0}}
        self.assertFalse(MODULE.validate(*args)["passed"])

    def test_rejects_tampered_manifest_even_with_correct_counts(self):
        args = fixture()
        args[2] = ["not the original row"]
        self.assertFalse(MODULE.validate(*args)["passed"])

    def test_rejects_wrong_locked500_seed(self):
        args = fixture()
        args[9] = copy.deepcopy(args[9])
        args[9]["seed"] += 1
        self.assertFalse(MODULE.validate(*args)["passed"])

    def test_rejects_modified_repair_or_duration(self):
        args = fixture()
        args[6]["kept"]["translation"] = "not the frozen repair"
        args[5]["total_duration_hours"] = 99
        self.assertFalse(MODULE.validate(*args)["passed"])


if __name__ == "__main__":
    unittest.main()
