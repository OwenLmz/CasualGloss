import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/reproduction"))
import experiment_matrix as matrix
import run_checkpoint_suite as runner
import run_training_plan as training


class ReproductionMatrixTests(unittest.TestCase):
    def setUp(self):
        self.matrix = matrix.load_matrix()

    def test_exact_82_historical_outputs(self):
        spec = importlib.util.spec_from_file_location("verification_fetch_for_matrix_test", ROOT / "scripts/verification/fetch_verification_inputs.py")
        fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch)
        expected = {(r, local) for r, local in fetch.target_paths() if r.endswith("sentence_rollouts.jsonl") or "/outputs/v" in r}
        actual = {(r["historical_output_relative"].removeprefix("data/how2sign_english_asl_gloss_v1/"), r["verification_local_relative"])
                  for r in self.matrix["experiments"]}
        self.assertEqual(82, len(expected))
        self.assertEqual(expected, actual)

    def test_no_unreported_lexical_clerc(self):
        self.assertFalse(any("lexical" in row["id"] for row in self.matrix["experiments"] if row["family"] == "clerc401"))

    def test_recorded_batch_and_precision_not_global_default(self):
        rows = self.matrix["experiments"]
        self.assertTrue(all(row["precision"] == "float16" and row["generation_batch_size"] == 16 for row in rows if row["family"] == "clerc401"))
        self.assertEqual({32}, {r["generation_batch_size"] for r in rows if r["id"].startswith("locked_causalgloss_")})
        self.assertEqual({16}, {r["generation_batch_size"] for r in rows if r["id"].startswith("locked_fixed_")})
        self.assertEqual({32}, {r["generation_batch_size"] for r in rows if r["id"].startswith("locked_transllama_")})
        self.assertEqual({1}, {r["generation_batch_size"] for r in rows if r["kind"] == "speech"})

    def test_config_seed_and_revision(self):
        for row in self.matrix["experiments"]:
            config = json.loads((ROOT / row["config"]).read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], row["seed"])
            self.assertEqual(config["model_revision"], row["model_revision"])

    def test_adapter_groups_and_inputs_registered(self):
        manifest = json.loads((ROOT / "provenance/artifact_manifest.json").read_text(encoding="utf-8"))
        registered = {r["relative_to_remote_root"] for r in manifest["entries"]}
        fingerprints = {r["relative_to_remote_root"]: r.get("sha256") for r in manifest["entries"]}
        for row in self.matrix["experiments"]:
            self.assertIn(row["adapter_relative"] + "/adapter_model.safetensors", registered)
            for value in row["inputs"].values():
                self.assertIn(value, registered)
            for name, value in row["inputs"].items():
                self.assertEqual(fingerprints[value], row["input_sha256"][name])
            for name, value in row["adapter_sha256"].items():
                self.assertEqual(fingerprints[row["adapter_relative"] + "/" + name], value)

    def test_no_source_data_overwrite(self):
        data_root, models_root = Path("D:/private/bundle"), Path("D:/private/models")
        for output in (ROOT, data_root, data_root / "data/new_results",
                       data_root / "new_outputs", data_root / "adapters/new_outputs",
                       models_root / "new_results", data_root.parent):
            with self.assertRaises(ValueError):
                matrix.protected_output(output, data_root, models_root)

    def test_output_sibling_of_installed_data_is_allowed(self):
        data_root, models_root = Path("D:/private/bundle"), Path("D:/private/models")
        output = Path("D:/private/fresh_outputs")
        self.assertEqual(output.resolve(), matrix.protected_output(output, data_root, models_root))

    def test_path_traversal_rejected(self):
        for value in ("../anything", "/absolute/file", "safe/../../escape", "safe//double", "safe\\windows"):
            with self.assertRaises(ValueError):
                matrix.relative_path(value)

    def test_model_shards_not_inferred_from_directory_only(self):
        snapshot = ROOT / "tests/nonexistent_snapshot_for_readonly_test" / matrix.MODEL_REVISION
        self.assertFalse(snapshot.exists())
        self.assertTrue(matrix.check_snapshot(snapshot))

    def test_grouped_sweeps_keep_all_82_unique_outputs(self):
        plans = [runner.make_plan(row, Path("D:/private/bundle"), Path("D:/private/models"), Path("D:/new/results"))
                 for row in self.matrix["experiments"]]
        grouped = runner.coalesce_plans(plans)
        self.assertEqual(40, len(grouped))
        members = [member for plan in grouped for member in plan["members"]]
        self.assertEqual(82, len(members))
        self.assertEqual(82, len({member["result"] for member in members}))
        sweeps = [plan for plan in grouped if plan["entry"]["kind"] in ("learned", "epee_learned")]
        self.assertEqual(6, len(sweeps))
        self.assertTrue(all(plan["command"][plan["command"].index("--k-values") + 1] == "1,2,3,4,5,6,7,8" for plan in sweeps))

    def test_only_config_model_location_rewritten(self):
        row = self.matrix["experiments"][0]
        original = json.loads((ROOT / row["config"]).read_text(encoding="utf-8"))
        plan = runner.make_plan(row, Path("D:/private/bundle"), Path("D:/private/models"), Path("D:/new/results"))
        altered = plan["runtime_config_contents"]
        self.assertEqual({"model_id"}, {key for key in original if original[key] != altered[key]})

    def test_each_checkpoint_links_to_recorded_training_run(self):
        known = {row["id"] for row in training.load_training()["runs"]}
        self.assertTrue(all(row["training_run_id"] in known for row in self.matrix["experiments"]))

    def test_new_training_receipt_cannot_point_to_archived_results(self):
        entry = self.matrix["experiments"][0]
        receipt = {"id":entry["training_run_id"],"status":"completed","fresh_training_executed":True,
                   "selected_adapter":"D:/private/bundle/data/old/adapter_best"}
        with mock.patch.object(Path,"read_text",return_value=json.dumps(receipt)):
            with self.assertRaises(ValueError):
                runner.resolved_adapter(entry,Path("D:/private/bundle"),Path("D:/new/training"))

    def test_new_training_checkpoint_requires_its_receipt_hashes(self):
        entry = self.matrix["experiments"][0]
        receipt = {"id":entry["training_run_id"],"status":"completed","fresh_training_executed":True,
                   "selected_adapter":"D:/new/training/some_run/outputs/adapter_best","selected_adapter_sha256":{}}
        with (mock.patch.object(Path,"read_text",return_value=json.dumps(receipt)),
              mock.patch.object(Path,"is_file",return_value=True),mock.patch.object(runner,"sha256",return_value="0"*64)):
            with self.assertRaises(ValueError):
                runner.resolved_adapter(entry,Path("D:/private/bundle"),Path("D:/new/training"))

    def test_unknown_selection_fails_instead_of_running_everything(self):
        with self.assertRaises(ValueError):
            matrix.select_entries(self.matrix, ids=["typo_run"])
        with self.assertRaises(ValueError):
            matrix.select_entries(self.matrix, families=["clerc401"], ids=["speech_lexical_internal"])


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = training.load_training()

    def test_thirty_runs_with_evidence(self):
        self.assertEqual(30,len(self.plan["runs"]))
        self.assertTrue(all(row["historical_summary_evidence_available"] for row in self.plan["runs"]))
        self.assertTrue(all(row["status"] == "resolved" and not row["unresolved"] for row in self.plan["runs"]))

    def test_world_sizes_and_ddp_single_device(self):
        rows = {row["id"]:row for row in self.plan["runs"]}
        self.assertEqual(4,rows["upstream_finetune1"]["world_size"])
        self.assertEqual(4,rows["lexical_internal_seed2"]["world_size"])
        self.assertEqual(2,rows["fixed_k3_seed20260821"]["world_size"])
        self.assertTrue(all(row["world_size"] == 2 for row in rows.values() if row["id"].startswith("causalgloss_stage_b")))
        selected = rows["causalgloss_stage_a"]
        invocation = training.training_invocation(selected,Path("D:/private/bundle"),Path("D:/private/models"),Path("D:/new/training"))
        self.assertIn("--nproc_per_node=1",invocation["command"])
        self.assertIn("torch.distributed.run",invocation["command"])

    def test_budget_not_replaced_with_observed_steps(self):
        for row in self.plan["runs"]:
            invocation = training.training_invocation(row,Path("D:/private/bundle"),Path("D:/private/models"),Path("D:/new/training"))
            self.assertNotIn("--max-updates",invocation["command"])
            self.assertNotIn("--epochs",invocation["command"])

    def test_all_training_inputs_hash_registered(self):
        assets = json.loads((ROOT/"provenance/artifact_manifest.json").read_text(encoding="utf-8"))
        registry = {item["relative_to_remote_root"]:item.get("sha256") for item in assets["entries"]}
        for row in self.plan["runs"]:
            for name,relative in row["inputs"].items():
                self.assertEqual(registry[relative],row["input_sha256"][name])

    def test_dependency_order_and_expansion(self):
        done = set()
        for row in self.plan["runs"]:
            self.assertTrue(set(row["dependencies"]) <= done)
            done.add(row["id"])
        selected = training.select_training(self.plan,ids=["causalgloss_stage_b_20260829"],include_dependencies=True)
        self.assertEqual(["upstream_finetune1","causalgloss_stage_a","causalgloss_stage_b_20260829"],[row["id"] for row in selected])

    def test_new_and_archived_initializations_are_explicit(self):
        row = next(row for row in self.plan["runs"] if row["id"] == "ablation_no_burst_stage_b")
        fresh = training.source_adapter(row,Path("D:/private/bundle"),Path("D:/new/training"),"new")
        archived = training.source_adapter(row,Path("D:/private/bundle"),Path("D:/new/training"),"archived")
        self.assertNotEqual(fresh,archived)
        self.assertIn("ablation_no_burst_stage_a",str(fresh))

    def test_default_dry_run_cannot_start_training_or_write(self):
        # Absolute paths on BOTH Windows and POSIX. Literal "D:/..." is a
        # source-relative child on Linux and must correctly fail the guard.
        # These paths need not exist: dry-run must not materialize anything.
        outside = Path(ROOT.anchor) / "causalgloss_readonly_dry_run_test"
        argv = ["run_training_plan.py","--id","upstream_finetune1","--data-root",str(outside/"bundle"),
                "--models-root",str(outside/"models"),"--output-root",str(outside/"training")]
        with (mock.patch.object(sys,"argv",argv), mock.patch.object(training.subprocess,"run") as execute,
              mock.patch.object(Path,"mkdir") as mkdir, contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(0,training.main())
            self.assertIn("dry_run_no_files_written_no_training",output.getvalue())
            execute.assert_not_called()
            mkdir.assert_not_called()

    def test_causal_selection_uses_development_not_locked_test(self):
        row = next(row for row in self.plan["runs"] if row["id"] == "causalgloss_stage_b_20260829")
        invocation = training.training_invocation(row,Path("D:/private/bundle"),Path("D:/private/models"),Path("D:/new/training"))
        self.assertEqual(4,len(invocation["post_training_selection_commands"]))
        for command in invocation["post_training_selection_commands"][:3]:
            self.assertIn("v2_3_development",command)
            self.assertIn("32",command)
        self.assertTrue(invocation["post_training_selection_commands"][-1][1].endswith("assess_screen_gate.py"))


if __name__ == "__main__":
    unittest.main()
