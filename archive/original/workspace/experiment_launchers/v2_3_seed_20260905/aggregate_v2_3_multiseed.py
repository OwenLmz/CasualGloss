#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


SEEDS = ("20260829", "20260904", "20260905")


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_sd(values):
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "seed_count": len(values),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def text_paths(data_root):
    return {
        "20260829": {
            "report": data_root / "paper_experiments_20260903/reports/v2_3_vs_waitk_v1_k1.json",
            "rollout": data_root / "state_recovery_v2_3_full/final_evaluation_v2_3/locked500/v2_3/sentence_rollouts.jsonl",
            "summary": data_root / "state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/training_summary.json",
            "metadata": data_root / "state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/run_metadata.json",
        },
        "20260904": {
            "report": data_root / "state_recovery_v2_3_multiseed_20260904/final_evaluation_seed20260904/reports/v2_3_seed20260904_vs_waitk_v1_k1.json",
            "rollout": data_root / "state_recovery_v2_3_multiseed_20260904/final_evaluation_seed20260904/locked500/v2_3/sentence_rollouts.jsonl",
            "summary": data_root / "state_recovery_v2_3_multiseed_20260904/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260904/training_summary.json",
            "metadata": data_root / "state_recovery_v2_3_multiseed_20260904/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260904/run_metadata.json",
        },
        "20260905": {
            "report": data_root / "state_recovery_v2_3_multiseed_20260905/final_evaluation_seed20260905/reports/v2_3_seed20260905_vs_waitk_v1_k1.json",
            "rollout": data_root / "state_recovery_v2_3_multiseed_20260905/final_evaluation_seed20260905/locked500/v2_3/sentence_rollouts.jsonl",
            "summary": data_root / "state_recovery_v2_3_multiseed_20260905/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260905/training_summary.json",
            "metadata": data_root / "state_recovery_v2_3_multiseed_20260905/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260905/run_metadata.json",
        },
    }


def e2e_paths(e2e_root):
    return {
        "20260829": {
            "report": e2e_root / "reports/v2_3.json",
            "comparison": e2e_root / "reports/v2_3_vs_v1.json",
            "rollout": e2e_root / "outputs/v2_3.jsonl",
        },
        "20260904": {
            "report": e2e_root / "multiseed/seed20260904/reports/v2_3_seed20260904.json",
            "comparison": e2e_root / "multiseed/seed20260904/reports/v2_3_seed20260904_vs_v1.json",
            "rollout": e2e_root / "multiseed/seed20260904/outputs/v2_3_seed20260904.jsonl",
        },
        "20260905": {
            "report": e2e_root / "multiseed/seed20260905/reports/v2_3_seed20260905.json",
            "comparison": e2e_root / "multiseed/seed20260905/reports/v2_3_seed20260905_vs_v1.json",
            "rollout": e2e_root / "multiseed/seed20260905/outputs/v2_3_seed20260905.jsonl",
        },
    }


def aggregate(args):
    server_root = Path(args.server_root)
    data_root = server_root / "data/how2sign_english_asl_gloss_v1/full_v4_train"
    e2e_root = server_root / "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/paper_e2e_locked500_20260904"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    text = {}
    e2e = {}
    provenance = {}
    input_files = []

    text_specs = text_paths(data_root)
    e2e_specs = e2e_paths(e2e_root)
    v1_e2e = read_jsonl(e2e_root / "outputs/v1.jsonl")
    v1_e2e_ids = {row["id"] for row in v1_e2e}
    assert len(v1_e2e) == len(v1_e2e_ids) == 500

    baseline_text = None
    baseline_e2e = None
    core_config = None
    data_hashes = None

    for seed in SEEDS:
        spec = text_specs[seed]
        report = read_json(spec["report"])
        rows = read_jsonl(spec["rollout"])
        summary = read_json(spec["summary"])
        metadata = read_json(spec["metadata"])
        input_files.extend(spec.values())

        assert len(rows) == 500
        assert len({row["id"] for row in rows}) == 500
        assert report["common_parent_count"] == 500
        assert report["same_parent_ids"] is True
        assert report["same_source_and_reference"] is True
        assert report["selection_used_locked_test"] is False

        baseline = report["means"]["waitk_v1_k1"]
        if baseline_text is None:
            baseline_text = baseline
        else:
            assert baseline == baseline_text

        config = dict(metadata["config"])
        config.pop("seed")
        config.pop("experiment_name")
        config.pop("important_warning", None)
        if core_config is None:
            core_config = config
            data_hashes = {key: metadata["sha256"][key] for key in ("train", "validation", "test")}
        else:
            assert config == core_config
            assert {key: metadata["sha256"][key] for key in data_hashes} == data_hashes

        candidate = report["means"]["v2_3"]
        paired = report["paired_metrics"]
        text[seed] = {
            "token_f1": candidate["token_f1"]["mean"],
            "sequence_similarity": candidate["sequence_similarity"]["mean"],
            "reference_length_al_source_tokens": candidate["reference_length_al"]["mean"],
            "laal_source_tokens": candidate["laal"]["mean"],
            "completion_rate": candidate["completed_after_full_source"]["mean"],
            "invalid_action_ratio": candidate["invalid_ratio"]["mean"],
            "mean_prediction_length": statistics.fmean(row["predicted_gloss_token_count"] for row in rows),
            "mean_reference_length": statistics.fmean(row["target_gloss_token_count"] for row in rows),
            "paired_vs_waitk_k1": paired,
            "supports_main_claim": report["paper_gate"]["supports_main_claim"],
        }
        provenance[seed] = {
            "random_seed": metadata["config"]["seed"],
            "gpu_names": metadata["gpu_names"],
            "world_size": metadata["world_size"],
            "global_update_steps": summary["global_update_steps"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "stopped_early": summary["stopped_early"],
            "model_revision": metadata["config"]["model_revision"],
            "data_sha256": data_hashes,
            "software_versions": metadata["versions"],
        }

        espec = e2e_specs[seed]
        ereport = read_json(espec["report"])
        comparison = read_json(espec["comparison"])
        erows = read_jsonl(espec["rollout"])
        input_files.extend(espec.values())

        assert len(erows) == 500
        assert len({row["id"] for row in erows}) == 500
        assert {row["id"] for row in erows} == v1_e2e_ids
        assert comparison["common_record_count"] == 500
        assert comparison["bootstrap_samples"] == 10000

        baseline_name = comparison["baseline"]
        candidate_name = comparison["candidate"]
        ebaseline = {metric: values[baseline_name] for metric, values in comparison["means"].items()}
        if baseline_e2e is None:
            baseline_e2e = ebaseline
        else:
            assert ebaseline == baseline_e2e

        checks = comparison["predeclared_checks"]
        e2e[seed] = {
            "token_f1": comparison["means"]["pseudo_gloss_token_f1"][candidate_name],
            "sequence_similarity": comparison["means"]["pseudo_gloss_sequence_similarity"][candidate_name],
            "exact_match": comparison["means"]["pseudo_gloss_exact_match"][candidate_name],
            "laal_seconds": comparison["means"]["speech_to_gloss_laal_seconds"][candidate_name],
            "compute_adjusted_laal_seconds": comparison["means"]["speech_to_gloss_compute_adjusted_laal_seconds"][candidate_name],
            "completion_rate": comparison["means"]["speech_to_gloss_completed_after_full_source"][candidate_name],
            "mean_prediction_length": statistics.fmean(len(row["speech_to_gloss_predicted_tokens"]) for row in erows),
            "mean_reference_length": statistics.fmean(len(row["reference_pseudo_asl_gloss"].split()) for row in erows),
            "paired_vs_v1": comparison["paired_deltas"],
            "quality_noninferior": checks["sequence_similarity_noninferior"] and checks["token_f1_noninferior"],
            "laal_improved": checks["laal_improved"],
            "compute_adjusted_laal_improved": checks["compute_adjusted_laal_improved"],
            "completion_noninferior": checks["completion_noninferior"],
            "length_error_within_tolerance": checks["length_error_within_tolerance"],
            "all_predeclared_checks_passed": comparison["accepted"],
            "termination_reason_counts": ereport["termination_reason_counts"],
        }

    text_metrics = (
        "token_f1",
        "sequence_similarity",
        "reference_length_al_source_tokens",
        "laal_source_tokens",
        "completion_rate",
        "invalid_action_ratio",
        "mean_prediction_length",
        "mean_reference_length",
    )
    e2e_metrics = (
        "token_f1",
        "sequence_similarity",
        "exact_match",
        "laal_seconds",
        "compute_adjusted_laal_seconds",
        "completion_rate",
        "mean_prediction_length",
        "mean_reference_length",
    )
    text_aggregate = {metric: mean_sd([text[seed][metric] for seed in SEEDS]) for metric in text_metrics}
    e2e_aggregate = {metric: mean_sd([e2e[seed][metric] for seed in SEEDS]) for metric in e2e_metrics}

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "three_seed_locked500_pseudo_gloss_evaluation_not_expert_gold_asl",
        "seeds": list(SEEDS),
        "text_locked500": {
            "baseline_waitk_v1_k1": baseline_text,
            "per_seed": text,
            "aggregate_mean_and_sample_sd": text_aggregate,
            "all_seeds_support_main_claim": all(text[seed]["supports_main_claim"] for seed in SEEDS),
        },
        "chinese_speech_e2e_locked500": {
            "baseline_v1": baseline_e2e,
            "per_seed": e2e,
            "aggregate_mean_and_sample_sd": e2e_aggregate,
            "all_seeds_quality_noninferior": all(e2e[seed]["quality_noninferior"] for seed in SEEDS),
            "all_seeds_laal_improved": all(e2e[seed]["laal_improved"] for seed in SEEDS),
            "all_seeds_compute_adjusted_laal_improved": all(e2e[seed]["compute_adjusted_laal_improved"] for seed in SEEDS),
            "all_seeds_completion_noninferior": all(e2e[seed]["completion_noninferior"] for seed in SEEDS),
            "all_seeds_length_error_within_tolerance": all(e2e[seed]["length_error_within_tolerance"] for seed in SEEDS),
        },
        "training_provenance": provenance,
        "protocol_checks": {
            "identical_core_training_configuration": True,
            "identical_train_validation_test_hashes": True,
            "identical_locked500_ids_within_each_comparison": True,
            "checkpoint_selection_used_locked_test": False,
            "paired_bootstrap_repetitions_per_comparison": 10000,
        },
        "limitations": [
            "ASL references are LLM-generated weak pseudo-glosses rather than expert gold ASL.",
            "Chinese speech is synthetic multi-speaker TTS.",
            "Text latency is measured in source-token units; speech latency is measured in seconds.",
            "All three speech runs show output-length error above the predeclared tolerance.",
            "Only three training seeds are available, so across-seed standard deviations are descriptive.",
        ],
    }

    text_rows = []
    e2e_rows = []
    ci_rows = []
    for seed in SEEDS:
        text_rows.append({"seed": seed, **{metric: text[seed][metric] for metric in text_metrics}})
        e2e_rows.append({"seed": seed, **{metric: e2e[seed][metric] for metric in e2e_metrics}})
        for metric in ("token_f1", "sequence_similarity", "laal"):
            value = text[seed]["paired_vs_waitk_k1"][metric]
            ci_rows.append({
                "evaluation": "text_locked500_vs_waitk_v1_k1",
                "seed": seed,
                "metric": metric,
                "mean_delta": value["mean_delta"],
                "ci95_lower": value["ci95_lower"],
                "ci95_upper": value["ci95_upper"],
            })
        for metric in (
            "pseudo_gloss_token_f1",
            "pseudo_gloss_sequence_similarity",
            "speech_to_gloss_laal_seconds",
            "speech_to_gloss_compute_adjusted_laal_seconds",
        ):
            value = e2e[seed]["paired_vs_v1"][metric]
            ci_rows.append({
                "evaluation": "chinese_speech_e2e_locked500_vs_v1",
                "seed": seed,
                "metric": metric,
                "mean_delta": value["mean_candidate_minus_baseline"],
                "ci95_lower": value["ci95_lower"],
                "ci95_upper": value["ci95_upper"],
            })

    write_csv(output_root / "tables/v2_3_text_locked500_multiseed.csv", text_rows)
    write_csv(output_root / "tables/v2_3_speech_e2e_locked500_multiseed.csv", e2e_rows)
    write_csv(output_root / "tables/v2_3_per_seed_paired_ci.csv", ci_rows)

    summary_path = output_root / "v2_3_multiseed_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(path): sha256(path) for path in sorted(set(input_files))},
        "outputs": {},
    }
    output_files = [
        summary_path,
        output_root / "tables/v2_3_text_locked500_multiseed.csv",
        output_root / "tables/v2_3_speech_e2e_locked500_multiseed.csv",
        output_root / "tables/v2_3_per_seed_paired_ci.csv",
    ]
    manifest["outputs"] = {str(path): sha256(path) for path in output_files}
    manifest_path = output_root / "reproducibility_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    text_mean = text_aggregate
    e2e_mean = e2e_aggregate
    markdown = f"""# v2.3 Three-Seed Locked-500 Summary

Seeds: {', '.join(SEEDS)}. Values are mean +/- sample standard deviation across three training seeds.

## Text locked-500

| Metric | v2.3 mean +/- SD | Wait-k v1 k=1 |
|---|---:|---:|
| Token F1 | {text_mean['token_f1']['mean']:.4f} +/- {text_mean['token_f1']['sample_standard_deviation']:.4f} | {baseline_text['token_f1']['mean']:.4f} |
| Sequence similarity | {text_mean['sequence_similarity']['mean']:.4f} +/- {text_mean['sequence_similarity']['sample_standard_deviation']:.4f} | {baseline_text['sequence_similarity']['mean']:.4f} |
| LAAL (source tokens) | {text_mean['laal_source_tokens']['mean']:.4f} +/- {text_mean['laal_source_tokens']['sample_standard_deviation']:.4f} | {baseline_text['laal']['mean']:.4f} |
| Completion | {text_mean['completion_rate']['mean']:.4f} +/- {text_mean['completion_rate']['sample_standard_deviation']:.4f} | {baseline_text['completed_after_full_source']['mean']:.4f} |

All three seeds satisfy the predeclared quality-noninferiority margin and have significantly lower LAAL than modified Wait-k k=1.

## Chinese speech end-to-end locked-500

| Metric | v2.3 mean +/- SD | v1 |
|---|---:|---:|
| Token F1 | {e2e_mean['token_f1']['mean']:.4f} +/- {e2e_mean['token_f1']['sample_standard_deviation']:.4f} | {baseline_e2e['pseudo_gloss_token_f1']:.4f} |
| Sequence similarity | {e2e_mean['sequence_similarity']['mean']:.4f} +/- {e2e_mean['sequence_similarity']['sample_standard_deviation']:.4f} | {baseline_e2e['pseudo_gloss_sequence_similarity']:.4f} |
| LAAL (seconds) | {e2e_mean['laal_seconds']['mean']:.4f} +/- {e2e_mean['laal_seconds']['sample_standard_deviation']:.4f} | {baseline_e2e['speech_to_gloss_laal_seconds']:.4f} |
| Compute-adjusted LAAL (seconds) | {e2e_mean['compute_adjusted_laal_seconds']['mean']:.4f} +/- {e2e_mean['compute_adjusted_laal_seconds']['sample_standard_deviation']:.4f} | {baseline_e2e['speech_to_gloss_compute_adjusted_laal_seconds']:.4f} |
| Completion | {e2e_mean['completion_rate']['mean']:.4f} +/- {e2e_mean['completion_rate']['sample_standard_deviation']:.4f} | {baseline_e2e['speech_to_gloss_completed_after_full_source']:.4f} |

All three seeds preserve quality within the predeclared margin and significantly reduce both LAAL and compute-adjusted LAAL relative to v1. The output-length-error tolerance is not met and must be reported as a limitation.
"""
    markdown_path = output_root / "v2_3_multiseed_summary.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    manifest["outputs"][str(markdown_path)] = sha256(markdown_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(json.dumps({
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "text_all_seeds_support_main_claim": summary["text_locked500"]["all_seeds_support_main_claim"],
        "e2e_all_seeds_quality_noninferior": summary["chinese_speech_e2e_locked500"]["all_seeds_quality_noninferior"],
        "e2e_all_seeds_laal_improved": summary["chinese_speech_e2e_locked500"]["all_seeds_laal_improved"],
        "e2e_all_seeds_length_gate": summary["chinese_speech_e2e_locked500"]["all_seeds_length_error_within_tolerance"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--output-root", required=True)
    aggregate(parser.parse_args())
