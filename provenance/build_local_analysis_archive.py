"""Archive an explicit reviewed allowlist of project analysis source originals.

This script never executes the source files, touches remote hosts, reads raw data,
or searches credential files. Source copies preserve their D:/Infinisst hierarchy.
Run without --copy to inspect the candidate list and dependency evidence first.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


WORKSPACE = Path("D:/Infinisst")
RELEASE = WORKSPACE / "releases/CasualGloss"
ARCHIVE = RELEASE / "archive/original/workspace"
PROVENANCE = RELEASE / "provenance"
DATA = "InfiniSST/data/how2sign_english_asl_gloss_v1"
FULL = DATA + "/full_v4_train"
FRONT = DATA + "/chinese_speech_frontend_v0"

# Each entry was selected for manuscript statistics, figures, or a local helper.
GROUPS = {
    "final_transllama_statistics": [
        "aggregate_transllama_results.py",
        "bootstrap_transllama_vs_causalgloss.py",
    ],
    "manuscript_figure_statistics": [
        "icassp2027_paper/scripts/build_core_results.py",
        "paper_revision_20260916/scripts/analyze_fixed_waitk_multiseed.py",
        "paper_revision_20260914/scripts/analyze_fixed_waitk.py",
        "paper_revision_20260914/scripts/analyze_component_ablations.py",
        "paper_revision_20260919_transllama/build_transllama_figure.py",
    ],
    "local_original_analysis": [
        "analysis/compare_three_locked_rollouts_reference.py",
        "analysis/compare_pair_versions.py",
        "analysis/compare_epee_multi_reference_reference.py",
        "analysis/completed_only_epee_sensitivity.py",
        "analysis/inspect_v2_2_recovery.py",
        "deliverables/ICASSP_2027_completed_only_analysis_20260921/completed_only_epee_sensitivity.py",
    ],
    "multiseed_main_and_speech_statistics": [
        "experiment_launchers/v2_3_seed_20260905/aggregate_v2_3_multiseed.py",
    ],
    "locked500_bleu_statistics": [
        "experiment_launchers/locked500_bleu_20260911/evaluate_locked500_bleu.py",
    ],
    "clerc_final_adaptive_statistics": [
        "remote_experiment_staging/clerc_epee_adaptive_comparison_20260920/scripts/compare_epee_adaptive.py",
        "remote_experiment_staging/clerc_epee_adaptive_comparison_20260920/scripts/epee_common.py",
        "remote_experiment_staging/clerc_epee_adaptive_comparison_20260920/scripts/validate_rollouts.py",
        "remote_experiment_staging/clerc_epee_adaptive_comparison_20260920/scripts/score_smoke_gate.py",
    ],
    "clerc_final_fixed_statistics": [
        "remote_fixed_wait_epee/compare_fixed_waitk_epee.py",
        "remote_fixed_wait_epee/compare_epee_adaptive.py",
        "remote_fixed_wait_epee/epee_common.py",
        "remote_fixed_wait_epee/validate_rollouts.py",
    ],
    "s1_speech_statistics_and_figures": [
        FRONT + "/scripts/build_locked500_e2e_artifacts.py",
        FRONT + "/scripts/compare_causal_e2e_versions.py",
        FRONT + "/scripts/evaluate_streaming_frontend.py",
        FRONT + "/scripts/evaluate_zh_asr.py",
        FRONT + "/scripts/compare_task5_frontends.py",
    ],
    "original_paper_statistics_and_figures": [
        FULL + "/paper_experiments_20260903/scripts/build_paper_artifacts.py",
        FULL + "/paper_experiments_20260903/scripts/compare_v2_3_vs_waitk.py",
        FULL + "/paper_experiments_20260903/scripts/select_v2_3_cases.py",
        FULL + "/paper_experiments_20260903/scripts/build_reproducibility_manifest.py",
    ],
    "original_statistics_helpers": [
        FULL + "/causal_alignment_latency_audit_v2_2/scripts/audit_latency.py",
    ],
}

DENIED_COMPONENTS = {
    ".ssh", ".codex", ".codex_remote_keys", ".git", "vendor", "_deps",
    "third_party", "models", "__pycache__", "node_modules", "releases",
}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret|private[_-]?key)\s*[:=]\s*[\"']([^\"'\r\n]{8,})[\"']"),
    re.compile(r"(?i)\b(?:https?|ssh)://[^/\s:@]+:[^/\s@]+@"),
]
PLACEHOLDERS = re.compile(r"(?i)example|placeholder|your[_ -]|changeme|dummy|redacted|os\.environ|getenv|\$\{|<[^>]+>")


def source_path(relative: str) -> Path:
    path = (WORKSPACE / relative).resolve()
    path.relative_to(WORKSPACE.resolve())
    if any(part.lower() in DENIED_COMPONENTS or "g2p" in part.lower() for part in path.relative_to(WORKSPACE).parts):
        raise ValueError(f"denied source: {relative}")
    if path.suffix.lower() not in {".py", ".sh", ".ps1", ".r"}:
        raise ValueError(f"non-source file: {relative}")
    return path


def secret_lines(text: str) -> list[int]:
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        for pattern in SECRET_PATTERNS:
            match = pattern.search(line)
            if match and not PLACEHOLDERS.search(match.group(0)):
                found.append(number)
                break
    return found


def imports_of(text: str, filename: str) -> list[dict]:
    if not filename.endswith(".py"):
        return []
    tree = ast.parse(text, filename=filename)
    output = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                output.append({"module": alias.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom) and node.module:
            output.append({"module": node.module, "line": node.lineno})
    return sorted(output, key=lambda item: (item["line"], item["module"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    candidates = []
    blocked = []
    missing = []
    for category, relatives in GROUPS.items():
        for relative in relatives:
            path = source_path(relative)
            if not path.is_file():
                missing.append({"path": relative, "category": category})
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig")
            hits = secret_lines(text)
            if hits:
                # Do not emit matched values, excerpts, or content hashes.
                blocked.append({"path": relative, "lines": hits})
                continue
            candidates.append({
                "relative_source_path": relative,
                "path": "archive/original/workspace/" + relative,
                "source": path.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "category": category,
                "imports": imports_of(text, relative),
                "external_helper_references": [
                    {"line": i, "kind": "Nature_skill_reference", "symbol": "audit_panel_alignment"}
                    for i, line in enumerate(text.splitlines(), 1)
                    if "audit_panel_alignment" in line or "nature-figure" in line or "require_matplotlib_panel_alignment" in line
                ],
                "path_portability_references": [
                    {"line": i, "kind": "absolute_path_or_personal_package_directory"}
                    for i, line in enumerate(text.splitlines(), 1)
                    if any(token in line for token in ["/123090274/", "D:/Infinisst", "D:\\\\Infinisst", "C:\\Users\\DELL", ".paper_python_packages"])
                ],
            })

    by_relative = {item["relative_source_path"]: item for item in candidates}
    module_paths = {}
    for item in candidates:
        module_paths.setdefault(Path(item["relative_source_path"]).stem, []).append(item["relative_source_path"])
    first_party_edges = []
    external_imports = Counter()
    for item in candidates:
        here = Path(item["relative_source_path"]).parent
        for imported in item["imports"]:
            top = imported["module"].split(".")[0]
            sibling = (here / (top + ".py")).as_posix()
            choices = [sibling] if sibling in by_relative else module_paths.get(top, [])
            if choices:
                first_party_edges.append({"importer": item["relative_source_path"], "line": imported["line"], "module": imported["module"], "archived_candidates": choices, "resolution": "sibling" if sibling in by_relative else "explicit_sys_path_requires_original_hierarchy"})
            elif top not in sys.stdlib_module_names and top != "__future__":
                external_imports[top] += 1

    plan = {
        "scope": "local project statistics, figure sources, original analysis, and explicit first-party helpers only",
        "source_workspace": WORKSPACE.as_posix(),
        "archive_root": ARCHIVE.as_posix(),
        "mode": "copy" if args.copy else "review_only",
        "source_files_executed": False,
        "third_party_libraries_copied": False,
        "raw_data_or_media_copied": False,
        "credential_scan": "selected source content pattern scan; credential files excluded; findings only path and line",
        "blocked_credential_candidates": blocked,
        "missing_allowlist_sources": missing,
        "file_count": len(candidates),
        "total_bytes": sum(item["size"] for item in candidates),
        "categories": dict(Counter(item["category"] for item in candidates)),
        "external_imports_not_archived": dict(sorted(external_imports.items())),
        "first_party_import_edges": first_party_edges,
        "local_first_party_helpers_included": [
            {"module": "audit_latency", "path": FULL + "/causal_alignment_latency_audit_v2_2/scripts/audit_latency.py", "symbol": "rollout_rows", "definition_line": 123, "dependencies": "Python_standard_library"},
            {"module": "epee_common", "paths": ["remote_fixed_wait_epee/epee_common.py", "remote_experiment_staging/clerc_epee_adaptive_comparison_20260920/scripts/epee_common.py"], "purpose": "CLERC_manifest_loading_hashes_normalization_and_numeric_summary"},
            {"module": "compare_epee_adaptive", "path": "remote_fixed_wait_epee/compare_epee_adaptive.py", "purpose": "shared_CLERC_scoring_and_bootstrap_used_by_fixed_comparator"},
        ],
        "embedded_first_party_plot_helpers": [
            {"path": "icassp2027_paper/scripts/build_core_results.py", "symbols": ["configure_style", "add_panel_label", "draw_forest"]},
            {"path": "paper_revision_20260919_transllama/build_transllama_figure.py", "symbols": ["configure_style", "draw_delta_panel", "add_panel_label"]},
        ],
        "entries": candidates,
        "closure_gaps": [
            {"kind": "external_helper_not_copied", "module": "audit_panel_alignment", "symbol": "require_matplotlib_panel_alignment", "reason": "External Nature skill library is explicitly excluded; no verified project-owned equivalent has been substituted."},
            {"kind": "incorrect_existing_import_path", "importer": "analysis/compare_three_locked_rollouts_reference.py", "line": 19, "resolved_original_local_path": "D:/causal_alignment_latency_audit_v2_2/scripts", "helper_present_in_archive": True, "reason": "Original parents[2] path calculation does not reach the real local helper. Source was preserved without modifying this historical path."},
            {"kind": "portability", "reason": "Archived originals retain hardcoded remote and personal package paths. This is source preservation, not a claim that unchanged scripts run from the archive directory."},
            {"kind": "data_inputs_out_of_scope", "reason": "No JSON/JSONL/TSV source data were copied by this source-only archiver. Existing result artifacts and exact input retrieval are a separate release task."},
        ],
    }
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    review_path = PROVENANCE / "local_analysis_source_review.json"
    if not args.copy:
        review_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for item in candidates:
            print(json.dumps({"path": item["relative_source_path"], "size": item["size"], "category": item["category"]}, ensure_ascii=True))
    else:
        # Refuse unreviewed selections and any change since the review step.
        reviewed = json.loads(review_path.read_text(encoding="utf-8"))
        expected = {(x["relative_source_path"], x["sha256"]) for x in reviewed["entries"]}
        observed = {(x["relative_source_path"], x["sha256"]) for x in candidates}
        if expected != observed:
            raise SystemExit("candidate selection or source content changed after review")
        for item in candidates:
            destination = (RELEASE / item["path"]).resolve()
            destination.relative_to(ARCHIVE.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                    raise SystemExit(f"refusing to overwrite differing archive file: {item['path']}")
            else:
                shutil.copy2(item["source"], destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                raise SystemExit(f"copy verification failed: {item['path']}")
            item["copy_verified"] = True
        (PROVENANCE / "local_analysis_source_manifest.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"mode": plan["mode"], "file_count": plan["file_count"], "total_bytes": plan["total_bytes"], "blocked": blocked, "missing": missing, "external_imports_not_archived": plan["external_imports_not_archived"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
