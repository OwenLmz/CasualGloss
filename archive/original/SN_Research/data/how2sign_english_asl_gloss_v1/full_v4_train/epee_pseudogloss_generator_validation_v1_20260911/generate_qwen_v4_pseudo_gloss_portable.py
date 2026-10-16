#!/usr/bin/env python3
"""Generate deterministic text-only pseudo glosses with batching and resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GLOSS_RE = re.compile(r"^[A-Z0-9][A-Z0-9'/-]*(?: [A-Z0-9][A-Z0-9'/-]*)*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output is not a JSON object")
    return value


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    original_gloss = " ".join(str(result.get("gloss", "")).strip().split())
    separator_normalized = original_gloss.replace("_", "-")
    gloss = separator_normalized.upper()
    normalization_flags: list[str] = []
    if separator_normalized != original_gloss:
        normalization_flags.append("model_output_underscore_normalized")
    if gloss != separator_normalized:
        normalization_flags.append("model_output_case_normalized")
    omissions = result.get("potential_omissions", [])
    return {
        "gloss": gloss,
        "original_gloss": original_gloss,
        "normalization_flags": normalization_flags,
        "uncertain": bool(result.get("uncertain", False)),
        "note": str(result.get("note", "")).strip(),
        "source_issue": str(result.get("source_issue", "")).strip(),
        "potential_omissions": [
            str(value).strip() for value in omissions if str(value).strip()
        ]
        if isinstance(omissions, list)
        else [],
    }


def write_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "sample_index",
        "id",
        "source_text_en",
        "target_pseudo_asl_gloss",
        "generation_status",
        "generator_uncertain",
        "source_issue",
        "generator_note",
        "generator_potential_omissions",
        "quality_flags",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        values = [
            row.get("sample_index", ""),
            row.get("id", ""),
            row.get("source_text_en", ""),
            row.get("target_pseudo_asl_gloss", ""),
            row.get("generation_status", ""),
            row.get("generator_uncertain", ""),
            row.get("source_issue", ""),
            row.get("generator_note", ""),
            json.dumps(row.get("generator_potential_omissions", []), ensure_ascii=False),
            json.dumps(row.get("quality_flags", []), ensure_ascii=False),
        ]
        lines.append("\t".join(str(value).replace("\t", " ").replace("\n", " ") for value in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--preview-tsv", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--revision", default="a09a35458c702b33eeacc393d103063234e8bc28"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Execution dtype. Use float16 for V100 replication runs.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    import transformers
    import transformers.utils.import_utils as transformers_import_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # This is a text-only experiment. Some V100 system images expose an
    # incompatible optional torchvision/onnx stack; keep it out of Qwen loading.
    transformers_import_utils._torchvision_available = False

    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("Batch size and max-new-tokens must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA was requested but is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    requests = read_jsonl(args.requests_jsonl)
    requests.sort(key=lambda row: int(row["request_index"]))
    if args.limit:
        requests = requests[: args.limit]
    request_ids = [str(row["request_id"]) for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise SystemExit("Duplicate request IDs")
    english_keys = [str(row["normalized_english_key"]) for row in requests]
    if len(english_keys) != len(set(english_keys)):
        raise SystemExit("Generation requests are not normalized-English unique")

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_sha256 = sha256_text(prompt)
    existing = {str(row["request_id"]): row for row in read_jsonl(args.output_jsonl)}
    unexpected = set(existing).difference(request_ids)
    if unexpected:
        raise SystemExit(f"Output contains unexpected request IDs: {sorted(unexpected)[:5]}")
    pending = [row for row in requests if str(row["request_id"]) not in existing]
    print(
        f"[resume] requested={len(requests)} existing={len(existing)} pending={len(pending)}",
        flush=True,
    )

    if pending:
        print(f"[setup] tokenizer {args.model_id}@{args.revision}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, revision=args.revision, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[setup] model on {args.device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": args.device},
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.eval()
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.output_jsonl.exists() and args.output_jsonl.stat().st_size else "w"
        with args.output_jsonl.open(mode, encoding="utf-8") as output_handle:
            for start in range(0, len(pending), args.batch_size):
                batch_rows = pending[start : start + args.batch_size]
                rendered_prompts: list[str] = []
                for row in batch_rows:
                    messages = [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"current_sentence": row["source_text_en"]},
                                ensure_ascii=False,
                            ),
                        },
                    ]
                    rendered_prompts.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                encoded = tokenizer(
                    rendered_prompts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(args.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated_only = generated[:, encoded["input_ids"].shape[1] :]
                raw_outputs = tokenizer.batch_decode(
                    generated_only, skip_special_tokens=True
                )
                generated_at = datetime.now(timezone.utc).isoformat()
                for row, raw_output in zip(batch_rows, raw_outputs):
                    raw_output = raw_output.strip()
                    errors: list[str] = []
                    normalized = {
                        "gloss": "",
                        "original_gloss": "",
                        "normalization_flags": [],
                        "uncertain": True,
                        "note": "",
                        "source_issue": "",
                        "potential_omissions": [],
                    }
                    status = "completed"
                    try:
                        normalized = normalize_result(parse_json_object(raw_output))
                        if not normalized["gloss"]:
                            errors.append("empty_gloss")
                        elif not GLOSS_RE.fullmatch(normalized["gloss"]):
                            errors.append("invalid_gloss_token_format")
                    except (json.JSONDecodeError, ValueError) as exc:
                        status = "parse_failed"
                        errors.append(f"json_parse_error:{type(exc).__name__}")
                    if errors and status == "completed":
                        status = "completed_with_quality_flags"
                    result = {
                        **row,
                        "target_pseudo_asl_gloss": normalized["gloss"],
                        "target_pseudo_asl_gloss_tokens": normalized["gloss"].split(),
                        "model_gloss_before_normalization": normalized["original_gloss"],
                        "pseudo_gloss_is_gold": False,
                        "generator_uncertain": normalized["uncertain"],
                        "generator_note": normalized["note"],
                        "source_issue": normalized["source_issue"],
                        "generator_potential_omissions": normalized["potential_omissions"],
                        "generation_status": status,
                        "quality_flags": normalized["normalization_flags"]
                        + errors
                        + ["manual_linguistic_review_required"],
                        "manual_review_status": "pending",
                        "generation": {
                            "method": "local_transformers_batched_greedy_decoding",
                            "model_id": args.model_id,
                            "requested_revision": args.revision,
                            "resolved_revision": resolved_revision,
                            "prompt_file": str(args.prompt_file),
                            "prompt_sha256": prompt_sha256,
                            "context_sentences": 0,
                            "do_sample": False,
                            "temperature": None,
                            "top_p": None,
                            "seed": args.seed,
                            "max_new_tokens": args.max_new_tokens,
                            "batch_size": args.batch_size,
                            "dtype": args.dtype,
                            "device": args.device,
                            "torch_version": torch.__version__,
                            "transformers_version": transformers.__version__,
                            "generated_at_utc": generated_at,
                        },
                        "raw_model_output": raw_output,
                    }
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    existing[str(row["request_id"])] = result
                completed = min(start + len(batch_rows), len(pending))
                print(
                    f"[generate] new={completed}/{len(pending)} total={len(existing)}/{len(requests)}",
                    flush=True,
                )

    ordered = [existing[request_id] for request_id in request_ids]
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)
    write_preview(args.preview_tsv, ordered)
    print(f"[done] {len(ordered)} records: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
