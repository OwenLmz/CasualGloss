#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from transllama_data import detokenize_source, file_sha256, read_jsonl, write_json, write_jsonl
from transllama_metrics import finalize_state, summarize_rows
from transllama_training import set_seed, wait_token_ids


GLOSS_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9'/_-]*$")
ALLOWED_PAIR_FIELDS = {
    "id",
    "split",
    "source_text_en",
    "source_tokens",
    "target_pseudo_asl_gloss",
    "target_pseudo_asl_gloss_tokens",
}


def parse_k_values(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("k values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("1,2,3,4,5,6,7,8"))
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--action-max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-locked-test", action="store_true")
    return parser.parse_args()


def native_context_window(model: Any, tokenizer: Any) -> int:
    """Return the fixed model/tokenizer inference context capacity."""
    candidates: list[int] = []
    for value in (
        getattr(model.config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 100_000_000:
            candidates.append(value)
    if not candidates:
        raise ValueError("unable to determine the native model context window")
    return min(candidates)


def find_repeating_patterns(value: str, min_length: int = 3) -> dict[str, int]:
    """Mirror the official TransLLaMA repeated-pattern detector."""
    pattern_counts: dict[str, int] = {}
    for length in range(min_length, len(value) // 2 + 1):
        for match in re.finditer(r"(?=(.{%d,}?)\1)" % length, value):
            pattern = match.group(1)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    return pattern_counts


def check_for_loop(value: str, max_repeated_chars: int = 6) -> bool:
    if re.search(r"(.)\1{%d,}" % (max_repeated_chars - 1), value):
        return True
    patterns = find_repeating_patterns(value)
    return any(len(pattern) >= 1 and count > 5 for pattern, count in patterns.items())


def parse_generated_action(
    tokenizer: Any,
    generated_ids: list[int],
    wait_ids: set[int],
    pending_ids: list[int] | None = None,
) -> tuple[str, str, str, list[int], list[int], bool]:
    """Parse one external action while retaining incomplete target words.

    TransLLaMA generates model tokens internally but exposes complete words to
    SimulEval.  A WAIT may occur after only part of a word (for example ``WE``
    before ``'RE``); those model tokens must remain in the hidden target
    history until a later token establishes the word boundary.
    """
    eos_id = int(tokenizer.eos_token_id)
    pending = list(pending_ids or [])
    if not generated_ids:
        return "<INVALID>", "INVALID", "", [], [], False
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(generated_ids)
    accepted_ids: list[int] = []
    for token_id, piece in zip(generated_ids, pieces):
        token_id = int(token_id)
        candidate_ids = pending + accepted_ids
        if token_id in wait_ids:
            return "@@", "WAIT", raw, accepted_ids, candidate_ids, False
        if token_id == eos_id:
            if not candidate_ids:
                return "<END>", "END", raw, accepted_ids, [], False
            action = tokenizer.decode(
                candidate_ids, skip_special_tokens=True
            ).strip().upper()
            if GLOSS_PATTERN.fullmatch(action):
                return action, "WRITE", raw, accepted_ids, [], True
            return action or "<INVALID>", "INVALID", raw, [], [], False
        starts_new_word = str(piece).startswith("Ġ") or str(piece).startswith("Ċ")
        if candidate_ids and starts_new_word:
            action = tokenizer.decode(
                candidate_ids, skip_special_tokens=True
            ).strip().upper()
            if GLOSS_PATTERN.fullmatch(action):
                return action, "WRITE", raw, accepted_ids, [], False
            return action or "<INVALID>", "INVALID", raw, [], [], False
        accepted_ids.append(token_id)

    candidate_ids = pending + accepted_ids
    return "<CONTINUE>", "CONTINUE", raw, accepted_ids, candidate_ids, False


def initialize_state(pair: dict[str, Any], wait_k: int) -> dict[str, Any]:
    source_tokens = list(pair["source_tokens"])
    target_tokens = list(pair["target_pseudo_asl_gloss_tokens"])
    return {
        "id": str(pair["id"]),
        "source_text_en": str(pair["source_text_en"]),
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "wait_k": wait_k,
        "prefix_length": min(wait_k, len(source_tokens)),
        "committed": [],
        "target_token_ids": [],
        "pending_token_ids": [],
        "write_positions": [],
        "events": [],
        "done": False,
        "termination_reason": None,
        "max_actions": len(source_tokens) + max(20, 3 * len(target_tokens) + 10),
    }


def prompt_text(tokenizer: Any, system_prompt: str, state: dict[str, Any]) -> str:
    source_prefix = detokenize_source(
        state["source_tokens"][: state["prefix_length"]]
    )
    base = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source_prefix},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    target_history = tokenizer.decode(
        state["target_token_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return base + target_history


@torch.inference_mode()
def generate_actions(
    model: Any,
    tokenizer: Any,
    states: list[dict[str, Any]],
    system_prompt: str,
    wait_ids: set[int],
    batch_size: int,
    max_new_tokens: int,
    context_window: int,
    context_stats: dict[str, int],
    device: torch.device,
) -> list[tuple[dict[str, Any], str, str, str, list[int], list[int], bool]]:
    results: list[
        tuple[dict[str, Any], str, str, str, list[int], list[int], bool]
    ] = []
    for source_complete in (False, True):
        group = [
            state
            for state in states
            if (state["prefix_length"] >= len(state["source_tokens"])) == source_complete
        ]
        suppress_tokens = sorted(wait_ids) if source_complete else [int(tokenizer.eos_token_id)]
        for start in range(0, len(group), batch_size):
            batch_states = group[start : start + batch_size]
            prompts = [prompt_text(tokenizer, system_prompt, state) for state in batch_states]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)
            prompt_length = int(encoded["input_ids"].shape[1])
            context_stats["max_observed_prompt_tokens"] = max(
                context_stats["max_observed_prompt_tokens"], prompt_length
            )
            if prompt_length + max_new_tokens > context_window:
                raise ValueError(
                    "rollout prompt plus action exceeds the native model context window: "
                    f"prompt={prompt_length}, action={max_new_tokens}, window={context_window}"
                )
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=int(tokenizer.eos_token_id),
                eos_token_id=int(tokenizer.eos_token_id),
                suppress_tokens=suppress_tokens,
            )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            for state, token_row in zip(batch_states, new_tokens.tolist()):
                action, action_type, raw, accepted_ids, pending_ids, finish = (
                    parse_generated_action(
                        tokenizer,
                        token_row,
                        wait_ids,
                        state["pending_token_ids"],
                    )
                )
                results.append(
                    (
                        state,
                        action,
                        action_type,
                        raw,
                        accepted_ids,
                        pending_ids,
                        finish,
                    )
                )
    return results


def apply_action(
    state: dict[str, Any],
    action: str,
    action_type: str,
    raw: str,
    accepted_ids: list[int],
    pending_ids: list[int],
    finish_after_write: bool,
    tokenizer: Any,
) -> None:
    source_count = len(state["source_tokens"])
    prefix_before = state["prefix_length"]
    pending_count_before = len(state["pending_token_ids"])
    if action_type != "INVALID":
        state["target_token_ids"].extend(accepted_ids)
        state["pending_token_ids"] = list(pending_ids)
    event = {
        "action_index": len(state["events"]),
        "source_prefix_length": prefix_before,
        "source_complete": prefix_before >= source_count,
        "committed_gloss_before": " ".join(state["committed"]),
        "raw_generation": raw,
        "predicted_action": action,
        "predicted_action_class": action_type,
        "pending_target_token_count_before": pending_count_before,
        "pending_target_token_count_after": len(state["pending_token_ids"]),
        "source_advanced_after_action": False,
    }
    if action_type == "CONTINUE":
        target_text = tokenizer.decode(
            state["target_token_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not check_for_loop(target_text, max_repeated_chars=6):
            return
        event["predicted_action"] = "<LOOP>"
        event["predicted_action_class"] = "LOOP"
        state["termination_reason"] = "loop_detected"
        state["done"] = True
        event["source_prefix_length_after"] = state["prefix_length"]
        event["committed_gloss_after"] = " ".join(state["committed"])
        state["events"].append(event)
        return
    if action_type == "END":
        state["termination_reason"] = (
            "end_after_full_source" if prefix_before >= source_count else "premature_end"
        )
        state["done"] = True
    elif action_type == "WAIT":
        if prefix_before < source_count:
            state["prefix_length"] += 1
            event["source_advanced_after_action"] = True
    elif action_type == "WRITE":
        state["committed"].append(action)
        state["write_positions"].append(prefix_before)
        if finish_after_write:
            state["termination_reason"] = "end_after_full_source"
            state["done"] = True
    else:
        if prefix_before < source_count:
            state["prefix_length"] += 1
            event["source_advanced_after_action"] = True
    event["source_prefix_length_after"] = state["prefix_length"]
    event["committed_gloss_after"] = " ".join(state["committed"])
    state["events"].append(event)
    if not state["done"] and len(state["events"]) >= state["max_actions"]:
        state["termination_reason"] = "maximum_actions_reached"
        state["done"] = True


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_locked_test:
        raise SystemExit("locked test evaluation requires --allow-locked-test")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    pairs = []
    for raw in read_jsonl(args.pairs):
        pair = {field: raw.get(field) for field in ALLOWED_PAIR_FIELDS}
        if pair["split"] != args.split:
            raise ValueError(f"unexpected split for {pair['id']}")
        pairs.append(pair)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    waits = wait_token_ids(tokenizer)
    device = torch.device("cuda", 0)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    context_window = native_context_window(base_model, tokenizer)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True

    states = [
        initialize_state(pair, wait_k)
        for wait_k in args.k_values
        for pair in pairs
    ]
    start_time = perf_counter()
    cycle = 0
    context_stats = {"max_observed_prompt_tokens": 0}
    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        results = generate_actions(
            model,
            tokenizer,
            active,
            config["system_prompt"],
            waits,
            args.generation_batch_size,
            args.action_max_new_tokens,
            context_window,
            context_stats,
            device,
        )
        for (
            state,
            action,
            action_type,
            raw,
            accepted_ids,
            pending_ids,
            finish_after_write,
        ) in results:
            apply_action(
                state,
                action,
                action_type,
                raw,
                accepted_ids,
                pending_ids,
                finish_after_write,
                tokenizer,
            )
        cycle += 1
        if cycle == 1 or cycle % 10 == 0:
            print(
                f"[rollout] cycle={cycle} completed={sum(state['done'] for state in states)}/{len(states)}",
                flush=True,
            )

    elapsed = perf_counter() - start_time
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sweep = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "independently_trained_transllama_style_rollout_not_gold_asl",
        "stage": args.stage,
        "split": args.split,
        "adapter_dir": str(args.adapter_dir),
        "k_values": args.k_values,
        "parent_count_per_k": len(pairs),
        "rollout_history_uses_gold_committed_gloss": False,
        "test_manifest_accessed": args.split == "test",
        "generation": {
            "decoding": "greedy",
            "normal_gloss": "buffer model subtokens and WRITE one complete gloss without reading source",
            "wait": "READ exactly one source token without committing WAIT",
            "eos_before_source_end": "suppressed",
            "wait_after_source_end": "suppressed",
            "modified_wait_k": "minimum initial read only",
            "invalid_action": "record and read one token before source end as finite-progress safeguard",
            "training_sequence_max_length": int(config["max_length"]),
            "native_inference_context_window": context_window,
            "max_observed_prompt_tokens": context_stats["max_observed_prompt_tokens"],
            "context_policy": "use the fixed base model native context capacity at inference",
            "loop_safeguard": "official TransLLaMA repeated-pattern check after each internal target-token extension",
        },
        "sha256": {
            "config": file_sha256(args.config),
            "pairs": file_sha256(args.pairs),
        },
        "elapsed_seconds": round(elapsed, 3),
        "results": [],
    }
    for wait_k in args.k_values:
        rows = [
            finalize_state(state, f"{args.stage}_k{wait_k}")
            for state in states
            if state["wait_k"] == wait_k
        ]
        output = args.output_dir / f"k{wait_k}"
        write_jsonl(output / "sentence_rollouts.jsonl", rows)
        metrics = summarize_rows(rows)
        metrics.update(
            {
                "stage": f"{args.stage}_k{wait_k}",
                "wait_k": wait_k,
                "adapter_dir": str(args.adapter_dir),
            }
        )
        write_json(output / "rollout_metrics.json", metrics)
        sweep["results"].append(metrics)
    write_json(args.output_dir / "sweep_summary.json", sweep)
    print(json.dumps(sweep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
