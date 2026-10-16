#!/usr/bin/env python3
"""Recompute paper results from private per-example outputs, never from summaries.

Dependencies: Python >=3.10, numpy (only bootstrap). No GPU/model imports.
All output is aggregate numeric data, file checksums, and protocol descriptions.
The inputs contain source/reference text and must remain outside the repository.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st
import sys

SEEDS = (20260829, 20260904, 20260905)
FULL = Path("full_v4_train")
EPEE = FULL / "clerc_epee_adaptive_comparison_20260920"
SPEECH = Path("chinese_speech_frontend_v0/paper_e2e_locked500_20260904")
RECOVERY = FULL / "state_recovery_v2_3_full"

def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def index(rows, n=None):
    out = {str(r["id"]): r for r in rows}
    if len(out) != len(rows) or (n is not None and len(rows) != n):
        raise ValueError(f"Expected {n} unique examples; got {len(rows)} rows / {len(out)} IDs")
    return out

def overlap(reference, prediction):
    hits = sum((Counter(reference) & Counter(prediction)).values())
    p, r = hits / max(1, len(prediction)), hits / max(1, len(reference))
    return 2*p*r/(p+r) if p+r else 0.0

def laal(delays, source_length, reference_length):
    if not delays or source_length <= 0 or reference_length <= 0:
        return None
    ratio = max(reference_length, len(delays)) / source_length
    tau = next((i for i, d in enumerate(delays, 1) if d >= source_length), len(delays))
    return round(st.fmean(delays[i] - i/ratio for i in range(tau)), 8)

def normalize_content(tokens):
    normalized = []
    for token in tokens:
        token = " ".join(str(token).upper().replace("’", "'").split()).rstrip("?")
        if token and token != "QUESTION":
            normalized.append(re.sub(r"_\d+$", "", token))
    for period in range(1, len(normalized)//2+1):
        if len(normalized) % period == 0 and normalized == normalized[:period] * (len(normalized)//period):
            return normalized[:period]
    return normalized

def corpus_bleu(rows):
    matches, totals = [0]*4, [0]*4
    hyp_len = ref_len = 0
    for r in rows:
        pred, ref = r["predicted_pseudo_asl_gloss_tokens"], r["target_pseudo_asl_gloss_tokens"]
        hyp_len += len(pred); ref_len += len(ref)
        for n in range(1, 5):
            pc = Counter(tuple(pred[i:i+n]) for i in range(len(pred)-n+1))
            rc = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            matches[n-1] += sum((pc & rc).values()); totals[n-1] += sum(pc.values())
    precisions = [m/t if t else 0 for m,t in zip(matches, totals)]
    if any(p <= 0 for p in precisions):
        return 0.0
    bp = min(1.0, math.exp(1-ref_len/max(1,hyp_len)))
    return 100*bp*math.exp(st.fmean(math.log(p) for p in precisions))

def finite_mean(values):
    vals = [float(x) for x in values if x is not None]
    return st.fmean(vals) if vals else None

def summarize(rows):
    return {name: finite_mean(r[name] for r in rows) for name in ("token_f1", "sequence_similarity", "laal", "completion")}

def aggregate(summaries):
    return {k: {"mean": st.fmean(s[k] for s in summaries),
                "sample_sd": st.stdev(s[k] for s in summaries) if len(summaries)>1 else None}
            for k in summaries[0] if all(s[k] is not None for s in summaries)}

class Verifier:
    def __init__(self, root, locked_root, repetitions):
        self.root, self.locked_root, self.repetitions = root, locked_root, repetitions
        self.inputs, self.discrepancies, self.checks = [], [], []

    def load(self, relative, n, locked=False):
        path = (self.locked_root if locked else self.root) / relative
        rows = read_jsonl(path)
        indexed = index(rows, n)
        self.inputs.append({"relative_path": ("locked500_from_local/" if locked else "") + str(relative).replace("\\", "/"),
                            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "rows": len(rows), "unique_ids": len(indexed)})
        return indexed

    def compare(self, label, actual, expected, tolerance):
        ok = ((actual is None and expected is None) or
              (actual is not None and expected is not None and abs(actual-expected) <= tolerance))
        if not ok:
            self.discrepancies.append({"check": label, "actual": actual, "expected": expected, "tolerance": tolerance})
        return ok

    def score_text(self, raw, label):
        scored = {}
        for i, (key, r) in enumerate(raw.items()):
            ref, pred = r["target_pseudo_asl_gloss_tokens"], r["predicted_pseudo_asl_gloss_tokens"]
            m = {"token_f1": round(overlap(ref, pred), 8),
                 "sequence_similarity": round(SequenceMatcher(None, ref, pred, autojunk=False).ratio(), 8),
                 "laal": laal(r["write_source_positions"], len(r["source_tokens"]), len(ref)),
                 "completion": float(r["completed_after_full_source"])}
            for field, actual in (("pseudo_gloss_token_f1", m["token_f1"]),
                                  ("pseudo_gloss_sequence_similarity", m["sequence_similarity"]),
                                  ("length_adaptive_average_lagging_source_tokens", m["laal"])):
                self.compare(f"{label}:row{i}:{field}", actual, r[field], 1e-7)
            scored[key] = m
        return scored

    def pair_check(self, left, right, raw=False):
        if set(left) != set(right):
            raise ValueError("Unpaired IDs; refusing comparison")
        if raw and any(left[k]["source_tokens"] != right[k]["source_tokens"] or
                       left[k]["target_pseudo_asl_gloss_tokens"] != right[k]["target_pseudo_asl_gloss_tokens"]
                       for k in left):
            raise ValueError("Source/reference token mismatch; refusing comparison")

    def paired_differences(self, left, right, metric):
        self.pair_check(left, right)
        return [left[k][metric]-right[k][metric] for k in left
                if left[k][metric] is not None and right[k][metric] is not None]

    def bootstrap(self, differences, seed, hierarchical=True, vector_batch=False, repetitions=None):
        import numpy as np
        draws = repetitions or self.repetitions
        rng = np.random.default_rng(seed)
        arrays = [np.asarray(v, dtype=float) for v in differences]
        samples = np.empty(draws)
        if vector_batch:
            matrix = np.asarray(arrays)
            for start in range(0, draws, 500):
                batch = min(500, draws-start)
                ss = rng.integers(0, len(arrays), size=(batch, len(arrays)))
                pp = rng.integers(0, matrix.shape[1], size=(batch, len(arrays), matrix.shape[1]))
                samples[start:start+batch] = np.nanmean(matrix[ss[:,:,None], pp], axis=(1,2))
        else:
            for i in range(draws):
                selected = rng.integers(0, len(arrays), len(arrays)) if hierarchical else [0]
                values = []
                for s in selected:
                    v = arrays[int(s)]; ix = rng.integers(0, len(v), len(v))
                    values.append(float(v[ix].mean()))
                samples[i] = st.fmean(values)
        point = float(np.nanmean(np.asarray(arrays))) if vector_batch else st.fmean(st.fmean(v) for v in arrays)
        return {"mean_difference": point,
                "ci95": [float(v) for v in np.quantile(samples, [.025,.975])],
                "repetitions": draws, "random_seed": seed,
                "paired_counts": [int(np.isfinite(v).sum()) for v in arrays],
                "design": "matched-seed then paired-parent" if hierarchical else "paired-parent"}

    def independent_bootstrap(self,left,right,metric,seed,repetitions):
        import numpy as np
        ids=[key for key in left[0] if all(r[key][metric] is not None for r in left+right)]
        a=np.asarray([[r[key][metric] for key in ids] for r in left],dtype=float)
        b=np.asarray([[r[key][metric] for key in ids] for r in right],dtype=float)
        rng=np.random.default_rng(seed); draws=np.empty(repetitions)
        for i in range(repetitions):
            pp=rng.integers(0,len(ids),len(ids))
            ss_a=rng.integers(0,len(left),len(left)); ss_b=rng.integers(0,len(right),len(right))
            draws[i]=a[ss_a][:,pp].mean()-b[ss_b][:,pp].mean()
        return {"mean_difference":float(a.mean()-b.mean()),"ci95":[float(x) for x in np.quantile(draws,[.025,.975])],
                "repetitions":repetitions,"random_seed":seed,"paired_parent_count":len(ids),
                "left_seed_count":len(left),"right_seed_count":len(right),"design":"independent system seeds, shared paired-parent draw"}

    def locked500(self, expected):
        raw_cg = [self.load(Path(f"cg_s{i}.jsonl"), 500, True) for i in range(1,4)]
        scored_cg = [self.score_text(r, f"locked_cg_{s}") for r,s in zip(raw_cg,SEEDS)]
        cg_sums = [summarize(list(s.values())) | {"bleu4": corpus_bleu(list(r.values()))} for s,r in zip(scored_cg,raw_cg)]
        table1 = {"causalgloss": aggregate(cg_sums)}
        fixed = {}; fixed_raw = {}
        for k in (1,3,5,7):
            suffixes = ("",) if k in (1,3) else ("", "_seed20260904", "_seed20260905")
            fixed_raw[k] = [self.load(Path(f"waitk{k}{s}.jsonl"), 500, True) for s in suffixes]
            fixed[k] = [self.score_text(r, f"fixed_k{k}_{i}") for i,r in enumerate(fixed_raw[k])]
            for r in fixed_raw[k]: self.pair_check(raw_cg[0],r,True)
        fs = [summarize(list(s.values())) | {"bleu4": corpus_bleu(list(r.values()))} for s,r in zip(fixed[7],fixed_raw[7])]
        table1["fixed_k7"] = aggregate(fs)
        learned = {}
        learned_sums = {}
        for k in range(1,9):
            rr = [self.load(FULL / f"transllama_learned_adaptive_20260918/runs/final_test/final_seed_{seed}/k{k}/sentence_rollouts.jsonl",500) for seed in SEEDS]
            for r in rr: self.pair_check(raw_cg[0],r,True)
            learned[k] = [self.score_text(r, f"transllama_{seed}_k{k}") for seed,r in zip(SEEDS,rr)]
            learned_sums[k] = aggregate([summarize(list(s.values())) | {"bleu4": corpus_bleu(list(r.values()))} for s,r in zip(learned[k],rr)])
        table1["transllama_k5"] = learned_sums[5]
        for system, values in expected["table1"].items():
            for metric, (mean,sd) in values.items():
                tol = .0051 if metric=="bleu4" else .000051
                self.compare(f"Table1:{system}:{metric}:mean", table1[system][metric]["mean"], mean, tol)
                self.compare(f"Table1:{system}:{metric}:sample_sd", table1[system][metric]["sample_sd"], sd, tol)
        cis = {}
        for k in range(1,9):
            cis[str(k)] = {}
            for mi,metric in enumerate(("token_f1","sequence_similarity","laal")):
                diffs = [[a[key][metric]-b[key][metric] if a[key][metric] is not None and b[key][metric] is not None else float("nan") for key in a] for a,b in zip(scored_cg,learned[k])]
                cis[str(k)][metric] = self.bootstrap(diffs, 2026091900+100*k+99, vector_batch=True)
        fixed_cis={str(k):{metric:self.independent_bootstrap(scored_cg,fixed[k],metric,20260916,10000) for metric in ("token_f1","laal")} for k in (5,7)}
        return {"table1": table1, "fixed_frontier": {str(k): aggregate([summarize(list(r.values())) for r in v]) for k,v in fixed.items()},
                "fixed_comparison_bootstrap":fixed_cis,
                "learned_frontier": {str(k): v for k,v in learned_sums.items()}, "learned_comparison_bootstrap": cis,
                "raw_token_and_id_pairing": "passed for all 35 locked500 rollouts"}

    def score_epee(self, raw, sources, label, historical=None):
        self.pair_check(raw,sources)
        scored = {}
        for i,(key,r) in enumerate(raw.items()):
            src=sources[key]; pred=normalize_content(r["predicted_gloss_tokens"])
            refs=[normalize_content(x["content_tokens"]) for x in src["references"] if x["content_tokens"]]
            m={"token_f1":max(overlap(ref,pred) for ref in refs),
               "laal":laal(r["write_source_positions"],len(src["source_tokens"]),src["median_content_reference_length"]),
               "completion":float(r["completed_after_full_source"])}
            self.compare(f"{label}:row{i}:laal",m["laal"],r["length_adaptive_average_lagging_source_tokens"],1e-7)
            if historical is not None:
                self.compare(f"{label}:row{i}:historical_f1",m["token_f1"],historical[key]["content_best_token_f1"],1e-12)
            scored[key]=m
        return scored

    def epee(self, expected):
        sources=self.load(FULL/"clerc_epee_external_eval_v1/manifests/epee_sources_401.jsonl",401)
        historical_rows=read_jsonl(self.root/EPEE/"reports/full_analysis/per_source_scores.jsonl")
        hp=self.root/EPEE/"reports/full_analysis/per_source_scores.jsonl"
        self.inputs.append({"relative_path":str(EPEE/"reports/full_analysis/per_source_scores.jsonl").replace("\\","/"),"sha256":hashlib.sha256(hp.read_bytes()).hexdigest(),"bytes":hp.stat().st_size,"rows":len(historical_rows),"unique_prompt_ids":len({r["id"] for r in historical_rows})})
        historical={}
        for r in historical_rows:
            group=(r["system"],str(r["seed"]),r["k"])
            if r["id"] in historical.setdefault(group,{}): raise ValueError("Duplicate historical Epee score")
            historical[group][r["id"]]=r
        cg=[self.score_epee(self.load(EPEE/f"outputs/full/causalgloss_{s}/sentence_rollouts.jsonl",401),sources,f"epee_cg_{s}",historical[("causalgloss",str(s),None)]) for s in SEEDS]
        learned={k:[self.score_epee(self.load(EPEE/f"outputs/full/transllama_{s}/k{k}/sentence_rollouts.jsonl",401),sources,f"epee_tl_{s}_{k}",historical[("transllama_style",str(s),k)]) for s in SEEDS] for k in range(1,9)}
        lexical=self.score_epee(self.load(EPEE/"outputs/full/lexical_v1/sentence_rollouts.jsonl",401),sources,"epee_lexical")
        fixed={}
        for k in (1,3,5,7):
            ss=(20260821,) if k in (1,3) else (20260821,20260904,20260905)
            fixed[k]=[self.score_epee(self.load(EPEE/f"fixed_waitk_extension_20260920/outputs/full/waitk{k}_seed{s}/sentence_rollouts.jsonl",401),sources,f"epee_fixed_{k}_{s}") for s in ss]
        def agg(rows):
            return aggregate([{metric: finite_mean(r[metric] for r in s.values()) for metric in ("token_f1","laal","completion")} for s in rows])
        summaries={"causalgloss":agg(cg), "learned":{str(k):agg(v) for k,v in learned.items()}, "fixed":{str(k):agg(v) for k,v in fixed.items()}, "lexical_control":agg([lexical])}
        ee=expected["clerc"]
        self.compare("CLERC:causal_f1",summaries["causalgloss"]["token_f1"]["mean"],ee["causal_f1"],.000051)
        self.compare("CLERC:causal_laal",summaries["causalgloss"]["laal"]["mean"],ee["causal_laal"],.000051)
        self.compare("CLERC:learned_k5_completion",summaries["learned"]["5"]["completion"]["mean"],ee["learned_k5_completion"],.00051)
        all_f1=self.bootstrap([self.paired_differences(a,b,"token_f1") for a,b in zip(cg,learned[5])],20260920)
        all_laal=self.bootstrap([self.paired_differences(a,b,"laal") for a,b in zip(cg,learned[5])],20260921)
        pairs=[]; counts=[]; left_means=[]; right_means=[]
        for a,b in zip(cg,learned[5]):
            ids=sorted(key for key in b if b[key]["completion"]==1)
            counts.append(len(ids)); left_means.append(st.fmean(a[k]["token_f1"] for k in ids)); right_means.append(st.fmean(b[k]["token_f1"] for k in ids))
            pairs.append([a[k]["token_f1"]-b[k]["token_f1"] for k in ids])
        completed=self.bootstrap(pairs,20260921)
        completed.update({"causal_f1":st.fmean(left_means),"learned_f1":st.fmean(right_means),"n_by_seed":counts,
                          "post_hoc":True,"primary_401_analysis_unchanged":True})
        for metric in ("causal_f1","learned_f1"):
            self.compare("completed_only:"+metric,completed[metric],expected["completed_only"][metric],.000051)
        self.compare("completed_only:delta",completed["mean_difference"],expected["completed_only"]["delta"],.000051)
        if counts!=expected["completed_only"]["n_by_seed"]: raise ValueError("Completed subset counts differ")
        for i,v in enumerate(expected["completed_only"]["ci95"]):
            self.compare(f"completed_only:ci{i}",completed["ci95"][i],v,.00051)
        self.compare("CLERC:causal_minus_learned_k5_f1",all_f1["mean_difference"],ee["causal_minus_learned_k5_f1"],.000051)
        self.compare("CLERC:causal_minus_fixed_k7_f1",summaries["causalgloss"]["token_f1"]["mean"]-summaries["fixed"]["7"]["token_f1"]["mean"],ee["fixed_k7_f1_delta"],.000051)
        fixed_cis={str(k):{metric:self.independent_bootstrap(cg,fixed[k],metric,20260921+(2 if k==5 else 3)*10+j,self.repetitions) for j,metric in enumerate(("token_f1","laal"))} for k in (5,7)}
        return {"systems":summaries,"confirmatory_k5_f1":all_f1,"confirmatory_k5_laal":all_laal,"completed_only":completed,"fixed_comparison_bootstrap":fixed_cis,
                "reference_normalization":"remove QUESTION / trailing ?, strip _number variants, collapse exact periodic repetitions; best reference per prompt",
                "reference_count":sum(len(s["references"]) for s in sources.values()),
                "historical_per_source_f1_checks":27*401,
                "unit":"401 English prompts; references remain nested, never treated as independent rows"}

    def ablations(self, expected):
        paths={"full":RECOVERY/"runs/qwen2_5_7b_lora_state_recovery_v2_3_full/rollout_confirmation_selected/sentence_rollouts.jsonl",
               "no_recovery":RECOVERY/"baselines/v2_2_native_confirmation/sentence_rollouts.jsonl",
               "guard_only":RECOVERY/"baselines/v2_2_prompt_guard_confirmation/sentence_rollouts.jsonl"}
        for v in ("no_semantic","no_rescue","no_negation","no_occurrence","no_burst"):
            paths[v]=FULL/f"professor_feedback_experiments_20260912/ablations/{v}/evaluation/confirmation/sentence_rollouts.jsonl"
        raw={v:self.load(p,250) for v,p in paths.items()}
        for r in raw.values(): self.pair_check(raw["full"],r,True)
        scored={v:self.score_text(r,"ablation_"+v) for v,r in raw.items()}
        sums={v:summarize(list(r.values())) for v,r in scored.items()}
        ee=expected["table2"]
        self.compare("Table2:full_f1",sums["full"]["token_f1"],ee["full_f1"],.000051)
        self.compare("Table2:full_laal",sums["full"]["laal"],ee["full_laal"],.000051)
        deltas={}
        for i,(v,rows) in enumerate(scored.items()):
            if v=="full":continue
            deltas[v]={metric:self.bootstrap([self.paired_differences(scored["full"],rows,metric)],20260829+i*10+j,hierarchical=False,repetitions=10000)
                       for j,metric in enumerate(("token_f1","laal"))}
            if v in ee:
                for j,metric in enumerate(("token_f1","laal")):
                    self.compare(f"Table2:{v}:{metric}",deltas[v][metric]["mean_difference"],ee[v][j],.00051)
        self.compare("Table2:guard_only_f1_delta",deltas["guard_only"]["token_f1"]["mean_difference"],ee["guard_only_f1_delta"],.00051)
        return {"summaries":sums,"full_minus_ablated":deltas,"paired_finite_laal_note":"Recovery/guard comparison may exclude rows with no finite LAAL; counts are reported separately.",
                "reporting_caveat":"The five component ablations, full model, and guard-only control complete 100%; the native no-recovery comparison completes 249/250 (99.6%). Do not extend the paper's all-variants 100% wording to that native control.",
                "ci_note":"New paired-parent intervals with documented RNGs; point estimates verified against paper; historical CI bitwise identity not claimed."}

    def speech(self, expected):
        paths={"lexical":SPEECH/"outputs/v1.jsonl","s1":SPEECH/"outputs/v2_3.jsonl",
               "s2":SPEECH/"multiseed/seed20260904/outputs/v2_3_seed20260904.jsonl",
               "s3":SPEECH/"multiseed/seed20260905/outputs/v2_3_seed20260905.jsonl"}
        raw={v:self.load(p,500) for v,p in paths.items()}
        scored={}
        for v,rows in raw.items():
            self.pair_check(raw["s1"],rows)
            scored[v]={}
            for i,(key,r) in enumerate(rows.items()):
                ref=str(r["reference_pseudo_asl_gloss"]).split(); pred=r["speech_to_gloss_predicted_tokens"]
                d=float(r["audio_duration_seconds"])
                m={"token_f1":round(overlap(ref,pred),8),
                   "laal":laal(r["speech_to_gloss_write_audio_seconds"],d,len(ref)),
                   "compute_laal":laal(r["speech_to_gloss_write_compute_adjusted_seconds"],d,len(ref)),
                   "completion":float(r["speech_to_gloss_completed_after_full_source"]),
                   "rtf":float(r["speech_to_gloss_processing_realtime_factor"]),
                   "relative_length_error":abs(len(pred)/max(1,len(ref))-1)}
                for field,value in (("pseudo_gloss_token_f1",m["token_f1"]),("speech_to_gloss_laal_seconds",m["laal"]),("speech_to_gloss_compute_adjusted_laal_seconds",m["compute_laal"])):
                    self.compare(f"speech_{v}:row{i}:{field}",value,r[field],1e-7)
                scored[v][key]=m
        sums={v:{m:finite_mean(r[m] for r in rows.values()) for m in ("token_f1","laal","compute_laal","completion","rtf","relative_length_error")} for v,rows in scored.items()}
        ag=aggregate([sums[s] for s in ("s1","s2","s3")]); ee=expected["speech"]
        for field,metric,kind,tol in (("mean_f1","token_f1","mean",.000051),("sample_sd_f1","token_f1","sample_sd",.000051),("mean_raw_laal","laal","mean",.00051),("sample_sd_raw_laal","laal","sample_sd",.00051),("completion","completion","mean",.000051)):
            self.compare("speech:"+field,ag[metric][kind],ee[field],tol)
        differences={m:self.bootstrap([self.paired_differences(scored["s1"],scored["lexical"],m)],20260904+j,hierarchical=False,repetitions=10000) for j,m in enumerate(("token_f1","laal","compute_laal"))}
        for m,f in (("laal","s1_raw_latency_delta"),("compute_laal","s1_compute_latency_delta")):
            self.compare("speech:"+f,differences[m]["mean_difference"],ee[f],.00051)
        historical=self.historical_speech_bootstrap(raw["s1"],raw["lexical"])
        return {"per_seed":sums,"causal_three_seed":ag,"s1_minus_lexical":differences,"s1_historical_python_random_bootstrap":historical,
                "limitations":"Synthetic speech, weak pseudo-gloss references, serial processing simulation. All causal seeds retain output over-generation; successful metric replication is not successful task validation.",
                "ci_note":"Both fresh NumPy paired-parent intervals and exact historical Python-random shared-stream intervals across 14 metrics are recomputed."}

    def historical_speech_bootstrap(self,candidate,baseline):
        # Preserve the original 14-metric order because a single RNG was shared.
        metrics=("pseudo_gloss_sequence_similarity","pseudo_gloss_token_f1","pseudo_gloss_exact_match",
                 "speech_to_gloss_completed_with_end","speech_to_gloss_completed_after_full_source","speech_to_gloss_premature_end",
                 "prediction_reference_length_absolute_error","speech_to_gloss_first_write_audio_seconds",
                 "speech_to_gloss_reference_length_al_seconds","speech_to_gloss_laal_seconds","speech_to_gloss_average_proportion",
                 "speech_to_gloss_dal_seconds","speech_to_gloss_compute_adjusted_laal_seconds","speech_to_gloss_processing_realtime_factor")
        rng=random.Random(20260904); result={}
        def number(row,metric):
            if metric=="prediction_reference_length_absolute_error":
                return abs(len(row["speech_to_gloss_predicted_tokens"])/max(1,len(row["reference_pseudo_asl_gloss"].split()))-1)
            return row.get(metric)
        wanted={"pseudo_gloss_token_f1","speech_to_gloss_laal_seconds","speech_to_gloss_compute_adjusted_laal_seconds"}
        for metric in metrics:
            diffs=[float(number(candidate[k],metric))-float(number(baseline[k],metric)) for k in sorted(candidate)
                   if number(candidate[k],metric) is not None and number(baseline[k],metric) is not None]
            if not diffs:continue
            draws=[]
            for _ in range(10000):
                values=[diffs[rng.randrange(len(diffs))] for _ in diffs]
                if metric in wanted: draws.append(st.mean(values))
            if metric in wanted:
                draws.sort()
                def quantile(q):
                    position=(len(draws)-1)*q; lower=math.floor(position); upper=math.ceil(position)
                    return draws[lower] if lower==upper else draws[lower]*(upper-position)+draws[upper]*(position-lower)
                result[metric]={"mean_difference":round(st.mean(diffs),8),"ci95":[round(quantile(.025),8),round(quantile(.975),8)],"repetitions":10000,"shared_random_seed":20260904}
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--locked-root",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--expected",type=Path,default=Path(__file__).resolve().parents[2]/"results/expected_paper_results.json")
    p.add_argument("--bootstrap-repetitions",type=int,default=20000)
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError("Choose a new output directory; do not overwrite an audit.")
    expected=json.loads(a.expected.read_text(encoding="utf-8"))
    v=Verifier(a.input_root,a.locked_root or a.input_root/"locked500_from_local",a.bootstrap_repetitions)
    report={"generated_utc":datetime.now(timezone.utc).isoformat(),"paper_sha256":expected["paper_sha256"],
            "verification_level":"saved per-example outputs independently rescored; no inference or training rerun",
            "locked500":v.locked500(expected),"clerc":v.epee(expected),"ablations":v.ablations(expected),"speech":v.speech(expected)}
    ci_targets={
        "locked_fixed_k7_f1":(report["locked500"]["fixed_comparison_bootstrap"]["7"]["token_f1"]["ci95"],.00051),
        "locked_fixed_k7_laal":(report["locked500"]["fixed_comparison_bootstrap"]["7"]["laal"]["ci95"],.00051),
        "locked_learned_k5_f1":(report["locked500"]["learned_comparison_bootstrap"]["5"]["token_f1"]["ci95"],.000051),
        "locked_learned_k5_laal":(report["locked500"]["learned_comparison_bootstrap"]["5"]["laal"]["ci95"],.000051),
        "clerc_learned_k5_f1":(report["clerc"]["confirmatory_k5_f1"]["ci95"],.00051),
        "clerc_fixed_k7_f1":(report["clerc"]["fixed_comparison_bootstrap"]["7"]["token_f1"]["ci95"],.00051)}
    hist=report["speech"]["s1_historical_python_random_bootstrap"]
    for label,field,tol in (("speech_s1_f1","pseudo_gloss_token_f1",.000051),("speech_s1_raw_laal","speech_to_gloss_laal_seconds",.00051),("speech_s1_compute_laal","speech_to_gloss_compute_adjusted_laal_seconds",.00051)):
        ci_targets[label]=(hist[field]["ci95"],tol)
    if a.bootstrap_repetitions==20000:
        for label,(actual,tol) in ci_targets.items():
            for i,e in enumerate(expected["confidence_intervals"][label]):v.compare("PDF_CI:"+label+f":{i}",actual[i],e,tol)
    remote_manifest=a.input_root/"verification_input_manifest.json"
    if remote_manifest.exists():
        remote=json.loads(remote_manifest.read_text(encoding="utf-8"))
        remote_by_path={r["relative_path"]:r for r in remote["files"]}
        for r in v.inputs:
            record=remote_by_path.get(r["relative_path"],{})
            if record.get("remote_sha256")!=r["sha256"]:
                v.discrepancies.append({"check":"remote_input_sha256","relative_path":r["relative_path"],"matches":False})
        report["remote_sha256_verification"]={"fetched_file_count":len(remote["files"]),"all_fetched_files_match_remote":all(r.get("remote_hash_matches",False) for r in remote["files"])}
    else:
        report["remote_sha256_verification"]={"status":"not_available; metric recomputation does not establish remote byte identity"}
    report.update({"input_files":v.inputs,"discrepancies":v.discrepancies,"all_numeric_checks_pass":not v.discrepancies,
                   "not_performed":["full model retraining","full model re-inference","independent expert ASL validation","raw audio duration remeasurement"]})
    a.output_dir.mkdir(parents=True)
    (a.output_dir/"verification_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"input_files":len(v.inputs),"all_numeric_checks_pass":not v.discrepancies,"discrepancy_count":len(v.discrepancies),"report":str(a.output_dir/"verification_report.json")},indent=2))
    raise SystemExit(bool(v.discrepancies))

if __name__=="__main__": main()
