"""Run through SSH stdin. Emit metadata only; no remote writes or sample text."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import os

ROOT = Path('/123090274/SN_Research')
DATA = ROOT / 'data/how2sign_english_asl_gloss_v1'
FULL = DATA / 'full_v4_train'
FRONT = DATA / 'chinese_speech_frontend_v0'
entries = {}
missing = []
groups = []
hash_budget = 3 * 1024 ** 3
hashed_bytes = 0


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def file_record(path, category, role, evidence=None, selected=False, force_hash=False):
    global hashed_bytes
    path = Path(path)
    path.relative_to(ROOT)
    if not path.is_file():
        missing.append({'remote_path': str(path), 'category': category, 'role': role})
        return
    key = str(path)
    if key in entries:
        entries[key]['additional_roles'].append(role)
        return
    stat = path.stat()
    record = {
        'remote_path': key, 'relative_to_remote_root': str(path.relative_to(ROOT)),
        'type': 'file', 'category': category, 'role': role,
        'size_bytes': stat.st_size, 'mtime_utc': stamp(stat.st_mtime),
        'exists_checked_live': True, 'selected': selected,
        'selection_evidence': evidence or [], 'sha256': None,
        'hash_verification': 'not_hashed_live_size_budget_or_large_data',
        'additional_roles': [],
        'acquisition': {'method': 'authenticated_ssh_sftp_read_only', 'remote_path': key,
                        'downloaded_in_this_audit': False,
                        'redistribution_permission': 'not_determined_by_this_metadata_audit'},
    }
    high_value_training_input = any(str(path).endswith(suffix) for suffix in [
        '/state_recovery_v2_3_full/manifests/train_actions_full.jsonl',
        '/semantic_causal_alignment_v2_2_full/manifests_treebank_v2_2_1/train_actions.jsonl',
        '/fine_tuning_v2_semantic_v2_2_full/manifests_wait_balanced_v2_2_1/train_actions.jsonl',
    ])
    allowed_hash = force_hash or high_value_training_input or category == 'prefix_scores' or stat.st_size <= 32 * 1024 ** 2
    if allowed_hash and hashed_bytes + stat.st_size <= hash_budget:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
                digest.update(block)
        record['sha256'] = digest.hexdigest()
        record['hash_verification'] = 'sha256_computed_live_remote_20261016'
        hashed_bytes += stat.st_size
    entries[key] = record


def flat(directory, category, role, patterns=('*.jsonl',), evidence=None):
    directory = Path(directory)
    if not directory.is_dir():
        missing.append({'remote_path': str(directory), 'category': category, 'role': role, 'type': 'directory'})
        return
    seen = set()
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                file_record(path, category, role, evidence)


def model_group(name, run_relative, adapter, role, evidence):
    run = FULL / run_relative
    adapter_path = run / adapter
    before = len(entries)
    if not adapter_path.is_dir():
        missing.append({'remote_path': str(adapter_path), 'category': 'selected_lora', 'role': role, 'group': name})
    else:
        # Weights first, then small adapter/tokenizer configuration assets.
        files = sorted(adapter_path.iterdir(), key=lambda p: (p.name != 'adapter_model.safetensors', p.name))
        for path in files:
            if path.is_file():
                file_record(path, 'selected_lora', role, evidence, True, path.name == 'adapter_model.safetensors')
    for filename in ['run_metadata.json', 'training_summary.json', 'training_history.jsonl']:
        path = run / filename
        if path.is_file():
            file_record(path, 'selected_training_metadata', role, evidence, True)
        else:
            missing.append({'remote_path': str(path), 'category': 'selected_training_metadata', 'role': role, 'group': name})
    for path in sorted(run.glob('*history*.jsonl')):
        if path.name != 'training_history.jsonl':
            file_record(path, 'selected_training_curve', role, evidence, True)
    groups.append({'name': name, 'run_directory': str(run), 'selected_adapter_directory': str(adapter_path),
                   'role': role, 'selection_evidence': evidence, 'new_file_entries': len(entries)-before})


clerc_run = str(FULL / 'clerc_epee_adaptive_comparison_20260920/scripts/run_full.sh')
fixed_run = str(FULL / 'clerc_epee_adaptive_comparison_20260920/scripts/run_fixed_waitk_epee.sh')
ablation_run = str(FULL / 'professor_feedback_experiments_20260912/scripts/run_ablation.sh')
speech_run = str(FRONT / 'scripts/run_locked500_speech_e2e_pipeline.sh')
for seed in [20260829, 20260904, 20260905]:
    if seed == 20260829:
        rel = 'state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full'
        selection = FULL / 'state_recovery_v2_3_full/reports/full_development_selection.json'
    else:
        rel = f'state_recovery_v2_3_multiseed_{seed}/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed{seed}'
        selection = FULL / f'state_recovery_v2_3_multiseed_{seed}/reports/development_selection_seed{seed}.json'
    model_group(f'causalgloss_s{seed}', rel, 'adapter_final', f'paper_main_selected_final_seed{seed}', [clerc_run, str(selection)])
    file_record(selection, 'checkpoint_selection_evidence', f'paper_main_seed{seed}_development_selection')
model_group('common_finetune1', 'fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1', 'adapter_best', 'common_finetune1_initialization', [str(FULL / 'state_recovery_v2_3_full/scripts/run_v2_3_full.sh')])
model_group('v2_2_1_initialization', 'fine_tuning_v2_semantic_v2_2_full/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full', 'adapter_best', 'v2_3_selected_initialization', [str(FULL / 'state_recovery_v2_3_full/scripts/run_v2_3_full.sh')])
model_group('lexical_clerc', 'fine_tuning_v2_causal_full/runs/qwen2_5_7b_lora_causal_action_full_corrected_v1', 'adapter_best', 'lexical_clerc_selected_best', [clerc_run])
model_group('lexical_speech', 'fine_tuning_v2_causal_full_seed2/runs/qwen2_5_7b_lora_causal_action_full_seed2_v1', 'adapter_best', 'lexical_speech_selected_best_distinct_from_clerc', [speech_run])
for k in [1,3,5,7]:
    name = f'qwen2_5_7b_lora_wait_k{k}_full_v1' + ('_recovery_a6000_2gpu' if k == 3 else '')
    model_group(f'fixed_k{k}_seed20260821', f'wait_k_baselines_v1/runs/{name}', 'adapter_best', f'fixed_waitk{k}_seed20260821_selected_best', [fixed_run])
for k in [5,7]:
    for seed in [20260904,20260905]:
        model_group(f'fixed_k{k}_seed{seed}', f'wait_k_baselines_multiseed_20260914/runs/qwen2_5_7b_lora_wait_k{k}_full_seed{seed}', 'adapter_best', f'fixed_waitk{k}_seed{seed}_selected_best', [fixed_run])
for seed in [20260829,20260904,20260905]:
    model_group(f'transllama_seed{seed}', f'transllama_learned_adaptive_20260918/runs/final_seeds/final_seed_{seed}', 'adapter_best', f'final_independently_trained_transllama_seed{seed}_selected_best', [clerc_run])
for variant in ['no_semantic','no_rescue','no_negation','no_occurrence','no_burst']:
    for stage in ['a','b']:
        role = f'ablation_{variant}_stage_{stage}_' + ('selected_downstream_initialization' if stage == 'a' else 'selected_confirmation250_evaluation')
        model_group(f'{variant}_stage_{stage}', f'professor_feedback_experiments_20260912/ablations/{variant}/runs/stage_{stage}', 'adapter_best', role, [ablation_run])

log_map={
    'causalgloss_s20260829':'state_recovery_v2_3_full/logs/run_v2_3_full.log',
    'causalgloss_s20260904':'state_recovery_v2_3_multiseed_20260904/logs/run_v2_3_seed20260904.log',
    'causalgloss_s20260905':'state_recovery_v2_3_multiseed_20260905/logs/full_pipeline.log',
    'common_finetune1':'fine_tuning_v1/logs/full_finetuning_4gpu.log',
    'v2_2_1_initialization':'fine_tuning_v2_semantic_v2_2_full/logs/run_v2_2_1_full.log',
    'lexical_clerc':'fine_tuning_v2_causal_full/logs/run_finetune2_full.log',
    'lexical_speech':'fine_tuning_v2_causal_full_seed2/logs/run_finetune2_full_seed2.log',
    'fixed_k1_seed20260821':'wait_k_baselines_v1/logs/run_wait_k1_full.a800_gpu0.log',
    'fixed_k3_seed20260821':'wait_k_baselines_v1/logs/run_wait_k3_full_recovery_a6000_2gpu.log',
    'fixed_k5_seed20260821':'wait_k_baselines_v1/logs/run_wait_k5_full.a800_gpu1.log',
    'fixed_k7_seed20260821':'wait_k_baselines_v1/logs/run_wait_k7_full.a800_gpu2.log',
}
for k in [5,7]:
    for seed,gpu in [(20260904,0),(20260905,1)]:
        log_map[f'fixed_k{k}_seed{seed}']=f'wait_k_baselines_multiseed_20260914/logs/k{k}_seed{seed}_gpu{gpu}.log'
for seed in [20260829,20260904,20260905]:
    log_map[f'transllama_seed{seed}']=f'transllama_learned_adaptive_20260918/logs/final_seed_{seed}.log'
for variant in ['no_semantic','no_rescue','no_negation','no_occurrence','no_burst']:
    for stage in ['a','b']:
        log_map[f'{variant}_stage_{stage}']=f'professor_feedback_experiments_20260912/logs/{variant}.log'
for group in groups:
    log_path=FULL/log_map[group['name']]
    group['training_log_association']={'remote_path':str(log_path),'association':'selected_run_pipeline_log; ablation stage_a and stage_b share variant pipeline log' if group['name'].startswith('no_') else 'selected_run_training_launcher_log'}
    file_record(log_path,'selected_training_log',group['role'],group['selection_evidence'],True)
    if group['name'].startswith('fixed_k'):
        file_record(Path(group['run_directory'])/'rollout_locked500/sentence_rollouts.jsonl','paper_raw_rollouts',group['name']+'_locked500_test')
    elif group['name'].startswith('no_') and group['name'].endswith('_stage_b'):
        variant=group['name'].removesuffix('_stage_b')
        file_record(FULL/f'professor_feedback_experiments_20260912/ablations/{variant}/evaluation/confirmation/sentence_rollouts.jsonl','paper_raw_rollouts',variant+'_confirmation250')

# Required final data products, not intermediate failed/smoke training runs.
dataset_dirs = [
    ('manifests', 'official_train_and_unique_generation_inputs'),
    ('fine_tuning_v1/manifests', 'common_finetune1_train_validation_test_inputs'),
    ('causal_alignment_v1_corrected_5k/manifests', 'locked500_original_parent_split_provenance'),
    ('causal_alignment_v1_corrected_full/manifests', 'corrected_full_frozen_parent_and_action_splits'),
    ('semantic_causal_alignment_v2_1_full/manifests', 'semantic_prefilter_reconciled_parents_and_action_splits'),
    ('semantic_causal_alignment_v2_2_full/manifests', 'semantic_v2_2_pair_and_action_splits'),
    ('semantic_causal_alignment_v2_2_full/manifests_treebank_v2_2_1', 'treebank_v2_2_1_action_splits'),
    ('fine_tuning_v2_semantic_v2_2_full/manifests_wait_balanced_v2_2_1', 'wait_balanced_v2_2_1_training_actions'),
    ('state_recovery_v2_3_full/manifests', 'main_shared_train_development_confirmation_inputs'),
    ('transllama_learned_adaptive_20260918/alignments', 'final_transllama_train_validation_alignments'),
    ('clerc_epee_external_eval_v1/manifests', 'clerc401_1200_reference_source_manifests'),
]
for rel, role in dataset_dirs:
    flat(FULL / rel, 'paper_dataset_or_alignment', role)
for k in [1,3,5,7]:
    flat(FULL/f'wait_k_baselines_v1/manifests/k{k}/full','fixed_training_and_evaluation_inputs',f'fixed_waitk{k}_full_train_validation_test_shared_across_seeds')
for variant in ['no_semantic','no_rescue','no_negation','no_occurrence','no_burst']:
    flat(FULL / f'professor_feedback_experiments_20260912/ablations/{variant}/manifests', 'ablation_training_and_confirmation_inputs', variant)
flat(FULL / 'outputs', 'pseudo_gloss_targets', 'official_train_final_weak_pseudo_labels', ('*.final.jsonl',))
flat(FULL / 'causal_alignment_v1_corrected_full/outputs', 'lexical_alignment_outputs', 'corrected_full_lexical_alignments_and_exclusions')
flat(FULL / 'semantic_causal_alignment_v2_1_full/shards', 'prefix_scores', 'full_prefix_scores_and_contraction_fix_delta', ('prefix_semantic_scores*.jsonl',))
flat(FULL / 'semantic_causal_alignment_v2_pilot500/shards', 'prefix_scores', 'pilot_prefix_scores_used_for_rule_development', ('prefix_semantic_scores*.jsonl',))
flat(FULL / 'semantic_causal_alignment_v2_1_full/outputs', 'semantic_alignment_outputs', 'v2_1_semantic_alignment_output')
flat(FULL / 'semantic_causal_alignment_v2_2_full/outputs', 'semantic_alignment_outputs', 'v2_2_confident_burst_alignment_output')
for corpus in ['paper_e2e_locked500_20260904','chinese_speech_to_pseudo_asl_corpus_v1_20260905']:
    flat(FRONT / corpus / 'manifests', 'corpus_manifests', corpus, ('locked500.*.jsonl',) if corpus.startswith('paper_') else ('full.*.jsonl',))
    if corpus.startswith('chinese_speech_to'):
        flat(FRONT / corpus, 'corpus_configuration', corpus, ('config*.json', 'DATASET_CARD*.md'))
        flat(FRONT / corpus / 'reports', 'corpus_quality_metadata', corpus, ('corpus_summary*.json', 'audio_validation*.json', 'audio_exclusions*.json', 'manifest_build.json', '*sha256*.txt'))

# Paper evaluation records: paths and hashes only, no gloss or source text.
for seed in [20260829,20260904,20260905]:
    if seed==20260829:
        eval_dir=FULL/'state_recovery_v2_3_full/final_evaluation_v2_3/locked500/v2_3'
    else:
        eval_dir=FULL/f'state_recovery_v2_3_multiseed_{seed}/final_evaluation_seed{seed}/locked500/v2_3'
    file_record(eval_dir/'sentence_rollouts.jsonl','paper_raw_rollouts',f'main_locked500_seed{seed}')
    for k in range(1,9):
        file_record(FULL/f'transllama_learned_adaptive_20260918/runs/final_test/final_seed_{seed}/k{k}/sentence_rollouts.jsonl','paper_raw_rollouts',f'final_transllama_locked500_seed{seed}_k{k}')
        file_record(FULL/f'clerc_epee_adaptive_comparison_20260920/outputs/full/transllama_{seed}/k{k}/sentence_rollouts.jsonl','paper_raw_rollouts',f'clerc401_transllama_seed{seed}_k{k}')
    file_record(FULL/f'clerc_epee_adaptive_comparison_20260920/outputs/full/causalgloss_{seed}/sentence_rollouts.jsonl','paper_raw_rollouts',f'clerc401_main_seed{seed}')
file_record(FULL/'clerc_epee_adaptive_comparison_20260920/outputs/full/lexical_v1/sentence_rollouts.jsonl','paper_raw_rollouts','clerc401_lexical_control')
for k,seed in [(1,20260821),(3,20260821),(5,20260821),(7,20260821),(5,20260904),(5,20260905),(7,20260904),(7,20260905)]:
    file_record(FULL/f'clerc_epee_adaptive_comparison_20260920/fixed_waitk_extension_20260920/outputs/full/waitk{k}_seed{seed}/sentence_rollouts.jsonl','paper_raw_rollouts',f'clerc401_fixed_k{k}_seed{seed}')
flat(FULL/'clerc_epee_adaptive_comparison_20260920/reports/full_analysis','paper_per_source_scores','clerc401_adaptive_and_completed_only_inputs',('per_source_scores.jsonl','comparison_report.json'))
flat(FULL/'clerc_epee_adaptive_comparison_20260920/fixed_waitk_extension_20260920/reports/full_analysis','paper_per_source_scores','clerc401_fixed_inputs',('per_source_scores.jsonl','comparison_report.json'))
flat(FRONT/'paper_e2e_locked500_20260904/outputs','paper_raw_rollouts','locked500_speech_shared_frontend_and_s1_lexical',('*.jsonl',))
for seed in [20260904,20260905]:
    file_record(FRONT/f'paper_e2e_locked500_20260904/multiseed/seed{seed}/outputs/v2_3_seed{seed}.jsonl','paper_raw_rollouts',f'locked500_speech_seed{seed}')

# Numeric provenance/selection files are indexed; their text contents are not emitted.
for path in [Path(clerc_run),Path(fixed_run),Path(ablation_run),Path(speech_run)]:
    file_record(path, 'selection_evidence_source', 'checkpoint_selection_launcher_evidence')

def audio_directory(path):
    if not path.is_dir():
        missing.append({'remote_path': str(path), 'category': 'audio_directory'})
        return
    counts=Counter(); total_bytes=0; all_count=0; earliest=None; latest=None; children={}
    for directory, dirs, files in os.walk(path, followlinks=False):
        dirs[:]=[name for name in dirs if name not in ['.cache','__pycache__']]
        for name in files:
            f=Path(directory)/name
            if not f.is_file() or f.is_symlink():continue
            stat=f.stat(); all_count+=1; total_bytes+=stat.st_size; counts[f.suffix.lower()]+=1
            earliest=stat.st_mtime if earliest is None else min(earliest,stat.st_mtime)
            latest=stat.st_mtime if latest is None else max(latest,stat.st_mtime)
            rel=f.relative_to(path); bucket=rel.parts[0] if len(rel.parts)>1 else '.'
            item=children.setdefault(bucket,{'file_count':0,'total_bytes':0});item['file_count']+=1;item['total_bytes']+=stat.st_size
    entries[str(path)]={'remote_path':str(path),'relative_to_remote_root':str(path.relative_to(ROOT)),
        'type':'directory_index','category':'audio_directory','role':path.name,
        'exists_checked_live':True,'file_count':all_count,'total_bytes':total_bytes,
        'extension_counts':dict(counts),'child_directory_index':children,
        'file_mtime_min_utc':stamp(earliest) if earliest else None,
        'file_mtime_max_utc':stamp(latest) if latest else None,
        'sha256':None,'hash_verification':'not_hashed_no_per_audio_hash_requested',
        'sample_text_emitted':False,'acquisition':{'method':'authenticated_ssh_sftp_directory_copy_if_authorized_and_licensed','remote_path':str(path),'downloaded_in_this_audit':False,'manifest_path_mapping_required':True}}

for name in ['paper_e2e_locked500_20260904','chinese_speech_to_pseudo_asl_corpus_v1_20260905']:
    audio_directory(FRONT / 'audio' / name)

result={
    'audit_date':'2026-10-16','remote_root':str(ROOT),'mode':'read_only_metadata_and_in_place_hashes',
    'remote_files_written':False,'artifact_binaries_downloaded':False,'sample_text_emitted':False,
    'hash_policy':{'live_hash_budget_bytes':hash_budget,'bytes_hashed_live':hashed_bytes,'default_max_data_file_hash_bytes':32*1024**2,'selected_lora_weights_priority':True,'per_audio_hashes':False},
    'selected_model_groups':groups,'selected_model_group_count':len(groups),
    'entries':list(entries.values()),'missing_expected_paths':missing,
    'excluded_from_selection':['failed_nonfinite_waitk3_run','aborted_v2_2_manual_detokenizer','old_professor_feedback_learned_baseline','smoke_training_runs','screen_checkpoint_as_main_final','backbone_model_weights','G2P','credential_files'],
    'limitations':['Metadata existence and hashes do not establish redistribution permission.','File metadata were inspected live; JSONL sample contents were not returned.','Model selection roles are tied to explicit existing launchers/protocol reports; stage_a ablation checkpoints are downstream initialization, not final direct evaluations.']
}
print(json.dumps(result,ensure_ascii=True))
