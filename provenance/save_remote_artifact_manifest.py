"""Store metadata-only SSH stdout, annotating previously verified hashes."""
from pathlib import Path
import hashlib
import json
import sys

root=Path('D:/Infinisst')
result=json.load(sys.stdin)
prior={}
for report in ['server_audit.json','server_data_audit.json']:
    path=root/'research_reports/causalgloss_release_audit_20261016'/report
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if report=='server_audit.json':
        rows=data.get('artifacts',[])
        for row in rows:
            if row.get('sha256'):prior[row['path']]={'sha256':row['sha256'],'source_report':str(path),'verified_by_prior_audit':True}
    else:
        for row in data.get('live_verified_files',[]):
            prior[data['remote_root']+'/'+row['path']]={'sha256':row['sha256'],'source_report':str(path),'verified_by_prior_audit':True}
for row in result['entries']:
    old=prior.get(row['remote_path'])
    if old:
        row['prior_hash_evidence']=old
        if row.get('sha256'):
            row['live_hash_matches_prior_audit']=row['sha256']==old['sha256']
        else:
            row['sha256']=old['sha256']
            row['hash_verification']='reused_prior_verified_audit_live_path_size_mtime_rechecked_no_rehash'
result['entry_count']=len(result['entries'])
result['audio_layout_notes']={
    'raw_corpus_utterance_count_reported':19947,
    'cleaned_corpus_utterance_count_reported':19924,
    'raw_full_corpus_audio_utterances_in_own_directory':19447,
    'test_utterances_reusing_locked500_audio':500,
    'raw_waveform_total_in_two_directories':39894,
    'raw_training_exclusions_retained_as_audio_assets':23,
    'explanation':'The full corpus references 500 existing locked-test audio pairs instead of duplicating them. Each generated utterance has an original WAV and a normalized 16 kHz mono WAV. Cleaned manifest excludes 23 training items while raw waveforms remain.',
    'evidence_report':'data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/chinese_speech_to_pseudo_asl_corpus_v1_20260905/reports/manifest_build.json and corpus_summary.cleaned.json',
    'directory_counts_checked_live':True,
    'per_audio_manifest_existence_not_revalidated_in_this_metadata_only_audit':True,
}
result['category_counts']={}
for row in result['entries']:result['category_counts'][row['category']]=result['category_counts'].get(row['category'],0)+1
destination=root/'releases/CasualGloss/provenance/artifact_manifest.json'
destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'output':str(destination),'entries':result['entry_count'],'model_groups':result['selected_model_group_count'],'missing_expected_paths':result['missing_expected_paths'],'category_counts':result['category_counts'],'manifest_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()},ensure_ascii=True,indent=2))
