#!/usr/bin/env python3
"""Build a self-contained browser viewer for the Task 5 end-to-end pilot."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def quality_bucket(similarity: float) -> str:
    if similarity >= 0.75:
        return "high"
    if similarity >= 0.35:
        return "medium"
    return "low"


def prepare_samples(
    rows: list[dict[str, Any]],
    asr_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    output_dir: Path,
    root: Path,
) -> list[dict[str, Any]]:
    asr_by_id = {row["id"]: row for row in asr_rows}
    video_by_id = {row["id"]: row for row in video_rows}
    samples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["pilot_index"])):
        asr = asr_by_id.get(row["id"], {})
        video = video_by_id.get(row["id"], {})
        reference_words = normalize_words(row["reference_text_en"])
        hypothesis_words = normalize_words(row["streaming_frontend_hypothesis_en"])
        wer = edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))
        audio_path = root / row["tts_audio_16k_mono"]
        audio_relative = Path("..") / audio_path.relative_to(root)
        video_path = Path(video["video_path"]) if video.get("video_path") else None
        video_relative = (
            video_path.relative_to(output_dir).as_posix()
            if video_path is not None and video_path.is_relative_to(output_dir)
            else None
        )
        duration = float(row["audio_duration_seconds"])
        first_english = row.get("streaming_frontend_first_commit_audio_seconds")
        first_gloss = row.get("speech_to_gloss_first_write_audio_seconds")
        action_events = row.get("speech_to_gloss_action_events", [])
        samples.append(
            {
                "id": row["id"],
                "index": int(row["pilot_index"]),
                "speaker": row.get("tts_speaker"),
                "audio": audio_relative.as_posix(),
                "audio_exists": audio_path.exists(),
                "video": video_relative,
                "video_exists": bool(video_path and video_path.exists()),
                "video_official_text_match": video.get("official_text_exact_match"),
                "video_archive_sha256": video.get("archive_sha256"),
                "duration": duration,
                "zh": row["source_text_zh"],
                "zhAsr": asr.get("hypothesis_zh"),
                "zhCer": asr.get("character_error_rate"),
                "enRef": row["reference_text_en"],
                "enHyp": row["streaming_frontend_hypothesis_en"],
                "enWer": round(wer, 6),
                "glossRef": row["reference_pseudo_asl_gloss"],
                "glossHyp": row["speech_to_gloss_predicted_pseudo_asl_gloss"],
                "similarity": float(row["pseudo_gloss_sequence_similarity"]),
                "tokenF1": float(row["pseudo_gloss_token_f1"]),
                "exact": bool(row["pseudo_gloss_exact_match"]),
                "firstEnglish": first_english,
                "firstGloss": first_gloss,
                "glossLead": round(duration - float(first_gloss), 6)
                if first_gloss is not None
                else None,
                "averageLaggingMs": row.get("streaming_frontend_average_lagging_ms"),
                "englishEventCount": int(row["streaming_frontend_commit_event_count"]),
                "actionCount": int(row["speech_to_gloss_action_count"]),
                "waitCount": sum(
                    event.get("predicted_action_class") == "WAIT" for event in action_events
                ),
                "writeCount": sum(
                    event.get("predicted_action_class") == "WRITE" for event in action_events
                ),
                "invalidActionCount": sum(
                    event.get("predicted_action_class") == "INVALID"
                    for event in action_events
                ),
                "completedEnd": bool(
                    row.get(
                        "speech_to_gloss_completed_after_full_source",
                        row["speech_to_gloss_completed_with_end"]
                        and not row.get("speech_to_gloss_premature_end", False),
                    )
                ),
                "prematureEnd": bool(row["speech_to_gloss_premature_end"]),
                "bucket": quality_bucket(float(row["pseudo_gloss_sequence_similarity"])),
                "frontendEvents": row.get("streaming_frontend_commit_events", []),
                "actionEvents": action_events,
            }
        )
    return samples


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech-to-Gloss Timeline Review</title>
<style>
:root {
  --ink: #17324d;
  --muted: #5f6975;
  --line: #d6dee7;
  --canvas: #f4f7fa;
  --paper: #ffffff;
  --teal: #167d74;
  --teal-soft: #e8f4f2;
  --amber: #be6f18;
  --amber-soft: #fff2df;
  --red: #b9473f;
  --red-soft: #fcebea;
  --blue-soft: #eaf1f8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--canvas);
  font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  font-size: 15px;
  letter-spacing: 0;
}
button, select, input { font: inherit; letter-spacing: 0; }
button:focus-visible, select:focus-visible, input:focus-visible {
  outline: 3px solid rgba(22, 125, 116, 0.24);
  outline-offset: 2px;
}
.topbar {
  background: var(--paper);
  border-bottom: 1px solid var(--line);
  padding: 18px 24px 14px;
}
.topline, .toolbar, .metrics, .live-grid, .reference-grid {
  max-width: 1520px;
  margin: 0 auto;
}
.topline { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; }
h1 { margin: 0; font-size: 28px; line-height: 1.18; font-weight: 750; }
.subtitle { margin: 6px 0 0; color: var(--muted); font-size: 15px; }
.status-badge {
  flex: 0 0 auto;
  color: var(--red);
  background: var(--red-soft);
  border: 1px solid #e3aaa6;
  border-radius: 5px;
  padding: 8px 10px;
  font-weight: 650;
  font-size: 13px;
}
.toolbar-band { background: var(--paper); border-bottom: 1px solid var(--line); padding: 12px 24px; }
.toolbar { display: grid; grid-template-columns: auto minmax(300px, 1fr) auto auto; gap: 10px; align-items: center; }
.icon-btn, .command-btn {
  height: 38px;
  border: 1px solid #aebdca;
  border-radius: 5px;
  background: var(--paper);
  color: var(--ink);
  cursor: pointer;
  font-weight: 650;
}
.icon-btn { width: 40px; font-size: 20px; }
.command-btn { padding: 0 14px; }
.command-btn.primary { background: var(--teal); border-color: var(--teal); color: white; }
.command-btn:disabled, .icon-btn:disabled { opacity: .45; cursor: default; }
select {
  width: 100%; height: 38px; border: 1px solid #aebdca; border-radius: 5px;
  background: white; color: var(--ink); padding: 0 10px;
}
.segmented { display: flex; height: 38px; border: 1px solid #aebdca; border-radius: 5px; overflow: hidden; }
.segmented button { border: 0; border-right: 1px solid #aebdca; padding: 0 11px; background: white; color: var(--muted); cursor: pointer; }
.segmented button:last-child { border-right: 0; }
.segmented button.active { background: var(--blue-soft); color: var(--ink); font-weight: 700; }
main { padding: 16px 24px 30px; }
.metrics { display: grid; grid-template-columns: repeat(8, minmax(92px, 1fr)); gap: 8px; margin-bottom: 12px; }
.metric {
  min-height: 82px; background: var(--paper); border: 1px solid var(--line); border-radius: 6px;
  padding: 11px 12px; overflow: hidden;
}
.metric .label { color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; }
.metric .value { margin-top: 7px; font-size: 22px; font-weight: 760; white-space: nowrap; }
.metric.good .value { color: var(--teal); }
.metric.warn .value { color: var(--amber); }
.metric.bad .value { color: var(--red); }
.workspace { max-width: 1520px; margin: 0 auto; display: grid; grid-template-columns: minmax(350px, .84fr) minmax(560px, 1.36fr); gap: 12px; }
.panel { background: var(--paper); border: 1px solid var(--line); border-radius: 6px; min-width: 0; }
.panel-head { padding: 13px 16px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.panel-head h2 { margin: 0; font-size: 17px; }
.panel-body { padding: 14px 16px; }
.sample-meta { color: var(--muted); font-size: 12px; font-family: Consolas, monospace; overflow-wrap: anywhere; }
audio { width: 100%; height: 42px; margin: 0 0 8px; }
.sign-video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #101820; object-fit: contain; border-radius: 5px; }
.media-label { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin: 11px 0 6px; color: var(--muted); font-size: 12px; font-weight: 700; }
.verified { color: var(--teal); text-align: right; }
.audio-controls { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.playback-status { margin-left: auto; color: var(--muted); align-self: center; font-size: 12px; }
.progress-wrap { position: relative; height: 30px; margin: 4px 0 16px; }
.progress-track { position: absolute; left: 0; right: 0; top: 12px; height: 6px; background: #e3e9ef; border-radius: 3px; overflow: hidden; }
.progress-fill { height: 100%; width: 0; background: var(--teal); }
.marker { position: absolute; top: 7px; width: 2px; height: 16px; transform: translateX(-1px); background: var(--amber); }
.marker.english { background: #3d6f9f; height: 12px; top: 4px; }
.time-readout { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
.text-block { border-top: 1px solid var(--line); padding: 12px 0; }
.text-block:first-of-type { border-top: 0; }
.text-block .label { color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 6px; }
.text-block .text { font-size: 17px; line-height: 1.5; overflow-wrap: anywhere; }
.text-block .text.mono { font-family: Consolas, monospace; font-size: 15px; }
.live-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.live-box { border: 1px solid var(--line); border-radius: 5px; padding: 12px; min-height: 126px; }
.live-box.english { background: var(--blue-soft); }
.live-box.gloss { background: var(--teal-soft); }
.live-title { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 8px; }
.live-text { font-size: 20px; line-height: 1.38; font-weight: 680; overflow-wrap: anywhere; }
.action-pill { display: inline-block; border-radius: 4px; padding: 3px 7px; font-family: Consolas, monospace; font-size: 12px; background: var(--amber-soft); color: var(--amber); }
.action-pill.invalid { background: var(--red-soft); color: var(--red); }
.timeline { margin-top: 12px; border: 1px solid var(--line); border-radius: 5px; overflow: auto; height: 430px; }
.event { display: grid; grid-template-columns: 66px 88px minmax(130px, .85fr) minmax(190px, 1.15fr); gap: 10px; padding: 9px 11px; border-bottom: 1px solid #e7ecf1; align-items: start; opacity: .4; }
.event:last-child { border-bottom: 0; }
.event.revealed { opacity: 1; }
.event.current { background: var(--amber-soft); box-shadow: inset 3px 0 var(--amber); }
.event-time { font-family: Consolas, monospace; font-weight: 700; }
.event-type { font-size: 12px; font-weight: 750; }
.event-type.en { color: #3d6f9f; }
.event-type.write { color: var(--teal); }
.event-type.wait { color: var(--amber); }
.event-type.end { color: var(--red); }
.event-type.invalid { color: var(--red); }
.event-input, .event-output { line-height: 1.35; overflow-wrap: anywhere; }
.event-input { color: var(--muted); font-size: 13px; }
.event-output { font-family: Consolas, monospace; font-size: 13px; }
.reference-band { max-width: 1520px; margin: 12px auto 0; background: var(--paper); border: 1px solid var(--line); border-radius: 6px; }
.reference-grid { display: grid; grid-template-columns: 1fr 1fr; }
.reference-grid .text-block { padding: 13px 16px; border-top: 0; }
.reference-grid .text-block:nth-child(odd) { border-right: 1px solid var(--line); }
.footer-note { max-width: 1520px; margin: 12px auto 0; color: var(--muted); font-size: 12px; }
@media (max-width: 1050px) {
  .metrics { grid-template-columns: repeat(4, 1fr); }
  .workspace { grid-template-columns: 1fr; }
  .timeline { height: 390px; }
}
@media (max-width: 700px) {
  .topbar, .toolbar-band, main { padding-left: 12px; padding-right: 12px; }
  .topline { display: block; }
  .status-badge { display: inline-block; margin-top: 10px; }
  .toolbar { grid-template-columns: 40px minmax(0, 1fr) 40px; }
  .segmented { grid-column: 1 / -1; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .live-grid, .reference-grid { grid-template-columns: 1fr; }
  .reference-grid .text-block:nth-child(odd) { border-right: 0; border-bottom: 1px solid var(--line); }
  .event { grid-template-columns: 58px 72px 1fr; }
  .event-output { grid-column: 3; }
  h1 { font-size: 23px; }
}
</style>
</head>
<body>
<header class="topbar">
  <div class="topline">
    <div>
      <h1>Chinese Speech → Incremental English → Pseudo ASL Gloss</h1>
      <p class="subtitle">Task 5 v0 · synchronized end-to-end demonstration</p>
    </div>
    <div class="status-badge">Weak pseudo ASL labels · not gold ASL</div>
  </div>
</header>
<section class="toolbar-band">
  <div class="toolbar">
    <button id="prevSample" class="icon-btn" title="Previous sample" aria-label="Previous sample">‹</button>
    <select id="sampleSelect" aria-label="Select sample"></select>
    <button id="nextSample" class="icon-btn" title="Next sample" aria-label="Next sample">›</button>
    <div class="segmented" aria-label="Quality filter">
      <button data-filter="all" class="active">All</button>
      <button data-filter="high">High</button>
      <button data-filter="medium">Medium</button>
      <button data-filter="low">Low</button>
    </div>
  </div>
</section>
<main>
  <section class="metrics" id="metrics"></section>
  <section class="workspace">
    <div class="panel">
      <div class="panel-head"><h2>Reference video and source</h2><span class="sample-meta" id="sampleMeta"></span></div>
      <div class="panel-body">
        <div class="media-label"><span>How2Sign ASL reference video</span><span class="verified" id="videoStatus">VERIFIED TRAIN CLIP</span></div>
        <video id="signVideo" class="sign-video" controls preload="metadata" playsinline></video>
        <div class="media-label"><span>Synthetic Mandarin speech input</span><span>Qwen3-TTS · 16 kHz mono</span></div>
        <audio id="audio" controls preload="metadata"></audio>
        <div class="audio-controls">
          <button id="togglePlay" class="command-btn primary" title="Play or pause synchronized replay">Play</button>
          <button id="reset" class="command-btn" title="Reset replay">Reset</button>
          <button id="prevEvent" class="icon-btn" title="Previous recorded event" aria-label="Previous event">‹</button>
          <button id="nextEvent" class="icon-btn" title="Next recorded event" aria-label="Next event">›</button>
          <span class="playback-status" id="playbackStatus">Audio-synced mode</span>
        </div>
        <div class="progress-wrap" id="progressWrap">
          <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
          <div id="markers"></div>
        </div>
        <div class="time-readout"><span id="currentTime">0.00 s</span><span id="duration">0.00 s</span></div>
        <div class="text-block"><div class="label">Chinese source</div><div class="text" id="zh"></div></div>
        <div class="text-block"><div class="label">Independent Mandarin ASR audit</div><div class="text" id="zhAsr"></div></div>
        <div class="text-block"><div class="label">Official How2Sign English</div><div class="text" id="enRef"></div></div>
        <div class="text-block"><div class="label">SeamlessStreaming final English</div><div class="text" id="enHyp"></div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>Synchronized commitments</h2><span class="action-pill" id="latestAction">WAITING</span></div>
      <div class="panel-body">
        <div class="live-grid">
          <div class="live-box english"><div class="live-title"><span>Committed English</span><span id="sourceState">INCOMPLETE</span></div><div class="live-text" id="liveEnglish">—</div></div>
          <div class="live-box gloss"><div class="live-title"><span>Committed pseudo gloss</span><span id="glossCount">0 tokens</span></div><div class="live-text" id="liveGloss">—</div></div>
        </div>
        <div class="timeline" id="timeline"></div>
      </div>
    </div>
  </section>
  <section class="reference-band">
    <div class="reference-grid">
      <div class="text-block"><div class="label">Qwen pseudo-gloss reference</div><div class="text mono" id="glossRef"></div></div>
      <div class="text-block"><div class="label">Fine-tune 2 final prediction</div><div class="text mono" id="glossHyp"></div></div>
    </div>
  </section>
  <p class="footer-note">The ASL video is the sentence-level How2Sign train RGB-front reference for the same SENTENCE_NAME. It is not frame-aligned to the synthetic Mandarin audio. English metrics use official How2Sign transcriptions; gloss metrics measure agreement with Qwen-generated pseudo labels, not ASL linguistic correctness.</p>
</main>
<script>
const DATA = __DATA__;
const samples = DATA.samples;
const audio = document.getElementById('audio');
const signVideo = document.getElementById('signVideo');
const select = document.getElementById('sampleSelect');
let visibleSamples = samples;
let sample = samples[0];
let combinedEvents = [];
let stepIndex = -1;
let stepMode = false;

const fmt = (value, digits = 2) => value === null || value === undefined ? '—' : Number(value).toFixed(digits);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const bucketLabel = value => value === 'high' ? 'High' : value === 'medium' ? 'Medium' : 'Low';

function buildCombinedEvents(s) {
  const events = [];
  s.frontendEvents.forEach((event, index) => events.push({
    time: Number(event.audio_read_seconds), order: 0, index,
    type: 'EN COMMIT', css: 'en',
    input: event.committed_english_prefix,
    output: (event.emitted_english_tokens || []).join(' '),
    english: event.committed_english_prefix,
    gloss: null,
    sourceStatus: event.source_status
  }));
  s.actionEvents.forEach((event, index) => events.push({
    time: Number(event.audio_read_seconds), order: 1, index,
    type: event.predicted_action_class,
    css: String(event.predicted_action_class || '').toLowerCase(),
    input: event.observed_english_prefix,
    output: event.predicted_action,
    english: event.observed_english_prefix,
    gloss: (event.committed_gloss_after || []).join(' '),
    sourceStatus: event.source_status
  }));
  events.sort((a, b) => a.time - b.time || a.order - b.order || a.index - b.index);
  return events;
}

function metric(label, value, cls = '') {
  return `<div class="metric ${cls}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`;
}

function renderMetrics() {
  const qualityClass = sample.similarity >= .75 ? 'good' : sample.similarity >= .35 ? 'warn' : 'bad';
  const leadClass = sample.glossLead > 0.25 ? 'good' : 'warn';
  document.getElementById('metrics').innerHTML = [
    metric('Audio duration', `${fmt(sample.duration)} s`),
    metric('First English', `${fmt(sample.firstEnglish)} s`, sample.firstEnglish < sample.duration ? 'good' : 'warn'),
    metric('Average lagging', `${fmt(sample.averageLaggingMs / 1000)} s`),
    metric('First gloss', `${fmt(sample.firstGloss)} s`, sample.firstGloss < sample.duration ? 'good' : 'warn'),
    metric('Before audio end', `${fmt(sample.glossLead)} s`, leadClass),
    metric('English WER', fmt(sample.enWer, 3), sample.enWer <= .35 ? 'good' : sample.enWer <= .7 ? 'warn' : 'bad'),
    metric('Gloss similarity', fmt(sample.similarity, 3), qualityClass),
    metric('Gloss token F1', fmt(sample.tokenF1, 3), qualityClass)
  ].join('');
}

function renderTimeline(revealCount = 0) {
  const htmlRows = combinedEvents.map((event, index) => `
    <div class="event ${index < revealCount ? 'revealed' : ''} ${index === revealCount - 1 ? 'current' : ''}" data-event-index="${index}">
      <div class="event-time">${fmt(event.time)} s</div>
      <div class="event-type ${event.css}">${esc(event.type)}</div>
      <div class="event-input">${esc(event.input)}</div>
      <div class="event-output">${esc(event.output)}</div>
    </div>`).join('');
  document.getElementById('timeline').innerHTML = htmlRows;
  const current = document.querySelector('.event.current');
  if (current) current.scrollIntoView({block: 'nearest'});
}

function setLiveState(revealCount) {
  const revealed = combinedEvents.slice(0, revealCount);
  const englishEvent = [...revealed].reverse().find(event => event.type === 'EN COMMIT');
  const actionEvent = [...revealed].reverse().find(event => event.type !== 'EN COMMIT');
  const english = actionEvent?.english || englishEvent?.english || '';
  const gloss = actionEvent?.gloss || '';
  document.getElementById('liveEnglish').textContent = english || '—';
  document.getElementById('liveGloss').textContent = gloss || '—';
  document.getElementById('sourceState').textContent = actionEvent?.sourceStatus || englishEvent?.sourceStatus || 'INCOMPLETE';
  document.getElementById('latestAction').textContent = actionEvent?.output || (englishEvent ? 'EN COMMIT' : 'WAITING');
  document.getElementById('latestAction').classList.toggle('invalid', actionEvent?.type === 'INVALID');
  document.getElementById('glossCount').textContent = `${gloss ? gloss.split(/\s+/).length : 0} tokens`;
}

function revealByAudioTime(time) {
  const revealCount = combinedEvents.filter(event => event.time <= time + .001).length;
  renderTimeline(revealCount);
  setLiveState(revealCount);
  document.getElementById('progressFill').style.width = `${Math.min(100, time / sample.duration * 100)}%`;
  document.getElementById('currentTime').textContent = `${fmt(time)} s`;
}

function renderStep() {
  const revealCount = stepIndex + 1;
  renderTimeline(revealCount);
  setLiveState(revealCount);
  const time = stepIndex >= 0 ? combinedEvents[stepIndex].time : 0;
  audio.currentTime = Math.min(time, sample.duration);
  document.getElementById('progressFill').style.width = `${Math.min(100, time / sample.duration * 100)}%`;
  document.getElementById('currentTime').textContent = `${fmt(time)} s`;
  document.getElementById('playbackStatus').textContent = `Step mode · ${Math.max(0, revealCount)}/${combinedEvents.length}`;
  document.getElementById('prevEvent').disabled = stepIndex < 0;
  document.getElementById('nextEvent').disabled = stepIndex >= combinedEvents.length - 1;
}

function renderMarkers() {
  const markers = combinedEvents.map(event => {
    const left = Math.min(100, Math.max(0, event.time / sample.duration * 100));
    return `<span class="marker ${event.type === 'EN COMMIT' ? 'english' : ''}" style="left:${left}%" title="${esc(event.type)} at ${fmt(event.time)} s"></span>`;
  }).join('');
  document.getElementById('markers').innerHTML = markers;
}

function loadSample(id) {
  sample = samples.find(item => item.id === id) || visibleSamples[0] || samples[0];
  combinedEvents = buildCombinedEvents(sample);
  stepIndex = -1;
  stepMode = false;
  audio.pause();
  signVideo.pause();
  audio.src = sample.audio;
  audio.load();
  if (sample.video) {
    signVideo.src = sample.video;
    signVideo.load();
    signVideo.hidden = false;
    document.getElementById('videoStatus').textContent = sample.video_official_text_match ? 'ID + TEXT VERIFIED' : 'ID VERIFIED';
  } else {
    signVideo.removeAttribute('src');
    signVideo.load();
    signVideo.hidden = true;
    document.getElementById('videoStatus').textContent = 'VIDEO MISSING';
  }
  const invalidNote = sample.invalidActionCount ? ` · ${sample.invalidActionCount} invalid actions` : '';
  document.getElementById('sampleMeta').textContent = `#${String(sample.index).padStart(3,'0')} · ${sample.speaker} · ${sample.id}${invalidNote}`;
  document.getElementById('zh').textContent = sample.zh;
  document.getElementById('zhAsr').textContent = `${sample.zhAsr || '—'}${sample.zhCer === null || sample.zhCer === undefined ? '' : `  (CER ${fmt(sample.zhCer, 3)})`}`;
  document.getElementById('enRef').textContent = sample.enRef;
  document.getElementById('enHyp').textContent = sample.enHyp;
  document.getElementById('glossRef').textContent = sample.glossRef;
  document.getElementById('glossHyp').textContent = sample.glossHyp;
  document.getElementById('duration').textContent = `${fmt(sample.duration)} s`;
  document.getElementById('playbackStatus').textContent = 'Audio-synced mode';
  document.getElementById('togglePlay').textContent = 'Play';
  renderMetrics();
  renderMarkers();
  revealByAudioTime(0);
  updateNav();
}

function populateSelect(preferredId = null) {
  select.innerHTML = visibleSamples.map(item => {
    const prefix = item.enHyp.length > 62 ? `${item.enHyp.slice(0, 62)}…` : item.enHyp;
    const issue = item.invalidActionCount ? ` · INVALID ${item.invalidActionCount}` : '';
    return `<option value="${esc(item.id)}">#${String(item.index).padStart(3,'0')} · ${bucketLabel(item.bucket)} ${fmt(item.similarity, 3)}${issue} · ${esc(prefix)}</option>`;
  }).join('');
  const selected = visibleSamples.some(item => item.id === preferredId) ? preferredId : visibleSamples[0]?.id;
  if (selected) {
    select.value = selected;
    loadSample(selected);
  }
}

function updateNav() {
  const index = visibleSamples.findIndex(item => item.id === sample.id);
  document.getElementById('prevSample').disabled = index <= 0;
  document.getElementById('nextSample').disabled = index < 0 || index >= visibleSamples.length - 1;
  if (visibleSamples.some(item => item.id === sample.id)) select.value = sample.id;
}

audio.addEventListener('timeupdate', () => {
  if (!stepMode) revealByAudioTime(audio.currentTime);
});
audio.addEventListener('play', () => {
  stepMode = false;
  document.getElementById('playbackStatus').textContent = 'Audio-synced mode';
  document.getElementById('togglePlay').textContent = 'Pause';
});
audio.addEventListener('pause', () => document.getElementById('togglePlay').textContent = 'Play');
audio.addEventListener('ended', () => {
  revealByAudioTime(sample.duration + .01);
  document.getElementById('togglePlay').textContent = 'Play';
});
document.getElementById('togglePlay').addEventListener('click', () => audio.paused ? audio.play() : audio.pause());
document.getElementById('reset').addEventListener('click', () => {
  audio.pause(); stepMode = false; stepIndex = -1; audio.currentTime = 0;
  document.getElementById('playbackStatus').textContent = 'Audio-synced mode';
  revealByAudioTime(0);
});
document.getElementById('prevEvent').addEventListener('click', () => {
  audio.pause(); stepMode = true; stepIndex = Math.max(-1, stepIndex - 1); renderStep();
});
document.getElementById('nextEvent').addEventListener('click', () => {
  audio.pause(); stepMode = true; stepIndex = Math.min(combinedEvents.length - 1, stepIndex + 1); renderStep();
});
select.addEventListener('change', () => loadSample(select.value));
document.getElementById('prevSample').addEventListener('click', () => {
  const index = visibleSamples.findIndex(item => item.id === sample.id); if (index > 0) loadSample(visibleSamples[index - 1].id);
});
document.getElementById('nextSample').addEventListener('click', () => {
  const index = visibleSamples.findIndex(item => item.id === sample.id); if (index >= 0 && index < visibleSamples.length - 1) loadSample(visibleSamples[index + 1].id);
});
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-filter]').forEach(item => item.classList.toggle('active', item === button));
  const filter = button.dataset.filter;
  visibleSamples = filter === 'all' ? samples : samples.filter(item => item.bucket === filter);
  populateSelect(sample.id);
}));

populateSelect(DATA.initialSampleId);
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-jsonl", type=Path, required=True)
    parser.add_argument("--asr-preview-jsonl", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--video-manifest-jsonl", type=Path)
    parser.add_argument(
        "--sample-indices",
        help="Optional comma-separated pilot indices for a compact demo viewer",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-index", type=int, default=34)
    args = parser.parse_args()

    rows = read_jsonl(args.e2e_jsonl)
    if args.sample_indices:
        selected_indices = {int(value) for value in args.sample_indices.split(",")}
        rows = [row for row in rows if int(row["pilot_index"]) in selected_indices]
        if len(rows) != len(selected_indices):
            raise SystemExit("At least one requested sample index was not found")
    asr_rows = read_jsonl(args.asr_preview_jsonl)
    video_rows = read_jsonl(args.video_manifest_jsonl) if args.video_manifest_jsonl else []
    selection = read_json(args.selection_report)
    samples = prepare_samples(rows, asr_rows, video_rows, args.output.parent, args.root)
    initial = next(
        (sample["id"] for sample in samples if sample["index"] == args.initial_index),
        samples[0]["id"],
    )
    payload = {
        "samples": samples,
        "initialSampleId": initial,
        "selectionSummary": selection,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    document = HTML_TEMPLATE.replace("__DATA__", serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8", newline="\n")
    missing_audio = sum(not sample["audio_exists"] for sample in samples)
    missing_video = sum(not sample["video_exists"] for sample in samples)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": len(samples),
                "missing_audio_count": missing_audio,
                "missing_video_count": missing_video,
                "initial_sample_id": initial,
                "html_bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if missing_audio:
        raise SystemExit(1)
    if args.video_manifest_jsonl and missing_video:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
