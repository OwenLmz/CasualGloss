# 最终论文 → 代码

依据提交 PDF SHA-256 `5f7793358d54f9d754aa2fa3c8f1d5fbb723ed4506105c21ca886771e9856e1a`。逐文件来源、章节和修改范围见 `provenance/source_manifest.json`。

| 论文位置 | 对应实现 |
|---|---|
| §3.1 WAIT/WRITE/END、源前缀和自身历史 | `method/causalgloss/evaluate_free_running_rollout.py` |
| §3.2 词法 anchors / occurrence matching | `data_preparation/lexical_alignment/`、`data_preparation/semantic_alignment/finalize_semantic_alignment_v2_1.py` |
| §3.2 prefix semantic evidence、confidence / rescue / negation / burst | `data_preparation/semantic_alignment/score_prefix_semantics.py`、`build_confident_burst_alignment_v2_2.py` |
| §3.3 WAIT balancing 与 recovery histories | `data_preparation/recovery/` |
| §4 上游 weak-gloss 模型 | `data_preparation/pseudo_gloss/`、`training/upstream_gloss/` |
| §4 主模型训练与选择 | `method/causalgloss/`、`configs/method/` |
| §4 固定 k=1,3,5,7 | `baselines/fixed_waitk/`、`configs/fixed_waitk/`，训练复用主 trainer |
| §4 最终 TransLLaMA-style / LR选择 / k=1…8 | `baselines/transllama/`、`configs/transllama/` |
| §5.1 Table 1 / Fig.1 | `scripts/verification/`、`figures/`、`evaluation/analysis/evaluate_locked500_bleu.py` |
| §5.2 CLERC / completed-only | `evaluation/clerc/`、`scripts/verification/` |
| §5.2 语音链 / lexical-only 对照 | `speech/frontend/`、`baselines/lexical_control/` |
| §4、Abstract 完整中文语料 | `speech/corpus/`、`configs/speech/` |
| §5.3 Table 2 | `ablations/`、`configs/ablations/`；no-recovery / guard-only 复用主模型 evaluator 和 Stage A checkpoint |

不在论文中的旧 learned-baseline、local-agreement、5K 模型试验和其他 speech frontends 不在当前代码树。必要预处理函数虽有旧版名称，保留理由是实际数据依赖，不是保留旧模型实验。
