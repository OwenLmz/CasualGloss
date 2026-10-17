# CausalGloss — final-paper code

**CausalGloss: Multi-Evidence Action Control for a Simultaneous Mandarin Speech-to-ASL Gloss Cascade**

Mingze Li · Satoshi Nakamura · ICASSP 2027 submission

本仓库只保留最终提交论文的方法、必要的数据处理、正式基线、消融和评测，**不是全部实验历史归档**。仓库名 `CasualGloss` 沿用原名，方法名为 `CausalGloss`；不包含后续 gloss→pose/video 扩展。

## 代码直接从这里看

| 模块 | 位置 | 论文对应 |
|---|---|---|
| **主方法：数据加载、训练、自由推理** | [method/causalgloss](method/causalgloss) | §3、§4 |
| 上游 weak-gloss 模型训练 | [training/upstream_gloss](training/upstream_gloss) | §4 |
| 弱标签、词法/语义对齐、recovery 数据 | [data_preparation](data_preparation) | §3.2–3.3、§4 |
| 固定 Wait-k、最终 TransLLaMA-style、语音 lexical control | [baselines](baselines) | §4、§5.1–5.2 |
| 五个组件消融 | [ablations](ablations) | §5.3、Table 2 |
| CLERC 与论文指标计算 | [evaluation/clerc](evaluation/clerc)、[scripts/verification](scripts/verification) | §5、Tables 1–2 |
| 最终语音链与完整合成语料构建 | [speech](speech) | §4、§5.2 |
| 最终实验配置、图 1、论文数值 | [configs](configs)、[figures](figures)、[results](results) | Fig. 1、Tables 1–2 |

**建议第一遍只看三个文件：**

1. [evaluate_free_running_rollout.py](method/causalgloss/evaluate_free_running_rollout.py)：怎样 WAIT、WRITE、END。
2. [causal_action_common.py](method/causalgloss/causal_action_common.py)：训练样本和 batch。
3. [train_qwen_lora_causal_ddp.py](method/causalgloss/train_qwen_lora_causal_ddp.py)：模型、loss、训练和保存。

进一步阅读：[中文学习路线](docs/LEARNING_GUIDE_ZH.md) · [论文—代码映射](docs/PAPER_TO_CODE.md) · [运行说明](docs/REPRODUCTION.md) · [评估协议](docs/EVALUATION.md) · [仓库范围](docs/SCOPE.md)。

## 最轻量的检查

```bash
python scripts/paper.py list
python scripts/check_release.py
python -m unittest discover -s tests -v
```

指标计算需 NumPy；训练/推理需相应 ML 依赖、冻结数据和正确 checkpoint：

```bash
python -m unittest discover -s tests/verification -v
python scripts/paper.py run --dry-run rollout -- --help
```

`results/paper_results.json` 保留原论文预测对应的聚合数值，供查表和绘图使用；`results/expected_paper_results.json` 保留论文显示值和计分输入约定。训练、推理及数据处理命令见[运行说明](docs/REPRODUCTION.md)。

## 数据、模型和边界

冻结数据、逐样本结果、选定 LoRA 和必要合成音频放在同一私有仓库的 [Release](https://github.com/OwenLmz/CasualGloss/releases/tag/paper-reproduction-20261016)，不塞入 Git 历史。`provenance/release_assets.json` 记录附件目录及校验和；官方 Qwen 等基座按固定版本另行获取。见[数据与模型说明](docs/DATA_AND_MODELS.md)及[运行说明](docs/REPRODUCTION.md)。

弱 pseudo-gloss 不是专家 ASL 金标准；语音过度生成仍未解决；没有声称视频生成或专家可理解性验证。数据、模型和作者代码许可分开处理，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

代码来源和必要的可移植性适配见 [source_manifest.json](provenance/source_manifest.json)。实验的数据关系、生成预算及指标口径见[数据与模型说明](docs/DATA_AND_MODELS.md)和[评估协议](docs/EVALUATION.md)。
