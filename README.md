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
| CLERC 与论文统计复核 | [evaluation/clerc](evaluation/clerc)、[scripts/verification](scripts/verification) | §5、Tables 1–2 |
| 最终语音链与完整合成语料构建 | [speech](speech) | §4、§5.2 |
| 最终实验配置、图 1、已复核数值 | [configs](configs)、[figures](figures)、[results](results) | Fig. 1、Tables 1–2 |

**建议第一遍只看三个文件：**

1. [evaluate_free_running_rollout.py](method/causalgloss/evaluate_free_running_rollout.py)：怎样 WAIT、WRITE、END。
2. [causal_action_common.py](method/causalgloss/causal_action_common.py)：训练样本和 batch。
3. [train_qwen_lora_causal_ddp.py](method/causalgloss/train_qwen_lora_causal_ddp.py)：模型、loss、训练和保存。

进一步阅读：[中文学习路线](docs/LEARNING_GUIDE_ZH.md) · [论文—代码映射](docs/PAPER_TO_CODE.md) · [运行说明](docs/REPRODUCTION.md) · [保留/剔除标准](docs/SCOPE.md)。

## 最轻量的检查

```bash
python scripts/paper.py list
python scripts/check_release.py
python -m unittest discover -s tests -v
```

数值复核需 NumPy；训练/推理需相应 ML 依赖、冻结数据和正确 checkpoint：

```bash
python -m unittest discover -s tests/verification -v
python scripts/paper.py run --dry-run rollout -- --help
```

本版从 82 份论文对应 rollout、35,535 条记录独立复算论文数值及相关 bootstrap，详见[验证说明](docs/VERIFICATION.md)。这不等于从头重新训练。82 项已选权重推理和 30 项训练计划分别记录在 `configs/reproduction/`，实际执行状态以最终验证报告为准。

## 数据、模型和边界

冻结数据、逐样本结果、选定 LoRA 和必要合成音频放在同一私有仓库的 [Releases](https://github.com/OwenLmz/CasualGloss/releases)，不塞入 Git 历史。`provenance/release_assets.json` 记录每个实际附件的 SHA/大小；安装器逐文件校验并保留原数据。官方 Qwen 等基座按固定版本另行获取。见[数据与模型说明](docs/DATA_AND_MODELS.md)及[运行说明](docs/REPRODUCTION.md)。未有附件清单或未通过下载校验时，不应认为“数据已经包含”。

弱 pseudo-gloss 不是专家 ASL 金标准；语音过度生成仍未解决；没有声称视频生成或专家可理解性验证。数据、模型和作者代码许可分开处理，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

数据血缘和既有报告边界已明确记录在[验证说明](docs/VERIFICATION.md)：上游已见 locked500 样本、弱目标实际生成器、参考长度参与预算、语音块描述及 native no-recovery 完成率范围。没有修改原目标、划分、预算或结果来掩盖这些事实。

原始服务器代码、模型、数据及提交 PDF 未修改。必要路径适配和删除未报告输出分支均记录在 [source_manifest.json](provenance/source_manifest.json)，保留原始和当前校验和。旧实验已经从当前代码树移出，但之前 Git 提交仍可恢复；没有改写提交历史。
