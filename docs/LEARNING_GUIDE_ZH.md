# 从哪里开始学习 CausalGloss

这份仓库的目的不是只放一份“最终 trainer”，而是让你能逐步回答：数据从哪里来、监督标签怎么生成、训练什么、推理时模型看到什么、论文数字如何得到。

先不用重新训练。第一遍只读下列六个模块，每次看一个模块即可。运行 `python scripts/paper.py list` 可以找到对应原文件。

## 1. 推理入口：先理解模型在做什么

阅读 `fine_tuning_v2_semantic_v2_2_full/scripts/evaluate_free_running_rollout.py`，先看 `initialize_state`、`state_prompt` 和 `main` 中的行动循环。

- 输入：已经读到的英文前缀、已提交的 gloss、源句是否结束。
- 输出：一个 gloss、`<WAIT>` 或 `<END>`。一次 WRITE 不读取新的英文，因此同一个前缀可以连续 WRITE。
- 真实运行使用模型自己的历史，不把下一步正确 gloss 偷给模型。
- WAIT 才推进英文输入；END 是否在完整源句之后决定 completion。
- `max_actions` 是避免无限循环的评测保护，不是学习成功的证据。

先画出一条句子的行动轨迹，再看损失和神经网络细节。

## 2. 训练样本和 loss

阅读同目录的 `causal_action_common.py`，再看 `train_qwen_lora_causal_ddp.py`。

重点回答：一条 action 样本有哪些字段？prompt、答案、padding 哪些位置被标记 `-100`？batch 怎样合并？LoRA 从哪个 adapter 继续训练？有效 batch 为什么是每卡 batch × 卡数 × 梯度累计？

教师强制训练与自由推理不同：训练时提供标准历史并计算答案位置的 loss；自由推理时只能使用已经生成的历史。

## 3. 这些 action 标签从哪里来

依次读：corrected lexical alignment → prefix semantic scoring → confident/burst alignment → Treebank detokenization → 第一段 policy 训练 → 第二段的 WAIT-balanced + recovery data。第一段选中的 v2.2.1 初始化在 534,877 条 Treebank action 上训练；不要把第二段的 WAIT-balanced 数据构造误记为第一段的训练输入。这些都是监督构造步骤，不是推理时额外读取未来答案的模块。

注意：`semantic_causal_alignment_v2_pilot500` 和 `v2_1_pilot500` 中仍有被 full 管线调用的 helper。名称看起来旧，不代表可以删掉。

## 4. 为什么增加 recovery

阅读 `state_recovery_v2_3_full/scripts/prepare_v2_3_manifests.py` 和对应 full config。最后的训练包含 367,267 条 balanced action 和 71,836 条 recovery action，共 439,103 条，涉及 18,632 个训练 parent。

核对每一种不完整历史如何构造，以及为什么它有助于源句读完后继续补出内容。最后三个主模型选用的都是 `adapter_final`；S1 名为 `adapter_best` 的文件实际是初始化，不能误用。

## 5. 两类 baseline 和消融

固定 Wait-k 是按各自时序重新构造 action 后训练的基线。最终 TransLLaMA-style 是 `transllama_learned_adaptive_20260918` 中独立 SimAlign 轨迹训练的模型，不是早期仅修改解码规则的实验。

消融在 diagnostic/confirmation 250 上评估，不是 locked500；“没有显著差异”不等于两者完全一样。先看实验协议，再看模型结果。

## 6. 从逐样本轨迹到论文

最后学习 [验证说明](VERIFICATION.md)：Token F1 的重复词如何计数、SequenceMatcher 衡量什么、LAAL 单位何时是 token/何时是秒、为什么三随机种子要报告均值和样本标准差，以及 paired bootstrap 为什么要按同一 parent 配对。

区分四件事：源码齐全、已有输出可重算、checkpoint 可以推理、从零重训能复现。前一项通过不能自动证明后一项通过。

## 边界

此仓库只讲提交论文的 gloss 研究。G2P-DDM、pose、SMPL-X 和视频生成属于后续扩展，不应混入本文实验或当作本文已有能力。
