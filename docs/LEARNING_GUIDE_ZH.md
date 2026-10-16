# 按最终论文学习代码

先理解系统做什么，再读数据、训练和实验。暂时不要一上来重训全部模型。

## 1. 推理：模型每次看到什么、做什么

从 `method/causalgloss/evaluate_free_running_rollout.py` 的 `initialize_state`、`state_prompt`、`main` 开始。

- 输入：英文前缀、已经生成的 gloss、源句是否完整。
- WRITE 提交一个 gloss，不读新词；WAIT 才推进英文前缀；END 结束。
- 连续 WRITE 形成 burst。推理历史来自模型自身，不提供正确答案历史。
- 完成率要求在完整源句之后 END；行动次数上限只是防止无限循环。

## 2. 数据与 loss

读同目录 `causal_action_common.py`，再看 trainer。

重点回答：一条 action 样本有哪些字段？prompt、答案、padding 哪些位置不算 loss？LoRA 从哪个 adapter 继续？有效 batch 如何由每卡 batch、GPU 数和梯度累计形成？

## 3. 标签与时序怎样生成

按 `data_preparation/pseudo_gloss` → `lexical_alignment` → `semantic_alignment` 阅读。最终流程确实复用了两个较早写出的 helper：`score_prefix_semantics.py` 和 `finalize_semantic_alignment_v2_1.py`。它们现已放在同一个语义处理目录；保留它们不等于保留早期 v2.1 模型实验。

这些模块生成训练监督；不能理解为推理时可以读取未来句子或标准答案。

## 4. 两段 policy 训练与 recovery

- Stage A：534,877 条 Treebank action，初始化来自上游 weak-gloss adapter。
- Stage B：367,267 条 balanced action + 71,836 条 recovery action，共 439,103 条；初始化来自选中 Stage A adapter。
- 三个最终种子为 20260829、20260904、20260905，均使用选定的 `adapter_final`。

读 `data_preparation/recovery/prepare_v2_3_manifests.py`，理解四类不完整历史。本文完整方法、action-only/no-recovery 和 guard-only 对照都需要保留。

## 5. 比较方法与消融

- `baselines/fixed_waitk`：各 k 自己构造监督并训练，不是在主模型上临时加等待。
- `baselines/transllama`：最终独立 SimAlign 方法；LR screen 是论文明确报告的选择过程，不能删掉。
- `ablations`：五项组件消融，保留独立 trainer/common/evaluator，不能因为同名就和主模型混用。
- `baselines/lexical_control`：仅供论文语音链中的旧 lexical-only 内部对照，不是最终外部基线。

## 6. 从预测到论文数字

最后读 `scripts/verification/verify_paper_results.py` 的 `overlap`、`laal`、`normalize_content`，再读三种 seed/parent bootstrap。Table 2 用 diagnostic 250，locked500 是另外的测试集。CLERC 的 1,200 参考不等于 1,200 个独立 source。

正文的一个边界已保留在验证说明：native no-recovery 完成率是 249/250；不能把所有列出的控制都说成 100%。语音输出长度问题也不能因运行完成而忽略。

根目录的 `scripts/paper.py` 是导航和运行入口；主方法本身在 `method/causalgloss`。先看真正实现，再使用入口，不需要在深层 archive 中找代码。
