# 最终论文结果复核 / Final-paper verification

**当前状态：验证进行中，尚未宣称完整复现。** 以 [reproduction_status.json](../results/reproduction_status.json) 的逐项实测记录为准。本工具只验证最终提交 PDF 中呈现的实验，不是全部服务器实验的归档。新增检查不是原论文结果；新逐样本输出应留在独立私有运行目录，不能冒充历史预测。

论文锚点为 `20260923032900_403947_8500 (1).pdf`，SHA-256：

`5f7793358d54f9d754aa2fa3c8f1d5fbb723ed4506105c21ca886771e9856e1a`

原始 PDF、服务器源码、训练 checkpoint 和历史逐样本输出均未修改。

## 三种“复现”必须分开

| 层级 | 当前能够成立的结论 | 不能由此推出 |
|---|---|---|
| 冻结预测重新计分 | 83 项输入、35,535 条预测已独立重算；已核对的论文显示值/CI 无数值差异。下面保留 CI 方法范围说明。 | 没有重新生成预测，不等于权重推理或训练通过。 |
| 选定 checkpoint 重新推理 | 40 次调用、82 份输出、35,535 条记录已实际完成；输入与指标完整性通过，但新结果与论文并非逐项精确一致。 | 不能把运行成功或主要均值接近称为所有纸面数字复现。 |
| 新训练 | 30 项训练计划可检查，执行必须明确加 `--execute`；完整新训练尚未运行。 | 配置齐全、dry-run 通过、复用旧 adapter 都不能称为从头训练复现。 |

私有 Release 安装、原始波形 QC、语料清洗及干净环境安装各有独立状态。环境锁只是当前服务器记录，不是历史环境逐包证明，也不是新环境安装成功的保证。操作命令见 [REPRODUCTION.md](REPRODUCTION.md)。

## 2026-10-17 实际重新推理结果

完整记录见 [checkpoint_reinference_20261017.json](../results/checkpoint_reinference_20261017.json)。独立验证了全部 40 个 receipt、82 份输出、130 份原始输入/adapter/参考文件及 161 个执行源码核心文件。ID 顺序、输入/参考和存储指标重算无错误。

| 指标 | 论文显示值 | 本次重新推理 |
|---|---:|---:|
| Locked500 CausalGloss F1 均值 | 0.7047 | 0.70386672 |
| Locked500 CausalGloss LAAL 均值 | 1.7823 | 1.77919386 |
| CLERC CausalGloss F1 均值 | 约 0.649 | 0.64871802 |
| completed-only F1 差 | 0.0357 | 0.03471956 |
| completed-only 各种子样本数 | 349 / 344 / 353 | 349 / 343 / 352 |
| Speech 三种子 F1 均值 | 0.3944 | 0.39248431 |

32,912/35,535 条预测 token 完全相同（92.6185%），但仅 1/82 份 rollout 的全部确定性输出一致。未设置事后近似通过阈值。原始纸面 verifier 因 completed-only 样本数变化拒绝通过；分节重算还记录了显示值及 CI 差异，不能掩盖为成功。**结论是推理工作流已完整执行和核验，不是所有论文数字的精确复现。** 数值漂移的原因尚未被完全隔离，硬件/精度/环境差异只是合理候选，不能断言已证明原因。

历史冻结结果重计分仍可精确复得原论文显示值。这两类结果必须分别保留；不得用新预测替换历史文件、改 expected 或调整科学参数来制造一致。

## 验证范围

使用明确 allowlist 中的 **83 个输入文件**：82 份历史预测输出，共 **35,535 条逐样本记录**，另加一份 401-prompt CLERC 参考 manifest。原始文本不收入 Git；仓库只保留代码、数值汇总、输入相对路径及校验和。

| 论文位置 | 保留的验证输入与操作 |
|---|---|
| Table 1 / Figure 1 | 35 份 locked500 输出：CausalGloss 三种子、固定 Wait-k 八份、最终独立训练 learned baseline 的 3×8 份。核对 parent IDs、原文/参考一致性，独立重算多重集 Token F1、序列相似度、LAAL、无平滑 corpus BLEU-4、均值与样本 SD。 |
| CLERC 主分析与描述性曲线 | 35 份 401-prompt 输出：CausalGloss 三种子、learned k=1…8 各三种子、fixed k=1/3 单种子及 k=5/7 三种子。直接从预测 token 和 1,200 条嵌套人类参考重算归一化 best-reference F1 / LAAL / 完成率。 |
| Completed-only 稳健性检查 | 重新选择每个 learned k=5 种子完成的 prompts，在同一子集上配对 CausalGloss；重算 20,000 次 seed/parent bootstrap。此项是 post-hoc，不替代全部 401 prompts 的主分析。 |
| Table 2 与 guard-only 对照 | 8 份 diagnostic 输出，各 250 parents：full、五个组件消融、native no-recovery、guard-only。重算配对差值；LAAL 只使用双方均有限的样本。 |
| Speech cascade | 4 份 locked500 输出：论文使用的 speech lexical control 与三个 CausalGloss 种子。重算 F1、raw / computation-aware LAAL、完成率、RTF 和长度误差。 |

CLERC lexical control、LocalAgreement、旧 learned baseline、筛选后未选中的训练候选，以及新推理检查，不进入这个数值工具的输入清单。Speech lexical control 则保留，因为论文明确报告了该比较。混有额外实验的历史 summary / per-source score 文件也不作为输入；所有点估计直接从对应预测和参考计算。

## 实际验证层级

- 独立重新计分历史逐样本输出，而非复制历史 `metrics.json`。
- 根据原始数据顺序和 RNG 复算论文报告的关键 bootstrap CI。locked learned 和 CLERC 默认 20,000 次；fixed locked500、Table 2 及 speech 原方法 10,000 次。
- 已保存输入 SHA-256 和与远端原件的匹配状态；只检查实际使用的 83 项，即使本地私有缓存另有历史文件，也不将其纳入。
- `tests/verification` 的 19 项单元测试检查指标、重复 ID 拒绝、sample SD、纯论文 allowlist、连接参数、私有下载目录和逐文件符号链接约束；安全测试不发起网络连接。根 `tests` 另覆盖 Release 安装、单卡 torchrun、去除未报告分支、语料清洗、83项物化及82项推理矩阵。通过单元测试不代表模型执行通过。
- **上述数值复核本身没有从头重训或重新推理，也没有新增 ASL 专家可理解性验证。** 本轮新增 checkpoint 执行范围另记，不能用历史重计分报告替代其结果。“历史输出重新计分一致”不能称为完整训练复现。

完整安全聚合结果见 [recomputed_paper_results.json](../results/recomputed_paper_results.json)，PDF 显示值及输入数量约束见 [expected_paper_results.json](../results/expected_paper_results.json)。显示值按论文实际小数位比较，不要求四位小数等于完整双精度数值。

## 核心复得数值

- Locked500 CausalGloss：F1 **0.70467667 ± 0.00554179**，LAAL **1.78229948 ± 0.20308548**；`±` 为三次独立训练的样本 SD。
- CLERC CausalGloss：F1 **0.64897663**；相对 learned k=5 的 F1 差 **0.10012395**，95% CI **[0.08167949, 0.11844157]**。
- Completed-only：**0.61962647 vs. 0.58394580**；差 **0.03568067**，95% CI **[0.02097545, 0.05084107]**；各种子完成子集大小 **349 / 344 / 353**。
- Speech 三种子：F1 **0.39442905 ± 0.00026439**，raw LAAL **2.41485810 ± 0.04178797 s**。S1 相对 lexical 的 raw / computation-aware 延迟差 **−0.15197128 / −0.12292145 s**。RTF 分别为 lexical **0.38766424**、CausalGloss 三种子均值 **0.38014896**；后者不是单独 S1 的 RTF。

## 必须保留的证据边界

1. How2Sign-derived gloss 是弱生成标签，speech 是合成中文语音；复得数值不证明真实 ASL 用户能理解输出，也不证明已能生成视频。
2. Speech over-generation 未解决。三个种子的长度误差均高于 lexical，原预声明长度门槛未通过。END 完成率不是输出长度合格率。
3. **Table 2 的完成率范围需谨慎。** 五个组件消融、full 和 guard-only 为 100%；native no-recovery 为 **249/250 = 99.6%**。若将正文 “all variants completed 100%” 扩展至此 native 对照，会超出原始记录。本仓库保留该事实，不改 PDF，不伪造完成率。
4. LAAL 缺失时必须对双方共同有限的配对样本计分；不能以各自有效均值相减冒充配对差。
5. Fixed k=1/3 各只有一个训练种子；k=5/7 才有三种子。固定基线与 CausalGloss 的种子不全相同，相关比较独立重采样两系统种子。
6. 最终 CausalGloss 三种子使用 `adapter_final`；S1 的 `adapter_best` 实为初始化。最终 learned baseline 来自 2026-09-18 的独立训练版本，不能替换为早期 baseline 或 inference-only 改写。
7. Table 2 的新配对 bootstrap 保留固定 RNG，验证点估计，但不声称其 CI 与历史 CI 逐位相同。Speech 同时给出新的 NumPy CI 与保留 14 个 metric 顺序、共享 Python RNG 流的历史方法 CI；论文 CI 比较采用后者。
8. **语音分块描述存在既有差异。** 选定 SeamlessStreaming launcher 使用 `--source-segment-size 320`，单位为毫秒。500 条原始日志中 1,397 个非末尾唯一提交时刻全部落在 320-ms 网格，其中 1,316 个不在 1.2-s 网格；转换后事件也支持该结论。未找到独立的固定 1.2-s 模型块，不能把读取步长、自适应输出间隔或编码器启动等待混为一谈。旧 1.2-s 设置属于未选中的 Whisper/AlignAtt 分支，而最终前端报告沿用了该分支的默认描述，可能是文字遗留，但其写作过程未经证明。当前仓库保留真实 320-ms 参数，只更正今后新生成报告的默认标签；原 PDF、原日志和原结果不改。保存预测的数值复核通过，不等于此处论文文字也已核实一致。
9. **上游训练暴露。** 全量 ID 核对确认，locked500 的 500/500 样本、132/132 源视频均出现在 Fine-tune1 的 27,079 条训练记录中；下游 action train/validation/test 的 ID 和视频两两交集仍为零。因此本实验的下游 policy 划分互斥，但不能据此宣称整条训练链从未见过测试样本。不改划分、不重建参考来隐藏此事实。
10. **冻结弱目标的真实血缘。** 30,102 个 v4 目标由 pinned base Qwen 生成，无 adapter。上游训练和最终 action/semantic pair 中目标字符串按 request ID 全量与该 v4 一致。Fine-tune1 用于初始化和 prefix scoring，不是这些冻结目标的重新生成器；这与论文相应生成顺序文字存在差异。
11. **参考长度影响生成预算。** CLERC 不把人类 gloss 内容放入生成历史，但其最长参考长度用于共同 action cap；k=5 三种子有 52/55/46 条因 cap 终止，得到 completed-only 的 349/344/353。locked500 和 speech 也使用目标长度确定上限。精确复现必须保留原 cap，不能将其称为“参考只参与评分”；改成 source-only cap 属于另一个实验。

## 修复后的验证层级

`scripts/reproduction/release_assets.py install/verify` 校验私有 Release 分块、外部文件 manifest 和逐文件 SHA/大小；安装位置必须在仓库外。它不修改原始 JSONL，也不证明波形合格或模型数值正确。

`scripts/reproduction/download_policy_model.py` 获取固定 Qwen revision，或以 `--local-only` 检查缓存；`preflight.py` 只读检查依赖、CUDA、模型和输入，不安装、不执行评测。

`scripts/reproduction/materialize_verification_inputs.py` 从已安装附件产生严格 83 项本地输入别名，不重新序列化 JSONL、不伪造当前远端校验清单；随后运行 CPU verifier 才是数值复核。

`scripts/reproduction/run_checkpoint_suite.py` 按 82 项矩阵生成新输出，默认完整选择是 **40 次模型调用，其中六次 learned k-sweep**，产生 82 份预测。它保留历史批大小、精度和预算；不应用 `--separate-k` 冒充精确历史运行。`--dry-run` 不执行，去掉它会启动模型。receipt 只证明退出状态/样本数量，数值仍须重算比较。

四项 speech checkpoint 只重运行 policy，使用已录制共同前端；**不重新合成 TTS、不重新运行 Seamless，也不重新测量其历史时延**。这四项通过也不能称为完整语音链重新生成。

`scripts/reproduction/run_training_plan.py` 默认 dry-run，明确 `--execute` 才执行 30 项训练计划或其子集。`new` 初始化遵循新训练依赖，`archived` 使用历史上游 adapter，后者不是整链重训。没有执行的项目必须明确为未运行。

原始波形 QC、corpus finalizer、保存预测重新计分、选定权重推理、从头训练，是五项不同证据。最终状态由 `results/reproduction_status.json` 中逐项实际记录决定；缺少、未执行、近似匹配和精确匹配不得合并写为“全部复现”。

## 自己运行

以下是冻结结果的 CPU 路线，需要 Python ≥3.10 和 NumPy，无须加载 7B 模型。先合法取得 Release 数据；所有输入/输出目录在仓库外，两个输出目录此前均不能存在。在仓库根目录执行：

```bash
python -m unittest discover -s tests/verification -v

python scripts/reproduction/materialize_verification_inputs.py \
  --data-root /absolute/path/to/installed_paper_assets \
  --output-dir /absolute/path/to/private_artifacts

python scripts/verification/verify_paper_results.py \
  --input-root /absolute/path/to/private_artifacts \
  --output-dir /absolute/path/to/a_new_audit_run
```

`--output-dir` 必须是此前不存在的新目录，以免覆盖审计。标准输出只给状态；完整数值、输入数量、校验和和任何差异写入 `verification_report.json`。默认进行 20,000 次相关重采样；其他次数可用于调试，但不能据此声称已重新验证论文 CI。

推荐通过 [REPRODUCTION.md](REPRODUCTION.md) 的私有 Release 安装器取得数据。拥有原服务器访问权限时，也可查看同一 83 项 allowlist，再明确下载：

```bash
python scripts/verification/fetch_verification_inputs.py \
  --destination /absolute/path/to/private_artifacts \
  --ssh-key /absolute/path/to/your_existing_private_key \
  --host USER@HOST --port PORT --list-only

# 检查清单后，去掉 --list-only 执行实际下载和 SHA-256 核对。
```

脚本只调用 `scp` 和只读 `sha256sum`；不训练、不下载模型、不改远端文件、不读取或显示密钥内容。它要求下载目标位于仓库外，并保持 SSH 主机密钥检查开启。主机须已由你可信地加入 `known_hosts`。清单内原服务器路径用于追溯，不是公共数据下载地址；无访问权者需合法取得数据并构建同 schema 输入。

## 私有输入 schema

| 输入组 | 必须的标识、参考与动作信息 |
|---|---|
| Locked500 / Table 2 | 唯一 `id`；`source_tokens`、`target_pseudo_asl_gloss_tokens`、`predicted_pseudo_asl_gloss_tokens`；`write_source_positions`、`completed_after_full_source` 及历史逐行指标作交叉检查。 |
| CLERC 预测 | 唯一 `id`；`predicted_gloss_tokens`、`write_source_positions`、`completed_after_full_source` 和历史 LAAL。 |
| CLERC 参考 manifest | 同一 401 个 `id`；`source_tokens`、嵌套 `references[*].content_tokens`、`median_content_reference_length`。不得把嵌套参考当独立 prompt。 |
| Speech | 唯一 `id`；`reference_pseudo_asl_gloss`、`speech_to_gloss_predicted_tokens`、`audio_duration_seconds`；音频读取时间 / compute-adjusted WRITE 时间、END/full-source 完成字段、RTF、原历史 bootstrap 使用的其他逐行指标。 |

学习代码可依次看 `overlap` / `laal` / `normalize_content`，再读 `locked500()`、`epee()`，最后比较两种 seed/parent bootstrap。
