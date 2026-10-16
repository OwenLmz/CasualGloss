# 结果复核与复现边界 / Verification

本仓库对应提交版论文，不是当前 gloss→pose/video 扩展。论文锚点 SHA-256：

`5f7793358d54f9d754aa2fa3c8f1d5fbb723ed4506105c21ca886771e9856e1a`

本次没有改动该 PDF、服务器原始代码、已有模型或历史输出。

## 本次实际完成了什么

不是把历史 `metrics.json` 换个位置，也不是重新训练一次后挑接近论文的结果。我们下载了明确列出的历史逐样本输出，独立实现轻量指标核算，并逐项对照提交 PDF。

| 层次 | 实际状态 |
|---|---|
| 原始文件身份 | 91 个指定远程文件逐一比对本地/远端 SHA-256，相同 |
| 逐样本独立计分 | 83 份原始 rollout，共 35,936 条记录；另读 CLERC 参考 manifest 和历史逐样本分数，总计 85 个直接输入文件 |
| 表1 / 图1 | 35 份 locked500 输出：CausalGloss 三种子、固定 Wait-k 八份、最终独立训练 learned baseline 的 3×8 份；检查 ID、source/reference 一致性，重算 Token F1、序列相似度、LAAL、无平滑 corpus BLEU-4、种子均值与样本 SD |
| CLERC | 从预测 token 和人类参考重新归一化、取逐 prompt 最佳参考；401 prompts、1,200 条嵌套参考；主方法、全部 learned k=1…8、固定基线及 lexical control；其中 10,827 个主方法/learned 逐 prompt F1 与历史分数逐一吻合 |
| Completed-only | 在每个 learned k=5 种子完成的相同 prompts 上配对主方法；重新选择子集并重算 20,000 次 bootstrap |
| 表2 | 250-parent diagnostic 集上的完整方法、五个组件消融、native no-recovery 和 guard-only 控制；重算配对差值，LAAL 只用双方有限的配对样本 |
| Speech | lexical 与三个 CausalGloss 种子的各 500 条原始输出；重算 F1、按音频读取时间和序列执行完成时间分别计算 LAAL；核对完成率、RTF、相对长度误差 |
| Bootstrap | locked learned 比较 20,000 次匹配种子/parent 重采样；固定比较独立重采样两系统种子；CLERC 20,000 次；speech 按原顺序共享 Python RNG 的 10,000 次原方法也已重算 |
| 合成单元测试 | 六项：多重集重叠、LAAL 截止点、CLERC 内容归一化、无平滑 BLEU、重复 ID 拒绝、sample SD |
| 模型重新推理 / 从头训练 | **本数值工具没有进行**。另行用选定 S1 checkpoint 完成了 8 条 smoke 和完整 locked500 推理；见下节。没有从头重训，不能称为完整训练复现 |

完整安全聚合报告见 [recomputed_paper_results.json](../results/recomputed_paper_results.json)。期望值及其显示精度见 [expected_paper_results.json](../results/expected_paper_results.json)。报告只包含汇总数值、协议、文件相对路径和校验和，不发布原始正文、参考或预测文本。

## 另行实际执行的 checkpoint 检查

在当前 RTX 4090 D 上，使用归档的原始 evaluator、原选中 S1 `adapter_final`、同一 pinned backbone、BF16 和原 generation batch size 32，重新运行完整 500 条。新 F1 **0.70405125**，历史 S1 **0.70158607**；新 LAAL **1.65958975**，历史 **1.66535143**；均 100% 完成。465/500 gloss 序列、459/500 WRITE 位置、453/500 联合轨迹完全一致。

这是**结果接近但不逐条一致**的重新推理，不能写成“完整复现通过”。硬件不同，但没有单独实验证明所有差异都由硬件造成。新旧输出都保留，不用新输出替换历史论文证据；没有重新选择模型。安全聚合见 [checkpoint_rerun_s1_500.json](../results/checkpoint_rerun_s1_500.json)，原始新输出保存在独立私有审计目录。8 条固定前缀 smoke 的不一致也保留在 [checkpoint_smoke_s1_8.json](../results/checkpoint_smoke_s1_8.json)。

最初误用语音环境时缺少 NLTK，模型推理没有启动；随后使用既有 policy 环境成功运行。没有安装依赖或修改原始环境。

## 关键数值复得

- CausalGloss locked500：F1 **0.70467667 ± 0.00554179**，LAAL **1.78229948 ± 0.20308548**；具体末位以 JSON 为准。`±` 是三次独立训练结果的样本 SD，不是置信区间。
- CLERC CausalGloss：F1 **0.64897663**；对 learned k=5 差 **0.10012395**，95% CI **[0.08167949, 0.11844157]**。
- Completed-only：**0.61962647 vs. 0.58394580**；差 **0.03568067**，95% CI **[0.02097545, 0.05084107]**；三种子的完成子集大小分别为 **349 / 344 / 353**。这是事后稳健性检查，不替代全部 401 prompts 的主分析。
- Speech 三种子：F1 **0.39442905 ± 0.00026439**，raw LAAL **2.41485810 ± 0.04178797 s**。S1 对 lexical 的 raw/compute-aware 延迟差分别 **−0.15197128 / −0.12292145 s**。

数字与 PDF 对照采用其实际显示位数的舍入容差，不要求四位小数等于原始双精度数值。完整逐样本核对则使用更严格的 `1e-7`（八位小数记录）/ `1e-12`（未舍入 CLERC F1）容差。

## 不能隐藏的边界

1. **“复得论文数值”不等于模型在真实 ASL 视频上有效。** How2Sign-derived 标签为弱 pseudo-gloss；speech 是合成中文语音；没有新专家 ASL 可理解性评估。
2. **Speech over-generation 没有解决。** 三个种子的长度误差均高于 lexical control，原始预声明长度门槛未通过。运行完成和 END 完成率不是同一个概念。
3. **表2正文有一个范围需要谨慎理解。** 五个组件消融、full、guard-only 均为 100% 完成；native no-recovery 对照是 **249/250 = 99.6%**。若把正文 “all variants completed 100%” 延伸到这个 native 对照，则超出了原始记录。仓库保留这个事实，不改动提交 PDF，也不伪造 100%。
4. 部分 LAAL 为空，配对比较必须同时剔除双方不有限的记录；不能拿两组各自均值相减冒充配对差。每个报告保留有效数量。
5. 固定 k=1/3 只有单一训练种子；不能称它们为“三种子基线”。固定 k=5/7 与 CausalGloss 的训练种子不全相同，应独立重采样系统种子。
6. CausalGloss 的提交模型是三个 `adapter_final`。S1 名称为 `adapter_best` 的文件实际为初始化，不是提交模型。最终 learned baseline 必须用 2026-09-18 的独立训练版本，不能换成 9/12 旧 baseline 或 inference-only 改写。
7. 表2中新增的配对 bootstrap 用记录在报告里的固定 RNG；其点估计对照论文，但**不声称表2历史 CI 逐位完全相同**。speech 同时保留新的 NumPy CI 与原 Python RNG CI，二者因随机数流不同而有轻微蒙特卡洛差别，原 CI 已精确复得。

## 如何自己运行

在仓库根目录执行。CPU 即可；需要 Python ≥3.10 和 NumPy。不要把下述私有输入文件夹放进 Git。

```bash
python -m unittest discover -s tests/verification -v

python scripts/verification/verify_paper_results.py \
  --input-root /absolute/path/to/private_artifacts \
  --output-dir /absolute/path/to/a_new_audit_run
```

`--output-dir` 必须是一个不存在的新目录，防止覆盖以前的审计。默认重采样次数 20,000；表2、固定 locked500 和 speech 原方法维持 10,000。标准输出只给简短状态，完整结果写入 `verification_report.json`。

已有服务器访问权限时，可显式下载 allowlist 并核对远端哈希：

```bash
python scripts/verification/fetch_verification_inputs.py \
  --destination /absolute/path/to/private_artifacts \
  --ssh-key /absolute/path/to/your_existing_private_key \
  --host USER@HOST --port PORT
```

先用 `--list-only` 看明确的 91 个目标文件。脚本仅调用 `scp` 与只读 `sha256sum`，不执行训练、不下载基座/adapter、不改服务器文件、不读取或显示密钥内容。SSH 主机必须已由你可信地加入 `known_hosts`，脚本不关闭主机密钥检查。原始服务器路径是审计映射，不是公开数据下载地址；没有访问权限的人应按数据访问文档合法获取原资产，再通过同样 schema 复算。

阅读代码建议：先看 `overlap` / `laal` / `normalize_content` 三个小函数，再看 `locked500()` 的表1聚合，再读 `epee()` 的完成子集分析；最后学习两种 seed/parent bootstrap 的差别。这个顺序不需要先加载 7B 模型。
