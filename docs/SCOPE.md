# Paper-only scope

当前分支是最终论文代码，不是实验历史备份。纳入标准仅有两个：

1. 最终论文明确报告的方法、配置、基线、消融、数据语料、评测或图表；
2. 重建这些内容所必需且有实际调用/数据生产证据的依赖。

“在同一服务器目录”“某个旧 launcher 顺带调用”“曾经跑过”都不足以纳入。

## 此次纠正

上一版是过宽的溯源归档，包含 690 份原件。此次选取 **126 份论文相关原始源码/配置/协议**，重新分到浅层模块；另有简短运行/验证工具、文档和纯数值源数据。126 不是 Python 文件数。

移出了早期 5K 模型试验、NaN 失败记录、旧 v2.1 模型实验、9/12 废弃 learned baseline、local-agreement、未选中的语音前端、未在论文报告的 CLERC lexical 对照、大量训练历史/日志/模型卡和本次新增 checkpoint rerun。本文语音 lexical 内部对照、no-recovery/guard 控制和最终 learned baseline 的 LR screen 则保留。

## 适配而非暗改

17 份来源文件有明确记录的适配：路径、CLI 输入、测试路径、辅助哈希清单、从混合脚本删除未报告输出、移除旧 pilot 默认输入/过期报告标签，或把 Fig.1 改成同内容可移植重绘。原始 SHA 和当前 SHA 同时保留，不把改过的文件冒称逐字节原件。

五项 ablation 变换、动作构造、WAIT balancing、recovery 状态和 bootstrap 核心函数，与原函数的 AST 一致；见 `provenance/algorithm_function_manifest.json`。主训练器、模型输入和自由推理算法不因目录整理而改变。

旧 `run_after_scoring.sh` 既执行必要预处理，又启动旧 v2.1 训练，因此不保留整条旧编排；保留实际被最终流程复用的函数。消融 builder 不再生成废弃 learned-baseline；recovery builder 不再额外导出早期 screen 子集，完整训练样本构造和冻结拆分算法不变。

旧多种子聚合脚本混入了未报告的 `waitk_v1_k1` 比较及旧 claim gate，故不保留。最终主方法、基线、消融和语音数值统一由 `scripts/verification/verify_paper_results.py` 从对应原始预测复核。

## 数据和历史

239 项资产索引只定位最终论文及必要生产输入，不收集所有训练日志/旧试验数据。实际大型数据和模型仍分开保存，GitHub 不是完整离线数据备份。

服务器原始文件、旧本地审计和已提交 PDF 均不动。此前 Git 提交可恢复；此次只清理当前代码树，不强制推送、不改写历史。需要查看历史时用旧提交，不把它们放回当前学习入口。
