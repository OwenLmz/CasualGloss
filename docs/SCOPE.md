# Paper-only scope

当前分支是最终论文代码，不是实验历史备份。纳入标准仅有两个：

1. 最终论文明确报告的方法、配置、基线、消融、数据语料、评测或图表；
2. 重建这些内容所必需且有实际调用/数据生产证据的依赖。

“在同一服务器目录”“某个旧 launcher 顺带调用”“曾经跑过”都不足以纳入。

## 包含范围

论文相关源码、配置和协议按数据准备、主方法、基线、消融、语音链和评测划分；逐文件对应关系见 `provenance/source_manifest.json` 和 [论文到代码映射](PAPER_TO_CODE.md)。本文语音 lexical 内部对照、no-recovery/guard 控制和最终 learned baseline 的 LR screen 均属于论文所需内容。

未在最终论文中报告的候选模型、失败实验、其他语音前端、扩展研究及私人录音不属于本仓库范围。

## 源码组织与适配

可移植适配包括路径、CLI 输入、测试路径、辅助哈希清单、从混合脚本分离最终流程、提取正式 locked500 参数，以及 Fig.1 的同内容重绘。原始 SHA 和仓库 SHA 同时保留在来源清单中。

五项 ablation 变换、动作构造、WAIT balancing、recovery 状态和 bootstrap 核心函数的来源映射见 `provenance/algorithm_function_manifest.json`。

完整训练样本构造、冻结拆分算法以及最终方法依赖的预处理函数保留；未报告的训练编排和输出不纳入。评测指标与统计协议见 [EVALUATION.md](EVALUATION.md)。

## 数据与资产

资产索引只定位最终论文及必要生产输入，不收集所有训练日志或候选实验数据。大型冻结输入、所选 LoRA、逐样本预测及必要音频以同一私有仓库的 Release 附件存放，下载位置见 `provenance/release_assets.json`；Git 源码树提供校验目录和安全安装器。官方预训练模型单独按版本获取。

从 19,947 原始语料中排除的 23 条 QC 输入保留，用于重建正式 19,924 清洗版。音频质检实现见 `speech/frontend/validate_audio_manifest.py`，相应输入报告和调用说明随数据提供。

最终 timing 规则的开发与选择使用 482 个 eligible pilot parents，对应两个 pilot prefix-score 分片。它们用于规则选择，不是最终训练或评测输入，也不能代替 full prefix-score 分片。

论文数值汇总见 `results/paper_results.json`。
