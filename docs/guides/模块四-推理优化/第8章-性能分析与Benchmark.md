---
title: "第8章：性能分析与 Benchmark"
description: "建立完整的推理性能指标体系，掌握 vllm bench、GenAI-Perf 等压测工具和性能分析工具，制定性能回归门禁"
pubDate: 2026-04-16
updatedDate: 2026-10-24
category: "inference-optimization"
order: 37
tags: ["性能分析", "Benchmark", "vLLM Bench", "GenAI-Perf", "性能回归"]
---

## 📑 目录

- [本章简介](#本章简介)
- [新手学习路线](#新手学习路线)
  - [从用户的一句“慢”开始](#从用户的一句慢开始)
  - [先完成一个可解释的小实验](#先完成一个可解释的小实验)
  - [先认识六个经常出现的词](#先认识六个经常出现的词)
- [本章小节](#本章小节)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

---

## 本章简介

优化效果需要量化验证，本章建立推理场景的完整性能评估体系，压测工具以 vLLM 自带的基准工具为主。

**推理指标体系**定义完整指标集：QPS、TTFT(P50/P95)、TPOT(P50/P95)、Throughput、显存峰值、GPU 利用率，分析指标间的关联与 trade-off，强调单一指标无法反映全貌。

**压测工具**覆盖 vLLM 自带的 `vllm bench serve` / `vllm bench latency` / `vllm bench throughput`、GenAI-Perf（LLM 指标一站式输出）、Triton Perf Analyzer，以及自定义压测脚本的设计要点和可复现的 Benchmark 配置管理。

**性能分析工具**在推理场景下的最佳实践：torch.profiler 算子级分析、Nsight Systems 全链路分析、Nsight Compute Kernel 级下钻。

**权威基准**介绍 MLPerf Inference（Datacenter）和 LLM Perf 评测趋势。

**性能回归门禁**制定规则（如 TPOT P95 退化 > 5% 则 Block Merge）、CI 自动化集成和退化定位方法（git bisect + Nsight 对比）。

**动手实验**贯穿各节：先手算请求指标，再运行固定负载；用 CPU 算子实验认识 Profiler，用人工退化摘要验证门禁，具备 GPU 环境后再采集真实 vLLM 与 Nsight 时间线。教学数字和 GPU 实测分开记录。

## 新手学习路线

### 从用户的一句“慢”开始

先分清首字慢、后续暂停、总任务慢和接不住流量，再选择指标与工具。Token、Prefill/Decode、KV Cache、TP/PP/DP 等基础概念见前面章节；本章从这些机制回到“怎样证明优化有效”。

| 小节 | 学完能回答 | 第一次可以做什么 |
|------|------------|------------------|
| 8.1 指标 | 首字、出字、吞吐和 Goodput 分别是什么 | 看自制时间线，手算四条请求 |
| 8.2 压测 | 离线成绩与在线体验有何不同 | 写配置卡，分终端运行小规模对照 |
| 8.3 定位 | 时间花在计算、通信还是等待 | CPU 上导出算子 Trace，再学 GPU 工具 |
| 8.4 基准 | 公开结果在什么条件下可比较 | 给两条成绩填完整结果卡 |
| 8.5 门禁 | 一次变慢怎样被发现与定位 | 运行 PASS/FAIL/REVIEW 教学比较器 |

### 先完成一个可解释的小实验

本章 vLLM 命令以 v0.19.0 为接口基线，使用 Linux Bash。先固定模型与 Tokenizer 快照、精度、长度、并发/到达率、缓存和质量，再改一个因素；需要 GPU、Profiler 或额外工具的步骤都有相应前置条件。

没有 GPU 时也能学习指标、结果卡、门禁和 CPU 算子分析。真实设备成绩、Prometheus 查询与集群验证要在目标环境完成，不能把代码语法通过写成性能结论。

### 先认识六个经常出现的词

| 词语 | 通俗理解 | 学习时先留意 |
|------|----------|--------------|
| Benchmark / 压测 | 用确定的题目和发流规则测成绩 | 先说明测的是本地引擎还是完整在线路径 |
| Baseline / 基线 | 拿来比较的参照配置或版本 | 参照本身也需要质量合格、环境可复查 |
| Workload / 工作负载 | 这次到底让系统做什么工作 | 不仅有请求数，还有输入输出长度、到达和缓存 |
| Percentile / 分位数 | 把样本排好队后看某个位置 | P95、平均值和最慢值不能互相替代 |
| Profile / Trace | 测量并记录时间花在哪里 / 导出的活动记录 | 采样范围与开销要明确，文件生成不等于已找到原因 |
| Regression / 回归 | 改动后某项能力比可信参照变差 | 先排除工作量或环境变化，再判断是否来自代码 |

可以先完成这条最短闭环，再阅读工具细节：

```mermaid
flowchart LR
    A["说清哪种慢"] --> B["固定负载与基线"]
    B --> C["测结果，再定位"]
    C --> D["关闭采样后验收"]
```

第一次自学，做到“手算一组指标 → 看懂一张时间线 → 写出一条有证据的结论”就有收获。每节新图下都有读图说明或小练习；教学数值帮助理解方法，真实性能表由自己的实验填写。

## 本章小节

- [**8.1 推理指标体系：从用户等多久，到一张卡能服务多少人**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-性能分析与benchmark/81-推理指标体系/)：用请求时间线和可手算记录解释 TTFT、TPOT、ITL、吞吐与 Goodput，区分客户端体验、服务端指标、显存与 GPU 利用率
- [**8.2 压测工具与可复现实验：先测同一道题，再比较谁更快**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-性能分析与benchmark/82-压测工具与可复现实验/)：区分离线延迟、离线吞吐与在线服务压测，逐步运行 vLLM 对照，理解并发、到达率、预热、GenAI-Perf 和实验配置记录
- [**8.3 性能分析工具：先看整条时间线，再钻进一个算子**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-性能分析与benchmark/83-性能分析工具与定位流程/)：以可运行的 torch.profiler 小实验认识 CPU/GPU 时间，再用 Nsight Systems、Nsight Compute 和 vLLM 短窗口采样逐层定位瓶颈
- [**8.4 权威基准与结果解读：排行榜上的高分，怎样变成自己的判断**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-性能分析与benchmark/84-权威基准与结果解读/)：认识 MLPerf Inference 的场景、质量与提交约束，区分 LLMPerf 工具与正式基准，用结果卡避免跨模型、预算和指标口径误比
- [**8.5 性能回归门禁：让一次变慢有证据，也有处理流程**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第8章-性能分析与benchmark/85-性能回归门禁与退化定位/)：用可运行的摘要比较器理解性能门禁，固定实验身份、质量与样本要求，区分通过、退化和证据不足，再连接 CI 与 git bisect 定位

## 📝 总结

- 从计时边界、样本和窗口开始，统一指标口径。
- 压测测结果，Profiler 查原因，正式基准提供共同规则。
- 工作量、质量与资源不一致时，不能只按一个吞吐数字排名。
- 性能门禁依赖受控环境、原始证据与可重复的好坏判定。

## 🎯 自我检验清单

- 能画出请求从发出、首输出到完成的计时路径
- 能区分在线/离线工具与客户端/服务端计时
- 能从队列和时间线解释一次慢，而不只说 GPU 忙
- 能核对公开成绩的任务、质量、场景、资源和单位
- 能设计带质量、失败、波动和证据不足处理的性能门禁

## 📚 参考资料

- [vLLM v0.19.0 — Benchmark Serve](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/benchmarks/serve.py)
- [vLLM v0.19.0 — Profiling](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/contributing/profiling.md)
- [MLCommons — Inference Policies](https://github.com/mlcommons/inference_policies)
