---
title: "第7章：Prefill/Decode 解耦架构"
description: "理解 P/D 混合 Batching 的问题、DistServe/Splitwise 等解耦方案、KV Cache 传输、Goodput 与 SLO 感知调度"
pubDate: 2026-04-16
updatedDate: 2026-10-21
category: "inference-optimization"
order: 36
tags: ["P/D解耦", "DistServe", "Splitwise", "KV Cache传输", "Goodput", "SLO"]
---

## 📑 目录

- [本章简介](#本章简介)
- [本章小节](#本章小节)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

## 本章简介

Prefill 与低并发 Decode 常分别偏计算和访存密集，具体瓶颈取决于模型与工作负载；混批时还可能出现资源互扰。本章深入解耦架构的设计与挑战，并落到 vLLM 的 Disaggregated Prefill 能力。

**混合 Batching 的问题**通过时间线、教学模型和分组实验分析 Prefill 对 Decode 的干扰，区分请求平均 TPOT 与逐 Token 的 ITL；具体倍数由模型、硬件、负载和统计口径决定。

**解耦架构设计**覆盖 DistServe（OSDI'24，系统化论证 P/D 解耦）、Splitwise（ISCA'24，分配到不同 GPU 池）、TaiChi（聚合与解耦统一框架）和 MLC Microserving（跨引擎编排）。

**KV Cache 传输与 Connector**深入解耦架构最核心的工程挑战：Prefill 节点算出的 KV Cache 如何高效搬到 Decode 节点。介绍 vLLM 的 KV Connector 抽象、NIXL/NCCL 等传输后端，以及跨节点带宽对整体延迟的影响。

**Goodput 与 SLO 感知调度**强调 Raw QPS 不等于用户体验，Goodput（满足 SLO 的有效吞吐）才是真正的优化目标。

**解耦架构的挑战**包括调度器复杂度、P/D GPU 池的资源配比推导，以及何时该用/不该用解耦。

**vLLM Disaggregated Prefill 实战**：用 vLLM 的解耦部署构造混合负载，量化 P/D 互扰程度并推导给定工作负载下的 P/D GPU 池配比。

## 本章小节

- [**7.1 混合 Batching 与 P/D 互扰：为什么出字会突然停一下**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/71-混合batching与pd互扰/)：从迭代时间线区分 Prefill/Decode 的资源竞争、排队、TPOT 与 ITL，用可运行模型理解 Chunked Prefill 与 P/D 解耦的取舍
- [**7.2 解耦架构设计：从 DistServe、Splitwise 到混合编排**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计/)：沿请求和 KV 的生命周期比较 DistServe、Splitwise、TaiChi 与 MLC Microserving，理解解耦的资源、路由和故障边界
- [**7.3 KV Cache 传输与 Connector：搬走的不只是一个张量**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/73-kvcache传输与connector/)：核算 KV 载荷与带宽下界，理解 vLLM V1 Connector 的 Scheduler/Worker 分工、NIXL/NCCL 和缓存移交的正确性与失败处理
- [**7.4 Goodput 与 SLO 感知调度：忙得很，为什么合格请求反而少**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/74-goodput与slo感知调度/)：定义请求级 TTFT/TPOT 与联合达标 Goodput，区分分位数和逐请求判断，讨论准入、期限与长度预测及失败统计
- [**7.5 资源配比与调度挑战：P 池和 D 池到底各放几张卡**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/75-资源配比与调度挑战/)：用服务需求和排队余量推导 P/D 初始配比，区分实例数与 GPU 数，加入网络、KV 驻留、背压和动态流量约束
- [**7.6 vLLM P/D 解耦实战：从 NIXL 跑通到同预算对照**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/76-vllm-pd解耦实战/)：按 vLLM v0.19.0 配置双 GPU NixlConnector 与官方代理，生成混合负载并比较聚合、双副本和解耦的互扰、Goodput 与质量

## 📝 总结

- 解耦让阶段资源与调度可分别配置，也引入 KV 迁移和状态治理成本。
- Goodput 需要明确同一请求、统一窗口和联合 SLO 的统计规则。
- P/D 配比要一起满足计算、网络、缓存驻留与尾延迟约束。

## 🎯 自我检验清单

- 能画出请求与 KV 的生命周期及故障处理
- 能区分各系统论文、编排接口和 Connector 的层次
- 能估算迁移成本并推导工作负载相关的资源配比
- 能设计同 GPU 预算的聚合与解耦对照实验

## 📚 参考资料

- [DistServe — OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [vLLM v0.19.0 — Disaggregated Prefilling](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/features/disagg_prefill.md)
- [vLLM v0.19.0 — NixlConnector](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/features/nixl_connector_usage.md)
