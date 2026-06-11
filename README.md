# 多 Agent 异构资源博弈管理系统 (Multi-Agent Resource Orchestrator)

基于 Ray 分布式框架构建的多智能体资源调度模拟系统。本项目模拟了在多台异构老旧设备（具备不同的 CPU、GPU、NPU 算力）上，通过“电费 Token”经济学机制进行大语言模型（LLM）推理任务分配与物理功耗平衡的过程。

## 🌟 核心特性

- **异构硬件模拟**：细粒度模拟大模型推理流水线中 `CPU`（数据分词）、`GPU`（重载预填充）和 `NPU`（低功耗边缘推理）的功耗与发热特性。
- **Token 经济学**：引入“饥饿感”与“电费 Token”机制。设备必须通过低功耗运行赚取 Token，才能在后续回合抢占高收益的高性能核心任务。
- **热力学与过热降频 (Thermal Throttling)**：高度拟真的物理约束。若节点长期满载导致温度失控（> 85°C）或 GPU 撞击温度墙（> 95%），将触发严格的 Token 罚款，强制节点进入休眠降温状态。
- **分布式 Actor 模型**：采用 `Ray` 框架，基于 Master-Worker 架构实现完全异步的进程隔离与 RPC 状态同步。

## 🛠️ 系统架构

系统包含两类核心 Agent：
1. **Master Agent (中央调度中枢)**：负责全局状态监控、回合结算、异常惩罚下发以及任务的宏观派发。
2. **Worker Agent (边缘节点)**：独立的 Ray Actor 进程，维护各自的物理状态（Tokens、CPU/GPU/NPU 占用率、温度），并根据资源状况向 Master 申请任务。

## 🚀 快速开始

### 1. 环境依赖
确保你的机器上安装了 Python 3.8 或更高版本。
安装必需的依赖库（主要是 Ray）：
```bash
pip install -r requirements.txt

### 2. 开始运行
```bash
python main.py