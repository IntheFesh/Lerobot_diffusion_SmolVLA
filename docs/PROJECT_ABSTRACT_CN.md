# 项目摘要：PhaseQFlow —— 面向 LIBERO 的阶段感知与质量加权生成式模仿学习

本项目基于 LeRobot 与 LIBERO 仿真基准，构建一条面向博士申请的“可复现、可扩展、可工程化交付”的具身智能研究主线。项目首先复现 Diffusion Policy 模仿学习基线，并进一步提出 **PhaseQFlow**：一种 phase‑aware + quality‑weighted 的生成式动作块策略。该策略将轨迹内进度阶段（phase）与示例质量权重（quality weight）两类结构化先验注入生成模型条件与训练目标，从而在小样本调试与长时序操作任务中提升训练稳定性与策略有效性。

PhaseQFlow 的核心做法包括：

1. **阶段感知（phase‑aware）**：基于 `(frame_index / episode_length)` 计算阶段标号，并通过可学习的 phase embedding 注入到动作生成过程，使策略对任务不同阶段的决策模式更加敏感。
2. **质量加权（quality‑weighted）**：基于动作序列的平滑度（jerk）计算样本质量权重，对更平滑、更一致的专家片段赋予更高损失权重，以提升小数据情况下的收敛质量。
3. **流匹配目标（flow matching）**：采用 flow matching/rectified flow 目标取代传统扩散去噪采样，在推理阶段以更少采样步数生成动作块，适配 2025–2026 年生成式机器人策略的效率趋势，并为接入 Real‑Time Chunking 等流式推理栈预留接口。

工程实现方面，项目严格对齐 LeRobot policy 插件与 processor pipeline 规范，提供可一键运行的训练/评估脚本（本地 Smol‑LIBERO 小样本调试 + 云端 LIBERO 全量训练），并以成功率、平均奖励与推理延迟三类指标进行评估。除主线外，项目给出 SmolVLA 的 PEFT/LoRA 微调与 phase prompt/adapter 原型扩展，为从 diffusion 主线走向 VLA/RL 扩展提供清晰的后续研究接口。
