## 简历条目（中文）

- **仿真具身智能工程**：基于 LeRobot + LIBERO 构建仿真具身智能模仿学习项目，复现 Diffusion Policy 基线并实现可一键训练/评估的工程化流水线，支持本地 12GB GPU 小样本调试与云端多卡规模训练。
- **策略创新**：提出 PhaseQFlow 生成式动作块策略，引入阶段感知（phase‑aware）条件与质量加权（quality‑weighted）训练机制，结合 flow‑matching 目标提升长序列操作任务的训练稳定性与成功率。
- **扩展研究**：根据官方指南实现 SmolVLA PEFT/LoRA 微调，并设计 phase prompt / adapter 原型，为从模仿学习扩展到 VLA 与在线 RL 提供清晰接口。

## CV bullets (English)

- **Simulation‑first embodied intelligence engineering**: Built an imitation learning project on LIBERO using LeRobot’s training/evaluation pipeline, reproducing the Diffusion Policy baseline and providing a fully engineered, reproducible repo.  Supports local 12 GB GPU debugging and cloud‑scale multi‑GPU training.
- **Policy innovation**: Proposed PhaseQFlow, a generative action‑chunk policy with phase‑aware conditioning and quality‑weighted imitation loss, leveraging a flow‑matching objective to improve stability and success on long‑horizon manipulation tasks.
- **Research extensions**: Implemented SmolVLA PEFT/LoRA fine‑tuning per official guidance and designed a phase prompt / adapter prototype, offering a clear path from imitation learning toward vision‑language action and online RL.
