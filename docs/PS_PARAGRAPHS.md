## PS 段落（中文）

我希望在博士阶段研究“生成式策略学习”在具身智能中的可解释结构化归纳偏置与高效推理。为此，我基于 LeRobot 与 LIBERO 仿真基准搭建了一个从基线复现到策略创新的完整项目：在复现 Diffusion Policy 模仿学习基线后，我提出 PhaseQFlow，将轨迹内任务进度离散为阶段（phase）并以可学习嵌入注入生成条件，同时根据动作序列平滑性构造质量权重（quality weight）来稳定小样本训练。该设计不仅能对齐 2025–2026 年生成式动作块策略的效率趋势（flow matching/流式推理），也为我后续把模仿学习扩展到更一般的 VLA 与在线 RL 提供了清晰接口。我相信这种“从复现到可交付创新”的研究型工程训练，使我具备在博士阶段持续推进可复现机器人学习研究的能力。

## PS paragraph (English)

I am interested in generative policy learning for embodied intelligence, particularly how structured inductive biases and efficient inference can improve long‑horizon manipulation.  To explore this, I built a complete simulation‑first pipeline on LIBERO using LeRobot: after reproducing the Diffusion Policy baseline, I proposed PhaseQFlow, which discretizes trajectory progress into phases and injects a learnable embedding while deriving quality weights from action smoothness to stabilize low‑data imitation.  This design aligns with the 2025–2026 trend of flow‑matching generative policies and streaming inference, and it provides a clear bridge from imitation learning to vision‑language action and online reinforcement learning.  This project reflects my ability to transform reproducible baselines into deliverable research improvements, which I aim to continue during my PhD.
