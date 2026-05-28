<h1 align="center">💫 Awesome LLM Hint-based RLVR</h1>


<p align="center">
  <b>A curated collection of papers and resources on Hint-Based RLVR for Large Language Models.</b>
</p>

## 😳 What are Hints?

**Hints** are **explicit textual signals** that carry task-relevant information beyond what ordinary rollouts under the current policy can reliably produce. They are introduced into the RL training process to reshape rollout-group construction, making otherwise uninformative samples more likely to yield useful policy updates. 

## 🧐 Why Hints?

Group-relative advantage methods like GRPO rely on **reward variation** within each rollout group to produce a gradient signal. When all rollouts in a group receive identical rewards, whether all correct on easy problems or all wrong on hard ones, the advantage collapses to zero and the policy-gradient signal vanishes. This **zero-advantage failure** wastes both computation and valuable training data, especially the hardest questions that matter most for learning. 

While searching more aggressively within the policy's own distribution (e.g., resampling or tree search) can help, it only sharpens the existing distribution without **expanding its capability boundary**. 

Hints address this bottleneck by injecting **external textual signals** that restore learnable contrasts within rollout groups, enabling the policy to learn from otherwise wasted samples while pushing beyond its current capability frontier.

<p align="center">
  📖 <b>Survey Paper:</b> <a href="#">A Survey on Hint-Based RLVR: Overcoming Zero-Advantage Failures with External Textual Signals</a>
</p>

<p align="center">
  🌕️ = Covered in our <a href="#">survey paper</a> &nbsp;|&nbsp; 🌒 = Indexed here, pending inclusion in the next survey revision
</p>

## 🛸 Full Tree
```text
Hint-based Reinforcement Learning (Current Survey Structure)
│
├── §3 Sample-level Hints
│   │
│   ├── §3.1 Trajectory-based Hints
│   │   ├── §3.1.1 Trajectory Injection
│   │   │   ├── Reference Prefix
│   │   │   │         (QuestA, POPE, CCL, SEELE, GHPO, BREAD)
│   │   │   ├── Self-Replay Prefix
│   │   │   │         (HiPO, PROS, RPO, Failure-Prefix)
│   │   │   └── Hybrid Prefix
│   │   │             (CORE)
│   │   ├── §3.1.2 Trajectory Continuation
│   │   │   ├── Scheduled Prefix Decay
│   │   │   │         (UFT, Prefix-RFT, EvoCoT)
│   │   │   ├── Adaptive Prefix Control
│   │   │   │         (G²RPO-A, ADHint, Hint-GRPO, BHA, TRAPO)
│   │   │   └── Intra-group Prefix Mixing
│   │   │             (StepHint)
│   │   ├── §3.1.3 Trajectory Replacement
│   │   │   ├── Unconditional Augmentation
│   │   │   │         (ANCHOR, LUFFY)
│   │   │   ├── Triggered Substitution
│   │   │   │         (AMPO, S-GRPO, HAPO, HiPO)
│   │   │   └── Reconstructive Repair
│   │   │             (SCOPE)
│   │   └── §3.1.4 Trajectory Optimization
│   │             (ExPO, MENTOR)
│   │
│   └── §3.2 Scaffold-based Hints
│       ├── §3.2.1 Scaffold Injection
│       │   ├── Answer-level Scaffold
│       │   │         (RAVR, CoVRL, HDPO)
│       │   ├── Solution-blueprint Scaffold
│       │   │         (Guide-GRPO, A2D, PieceHint, SAGE_scaf, AVATAR, RLAD)
│       │   ├── Knowledge-level Scaffold
│       │   │         (KnowRL, NuRL)
│       │   └── Format-level Scaffold
│       │             (MeRF, RuscaRL)
│       ├── §3.2.2 Scaffold Replacement
│       │   ├── Pedagogical Guidance
│       │   │         (HINT, Scaf-GRPO, KEPO, HiLL)
│       │   ├── Critique-driven Refinement
│       │   │         (LTE, RGR-GRPO, R³L, GOLF, ECHO)
│       │   └── Action-level Guidance
│       │             (InfoFlow)
│       └── §3.2.3 Scaffold Optimization
│           ├── Auxiliary Supervision
│           │         (MEL, ThinkTuning, RLTF, KEPO)
│           └── Distribution Alignment
│                     (HDPO, RAVR, CoVRL)
│
├── §4 Task-level Hints
│   ├── §4.1 Static Experience Base
│   │   ├── Demonstration-based Experience
│   │   │         (CBRL, ICPO)
│   │   └── Skill-based Experience
│   │             (TemplateRL, SKILL0)
│   └── §4.2 Evolving Experience Base
│       ├── Accumulative Experience Base
│       │   ├── Reflection Experience
│       │   │         (EvolveR, RetroAgent, ERL, MAGE)
│       │   └── Strategy Experience
│       │             (SGE, CRMWeaver, IntPro, MetaClaw)
│       ├── Curated Experience Base
│       │   ├── Skill Evolution
│       │   │         (SkillRL, ARISE, COS-PLAY, SAGE_exp, K²-Agent)
│       │   └── Experience Revision
│       │             (BEPA, Comp.RL, INSPO, PEARL, AgentEvolver, DGO)
│       └── Optimized Experience Base
│           ├── Experience-derived Utility
│           │         (D2Skill, SLEA-RL, Skill-SD)
│           └── Trainable Experience Module
│                     (Trainable Graph Memory, UMEM)
│
├── §5 Domain-specific Hints
│   ├── Technical Domains
│   │         (Kevin, SGS, C2F-Thinker)
│   ├── Interactive Agent Domains
│   │         (COS-PLAY, WebGen-Agent, UI-S1)
│   └── Business & Personal-service Domains
│             (TaoSR-AGRL, CRMWeaver, VeriRole, PEARL)
│
└── §6 Discussion, Challenges and Future Directions
    ├── §6.1 Scope and Boundaries
    ├── §6.2 Cross-level Analysis of Hints
    └── §6.3 Challenges and Future Directionsurce, utilization mechanism, inference-time hint use)
```

## §3 Sample-level Hints

> **Sample-level hints** are the foundation of Hint-based RL. Each hint is exclusive to a single training sample and not shared across samples. Hints can originate from offline human annotations, verifiable text, or text selected and processed from the policy or teacher model during training. Their utilization falls into four categories, spanning all stages of the rollout. We further divide hints by content refinement into trajectories and scaffolds, and organize the taxonomy around the basic utilization and construction mechanisms. 

### §3.1 Trajectory-based Hints

> **Trajectory-based hints** take the form of policy-like responses, typically full solution trajectories or their prefixes. They expose concrete reasoning paths that the current policy may fail to sample reliably, making informative states more reachable and giving low-variance samples new rollout variations.

#### §3.1.1 Trajectory Injection

> Trajectory Injection places a trajectory segment in the input context while the policy still generates a full response.

| Abbr. | Paper | Data (1st time) | Publication | Resources |
|-|-|:-:|:-:|:-:|
| BREAD | 🌕️ [BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning](https://openreview.net/forum?id=NUDaln2vCe) | 2025.6 | NeurIPS 2025 |  |
| QuestA | 🌕️ [QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation](https://openreview.net/forum?id=3MifB0f7qR) | 2025.7 | ICLR 2026 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/foreverlasting1202/QuestA) |
| CCL | 🌕️ [Progressive Mastery: Customized Curriculum Learning with Guided Prompting for Mathematical Reasoning](https://arxiv.org/abs/2506.04065) | 2025.7 | arXiv preprint |  |
| GHPO | 🌕️ [GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning](https://arxiv.org/abs/2507.10628) | 2025.7 | arXiv preprint | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/hkgc-1/GHPO) |
| SEELE | 🌕️ [Staying in the Sweet Spot: Responsive Reasoning Evolution via Capability-Adaptive Hint Scaffolding](https://arxiv.org/abs/2509.06923) | 2025.9 | arXiv preprint | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/ChillingDream/seele) |
| RPO | 🌕️ [RPO: Reinforcement Fine-Tuning with Partial Reasoning Optimization](https://arxiv.org/abs/2601.19404) | 2026.1 | arXiv preprint | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/yhz5613813/RPO) |
| Failure-Prefix | 🌕️ [Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning](https://arxiv.org/abs/2601.20829) | 2026.1 | arXiv preprint | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/minwukim/training-on-saturated-problems) |
| CORE | 🌕️ [CORE: Collaborative Reasoning via Cross Teaching](https://arxiv.org/abs/2601.21600) | 2026.1 | arXiv preprint |  |
| POPE | 🌕️ [POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration](https://arxiv.org/abs/2601.18779) | 2026.1 | arXiv preprint | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/CMU-AIRe/POPE) |
| HiPO | 🌕️ [HiPO: Self-Hint Policy Optimization for RLVR](https://openreview.net/forum?id=rcb20pHmT1) | 2026.2 | ICLR 2026 |  |
| PROS | 🌕️ [PROS: Towards Compute-Efficient RLVR via Rollout Prefix Reuse](https://openreview.net/forum?id=lz1SRTcnUb) | 2026.2 | ICLR 2026 |  |

#### §3.1.2 Trajectory Continuation

> Trajectory Continuation uses a trajectory prefix as a fixed generation context and trains the policy to complete the suffix. The main design variable is how much prefix to reveal during training.

| Abbr. | Paper | Data (1st time) | Publication | Resources |
|-|-|:-:|:-:|:-:|
| Hint-GRPO | 🌕️ [Boosting MLLM Reasoning with Text-Debiased Hint-GRPO](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Boosting_MLLM_Reasoning_with_Text-Debiased_Hint-GRPO_ICCV_2025_paper.pdf) | 2025.5 | ICCV 2025 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/hqhQAQ/Hint-GRPO) |
| UFT | 🌕️ [UFT: Unifying Supervised and Reinforcement Fine-Tuning](https://openreview.net/forum?id=usOkGv1S7M) | 2025.5 | NeurIPS 2025 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/liumy2010/UFT) |
| Prefix-RFT | 🌕️ [Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling](https://arxiv.org/abs/2507.01679) | 2025.7 | ICML 2026 |  |
| StepHint | 🌕️ [StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason](https://arxiv.org/abs/2507.02841) | 2025.7 | ACL 2026 |  |
| EvoCoT | 🌕️ [EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning for LLMs](https://arxiv.org/abs/2508.07809) | 2025.8 | ACL 2026 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/gtxygyzb/EvoCoT) |
| G²RPO-A | 🌕️ [G²RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance](https://arxiv.org/abs/2508.13023) | 2025.8 | ACL 2026 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/T-Lab-CUHKSZ/G2RPO-A) |
| ADHint | 🌕️ [ADHint: Adaptive Hints with Difficulty Priors for Reinforcement Learning](https://arxiv.org/abs/2512.13095) | 2025.12 | arXiv preprint |  |
| TRAPO | 🌕️ [Trust-Region Adaptive Policy Optimization](https://openreview.net/forum?id=oXlSEcxD6N) | 2025.12 | ICLR 2026 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/Su-my/TRAPO) |
| BHA | 🌕️ [Mitigating Distribution Sharpening in Math RLVR via Distribution-Aligned Hint Synthesis and Backward Hint Annealing](https://arxiv.org/abs/2604.07747) | 2026.4 | arXiv preprint |  |

#### §3.1.3 Trajectory Continuation
> Trajectory Replacement modifies the rollout group with trajectory-level samples before the RL update. It changes the candidate set by inserting reliable trajectories or repairing failed ones.

| Abbr. | Paper | Data (1st time) | Publication | Resources |
|-|-|:-:|:-:|:-:|
| LUFFY | 🌕️ [Learning to Reason under Off-Policy Guidance](https://openreview.net/forum?id=vO8LLoNWWk) | 2025.9 | NeurIPS 2025 |  |
| AMPO | 🌕️ [More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration](https://arxiv.org/abs/2510.02227) | 2025.10 | arXiv preprint | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/EnigmaYYYY/AMPO) |
| ANCHOR | 🌕️ [Toward Honest Language Models for Deductive Reasoning](https://arxiv.org/abs/2511.09222) | 2025.11 | arXiv preprint |  |
| HiPO | 🌕️ [HiPO: Self-Hint Policy Optimization for RLVR](https://openreview.net/forum?id=rcb20pHmT1) | 2026.2 | ICLR 2026 |  |
| SCOPE | 🌕️ [Recycling Failures: Salvaging Exploration in RLVR via Fine-Grained Off-Policy Guidance](https://arxiv.org/abs/2602.24110) | 2026.2 | arXiv preprint |  |
| HAPO | 🌕️ [Hindsight-Anchored Policy Optimization: Turning Failure into Feedback in Sparse Reward Settings](https://openreview.net/forum?id=86xIZZ9lnT#discussion) | 2026.3 | ICLR-workshop 2026 |  |
| S-GRPO | 🌕️ [S-GRPO: Unified Post-Training for Large Vision-Language Models](https://arxiv.org/abs/2604.16557) | 2026.4 | arXiv preprint |  |

#### §3.1.4 Trajectory Optimization
> Trajectory Optimization converts trajectory hints into auxiliary update signals. 

[![Code](https://img.shields.io/badge/Code-GitHub-blue)]()


| Abbr. | Paper | Data (1st time) | Publication | Resources |
|-|-|:-:|:-:|:-:|
| ExPO | 🌕️ [ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](https://openreview.net/forum?id=D1PeGJtVEu) | 2025.7 | NeurIPS 2025 |  |
| MENTOR | 🌕️ [Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs](https://openreview.net/forum?id=axlFycAkoL) | 2025.10 | ICLR 2026 | [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/Jiangzs1028/MENTOR) |





