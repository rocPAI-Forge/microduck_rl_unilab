# microduck_rl_unilab

[English](README.md) | [中文](README.zh.md)

Pollen Robotics MicroDuck 机器人 RL 训练仓库，**只依赖 UniLab 的包分发，不依赖 UniLab 源码**。
任务复刻上游 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
@ `29e887e`（对齐契约见 `src/microduck_rl_unilab/tasks/microduck/alignment_contract.py`，
审计脚本 `scripts/audit_microduck_alignment.py`）。

本仓库展示 UniLab 对外部分发的两个对接缝（seam）：

1. **任务注册**：`UNILAB_EXTRA_REGISTRY_PACKAGES` 环境变量让 UniLab 的
   `ensure_registries()` 导入本仓库的任务包（含 spawn 子进程）。
2. **配置组合**：Hydra `--config-dir` 把 `src/microduck_rl_unilab/conf/<algo>` 追加进
   对应算法 UniLab 训练脚本的 config search path，外部 `task=<task>/<sim>` owner YAML
   直接参与组合；conf 树按算法分目录（`conf/ppo/`、`conf/sac/`），与 UniLab 仓内布局一致。

训练、回放、runner/learner/collector 全部来自 `unilab` wheel；
本仓库只携带任务代码（manager terms / recovery terms / BAM action 等）、
owner 配置和机器人 XML 资产。

## 任务 × 算法 × 后端矩阵

| task | ppo | sac |
|------|-----|-----|
| `microduck_velocity_flat` | mujoco, motrix | mujoco, motrix |
| `microduck_sprint_flat` | mujoco | — |
| `microduck_sprint_robust_flat` | mujoco | — |
| `microduck_sprint_gaitfix_flat` | mujoco | — |
| `microduck_sprint_rollfix_flat` | mujoco | — |
| `microduck_velstand_flat` | mujoco | — |
| `microduck_standup_flat` | mujoco | — |

`microduck_velocity_flat` 的 PPO 已按 owner 预算训完（2026-10-15，2048 环境，500 轮，`algo.seed=42`，不回放）。

| 后端 | 最终平均回报 | 权重 |
|---|---:|---|
| mujoco | 93.01 | [`examples/velocity_flat_mujoco/model_499.pt`](examples/velocity_flat_mujoco/model_499.pt) |
| motrix（`motrixsim-core==0.8.2`） | 145.44 | [`examples/velocity_flat_motrix/model_499.pt`](examples/velocity_flat_motrix/model_499.pt) |

训练复现状态（只有指标和推理视频都通过才记为完成）：

| 策略 | 状态 | checkpoint / 视频 |
|---|---|---|
| `microduck_standup_flat` PPO | 完成；15000 轮，困难初态混合下可从趴、仰、坐姿站起 | [`examples/standup/model_14999.pt`](examples/standup/model_14999.pt) / [`play_video.mp4`](examples/standup/play_video.mp4) |
| `microduck_velstand_flat` PPO | 已观察；强制趴下开场时 4 台里 2 台在 1 秒内起身。侧面演示见下方示例 | [`examples/velstand_getup/model_2250.pt`](examples/velstand_getup/model_2250.pt) / [`play_video_side.mp4`](examples/velstand_getup/play_video_side.mp4) |
| `microduck_velocity_flat` SAC | **未通过**；5000 轮训练完成，但推理反复倒地重置，不能作为成品策略。失败样本留在本地 `logs/`，不入库 | — |

任务与上游 microduck_rl 保持 1:1：BAM（bam xl330-m6 电压伺服模型，
`BamVoltageAction` 逐 substep 力矩路径）是上游唯一的驱动器模型，因此本仓库
所有任务统一走 BAM，不再保留早期移植的简化位置驱动变体。
BAM 的 substep 状态反馈契约（`SimBackend.set_pre_step_control`）在 mujoco 与
motrix 后端可用。motrix 上关掉了该后端没有的速度推力、关节 armature、足端摩擦和惯量随机化。

## 安装

```bash
git clone https://github.com/unilabsim/microduck_rl_unilab.git
cd microduck_rl_unilab
# ROCm（默认）：Linux x86_64 从 ROCm 7.2 索引解析 torch 2.11.0 与 triton-rocm 3.6.0。
# 需要系统 ROCm >= 7.1。
uv sync
```

CUDA 是另一套配置（Linux x86_64 与 Windows 上为 cu128 索引的 `torch==2.8.0`）。
`make sync-cuda` 会把 `pyproject.cuda.toml` 和 `uv.cuda.lock` 换成当前工程文件并同步。
`make sync-rocm` 切回默认的 ROCm。两套环境的训练设备名都仍写 `cuda`。不要把换进去的文件提交。

> **依赖**：`unilab[mujoco]==1.0.0`、`unilab-rl==1.0.0`、`unisim-core==1.0.0`
> 全部从生产 PyPI 解析。unilab 1.0.0 起包含仓内 microduck 任务移除（PR #1495）、
> `read_reset_root_pose` 基础 API（PR #1494）与 `unilab-rl==1.0.0` 升级（PR #1496）。
> `--sim motrix` 另需 `unilab[motrix]`（`motrixsim-core==0.8.2`）。

## 训练

在仓库根目录执行（`env.scene.model_file` 相对仓库根解析）：

```bash
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco
uv run microduck-train --algo ppo --task microduck_sprint_flat --sim mujoco
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim motrix
uv run microduck-train --algo ppo --task microduck_standup_flat --sim mujoco
uv run microduck-train --algo sac --task microduck_velocity_flat --sim mujoco
```

短程冒烟（4 个 env、2 次迭代、不回放）：

```bash
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

任意 UniLab Hydra override 均可透传（如 `algo.num_envs=512`、
`training.logger=wandb`）。

## 回放

```bash
uv run microduck-eval --algo ppo --task microduck_velocity_flat --sim mujoco --load-run -1
```

Sprint 评估口径为 2.20 m/s 前向命令、1 秒预热、10 秒测量：

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_speed_1p68/model_11997.pt
```

从已训好的速度策略做三段 robustify（钉住 1.65–2.20 m/s 命令带，逐步加 push / CoM / 倾角）：

```bash
uv run --no-sync scripts/train_sprint_robust.py \
  --load-run <sprint-run> --checkpoint <last-iter> --stages B
```

## 示例 checkpoint

| 示例 | 速度 | 存活 | 头部倒置 | 说明 |
|---|---|---|---|---|
| [`examples/sprint_speed_1p68/`](examples/sprint_speed_1p68/) | 1.682 m/s | 89.8% | ~91% | 速度配方。大腿已左右交替；颈部折叠，头大部分时间倒置。 |
| [`examples/sprint_head_upright/`](examples/sprint_head_upright/) | 1.315 m/s | 97.9% | 0.6% | 同一套交替步态，头顶朝天、面朝前；10 s 航向仍约 42° 左偏。 |
| [`examples/sprint_straight_long/`](examples/sprint_straight_long/) | ~0.83 m/s 长时 | 300 s 100% | 0 | 续训后闭环节航向 + 横向误差。300 s 平均航向 1.7°，直姿窗口 295 s。 |

每个目录含 `model_*.pt`、可视化、`metrics.json` 和复现命令。
`model.pt` 约 4.7 MB，直接入 git。

跌倒起身示例 [`examples/velstand_getup/`](examples/velstand_getup/)：

| 文件 | 内容 |
|---|---|
| [`model_2250.pt`](examples/velstand_getup/model_2250.pt) | 已观察权重，第 2250 轮。standup actor 热启动后训了 4000 轮 |
| [`play_video_side.mp4`](examples/velstand_getup/play_video_side.mp4) | 侧面近景，正常速度，约 3 秒：趴下、撑起、蹲起、站稳 |
| [`play_video_side_slow.mp4`](examples/velstand_getup/play_video_side_slow.mp4) | 同一段，3 倍慢放 |

seed 42 下 4 个趴下开场里，环境 0 和 2 在 1 秒内站起，1 和 3 在 8 秒窗口内一直躺着。视频跟的是环境 2。直接回放会从站立开场，因为课程表第 0 阶段把 `prone_prob` 写回 0；复现命令在该目录的 README 里。

长时直线评测（需要闭环 owner，不要用默认 `microduck_sprint_flat`）：

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_straight_long/model_19297.pt \
  --owner microduck_sprint_straightfix_flat/mujoco \
  --duration 300 --num-envs 128
```

## 吞吐

短程 PPO 吞吐记在 [`benchmark/`](benchmark/README.md)。2026-10-14 在 RX 9070 XT 上，`microduck_velocity_flat`、256 环境、5 轮的最终吞吐：MuJoCo 14,629 steps/s，MotrixSim 13,503 steps/s。

## 测试

```bash
uv run pytest tests/ -x -q
```

`tests/conftest.py` 复现两个 seam：设置 `UNILAB_EXTRA_REGISTRY_PACKAGES` 并通过
Hydra `SearchPathPlugin` 把本仓库 `conf/<algo>` 追加进 config search path
（测试不经 CLI，无法用 `--config-dir`）。

## 资产策略

全部 MicroDuck 资产（7 个 XML + 47 个 STL + 上游 LICENSE + sha256 清单）随仓库
入 git，开箱即用；`assets.py` 仅作为兜底——文件缺失时才从 Hugging Face 数据集
[`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots)
补齐（冷路径）。STL 来源为上游 pollen-robotics/microduck_rl（Apache-2.0，
见 `assets/robots/microduck/LICENSE.pollen-robotics.txt`），完整性由
`assets/robots/microduck/assets.sha256` 与 `tests/test_asset_contract.py` 保证。

## 仓库结构

```text
assets/robots/microduck/            # 机器人 XML + STL + LICENSE + sha256 清单（全部入 git）
src/microduck_rl_unilab/
├── assets.py                       # 冷路径资产物化（snapshot_download 兜底）
├── cli.py                          # microduck-train / microduck-eval：env var + --config-dir 注入
├── conf_searchpath.py              # Hydra SearchPathPlugin（测试/脚本的程序化 compose 用）
├── conf/
│   ├── ppo/task/microduck_{velocity_flat,sprint_flat,sprint_robust_flat,velstand_flat,standup_flat,ground_pick_flat,sitstand_flat}/
│   └── sac/task/microduck_velocity_flat/
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── microduck/
        ├── __init__.py             # 任务 registry.register_env
        ├── manager_terms.py        # velocity command / 奖励 terms
        ├── sprint_terms.py         # sprint 速度/航向奖励与速度课程
        ├── recovery_terms.py       # velstand 跌倒恢复 terms
        ├── standup_terms.py        # standup / ground_pick / sitstand terms
        ├── bam_action.py           # BAM 电压驱动 action term
        ├── deploy_contract.py      # obs/action 维度契约
        ├── alignment_contract.py   # 上游 microduck_rl @ 29e887e 对齐契约表
        └── sim2real_notes.py       # sim2real 注意事项
benchmark/                          # 吞吐实测（硬件、规模、对照）
tests/                              # 迁移自 UniLab 的契约/对齐测试套件
scripts/                            # 对齐审计 / 对比 / rollout 脚本
```
