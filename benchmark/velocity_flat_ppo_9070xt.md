# Velocity flat PPO 吞吐：RX 9070 XT

2026-10-14，在本仓库跑 `microduck_velocity_flat` 的短程 PPO。同一套步态测了 MuJoCo 和 MotrixSim，并和 [Dive into Embodied AI 的 AMD MicroDuck 记录](https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/docs/practices/amd/microduck-rl/index.md) 里同一规模的 smoke 对照。

吞吐定义与 UniLab 训练日志一致：每轮

`steps/s = num_envs × num_steps_per_env / (collection_time + learning_time)`。

第 0 轮含初始化。下面的「最终」取最后一轮，「最高」取第 1 轮之后的最大值。

## 硬件

| 项 | 值 |
|---|---|
| 机器 | Ubuntu 24.04.3 LTS，内核 7.0.0-34-generic |
| CPU | AMD Ryzen AI 9 HX 370，24 逻辑核，内存 61 GiB |
| 训练 GPU | AMD Radeon RX 9070 XT，`gfx1201`，显存 16 GiB（`rocm-smi` 报告 17095983104 B） |
| 未参与训练 | 核显 Radeon 890M（`HIP_VISIBLE_DEVICES=0`，只露出 9070 XT） |
| ROCm | 7.2.0 |

## 软件与规模

两边共用：PPO，`num_steps_per_env=24`，`max_iterations=5`，`algo.seed=42`，`env.seed=42`，BAM 电压伺服，物理在 CPU，PPO 更新在 `cuda:0`。`training.no_play=true`，`training.logger=tensorboard`。Python 3.12.12，Torch 2.11.0+rocm7.2（HIP 7.2.26015），`unilab==1.0.0`，`unilab-rl==1.0.0`，`unisim-core==1.0.0`。

| 后端 | 额外包 | 相对 MuJoCo owner 关掉的事件 |
|---|---|---|
| `mujoco` | `mujoco-uni-runtime==0.4.0`，MuJoCo 3.8.0（`mujoco-uni`） | 无 |
| `motrix` | `motrixsim-core==0.8.1.dev104665` | `push_robot`、`randomize_armature`、`foot_friction`、`randomize_mass_inertia` |

这次机器上的 MotrixSim 是 UniLab ROCm 环境里已有的 `0.8.1.dev104665`。仓库安装说明里的针是 `unilab[motrix]` 的 `motrixsim-core==0.8.2`。

```text
microduck-train --algo ppo --task microduck_velocity_flat --sim <mujoco|motrix> \
  algo.num_envs=<64|256> algo.max_iterations=5 algo.seed=42 \
  training.no_play=true training.device=cuda:0 training.logger=tensorboard
```

MuJoCo 那次还加了 `+env.seed=42` 和 `MUJOCO_GL=egl`。Motrix owner 自己写了 `env.seed=42`。

## MuJoCo

64 环境 × 5 轮 = 7,680 transitions。最终 **6,604 steps/s**，最高 **7,199 steps/s**。

| 轮 | 累计步数 | steps/s | 采样 | 学习 |
|---:|---:|---:|---:|---:|
| 0 | 1,536 | 575 | 1.276 s | 1.394 s |
| 1 | 3,072 | 6,090 | 0.175 s | 0.077 s |
| 2 | 4,608 | 7,199 | 0.151 s | 0.063 s |
| 3 | 6,144 | 6,752 | 0.154 s | 0.073 s |
| 4 | 7,680 | 6,604 | 0.173 s | 0.059 s |

256 环境 × 5 轮 = 30,720 transitions。最终 **14,629 steps/s**，最高 **16,252 steps/s**。稳定后每轮大约 0.31 秒采样、0.10 秒学习。

| 轮 | 累计步数 | steps/s | 采样 | 学习 |
|---:|---:|---:|---:|---:|
| 0 | 6,144 | 5,433 | 0.629 s | 0.502 s |
| 1 | 12,288 | 15,122 | 0.311 s | 0.095 s |
| 2 | 18,432 | 16,252 | 0.283 s | 0.095 s |
| 3 | 24,576 | 15,090 | 0.312 s | 0.095 s |
| 4 | 30,720 | 14,629 | 0.324 s | 0.096 s |

## MotrixSim

64 环境 × 5 轮 = 7,680 transitions。最终 **5,082 steps/s**，最高 **5,716 steps/s**。

| 轮 | 累计步数 | steps/s | 采样 | 学习 |
|---:|---:|---:|---:|---:|
| 0 | 1,536 | 1,501 | 0.536 s | 0.487 s |
| 1 | 3,072 | 5,716 | 0.206 s | 0.062 s |
| 2 | 4,608 | 4,818 | 0.257 s | 0.062 s |
| 3 | 6,144 | 4,657 | 0.270 s | 0.059 s |
| 4 | 7,680 | 5,082 | 0.218 s | 0.084 s |

256 环境 × 5 轮 = 30,720 transitions。最终 **13,503 steps/s**，最高 **15,207 steps/s**。稳定后每轮大约 0.35 秒采样、0.10 秒学习。

| 轮 | 累计步数 | steps/s | 采样 | 学习 |
|---:|---:|---:|---:|---:|
| 0 | 6,144 | 5,350 | 0.643 s | 0.505 s |
| 1 | 12,288 | 15,207 | 0.309 s | 0.095 s |
| 2 | 18,432 | 13,178 | 0.371 s | 0.096 s |
| 3 | 24,576 | 13,847 | 0.348 s | 0.095 s |
| 4 | 30,720 | 13,503 | 0.360 s | 0.095 s |

## 同机对照

两边都是本仓库、同一台 9070 XT、同一 PPO 规模。256 环境下学习时间都是约 0.10 秒，差别在采样。MotrixSim 少了上面四项随机化，所以这组数字是各自可运行配置的吞吐。

| 规模 | transitions | MuJoCo 最终 / 最高 | MotrixSim 最终 / 最高 |
|---|---:|---:|---:|
| 64 env × 5 | 7,680 | 6,604 / 7,199 steps/s | 5,082 / 5,716 steps/s |
| 256 env × 5 | 30,720 | 14,629 / 16,252 steps/s | 13,503 / 15,207 steps/s |

## 与 R9700 / mjlab 记录对照

对照来源是上面那篇文档里的 smoke，任务同为平地速度跟踪，规模同为 64 与 256 环境、5 轮。对方机器是 Radeon AI PRO R9700（同样 `gfx1201`，32 GiB），ROCm 7.2.4，栈是 mjlab 1.3.0 + ROCm MuJoCo Warp，物理在 GPU 上。表里的本仓库数字取 MuJoCo 后端。

| 规模 | transitions | 本仓库 MuJoCo 最终 / 最高 | 对方记录 |
|---|---:|---:|---:|
| 64 env × 5 | 7,680 | 6,604 / 7,199 steps/s | 约 1,033 steps/s |
| 256 env × 5 | 30,720 | 14,629 / 16,252 steps/s | 约 3,741 steps/s |

对方长训（256 环境、500 轮及后续续训）的最终 / 最高吞吐约 3,991 / 4,301 steps/s，续训阶段约 5,275 / 5,517 steps/s。那是更长的运行，和这次 5 轮 smoke 不是同一口径。

两条栈的物理后端不同：本记录是 UniLab Manager 运行时加 CPU MuJoCo，对方是 GPU 上的 MuJoCo Warp，并且为了修正动态接触，每个 HIP step 都重算 broadphase。小批量时 CPU MuJoCo 的单轮吞吐可以高于那条 Warp 路径。
