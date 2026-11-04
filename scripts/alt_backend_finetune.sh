#!/usr/bin/env bash
# Alternating-backend fine-tune from sprint_stand/model_59000.
set -u
cd "$(dirname "$0")/.."
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
run_dir="$(pwd)/examples/sprint_stand"
ckpt=59000
for phase in 1 2 3 4 5 6 7 8; do
  if (( phase % 2 )); then sim=motrix; else sim=mujoco; fi
  name=rsl_rl_ppo_alt_p${phase}
  echo "phase $phase sim=$sim from $run_dir @ $ckpt"
  uv run --no-sync microduck-train --algo ppo --task microduck_sprint_standturn_flat --sim $sim \
    algo.num_envs=2048 algo.max_iterations=200 algo.save_interval=200 algo.seed=$((42 + phase)) \
    algo.load_run="$run_dir" algo.checkpoint=$ckpt algo.algo_log_name=$name \
    training.task_name=MicroduckSprintFlat training.no_play=true training.device=cuda:0 \
    > logs/alt_p${phase}.log 2>&1 || { echo "phase $phase failed"; exit 1; }
  latest=$(ls -t $(find logs/$name -name "model_*.pt") | head -1)
  run_dir="$(pwd)/$(dirname "$latest")"
  ckpt=$(basename "$latest" .pt | sed 's/model_//')
  echo "phase $phase done -> $latest"
done
echo ALT_DONE "$run_dir/model_$ckpt.pt"
