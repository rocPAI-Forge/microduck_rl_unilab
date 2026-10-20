"""Project-owned FastSAC runtime for phase-conditioned MicroDuck velocity.

The runtime adds two algorithm changes without using PPO data or weights:

* sagittal mirror consistency on the deterministic actor mean;
* a linear target-entropy-ratio schedule.

It is selected through UniLab's ``algo.runtime_resolver`` extension point, so
normal PPO and non-MicroDuck SAC runs are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from uni_rl.algos.fast_sac.learner import FastSACLearner
from uni_rl.offpolicy.runtime import OffPolicyRuntime

from microduck_rl_unilab.tasks.microduck.symmetry import (
    mirror_microduck_velocity_tensors,
)


class MicroduckFastSACLearner(FastSACLearner):
    """FastSAC learner with actor mirror loss and target-entropy annealing."""

    def __init__(
        self,
        *args: Any,
        mirror_loss_coeff: float = 2.0,
        entropy_start_ratio: float = 0.0,
        entropy_end_ratio: float = 0.55,
        entropy_schedule_steps: int = 56_000,
        **kwargs: Any,
    ):
        if mirror_loss_coeff < 0.0:
            raise ValueError("mirror_loss_coeff must be non-negative")
        if not 0.0 <= entropy_start_ratio <= 1.0:
            raise ValueError("entropy_start_ratio must be within [0, 1]")
        if not 0.0 <= entropy_end_ratio <= 1.0:
            raise ValueError("entropy_end_ratio must be within [0, 1]")
        if entropy_schedule_steps <= 0:
            raise ValueError("entropy_schedule_steps must be positive")
        if bool(kwargs.get("obs_normalization", False)):
            raise ValueError(
                "Microduck mirror loss requires algo.obs_normalization=false "
                "so sign/permutation symmetry is applied in the raw actor space"
            )
        # Custom actor/alpha losses are intentionally kept off CUDA graphs.
        kwargs["use_cuda_graph_actor"] = False
        kwargs["use_cuda_graph_critic"] = False
        super().__init__(*args, **kwargs)
        self.mirror_loss_coeff = float(mirror_loss_coeff)
        self.entropy_start_ratio = float(entropy_start_ratio)
        self.entropy_end_ratio = float(entropy_end_ratio)
        self.entropy_schedule_steps = int(entropy_schedule_steps)
        self._microduck_entropy_step = 0
        self._microduck_entropy_ratio = self.entropy_start_ratio
        self._microduck_mirror_loss = torch.zeros((), device=self.device)

    def _actor_loss_tensors(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action_eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, entropy, action_std = super()._actor_loss_tensors(
            obs, critic_obs, action_eps
        )
        if self.mirror_loss_coeff == 0.0 or obs.shape[-1] not in (61, 64):
            self._microduck_mirror_loss = torch.zeros((), device=obs.device)
            return loss, entropy, action_std
        with self._autocast():
            action, _, _ = self.actor(obs)
            mirrored_obs, expected_action = mirror_microduck_velocity_tensors(obs, action)
            assert expected_action is not None
            mirrored_action, _, _ = self.actor(mirrored_obs)
            mirror_loss = (mirrored_action - expected_action.detach()).square().mean()
        self._microduck_mirror_loss = mirror_loss.detach()
        return loss + self.mirror_loss_coeff * mirror_loss, entropy, action_std

    def _alpha_loss_tensor(self, next_log_probs: torch.Tensor) -> torch.Tensor:
        fraction = min(1.0, self._microduck_entropy_step / self.entropy_schedule_steps)
        ratio = self.entropy_start_ratio + fraction * (
            self.entropy_end_ratio - self.entropy_start_ratio
        )
        action_dim = int(self.actor.action_scale.numel())
        self.target_entropy = -action_dim * ratio
        self._microduck_entropy_ratio = ratio
        self._microduck_entropy_step += 1
        return super()._alpha_loss_tensor(next_log_probs)

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        metrics = super().update_actor(batch)
        metrics["mirror_loss"] = float(self._microduck_mirror_loss.item())
        metrics["target_entropy_ratio"] = float(self._microduck_entropy_ratio)
        return metrics

    def get_state_dict(self) -> dict[str, Any]:
        state = super().get_state_dict()
        state["microduck_entropy_step"] = self._microduck_entropy_step
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._microduck_entropy_step = int(state_dict.get("microduck_entropy_step", 0))


@dataclass(frozen=True)
class MicroduckPureSACRuntime(OffPolicyRuntime):
    learner_cls: type[Any] | None = MicroduckFastSACLearner
    # The collector/playback action API is standard SAC; only the learner is
    # specialized.
    algo_type: str | None = "sac"
    learner_cfg: dict[str, Any] = field(default_factory=dict)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        del critic_obs_dim
        if obs_dim not in (61, 64):
            raise ValueError(f"Microduck pure SAC expects 61D or 64D actor obs, got {obs_dim}")
        return {
            "mirror_loss_coeff": float(self.learner_cfg.get("mirror_loss_coeff", 2.0)),
            "entropy_start_ratio": float(self.learner_cfg.get("entropy_start_ratio", 0.0)),
            "entropy_end_ratio": float(self.learner_cfg.get("entropy_end_ratio", 0.55)),
            "entropy_schedule_steps": int(
                self.learner_cfg.get("entropy_schedule_steps", 56_000)
            ),
        }


def resolve_microduck_pure_sac_runtime(
    rl_cfg: dict[str, Any],
) -> MicroduckPureSACRuntime | None:
    if rl_cfg.get("runtime_impl") != "microduck_pure_sac":
        return None
    learner_cfg = rl_cfg.get("actor", {})
    if not isinstance(learner_cfg, dict):
        raise TypeError("algo.actor must be a mapping for microduck_pure_sac")
    return MicroduckPureSACRuntime(learner_cfg=dict(learner_cfg))


__all__ = [
    "MicroduckFastSACLearner",
    "MicroduckPureSACRuntime",
    "resolve_microduck_pure_sac_runtime",
]
