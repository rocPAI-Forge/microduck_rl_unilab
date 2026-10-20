"""Bilateral symmetry mapping for the MicroDuck 61D policy contract."""

from __future__ import annotations

import torch
from tensordict import TensorDict

_JOINT_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
_JOINT_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

_OBS_PERM = (
    (0, 1, 2)
    + (3, 4, 5)
    + tuple(6 + joint for joint in _JOINT_PERM)
    + tuple(20 + joint for joint in _JOINT_PERM)
    + tuple(34 + joint for joint in _JOINT_PERM)
    + (48, 49, 50)
    + (51, 52, 53, 54)
    + (55, 56, 57, 58, 59, 60)
)
_OBS_SIGN = (
    (-1.0, 1.0, -1.0)
    + (1.0, -1.0, 1.0)
    + _JOINT_SIGN
    + _JOINT_SIGN
    + _JOINT_SIGN
    + (1.0, -1.0, -1.0)
    + (1.0, 1.0, -1.0, -1.0)
    + (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
)
# SAC appends ``[cos(phase), sin(phase), 0]``. Mirroring exchanges stance
# legs, which is a half-cycle phase shift and therefore negates cos and sin.
_SAC_PHASE_PERM = (61, 62, 63)
_SAC_PHASE_SIGN = (-1.0, -1.0, 1.0)

_TENSOR_CACHE: dict[
    torch.device,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _mapping_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = _TENSOR_CACHE.get(device)
    if cached is None:
        cached = (
            torch.tensor(_OBS_PERM, dtype=torch.long, device=device),
            torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device),
            torch.tensor(_JOINT_PERM, dtype=torch.long, device=device),
            torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device),
        )
        _TENSOR_CACHE[device] = cached
    return cached


def microduck_velocity_symmetry(
    env: object,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append sagittal-plane mirrored actor observations and actions."""
    del env
    augmented_obs: TensorDict | None = None
    augmented_actions: torch.Tensor | None = None

    if obs is not None:
        actor = obs["policy"]
        critic = obs["critic"]
        obs_perm, obs_sign, _, _ = _mapping_tensors(actor.device)
        mirrored_actor = actor[:, obs_perm] * obs_sign
        augmented_obs = TensorDict(
            {
                "policy": torch.cat((actor, mirrored_actor), dim=0),
                # Mirror loss consumes actor observations only. Keep the critic
                # batch shape valid without inventing privileged-foot semantics.
                "critic": torch.cat((critic, critic), dim=0),
            },
            batch_size=[actor.shape[0] * 2],
            device=actor.device,
        )

    if actions is not None:
        _, _, action_perm, action_sign = _mapping_tensors(actions.device)
        mirrored_actions = actions[:, action_perm] * action_sign
        augmented_actions = torch.cat((actions, mirrored_actions), dim=0)

    return augmented_obs, augmented_actions


def mirror_microduck_velocity_tensors(
    actor: torch.Tensor,
    actions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mirror raw PPO (61D) or phase-conditioned SAC (64D) tensors."""
    if actor.ndim != 2 or actor.shape[1] not in (61, 64):
        raise ValueError(f"actor observations must have shape (N, 61|64), got {actor.shape}")
    obs_perm, obs_sign, action_perm, action_sign = _mapping_tensors(actor.device)
    if actor.shape[1] == 64:
        obs_perm = torch.cat(
            (obs_perm, torch.tensor(_SAC_PHASE_PERM, dtype=torch.long, device=actor.device))
        )
        obs_sign = torch.cat(
            (obs_sign, torch.tensor(_SAC_PHASE_SIGN, dtype=actor.dtype, device=actor.device))
        )
    mirrored_actor = actor[:, obs_perm] * obs_sign.to(dtype=actor.dtype)
    if actions is None:
        return mirrored_actor, None
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"actions must have shape (N, 14), got {actions.shape}")
    mirrored_actions = actions[:, action_perm] * action_sign.to(dtype=actions.dtype)
    return mirrored_actor, mirrored_actions


__all__ = ["microduck_velocity_symmetry", "mirror_microduck_velocity_tensors"]
