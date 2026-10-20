"""Project-owned FastSAC runtime contracts."""

from __future__ import annotations

import pytest
import torch

from microduck_rl_unilab.tasks.microduck.pure_sac import (
    MicroduckFastSACLearner,
    MicroduckPureSACRuntime,
    resolve_microduck_pure_sac_runtime,
)


def test_runtime_resolver_builds_project_learner_kwargs() -> None:
    runtime = resolve_microduck_pure_sac_runtime(
        {
            "runtime_impl": "microduck_pure_sac",
            "actor": {
                "mirror_loss_coeff": 3.0,
                "entropy_start_ratio": 0.1,
                "entropy_end_ratio": 0.8,
                "entropy_schedule_steps": 123,
            },
        }
    )

    assert isinstance(runtime, MicroduckPureSACRuntime)
    assert runtime.learner_cls is MicroduckFastSACLearner
    assert runtime.build_model_kwargs(obs_dim=64, critic_obs_dim=79) == {
        "mirror_loss_coeff": 3.0,
        "entropy_start_ratio": 0.1,
        "entropy_end_ratio": 0.8,
        "entropy_schedule_steps": 123,
    }


def test_pure_sac_actor_loss_and_entropy_schedule_are_finite() -> None:
    learner = MicroduckFastSACLearner(
        obs_dim=64,
        critic_obs_dim=79,
        action_dim=14,
        device="cpu",
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        num_atoms=11,
        use_layer_norm=False,
        mirror_loss_coeff=2.0,
        entropy_start_ratio=0.0,
        entropy_end_ratio=0.5,
        entropy_schedule_steps=2,
    )
    obs = torch.randn(8, 64)
    critic = torch.randn(8, 79)

    loss, entropy, action_std = learner._actor_loss_tensors(obs, critic)
    assert torch.isfinite(loss)
    assert torch.isfinite(entropy)
    assert torch.isfinite(action_std)
    assert learner._microduck_mirror_loss.item() >= 0.0

    log_probs = torch.zeros(8)
    learner._alpha_loss_tensor(log_probs)
    assert learner._microduck_entropy_ratio == pytest.approx(0.0)
    learner._alpha_loss_tensor(log_probs)
    assert learner._microduck_entropy_ratio == pytest.approx(0.25)
    learner._alpha_loss_tensor(log_probs)
    assert learner._microduck_entropy_ratio == pytest.approx(0.5)


def test_pure_sac_rejects_normalized_observation_symmetry() -> None:
    with pytest.raises(ValueError, match="obs_normalization=false"):
        MicroduckFastSACLearner(
            obs_dim=64,
            critic_obs_dim=79,
            action_dim=14,
            device="cpu",
            obs_normalization=True,
        )
