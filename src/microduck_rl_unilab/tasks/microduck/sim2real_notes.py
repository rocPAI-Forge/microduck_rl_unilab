"""Known deployment gaps for the MicroDuck Manager-Based task."""

from __future__ import annotations

from .deploy_contract import MICRODUCK_ACTOR_OBS_DIM, MICRODUCK_OBS_SEGMENTS

POLLEN_RUNTIME_OBS_DIM = MICRODUCK_ACTOR_OBS_DIM

# Known sim-to-real gaps (docs/DEPLOY.md has the full list):
# 1. Every task drives joints through the BAM XL330 M6 voltage-servo model.
#    Only policies trained with clip_target_to_joint_limits clamp q_des the way
#    the servo clamps its goal position; earlier sprint checkpoints rely on
#    targets far past the stops.
# 2. Optional backlash hinges are not represented in the current model.
# 3. Observation delay/noise envelopes still need hardware calibration.
# 4. The 1.0 m/s sprint gait loses 16-36 points of 300 s survival on
#    MotrixSim (docs/audits/motrix_crosscheck.md); 0-0.6 m/s transfers.


def obs_layout_for_export() -> list[tuple[str, int]]:
    """Return the stable actor observation layout used by deploy tooling."""
    return list(MICRODUCK_OBS_SEGMENTS)
