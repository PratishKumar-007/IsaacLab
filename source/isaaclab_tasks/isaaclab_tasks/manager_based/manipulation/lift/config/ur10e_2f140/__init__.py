# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Cube-UR10e-2F140-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140CubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10e2F140LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-UR10e-2F140-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140CubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10e2F140LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

