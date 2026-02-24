# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Joint Position Control
##

gym.register(
    id="Isaac-Pick-Cube-UR10e-2F140-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_joint_pos_env_cfg:UR10e2F140CubePickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10e2F140PickPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pick-Cube-UR10e-2F140-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_joint_pos_env_cfg:UR10e2F140CubePickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10e2F140PickPPORunnerCfg",
    },
    disable_env_checker=True,
)

