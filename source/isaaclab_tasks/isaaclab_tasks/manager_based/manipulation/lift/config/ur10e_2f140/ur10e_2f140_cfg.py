# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of UR10e arm with Robotiq 2F-140 gripper using implicit actuators."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_NAT_FREQ = 0.5
_DAMP_RATIO = 1.0
_K_P = (2.0 * 3.141592653589793 * _NAT_FREQ) ** 2
_K_D = 2.0 * _DAMP_RATIO * 2.0 * 3.141592653589793 * _NAT_FREQ

UR10_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        variants={"Gripper": "Robotiq_2f_140"},
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Arm joints
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.0,
            "elbow_joint": 1.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            # Gripper joints
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
            "left_inner_finger_pad_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit_sim=20.0,
            effort_limit_sim=87.0,
            stiffness=_K_P,
            damping=_K_D,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_joint",
                "right_inner_finger_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_finger_pad_joint",
                "left_inner_finger_pad_joint",
            ],
            velocity_limit_sim=20.0,
            effort_limit_sim=200.0,
            stiffness=_K_P,
            damping=_K_D,
        ),
    },
)

