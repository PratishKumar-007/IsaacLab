# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from .ur10e_2f140_cfg import UR10_ROBOTIQ_CFG  # isort: skip


def close_gripper_when_near_command(env, env_ids, *, asset_cfg: SceneEntityCfg):
    """Close the gripper when EE is near the pose command."""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    ee_id = asset_cfg.body_ids[0]
    cmd = env.command_manager.get_command("object_pose")
    des_pos_b = cmd[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, ee_id, :]
    dist = torch.norm(des_pos_w[env_ids] - curr_pos_w[env_ids], dim=1)

    target = torch.where(
        dist < 0.10,
        torch.full((env_ids.numel(),), 0.4, device=env.device),
        torch.full((env_ids.numel(),), 0.0, device=env.device),
    )

    joint_id = asset_cfg.joint_ids[0] if isinstance(asset_cfg.joint_ids, list) else asset_cfg.joint_ids
    asset.data.joint_pos_target[env_ids, joint_id] = target


def apply_gripper_friction_material(
    env,
    env_ids,
    *,
    static_friction: float,
    dynamic_friction: float,
    link_names: list[str],
):
    """Apply a high-friction material to the gripper finger links."""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    stage = sim_utils.get_current_stage()
    material_cfg = sim_utils.materials.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )

    for env_id in env_ids.tolist():
        env_prim = env.scene.env_prim_paths[env_id]
        material_path = f"{env_prim}/Robot/GripperMaterial"
        material_cfg.func(material_path, material_cfg)
        for link_name in link_names:
            candidate_paths = [
                f"{env_prim}/Robot/{link_name}",
                f"{env_prim}/Robot/robotiq_base_link/{link_name}",
            ]
            for prim_path in candidate_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    sim_utils.bind_physics_material(prim_path, material_path)
                    break


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    auto_close_gripper = EventTerm(
        func=close_gripper_when_near_command,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["wrist_3_link"], joint_names=["finger_joint"], preserve_order=True
            ),
        },
    )

    gripper_friction_material = EventTerm(
        func=apply_gripper_friction_material,
        mode="startup",
        params={
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "link_names": [
                "left_inner_finger",
                "right_inner_finger",
                "left_outer_finger",
                "right_outer_finger",
                "left_inner_knuckle",
                "right_inner_knuckle",
                "left_outer_knuckle",
                "right_outer_knuckle",
            ],
        },
    )


@configclass
class UR10e2F140CubeLiftEnvCfg(LiftEnvCfg):
    """Lift task with UR10e + Robotiq 2F-140."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # reduce default parallel envs for stability (can be overridden via CLI)
        self.scene.num_envs = min(self.scene.num_envs, 256)
        # improve velocity accuracy
        self.sim.physx.enable_external_forces_every_iteration = True

        # Set UR10e with gripper
        self.scene.robot = UR10_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.4},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "wrist_3_link"

        # Set cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # Override events to include auto-gripper close
        self.events = EventCfg()

        # soften negative reward scaling to avoid divergence
        self.rewards.action_rate.weight = -5.0e-5
        self.rewards.joint_vel.weight = -5.0e-5
        self.rewards.object_goal_tracking.weight = 8.0
        self.rewards.object_goal_tracking_fine_grained.weight = 2.0


@configclass
class UR10e2F140CubeLiftEnvCfg_PLAY(UR10e2F140CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

