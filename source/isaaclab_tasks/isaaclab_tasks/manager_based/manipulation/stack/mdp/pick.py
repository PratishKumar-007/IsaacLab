from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_position_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """The position of the cube in the world frame."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w


def cube_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """The orientation of the cube in the world frame."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_quat_w


def pick_object_obs(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object observations for single-cube picking."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w
    cube_quat_w = cube.data.root_quat_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    gripper_to_cube = cube_pos_w - ee_pos_w

    return torch.cat(
        (
            cube_pos_w - env.scene.env_origins,
            cube_quat_w,
            gripper_to_cube,
        ),
        dim=1,
    )


def gripper_drive_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the drive joint position for Robotiq grippers (single joint)."""
    robot: Articulation = env.scene[robot_cfg.name]
    drive_joint_name = getattr(env.cfg, "gripper_drive_joint_name", None)
    if drive_joint_name is None:
        raise ValueError("gripper_drive_joint_name must be set in the environment config")

    joint_ids, _ = robot.find_joints([drive_joint_name])
    if len(joint_ids) != 1:
        raise ValueError(f"Expected a single drive joint for gripper, got {len(joint_ids)}")

    return robot.data.joint_pos[:, joint_ids[0]].unsqueeze(1)


def object_grasped_drive_joint(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped using a single gripper drive joint."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    drive_joint_name = getattr(env.cfg, "gripper_drive_joint_name", None)
    if drive_joint_name is None:
        raise ValueError("gripper_drive_joint_name must be set in the environment config")

    joint_ids, _ = robot.find_joints([drive_joint_name])
    if len(joint_ids) != 1:
        raise ValueError(f"Expected a single drive joint for gripper, got {len(joint_ids)}")

    drive_joint_pos = robot.data.joint_pos[:, joint_ids[0]]
    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
    closed = torch.abs(drive_joint_pos - open_val) > env.cfg.gripper_threshold

    return torch.logical_and(pose_diff < diff_threshold, closed)


def arm_home_reached(
    env: ManagerBasedRLEnv,
    arm_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_threshold: float = 0.08,
) -> torch.Tensor:
    """Check if the arm joints are close to their default positions."""
    robot: Articulation = env.scene[arm_joint_cfg.name]
    joint_pos = robot.data.joint_pos[:, arm_joint_cfg.joint_ids]
    joint_default = robot.data.default_joint_pos[:, arm_joint_cfg.joint_ids]

    return torch.all(torch.abs(joint_pos - joint_default) < joint_threshold, dim=1)


def object_grasped_and_home(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    arm_joint_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    joint_threshold: float = 0.08,
) -> torch.Tensor:
    """Success condition: object grasped and arm returned near default pose."""
    grasped = object_grasped_drive_joint(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        diff_threshold=diff_threshold,
    )
    home = arm_home_reached(env, arm_joint_cfg=arm_joint_cfg, joint_threshold=joint_threshold)
    return torch.logical_and(grasped, home)

