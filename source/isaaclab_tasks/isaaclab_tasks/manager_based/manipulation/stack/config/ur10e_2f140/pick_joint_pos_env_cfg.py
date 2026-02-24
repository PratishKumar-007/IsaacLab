# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG  # isort: skip


UR10E_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.1, 0.1), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube")],
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the pick task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES)},
        )
        object = ObsTerm(func=mdp.pick_object_obs)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_drive_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES)},
        )
        object = ObsTerm(func=mdp.pick_object_obs)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_drive_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasped = ObsTerm(
            func=mdp.object_grasped_drive_joint,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        home = ObsTerm(
            func=mdp.arm_home_reached,
            params={"arm_joint_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the pick task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
    )

    success = DoneTerm(
        func=mdp.object_grasped_and_home,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube"),
            "arm_joint_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES),
            "diff_threshold": 0.06,
            "joint_threshold": 0.08,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the pick task."""

    reach_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={"std": 0.5, "object_cfg": SceneEntityCfg("cube")},
        weight=2.0,
    )
    lift_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube")},
        weight=10.0,
    )
    grasped = RewTerm(
        func=mdp.object_grasped_drive_joint,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube"),
            "diff_threshold": 0.06,
        },
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class UR10e2F140CubePickEnvCfg(StackEnvCfg):
    """Configuration for UR10e 2F-140 cube pick and return."""
    # disable XR in training to avoid hydra callable parsing issues
    xr = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override MDP configs
        self.observations = ObservationsCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.events = EventCfg()

        # robot
        base_joint_pos = dict(UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos)
        base_joint_pos.update(
            {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5707963267948966,
                "elbow_joint": 1.5707963267948966,
                "wrist_1_joint": -1.5707963267948966,
                "wrist_2_joint": -1.5707963267948966,
                "wrist_3_joint": 0.0,
            }
        )
        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=base_joint_pos, pos=(0.0, 0.0, 0.0)),
        )

        # action configs
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=UR10E_ARM_JOINT_NAMES, scale=0.15, use_default_offset=True
        )
        self.gripper_drive_joint_name = "finger_joint"
        self.gripper_open_val = 0.0
        self.gripper_closed_val = 0.69
        self.gripper_threshold = 0.02
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[self.gripper_drive_joint_name],
            open_command_expr={self.gripper_drive_joint_name: self.gripper_open_val},
            close_command_expr={self.gripper_drive_joint_name: self.gripper_closed_val},
        )

        # soften arm gains to reduce jerky motion and improve contact compliance
        self.scene.robot.actuators["shoulder"].stiffness = 800.0
        self.scene.robot.actuators["shoulder"].damping = 90.0
        self.scene.robot.actuators["elbow"].stiffness = 400.0
        self.scene.robot.actuators["elbow"].damping = 45.0
        self.scene.robot.actuators["wrist"].stiffness = 150.0
        self.scene.robot.actuators["wrist"].damping = 35.0

        # cube properties
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube")],
            ),
        )

        # end-effector frame
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
                    offset=OffsetCfg(pos=(0.22, 0.0, 0.0)),
                ),
            ],
        )


@configclass
class UR10e2F140CubePickEnvCfg_PLAY(UR10e2F140CubePickEnvCfg):
    """Play configuration for UR10e 2F-140 cube pick and return."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable observation corruption for play
        self.observations.policy.enable_corruption = False

