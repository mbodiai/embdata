# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Motions to control a robot.

This module defines the motions to control a robot as pydantic models.

Motions:
    JointControl: A joint value, typically an angle.
    FullJointControl: A lsit of joint values.
    HandControl: A 7D space representing x, y, z, roll, pitch, yaw, and openness of the hand.
    HeadControl: Tilt and pan.
    MobileSingleArmControl: Control for a robot that can move in planar space with a single arm.
    BimanualArmControl: Control for a robot that can move planar space with two arms.
    HumanoidControl: Control for a robot with two arms, two legs, and a head.

Example:
    To create a new Pydantic model for a motion, inherit from the Motion class and define pydantic fields with the MotionField,
    function as you would with any other Pydantic field.

Example:
    from embdata.motion import Motion, AbsoluteMotionField, MotionField, MotionType, VelocityMotionField
    from embdata.geometry import PlanarPose, Pose

    class HandControl(Motion):
        pose: Pose = RelativeMotionField(default_factory=Pose, description="Pose of the robot hand.", motion_type="relative")
        grasp: float = AbsoluteMotionField(
            default=0,
            bounds=[-1,1],
            description="Openness of the robot hand. -1 is closed, 1 is open.",
            )

    You can also use the RelativeMotionField and VelocityMotionField or TorqueMotionField for different types of motions.
"""

import numpy as np

from embdata.geometry import PlanarPose, Pose
from embdata.motion import AbsoluteMotionField, Motion, MotionField, RelativeMotionField
from embdata.ndarray import NumpyArray


class HandControl(Motion):
    """Action for a 7D space representing x, y, z, roll, pitch, yaw, and oppenness of the hand."""

    pose: Pose = MotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: float = AbsoluteMotionField(
        default=0,
        bounds=[-1, 1],
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )


class AbsoluteHandControl(Motion):
    pose: Pose = AbsoluteMotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: float = AbsoluteMotionField(
        default=0,
        bounds=[-1, 1],
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )


class RelativePoseHandControl(Motion):
    pose: Pose = RelativeMotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: float = AbsoluteMotionField(
        0,
        bounds=[-1, 1],  # [-1,1] so that 0 is a no-op.
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )


class HeadControl(Motion):
    tilt: float = MotionField(0.0, description="Tilt of the robot head in radians (down is negative).")
    pan: float = MotionField(0.0, description="Pan of the robot head in radians (left is negative).")


class MobileSingleHandControl(Motion):
    """Control for a robot that can move its base in 2D space with a 6D EEF control + grasp."""

    # Location of the robot on the ground.
    base: PlanarPose | None = MotionField(
        default_factory=PlanarPose,
        description="Location of the robot on the ground.",
    )
    hand: HandControl | None = MotionField(default_factory=HandControl, description="Control for the robot hand.")
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


class MobileSingleArmControl(Motion):
    """Control for a robot that can move in 2D space with a single arm."""

    base: PlanarPose | None = MotionField(
        default_factory=PlanarPose,
        description="Location of the robot on the ground.",
    )
    arm: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the robot arm.",
    )
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


class MobileBimanualArmControl(Motion):
    """Control for a robot that can move in 2D space with two arms."""

    base: PlanarPose | None = MotionField(
        default_factory=PlanarPose,
        description="Location of the robot on the ground.",
    )
    left_arm: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the left robot arm.",
    )
    right_arm: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the right robot arm.",
    )
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


class HumanoidControl(Motion):
    """Control for a humanoid robot."""

    left_arm: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the left robot arm.",
    )
    right_arm: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(7),
        description="Control for the right robot arm.",
    )
    left_leg: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(6),
        description="Control for the left robot leg.",
    )
    right_leg: NumpyArray | None = MotionField(
        default_factory=lambda: np.zeros(6),
        description="Control for the right robot leg.",
    )
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")
