# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
import numpy as np
from embdata.geometry import CoordinateField, Pose6D, Pose3D, Coordinate

def test_angular_conversion():
    class PoseRollDegrees(Pose6D):
        roll: float = CoordinateField(unit="deg")

    pose = PoseRollDegrees(roll=45.0)
    pose_in_radians: Pose6D = pose.to(angular_unit="rad")
    assert pytest.approx(pose_in_radians.roll, abs=1e-6) == np.pi / 4

def test_linear_conversion():
    class PoseMM(Pose6D):
        x: float = CoordinateField(unit="mm")
        y: float = CoordinateField(unit="mm")
        z: float = CoordinateField(unit="mm")

    pose = PoseMM(x=1000.0, y=0.0, z=1.0)
    pose_in_meters: PoseMM = pose.to(unit="m")
    assert pose_in_meters.x == 1.0
    assert pose_in_meters.y == 0.0
    assert pose_in_meters.z == 0.001

    pose_in_centimeters = pose.to(unit="cm")
    assert pose_in_centimeters.x == 100.0
    assert pose_in_centimeters.y == 0.0
    assert pose_in_centimeters.z == 0.1

def test_coordinate_creation():
    coord = Coordinate()
    assert coord is not None

def test_coordinate_bounds_validation():
    class BoundedCoordinate(Coordinate):
        x: float = CoordinateField(bounds=(0, 10))

    with pytest.raises(ValueError):
        BoundedCoordinate(x=11)

    valid_coord = BoundedCoordinate(x=5)
    assert valid_coord.x == 5

def test_pose3d_creation():
    pose = Pose3D(x=1, y=2, theta=np.pi/2)
    assert pose.x == 1
    assert pose.y == 2
    assert pytest.approx(pose.theta, abs=1e-6) == np.pi/2

def test_pose6d_quaternion():
    pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=np.pi/2, yaw=0)
    quat = pose.quaternion()
    assert pytest.approx(quat, abs=1e-3) == [0.0, 0.707, 0.0, 0.707]

def test_pose6d_rotation_matrix():
    pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=np.pi/2, yaw=0)
    rot_matrix = pose.rotation_matrix()
    expected_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(rot_matrix, expected_matrix, atol=1e-6)
