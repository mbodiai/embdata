import pytest
import numpy as np
from embdata.coordinate import Coordinate, Pose6D, CoordinateField, PlanarPose


@pytest.fixture
def pose():
    class Pose(Pose6D):
        x: float = CoordinateField(0.0, unit="m", bounds=(0, 10))

    return Pose


def test_coordinate_creation():
    coord = Coordinate()
    assert coord is not None


def test_coordinate_fields():
    coord = PlanarPose()
    assert coord.x == 0.0
    assert coord.y == 0.0
    assert coord.theta == 0.0


def test_coordinate_bounds():
    coord = PlanarPose()
    coord.x = 5.0
    coord.y = 10.0
    coord.theta = 1.57
    assert coord.x == 5.0
    assert coord.y == 10.0
    assert pytest.approx(coord.theta, abs=1e-6) == 1.57


def test_pose6d_fields(pose):
    pose = pose()
    assert pose.x == 0.0
    assert pose.y == 0.0
    assert pose.z == 0.0
    assert pose.roll == 0.0
    assert pose.pitch == 0.0
    assert pose.yaw == 0.0


def test_pose6d_bounds(pose):
    pose = pose()
    pose.x = 5.0
    pose.y = 10.0
    pose.z = 2.5
    pose.roll = 0.5
    pose.pitch = 0.3
    pose.yaw = 1.57
    assert pose.x == 5.0
    assert pose.y == 10.0
    assert pose.z == 2.5
    assert pytest.approx(pose.roll, abs=1e-6) == 0.5
    assert pytest.approx(pose.pitch, abs=1e-6) == 0.3
    assert pytest.approx(pose.yaw, abs=1e-6) == 1.57


def test_pose6d_bounds_validation(pose):
    pose_instance = pose(x=10)
    with pytest.raises(ValueError):
        pose_instance = pose(x=11)


def test_pose6d_to_conversion():
    pose = Pose6D(x=1, y=2, z=3, roll=np.pi / 4, pitch=np.pi / 3, yaw=np.pi / 2)

    # Test unit conversion
    pose_cm = pose.to(unit="cm")
    assert pose_cm.x == 100
    assert pose_cm.y == 200
    assert pose_cm.z == 300

    # Test angular unit conversion
    pose_deg = pose.to(angular_unit="deg")
    assert pytest.approx(pose_deg.roll, abs=1e-6) == 45
    assert pytest.approx(pose_deg.pitch, abs=1e-6) == 60
    assert pytest.approx(pose_deg.yaw, abs=1e-6) == 90

    # Test quaternion conversion
    quat = pose.to("quaternion")
    expected_quat = np.array([0.70105738, 0.09229596, 0.56098553, 0.43045933])
    assert np.allclose(quat, expected_quat, atol=1e-6)

    # Test rotation matrix conversion
    rot_matrix = pose.to("rotation_matrix")
    expected_matrix = np.array(
        [[0.35355339, -0.35355339, 0.8660254], [0.61237244, -0.61237244, -0.5], [0.70710678, 0.70710678, 0.0]]
    )
    assert np.allclose(rot_matrix, expected_matrix, atol=1e-6)


def test_planar_pose_to_conversion():
    pose = PlanarPose(x=1, y=2, theta=np.pi / 2)

    # Test unit conversion
    pose_cm = pose.to(unit="cm")
    assert pose_cm.x == 100
    assert pose_cm.y == 200

    # Test angular unit conversion
    pose_deg = pose.to(angular_unit="deg")
    assert pytest.approx(pose_deg.theta, abs=1e-6) == 90


def test_coordinate_conversion():
    coord = Pose6D(x=1.0, y=2.0, yaw=0.5)

    # Test linear unit conversion
    coord_cm = coord.to(unit="cm")
    assert coord_cm.x == 100.0
    assert coord_cm.y == 200.0
    assert coord_cm.yaw == 0.5

    # Test angular unit conversion
    coord_deg = coord.to(angular_unit="deg")
    assert pytest.approx(coord_deg.yaw, abs=1e-6) == 28.64788975654116

    # # Test invalid unit conversion
    # with pytest.raises(ValueError, KeyError):
    #     coord_invalid = coord.to(unit="invalid")


def test_coordinate_relative():
    coord1 = Pose6D(x=1.0, y=2.0, yaw=0.5)
    coord2 = Pose6D(x=0.5, y=1.0, yaw=0.25)

    # Test relative pose calculation
    relative_coord = coord1.relative_to(coord2)
    assert relative_coord.x == 0.5
    assert relative_coord.y == 1.0
    assert relative_coord.yaw == 0.25


def test_coordinate_absolute():
    coord1 = Pose6D(x=1.0, y=2.0, yaw=0.5)
    coord2 = Pose6D(x=0.5, y=1.0, yaw=0.25)

    # Test absolute pose calculation
    absolute_coord = coord1.absolute_from(coord2)
    assert absolute_coord.x == 1.5
    assert absolute_coord.y == 3.0
    assert absolute_coord.yaw == 0.75
