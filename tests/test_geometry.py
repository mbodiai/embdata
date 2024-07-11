import pytest
from embdata.geometry import Coordinate, Pose6D, CoordinateField, PlanarPose


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
    assert coord.theta == 1.57


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
    assert pose.roll == 0.5
    assert pose.pitch == 0.3
    assert pose.yaw == 1.57


def test_pose6d_bounds_validation(pose):
    pose_instance = pose(x=10)
    with pytest.raises(ValueError):
        pose_instance = pose(x=11)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
