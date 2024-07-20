import numpy as np
import pytest
from embdata.utils.geometry_utils import pose_to_transformation_matrix
from embdata.geometry import Pose, Pose6D
from embdata.motion import Motion


def test_pose_to_transformation_matrix():
    # Test with Pose object
    pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=0)
    result = pose_to_transformation_matrix([1, 2, 3, 0, 0, 0])
    expected = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    np.testing.assert_array_almost_equal(result, expected)

    # Test with numpy array
    pose_array = np.array([1, 2, 3, 0, 0, 0])
    result = pose_to_transformation_matrix(pose_array)
    np.testing.assert_array_almost_equal(result, expected)


