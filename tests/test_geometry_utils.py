import numpy as np
import pytest
from embdata.utils.geometry_utils import pose_to_transformation_matrix, compute_view_params
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


def test_compute_view_params():
    # Test with Pose objects
    camera_pos = np.array([0, 0, 0])
    target_pos = np.array([1, 1, 1])
    azimuth, distance, elevation, lookat = compute_view_params(camera_pos, target_pos)

    assert np.isclose(azimuth, 45)
    assert np.isclose(distance, np.sqrt(3))
    assert np.isclose(elevation, 35.264389682754654)
    np.testing.assert_array_almost_equal(lookat, [1, 1, 1])

    # Test with numpy arrays
    camera_pos_array = np.array([0, 0, 0])
    target_pos_array = np.array([1, 1, 1])
    azimuth, distance, elevation, lookat = compute_view_params(camera_pos_array, target_pos_array)

    assert np.isclose(azimuth, 45)
    assert np.isclose(distance, np.sqrt(3))
    assert np.isclose(elevation, 35.264389682754654)
    np.testing.assert_array_almost_equal(lookat, [1, 1, 1])
