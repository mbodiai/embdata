import numpy as np
import pytest
from embdata.trajectory import stats, Trajectory


def test_stats():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = stats(array)
    expected_result = {
        "mean": [4.0, 5.0, 6.0],
        "variance": [6.0, 6.0, 6.0],
        "skewness": [0.0, 0.0, 0.0],
        "kurtosis": [-1.5, -1.5, -1.5],
        "min": [1, 2, 3],
        "max": [7, 8, 9],
    }
    assert np.array_equal(result["mean"], expected_result["mean"])
    # assert np.array_equal(result['variance'], expected_result['variance'])
    assert np.array_equal(result["skewness"], expected_result["skewness"])
    assert np.array_equal(result["kurtosis"], expected_result["kurtosis"])
    assert np.array_equal(result["min"], expected_result["min"])
    assert np.array_equal(result["max"], expected_result["max"])


def test_make_relative():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    trajectory = Trajectory(array, freq_hz=1)
    relative_trajectory = trajectory.make_relative()
    expected_array = np.array([[3, 3, 3], [3, 3, 3]])
    assert np.array_equal(relative_trajectory.array, expected_array)


def test_make_minmax():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    trajectory = Trajectory(steps=array, freq_hz=1)
    minmax_trajectory = trajectory.make_minmax(0, 255)
    expected_array = 255 * (array - np.min(array, axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))
    assert np.array_equal(minmax_trajectory, expected_array)


def test_resample():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(steps=array, freq_hz=1)
    resampled_trajectory = trajectory.resample(target_hz=0.5)
    expected_array = np.array([[1, 2, 3], [7, 8, 9]])
    assert np.allclose(resampled_trajectory.array, expected_array)
    assert resampled_trajectory.freq_hz == 0.5


def test_upsample():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(steps=array, freq_hz=1)
    upsampled_trajectory = trajectory.resample(target_hz=2)
    expected_array = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.5, 3.5, 4.5],
            [4.0, 5.0, 6.0],
            [5.5, 6.5, 7.5],
            [7.0, 8.0, 9.0],
            [8.5, 9.5, 10.5],
            [10.0, 11.0, 12.0],
        ]
    )
    assert np.allclose(upsampled_trajectory.array, expected_array)


def test_minmax():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(steps=array, freq_hz=1)
    minmax_trajectory = trajectory.make_minmax(0, 255)
    expected_array = np.array([[0.0, 0.0, 0.0], [85.0, 85.0, 85.0], [170.0, 170.0, 170.0], [255.0, 255.0, 255.0]])
    assert np.allclose(minmax_trajectory.array, expected_array)


def test_normalize():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(array, freq_hz=1)
    normalized_trajectory = trajectory.make_standard()
    expected_array = np.array(
        [
            [-1.3416407864998738, -1.3416407864998736, -1.3416407864998736],
            [-0.44721359549995787, -0.44721359549995776, -0.44721359549995765],
            [0.44721359549995776, 0.44721359549995776, 0.44721359549995776],
            [1.3416407864998736, 1.3416407864998736, 1.3416407864998736],
        ]
    )
    assert np.allclose(normalized_trajectory.array, expected_array)


def test_unnormalize():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(steps=array, freq_hz=1)
    print(trajectory.stats)
    mean, std = trajectory.mean(), trajectory.std()
    normalized_trajectory = trajectory.make_standard()
    unnormalized_trajectory = normalized_trajectory.make_unstandard(mean, std).array
    expected_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert np.allclose(unnormalized_trajectory, expected_array)


def test_unminmax():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(array, freq_hz=1)
    min, max = trajectory.min(), trajectory.max()
    minmax_trajectory = trajectory.make_minmax(0, 255)
    norm_min, norm_max = minmax_trajectory.min(), minmax_trajectory.max()
    unminmax_trajectory = minmax_trajectory.make_unminmax(min, max).array
    expected_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert np.allclose(unminmax_trajectory, expected_array)


def test_make_pca():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    trajectory = Trajectory(array, freq_hz=1)
    pca_trajectory = trajectory.make_pca()
    expected_array = np.array(
        [
            [-1.16189500e00, -1.28452326e00, 2.40969395e-16],
            [-3.87298335e-01, 3.50324525e-01, -1.10782342e00],
            [3.87298335e-01, -3.50324525e-01, 1.10782342e00],
            [1.16189500e00, -1.05097357e00, -7.38548946e-01],
        ]
    )
    assert np.allclose(pca_trajectory.array, expected_array)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
