from embdata.utils.lazy import LazyCall
import numpy as np
import plotext as plt
import pytest

@pytest.fixture
def traj_class():
    class TestTrajectory:
        def __init__(self, points):
            self.points = points
            self.lazy_call = LazyCall()

        @LazyCall()
        def transform(self, matrix):
            """Applies a transformation matrix to the trajectory points."""
            self.points = [matrix @ point for point in self.points]

        @LazyCall()
        def translate(self, vector):
            """Translates the trajectory points by a given vector."""
            self.points = [point + vector for point in self.points]

        @LazyCall()
        def scale(self, factor):
            """Scales the trajectory points by a given factor."""
            self.points = [point * factor for point in self.points]

        def plot(self):
            """Plots the trajectory points."""
            self.lazy_call.apply()  # Ensure all transformations are applied before plotting
            x, y = zip(*self.points)
            plt.plot(x, y, marker="o")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Trajectory")
            plt.show()

    return TestTrajectory


def test_lazy_traj(traj_class):
    points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])]
    trajectory = traj_class(points)

    # Define a transformation matrix (e.g., rotation)
    rotation_matrix = np.array([[0, -1], [1, 0]])
    translation_vector = np.array([1, 1])
    scale_factor = 2

    # Queue the transformations
    trajectory.transform(matrix=rotation_matrix)
    trajectory.translate(vector=translation_vector)
    trajectory.scale(factor=scale_factor)

    # Plot the trajectory (this will apply the transformations first)
    trajectory.plot()


if __name__ == "__main__":
    test_lazy_traj()
