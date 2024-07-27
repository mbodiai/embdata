from collections import deque
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np


class LazyCall:
    """A class that allows queuing and applying function calls with their respective arguments."""

    def __init__(self):
        """Initializes a new instance of the LazyCall class."""
        self.function_calls = deque()
        self.kwargs = deque()

    def add_call(self, function, instance, *args, **kwargs) -> None:
        """Adds a function call to the queue with the specified arguments.

        Parameters:
        function (callable): The function to be called.
        instance (object): The instance the function is called on.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
        """
        self.function_calls.append((function, instance, args, kwargs))

    def apply(self) -> None:
        """Applies the queued function calls with their respective arguments."""
        while self.function_calls:
            function, instance, args, kwargs = self.function_calls.popleft()
            function(instance, *args, **kwargs)

    def __call__(self, function):
        """Decorator to add a function call to the queue with the specified arguments.

        Parameters:
        function (callable): The function to be called.

        Returns:
        callable: The wrapped function.
        """
        @wraps(function)
        def wrapper(instance, *args, **kwargs):
            instance.lazy_call.add_call(function, instance, *args, **kwargs)
        return wrapper

class TestTrajectory:
    def __init__(self, points):
        self.points = points
        self.lazy_call = LazyCall()

    @LazyCall()
    def transform(self, matrix) -> None:
        """Applies a transformation matrix to the trajectory points."""
        self.points = [matrix @ point for point in self.points]

    @LazyCall()
    def translate(self, vector) -> None:
        """Translates the trajectory points by a given vector."""
        self.points = [point + vector for point in self.points]

    @LazyCall()
    def scale(self, factor) -> None:
        """Scales the trajectory points by a given factor."""
        self.points = [point * factor for point in self.points]

    def plot(self) -> None:
        """Plots the trajectory points."""
        self.lazy_call.apply()  # Ensure all transformations are applied before plotting
        x, y = zip(*self.points, strict=False)
        plt.plot(x, y, marker="o")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory")
        plt.show()

if __name__ == "__main__":
  # Example usage
  points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])]
  trajectory = TestTrajectory(points)

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
