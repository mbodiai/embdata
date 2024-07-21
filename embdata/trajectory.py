import inspect
from functools import partial
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.stats as sstats
from pydantic import Field
from pydantic.dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.spatial.transform import RotationSpline
from sklearn import decomposition

from embdata.ndarray import NumpyArray
from embdata.sample import Sample


@dataclass
class Stats:
    mean: Any | None = None
    variance: Any | None = None
    skewness: Any | None = None
    kurtosis: Any | None = None
    min: Any | None = None
    max: Any | None = None
    lower_quartile: Any | None = None
    median: Any | None = None
    upper_quartile: Any | None = None
    non_zero_count: Any | None = None
    zero_count: Any | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __repr__(self) -> str:
        sep = "\n  "
        if isinstance(self.mean, float | int):
            return f"Stats(\n  {sep.join([f'{key}={round(v, 3)}' for key,v in self.__dict__.items()])}\n)"
        return f"Stats(\n  {sep.join([f'{key}={[round(x, 3) for x in value]}' for key, value in self.__dict__.items()])}\n)"

    def __str__(self) -> str:
        return self.__repr__()


def stats(array: np.ndarray, axis=0, bias=True, sample_type: type[Sample] | None = None) -> dict:
    """Compute statistics for an array along a given axis. Includes mean, variance, skewness, kurtosis, min, and max.

    Args:
      array (np.ndarray): The array to compute statistics for.
      axis (int, optional): The axis to compute statistics along. Defaults to 0.
      bias (bool, optional): Whether to use a biased estimator for the variance. Defaults to False.
      sample_type (type[Sample], optional): The type corresponding to a row in the array. Defaults to None.

    """
    stats_result = sstats.describe(array, axis=axis, bias=bias)
    mean = stats_result.mean
    variance = stats_result.variance
    skewness = stats_result.skewness
    kurtosis = stats_result.kurtosis
    min_val = np.min(array, axis=axis)
    max_val = np.max(array, axis=axis)
    lower_quartile = np.percentile(array, 25, axis=axis)
    median = np.percentile(array, 50, axis=axis)
    upper_quartile = np.percentile(array, 75, axis=axis)
    non_zero_count = np.count_nonzero(array, axis=axis)
    length = array.shape[axis]
    zero_count = length - non_zero_count

    if sample_type is not None:
        mean = sample_type(mean)
        variance = sample_type(variance)
        skewness = sample_type(skewness)
        kurtosis = sample_type(kurtosis)
        min_val = sample_type(min_val)
        max_val = sample_type(max_val)
        lower_quartile = sample_type(lower_quartile)
        median = sample_type(median)
        upper_quartile = sample_type(upper_quartile)
        non_zero_count = sample_type(non_zero_count)
        zero_count = sample_type(zero_count)

    return Stats(
        mean=mean,
        variance=variance,
        skewness=skewness,
        kurtosis=kurtosis,
        min=min_val,
        max=max_val,
        lower_quartile=lower_quartile,
        median=median,
        upper_quartile=upper_quartile,
        non_zero_count=non_zero_count,
        zero_count=zero_count,
    )


def plot_trajectory(trajectory: np.ndarray, labels: list[str] | None = None, time_step: float = 0.1, show=True) -> None:
    """Plot the trajectory.

    Args:
      trajectory (np.ndarray): The trajectory array.
      labels (list[str], optional): The labels for each dimension of the trajectory. Defaults to None.
      time_step (float, optional): The time step between each step in the trajectory. Defaults to 0.1.
      show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
      None
    """
    num_steps = trajectory.shape[0]
    num_plots = trajectory.shape[1]
    if labels is None and num_plots == 6:
        labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    elif labels is None and num_plots == 7:
        labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Grasp"]
    elif labels is None:
        labels = [f"Dimension {i}" for i in range(num_plots)]
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15), sharex=True)

    for i in range(num_plots):
        axs[i].plot(np.arange(num_steps) * time_step, trajectory[:, i])
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel("Time (s)")
    if show:
        plt.show()
    return fig


@dataclass
class Trajectory:
    """A trajectory of steps representing a time series of multidimensional data.

    This class provides methods for analyzing, visualizing, and manipulating trajectory data,
    such as robot movements, sensor readings, or any other time-series data.

    Attributes:
        steps (NumpyArray | List[Sample | NumpyArray]): The trajectory data.
        freq_hz (float | None): The frequency of the trajectory in Hz.
        time_idxs (NumpyArray | None): The time index of each step in the trajectory.
        dim_labels (list[str] | None): The labels for each dimension of the trajectory.
        angular_dims (list[int] | list[str] | None): The dimensions that are angular.

    Methods:
        plot: Plot the trajectory.
        map: Apply a function to each step in the trajectory.
        make_relative: Convert the trajectory to relative actions.
        resample: Resample the trajectory to a new sample rate.
        frequencies: Plot the frequency spectrogram of the trajectory.
        frequencies_nd: Plot the n-dimensional frequency spectrogram of the trajectory.
        low_pass_filter: Apply a low-pass filter to the trajectory.
        stats: Compute statistics for the trajectory.
        transform: Apply a transformation to the trajectory.

    Example:
        >>> import numpy as np
        >>> from embdata.trajectory import Trajectory
        >>> # Create a simple 2D trajectory
        >>> steps = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
        >>> traj = Trajectory(steps, freq_hz=10, dim_labels=["X", "Y"])
        >>> # Plot the trajectory
        >>> traj.plot().show()
        >>> # Compute and print statistics
        >>> print(traj.stats())
        >>> # Apply a low-pass filter
        >>> filtered_traj = traj.low_pass_filter(cutoff_freq=2)
        >>> filtered_traj.plot().show()
    """

    steps: NumpyArray | List[Sample | NumpyArray]
    freq_hz: float | None = Field(default=None, description="The frequency of the trajectory in Hz")
    time_idxs: NumpyArray | None = Field(default=None, description="The time index of each step in the trajectory")
    dim_labels: list[str] | None = Field(default=None, description="The labels for each dimension of the trajectory")
    angular_dims: list[int] | list[str] | None = Field(default=None, description="The dimensions that are angular")

    _fig: Any | None = None
    _stats: Stats | None = None
    _sample_class: type[Sample] = None
    _array: NumpyArray | None = None
    _map_history: list[Callable] = Field(default_factory=list)
    _map_history_kwargs: list[dict] = Field(default_factory=list)
    _episode: Any | None = None

    def __repr__(self) -> str:
        return f"Trajectory({self.stats()})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: "Trajectory") -> bool:
        return np.allclose(self.array, other.array)

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            if isinstance(self.steps[0], Sample):
                self._array = np.array([step.numpy() for step in self.steps])
            else:
                self._array = np.array(self.steps)
        return self._array

    def stats(self) -> Stats:
        """Compute statistics for the trajectory.

        Returns:
          dict: A dictionary containing the computed statistics, including mean, variance, skewness, kurtosis, min, and max.
        """
        if self._stats is None:
            self._stats = stats(self.array, axis=0, sample_type=self._sample_class)
        return self._stats

    def plot(self, labels: list[str] = None) -> "Trajectory":
        """Plot the trajectory. Saves the figure to the trajectory object. Call show() to display the figure.

        Args:
          labels (list[str], optional): The labels for each dimension of the trajectory. Defaults to None.
          time_step (float, optional): The time step between each step in the trajectory. Defaults to 0.1.

        Returns:
          Trajectory: The original trajectory.

        """
        self._fig = plot_trajectory(self.array, labels, time_step=1 / self.freq_hz, show=False)
        return self

    def map(self, fn) -> "Trajectory":
        """Apply a function to each step in the trajectory.

        Args:
          fn: The function to apply to each step.

        Returns:
          Trajectory: The modified trajectory.
        """
        return Trajectory(
            [fn(step) for step in self.steps],
            self.freq_hz,
            self.time_idxs,
            self.dim_labels,
            self.angular_dims,
            _episode=self._episode)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __iter__(self):
        return iter(self.steps)

    def __post_init__(self, *args, **kwargs):
        if self.dim_labels is None:
            self.dim_labels = [f"Dimension {i}" for i in range(self.array.shape[1])]
        if self.time_idxs is None:
            self.time_idxs = np.arange(0, len(self.array)) / self.freq_hz

        super().__init__(*args, **kwargs)


    def make_relative(self) -> "Trajectory":
        """Convert trajectory of absolute actions to relative actions.

        Returns:
          Trajectory: The converted relative trajectory.
        """
        return Trajectory(
            np.diff(self.array, axis=0),
            self.freq_hz,
            self.time_idxs[1:],
            self.dim_labels,
            self.angular_dims,
        )

    def make_absolute(self, initial_state: None | np.ndarray = None) -> "Trajectory":
        """Convert trajectory of relative actions to absolute actions.

        Args:
          initial_state (np.ndarray): The initial state of the trajectory. Defaults to zeros.

        Returns:
          Trajectory: The converted absolute trajectory.
        """
        if initial_state is None:
            initial_state = np.zeros(self.array.shape[1])
        self._map_history.append(partial(self.make_relative))
        self._map_history_kwargs.append({})
        return Trajectory(
            np.cumsum(self.array, axis=0) + initial_state,
            self.freq_hz,
            self.time_idxs,
            self.dim_labels,
            self.angular_dims,
        )

    def episode(self) -> Any:
        if self._episode is None:
            msg = "This trajectory is not part of an episode"
            raise ValueError(msg)
        return self._episode

    def resample(self, target_hz: float) -> "Trajectory":
        if self.freq_hz is None:
            msg = "Cannot resample a trajectory without a frequency"
            raise ValueError(msg)
        if self.freq_hz == target_hz:
            return self
        if self.array.shape[0] == 0:
            msg = "Cannot resample an empty trajectory"
            raise ValueError(msg)

        # Calculate total duration
        total_duration = (len(self.array) - 1) / self.freq_hz

        # Calculate the number of samples in the resampled trajectory, ensuring to include the last sample
        num_samples = np.ceil(total_duration * target_hz) + 1

        # Generate resampled_time_idxs including the end of the duration
        resampled_time_idxs = np.linspace(0, total_duration, int(num_samples))

        if target_hz < self.freq_hz:
            print("Downsampling...")  # noqa
            # For downsampling, just take every nth sample.
            downsampling_factor = int(self.freq_hz / target_hz)
            resampled_array = self.array[::downsampling_factor, :]
        else:
            if len(self.array) < 4:
                msg = "Cannot upsample a trajectory with bicubic interpolationwith less than 4 samples"
                raise ValueError(msg)
            print("Upsampling using bicubic interpolation and rotation splines...")  # noqa
            # Upsampling requires interpolation.
            num_dims = self.array.shape[1]
            resampled_array = np.zeros((len(resampled_time_idxs), num_dims))

            for i in range(num_dims):
                spline = interp1d(
                    np.arange(0, len(self.array)) / self.freq_hz,
                    self.array[:, i],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                resampled_array[:, i] = spline(resampled_time_idxs)

            if self.angular_dims:
                angular_dims = (
                    [self.dim_labels.index(dim) for dim in self.angular_dims]
                    if isinstance(self.angular_dims[0], str)
                    else self.angular_dims
                )
                for i in angular_dims:
                    spline = RotationSpline(np.arange(0, len(self.array)) / self.freq_hz, self.array[:, i])
                    resampled_array[:, i] = spline(resampled_time_idxs)

        return Trajectory(resampled_array, target_hz, resampled_time_idxs, self.dim_labels, self.angular_dims)

    def save(self, filename: str = "trajectory.png") -> None:
        """Save the current figure to a file.

        Args:
          filename (str, optional): The filename to save the figure. Defaults to "trajectory.png".

        Returns:
          None
        """
        self._fig.savefig(filename)
        return self

    def show(self) -> "Trajectory":
        """Display the current figure.

        Returns:
          None
        """
        plt.show()

    def frequencies(self) -> "Trajectory":
        """Plot the frequency spectrogram of the trajectory.

        Returns:
          Trajectory: The modified trajectory.
        """
        x = self.array
        nrows, ncols = 3, 3  # Adjust based on the number of dimensions
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15), sharex=True, sharey=True)
        axes = axes.flatten()

        N = x.shape[0]
        T = 1.0 / self.freq_hz
        freqs = np.fft.fftfreq(N, T)[: N // 2]

        dim_labels = self.dim_labels or [f"Dimension {i}" for i in range(x.shape[1])]

        for i in range(x.shape[1]):
            ax = axes[i] if i < len(axes) else axes[-1]
            fft_vals = np.fft.fft(x[:, i])
            magnitude = 2.0 / N * np.abs(fft_vals[0 : N // 2])
            ax.plot(freqs, magnitude)
            ax.set_title(dim_labels[i])
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Magnitude")
            ax.set_xlim(0, self.freq_hz / 2)  # Nyquist frequency
            ax.grid(True)

        for ax in axes[x.shape[1] :]:
            ax.remove()  # Remove unused subplots

        plt.tight_layout()
        self._fig = fig
        return self

    def frequencies_nd(self) -> "Trajectory":
        """Plot the n-dimensional frequency spectrogram of the trajectory.

        Returns:
          Trajectory: The modified trajectory.
        """
        n_dims = self.array.shape[1]
        N = len(self.array)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        axs = axs.flatten()

        t = np.arange(N) / self.freq_hz
        freqs = fftpack.fftfreq(N, d=1 / self.freq_hz)
        Sxx = np.abs(fftpack.fft2(self.array, axes=(0, 1)))
        Sxx = np.fft.fftshift(Sxx, axes=1)
        Sxx = np.log10(Sxx + 1e-10)  # Add small constant to avoid log(0)

        dim_labels = self.dim_labels[:n_dims] if self.dim_labels else ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

        for i, ax in enumerate(axs):
            im = ax.pcolormesh(freqs, t, Sxx[:, :, i], shading="gouraud", cmap="viridis")
            ax.set_title(dim_labels[i])
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Time [s]")
            ax.set_ylim(t[0], t[-1])
            ax.set_xlim(-self.freq_hz / 2, self.freq_hz / 2)
            fig.colorbar(im, ax=ax, label="Log Magnitude")

        plt.tight_layout()
        self._fig = fig
        return self

    def low_pass_filter(self, cutoff_freq: float) -> "Trajectory":
        """Apply a low-pass filter to the trajectory.

        Args:
          cutoff_freq (float): The cutoff frequency for the low-pass filter.

        Returns:
          Trajectory: The filtered trajectory.
        """
        fft = fftpack.fft(self.array, axis=0)
        frequencies = fftpack.fftfreq(len(fft), d=1.0 / self.freq_hz)
        fft[np.abs(frequencies) > cutoff_freq] = 0
        filtered_trajectory = fftpack.ifft(fft, axis=0)

        return Trajectory(filtered_trajectory, self.freq_hz, self.time_idxs)

    # def spectrogram(self) -> "Trajectory":
    #     """Plot the spectrogram of the trajectory.

    #     Returns:
    #       Trajectory: The modified trajectory.
    #     """
    #     x = self.array
    #     f, t, Sxx = spectrogram(x.T, fs=self.freq_hz)
    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis")
    #     ax.set_ylabel("Frequency [Hz]")
    #     ax.set_xlabel("Time [s]")
    #     fig.colorbar(cax, label="Power [dB]")
    #     self._fig = fig
    #     return self

    def q01(self) -> float:
        return np.percentile(self.array, 1, axis=0)

    def q99(self) -> float:
        return np.percentile(self.array, 99, axis=0)

    def mean(self) -> np.ndarray | Sample:
        return np.mean(self.array, axis=0)

    def variance(self) -> np.ndarray | Sample:
        return np.var(self.array, axis=0)

    def std(self) -> float:
        return np.std(self.array, axis=0)

    def skewness(self) -> float:
        return self.stats().skewness

    def kurtosis(self) -> float:
        return self.stats().kurtosis

    def min(self) -> float:
        return self.stats().min

    def max(self) -> float:
        return self.stats().max

    def lower_quartile(self) -> float:
        return self.stats().lower_quartile

    def median(self) -> float:
        return self.stats().median

    def upper_quartile(self) -> float:
        return self.stats().upper_quartile

    def non_zero_count(self) -> float:
        return self.stats().non_zero_count

    def zero_count(self) -> float:
        return self.stats().zero_count

    def transform(self, operation: Callable[[np.ndarray], np.ndarray] | str, **kwargs) -> "Trajectory":
        """Apply a transformation to the trajectory.

        Available operations are:
        - [un]minmax: Apply min-max normalization to the trajectory.
        - [un]standard: Apply standard normalization to the trajectory.
        - pca: Apply PCA normalization to the trajectory.
        - absolute: Convert the trajectory to absolute actions.
        - relative: Convert the trajectory to relative actions.


        Args:
          operation (Callable | str): The operation to apply. Can be a callable or a string corresponding to a `make_` method on the Trajectory class.
          **kwargs: Additional keyword arguments to pass to the operation.
        """
        if isinstance(operation, str):
            try:
                operation = getattr(self, "make_" + operation)
            except AttributeError as e:
                raise ValueError(f""""peration {operation} not found in Trajectory. Available operations are {[
                    inspect.getmembers(self, predicate=lambda x: inspect.ismethod(x) and x.__name__.startswith("make_"))
                ]}. See the corresponding methods starting with `make_` for kwargs.""") from e
        elif not isinstance(operation, Callable):
            raise ValueError("operation must be a callable or a string")

        try:
            return operation(**kwargs)
        except TypeError as e:
            raise ValueError(
                f"Operation {operation} failed with kwargs {kwargs}. Signature is {inspect.signature(operation)}"
            ) from e

    def make_minmax(self, min: float = 0, max: float = 1) -> "Trajectory":
        """Apply min-max normalization to the trajectory.

        Args:
          min (float, optional): The minimum value for the normalization. Defaults to 0.
          max (float, optional): The maximum value for the normalization. Defaults to 1.

        Returns:
          Trajectory: The normalized trajectory.
        """
        min_vals = np.min(self.array, axis=0)
        max_vals = np.max(self.array, axis=0)
        return Trajectory(
            (self.array - min_vals) / (max_vals - min_vals) * (max - min) + min,
            self.freq_hz,
            self.time_idxs,
            self.dim_labels,
            self.angular_dims,
        )

    def make_pca(self, whiten=True) -> "Trajectory":
        """Apply PCA normalization to the trajectory.

        Returns:
          Trajectory: The PCA-normalized trajectory.
        """
        pca = decomposition.PCA(n_components=self.array.shape[1], whiten=whiten)
        return Trajectory(
            pca.fit_transform(self.array),
            self.freq_hz,
            self.time_idxs,
            self.dim_labels,
            self.angular_dims,
        )

    def make_standard(self) -> "Trajectory":
        """Apply standard normalization to the trajectory.

        Returns:
          Trajectory: The standardized trajectory.
        """
        mean = np.mean(self.array, axis=0)
        std = np.std(self.array, axis=0)
        return Trajectory((self.array - mean) / std, self.freq_hz, self.time_idxs, self.dim_labels, self.angular_dims)

    def make_unminmax(
        self,
        orig_min: np.ndarray | Sample,
        orig_max: np.ndarray | Sample,
    ) -> "Trajectory":
        """Reverse min-max normalization on the trajectory."""
        norm_min = np.min(self.array, axis=0)
        norm_max = np.max(self.array, axis=0)
        array = (self.array - norm_min) / (norm_max - norm_min) * (orig_max - orig_min) + orig_min
        steps = [self._sample_class(step) for step in array] if self._sample_class is not None else array
        return Trajectory(steps, self.freq_hz, self.time_idxs, self.dim_labels, self.angular_dims)

    def make_unstandard(self, mean: np.ndarray, std: np.ndarray) -> "Trajectory":
        array = (self.array * std) + mean
        steps = [self._sample_class(step) for step in array] if self._sample_class is not None else array
        return Trajectory(steps, self.freq_hz, self.time_idxs, self.dim_labels, self.angular_dims)


def main() -> None:
    from datasets import Dataset, load_dataset

    from embdata.motion.control import HandControl

    ds = Dataset.from_list(
        list(
            load_dataset("mbodiai/oxe_bridge", "default", split="default", streaming=True)
            .take(100)
            .filter(lambda x: x["episode_idx"] == 1),
        ),
    )

    ds = np.array([HandControl(**a["action"]).numpy() for a in ds])
    trajectory = Trajectory(
        ds,
        freq_hz=5,
        dim_labels=[
            "X",
            "Y",
            "Z",
            "Roll",
            "Pitch",
            "Yaw",
            "Grasp",
        ],
        angular_dims=["Roll", "Pitch", "Yaw"],
    )
    trajectory.spectrogram().save("spectrogram.png")
    trajectory.plot().save("trajectory.png")
    trajectory.frequencies_nd().save("nd_spectrogram.png")
    trajectory.resample(5).spectrogram().save("resampled_trajectory.png")
    trajectory.resample(5).plot().save("resampled_trajectory_plot.png")


if __name__ == "__main__":
    main()
