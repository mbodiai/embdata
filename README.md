# embodied data

## Data, types, pipes, manipulation for embodied learning.

[![PyPI - Version](https://img.shields.io/pypi/v/embdata.svg)](https://pypi.org/project/embdata)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/embdata.svg)](https://pypi.org/project/embdata)

-----



### A good chunk of data wrangling and exploratory data analysis that just works. See [embodied-agents](https://github.com/mbodiai/embodied-agents) for real world usage.


## Plot, filter and transform your data with ease. On any type of data structure.


[![Video Title](https://img.youtube.com/vi/L5JqM2_rIRM/0.jpg)](https://www.youtube.com/watch?v=L5JqM2_rIRM)

## Table of Contents

- [embodied data](#embodied-data)
  - [Data, types, pipes, manipulation for embodied learning.](#data-types-pipes-manipulation-for-embodied-learning)
    - [A good chunk of data wrangling and exploratory data analysis that just works. See embodied-agents for real world usage.](#a-good-chunk-of-data-wrangling-and-exploratory-data-analysis-that-just-works-see-embodied-agents-for-real-world-usage)
  - [Plot, filter and transform your data with ease. On any type of data structure.](#plot-filter-and-transform-your-data-with-ease-on-any-type-of-data-structure)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Classes](#classes)
    - [Coordinate](#coordinate)
      - [Pose](#pose)
    - [Episode](#episode)
    - [Exploratory data analysis for common minmax and standardization normalization methods.](#exploratory-data-analysis-for-common-minmax-and-standardization-normalization-methods)
    - [Upsample and downsample the trajectory to a target frequency.](#upsample-and-downsample-the-trajectory-to-a-target-frequency)
    - [Make actions relative or absolute](#make-actions-relative-or-absolute)
  - [Applications](#applications)
    - [What are the grasping positions in the world frame?](#what-are-the-grasping-positions-in-the-world-frame)
  - [License](#license)
  - [Design Decisions](#design-decisions)
  - [API Reference](#api-reference)

## Installation

```console
pip install embdata
```

## Classes


### Coordinate

Classes for representing geometric data in cartesian and polar coordinates.

#### Pose 
-  (x, y, z, roll, pitch, and yaw) in some reference frame.
- Contains `.numpy()`, `.dict()`, `.dataset()` methods from `Sample` class we all know and love.

Example:
```python
    >>> import math
    >>> pose_3d = Pose3D(x=1, y=2, theta=math.pi/2)
    >>> pose_3d.to("cm")
    Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

    >>> pose_3d.to("deg")
    Pose3D(x=1.0, y=2.0, theta=90.0)

    >>> class BoundedPose6D(Pose6D):
    ...     x: float = CoordinateField(bounds=(0, 5))

    >>> pose_6d = BoundedPose6D(x=10, y=2, z=3, roll=0, pitch=0, yaw=0)
    Traceback (most recent call last):
    ...
    ValueError: x value 10 is not within bounds (0, 5)
```


### Episode

The `Episode` class provides a list-like interface for a sequence of observations, actions, and/or other data points. It is designed to streamline exploratory data analysis and manipulation of time series data. Episodes can be easily concatenated, iterated over, and manipulated similar to lists.

```python
class Episode(Sample):
    """A list-like interface for a sequence of observations, actions, and/or other.

    Meant to streamline exploratory data analysis and manipulation of time series data.

    Just append to an episode like you would a list and you're ready to start training models.

    To iterate over the steps in an episode, use the `iter` method.
```

Example:
```python
    >>> episode = Episode(steps=[TimeStep(), TimeStep(), TimeStep()])
    >>> for step in episode.iter():
    ...     print(step)
```
To concatenate two episodes, use the `+` operator.

Example:
```python
    >>> episode1 = Episode(steps=[TimeStep(), TimeStep()])
    >>> episode2 = Episode(steps=[TimeStep(), TimeStep()])
    >>> combined_episode = episode1 + episode2
    >>> len(combined_episode)
    4 
```

```python
 def __init__(
        self, 
        steps: List[Dict | Sample] | Iterable, 
        observation_key: str = "observation", 
        action_key: str = "action",
        state_key: str|None=None, supervision_key: str | None = None, 
        metadata: Sample | Any | None = None,
    ) -> None:
 """Create an episode from a list of dicts, samples, time steps or any iterable with at least two items per step.

        Args:
            steps (List[Dict|Sample]): The list of dictionaries representing the steps.
            action_key (str): The key for the action in each dictionary.
            observation_key (str): The key for the observation in each dictionary.
            state_key (str, optional): The key for the state in each dictionary. Defaults to None.
            supervision_key (str, optional): The key for the supervision in each dictionary. Defaults to None.

        Returns:
            'Episode': The created episode.
```
```python
        Example:
            >>> steps = [
            ...     {"observation": Image((224, 224)), "action": 1, "supervision": 0},
            ...     {"observation": Image((224, 224)), "action": 1, "supervision": 1},
            ...     {"observation": Image((224, 224)), "action": 100, "supervision": 0},
            ...     {"observation": Image((224, 224)), "action": 300, "supervision": 1},
            ... ]
            >>> episode = Episode(steps, "observation", "action", "supervision")
            >>> episode
            Episode(
              Stats(
                mean=[100.5]
                variance=[19867.0]
                skewness=[0.821]
                kurtosis=[-0.996]
                min=[1]
                max=[300]
                lower_quartile=[1.0]
                median=[50.5]
                upper_quartile=[150.0]
                non_zero_count=[4]
                zero_count=[0])
            )
        """
```
### Exploratory data analysis for common minmax and standardization normalization methods.

Example
```python
    >>> steps = [
    ...     {"observation": Image((224,224)), "action": 1, "supervision": 0},
    ...     {"observation": Image((224,224)), "action": 1, "supervision": 1},
    ...     {"observation": Image((224,224)), "action": 100, "supervision": 0},
    ...     {"observation": Image((224,224)), "action": 300, "supervision": 1},
    ]
    >>> episode = Episode.from_list(steps, "observation", "action", "supervision")
    >>> episode.trajectory().transform("minmax")
    Episode(
        Stats(
            mean=[0.335]
            variance=[0.198]
            skewness=[0.821]
            kurtosis=[-0.996]
            min=[0.0]
            max=[1.0]
            lower_quartile=[0.0]
            median=[0.168]
            upper_quartile=[0.503]
            non_zero_count=[4]
            zero_count=[0])
    )
    >>> episode.trajectory().transform("unminmax", orig_min=1, orig_max=300)
    Episode(
        Stats(
            mean=[100.5]
            variance=[19867.0]
            skewness=[0.821]
            kurtosis=[-0.996]
            min=[1]
            max=[300]
            lower_quartile=[1.0]
            median=[50.5]
            upper_quartile=[150.0]
            non_zero_count=[4]
            zero_count=[0])
    )
    >>> episode.trajectory().frequencies().show()  # .save("path/to/save.png") also works.
    >>> episode.trajectory().plot().show()
```
### Upsample and downsample the trajectory to a target frequency.

- Uses bicupic and rotation spline interpolation to upsample and downsample the trajectory to a target frequency.
```python
    >>> episode.trajectory().resample(target_hz=10).plot().show() # .save("path/to/save.png") also works.
```

### Make actions relative or absolute

- Make actions relative or absolute to the previous action.
```python
    >>> relative_actions = episode.trajectory("action").make_relative()
    >>>  absolute_again = episode.trajectory().make_absolute(initial_state=relative_actions[0])
    assert np.allclose(episode.trajectory("action"), absolute_again)
```

## Applications

### What are the grasping positions in the world frame?
    
```python
    >>> initial_state = episode.trajectory('state').flatten(to='end_effector_pose')[0]
    >>> episode.trajectory('action').make_absolute(initial_state=initial_state).filter(lambda x: x.grasp == 1)
```

## License

`embdata` is distributed under the terms of the [apache-2.0](https://spdx.org/licenses/apache-2.0.html) license.

## Design Decisions

- [x] Grasp value is [-1, 1] so that the default value is 0.
- [x] Motion rather than Action to distinguish from non-physical actions.

## API Reference

## API Reference

<details>
<summary>Episode</summary>

```python
class Episode(Sample):
    """A list-like interface for a sequence of observations, actions, and/or other data.

    This class is designed to streamline exploratory data analysis and manipulation of time series data.
    It provides methods for appending, iterating, concatenating, and analyzing episodes.

    Attributes:
        steps (list[TimeStep]): A list of TimeStep objects representing the episode's steps.
        metadata (Sample | Any | None): Additional metadata for the episode.
        freq_hz (int | None): The frequency of the episode in Hz.

    Example:
        >>> from embdata.image import Image
        >>> from embdata.motion import Motion
        >>> steps = [
        ...     VisionMotorStep(
        ...         observation=ImageTask(image=Image((224, 224, 3)), task="grasp"),
        ...         action=Motion(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
        ...     ),
        ...     VisionMotorStep(
        ...         observation=ImageTask(image=Image((224, 224, 3)), task="lift"),
        ...         action=Motion(position=[0.2, 0.3, 0.4], orientation=[0, 0, 1, 0])
        ...     )
        ... ]
        >>> episode = Episode(steps=steps)
        >>> len(episode)
        2
        >>> for step in episode.iter():
        ...     print(f"Task: {step.observation.task}, Action: {step.action.position}")
        Task: grasp, Action: [0.1, 0.2, 0.3]
        Task: lift, Action: [0.2, 0.3, 0.4]

    To concatenate two episodes, use the `+` operator:
        >>> episode1 = Episode(steps=steps[:1])
        >>> episode2 = Episode(steps=steps[1:])
        >>> combined_episode = episode1 + episode2
        >>> len(combined_episode)
        2
    """

    def trajectory(self, field: str = "action", freq_hz: int = 1) -> Trajectory:
        """Extract a trajectory from the episode for a specified field.

        This method creates a Trajectory object from the specified field of each step in the episode.
        The resulting Trajectory object allows for various operations such as frequency analysis,
        subsampling, super-sampling, and min-max scaling.

        Args:
            field (str, optional): The field to extract from each step. Defaults to "action".
            freq_hz (int, optional): The frequency in Hz of the trajectory. Defaults to 1.

        Returns:
            Trajectory: The trajectory of the specified field.

        Example:
            >>> from embdata.image import Image
            >>> from embdata.motion import Motion
            >>> episode = Episode(
            ...     steps=[
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="grasp"),
            ...             action=Motion(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
            ...         ),
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="move"),
            ...             action=Motion(position=[0.2, 0.3, 0.4], orientation=[0, 0, 1, 0])
            ...         ),
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="release"),
            ...             action=Motion(position=[0.3, 0.4, 0.5], orientation=[1, 0, 0, 0])
            ...         ),
            ...     ]
            ... )
            >>> action_trajectory = episode.trajectory(field="action", freq_hz=10)
            >>> action_trajectory.mean()
            array([0.2, 0.3, 0.4, 0.33333333, 0., 0.33333333, 0.33333333])
            >>> observation_trajectory = episode.trajectory(field="observation")
            >>> [step.task for step in observation_trajectory]
            ['grasp', 'move', 'release']
        """
```

</details>
