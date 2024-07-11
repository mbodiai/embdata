# embdata

## Data, types, pipes, manipulation for embodied learning.

[![PyPI - Version](https://img.shields.io/pypi/v/embdata.svg)](https://pypi.org/project/embdata)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/embdata.svg)](https://pypi.org/project/embdata)

-----



### A good chunk of data wrangling and exploratory data analysis that just works. See [embodied-agents](https://github.com/mbodiai/embodied-agents) for full examples.

[![Video Title](https://img.youtube.com/vi/L5JqM2_rIRM/0.jpg)](https://www.youtube.com/watch?v=L5JqM2_rIRM)

## Table of Contents

- [embdata](#embdata)
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
  - [Classes](#classes-1)
    - [Episode](#episode-1)
      - [concat](#concat)
      - [trajectory](#trajectory)
      - [map](#map)
      - [filter](#filter)
      - [unpack](#unpack)
      - [iter](#iter)
      - [append](#append)
      - [split](#split)
      - [dataset](#dataset)
      - [rerun](#rerun)
      - [show](#show)
    - [Trajectory](#trajectory-1)
      - [stats](#stats)
      - [plot](#plot)
      - [map](#map-1)
      - [make\_relative](#make_relative)
      - [make\_absolute](#make_absolute)
      - [resample](#resample)
      - [save](#save)
      - [show](#show-1)
      - [frequencies](#frequencies)
      - [frequencies\_nd](#frequencies_nd)
      - [low\_pass\_filter](#low_pass_filter)
      - [q01](#q01)
      - [q99](#q99)
      - [transform](#transform)
      - [make\_minmax](#make_minmax)
      - [make\_pca](#make_pca)
      - [make\_standard](#make_standard)
      - [make\_unminmax](#make_unminmax)
      - [make\_unstandard](#make_unstandard)

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

## Classes

# <span style='color: #C792EA;'>Episode</span>

```
A list-like interface for a sequence of observations, actions, and/or other.

Meant to streamline exploratory data analysis and manipulation of time series data.

Just append to an episode like you would a list and you're ready to start training models.

To iterate over the steps in an episode, use the `iter` method.

Example:
    >>> episode = Episode(steps=[TimeStep(), TimeStep(), TimeStep()])
    >>> for step in episode.iter():
    ...     print(step)

To concatenate two episodes, use the `+` operator.

Example:
    >>> episode1 = Episode(steps=[TimeStep(), TimeStep()])
    >>> episode2 = Episode(steps=[TimeStep(), TimeStep()])
    >>> combined_episode = episode1 + episode2
    >>> len(combined_episode)
    4
```



## <span style='color: #82AAFF;'>concat</span>

### <span style='color: #89DDFF;'>concat(<span style='color: #FFCB6B;'>episodes</span>: <span style='color: #C3E88D;'>List</span>[ForwardRef('Episode')]) <span style='color: #F07178;'>-> 'Episode'</span></span>

Concatenate a list of episodes into a single episode.

<details>
<summary><b>Full Documentation</b></summary>

```
Concatenate a list of episodes into a single episode.

This method combines multiple episodes into a single episode by concatenating their steps.

Args:
    episodes (List['Episode']): The list of episodes to concatenate.

Returns:
    'Episode': The concatenated episode.

Example:
    >>> episode1 = Episode(steps=[TimeStep(observation=1, action=1), TimeStep(observation=2, action=2)])
    >>> episode2 = Episode(steps=[TimeStep(observation=3, action=3), TimeStep(observation=4, action=4)])
    >>> combined_episode = Episode.concat([episode1, episode2])
    >>> len(combined_episode)
    4
    >>> [step.observation for step in combined_episode.iter()]
    [1, 2, 3, 4]
```
</details>

## <span style='color: #82AAFF;'>trajectory</span>

### <span style='color: #89DDFF;'>trajectory(self, <span style='color: #FFCB6B;'>field</span>: <span style='color: #C3E88D;'>str</span> = 'action', <span style='color: #FFCB6B;'>freq_hz</span>: <span style='color: #C3E88D;'>int</span> = 1) <span style='color: #F07178;'>-> embdata.trajectory.Trajectory</span></span>

Get a numpy array and perform frequency, subsampling, super-sampling, min-max scaling, and more.

<details>
<summary><b>Full Documentation</b></summary>

```
Get a numpy array and perform frequency, subsampling, super-sampling, min-max scaling, and more.

This method extracts the specified field from each step in the episode and creates a Trajectory object.

Args:
    field (str, optional): The field to extract from each step. Defaults to "action".
    freq_hz (int, optional): The frequency in Hz of the trajectory. Defaults to 1.

Returns:
    Trajectory: The trajectory of the specified field.

Example:
    >>> episode = Episode(steps=[
    ...     TimeStep(observation=Sample(value=1), action=Sample(value=10)),
    ...     TimeStep(observation=Sample(value=2), action=Sample(value=20)),
    ...     TimeStep(observation=Sample(value=3), action=Sample(value=30))
    ... ])
    >>> action_trajectory = episode.trajectory()
    >>> action_trajectory.mean()
    array([20.])
    >>> observation_trajectory = episode.trajectory(field="observation")
    >>> observation_trajectory.mean()
    array([2.])
```
</details>

## <span style='color: #82AAFF;'>map</span>

### <span style='color: #89DDFF;'>map(self, <span style='color: #FFCB6B;'>func</span>: <span style='color: #C3E88D;'>Callable</span>[[Union[embdata.episode.TimeStep, Dict]], embdata.episode.TimeStep]) <span style='color: #F07178;'>-> 'Episode'</span></span>

Apply a function to each step in the episode.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply a function to each step in the episode.

Args:
    func (Callable[[TimeStep], TimeStep]): The function to apply to each step.

Returns:
    'Episode': The modified episode.

Example:
    >>> episode = Episode(steps=[
    ...     TimeStep(observation=Sample(value=1), action=Sample(value=10)),
    ...     TimeStep(observation=Sample(value=2), action=Sample(value=20)),
    ...     TimeStep(observation=Sample(value=3), action=Sample(value=30))
    ... ])
    >>> episode.map(lambda step: TimeStep(observation=Sample(value=step.observation.value * 2), action=step.action))
    Episode(steps=[
      TimeStep(observation=Sample(value=2), action=Sample(value=10)),
      TimeStep(observation=Sample(value=4), action=Sample(value=20)),
      TimeStep(observation=Sample(value=6), action=Sample(value=30))
    ])
```
</details>

## <span style='color: #82AAFF;'>filter</span>

### <span style='color: #89DDFF;'>filter(self, <span style='color: #FFCB6B;'>condition</span>: <span style='color: #C3E88D;'>Callable</span>[[embdata.episode.TimeStep], bool]) <span style='color: #F07178;'>-> 'Episode'</span></span>

Filter the steps in the episode based on a condition.

<details>
<summary><b>Full Documentation</b></summary>

```
Filter the steps in the episode based on a condition.

Args:
    condition (Callable[[TimeStep], bool]): A function that takes a time step and returns a boolean.

Returns:
    'Episode': The filtered episode.

Example:
    >>> episode = Episode(steps=[
    ...     TimeStep(observation=Sample(value=1), action=Sample(value=10)),
    ...     TimeStep(observation=Sample(value=2), action=Sample(value=20)),
    ...     TimeStep(observation=Sample(value=3), action=Sample(value=30))
    ... ])
    >>> episode.filter(lambda step: step.observation.value > 1)
    Episode(steps=[
      TimeStep(observation=Sample(value=2), action=Sample(value=20)),
      TimeStep(observation=Sample(value=3), action=Sample(value=30))
    ])
```
</details>

## <span style='color: #82AAFF;'>unpack</span>

### <span style='color: #89DDFF;'>unpack(self) <span style='color: #F07178;'>-> tuple</span></span>

Unpack the episode into a tuple of lists with a consistent order.

<details>
<summary><b>Full Documentation</b></summary>

```
Unpack the episode into a tuple of lists with a consistent order.

Output order:
 observations, actions, states, supervision, ... other attributes.

Returns:
    tuple[list, list, list]: A tuple of lists containing the observations, actions, and states.

Example:
    >>> episode = Episode(steps=[
    ...     TimeStep(observation=1, action=10),
    ...     TimeStep(observation=2, action=20),
    ...     TimeStep(observation=3, action=30)
    ... ])
    >>> observations, actions, states = episode.unpack()
    >>> observations
    [Sample(value=1), Sample(value=2), Sample(value=3)]
    >>> actions
    [Sample(value=10), Sample(value=20), Sample(value=30)]
```
</details>

## <span style='color: #82AAFF;'>iter</span>

### <span style='color: #89DDFF;'>iter(self) <span style='color: #F07178;'>-> Iterator[embdata.episode.TimeStep]</span></span>

Iterate over the steps in the episode.

<details>
<summary><b>Full Documentation</b></summary>

```
Iterate over the steps in the episode.

Returns:
    Iterator[TimeStep]: An iterator over the steps in the episode.
```
</details>

## <span style='color: #82AAFF;'>append</span>

### <span style='color: #89DDFF;'>append(self, <span style='color: #FFCB6B;'>step</span>: <span style='color: #C3E88D;'>embdata</span>.episode.TimeStep) <span style='color: #F07178;'>-> None</span></span>

Append a time step to the episode.

<details>
<summary><b>Full Documentation</b></summary>

```
Append a time step to the episode.

Args:
    step (TimeStep): The time step to append.
```
</details>

## <span style='color: #82AAFF;'>split</span>

### <span style='color: #89DDFF;'>split(self, <span style='color: #FFCB6B;'>condition</span>: <span style='color: #C3E88D;'>Callable</span>[[embdata.episode.TimeStep], bool]) <span style='color: #F07178;'>-> list['Episode']</span></span>

Split the episode into multiple episodes based on a condition.

<details>
<summary><b>Full Documentation</b></summary>

```
Split the episode into multiple episodes based on a condition.

This method divides the episode into separate episodes based on whether each step
satisfies the given condition. The resulting episodes alternate between steps that
meet the condition and those that don't.

The episodes will be split alternatingly based on the condition:
- The first episode will contain steps where the condition is true,
- The second episode will contain steps where the condition is false,
- And so on.

If the condition is always or never met, one of the episodes will be empty.

Args:
    condition (Callable[[TimeStep], bool]): A function that takes a time step and returns a boolean.

Returns:
    list[Episode]: A list of at least two episodes.

Example:
    >>> episode = Episode(steps=[
    ...     TimeStep(observation=Sample(value=5)),
    ...     TimeStep(observation=Sample(value=10)),
    ...     TimeStep(observation=Sample(value=15)),
    ...     TimeStep(observation=Sample(value=8)),
    ...     TimeStep(observation=Sample(value=20))
    ... ])
    >>> episodes = episode.split(lambda step: step.observation.value <= 10)
    >>> len(episodes)
    3
    >>> [len(ep) for ep in episodes]
    [2, 1, 2]
    >>> [[step.observation.value for step in ep.iter()] for ep in episodes]
    [[5, 10], [15], [8, 20]]
```
</details>

## <span style='color: #82AAFF;'>dataset</span>

### <span style='color: #89DDFF;'>dataset(self) <span style='color: #F07178;'>-> datasets.arrow_dataset.Dataset</span></span>

Create a Hugging Face Dataset from the episode.

<details>


## <span style='color: #82AAFF;'>rerun</span>

### <span style='color: #89DDFF;'>rerun(self, local=True, port=5003, ws_port=5004) <span style='color: #F07178;'>-> 'Episode'</span></span>

Start a rerun server.

<details>
<summary><b>Full Documentation</b></summary>

```
Start a rerun server.
```
</details>

## <span style='color: #82AAFF;'>show</span>

### <span style='color: #89DDFF;'>show(self, local=True, port=5003) <span style='color: #F07178;'>-> None</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>


```
This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
    self: The BaseModel instance.
    __context: The context.
```
</details>




# <span style='color: #C792EA;'>Trajectory</span>

```
A trajectory of steps.

Methods:
- plot: Plot the trajectory.
- map: Apply a function to each step in the trajectory.
- make_relative: Convert the trajectory to relative actions.
- resample: Resample the trajectory to a new sample rate.
- spectrogram: Plot the spectrogram of the trajectory.
- spectrogram_nd: Plot the n-dimensional spectrogram of the trajectory.
- low_pass_filter: Apply a low-pass filter to the trajectory.
- stats: Compute statistics for the trajectory. Includes mean, variance, skewness, kurtosis, min, and max.
```
<details>
<summary><b>Examples</b></summary>

```python
# Examples would go here
```
</details>

## <span style='color: #82AAFF;'>stats</span>

### <span style='color: #89DDFF;'>stats(self) <span style='color: #F07178;'>-> embdata.trajectory.Stats</span></span>

Compute statistics for the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Compute statistics for the trajectory.

Returns:
  dict: A dictionary containing the computed statistics, including mean, variance, skewness, kurtosis, min, and max.
```
</details>

## <span style='color: #82AAFF;'>plot</span>

### <span style='color: #89DDFF;'>plot(self, <span style='color: #FFCB6B;'>labels</span>: <span style='color: #C3E88D;'>list</span>[str] = None) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Plot the trajectory. Saves the figure to the trajectory object. Call show() to display the figure.

<details>
<summary><b>Full Documentation</b></summary>

```
Plot the trajectory. Saves the figure to the trajectory object. Call show() to display the figure.

Args:
  labels (list[str], optional): The labels for each dimension of the trajectory. Defaults to None.
  time_step (float, optional): The time step between each step in the trajectory. Defaults to 0.1.

Returns:
  Trajectory: The original trajectory.
```
</details>

## <span style='color: #82AAFF;'>map</span>

### <span style='color: #89DDFF;'>map(self, fn) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Apply a function to each step in the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply a function to each step in the trajectory.

Args:
  fn: The function to apply to each step.

Returns:
  Trajectory: The modified trajectory.
```
</details>

## <span style='color: #82AAFF;'>make_relative</span>

### <span style='color: #89DDFF;'>make_relative(self) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Convert trajectory of absolute actions to relative actions.

<details>
<summary><b>Full Documentation</b></summary>

```
Convert trajectory of absolute actions to relative actions.

Returns:
  Trajectory: The converted relative trajectory.
```
</details>

## <span style='color: #82AAFF;'>make_absolute</span>

### <span style='color: #89DDFF;'>make_absolute(self, <span style='color: #FFCB6B;'>initial_state</span>: <span style='color: #C3E88D;'>None</span> | numpy.ndarray = None) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Convert trajectory of relative actions to absolute actions.

<details>
<summary><b>Full Documentation</b></summary>

```
Convert trajectory of relative actions to absolute actions.

Args:
  initial_state (np.ndarray): The initial state of the trajectory. Defaults to zeros.

Returns:
  Trajectory: The converted absolute trajectory.
```
</details>

## <span style='color: #82AAFF;'>resample</span>

### <span style='color: #89DDFF;'>resample(self, <span style='color: #FFCB6B;'>target_hz</span>: <span style='color: #C3E88D;'>float</span>) <span style='color: #F07178;'>-> 'Trajectory'</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>save</span>

### <span style='color: #89DDFF;'>save(self, <span style='color: #FFCB6B;'>filename</span>: <span style='color: #C3E88D;'>str</span> = 'trajectory.png') <span style='color: #F07178;'>-> None</span></span>

Save the current figure to a file.

<details>
<summary><b>Full Documentation</b></summary>

```
Save the current figure to a file.

Args:
  filename (str, optional): The filename to save the figure. Defaults to "trajectory.png".

Returns:
  None
```
</details>

## <span style='color: #82AAFF;'>show</span>

### <span style='color: #89DDFF;'>show(self) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Display the current figure.

<details>
<summary><b>Full Documentation</b></summary>

```
Display the current figure.

Returns:
  None
```
</details>

## <span style='color: #82AAFF;'>frequencies</span>

### <span style='color: #89DDFF;'>frequencies(self) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Plot the frequency spectrogram of the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Plot the frequency spectrogram of the trajectory.

Returns:
  Trajectory: The modified trajectory.
```
</details>

## <span style='color: #82AAFF;'>frequencies_nd</span>

### <span style='color: #89DDFF;'>frequencies_nd(self) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Plot the n-dimensional frequency spectrogram of the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Plot the n-dimensional frequency spectrogram of the trajectory.

Returns:
  Trajectory: The modified trajectory.
```
</details>

## <span style='color: #82AAFF;'>low_pass_filter</span>

### <span style='color: #89DDFF;'>low_pass_filter(self, <span style='color: #FFCB6B;'>cutoff_freq</span>: <span style='color: #C3E88D;'>float</span>) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Apply a low-pass filter to the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply a low-pass filter to the trajectory.

Args:
  cutoff_freq (float): The cutoff frequency for the low-pass filter.

Returns:
  Trajectory: The filtered trajectory.
```
</details>

## <span style='color: #82AAFF;'>q01</span>

### <span style='color: #89DDFF;'>q01(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>q99</span>

### <span style='color: #89DDFF;'>q99(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>mean</span>

### <span style='color: #89DDFF;'>mean(self) <span style='color: #F07178;'>-> numpy.ndarray | embdata.sample.Sample</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>variance</span>

### <span style='color: #89DDFF;'>variance(self) <span style='color: #F07178;'>-> numpy.ndarray | embdata.sample.Sample</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>std</span>

### <span style='color: #89DDFF;'>std(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>skewness</span>

### <span style='color: #89DDFF;'>skewness(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>kurtosis</span>

### <span style='color: #89DDFF;'>kurtosis(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>min</span>

### <span style='color: #89DDFF;'>min(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>max</span>

### <span style='color: #89DDFF;'>max(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>lower_quartile</span>

### <span style='color: #89DDFF;'>lower_quartile(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>median</span>

### <span style='color: #89DDFF;'>median(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>upper_quartile</span>

### <span style='color: #89DDFF;'>upper_quartile(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>non_zero_count</span>

### <span style='color: #89DDFF;'>non_zero_count(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>zero_count</span>

### <span style='color: #89DDFF;'>zero_count(self) <span style='color: #F07178;'>-> float</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>

## <span style='color: #82AAFF;'>transform</span>

### <span style='color: #89DDFF;'>transform(self, <span style='color: #FFCB6B;'>operation</span>: <span style='color: #C3E88D;'>Union</span>[Callable[[numpy.ndarray], numpy.ndarray], str], **kwargs) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Apply a transformation to the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply a transformation to the trajectory.

Available operations are:
- [un]minmax: Apply min-max normalization to the trajectory.
- [un]standard: Apply standard normalization to the trajectory.
- pca: Apply PCA normalization to the trajectory.
- absolute: Convert the trajectory to absolute actions.
- relative: Convert the trajectory to relative actions.


Args:
  operation (Callable | str): The operation to apply. Can be a callable or a string corresponding to a `make_` method on the Trajectory class.
  **kwargs: Additional keyword arguments to pass to the operation.
```
</details>

## <span style='color: #82AAFF;'>make_minmax</span>

### <span style='color: #89DDFF;'>make_minmax(self, <span style='color: #FFCB6B;'>min</span>: <span style='color: #C3E88D;'>float</span> = 0, <span style='color: #FFCB6B;'>max</span>: <span style='color: #C3E88D;'>float</span> = 1) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Apply min-max normalization to the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply min-max normalization to the trajectory.

Args:
  min (float, optional): The minimum value for the normalization. Defaults to 0.
  max (float, optional): The maximum value for the normalization. Defaults to 1.

Returns:
  Trajectory: The normalized trajectory.
```
</details>

## <span style='color: #82AAFF;'>make_pca</span>

### <span style='color: #89DDFF;'>make_pca(self, whiten=True) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Apply PCA normalization to the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply PCA normalization to the trajectory.

Returns:
  Trajectory: The PCA-normalized trajectory.
```
</details>

## <span style='color: #82AAFF;'>make_standard</span>

### <span style='color: #89DDFF;'>make_standard(self) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Apply standard normalization to the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Apply standard normalization to the trajectory.

Returns:
  Trajectory: The standardized trajectory.
```
</details>

## <span style='color: #82AAFF;'>make_unminmax</span>

### <span style='color: #89DDFF;'>make_unminmax(self, <span style='color: #FFCB6B;'>orig_min</span>: <span style='color: #C3E88D;'>numpy</span>.ndarray | embdata.sample.Sample, <span style='color: #FFCB6B;'>orig_max</span>: <span style='color: #C3E88D;'>numpy</span>.ndarray | embdata.sample.Sample) <span style='color: #F07178;'>-> 'Trajectory'</span></span>

Reverse min-max normalization on the trajectory.

<details>
<summary><b>Full Documentation</b></summary>

```
Reverse min-max normalization on the trajectory.
```
</details>

## <span style='color: #82AAFF;'>make_unstandard</span>

### <span style='color: #89DDFF;'>make_unstandard(self, <span style='color: #FFCB6B;'>mean</span>: <span style='color: #C3E88D;'>numpy</span>.ndarray, <span style='color: #FFCB6B;'>std</span>: <span style='color: #C3E88D;'>numpy</span>.ndarray) <span style='color: #F07178;'>-> 'Trajectory'</span></span>



<details>
<summary><b>Full Documentation</b></summary>

```

```
</details>


