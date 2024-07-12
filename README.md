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
  - [Classes](#classes-1)

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
## API Reference

<details>
<summary>Demo Module</summary>

### `load_and_process_dataset()`

Load a dataset, create a sample, and process it into actions, observations, and states.

**Returns:**
- `tuple`: A tuple containing actions, observations, and states as flattened samples.

**Example:**
```python
actions, observations, states = load_and_process_dataset()
print(type(actions), type(observations), type(states))
# Output: <class 'embdata.sample.Sample'> <class 'embdata.sample.Sample'> <class 'embdata.sample.Sample'>
```

### `create_and_analyze_episode(observations, actions, states)`

Create an episode from observations, actions, and states, and perform various analyses.

**Args:**
- `observations` (Sample): Flattened observations.
- `actions` (Sample): Flattened actions.
- `states` (Sample): Flattened states.

**Returns:**
- `Episode`: The created and analyzed episode.

**Example:**
```python
actions, observations, states = load_and_process_dataset()
episode = create_and_analyze_episode(observations, actions, states)
print(type(episode))
# Output: <class 'embdata.episode.Episode'>
```

</details>
## API Reference

<details>
<summary>embdata</summary>

### embdata

```python
"""embdata: A package for handling embodied AI data structures and operations.

This package provides classes and utilities for working with various data types
commonly used in embodied AI tasks, such as episodes, time steps, images, and samples.

Examples:
    >>> from embdata import Episode, TimeStep, Image, Sample
    >>> # Create a complex nested structure with image and text data
    >>> image_data = Image.from_base64("base64_encoded_image_data", encoding="jpeg")
    >>> text_data = Sample(text="This is a sample text")
    >>> action = Sample(velocity=1.0, rotation=0.5)
    >>> observation = Sample(image=image_data, text=text_data)
    >>> time_step = TimeStep(observation=observation, action=action)
    >>> episode = Episode(steps=[time_step])
    >>> print(len(episode))
    1
    >>> print(episode.steps[0].observation.image.encoding)
    'jpeg'
    >>> print(episode.steps[0].observation.text.text)
    'This is a sample text'
    >>> print(episode.steps[0].action.velocity)
    1.0
"""
```

This package includes the following main classes:
- `Episode`: Represents a sequence of time steps in an embodied AI task.
- `TimeStep`: Represents a single step in an episode, containing observation and action data.
- `ImageTask`: A specialized episode type for image-based tasks.
- `VisionMotorStep`: A specialized time step for vision and motor tasks.
- `Image`: Represents image data in various formats (base64, file path, numpy array, etc.).
- `Sample`: A base class for serializing, recording, and manipulating arbitrary data.

</details>
## API Reference

<details>
<summary><strong>to_vision_motor_step</strong></summary>

```python
def to_vision_motor_step(step: Dict, index: int | None = None) -> VisionMotorStep:
    """Convert a dictionary step to a VisionMotorStep object.

    This function takes a dictionary representing a step in a robotic arm dataset
    and converts it into a VisionMotorStep object. The step typically includes
    information about the observation (image and instruction) and the action taken.

    Args:
        step (Dict): A dictionary containing step information.
        index (int | None, optional): The index of the step. Defaults to None.

    Returns:
        VisionMotorStep: A VisionMotorStep object representing the converted step.

    Example:
        >>> step_dict = {
        ...     "episode": 1,
        ...     "observation": {
        ...         "image": {
        ...             "bytes": b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\nIDATx\\x9cc\\x00\\x01\\x00\\x00\\x05\\x00\\x01\\r\\n-\\xb4\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"
        ...         },
        ...         "instruction": "Move the robotic arm to grasp the red cube on the left",
        ...     },
        ...     "action": {"x": 0.1, "y": -0.2, "z": 0.05, "roll": 0.1, "pitch": -0.1, "yaw": 0.2, "gripper": 0.7},
        ... }
        >>> vision_motor_step = to_vision_motor_step(step_dict, index=0)
        >>> print(vision_motor_step)
        VisionMotorStep(step_idx=0, episode_idx=1, observation=ImageTask(...), relative_action=RelativePoseHandControl(...))
        >>> print(vision_motor_step.observation.task)
        Move the robotic arm to grasp the red cube on the left
        >>> print(vision_motor_step.relative_action)
        RelativePoseHandControl(x=0.1, y=-0.2, z=0.05, roll=0.1, pitch=-0.1, yaw=0.2, gripper=0.7)
    """
```

</details>

<details>
<summary><strong>to_vision_motor_episode</strong></summary>

```python
def to_vision_motor_episode(episode: List[Dict]) -> VisionMotorEpisode:
    """Convert a list of steps to a VisionMotorEpisode object.

    This function takes a list of dictionaries, each representing a step in an episode,
    and converts them into a VisionMotorEpisode object. This is useful for processing
    entire episodes of robotic arm interactions.

    Args:
        episode (List[Dict]): A list of dictionaries, each representing a step in the episode.

    Returns:
        VisionMotorEpisode: A VisionMotorEpisode object containing the converted steps.

    Example:
        >>> episode_steps = [
        ...     {
        ...         "episode": 1,
        ...         "observation": {
        ...             "image": {
        ...                 "bytes": b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\nIDATx\\x9cc\\x00\\x01\\x00\\x00\\x05\\x00\\x01\\r\\n-\\xb4\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"
        ...             },
        ...             "instruction": "Locate the blue sphere"
        ...         },
        ...         "action": {"x": 0.1, "y": -0.2, "z": 0.0, "roll": 0, "pitch": 0, "yaw": 0, "gripper": 0.5},
        ...     },
        ...     {
        ...         "episode": 1,
        ...         "observation": {
        ...             "image": {
        ...                 "bytes": b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\nIDATx\\x9cc\\x00\\x01\\x00\\x00\\x05\\x00\\x01\\r\\n-\\xb4\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"
        ...             },
        ...             "instruction": "Move towards the blue sphere"
        ...         },
        ...         "action": {"x": 0.2, "y": 0.1, "z": -0.1, "roll": 0.1, "pitch": 0, "yaw": -0.1, "gripper": 0.5},
        ...     },
        ...     {
        ...         "episode": 1,
        ...         "observation": {
        ...             "image": {
        ...                 "bytes": b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\nIDATx\\x9cc\\x00\\x01\\x00\\x00\\x05\\x00\\x01\\r\\n-\\xb4\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"
        ...             },
        ...             "instruction": "Grasp the blue sphere"
        ...         },
        ...         "action": {"x": 0.0, "y": 0.0, "z": -0.2, "roll": 0, "pitch": 0, "yaw": 0, "gripper": 1.0},
        ...     },
        ... ]
        >>> vision_motor_episode = to_vision_motor_episode(episode_steps)
        >>> print(len(vision_motor_episode.steps))
        3
        >>> print(vision_motor_episode.steps[1].observation.task)
        Move towards the blue sphere
        >>> print(vision_motor_episode.steps[2].relative_action)
        RelativePoseHandControl(x=0.0, y=0.0, z=-0.2, roll=0, pitch=0, yaw=0, gripper=1.0)
    """
```

</details>

<details>
<summary><strong>process_dataset</strong></summary>

```python
def process_dataset(dataset_name: str, num_episodes: int = 48) -> List[VisionMotorEpisode]:
    """Process a dataset and convert it into a list of VisionMotorEpisode objects.

    This function loads a specified dataset, processes a given number of episodes,
    and converts them into VisionMotorEpisode objects. It also performs some
    additional processing and visualization on the episodes.

    Args:
        dataset_name (str): The name of the dataset to process.
        num_episodes (int, optional): The number of episodes to process. Defaults to 48.

    Returns:
        List[VisionMotorEpisode]: A list of processed VisionMotorEpisode objects.

    Example:
        >>> processed_episodes = process_dataset("mbodiai/xarm_7_6_delta", num_episodes=2)
        >>> print(len(processed_episodes))
        2
        >>> print(type(processed_episodes[0]))
        <class 'embdata.episode.VisionMotorEpisode'>
        >>> print(len(processed_episodes[0].steps))
        # This will print the number of steps in the first episode
        >>> print(processed_episodes[1].steps[0].observation.task)
        # This will print the instruction for the first step of the second episode
        >>> print(processed_episodes[0].steps[0].observation.image)
        # This will print the Image object for the first step of the first episode
    """
```

</details>
