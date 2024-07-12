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
## Classes

<details>
<summary><strong>Trajectory</strong></summary>

### Trajectory

A trajectory of steps representing a time series of multidimensional data.

This class provides methods for analyzing, visualizing, and manipulating trajectory data,
such as robot movements, sensor readings, or any other time-series data.

#### Attributes:
- `steps` (NumpyArray | List[Sample | NumpyArray]): The trajectory data.
- `freq_hz` (float | None): The frequency of the trajectory in Hz.
- `time_idxs` (NumpyArray | None): The time index of each step in the trajectory.
- `dim_labels` (list[str] | None): The labels for each dimension of the trajectory.
- `angular_dims` (list[int] | list[str] | None): The dimensions that are angular.

#### Methods:
- `plot`: Plot the trajectory.
- `map`: Apply a function to each step in the trajectory.
- `make_relative`: Convert the trajectory to relative actions.
- `resample`: Resample the trajectory to a new sample rate.
- `frequencies`: Plot the frequency spectrogram of the trajectory.
- `frequencies_nd`: Plot the n-dimensional frequency spectrogram of the trajectory.
- `low_pass_filter`: Apply a low-pass filter to the trajectory.
- `stats`: Compute statistics for the trajectory.
- `transform`: Apply a transformation to the trajectory.

#### Example:
```python
import numpy as np
from embdata.trajectory import Trajectory

# Create a simple 2D trajectory
steps = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
traj = Trajectory(steps, freq_hz=10, dim_labels=['X', 'Y'])

# Plot the trajectory
traj.plot().show()

# Compute and print statistics
print(traj.stats())

# Apply a low-pass filter
filtered_traj = traj.low_pass_filter(cutoff_freq=2)
filtered_traj.plot().show()
```

</details>
## Classes

<details>
<summary><strong>Pose3D</strong></summary>

### Pose3D

Absolute coordinates for a 3D space representing x, y, and theta.

This class represents a pose in 3D space with x and y coordinates for position
and theta for orientation.

#### Attributes:
- `x` (float): X-coordinate in meters.
- `y` (float): Y-coordinate in meters.
- `theta` (float): Orientation angle in radians.

#### Methods:
- `to(container_or_unit=None, unit="m", angular_unit="rad", **kwargs) -> Any`: Convert the pose to a different unit or container.

#### Example:
```python
import math
from embdata.geometry import Pose3D

# Create a Pose3D instance
pose = Pose3D(x=1, y=2, theta=math.pi/2)
print(pose)  # Output: Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)

# Convert to centimeters
pose_cm = pose.to("cm")
print(pose_cm)  # Output: Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

# Convert theta to degrees
pose_deg = pose.to(angular_unit="deg")
print(pose_deg)  # Output: Pose3D(x=1.0, y=2.0, theta=90.0)

# Convert to a list
pose_list = pose.to("list")
print(pose_list)  # Output: [1.0, 2.0, 1.5707963267948966]

# Convert to a dictionary
pose_dict = pose.to("dict")
print(pose_dict)  # Output: {'x': 1.0, 'y': 2.0, 'theta': 1.5707963267948966}
```

</details>
## Classes

<details>
<summary><strong>Sample</strong></summary>

### Sample

A base model class for serializing, recording, and manipulating arbitrary data.

This class provides a flexible and extensible way to handle complex data structures,
including nested objects, arrays, and various data types. It offers methods for
flattening, unflattening, converting between different formats, and working with
machine learning frameworks.

#### Attributes:
- `model_config` (ConfigDict): Configuration for the model, including settings for validation, extra fields, and arbitrary types.

#### Methods:
- `__init__(self, item=None, **data)`: Initialize a Sample instance.
- `schema(self, include_descriptions=False)`: Get a simplified JSON schema of the data.
- `to(self, container)`: Convert the Sample instance to a different container type.
- `flatten(self, output_type="list", non_numerical="allow", ignore=None, sep=".", to=None)`: Flatten the Sample instance into a one-dimensional structure.
- `unflatten(cls, one_d_array_or_dict, schema=None)`: Unflatten a one-dimensional array or dictionary into a Sample instance.
- `space(self)`: Return the corresponding Gym space for the Sample instance.
- `random_sample(self)`: Generate a random Sample instance based on its attributes.

#### Example:
```python
from embdata import Sample
import numpy as np

# Create a simple Sample instance
sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)

# Flatten the sample
flat_sample = sample.flatten()
print(flat_sample)  # Output: [1, 2, 3, 4, 5]

# Get the schema
schema = sample.schema()
print(schema)

# Unflatten a list back to a Sample instance
unflattened_sample = Sample.unflatten(flat_sample, schema)
print(unflattened_sample)  # Output: Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)

# Create a complex nested structure
nested_sample = Sample(
    image=Sample(
        data=np.random.rand(32, 32, 3),
        metadata={"format": "RGB", "size": (32, 32)}
    ),
    text=Sample(
        content="Hello, world!",
        tokens=["Hello", ",", "world", "!"],
        embeddings=np.random.rand(4, 128)
    ),
    labels=["greeting", "example"]
)

# Get the schema of the nested structure
nested_schema = nested_sample.schema()
print(nested_schema)
```

</details>
## API Reference

<details>
<summary>to_features_dict</summary>

```python
def to_features_dict(indict: Any, exclude_keys: set | None = None) -> Dict[str, Any]:
    """
    Convert a dictionary to a Datasets Features object.

    This function recursively converts a nested dictionary into a format compatible with
    Hugging Face Datasets' Features. It handles various data types including strings,
    integers, floats, lists, and PIL Images.

    Args:
        indict: The input to convert. Can be a dictionary, string, int, float, list, tuple, numpy array, or PIL Image.
        exclude_keys: A set of keys to exclude from the conversion. Defaults to None.

    Returns:
        A dictionary representation of the Features object for Hugging Face Datasets.

    Raises:
        ValueError: If an empty list is provided or if the input type is not supported.

    Examples:
        Simple dictionary conversion:
        >>> to_features_dict({"name": "Alice", "age": 30})
        {'name': Value(dtype='string', id=None), 'age': Value(dtype='int64', id=None)}

        List conversion:
        >>> to_features_dict({"scores": [85, 90, 95]})
        {'scores': [Value(dtype='int64', id=None)]}

        Numpy array conversion:
        >>> import numpy as np
        >>> to_features_dict({"data": np.array([1, 2, 3])})
        {'data': [Value(dtype='int64', id=None)]}

        PIL Image conversion:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (60, 30), color="red")
        >>> to_features_dict({"image": img})
        {'image': Image(decode=True, id=None)}

        Nested structure with image and text:
        >>> complex_data = {
        ...     "user_info": {
        ...         "name": "John Doe",
        ...         "age": 28
        ...     },
        ...     "posts": [
        ...         {
        ...             "text": "Hello, world!",
        ...             "image": Image.new("RGB", (100, 100), color="blue"),
        ...             "likes": 42
        ...         },
        ...         {
        ...             "text": "Another post",
        ...             "image": Image.new("RGB", (200, 150), color="green"),
        ...             "likes": 17
        ...         }
        ...     ]
        ... }
        >>> features = to_features_dict(complex_data)
        >>> features
        {
            'user_info': {
                'name': Value(dtype='string', id=None),
                'age': Value(dtype='int64', id=None)
            },
            'posts': [
                {
                    'text': Value(dtype='string', id=None),
                    'image': Image(decode=True, id=None),
                    'likes': Value(dtype='int64', id=None)
                }
            ]
        }
    """
```

</details>
## API Reference

<details>
<summary>Image</summary>

```python
class Image(Sample):
    """An image sample that can be represented in various formats.

    The image can be represented as a NumPy array, a base64 encoded string, a file path, a PIL Image object,
    or a URL. The image can be resized to and from any size and converted to and from any supported format.

    Attributes:
        array (Optional[np.ndarray]): The image represented as a NumPy array.
        base64 (Optional[Base64Str]): The base64 encoded string of the image.
        path (Optional[FilePath]): The file path of the image.
        pil (Optional[PILImage]): The image represented as a PIL Image object.
        url (Optional[AnyUrl]): The URL of the image.
        size (Optional[tuple[int, int]]): The size of the image as a (width, height) tuple.
        encoding (Optional[Literal["png", "jpeg", "jpg", "bmp", "gif"]]): The encoding of the image.

    Example:
        >>> image = Image("https://example.com/image.jpg")
        >>> image = Image("/path/to/image.jpg")
        >>> image = Image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4Q3zaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA")

        >>> jpeg_from_png = Image("path/to/image.png", encoding="jpeg")
        >>> resized_image = Image(image, size=(224, 224))
        >>> pil_image = Image(image).pil
        >>> array = Image(image).array
        >>> base64 = Image(image).base64
    """

    @staticmethod
    def from_base64(base64_str: str, encoding: str, size=None, make_rgb=False) -> "Image":
        """Decodes a base64 string to create an Image instance.

        This method takes a base64 encoded string representation of an image,
        decodes it, and creates an Image instance from it. It's useful when
        you have image data in base64 format and want to work with it as an Image object.

        Args:
            base64_str (str): The base64 string to decode.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
            make_rgb (bool): Whether to convert the image to RGB format. Defaults to False.

        Returns:
            Image: An instance of the Image class with populated fields.

        Example:
            >>> base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
            >>> image = Image.from_base64(base64_str, encoding="png", size=(1, 1))
            >>> print(image.size)
            (1, 1)

            # Example with complex nested structure
            >>> nested_data = {
            ...     "image": Image.from_base64(base64_str, encoding="png"),
            ...     "metadata": {
            ...         "text": "A small red square",
            ...         "tags": ["red", "square", "small"]
            ...     }
            ... }
            >>> print(nested_data["image"].size)
            (1, 1)
            >>> print(nested_data["metadata"]["text"])
            A small red square
        """
```

</details>
## Classes

<details>
<summary><strong>Motion</strong></summary>

### Motion

Base class for defining motion-related data structures.

This class extends the Coordinate class and provides a foundation for creating
motion-specific data models. It does not allow extra fields and enforces
validation of motion type, shape, and bounds.

#### Attributes:
- Inherited from Coordinate

#### Usage:
Subclasses of Motion should define their fields using MotionField or its variants
(e.g., AbsoluteMotionField, VelocityMotionField) to ensure proper validation and
type checking.

#### Example:
```python
from embdata.motion import Motion
from embdata.motion.fields import VelocityMotionField

class Twist(Motion):
    x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    roll: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    pitch: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    yaw: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])

# Create a Twist instance
twist = Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)
print(twist)
# Output: Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

# Attempt to create an invalid Twist instance
try:
    invalid_twist = Twist(x=1.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: x value 1.5 is not within bounds [-1.0, 1.0]

# Example with complex nested structure
class RobotMotion(Motion):
    twist: Twist
    gripper: float = VelocityMotionField(default=0.0, bounds=[0.0, 1.0])

robot_motion = RobotMotion(
    twist=Twist(x=0.2, y=0.1, z=0.0, roll=0.0, pitch=0.0, yaw=0.1),
    gripper=0.5
)
print(robot_motion)
# Output: RobotMotion(twist=Twist(x=0.2, y=0.1, z=0.0, roll=0.0, pitch=0.0, yaw=0.1), gripper=0.5)
```

Note: The Motion class is designed to work with complex nested structures.
It can handle various types of motion data, including images and text,
as long as they are properly defined using the appropriate MotionFields.

</details>
## Classes

<details>
<summary><strong>HandControl</strong></summary>

### HandControl

Action for a 7D space representing x, y, z, roll, pitch, yaw, and openness of the hand.

This class represents the control for a robot hand, including its pose and grasp state.

#### Attributes:
- `pose` (Pose): The pose of the robot hand, including position and orientation.
- `grasp` (float): The openness of the robot hand, ranging from 0 (closed) to 1 (open).

#### Example:
```python
from embdata.geometry import Pose
from embdata.motion.control import HandControl

# Create a HandControl instance
hand_control = HandControl(
    pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
    grasp=0.5
)

# Access and modify the hand control
print(hand_control.pose.position)  # Output: [0.1, 0.2, 0.3]
hand_control.grasp = 0.8
print(hand_control.grasp)  # Output: 0.8

# Example with complex nested structure
from embdata.motion import Motion
from embdata.motion.fields import VelocityMotionField

class RobotControl(Motion):
    hand: HandControl
    velocity: float = VelocityMotionField(default=0.0, bounds=[0.0, 1.0])

robot_control = RobotControl(
    hand=HandControl(
        pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
        grasp=0.5
    ),
    velocity=0.3
)

print(robot_control.hand.pose.position)  # Output: [0.1, 0.2, 0.3]
print(robot_control.velocity)  # Output: 0.3
```

</details>
