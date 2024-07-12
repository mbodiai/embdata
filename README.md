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
  - [Plot, filter and transform your data with ease. On any type of data structure.](#plot-filter-and-transform-your-data-with-ease-on-any-type-of-data-structure)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
  - [Design Decisions](#design-decisions)
  - [API Reference](#api-reference)

## Installation

```console
pip install embdata
```

## Usage

<details>
<summary><strong>Sample</strong></summary>

The `Sample` class is a flexible base model for serializing, recording, and manipulating arbitrary data.

### Key Features
- Serialization and deserialization of complex data structures
- Flattening and unflattening of nested structures
- Conversion between different formats (e.g., dict, numpy arrays, torch tensors)
- Integration with machine learning frameworks and gym spaces

### Usage Example
```python
from embdata import Sample

# Create a simple Sample
sample = Sample(x=1, y=2, z={"a": 3, "b": 4})

# Flatten the sample
flat_sample = sample.flatten()
print(flat_sample)  # [1, 2, 3, 4]

# Convert to different formats
as_dict = sample.to("dict")
as_numpy = sample.to("np")
as_torch = sample.to("pt")

# Create a random sample based on the structure
random_sample = sample.random_sample()

# Get the corresponding Gym space
space = sample.space()

# Read a Sample from JSON or dictionary
sample_from_json = Sample.read('{"x": 1, "y": 2}')

# Get default value and space
default_sample = Sample.default_value()
default_space = Sample.default_space()

# Get model information
model_info = sample.model_info()

# Pack and unpack samples
samples = [Sample(a=1, b=2), Sample(a=3, b=4)]
packed = Sample.pack_from(samples)
unpacked = packed.unpack()

# Convert to HuggingFace Dataset and Features
dataset = sample.dataset()
features = sample.features()
```

### Methods
- `flatten()`: Flattens the nested structure into a 1D representation
- `unflatten()`: Reconstructs the original nested structure from a flattened representation
- `to(format)`: Converts the sample to different formats (dict, numpy, torch, etc.)
- `random_sample()`: Creates a random sample based on the current structure
- `space()`: Returns the corresponding Gym space for the sample
- `read()`: Reads a Sample instance from a JSON string, dictionary, or path
- `default_value()`: Gets the default value for the Sample instance
- `default_space()`: Returns the Gym space for the Sample class based on its class attributes
- `model_info()`: Gets the model information
- `pack_from()`: Packs a list of samples into a single sample with lists for attributes
- `unpack()`: Unpacks the packed Sample object into a list of Sample objects or dictionaries
- `dataset()`: Converts the Sample instance to a HuggingFace Dataset object
- `features()`: Converts the Sample instance to a HuggingFace Features object
- `space_for()`: Default Gym space generation for a given value
- `init_from()`: Initializes a Sample instance from various data types
- `from_space()`: Generates a Sample instance from a Gym space
- `model_field_info()`: Gets the extra json values set from a FieldInfo for a given attribute key
- `default_sample()`: Generates a default Sample instance from its class attributes
- `numpy()`: Converts the Sample instance to a numpy array
- `tolist()`: Converts the Sample instance to a list
- `torch()`: Converts the Sample instance to a PyTorch tensor
- `json()`: Converts the Sample instance to a JSON string

The `Sample` class provides a wide range of functionality for data manipulation, conversion, and integration with various libraries and frameworks.

</details>

<details>
<summary><strong>Image</strong></summary>

The `Image` class represents image data and provides methods for manipulation and conversion.

### Key Features
- Multiple representation formats (NumPy array, base64, file path, PIL Image, URL)
- Easy conversion between different image formats
- Resizing and encoding capabilities
- Integration with other data processing pipelines

### Usage Example
```python
from embdata import Image
import numpy as np

# Create an Image from a numpy array
array_data = np.random.rand(100, 100, 3)
img = Image(array=array_data)

# Convert to base64
base64_str = img.base64

# Open an image from a file
img_from_file = Image.open("path/to/image.jpg")

# Resize the image
resized_img = Image(img_from_file, size=(50, 50))

# Save the image
img.save("output_image.png")

# Create an Image from a URL
img_from_url = Image("https://example.com/image.jpg")

# Create an Image from a base64 string
img_from_base64 = Image.from_base64(base64_str, encoding="png")
```

### Methods
- `open(path)`: Opens an image from a file path
- `save(path, encoding, quality)`: Saves the image to a file
- `show()`: Displays the image using matplotlib
- `from_base64(base64_str, encoding, size, make_rgb)`: Creates an Image instance from a base64 string
- `load_url(url, download)`: Downloads an image from a URL or decodes it from a base64 data URI
- `from_bytes(bytes_data, encoding, size)`: Creates an Image instance from a bytes object
- `space()`: Returns the space of the image
- `dump(*args, as_field, **kwargs)`: Returns a dict or a field of the image
- `infer_features_dict()`: Infers features of the image

### Properties
- `array`: The image as a NumPy array
- `base64`: The image as a base64 encoded string
- `path`: The file path of the image
- `pil`: The image as a PIL Image object
- `url`: The URL of the image
- `size`: The size of the image as a (width, height) tuple
- `encoding`: The encoding format of the image

### Class Methods
- `supports(arg)`: Checks if the argument is supported by the Image class
- `pil_to_data(image, encoding, size, make_rgb)`: Creates an Image instance from a PIL image
- `bytes_to_data(bytes_data, encoding, size, make_rgb)`: Creates an Image instance from a bytes object

The `Image` class provides a convenient interface for working with image data in various formats and performing common image operations.

</details>

<details>
<summary><strong>Trajectory</strong></summary>

The `Trajectory` class represents a time series of multidimensional data, such as robot movements or sensor readings.

### Key Features
- Representation of time series data with optional frequency information
- Methods for statistical analysis, visualization, and manipulation
- Support for resampling and filtering operations

### Usage Example
```python
from embdata import Trajectory
import numpy as np

# Create a Trajectory
data = np.random.rand(100, 3)  # 100 timesteps, 3 dimensions
traj = Trajectory(data, freq_hz=10)

# Compute statistics
stats = traj.stats()
print(stats)

# Plot the trajectory
traj.plot()

# Resample the trajectory
resampled_traj = traj.resample(target_hz=5)

# Apply a low-pass filter
filtered_traj = traj.low_pass_filter(cutoff_freq=2)

# Save the plot
traj.save("trajectory_plot.png")
```

### Methods
- `stats()`: Computes statistics for the trajectory
- `plot()`: Plots the trajectory
- `resample(target_hz)`: Resamples the trajectory to a new frequency
- `low_pass_filter(cutoff_freq)`: Applies a low-pass filter to the trajectory
- `save(filename)`: Saves the trajectory plot to a file
- `show()`: Displays the trajectory plot

The `Trajectory` class offers methods for analyzing, visualizing, and manipulating trajectory data, making it easier to work with time series data in robotics and other applications.

</details>

<details>
<summary><strong>Episode</strong></summary>

The `Episode` class provides a list-like interface for a sequence of observations, actions, and other data, particularly useful for reinforcement learning scenarios.

### Key Features
- List-like interface for managing sequences of data
- Methods for appending, iterating, and splitting episodes
- Support for metadata and frequency information
- Integration with reinforcement learning workflows

### Usage Example
```python
from embdata import Episode, Sample

# Create an Episode
episode = Episode()

# Add steps to the episode
episode.append(Sample(observation=[1, 2, 3], action=0, reward=1))
episode.append(Sample(observation=[2, 3, 4], action=1, reward=0))
episode.append(Sample(observation=[3, 4, 5], action=0, reward=2))

# Iterate over the episode
for step in episode.iter():
    print(step.observation, step.action, step.reward)

# Split the episode based on a condition
def split_condition(step):
    return step.reward > 0

split_episodes = episode.split(split_condition)

# Extract a trajectory from the episode
action_trajectory = episode.trajectory(field="action", freq_hz=10)

# Access episode metadata
print(episode.metadata)
print(episode.freq_hz)
```

### Methods
- `append(step)`: Adds a new step to the episode
- `iter()`: Returns an iterator over the steps in the episode
- `split(condition)`: Splits the episode based on a given condition
- `trajectory(field, freq_hz)`: Extracts a trajectory from the episode for a specified field
- `filter(condition)`: Filters the episode based on a given condition

### Properties
- `metadata`: Additional metadata for the episode
- `freq_hz`: The frequency of the episode in Hz

The `Episode` class simplifies the process of working with sequential data in reinforcement learning and other time-series applications.

</details>

<details>
<summary><strong>Pose3D</strong></summary>

The `Pose3D` class represents absolute coordinates for a 3D space with x, y, and theta (orientation).

### Key Features
- Representation of 3D pose with position (x, y) and orientation (theta)
- Conversion between different units (meters, centimeters, radians, degrees)
- Conversion to different formats (list, dict)

### Usage Example
```python
from embdata.geometry import Pose3D
import math

# Create a Pose3D instance
pose = Pose3D(x=1, y=2, theta=math.pi/2)
print(pose)  # Pose3D(x=1.0, y=2.0,

 theta=1.5707963267948966)

# Convert to different units
pose_cm = pose.to("cm")
print(pose_cm)  # Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

pose_deg = pose.to(angular_unit="deg")
print(pose_deg)  # Pose3D(x=1.0, y=2.0, theta=90.0)

# Convert to different formats
pose_list = pose.to("list")
print(pose_list)  # [1.0, 2.0, 1.5707963267948966]

pose_dict = pose.to("dict")
print(pose_dict)  # {'x': 1.0, 'y': 2.0, 'theta': 1.5707963267948966}
```

### Methods
- `to(container_or_unit, unit, angular_unit)`: Converts the pose to different units or formats

The `Pose3D` class provides methods for converting between different units and representations of 3D poses, making it easier to work with spatial data in various contexts.

</details>

<details>
<summary><strong>HandControl</strong></summary>

The `HandControl` class represents an action for a 7D space, including the pose of a robot hand and its grasp state.

### Key Features
- Representation of robot hand pose and grasp state
- Integration with other motion control classes
- Support for complex nested structures

### Usage Example
```python
from embdata.geometry import Pose
from embdata.motion.control import HandControl

# Create a HandControl instance
hand_control = HandControl(
    pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
    grasp=0.5
)

# Access and modify the hand control
print(hand_control.pose.position)  # [0.1, 0.2, 0.3]
hand_control.grasp = 0.8
print(hand_control.grasp)  # 0.8

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

print(robot_control.hand.pose.position)  # [0.1, 0.2, 0.3]
print(robot_control.velocity)  # 0.3
```

### Attributes
- `pose`: The pose of the robot hand (Pose object)
- `grasp`: The openness of the robot hand (float, 0 to 1)

The `HandControl` class allows for easy manipulation and representation of robot hand controls in a 7D space, making it useful for robotics and motion control applications.

</details>

## License

`embdata` is distributed under the terms of the [apache-2.0](https://spdx.org/licenses/apache-2.0.html) license.

## Design Decisions

- [x] Grasp value is [-1, 1] so that the default value is 0.
- [x] Motion rather than Action to distinguish from non-physical actions.

## Classes

<details>
<summary><strong>Episode</strong></summary>

### Episode

The `Episode` class provides a list-like interface for a sequence of observations, actions, and/or other data. It's designed to streamline exploratory data analysis and manipulation of time series data.

#### Key Features
- List-like interface for managing sequences of data
- Methods for appending, iterating, and splitting episodes
- Support for metadata and frequency information
- Integration with reinforcement learning workflows

#### Usage Example

```python
from embdata import Episode, Sample

# Create an Episode
episode = Episode()

# Add steps to the episode
episode.append(Sample(observation=[1, 2, 3], action=0, reward=1))
episode.append(Sample(observation=[2, 3, 4], action=1, reward=0))
episode.append(Sample(observation=[3, 4, 5], action=0, reward=2))

# Iterate over the episode
for step in episode.iter():
    print(f"Observation: {step.observation}, Action: {step.action}, Reward: {step.reward}")

# Split the episode based on a condition
def split_condition(step):
    return step.reward > 0

split_episodes = episode.split(split_condition)

# Extract a trajectory from the episode
action_trajectory = episode.trajectory(field="action", freq_hz=10)

# Access episode metadata
print(episode.metadata)
print(episode.freq_hz)
```

#### Methods
- `append(step)`: Adds a new step to the episode
- `iter()`: Returns an iterator over the steps in the episode
- `split(condition)`: Splits the episode based on a given condition
- `trajectory(field, freq_hz)`: Extracts a trajectory from the episode for a specified field
- `filter(condition)`: Filters the episode based on a given condition

#### Properties
- `metadata`: Additional metadata for the episode
- `freq_hz`: The frequency of the episode in Hz

The `Episode` class simplifies the process of working with sequential data in reinforcement learning and other time-series applications.

</details>

<details>
<summary><strong>Image</strong></summary>

### Image

The `Image` class represents an image sample that can be represented in various formats, including NumPy arrays, base64 encoded strings, file paths, PIL Images, or URLs.

#### Key Features
- Multiple representation formats (NumPy array, base64, file path, PIL Image, URL)
- Easy conversion between different image formats
- Resizing and encoding capabilities
- Integration with other data processing pipelines

#### Usage Example

```python
from embdata import Image
import numpy as np

# Create an Image from a numpy array
array_data = np.random.rand(100, 100, 3)
img = Image(array=array_data)

# Convert to base64
base64_str = img.base64

# Open an image from a file
img_from_file = Image.open("path/to/image.jpg")

# Resize the image
resized_img = Image(img_from_file, size=(50, 50))

# Save the image
img.save("output_image.png")

# Create an Image from a base64 string
base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
image = Image.from_base64(base64_str, encoding="png", size=(1, 1))
print(image.size)  # Output: (1, 1)

# Example with complex nested structure
nested_data = {
    "image": Image.from_base64(base64_str, encoding="png"),
    "metadata": {
        "text": "A small red square",
        "tags": ["red", "square", "small"]
    }
}
print(nested_data["image"].size)  # Output: (1, 1)
print(nested_data["metadata"]["text"])  # Output: A small red square
```

#### Methods
- `open(path)`: Opens an image from a file path
- `save(path, encoding, quality)`: Saves the image to a file
- `show()`: Displays the image using matplotlib
- `from_base64(base64_str, encoding, size, make_rgb)`: Creates an Image instance from a base64 string

#### Properties
- `array`: The image as a NumPy array
- `base64`: The image as a base64 encoded string
- `path`: The file path of the image
- `pil`: The image as a PIL Image object
- `url`: The URL of the image
- `size`: The size of the image as a (width, height) tuple
- `encoding`: The encoding format of the image

The `Image` class provides a convenient interface for working with image data in various formats and performing common image operations.

</details>

<details>
<summary><strong>Sample</strong></summary>

### Sample

The `Sample` class is a base model for serializing, recording, and manipulating arbitrary data. It provides a flexible and extensible way to handle complex data structures, including nested objects, arrays, and various data types.

#### Key Features
- Serialization and deserialization of complex data structures
- Flattening and unflattening of nested structures
- Conversion between different formats (e.g., dict, numpy arrays, torch tensors)
- Integration with machine learning frameworks and gym spaces

#### Usage Example

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

#### Methods
- `flatten(output_type="list", non_numerical="allow", ignore=None, sep=".", to=None)`: Flattens the Sample instance into a one-dimensional structure
- `unflatten(one_d_array_or_dict, schema=None)`: Unflattens a one-dimensional array or dictionary into a Sample instance
- `to(container)`: Converts the Sample instance to a different container type
- `schema(include_descriptions=False)`: Get a simplified JSON schema of the data
- `space()`: Return the corresponding Gym space for the Sample instance
- `random_sample()`: Generate a random Sample instance based on its attributes

The `Sample` class provides a wide range of functionality for data manipulation, conversion, and integration with various libraries and frameworks.

</details>

<details>
<summary><strong>Trajectory</strong></summary>

### Trajectory

The `Trajectory` class represents a trajectory of steps, typically used for time series of multidimensional data such as robot movements or sensor readings.

#### Key Features
- Representation of time series data with optional frequency information
- Methods for statistical analysis, visualization, and manipulation
- Support for resampling and filtering operations
- Transformation and normalization capabilities

#### Usage Example

```python
import numpy as np
from embdata import Trajectory

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

# Resample the trajectory
resampled_traj = traj.resample(target_hz=5)

# Access data
print(traj.array)  # Output: [[0 0] [1 1] [2 0] [3 1] [4 0]]

# Get statistics
stats = traj.stats()
print(stats.mean)  # Output: [2. 0.4]
print(stats.std)   # Output: [1.41421356 0.48989795]

# Slice the trajectory
sliced_traj = traj[1:4]
print(sliced_traj.array)  # Output: [[1 1] [2 0] [3 1]]

# Transform the trajectory
normalized_traj = traj.transform('minmax')
normalized_traj.plot().show()
```

#### Methods
- `plot()`: Plot the trajectory
- `stats()`: Compute statistics for the trajectory
- `low_pass_filter(cutoff_freq)`: Apply a low-pass filter to the trajectory
- `resample(target_hz)`: Resample the trajectory to a new frequency
- `make_relative()`: Convert the trajectory to relative actions
- `make_absolute(initial_state)`: Convert relative actions to absolute actions
- `frequencies()`: Plot the frequency spectrogram of the trajectory
- `frequencies_nd()`: Plot the n-dimensional frequency spectrogram of the trajectory
- `transform(operation, **kwargs)`: Apply a transformation to the trajectory
- `make_minmax(min, max)`: Apply min-max normalization
- `make_pca(whiten)`: Apply PCA transformation
- `make_standard()`: Apply standard normalization
- `make_unminmax(orig_min, orig_max)`: Reverse min-max normalization
- `make_unstandard(mean, std)`: Reverse standard normalization
- `q01()`, `q99()`: Get 1st and 99th percentiles
- `mean()`, `variance()`, `std()`, `skewness()`, `kurtosis()`: Statistical measures
- `min()`, `max()`: Minimum and maximum values
- `lower_quartile()`, `median()`, `upper_quartile()`: Quartile values
- `non_zero_count()`, `zero_count()`: Count non-zero and zero values

#### Properties
- `array`: The trajectory data as a NumPy array
- `freq_hz`: The frequency of the trajectory in Hz
- `time_idxs`: The time index of each step in the trajectory
- `dim_labels`: The labels for each dimension of the trajectory

The `Trajectory` class offers comprehensive methods for analyzing, visualizing, manipulating, and transforming trajectory data, making it easier to work with time series data in robotics and other applications.

</details>

<details>
<summary><strong>Motion</strong></summary>

### Motion

The `Motion` class is a base class for defining motion-related data structures. It extends the `Coordinate` class and provides a foundation for creating motion-specific data models.

#### Key Features
- Base class for motion-specific data models
- Integration with MotionField and its variants for proper validation and type checking
- Support for defining bounds and motion types

#### Usage Example

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

# Create a Twist motion
twist = Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

print(twist)  # Output: Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

# Access individual fields
print(twist.x)  # Output: 0.5

# Validate bounds
try:
    invalid_twist = Twist(x=1.5)  # This will raise a ValueError
except ValueError as e:
    print(f"Validation error: {e}")

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

#### Key Concepts
- Subclasses of Motion should define their fields using MotionField or its variants (e.g., AbsoluteMotionField, VelocityMotionField) to ensure proper validation and type checking.
- The Motion class does not allow extra fields and enforces validation of motion type, shape, and bounds.
- It can handle various types of motion data, including nested structures with images and text, as long as they are properly defined using the appropriate MotionFields.

The `Motion` class provides a flexible foundation for creating motion-specific data models with built-in validation and type checking, making it easier to work with complex motion data in robotics and other applications.

</details>

<details>
<summary><strong>HandControl</strong></summary>

### HandControl

The `HandControl` class represents an action for a 7D space, including the pose of a robot hand and its grasp state.

#### Key Features
- Representation of robot hand pose and grasp state
- Integration with other motion control classes
- Support for complex nested structures

#### Usage Example

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

#### Attributes
- `pose` (Pose): The pose of the robot hand, including position and orientation.
- `grasp` (float): The openness of the robot hand, ranging from 0 (closed) to 1 (open).

The `HandControl` class allows for easy manipulation and representation of robot hand controls in a 7D space, making it useful for robotics and motion control applications. It can be integrated into more complex control structures and supports nested data representations.

</details>
## Geometry Module

The `embdata.geometry` module provides classes for representing geometric data in cartesian and polar coordinates.

### Classes

#### CoordinateField

A function to create a Pydantic Field with extra metadata for coordinates.

#### Coordinate

A base class for representing coordinates in an arbitrary space.

Methods:
- `convert_linear_unit(value: float, from_unit: str, to_unit: str) -> float`
- `convert_angular_unit(value: float, from_unit: str, to_unit: str) -> float`
- `validate_bounds()`

#### Pose3D

Represents absolute coordinates for a 3D space (x, y, theta).

Methods:
- `to(container_or_unit=None, unit="m", angular_unit="rad", **kwargs) -> Any`

#### Pose6D

Represents absolute coordinates for a 6D space (x, y, z, roll, pitch, yaw).

Methods:
- `to(container_or_unit=None, sequence="zyx", unit="m", angular_unit="rad", **kwargs) -> Any`
- `get_quaternion(sequence="zyx") -> np.ndarray`
- `get_rotation_matrix(sequence="zyx") -> np.ndarray`

### Usage

```python
import math
from embdata.geometry import Pose3D, Pose6D

# Create a 3D pose
pose_3d = Pose3D(x=1, y=2, theta=math.pi / 2)
print(pose_3d)  # Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)

# Convert to centimeters
pose_3d_cm = pose_3d.to("cm")
print(pose_3d_cm)  # Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

# Create a 6D pose
pose_6d = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=math.pi / 2)

# Get quaternion representation
quaternion = pose_6d.get_quaternion()
print(quaternion)  # [0. 0. 0.70710678 0.70710678]

# Get rotation matrix
rotation_matrix = pose_6d.get_rotation_matrix()
print(rotation_matrix)
# [[ 0. -1.  0.]
#  [ 1.  0.  0.]
#  [ 0.  0.  1.]]
```
## Geometry Module

The `embdata.geometry` module provides classes for representing geometric data in cartesian and polar coordinates.

### Classes

#### CoordinateField

A function to create a Pydantic Field with extra metadata for coordinates.

#### Coordinate

A base class for representing coordinates in an arbitrary space.

Methods:
- `convert_linear_unit(value: float, from_unit: str, to_unit: str) -> float`
- `convert_angular_unit(value: float, from_unit: str, to_unit: str) -> float`
- `validate_bounds()`

#### Pose3D

Represents absolute coordinates for a 3D space (x, y, theta).

Methods:
- `to(container_or_unit=None, unit="m", angular_unit="rad", **kwargs) -> Any`

#### Pose6D

Represents absolute coordinates for a 6D space (x, y, z, roll, pitch, yaw).

Methods:
- `to(container_or_unit=None, sequence="zyx", unit="m", angular_unit="rad", **kwargs) -> Any`
- `get_quaternion(sequence="zyx") -> np.ndarray`
- `get_rotation_matrix(sequence="zyx") -> np.ndarray`

### Usage

```python
import math
from embdata.geometry import Pose3D, Pose6D

# Create a 3D pose
pose_3d = Pose3D(x=1, y=2, theta=math.pi / 2)
print(pose_3d)  # Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)

# Convert to centimeters
pose_3d_cm = pose_3d.to("cm")
print(pose_3d_cm)  # Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

# Create a 6D pose
pose_6d = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=math.pi / 2)

# Get quaternion representation
quaternion = pose_6d.get_quaternion()
print(quaternion)  # [0. 0. 0.70710678 0.70710678]

# Get rotation matrix
rotation_matrix = pose_6d.get_rotation_matrix()
print(rotation_matrix)
# [[ 0. -1.  0.]
#  [ 1.  0.  0.]
#  [ 0.  0.  1.]]
```
