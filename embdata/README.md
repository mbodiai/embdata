# embdata

A Python package for working with embodied AI datasets.

## Installation

```bash
pip install embdata
```

## Usage

<details>
<summary>Click to expand</summary>

### Sample

```python
from embdata import Sample

# Create a sample
sample = Sample(field1="value1", field2=42)

# Convert to dictionary
sample_dict = sample.dict()

# Flatten the sample
flattened = sample.flatten()

# Unflatten a dictionary or array
unflattened = Sample.unflatten(flattened)

# Get the schema
schema = sample.schema()

# Convert to different formats
numpy_sample = sample.numpy()
torch_sample = sample.torch()

# Get the space representation
space = sample.space()
```

### Episode

```python
from embdata import Episode, TimeStep

# Create an episode
episode = Episode([
    TimeStep(observation=obs1, action=action1),
    TimeStep(observation=obs2, action=action2),
])

# Iterate through steps
for step in episode:
    print(step.observation, step.action)

# Filter steps
filtered_episode = episode.filter(lambda step: step.action > 0)

# Split episode
split_episodes = episode.split(lambda step: step.observation == target)

# Convert to dataset
dataset = episode.dataset()

# Visualize the episode
episode.show()
```

### Image

```python
from embdata import Image

# Create an image from various sources
image = Image.open("path/to/image.jpg")
image_from_array = Image(array=numpy_array)
image_from_base64 = Image(base64=base64_string)

# Save the image
image.save("output.png")

# Show the image
image.show()

# Convert to different formats
pil_image = image.pil
numpy_array = image.array
base64_string = image.base64
```

### Trajectory

```python
from embdata import Trajectory
import numpy as np

# Create a trajectory
trajectory = Trajectory(steps=np.random.rand(100, 3), freq_hz=10)

# Get statistics
stats = trajectory.stats()

# Plot the trajectory
trajectory.plot()

# Resample the trajectory
resampled = trajectory.resample(target_hz=20)

# Apply transformations
normalized = trajectory.make_minmax()
standardized = trajectory.make_standard()
```

### Coordinate and Pose

```python
from embdata import Coordinate, Pose3D, Pose6D

# Create coordinates
coord = Coordinate([1.0, 2.0, 3.0])

# Create poses
pose3d = Pose3D(x=1.0, y=2.0, theta=0.5)
pose6d = Pose6D(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)

# Convert units
pose3d_meters = pose3d.to(unit="m")
pose6d_degrees = pose6d.to(angular_unit="deg")

# Get rotation matrix or quaternion
rotation_matrix = pose6d.get_rotation_matrix()
quaternion = pose6d.get_quaternion()
```

### Motion Control

```python
from embdata.motion import HandControl, HeadControl, MobileSingleArmControl, MobileSingleHandControl, HumanoidControl
from embdata.motion import Motion

# Create motion controls
hand_control = HandControl(gripper=0.5)
head_control = HeadControl(pan=0.2, tilt=-0.1)
arm_control = MobileSingleArmControl(x=0.1, y=0.2, z=0.3, gripper=0.5)
mobile_hand_control = MobileSingleHandControl(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3, gripper=0.5)
humanoid_control = HumanoidControl(
    left_arm=[0.1, 0.2, 0.3, 0.4],
    right_arm=[0.5, 0.6, 0.7, 0.8],
    left_leg=[0.9, 1.0, 1.1],
    right_leg=[1.2, 1.3, 1.4]
)

# Access control values
print(hand_control.gripper)
print(head_control.pan, head_control.tilt)
print(arm_control.x, arm_control.y, arm_control.z, arm_control.gripper)
print(mobile_hand_control.x, mobile_hand_control.y, mobile_hand_control.z, mobile_hand_control.roll, mobile_hand_control.pitch, mobile_hand_control.yaw, mobile_hand_control.gripper)
print(humanoid_control.left_arm, humanoid_control.right_leg)

# Create a custom Motion control by subclassing
class QuadrupedControl(Motion):
    front_left: float
    front_right: float
    back_left: float
    back_right: float

# Use the custom Motion control
quadruped_control = QuadrupedControl(front_left=0.1, front_right=0.2, back_left=0.3, back_right=0.4)
print(quadruped_control.front_left, quadruped_control.back_right)
```

</details>
