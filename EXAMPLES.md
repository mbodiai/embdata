# embdata Examples

This document provides examples of how to use the `embdata` library for various tasks.

</details>

<details>
<summary><strong>Transforming Types</strong></summary>

```python
from embdata.sample import Sample
from embdata.geometry import Pose6D
import numpy as np

# Create a Sample object
sample = Sample(x=1, y=2, z=3)

# Transform to different types
as_dict = sample.to("dict")
as_list = sample.to("list")
as_numpy = sample.to("np")
as_torch = sample.to("pt")

print("As dict:", as_dict)
print("As list:", as_list)
print("As numpy:", as_numpy)
print("As torch:", as_torch)

# Transform between different Sample subclasses
pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=0)
as_sample = pose.to(Sample)
print("Pose6D as Sample:", as_sample)
```

</details>

<details>
<summary><strong>Working with Trajectories</strong></summary>

```python
import numpy as np
from embdata.trajectory import Trajectory

# Create a simple 2D trajectory
steps = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
traj = Trajectory(steps, freq_hz=10, dim_labels=["X", "Y"])

# Plot the trajectory
traj.plot().show()

# Compute and print statistics
print(traj.stats())

# Apply a low-pass filter
filtered_traj = traj.low_pass_filter(cutoff_freq=2)
filtered_traj.plot().show()

# Resample the trajectory
upsampled_traj = traj.resample(target_hz=20)
print(upsampled_traj)
```

</details>

<details>
<summary><strong>Using Geometric Transformations</strong></summary>

```python
from embdata.geometry import Pose6D
import numpy as np

# Create a Pose6D object
pose = Pose6D(x=1, y=2, z=3, roll=np.pi/4, pitch=np.pi/3, yaw=np.pi/2)

# Convert to different representations
pose_cm = pose.to(unit="cm")
pose_deg = pose.to(angular_unit="deg")
quat = pose.to("quaternion")
rot_matrix = pose.to("rotation_matrix")

print("Pose in cm:", pose_cm)
print("Pose in degrees:", pose_deg)
print("Quaternion:", quat)
print("Rotation matrix:", rot_matrix)
```

</details>

These examples demonstrate some of the key features of the `embdata` library. For more detailed information, please refer to the API documentation.
