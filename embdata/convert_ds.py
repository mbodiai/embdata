"""This module provides utilities for converting and processing robotic arm datasets.

It includes functions for converting dictionary steps to VisionMotorStep objects,
creating VisionMotorEpisode objects, and processing trajectories.
"""

from typing import Dict, List

import numpy as np
from datasets import load_dataset

from embdata.describe import describe
from embdata.episode import Episode, ImageTask, VisionMotorEpisode, VisionMotorStep
from embdata.image import Image
from embdata.motion.control import RelativePoseHandControl


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
        ...         "image": {"bytes": b"complex_image_data_encoded_as_bytes"},
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
    return VisionMotorStep(
        step_idx=index,
        episode_idx=step["episode"],
        observation=ImageTask(
            image=Image(step["observation"]["image"]["bytes"]),
            task=step["observation"]["instruction"],
        ),
        relative_action=RelativePoseHandControl(**step["action"]),
    )


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
        ...         "observation": {"image": {"bytes": b"image_data_step1"}, "instruction": "Locate the blue sphere"},
        ...         "action": {"x": 0.1, "y": -0.2, "z": 0.0, "roll": 0, "pitch": 0, "yaw": 0, "gripper": 0.5},
        ...     },
        ...     {
        ...         "episode": 1,
        ...         "observation": {"image": {"bytes": b"image_data_step2"}, "instruction": "Move towards the blue sphere"},
        ...         "action": {"x": 0.2, "y": 0.1, "z": -0.1, "roll": 0.1, "pitch": 0, "yaw": -0.1, "gripper": 0.5},
        ...     },
        ...     {
        ...         "episode": 1,
        ...         "observation": {"image": {"bytes": b"image_data_step3"}, "instruction": "Grasp the blue sphere"},
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
    return VisionMotorEpisode(
        steps=[to_vision_motor_step(step, index=step_idx) for step_idx, step in enumerate(episode)],
    )


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
    """
    ds = load_dataset(dataset_name, split="train")
    episodes = []
    for episode_idx in range(num_episodes):
        episode_data = ds.filter(lambda x: x["episode"] == episode_idx).to_list()
        episode = to_vision_motor_episode(episode_data)
        episodes.append(episode)
        describe(episodes[-1].model_dump(), compact=True, show=True)
        episode.trajectory().make_absolute([0, 0, 0]).resample(10).make_relative().plot()
    return episodes


def main() -> None:
    """Main function to demonstrate the usage of the module's components.

    This function showcases how to use various components of the module,
    including loading and processing datasets, working with trajectories,
    and using the VisionMotorStep and VisionMotorEpisode classes.
    """
    #  Load and process dataset
    steps = load_dataset("mbodiai/oxe_bridge_v2", split="shard_0").take(100).to_list()
    describe(steps, compact=True, show=True)
    episode = Episode.from_list(steps)

    # Process trajectories
    e0 = episode.trajectory("action")
    e = episode.trajectory("action").make_absolute()
    a = episode.trajectory("absolute_action")
    s = episode.trajectory("state")

    describe({"e0": e0, "e": e, "a": a, "s": s}, compact=True, show=True)

    # Verify trajectory calculations
    for i, ss in enumerate(s):
        e00 = e0[i]
        aa = a[i]
        assert np.allclose(ss + e00[i + 1], s[i + 1]), f"Step {i} failed. {ss} + {e00} != {s[i + 1]}"
        assert np.allclose(ss + e00, aa), f"Step {i} failed. {ss} + {e00} != {aa}"

    # Composite example using multiple classes
    sample_step = {
        "episode": 1,
        "observation": {
            "image": {"bytes": b"complex_image_data_encoded_as_bytes"},
            "instruction": "Navigate around the obstacle and pick up the green cylinder",
        },
        "action": {
            "x": 0.15,
            "y": -0.25,
            "z": 0.1,
            "roll": 0.05,
            "pitch": -0.1,
            "yaw": 0.2,
            "gripper": 0.6,
        },
    }

    to_vision_motor_step(sample_step, index=0)

    sample_episode = [sample_step, sample_step]  # Using the same step twice for simplicity
    vision_motor_episode = to_vision_motor_episode(sample_episode)

    trajectory = vision_motor_episode.trajectory()

    absolute_trajectory = trajectory.make_absolute([0, 0, 0])

    resampled_trajectory = absolute_trajectory.resample(5)

    # Additional composite example
    processed_episodes = process_dataset("mbodiai/xarm_7_6_delta", num_episodes=1)
    example_episode = processed_episodes[0]

    # Demonstrate trajectory processing
    episode_trajectory = example_episode.trajectory()
    absolute_trajectory = episode_trajectory.make_absolute([0, 0, 0])
    resampled_trajectory = absolute_trajectory.resample(20)
    resampled_trajectory.make_relative()


if __name__ == "__main__":
    main()
