import pytest
import io
import os
from typing import List
from PIL import Image as PILModule
from embdata.coordinate import Pose
import numpy as np
from pydantic import Field
from embdata.episode import Episode, TimeStep, VisionMotorStep, ImageTask
from embdata.sample import Sample
from datasets import load_dataset
from embdata.sense.image import Image
from embdata.motion.control import AnyMotionControl, RelativePoseHandControl
from mbench.profile import profileme


profileme()

@pytest.fixture
def time_step():
    return TimeStep(observation=Sample("observation"), action=Sample(1), supervision=Sample("supervision"))

@pytest.mark.network
def test_episode_push_to_hub(time_step):
    episode = Episode(steps=[time_step, time_step, time_step], freq_hz=0.2)
    episode.dataset().push_to_hub("mbodiai/episode_test", private=True)

@pytest.mark.network
def test_episode_from_steps_image():
    steps = [
        {
            "observation": {
                "image": Image(array=np.zeros((224, 224, 3), dtype=np.uint8), dtype=np.uint8),
                "task": "command",
            },
            "action": AnyMotionControl(joints=[0.5, 3.3]).dict(),
            "state": {"joint": [0.5, 3.3]},
        },
        {
            "observation": {
                "image": Image(array=np.zeros((224, 224, 3), dtype=np.uint8), dtype=np.uint8),
                "task": "command",
            },
            "action": AnyMotionControl(joints=[0.5, 3.3]).dict(),
            "state": {"joint": [0.5, 3.3]},
        },
        {
            "observation": {
                "image": Image(array=np.zeros((224, 224, 3), dtype=np.uint8), dtype=np.uint8),
                "task": "command",
            },
            "action": AnyMotionControl(joints=[0.5, 3.3]).dict(),
            "state": {"joint": [0.5, 3.3]},
        },
    ]

    episode = Episode(steps)
    episode.dataset().push_to_hub("mbodiai/episode_testing3", private=True, token=os.getenv("HF_TOKEN"))
    assert len(episode.steps) == 3

@pytest.mark.network
def test_episode_push_real_data(time_step):
    from embdata.episode import Episode, VisionMotorStep, ImageTask
    from embdata.motion.control import MobileSingleHandControl, Pose, PlanarPose, HandControl

    buffer = io.BytesIO()
    img = PILModule.new("RGB", (224, 224), (255, 0, 0))
    img.save(buffer, format="JPEG")
    obs = ImageTask(image={"bytes": buffer}, task="command")
    act = MobileSingleHandControl(
        base=PlanarPose(x=0.1, y=0.2, theta=0.3), hand=HandControl([0, 1, 2, 3, 4, 5, 0.1]), head=[0.1, 0.2]
    )
    state = Pose.unflatten(np.zeros(6))
    episode = Episode(steps=[VisionMotorStep(observation=obs, action=act, state=state) for _ in range(10)], freq_hz=5)

    episode.dataset().push_to_hub("mbodiai/episode_test22", private=True)

@pytest.mark.network
def test_episode_vision_motor_step_dataset():
    episode = Episode([])
    episode.append(
        VisionMotorStep(
            episode_idx=0,
            step_idx=0,
            observation=ImageTask(image=Image(size=(224, 224)), task="task"),
            action=RelativePoseHandControl(),
            state=Sample(),
        )
    )
    episode.dataset()

@pytest.mark.network
def test_episode_vision_motor_step_idx_dataset():
    episode = Episode([])
    episode.append(
        VisionMotorStep(
            episode_idx=0,
            step_idx=0,
            observation=ImageTask(image=Image(size=(224, 224)), task="task"),
            action=RelativePoseHandControl(),
            state=Sample(),
        )
    )
    episode.append(
        VisionMotorStep(
            episode_idx=1,
            step_idx=1,
            observation=ImageTask(image=Image(size=(224, 224)), task="task"),
            action=RelativePoseHandControl(),
            state=Sample(),
        )
    )
    ds = episode.dataset()
    assert ds[0]["episode_idx"] == 0
    assert ds[1]["episode_idx"] == 1
    assert ds[0]["step_idx"] == 0
    assert ds[1]["step_idx"] == 1

@pytest.mark.network
def test_dataset_to_episode(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode(steps=dataset.to_list())

@pytest.mark.network
def test_episode_from_dataset(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode.from_dataset(dataset)

@pytest.mark.network
def test_episode_from_list(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode.from_list(dataset.to_list(), observation_key="observation", action_key="action")

@pytest.mark.network
def test_object_scene(time_step):
    # ds = load_dataset("mbodiai/new_ds", split="train")
    # features = ds.features
    from embdata.describe import describe
    from datasets import concatenate_datasets
    from embdata.episode import Episode, TimeStep, VisionMotorEpisode, ImageTask

    episode = VisionMotorEpisode(steps=[])

    class SceneObject(Sample):
        """Model for Scene Object Poses."""

        object_name: str = ""
        object_pose: Pose = Field(default_factory=Pose, description="Object Pose")

    class SceneData(Sample):
        """Model for Scene Data."""

        image: Image
        depth_image: Image
        scene_objects: List[SceneObject] = Field(
            default_factory=lambda: [SceneObject()], description="List of Scene Objects"
        )

    episode.append(
        VisionMotorStep(
            episode_idx=0,
            step_idx=0,
            observation=ImageTask(image=Image(size=(224, 224)), task="task"),
            action=RelativePoseHandControl(),
            state=SceneData(
                image=Image(size=(224, 224)),
                depth_image=Image(size=(224, 224)),
                scene_objects=[
                    SceneObject(
                        object_name="object1", object_pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3)
                    ),
                    SceneObject(
                        object_name="object2", object_pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.21, pitch=0.2, yaw=0.3)
                    ),
                ],
            ),
        )
    )
    new_ds = episode.dataset()
    # print(f"New Features: ")
    from rich.pretty import pprint

    # pprint(new_ds.features)
    new_ds.push_to_hub("mbodiai/test_randss", private=True)
    # describe(new_ds.features)
    new_new_ds = load_dataset("mbodiai/test_randss", split="train")
    new_new_features = new_new_ds.features
    # print(f"New new Features:")
    # print_json(data=new_new_features)

    # describe(features)
    ds = concatenate_datasets([new_new_ds, episode.dataset()])

