import io
import os
from typing import List
from PIL import Image as PILModule
from embdata.geometry import Pose
from embdata.sense.depth import Depth
import numpy as np
from pydantic import Field
import pytest
from embdata.episode import Episode, TimeStep, VisionMotorStep, ImageTask
from embdata.sample import Sample
from datasets import load_dataset
from embdata.sense.image import Image
from embdata.motion.control import AnyMotionControl, RelativePoseHandControl


@pytest.fixture
def time_step():
    return TimeStep(observation=Sample("observation"), action=Sample("action"), supervision=Sample("supervision"))


def test_episode_initialization(time_step):
    episode = Episode(steps=[time_step])
    assert len(episode) == 1
    assert episode[0] == time_step


def test_episode_length(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    assert len(episode) == 3


def test_episode_get_item(time_step):
    episode = Episode(steps=[time_step])
    assert episode[0] == time_step


def test_episode_set_item(time_step):
    time_step2 = TimeStep(observation=Sample("observation"), action=Sample("action"), supervision=Sample("supervision"))
    episode = Episode(steps=[time_step])
    episode[0] = time_step2
    assert episode[0] == time_step2


def test_episode_iteration(time_step):
    episode = Episode(steps=[time_step, time_step])
    for i, step in enumerate(episode.iter()):
        assert step == episode[i]


def test_episode_addition(time_step):
    episode1 = Episode(steps=[time_step])
    episode2 = Episode(steps=[time_step, time_step])
    combined_episode = episode1 + episode2
    assert len(combined_episode) == 3


def test_episode_append(time_step):
    episode = Episode(steps=[])
    episode.append(time_step)
    assert len(episode) == 1
    assert episode[0] == time_step


def test_episode_split(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    episodes = episode.split(lambda step: False)
    assert len(episodes) == 2
    assert len(episodes[0]) == 0
    assert len(episodes[1]) == 3


def test_unpacked_episode(time_step):
    steps = [time_step, time_step, time_step]
    episode = Sample.unpack_from(steps)
    observations, actions, supervisions = episode.observation, episode.action, episode.supervision
    assert len(observations) == 3
    assert len(actions) == 3
    assert len(supervisions) == 3
    assert all(isinstance(observation, Sample) for observation in observations)
    assert all(isinstance(action, Sample) for action in actions)
    assert all(isinstance(supervision, Sample) for supervision in supervisions)


def test_episode_concatenate(time_step):
    episode1 = Episode(steps=[time_step, time_step])
    episode2 = Episode(steps=[time_step, time_step])
    episode3 = Episode(steps=[time_step, time_step])
    concatenated_episode = Episode.concat([episode1, episode2, episode3])
    assert len(concatenated_episode) == 6


def test_episode_from_lists(time_step):
    observations = [Sample("observation1"), Sample("observation2")]
    actions = [Sample("action1"), Sample("action2")]
    episode = Episode.from_lists(observations, actions)
    assert len(episode) == 2
    assert episode[0].observation == observations[0]
    assert episode[0].action == actions[0]
    assert episode[1].observation == observations[1]
    assert episode[1].action == actions[1]


def test_episode_from_list(time_step):
    steps = [
        {"observation": Sample("observation1"), "action": Sample("action1"), "supervision": Sample("supervision1")},
        {"observation": Sample("observation2"), "action": Sample("action2"), "supervision": Sample("supervision2")},
    ]
    episode = Episode.from_list(steps, "observation", "action", "supervision")
    assert len(episode) == 2
    assert episode[0].observation == steps[0]["observation"]
    assert episode[0].action == steps[0]["action"]
    assert episode[0].supervision == steps[0]["supervision"]
    assert episode[1].observation == steps[1]["observation"]
    assert episode[1].action == steps[1]["action"]
    assert episode[1].supervision == steps[1]["supervision"]


def test_episode_trajectory(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    trajectory = episode.trajectory("action", freq_hz=1)
    assert len(trajectory) == 3


def test_episode_append(time_step):
    episode = Episode(steps=[])
    episode.append(time_step)
    assert len(episode) == 1
    assert episode[0] == time_step


def test_episode_split(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    episodes = episode.split(lambda step: False)
    assert len(episodes) == 2
    assert len(episodes[0]) == 0
    assert len(episodes[1]) == 3


def test_episode_iteration(time_step):
    episode = Episode(steps=[time_step, time_step])
    for i, step in enumerate(episode.iter()):
        assert step == episode[i]


def test_episode_addition(time_step):
    episode1 = Episode(steps=[time_step])
    episode2 = Episode(steps=[time_step, time_step])
    combined_episode = episode1 + episode2
    assert len(combined_episode) == 3


def test_episode_get_item(time_step):
    episode = Episode(steps=[time_step])
    assert episode[0] == time_step


def test_episode_set_item(time_step):
    time_step2 = TimeStep(observation=Sample("observation"), action=Sample("action"), supervision=Sample("supervision"))
    episode = Episode(steps=[time_step])
    episode[0] = time_step2
    assert episode[0] == time_step2


def test_episode_push_to_hub(time_step):
    episode = Episode(steps=[time_step, time_step, time_step], freq_hz=0.2)
    episode.dataset().push_to_hub("mbodiai/episode_test", private=True)


def test_episode_from_ds(time_step):
    ds = load_dataset("mbodiai/test_dumb", split="train").to_list()
    episode = Episode(steps=ds)
    assert len(episode.steps) == len(ds)


def test_episode_from_zipped_ds(time_step):
    obs = [Sample("observation1"), Sample("observation2")]
    act = [Sample("action1"), Sample("action2")]
    sup = [Sample("supervision1"), Sample("supervision2")]

    episode = Episode(zip(obs, act, sup))
    assert len(episode.steps) == len(obs)


def test_episode_from_steps_image(time_step):
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


def test_dataset_to_episode(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode(steps=dataset.to_list())


def test_episode_from_dataset(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode.from_dataset(dataset)


def test_episode_from_list(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode.from_list(dataset.to_list(), observation_key="observation", action_key="action")


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
        depth_image: Depth
        scene_objects: List[SceneObject] = Field(default_factory=lambda: [SceneObject()], description="List of Scene Objects")

    episode.append(
        VisionMotorStep(
            episode_idx=0,
            step_idx=0,
            observation=ImageTask(image=Image(size=(224, 224)), task="task"),
            action=RelativePoseHandControl(),
            state=SceneData(image=Image(size=(224, 224), mode="RGB"), depth_image=Depth(size=(224, 224), mode="I"), scene_objects=[
                SceneObject(object_name="object1", object_pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3)),
                SceneObject(object_name="object2", object_pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.21, pitch=0.2, yaw=0.3)),
            ]),
        )
    )
    new_ds = episode.dataset()
    print(f"New Features: ")
    from rich import print_json
    # print_json(data=new_ds.features)
    new_ds.push_to_hub("mbodiai/test_randss", private=True)
    describe(new_ds.features)
    new_new_ds = load_dataset("mbodiai/test_randss", split="train")
    new_new_features = new_new_ds.features
    # print(f"New new Features:")
    # print_json(data=new_new_features)

    
        
def test_concatenate():
    ds = load_dataset("mbodiai/testinglatest", split="train").to_list()
    print(len(ds))
    for item in ds:
        print(item["action"]["pose"])
        print(item["action"]["grasp"])
    assert False
    # describe(features)
    # ds = concatenate_datasets([new_new_ds, episode.dataset()])

if __name__ == "__main__":
    pytest.main(["-vv", __file__])
