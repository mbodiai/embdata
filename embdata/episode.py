import atexit
import logging
import sys
from itertools import zip_longest
from threading import Thread
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal

import numpy as np
import rerun as rr
import torch
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from rerun.archetypes import Image as RRImage
from rich import print_json
from torchvision import transforms

from embdata.describe import describe
from embdata.motion import Motion
from embdata.motion.control import AnyMotionControl, RelativePoseHandControl
from embdata.sample import Sample
from embdata.sense.image import Image, SupportsImage
from embdata.trajectory import Trajectory
from embdata.utils.iter_utils import get_iter_class

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch
except ImportError:
    logging.warning("lerobot not found. Episode.lerobot() may not work.")



def convert_images(values: Dict[str, Any] | Any, image_keys: set[str] | str | None = "image") -> "TimeStep":
        if not isinstance(values, dict | Sample):
            if Image.supports(values):
                try:
                    return Image(values)
                except Exception:
                    return values
            return values
        if isinstance(image_keys, str):
            image_keys = {image_keys}
        obj = {}
        for key, value in values.items():
            if key in image_keys:
                try :
                    if isinstance(value, dict):
                        obj[key] = Image(**value)
                    else:
                        obj[key] = Image(value)
                except Exception as e: # noqa
                    logging.warning(f"Failed to convert {key} to Image: {e}")
                    obj[key] = value
            elif isinstance(value, dict | Sample):
                obj[key] = convert_images(value)
            else:
                obj[key] = value
        return obj

class TimeStep(Sample):
    """Time step for a task."""

    episode_idx: int | None = 0
    step_idx: int | None = 0
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    observation: Sample | None = None
    action: Sample | None = None
    state: Sample | None = None
    supervision: Any = None
    timestamp: float | None = None

    _observation_class: type[Sample] = PrivateAttr(default=Sample)
    _action_class: type[Sample] = PrivateAttr(default=Sample)
    _state_class: type[Sample] = PrivateAttr(default=Sample)

    @classmethod
    def from_dicts(cls, values: Dict[str, Any], image_keys: str | set | None = "image") -> "TimeStep":
        obs = values.get("observation")
        if not isinstance(obs, Sample):
            obs = Sample(obs)
        act = values.get("action")
        if not isinstance(act, Sample):
            act = Sample(act)
        sta = values.get("state")
        if not isinstance(sta, Sample):
            sta = Sample(sta)
        return cls(
            observation=obs, 
            action=act,
            state=sta, 
            supervision=values.get("supervision"),
            **{k: v for k, v in values.items() if k not in ["observation", "action", "state", "supervision"]},
        )

    def __init__(
        self,
        observation: Sample | Dict | np.ndarray, 
        action: Sample | Dict | np.ndarray,
        state: Sample | Dict | np.ndarray | None = None, 
        supervision: Any | None = None, 
        episode_idx: int = 0, step_idx: int = 0, 
        timestamp: float | None = None, **kwargs) -> None:
        Obs = self.__class__._observation_class.get_default()  # noqa: N806, SLF001
        Act = self.__class__._action_class.get_default()  # noqa: N806, SLF001
        Sta = self.__class__._state_class.get_default()  # noqa: N806, SLF001

        observation = Obs(**convert_images(observation)) if observation is not None else None
        action = Act(**convert_images(action)) if action is not None else None
        state = Sta(**convert_images(state)) if state is not None else None
        supervision = convert_images(supervision) if supervision is not None else None
        super().__init__(observation=observation, action=action, state=state, supervision=supervision, episode_idx=episode_idx, step_idx=step_idx, timestamp=timestamp, **kwargs)


class ImageTask(Sample):
    """Canonical Observation."""
    image: Image
    task: str

    def __init__(self, image: Image | SupportsImage, task: str) -> None:
        super().__init__(image=image, task=task)
class VisionMotorStep(TimeStep):
    """Time step for vision-motor tasks."""
    _observation_class: type[ImageTask] = PrivateAttr(default=ImageTask)
    observation: ImageTask
    action: Motion
    supervision: Any | None = None


class Episode(Sample):
    """A list-like interface for a sequence of observations, actions, and/or other data."""

    episode_id: str | int | None = None
    steps: list[TimeStep] = Field(default_factory=list)
    metadata: Sample | Any | None = None
    freq_hz: int | None = None
    _action_class: type[Sample] = PrivateAttr(default=Sample)
    _state_class: type[Sample] = PrivateAttr(default=Sample)
    _observation_class: type[Sample] = PrivateAttr(default=Sample)
    _step_class: type[TimeStep] = PrivateAttr(default=TimeStep)
    _image_keys: str | set[str] = PrivateAttr(default="image")
    _rr_thread: Thread | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_classes(self) -> "Episode":
        self._observation_class = get_iter_class("observation", self.steps)
        self._action_class = get_iter_class("action", self.steps)
        self._state_class = get_iter_class("state", self.steps)
        return self

    @staticmethod
    def concat(episodes: List["Episode"]) -> "Episode":
        return sum(episodes, Episode(steps=[]))

    @classmethod
    def from_observations_actions(cls, observations: List[Sample], actions: List[Sample]) -> "Episode":
        steps = [TimeStep(observation=o, action=a) for o, a in zip(observations, actions)]
        return cls(steps=steps)

    def __init__(
        self,
        steps: List[Dict | Sample | TimeStep] | Iterable,
        observation_key: str = "observation",
        action_key: str = "action",
        supervision_key: str | None = "supervision",
        metadata: Sample | Any | None = None,
        freq_hz: int | None = None,
    ) -> None:
        if not hasattr(steps, "__iter__"):
            msg = "Steps must be an iterable"
            raise ValueError(msg)
        steps = list(steps)

        Step = self.__class__._step_class.get_default()  # noqa: N806, SLF001

        if steps and not isinstance(steps[0], TimeStep):
            if isinstance(steps[0], dict) and observation_key in steps[0] and action_key in steps[0]:
                steps = [Step(observation=step[observation_key], action=step[action_key], supervision=step.get(supervision_key)) for step in steps]
            elif isinstance(steps[0], tuple):
                steps = [Step(observation=step[0], action=step[1], supervision=step[2] if len(step) > 2 else None) for step in steps]
            else:
                steps = [Step.from_dicts(step) if isinstance(step, dict) else Step(**step) for step in steps]

        super().__init__(steps=steps, metadata=metadata)
        self.freq_hz = freq_hz

    @classmethod
    def from_list(cls, steps: List[Dict], observation_key: str, action_key: str, supervision_key: str | None = None):
        Step = cls._step_class.get_default()
        processed_steps = [
            Step(
                observation=step[observation_key],
                action=step[action_key],
                supervision=step.get(supervision_key) if supervision_key else None
            )
            for step in steps
        ]
        return cls(steps=processed_steps)

    def __repr__(self) -> str:
        if not hasattr(self, "stats"):
            self.stats = self.trajectory().stats()
            stats = str(self.stats).replace("\n ", "\n  ")
        return f"Episode(\n  {stats}\n)"

    def trajectory(self, field: str = "action", freq_hz: int = 1) -> Trajectory:
        freq_hz = freq_hz or self.freq_hz or 1
        data = [getattr(step, field) for step in self.steps]
        if isinstance(data[0], Sample):
            data = [d.numpy() for d in data]
        return Trajectory(
            data,
            freq_hz=freq_hz,
            dim_labels=list(data[0].keys()) if isinstance(data[0], dict) else None,
            episode=self,
        )

    def __len__(self) -> int:
        """Get the number of steps in the episode.

        Returns:
            int: The number of steps in the episode.
        """
        return len(self.steps)

    def __getitem__(self, idx) -> TimeStep:
        """Get the step at the specified index.

        Args:
            idx: The index of the step.

        Returns:
            TimeStep: The step at the specified index.
        """
        return self.steps[idx]

    def __setitem__(self, idx, value) -> None:
        """Set the step at the specified index.

        Args:
            idx: The index of the step.
            value: The value to set.
        """
        self.steps[idx] = value

    def __iter__(self) -> Any:
        """Iterate over the episode.

        Returns:
            Any: An iterator over the episode.
        """
        logging.warning(
            "Iterating over an Episode will iterate over keys. Use the `iter()` method to iterate over the steps.",
        )
        return super().__iter__()

    def map(self, func: Callable[[TimeStep | Dict | np.ndarray],np.ndarray | TimeStep], field=None) -> "Episode":
        """Apply a function to each step in the episode.

        Args:
            func (Callable[[TimeStep], TimeStep]): The function to apply to each step.
            field (str, optional): The field to apply the function to. Defaults to None.

        Returns:
            'Episode': The modified episode.

        Example:
            >>> episode = Episode(
            ...     steps=[
            ...         TimeStep(observation=Sample(value=1), action=Sample(value=10)),
            ...         TimeStep(observation=Sample(value=2), action=Sample(value=20)),
            ...         TimeStep(observation=Sample(value=3), action=Sample(value=30)),
            ...     ]
            ... )
            >>> episode.map(lambda step: TimeStep(observation=Sample(value=step.observation.value * 2), action=step.action))
            Episode(steps=[
              TimeStep(observation=Sample(value=2), action=Sample(value=10)),
              TimeStep(observation=Sample(value=4), action=Sample(value=20)),
              TimeStep(observation=Sample(value=6), action=Sample(value=30))
            ])
        """
        if field is not None:
            return self.trajectory(field=field).map(func).episode()
        return Episode(steps=[func(step) for step in self.steps])

    def filter(self, condition: Callable[[TimeStep], bool]) -> "Episode":
        """Filter the steps in the episode based on a condition.

        Args:
            condition (Callable[[TimeStep], bool]): A function that takes a time step and returns a boolean.

        Returns:
            'Episode': The filtered episode.

        Example:
            >>> episode = Episode(
            ...     steps=[
            ...         TimeStep(observation=Sample(value=1), action=Sample(value=10)),
            ...         TimeStep(observation=Sample(value=2), action=Sample(value=20)),
            ...         TimeStep(observation=Sample(value=3), action=Sample(value=30)),
            ...     ]
            ... )
            >>> episode.filter(lambda step: step.observation.value > 1)
            Episode(steps=[
              TimeStep(observation=Sample(value=2), action=Sample(value=20)),
              TimeStep(observation=Sample(value=3), action=Sample(value=30))
            ])
        """
        return Episode(steps=[step for step in self.steps if condition(step)])

    def iter(self) -> Iterator[TimeStep]:
        """Iterate over the steps in the episode.

        Returns:
            Iterator[TimeStep]: An iterator over the steps in the episode.
        """
        return iter(self.steps)

    def __add__(self, other) -> "Episode":
        """Append episodes from another Episode.

        Args:
            other ('Episode'): The episode to append.

        Returns:
            'Episode': The combined episode.
        """
        if isinstance(other, Episode):
            self.steps += other.steps
        else:
            msg = "Can only add another Episode"
            raise TypeError(msg)
        return self

    def __truediv__(self, field: str) -> "Episode":
        """Group the steps in the episode by a key."""
        return self.group_by(field)

    def append(self, step: TimeStep) -> None:
        """Append a time step to the episode.

        Args:
            step (TimeStep): The time step to append.
        """
        self.steps.append(step)

    def split(self, condition: Callable[[TimeStep], bool]) -> list["Episode"]:
        """Split the episode into multiple episodes based on a condition.

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
            >>> episode = Episode(
            ...     steps=[
            ...         TimeStep(observation=Sample(value=5)),
            ...         TimeStep(observation=Sample(value=10)),
            ...         TimeStep(observation=Sample(value=15)),
            ...         TimeStep(observation=Sample(value=8)),
            ...         TimeStep(observation=Sample(value=20)),
            ...     ]
            ... )
            >>> episodes = episode.split(lambda step: step.observation.value <= 10)
            >>> len(episodes)
            3
            >>> [len(ep) for ep in episodes]
            [2, 1, 2]
            >>> [[step.observation.value for step in ep.iter()] for ep in episodes]
            [[5, 10], [15], [8, 20]]
        """
        episodes = []
        current_episode = Episode(steps=[])
        steps = iter(self.steps)
        current_episode_meets_condition = True
        for step in steps:
            if condition(step) != current_episode_meets_condition:
                episodes.append(current_episode)
                current_episode = Episode(steps=[])
                current_episode_meets_condition = not current_episode_meets_condition
            current_episode.steps.append(step)
        episodes.append(current_episode)
        return episodes

    def group_by(self, key: str) -> Dict:
        """Group the steps in the episode by a key.

        Args:
            key (str): The key to group by.

        Returns:
            Dict: A dictionary of lists of steps grouped by the key.

        Example:
            >>> episode = Episode(
            ...     steps=[
            ...         TimeStep(observation=Sample(value=5), action=Sample(value=10)),
            ...         TimeStep(observation=Sample(value=10), action=Sample(value=20)),
            ...         TimeStep(observation=Sample(value=5), action=Sample(value=30)),
            ...         TimeStep(observation=Sample(value=10), action=Sample(value=40)),
            ...     ]
            ... )
            >>> groups = episode.group_by("observation")
            >>> groups
            {'5': [TimeStep(observation=Sample(value=5), action=Sample(value=10)), TimeStep(observation=Sample(value=5), action=Sample(value=30)], '10': [TimeStep(observation=Sample(value=10), action=Sample(value=20)), TimeStep(observation=Sample(value=10), action=Sample(value=40)]}
        """
        groups = {}
        for step in self.steps:
            key_value = step[key]
            if key_value not in groups:
                groups[key_value] = []
            groups[key_value].append(step)
        return groups

    def dataset(self) -> Dataset:
        if self.steps is None or len(self.steps) == 0:
            msg = "Episode has no steps"
            raise ValueError(msg)
        data = [step.dump(as_field="pil") for step in self.steps]
        features = self.steps[0].features()
        return Dataset.from_list(data, features=features)

    def lerobot(self) -> "LeRobotDataset":
        """Convert the episode to LeRobotDataset compatible format.

        Refer to https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/push_dataset_to_hub.py for more details.

        Args:
            fps (int, optional): The frames per second for the episode. Defaults to 1.

        Returns:
            LeRobotDataset: The LeRobotDataset dataset.
        """
        data_dict = {
            "observation.image": [],
            "observation.state": [],
            "action": [],
            "episode_index": [],
            "frame_index": [],
            "timestamp": [],
            "next.done": [],
        }

        for i, step in enumerate(self.steps):
            data_dict["observation.image"].append(Image(step.observation.image).pil)
            data_dict["observation.state"].append(step.state.torch())
            data_dict["action"].append(step.action.torch())
            data_dict["episode_index"].append(torch.tensor(step.episode_idx, dtype=torch.int64))
            data_dict["frame_index"].append(torch.tensor(step.step_idx, dtype=torch.int64))
            fps = self.freq_hz if self.freq_hz is not None else 1
            data_dict["timestamp"].append(torch.tensor(i / fps, dtype=torch.float32))
            data_dict["next.done"].append(torch.tensor(i == len(self.steps) - 1, dtype=torch.bool))
        data_dict["index"] = torch.arange(0, len(self.steps), 1)

        features = Features(
            {
                "observation.image": HFImage(),
                "observation.state": Sequence(feature=Value(dtype="float32")),
                "action": Sequence(feature=Value(dtype="float32")),
                "episode_index": Value(dtype="int64"),
                "frame_index": Value(dtype="int64"),
                "timestamp": Value(dtype="float32"),
                "next.done": Value(dtype="bool"),
                "index": Value(dtype="int64"),
            },
        )

        hf_dataset = Dataset.from_dict(data_dict, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        episode_data_index = calculate_episode_data_index(hf_dataset)
        info = {
            "fps": fps,
            "video": False,
        }
        return LeRobotDataset.from_preloaded(
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
        )

    @classmethod
    def from_lerobot(cls, lerobot_dataset: "LeRobotDataset") -> "Episode":
        """Convert a LeRobotDataset compatible dataset back into an Episode.

        Args:
            lerobot_dataset: The LeRobotDataset dataset to convert.

        Returns:
            Episode: The reconstructed Episode.
        """
        steps = []
        dataset = lerobot_dataset.hf_dataset
        for _, data in enumerate(dataset):
            image = Image(transforms.ToPILImage()(data["observation.image"]))
            state = Sample(data["observation.state"])
            action = Sample(data["action"])
            observation = Sample(image=image, task=None)
            step = TimeStep(
                episode_idx=data["episode_index"],
                step_idx=data["frame_index"],
                observation=observation,
                action=action,
                state=state,
                supervision=None,
            )
            steps.append(step)
        return cls(
            steps=steps,
            freq_hz=lerobot_dataset.fps,
        )

    def rerun(self, mode=Literal["local", "remote"], port=5003, ws_port=5004) -> "Episode":
        """Start a rerun server."""
        rr.init("rerun-mbodied-data", spawn=mode == "local")
        blueprint = rr.blueprint.Blueprint(
            rr.blueprint.Spatial3DView(background=RRImage(Image(size=(224, 224, 3)).array)),
            auto_layout=True, auto_space_views=True)
        rr.serve(open_browser=False, web_port=port, ws_port=ws_port, default_blueprint=blueprint)
        for i, step in enumerate(self.steps):
            if not hasattr(step, "timestamp") or step.timestamp is None:
                step.timestamp = i / 5
            rr.set_time_sequence("frame_index", i)
            rr.set_time_seconds("timestamp", step.timestamp)
            rr.log("observation", RRImage(step.observation.image)) if step.observation.image else None
            rr.send_blueprint(rr.blueprint.Blueprint(
            rr.blueprint.Spatial3DView(background=RRImage(step.observation.image.array)),
            auto_layout=True, auto_space_views=True))
            for dim, val in step.action.flatten("dict").items():
                rr.log(f"action/{dim}", rr.Scalar(val))
            if step.action.flatten(to="pose"):
                origin = step.state.numpy()[:3] if step.state else [0, 0, 0]
                direction = step.action.numpy()[:3]
                rr.log("action/pose_arrow", rr.Arrows3D(vectors=[direction], origins=[origin]))

            try:
                while hasattr(self, "_rr_thread") and self._rr_thread.is_alive():
                    pass
            except KeyboardInterrupt:
                self.close_view()
                sys.exit()

    def show(self, mode: Literal["local", "remote"] | None = None, port=5003) -> None:
        if mode is None:
            msg = "Please specify a mode: 'local' or 'remote'"
            raise ValueError(msg)
        thread = Thread(target=self.rerun, kwargs={"port": port, "mode": mode})
        self._rr_thread = thread
        atexit.register(self.close_view)
        thread.start()

    def close_view(self) -> None:
        if hasattr(self, "_rr_thread"):
            self._rr_thread.join()
        self._rr_thread = None


class VisionMotorEpisode(Episode):
    """An episode for vision-motor tasks."""
    _step_class: type[VisionMotorStep] = PrivateAttr(default=VisionMotorStep)
    _observation_class: type[ImageTask] = PrivateAttr(default=ImageTask)
    _action_class: type[AnyMotionControl] = PrivateAttr(default=AnyMotionControl)
    steps: list[VisionMotorStep]


class VisionMotorHandEpisode(VisionMotorEpisode):
    """An episode for vision-motor tasks with hand control."""
    _action_class: type[RelativePoseHandControl] = PrivateAttr(default=RelativePoseHandControl)

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
