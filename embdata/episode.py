import atexit
from itertools import zip_longest
import logging
import sys
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
from embdata.motion import AnyMotionControl, Motion
from embdata.motion.control import RelativePoseHandControl
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

    episode_idx: int = 0
    step_idx: int = 0
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
        obs = step[observation_key] if observation_key else None
        if not isinstance(obs, Sample):
            obs = Obs(obs)
        act = step[action_key] if action_key else None
        if not isinstance(act, Sample):
            act = Act(act)
        sta = step[state_key] if state_key else None
        if not isinstance(sta, Sample):
            sta = State(sta)
        return Step(
            observation=obs, 
            action=act,
            state=sta, 
            supervision=step[supervision_key] if supervision_key else None,
            **{k: step[k] for k in step if k not in [observation_key, action_key, supervision_key, state_key]},
        )

    def __init__(
        self,
        observation: Sample | Dict | np.ndarray, 
        action: Sample | Dict | np.ndarray,
        state: Sample | Dict | np.ndarray | None = None, 
        supervision: Any | None = None, 
        episode_idx: int = 0, step_idx: int = 0, 
        timestamp: float | None = None, **kwargs) -> None:

        
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
    """A list-like interface for a sequence of observations, actions, and/or other data.

    This class is designed to streamline exploratory data analysis and manipulation of time series data.
    It provides methods for appending, iterating, concatenating, and analyzing episodes.

    Attributes:
        steps (list[TimeStep]): A list of TimeStep objects representing the episode's steps.
        metadata (Sample | Any | None): Additional metadata for the episode.
        freq_hz (int | None): The frequency of the episode in Hz.

    Example:
        >>> from embdata.sense.image import Image
        >>> from embdata.motion import Motion
        >>> steps = [
        ...     VisionMotorStep(
        ...         observation=ImageTask(image=Image((224, 224, 3)), task="grasp"),
        ...         action=Motion(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
        ...     ),
        ...     VisionMotorStep(
        ...         observation=ImageTask(image=Image((224, 224, 3)), task="lift"),
        ...         action=Motion(position=[0.2, 0.3, 0.4], orientation=[0, 0, 1, 0]),
        ...     ),
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
        """Validate the action class based on the first step in the episode.

        Args:
            values (Dict[str, Any]): The values of the episode.

        Returns:
            Dict[str, Any]: The validated values of the episode.
        """
        self._observation_class = get_iter_class("observation", self.steps)
        self._action_class = get_iter_class("action", self.steps)
        self._state_class = get_iter_class("state", self.steps)

        return self


    @staticmethod
    def concat(episodes: List["Episode"]) -> "Episode":
        """Concatenate a list of episodes into a single episode.

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
        """
        return sum(episodes, Episode(steps=[]))

    @classmethod
    def pack_from(cls, observations: List[Sample] | None = None, actions: List[Sample] | None = None,
                       states: List[Sample] | None = None) -> "Episode":
        """Create an episode from a list of observations and actions.

        Args:
            observations (List[Sample]): The list of observations.
            actions (List[Sample]): The list of actions.

        Returns:
            'Episode': The created episode.

        Example:
            >>> observations = [Sample(observation=Image((224, 224))), Sample(observation=Image((224, 224)))]
            >>> actions = [Sample(action=Motion()), Sample(action=Motion())]
            >>> episode = Episode.from_observations_actions(observations, actions)
        """
        return cls(steps=super().pack_from(observations=observations, actions=actions, states=states).steps)

    def __init__(
        self,
        steps: List[Dict | Sample] | Iterable,
        observation_key: str = "observation",
        action_key: str = "action",
        state_key: str | None = None,
        supervision_key: str | None = None,
        metadata: Sample | Any | None = None,
        freq_hz: int | None = None,
    ) -> None:
        """Create an episode from a list of dicts, samples, time steps or any iterable with at least two items per step.

        Args:
            steps (List[Dict|Sample]): The list of dictionaries representing the steps.
            action_key (str): The key for the action in each dictionary.
            observation_key (str): The key for the observation in each dictionary.
            state_key (str, optional): The key for the state in each dictionary. Defaults to None.
            supervision_key (str, optional): The key for the supervision in each dictionary. Defaults to None.
            metadata (Sample|Any, optional): Additional metadata for the episode. Defaults to None.
            freq_hz (int, optional): The frequency of the episode in Hz. Defaults to None.

        Returns:
            'Episode': The created episode.

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
        if not hasattr(steps, "__iter__"):
            msg = "Steps must be an iterable"
            raise ValueError(msg)
        if not hasattr(steps, "__getitem__"):
            steps = list(steps)

        Obs = self.__class__._observation_class.get_default() # noqa
        Act = self.__class__._action_class.get_default() # noqa
        State = self.__class__._state_class.get_default() # noqa
        Step = self.__class__._step_class.get_default() # noqa
        supervision_key = supervision_key or "supervision"
        action_key = action_key or "action"
        observation_key = observation_key or "observation"
        state_key = state_key or "state"
        if steps and not isinstance(steps[0], Dict | Sample) and not isinstance(steps[0], TimeStep):
            logging.debug("Converting not mapping steps to TimeStep")
            steps = [
              Step(
                observation=step[0],
                action=step[1],
                 state=step[2] if len(step) > 2 else None, 
                 supervision=step[3] if len(step) > 3 else None)
                for step in steps
            ]
        elif isinstance(steps[0], Dict | Sample) and not isinstance(steps[0], TimeStep):
            logging.debug("Converting dict or sample steps to TimeStep")
            print(f"a: {action_key}, o: {observation_key}, s: {state_key}, sup: {supervision_key}")
            print(f"Observation: {Obs}, Action: {Act}, State: {State}, Step: {Step}")
            print(f"steps[0]: {steps[0]}")

            steps = [Step.from_dicts(step) for step in steps]
        print(f"steps: {steps}")
        try:
            super().__init__(
                steps=steps,
                metadata=metadata,
            )
        except KeyError as e:
            raise KeyError(f"Missing key: {e}. Did you forget to set the observation_key or action_key?") from e
        self.freq_hz = freq_hz

    def __repr__(self) -> str:
        if not hasattr(self, "stats"):
            self.stats = self.trajectory().stats()
            stats = str(self.stats).replace("\n ", "\n  ")
        return f"Episode(\n  {stats}\n)"

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
            >>> from embdata.sense.image import Image
            >>> from embdata.motion import Motion
            >>> episode = Episode(
            ...     steps=[
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="grasp"),
            ...             action=Motion(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
            ...         ),
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="move"),
            ...             action=Motion(position=[0.2, 0.3, 0.4], orientation=[0, 0, 1, 0]),
            ...         ),
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="release"),
            ...             action=Motion(position=[0.3, 0.4, 0.5], orientation=[1, 0, 0, 0]),
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
        freq_hz = freq_hz or self.freq_hz or 1
        return Trajectory(
            [step[field].numpy() for step in self.steps],
            freq_hz=freq_hz,
            dim_labels=self.steps[0][field].flatten("dict").keys(),
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
        from rich import print
        for k, v in  self.steps[0].features().items():
            print(k, v, describe(data[0][k]))
        # print_json(self.steps[0].model_dump_json())
        return Dataset.from_list(data, features=self.steps[0].features())

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
