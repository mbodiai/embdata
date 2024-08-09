import logging
import os
import sys
import traceback
from itertools import zip_longest
from threading import Thread
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal

import numpy as np
import rerun as rr
import torch
from datasets import Dataset, DatasetDict, Features, IterableDataset, Sequence, Value
from datasets import Image as HFImage
from pydantic import Field, PrivateAttr, model_validator

from embdata.describe import describe
from embdata.features import to_features_dict
from embdata.motion.control import AnyMotionControl, RelativePoseHandControl
from embdata.sample import Sample
from embdata.sense.image import Image
from embdata.time import ImageTask, TimeStep, VisionMotorStep
from embdata.trajectory import Stats, Trajectory
from embdata.utils.iter_utils import get_iter_class
from embdata.utils.rerun_utils import get_blueprint, project_points_to_2d

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch
except ImportError:
    logging.info("lerobot not found. Go to https://github.com/huggingface/lerobot to install it.")

logger = logging.getLogger(__name__)
if logger.level == logging.DEBUG or "MBENCH" in os.environ:
    from mbench import profileme

    profileme()



class Episode(Sample):
    """A list-like interface for a sequence of observations, actions, and/or other data."""

    episode_id: str | int | None = None
    steps: list[TimeStep] = Field(default_factory=list)
    metadata: Sample | Any | None = None
    freq_hz: int | float | None = None
    _action_class: type[Sample] = PrivateAttr(default=Sample)
    _state_class: type[Sample] = PrivateAttr(default=Sample)
    _observation_class: type[Sample] = PrivateAttr(default=Sample)
    _step_class: type[TimeStep] = PrivateAttr(default=TimeStep)
    image_keys: str | list[str] = "image"
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

    def __init__(
        self,
        steps: List[Dict | Sample | TimeStep] | Iterable,
        observation_key: str = "observation",
        action_key: str = "action",
        supervision_key: str | None = "supervision",
        image_keys: str | list[str] | None = "image",
        metadata: Sample | Any | None = None,
        freq_hz: int | None = None,
        **kwargs,
    ) -> None:
        """Create an episode from a list of steps.

        Steps can be a list of dictionaries, samples, or time steps. If steps are dictionaries, the observation, action, and supervision keys must be provided.

        Args:
            steps (List[Dict | Sample | TimeStep]): A list of steps.
            observation_key (str, optional): The key to use for observations. Defaults to "observation".
            action_key (str, optional): The key to use for actions. Defaults to "action".
            supervision_key (str, optional): The key to use for supervisions. Defaults to "supervision".
            image_keys (str | list[str], optional): The keys to use for images. Defaults to "image".
            metadata (Sample | Any, optional): Metadata for the episode. Defaults to None.
            freq_hz (int, optional): The frequency in Hz. Defaults to None.
        """
        if not hasattr(steps, "__iter__"):
            msg = "Steps must be an iterable"
            raise ValueError(msg)
        steps = list(steps) if not isinstance(steps, list) else steps

        if "step_class" in kwargs:
            self._step_class = kwargs.pop("step_class")
            Step = self._step_class
        else:
            try:
                Step = self.__class__._step_class.get_default()  # noqa
            except AttributeError:
                Step = TimeStep

        if steps and not isinstance(steps[0], TimeStep | Sample):
            if isinstance(steps[0], dict) and observation_key in steps[0] and action_key in steps[0]:
                steps = [
                    Step(
                        observation=step[observation_key],
                        action=step[action_key],
                        supervision=step.get(supervision_key),
                        image_keys=image_keys,
                    )
                    for step in steps
                ]
            elif isinstance(steps[0], tuple):
                steps = [
                    Step(
                        observation=step[0],
                        action=step[1],
                        state=step[2] if len(step) > 2 else None,
                        supervision=step[3] if len(step) > 3 else None,
                        image_keys=image_keys,
                    )
                    for step in steps
                ]

        super().__init__(steps=steps, metadata=metadata, freq_hz=freq_hz, image_keys=image_keys, **kwargs)

    @classmethod
    def from_list(
        cls,
        steps: List[Dict],
        observation_key: str,
        action_key: str,
        state_key: str | None = None,
        supervision_key: str | None = None,
        freq_hz: int | None = None,
        **kwargs,
    ) -> "Episode":
        Step = cls._step_class.get_default()
        observation_key = observation_key or "observation"
        action_key = action_key or "action"
        state_key = state_key or "state"
        supervision_key = supervision_key or "supervision"
        freq_hz = freq_hz or 1
        processed_steps = [
            Step(
                observation=step.get(observation_key),
                action=step.get(action_key),
                state=step.get(state_key),
                supervision=step.get(supervision_key),
                timestamp=i / freq_hz,
                **{
                    k: v
                    for k, v in step.items()
                    if k not in ["timestamp", observation_key, action_key, state_key, supervision_key]
                },
            )
            for i, step in enumerate(steps)
        ]
        return cls(
            steps=processed_steps,
            freq_hz=freq_hz,
            observation_key=observation_key,
            action_key=action_key,
            state_key=state_key,
            supervision_key=supervision_key,
            **kwargs,
        )

    @classmethod
    def from_lists(
        cls,
        observations: List[Sample | Dict | np.ndarray],
        actions: List[Sample | Dict | np.ndarray],
        states: List[Sample | Dict | np.ndarray] | None = None,
        supervisions: List[Sample | Dict | np.ndarray] | None = None,
        image_keys: str | list[str] = "image",
        freq_hz: int | None = None,
        **kwargs,
    ) -> "Episode":
        """Create an episode from lists of observations, actions, states, and supervisions or other list of dicts.

        Args:
            observations (List[Sample | Dict | np.ndarray]): A list of observations.
            actions (List[Sample | Dict | np.ndarray]): A list of actions.
            states (List[Sample | Dict | np.ndarray], optional): A list of states. Defaults to None.
            supervisions (List[Sample | Dict | np.ndarray], optional): A list of supervisions. Defaults to None.
            image_keys (str | list[str], optional): The keys to use for images. Defaults to "image".
            freq_hz (int, optional): The frequency in Hz. Defaults to None.
        """

        Step: type[TimeStep] = cls._step_class.get_default()  # noqa: N806
        observations = observations or []
        actions = actions or []
        states = states or []
        supervisions = supervisions or []
        length = max(len(observations), len(actions), len(states), len(supervisions))
        freq_hz = freq_hz or 1.0
        kwargs.update({"freq_hz": freq_hz})
        steps = [
            Step(observation=o, action=a, state=s, supervision=sup, timestamp=i / freq_hz, image_keys=image_keys)
            for i, o, a, s, sup in zip_longest(
                range(length),
                observations,
                actions,
                states,
                supervisions,
                fillvalue=Sample(),
            )
        ]
        return cls(steps=steps, **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        image_keys: str | list[str] = "image",
        observation_key: str = "observation",
        action_key: str = "action",
        state_key: str | None = "state",
        supervision_key: str | None = "supervision",
        freq_hz_key: str | None = "freq_hz",
    ) -> "Episode":
        """Create an episode from a Hugging Face dataset.

        Args:
            dataset (Dataset): The dataset to create the episode from.
            image_keys (str | list[str], optional): The keys to use for images. Defaults to "image".
            observation_key (str, optional): The key to use for observations. Defaults to "observation".
            action_key (str, optional): The key to use for actions. Defaults to "action".
            state_key (str, optional): The key to use for states. Defaults to "state".
            supervision_key (str, optional): The key to use for supervisions. Defaults to "supervision".
            freq_hz_key (str, optional): The key to use for the frequency in Hz. Defaults to "freq_hz".
        """
        if isinstance(dataset, IterableDataset):
            msg = "IterableDataset not supported"
            raise TypeError(msg)

        if isinstance(dataset, DatasetDict):
            freq_hz = dataset.get(freq_hz_key, 1)
            freq_hz = freq_hz[0] if isinstance(freq_hz, list) else freq_hz
            list_keys = {key for key in dataset if isinstance(dataset[key], list)}
            list_keys.update((image_keys, observation_key, action_key, state_key, supervision_key))
            return cls.from_lists(
                *[dataset[key] for key in list_keys],
                image_keys,
                observation_key,
                action_key,
                state_key,
                supervision_key,
            )

        return cls(
            steps=[
                TimeStep.from_dict(step, image_keys, observation_key, action_key, supervision_key) for step in dataset
            ],
            image_keys=image_keys,
        )

    def dataset(self) -> Dataset:
        if self.steps is None or len(self.steps) == 0:
            msg = "Episode has no steps"
            raise ValueError(msg)

        image_feat = HFImage() if self.steps[0].flatten("dict",include=self.image_keys) else Value("string")
        feat = {
            "image": image_feat,
            "episode_idx": Value("int64"),
            "step_idx": Value("int64"),
            "timestamp": Value("float32"),
            **self.steps[0].infer_features_dict(),
        }
        model_info = self.steps[0].model_info()
        if model_info:
            feat["_info"] = to_features_dict(self.steps[0].model_info())
        feat = Features(feat)

        data = []
        for step in self.steps:
            try:
                images = step.flatten("list",include=self.image_keys)
                image = images[0] if isinstance(images, list) else images
                try:
                    image = Image(image).pil
                except (ValueError, TypeError, AttributeError):
                    traceback.print_exc()
                    image = "Absent"
            except (ValueError, AttributeError, TypeError, IndexError):
                image = "Absent"
            step = step.dump(as_field="pil")  # noqa
            step_idx = step.pop("step_idx", None)
            episode_idx = step.pop("episode_idx", None)
            timestamp = step.pop("timestamp", None)
            step = {  # noqa
                "image": image,
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "timestamp": timestamp,
                **step,
            }
            if model_info:
                step["_info"] = model_info
            data.append(step)

        return Dataset.from_list(data, features=feat)

    def stats(self, mode: Literal["full", "first10"] = "first10") -> Stats:
        """Get the statistics of the episode."""
        if not hasattr(self, "stats_"):
            self.stats_ = self.trajectory(mode=mode).stats()
        return self.stats_


    def __slice__(self, start: int, stop: int, step: int) -> "Episode":
        return Episode(steps=self.steps[start:stop:step])

    def trajectory(
        self, of: str | list[str] = "action", freq_hz: int | None = None, mode: Literal["full", "first10"] = "full"
    ) -> Trajectory:
        """Numpy array with rows (axis 0) consisting of the `of` argument. Can be steps or plural form of fields.

        Each step will be reduced to a flattened 1D numpy array. A conveniant
        feature is that transforms can be reversed with the `un` version of the method. This is done by keeping
        a stack of inverse functions with correct kwargs partially applied.

        Note:
        - This is a lazy operation and will not be computed until a field or method like "show" is called.
        - The `of` argument can be in plural or singular form which will result in the same output.

        Some operations that can be done are:
        - show
        - minmax scaling
        - gaussian normalization
        - resampling with interpolation
        - filtering
        - windowing
        - low pass filtering

        Args:
            of (str, optional): steps or the step field get the trajectory of. Defaults to "steps".
            freq_hz (int, optional): The frequency in Hz. Defaults to the episode's frequency.
            mode (str, optional): The mode to use for statistics. Defaults to "first10".

        Example:
            Understand the relationship between frequency and grasping.
        """
        
        step_keys = {"observations", "actions", "states", "steps", "supervisions"}
        of = of if isinstance(of, list) else [of]
        from embdata.describe import full_paths
        full_keys = full_paths(self.steps[0], of)
        logging.debug("Flattening episode steps for %s", of)
        of = [f[:-1] if f.endswith("s") and f in step_keys else f for f in of]
        if self.steps is None or len(self.steps) == 0:
            msg = "Episode has no steps"
            raise ValueError(msg)
        freq_hz = freq_hz or self.freq_hz or 1
        include = [k for k in self.steps[0].flatten("dict", exclude="info") if any(k in field for field in full_keys)]
        if of == ["step"]:
            if mode == "full":
                data = self.flatten(to="lists", include=include, exclude="info")
            elif mode == "first10":
                data = self.flatten(to="lists", include=include, exclude="info")[:10]
            if not data:
                msg = "Episode has no steps"
                raise ValueError(msg)
            return Trajectory(
                data,
                freq_hz=freq_hz,
                _episode=self,
                keys=include,
                _sample_cls=self.steps[0].__class__,
                _sample_keys=include,
            )
        logging.debug("Describe data: %s", list(self.steps[0].flatten("dict")))
        if not any(k in field for field in self.steps[0].flatten("dict", exclude="info") for k in of):
            msg = f"Field '{of}' not found in episode steps"
            raise ValueError(msg)
        logging.debug("Flattening episode steps for %s", of)
        if mode == "full":
                data = self.flatten(to="lists", include=of)
        elif mode == "first10":
            data = self.flatten(to="lists", include=of)[:10]
        else:
            msg = f"Mode {mode} not supported"
            raise ValueError(msg)
        if not data:
            msg = f"Field '{of}' not found with any numerical values in episode steps"
            raise ValueError(msg)

        keys = [key.removeprefix("steps.*.") for key in full_keys]
        keys = [subkey for key in keys for subkey in self.steps[0][key].flatten("dict")]
        return Trajectory(
            steps=data,
            freq_hz=freq_hz,
            keys=keys,
            _episode=self,
            _sample_keys=include,
            _sample_cls=self.steps[0].__class__ if len(keys) > 1 else self.steps[0][of[0]].__class__,
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
        """Set the step at the specified index."""
        self.steps[idx] = value

    def __iter__(self) -> Any:
        """Iterate over the keys in the dataset."""
        return iter(self.steps)
    
    def dump(self, as_field: str = "dict") -> Any:
        """Dump the episode to a dictionary or list of dictionaries.

        Args:
            as_field (str, optional): The format to dump the episode to. Defaults to "dict".

        Returns:
            Any: The dumped episode.
        """
        return [step.dump(as_field) for step in self.steps]

    def map(self, func: Callable[[TimeStep | Dict | np.ndarray], np.ndarray | TimeStep], field=None) -> "Episode":
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
        return Episode(steps=[step for step in self.steps if condition(step)], metadata=self.metadata)



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
            >>> [[step.observation.value for step in ep.steps] for ep in episodes]
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

    def group_by(self, key: str) -> Dict[str, "Episode"]:
        """Group the steps in the episode by a key or a list of discriminators.

        Args:
            key (str): The key to group by.
            discriminators (List[Callable[[TimeStep], Any]]): A function that takes a time step and returns a one of the choices.
            choices (List, optional): A list of choices to group by. Defaults to None.

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
            image = Image(data["observation.image"]).pil
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

    def window(self, nforward: int, nbackward: int = 0, current_n: int = 0, pad_value: Any = None) -> Iterable:
        """Create a sliding window over the episode.

        Args:
            nforward (int): The number of steps to look forward.
            nbackward (int, optional): The number of steps to look backward. Defaults to 0.
            current_n (int, optional): The current step index. Defaults to 0.
            pad_value (Any, optional): The value to pad the window with. Defaults to None.

        Yields:
            Iterable: An iterable of steps in the window.
        """
        for i in range(current_n - nbackward, current_n + nforward):
            if i < 0 or i >= len(self):
                yield pad_value
            else:
                yield self[i]

    def windowed(self, nforward: int, nbackward: int = 0, pad_value: Any = None) -> "Episode":
        """Create a windowed episode.

        Args:
            nforward (int): The number of steps to look forward.
            nbackward (int, optional): The number of steps to look backward. Defaults to 0.
            pad_value (Any, optional): The value to pad the window with. Defaults to None.

        Returns:
            Episode: The windowed episode.
        """
        self.__iter__ = yield from [self.window(nforward, nbackward, i, pad_value) for i in range(len(self))]
        return self

    def flatten(self, to: Literal["dicts", "lists", "torch"], include: list[str] | None = None, exclude: list[str] | None = None) -> Any:
        """Flatten the episode to a list of dictionaries, lists, or torch tensors.

        Args:
            to (str): The format to flatten the episode to.
            include (list[str], optional): The keys to include in the flattened data. Defaults to None.

        Returns:
            Any: The flattened episode.
        """
        to = [t[:-1] if t.endswith("s") else t for t in to]
        include = [include.removeprefix("steps.*.") for include in include]
        flattened = [step.flatten(to, include=include, exclude=exclude) for step in self.steps]
        if flattened and not isinstance(flattened[0], dict | Sample) and len(flattened[0]) == 1:
            return [f[0] for f in flattened]
        return flattened

    def log_scalar(self, name: str, value: float, step: int) -> None:

        rr.log(name, rr.Scalar(value))


    def rerun(self, mode: Literal["local", "remote"], port=3389, ws_port=8888) -> "Episode":
        """Start a rerun server."""
        rr.init("rerun-mbodied-data", spawn=True)
        # rr.serve(open_browser=False, web_port=port, ws_port=ws_port)

        blueprint = get_blueprint()

        states_to_log = ["RemoteControl/x",
                            "RemoteControl/y",
                            "RemoteControl/z",
                            "ef/x",
                            "ef/y",
                            "ef/z"]
        
        plot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        for state, color in zip(states_to_log, plot_colors, strict=False):
            print(f"Logging state {state}")
            rr.log(f"action/{state}", rr.SeriesLine(color=color, name=state), static=True)
        
        end_effector_offset = 0.175
        next_n = 4
        arrow_color = 0xF14F21 # Alpha set to 128 for 50% transparency
        radii = 7
        
        for i, step in enumerate(self.steps):
            print(f"Processing step {i}")
            if not hasattr(step, "timestamp") or step.timestamp is None:
                step.timestamp = i / 5
            rr.set_time_sequence("timeline0", i)
            rr.set_time_seconds("timestamp", step.timestamp)

            # rr.log(f"action/ef/x", rr.Scalar(step.absolute_pose.pose.x))
            rr.log(f"action/ef/y", rr.Scalar(step.absolute_pose.pose.y))
            # rr.log(f"action/ef/z", rr.Scalar(step.absolute_pose.pose.z - end_effector_offset))
            
            rr.log("scene/image", rr.Image(data=step.observation.image.array if step.observation.image else None))
            rr.log("augmented/image", rr.Image(data=step.inpaint_image.array if step.inpaint_image else None))
            # rr.log("segmented/image", rr.Image(data=step.segmentation_image.array if step.segmentation_image else None))
            
            params = step.camera_params

            detection_results = step.detection_result.objects

            for obj in detection_results:
                name = obj.name
                bbox_2d = obj.bbox_2d

                if bbox_2d:
                    min_corner = (bbox_2d.x1, bbox_2d.y1)
                    width = bbox_2d.x2 - bbox_2d.x1
                    height = bbox_2d.y2 - bbox_2d.y1

                rr.log(f"scene/detection/{name}", rr.Boxes2D(mins=[min_corner], sizes=[(width, height)], labels=[name]))

            for window in self.window(self.steps, current_n=i, nforward=next_n, nbackward=0):
                projected_start_points_2d = []
                projected_end_points_2d = []
                for j in range(len(window) - 1):
                    start_point_2d, end_point_2d = project_points_to_2d(params, window[j].absolute_pose.pose, window[j+1].absolute_pose.pose)
                    projected_start_points_2d.append(start_point_2d)
                    projected_end_points_2d.append(end_point_2d)

                start_points_2d_array = np.array(projected_start_points_2d)
                end_points_2d_array = np.array(projected_end_points_2d)
                vectors = end_points_2d_array - start_points_2d_array
                
                rr.log("scene/arrows", rr.Arrows2D(vectors=vectors, origins=start_points_2d_array, colors=(arrow_color, 50), radii=radii))

            scene_objects = step.state.scene.scene_objects

            for obj in scene_objects:

                object_name = obj['object_name'].replace(' ', '')
                rr.log(f"Objects/{object_name}", rr.Points3D(positions=[[obj['object_pose']["x"], obj['object_pose']["y"], obj['object_pose']["z"]]], labels=[obj['object_name']]))
                
                if object_name == "RemoteControl":
                    # rr.log(f"action/{object_name}/x", rr.Scalar(obj['object_pose']['x']))
                    rr.log(f"action/{object_name}/y", rr.Scalar(obj['object_pose']['y']))
                    # rr.log(f"action/{object_name}/z", rr.Scalar(obj['object_pose']['z']))


        rr.send_blueprint(blueprint)

    def show(self, mode: Literal["local", "remote"] | None = None, port=5003, ws_port=5004) -> None:
        if mode is None:
            msg = "Please specify a mode: 'local' or 'remote'"
            raise ValueError(msg)
        thread = Thread(target=self.rerun, kwargs={"port": port, "mode": mode, "ws_port": ws_port},
                        daemon=True)
        self._rr_thread = thread
        thread.start()
        try:
            while hasattr(self, "_rr_thread"):
                pass
        except KeyboardInterrupt:
            self.close_view()
            sys.exit()
        finally:
            self.close_view()


    def close_view(self) -> None:
        if hasattr(self, "_rr_thread"):
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
