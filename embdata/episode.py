import logging
from threading import Thread
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal

import rerun as rr
from datasets import Dataset
from pydantic import ConfigDict, PrivateAttr, model_validator
from rerun.archetypes import Image as RRImage

from embdata.image import Image
from embdata.motion import Motion
from embdata.motion.control import RelativePoseHandControl
from embdata.sample import Sample
from embdata.trajectory import Trajectory


class TimeStep(Sample):
    """Time step for a task."""
    episode_idx: int = 0
    step_idx: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    observation: Sample = None
    action: Sample = None
    state: Sample = None
    supervision: Any = None

    @model_validator(mode="before")
    @classmethod
    def validate_action_class(cls, values) -> Dict[str, Any]:
        return {
            k: Sample(v)
            # Omit supervision to allow float values.
            if not isinstance(v, Sample) and v is not None and k in ["observation", "action", "state"]
            else v
            for k, v in values.items()
        }


class ImageTask(Sample):
    """Canonical Observation."""

    image: Image
    task: str


class VisionMotorStep(TimeStep):
    """Time step for vision-motor tasks."""

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

    steps: list[TimeStep]
    metadata: Sample | Any | None = None
    freq_hz: int | None = None
    _action_class: type[Sample] = PrivateAttr(default=Sample)
    _rr_thread: Thread | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_action_class(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the action class based on the first step in the episode.

        Args:
            values (Dict[str, Any]): The values of the episode.

        Returns:
            Dict[str, Any]: The validated values of the episode.
        """
        if values.get("steps") is not None and len(values.get("steps")) > 0:  # noqa: SIM102
            if hasattr(values.get("steps")[0], "action") and values.get("steps")[0].action is not None:
                cls._action_class = type(values.get("steps")[0].action)
        return values

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
    def from_observations_actions(cls, observations: List[Sample], actions: List[Sample]) -> "Episode":
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
        return cls(
            steps=[TimeStep(observation=obs, action=action) for obs, action in zip(observations, actions, strict=True)],
        )

    def __init__(
        self,
        steps: List[Dict | Sample] | Iterable,
        observation_key: str = "observation",
        action_key: str = "action",
        state_key: str | None = None,
        supervision_key: str | None = None,
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
            raise ValueError("Steps must be an iterable")
        if not hasattr(steps, "__getitem__"):
            steps = list(steps)
        if steps and not isinstance(steps[0], Dict | Sample) and not isinstance(steps[0], TimeStep):
            steps = [
                {
                    "observation": step[0],
                    "action": step[1],
                    "state": step[2] if len(step) > 2 else None,
                    "supervision": step[3] if len(step) > 3 else None,
                }
                for step in steps
            ]
            supervision_key = "supervision"
            action_key = "action"
            observation_key = "observation"
            state_key = "state"
        steps = steps or [
            TimeStep(
                observation=step[observation_key],
                action=step[action_key],
                state=step[state_key] if state_key else None,
                supervision=step[supervision_key] if supervision_key else None,
                **{k: step[k] for k in step if k not in [observation_key, action_key, supervision_key, state_key]},
            )
            for step in steps
        ]
        try:
            super().__init__(
                steps=steps,
                metadata=metadata,
            )
        except KeyError as e:
            raise KeyError(f"Missing key: {e}. Did you forget to set the observation_key or action_key?") from e

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
        freq_hz = freq_hz or self.freq_hz or 1
        return Trajectory(
            [step[field].numpy() for step in self.steps],
            freq_hz=freq_hz,
            dim_labels=self.steps[0][field].flatten("dict").keys(),
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

    def map(self, func: Callable[[TimeStep | Dict], TimeStep]) -> "Episode":
        """Apply a function to each step in the episode.

        Args:
            func (Callable[[TimeStep], TimeStep]): The function to apply to each step.

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

    def unpack(self) -> tuple:
        """Unpack the episode into a tuple of lists with a consistent order.

        Output order:
         observations, actions, states, supervision, ... other attributes.

        Returns:
            tuple[list, list, list]: A tuple of lists containing the observations, actions, and states.

        Example:
            >>> episode = Episode(
            ...     steps=[
            ...         TimeStep(observation=1, action=10),
            ...         TimeStep(observation=2, action=20),
            ...         TimeStep(observation=3, action=30),
            ...     ]
            ... )
            >>> observations, actions, states = episode.unpack()
            >>> observations
            [Sample(value=1), Sample(value=2), Sample(value=3)]
            >>> actions
            [Sample(value=10), Sample(value=20), Sample(value=30)]
        """
        if not self.steps:
            msg = "Episode has no steps"
            raise ValueError(msg)
        return tuple([step[k] for step in self.steps] for k, _ in self.steps[0])

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

    def dataset(self) -> Dataset:
        if self.steps is None or len(self.steps) == 0:
            msg = "Episode has no steps"
            raise ValueError(msg)
        data = [step.dict() for step in self.steps]
        return Dataset.from_list(data, features=self.steps[0].features())

    def rerun(self, mode=Literal["local", "remote"], port=5003, ws_port=5004) -> "Episode":
        """Start a rerun server."""
        rr.init("rerun-mbodied-data", spawn=mode == "local")
        rr.serve(open_browser=False, web_port=port, ws_port=ws_port)
        for i, step in enumerate(self.steps):
            if not hasattr(step, "timestamp"):
                step.timestamp = i / 5
            rr.set_time_sequence("frame_index", i)
            rr.set_time_seconds("timestamp", step.timestamp)
            rr.log("observation", RRImage(step.observation.image)) if step.observation.image else None
            for dim, val in step.action.flatten("dict").items():
                rr.log(f"action/{dim}", rr.Scalar(val))
            if step.action.flatten(to="pose"):
                origin = step.state.numpy()[:3]
                direction = step.action.numpy()[:3]
                rr.log("action/pose_arrow", rr.Arrows3D(vectors=[direction], origins=[origin]))
        while hasattr(self, "_rr_thread") and self._rr_thread.is_alive():
            pass

    def show(self, mode: Literal["local", "remote"], port=5003) -> None:
        thread = Thread(target=self.rerun, kwargs={"port": port, "mode": mode})
        self._rr_thread = thread
        thread.start()

    def close_view(self) -> None:
        if hasattr(self, "_rr_thread"):
            self._rr_thread.join()
        self._rr_thread = None


class VisionMotorEpisode(Episode):
    """An episode for vision-motor tasks."""

    steps: list[VisionMotorStep]


class VisionMotorHandEpisode(VisionMotorEpisode):
    """An episode for vision-motor tasks with hand control."""

    _action_class: type[RelativePoseHandControl] = PrivateAttr(default=RelativePoseHandControl)

    def __init__(self, **data):
        super().__init__(**data)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
