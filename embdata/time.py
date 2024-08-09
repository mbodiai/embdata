from typing import Any, Dict, Iterable, List

import numpy as np
from pydantic import ConfigDict, PrivateAttr

from embdata.motion import Motion
from embdata.sample import Sample
from embdata.sense.image import Image
from embdata.utils.iter_utils import map_nested_with_keys


def convert_images(d, image_keys=None) -> dict:
    fn = lambda k, v: Image(v) if k and image_keys and k[-1] in image_keys else v
    return map_nested_with_keys(fn, d)


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
    image_keys: str | set[str] | None = "image"
    _observation_class: type[Sample] = PrivateAttr(default=Sample)
    _action_class: type[Sample] = PrivateAttr(default=Sample)
    _state_class: type[Sample] = PrivateAttr(default=Sample)
    _supervision_class: type[Sample] = PrivateAttr(default=Sample)

    @classmethod
    def from_dict(  # noqa: PLR0913
        cls,
        values: Dict[str, Any],
        image_keys: str | set | None = "image",
        observation_key: str | None = "observation",
        action_key: str | None = "action",
        supervision_key: str | None = "supervision",
        state_key: str | None = "state",
    ) -> "TimeStep":
        obs = values.pop(observation_key, None)
        act = values.pop(action_key, None)
        sta = values.pop(state_key, None)
        sup = values.pop(supervision_key, None)
        timestamp = values.pop("timestamp", 0)
        step_idx = values.pop("step_idx", 0)
        episode_idx = values.pop("episode_idx", 0)

        Obs = cls._observation_class.get_default()  # noqa: N806
        Act = cls._action_class.get_default()  # noqa: N806
        Sta = cls._state_class.get_default()  # noqa: N806
        Sup = cls._supervision_class.get_default()  # noqa: N806
        obs = Obs(**convert_images(obs, image_keys)) if obs is not None else None
        act = Act(**convert_images(act, image_keys)) if act is not None else None
        sta = Sta(**convert_images(sta, image_keys)) if sta is not None else None
        sup = Sup(**convert_images(sup, image_keys)) if sup is not None else None
        field_names = cls.model_fields.keys()
        return cls(
            observation=obs,
            action=act,
            state=sta,
            supervision=sup,
            episode_idx=episode_idx,
            step_idx=step_idx,
            timestamp=timestamp,
            **{k: v for k, v in values.items() if k not in field_names},
        )

    @classmethod
    def from_iterable(cls, step: tuple, image_keys="image", **kwargs) -> "TimeStep":
        return cls(*step, image_keys=image_keys, **kwargs)

    def __init__(
        self,
        observation: Sample | Dict | np.ndarray,
        action: Sample | Dict | np.ndarray,
        state: Sample | Dict | np.ndarray | None = None,
        supervision: Any | None = None,
        episode_idx: int | None = 0,
        step_idx: int | None = 0,
        timestamp: float | None = None,
        image_keys: str | set[str] | None = "image",
        **kwargs,
    ):
        obs = observation
        act = action
        sta = state
        sup = supervision

        Obs = TimeStep._observation_class.get_default() if not isinstance(obs, Sample) else lambda x: x  # noqa: N806
        Act = TimeStep._action_class.get_default() if not isinstance(act, Sample) else lambda x: x  # noqa: N806
        Sta = TimeStep._state_class.get_default() if not isinstance(sta, Sample) else lambda x: x  # noqa: N806
        Sup = TimeStep._supervision_class.get_default() if not isinstance(sup, Sample) else lambda x: x  # noqa: N806
        obs = Obs(convert_images(obs, image_keys)) if obs is not None else None
        act = Act(convert_images(act, image_keys)) if act is not None else None
        sta = Sta(convert_images(sta, image_keys)) if sta is not None else None
        sup = Sup(convert_images(supervision)) if supervision is not None else None

        super().__init__(
            observation=obs,
            action=act,
            state=sta,
            supervision=sup,
            episode_idx=episode_idx,
            step_idx=step_idx,
            timestamp=timestamp,
            **{k: v for k, v in kwargs.items() if k not in ["observation", "action", "state", "supervision"]},
        )


    def window(self, steps: List["TimeStep"], nforward: int = 1, nbackward: int = 1, pad_value: Any = None) -> Iterable["TimeStep"]:
        """Create a sliding window over the episode.

        Args:
            steps (List[TimeStep]): The steps in the episode.
            nforward (int): The number of steps to look forward.
            nbackward (int, optional): The number of steps to look backward. Defaults to 0.
            current_n (int, optional): The current step index. Defaults to 0.
            pad_value (Any, optional): The value to pad the window with. Defaults to None.

        Yields:
            Iterable: An iterable of steps in the window.
        """
        for i in range(self.step_idx - nbackward, self.step_idx + nforward):
            if i < 0 or i >= len(self):
                yield pad_value
            else:
                yield steps[i]

class ImageTask(Sample):
    """Canonical Observation."""

    image: Image
    task: str


class VisionMotorStep(TimeStep):
    """Time step for vision-motor tasks."""

    _observation_class: type[ImageTask] = PrivateAttr(default=ImageTask)
    observation: ImageTask
    action: Motion
    supervision: Any | None = None
