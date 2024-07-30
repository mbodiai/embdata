"""embdata: Embodied AI data structures and operations.

Includes Episode which is list-like and Sample which is dict-like.

```python
Episode(steps=any_list_of_dicts, image_keys="some.nested.image").dataset().push_to_hub("your_repo_name")
```
is gauranteed to work.

```python
Episode.trajectory("some.nested.field").minmax().upsample().plow().show()
```
Also works.

Examples:
    >>> from embdata import Episode
    >>> episode = Episode()
    >>> timestep = {
    ...     "observation": {"image": "path/pil_image/url/base64/bytes/whatever.jpg"},
            "whatever": "you_want",
            "action": {"any_type_of_action": "action"},
            "state": {
                "you_get_the_idea": "state"
            },
    }
    >>> episode.append(timestep)
"""

from typing import List

from . import episode, motion, sample, sense, trajectory
from .episode import Episode, ImageTask, TimeStep, VisionMotorStep
from .motion import Motion
from .sample import Sample

__all__ = [
    "Episode",
    "TimeStep",
    "ImageTask",
    "VisionMotorStep",
    "Sample",
    "Motion",
    "episode",
    "motion",
    "sample",
    "sense",
    "trajectory",
]
