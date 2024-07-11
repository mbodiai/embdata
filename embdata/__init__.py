"""embdata: A package for handling embodied AI data structures and operations.

This package provides classes and utilities for working with various data types
commonly used in embodied AI tasks, such as episodes, time steps, images, and samples.

Examples:
    >>> from embdata import Episode, TimeStep, Image, Sample
    >>> # Create a sample episode
    >>> episode = Episode(steps=[TimeStep(observation=Image(), action=Sample(velocity=1.0))])
    >>> print(len(episode))
    1
"""

from typing import List

from .episode import Episode, ImageTask, TimeStep, VisionMotorStep
from .image import Image
from .sample import Sample

__all__ = [
    "Episode",
    "TimeStep",
    "ImageTask",
    "VisionMotorStep",
    "Image",
    "Sample",
]
