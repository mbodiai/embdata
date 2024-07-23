"""embdata: A package for handling embodied AI data structures and operations.

This package provides classes and utilities for working with various data types
commonly used in embodied AI tasks, such as episodes, time steps, images, and samples.

Examples:
    >>> from embdata import Episode, TimeStep, Image, Sample
    >>> # Create a complex nested structure with image and text data
    >>> image_data = Image.from_base64("base64_encoded_image_data", encoding="jpeg")
    >>> text_data = Sample(text="This is a sample text")
    >>> action = Sample(velocity=1.0, rotation=0.5)
    >>> observation = Sample(image=image_data, text=text_data)
    >>> time_step = TimeStep(observation=observation, action=action)
    >>> episode = Episode(steps=[time_step])
    >>> print(len(episode))
    1
    >>> print(episode.steps[0].observation.image.encoding)
    'jpeg'
    >>> print(episode.steps[0].observation.text.text)
    'This is a sample text'
    >>> print(episode.steps[0].action.velocity)
    1.0
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
