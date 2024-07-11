from typing import Any
import numpy as np
import pytest
from embdata.sample import Sample
from pydantic import Field
from embdata.ndarray import NumpyArray


class NewSample(Sample):
    answer: str = Field(
        default="",
        description="Short, one sentence answer to any question a user might have asked. 20 words max.",
    )
    sleep: bool = Field(
        default=False,
        description="Whether the robot should go to sleep after executing the motion.",
    )
    home: bool = Field(
        default=False,
        description="Whether the robot should go to home after executing the motion.",
    )


def test_sample_schema():
    assert NewSample().schema(include_descriptions=True) == {
        "type": "object",
        "properties": {
            "answer": {
                "default": "",
                "description": "Short, one sentence answer to any question a user might have asked. 20 words max.",
                "type": "string",
            },
            "sleep": {
                "default": False,
                "description": "Whether the robot should go to sleep after executing the motion.",
                "type": "boolean",
            },
            "home": {
                "default": False,
                "description": "Whether the robot should go to home after executing the motion.",
                "type": "boolean",
            },
        },
    }


def test_sample_schema_with_numpy_array():
    class NewSample(Sample):
        answer: str = Field(
            default="",
            description="Short, one sentence answer to any question a user might have asked. 20 words max.",
        )
        sleep: bool = Field(
            default=False,
            description="Whether the robot should go to sleep after executing the motion.",
        )
        home: bool = Field(
            default=False,
            description="Whether the robot should go to home after executing the motion.",
        )
        image: NumpyArray = Field(
            default_factory=lambda: np.zeros((224, 224, 3)),
            description="Image data",
        )

    assert NewSample().schema(include_descriptions=False) == {
        "type": "object",
        "properties": {
            "answer": {"default": "", "type": "string"},
            "sleep": {"default": False, "type": "boolean"},
            "home": {"default": False, "type": "boolean"},
            "image": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    }


def test_sample_schema_with_null():
    class NewSample(Sample):
        answer: str = Field(
            default="",
            description="Short, one sentence answer to any question a user might have asked. 20 words max.",
        )
        sleep: bool | None = Field(
            description="Whether the robot should go to sleep after executing the motion.",
        )
        home: bool = Field(
            default=False,
            description="Whether the robot should go to home after executing the motion.",
        )
        image: None | NumpyArray = Field(
            default_factory=lambda: np.zeros((224, 224, 3)),
            description="Image data",
        )

    assert NewSample(sleep=True).schema(include_descriptions=False) == {
        "type": "object",
        "required": ["sleep"],
        "properties": {
            "answer": {"default": "", "type": "string"},
            "sleep": {"type": "boolean"},
            "home": {"default": False, "type": "boolean"},
            "image": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    }


def test_dynamic_field():
    assert Sample(list_field=["a", "b"]).schema(include_descriptions=False) == {
        "type": "object",
        "properties": {"list_field": {"type": "array", "items": {"type": "string"}}},
    }


def test_nested():
    class AnotherSample(Sample):
        child_field: Sample = Field(
            default_factory=lambda: Sample(list_field=["a", "b"]),
            description="Child field",
        )

    class NewSample(Sample):
        answer: str = Field(
            default="",
            description="Short, one sentence answer to any question a user might have asked. 20 words max.",
        )
        nested: AnotherSample = Field(
            default_factory=lambda: AnotherSample(child_field=Sample(list_field=["a", "b"])),
            description="Nested sample",
        )

    assert NewSample().schema(include_descriptions=False) == {
        "type": "object",
        "properties": {
            "answer": {"default": "", "type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "child_field": {
                        "type": "object",
                        "properties": {"list_field": {"type": "array", "items": {"type": "string"}}},
                    }
                },
            },
        },
    }


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
