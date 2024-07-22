# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


from embdata.describe import describe


class ImageModel(BaseModel):
    bytes: Optional[bytes]
    path: Optional[str]


class ActionModel(BaseModel):
    open_gripper: bool
    rotation_delta: List[float] = Field(...)
    terminate_episode: float
    world_vector: List[float] = Field(...)


class ObservationModel(BaseModel):
    image: ImageModel
    natural_language_embedding: List[float] = Field(...)
    natural_language_instruction: str
    state: List[float] = Field(...)


class StepModel(BaseModel):
    action: ActionModel
    is_first: bool
    is_last: bool
    is_terminal: bool
    observation: ObservationModel
    reward: float


class EpisodeModel(BaseModel):
    image_list: List[str]
    steps: List[StepModel]


class MetadataModel(BaseModel):
    depth_cams: int = Field(..., alias="# Depth Cams")
    episodes: str = Field(..., alias="# Episodes")
    rgb_cams: int = Field(..., alias="# RGB Cams")
    wrist_cams: int = Field(..., alias="# Wrist Cams")
    action_space: str = Field(..., alias="Action Space")
    control_frequency: str = Field(..., alias="Control Frequency")
    data_collect_method: str = Field(..., alias="Data Collect Method")
    dataset: str = Field(..., alias="Dataset")
    description: str = Field(..., alias="Description")
    file_size_gb: float = Field(..., alias="File Size (GB)")
    gripper: str = Field(..., alias="Gripper")
    has_camera_calibration: str = Field(..., alias="Has Camera Calibration?")
    has_proprioception: str = Field(..., alias="Has Proprioception?")
    has_suboptimal: str = Field(..., alias="Has Suboptimal?")
    language_annotations: str = Field(..., alias="Language Annotations")
    registered_dataset_name: str = Field(..., alias="Registered Dataset Name")
    robot: str = Field(..., alias="Robot")
    robot_morphology: str = Field(..., alias="Robot Morphology")
    scene_type: str = Field(..., alias="Scene Type")


class DatasetModel(BaseModel):
    key: str = Field(..., alias="__key__")
    metadata: MetadataModel
    episode: EpisodeModel


@pytest.fixture
def data():
    return {
        "__key__": "",
        "metadata": {
            "# Depth Cams": 0,
            "# Episodes": "",
            "# RGB Cams": 0,
            "# Wrist Cams": 0,
            "Action Space": "",
            "Control Frequency": "",
            "Data Collect Method": "",
            "Dataset": "",
            "Description": "",
            "File Size (GB)": 0.0,
            "Gripper": "",
            "Has Camera Calibration?": "",
            "Has Proprioception?": "",
            "Has Suboptimal?": "",
            "Language Annotations": "",
            "Registered Dataset Name": "",
            "Robot": "",
            "Robot Morphology": "",
            "Scene Type": "",
        },
        "episode": {
            "image_list": ["image"],
            "steps": [
                {
                    "action": {
                        "open_gripper": False,
                        "rotation_delta": [0.0, 0.0, 0.0],
                        "terminate_episode": 0.0,
                        "world_vector": [0.0, 0.0, 0.0],
                    },
                    "is_first": False,
                    "is_last": False,
                    "is_terminal": False,
                    "observation": {
                        "image": {"bytes": None, "path": None},
                        "natural_language_embedding": [0.0] * 512,
                        "natural_language_instruction": "",
                        "state": [0.0] * 7,
                    },
                    "reward": 0.0,
                }
            ],
        },
    }


def test_describe_full(data):
    schema = describe(data)
    assert schema == {
        "__key__": {"type": "str"},
        "metadata": {
            "# Depth Cams": {"type": "int"},
            "# Episodes": {"type": "str"},
            "# RGB Cams": {"type": "int"},
            "# Wrist Cams": {"type": "int"},
            "Action Space": {"type": "str"},
            "Control Frequency": {"type": "str"},
            "Data Collect Method": {"type": "str"},
            "Dataset": {"type": "str"},
            "Description": {"type": "str"},
            "File Size (GB)": {"type": "float"},
            "Gripper": {"type": "str"},
            "Has Camera Calibration?": {"type": "str"},
            "Has Proprioception?": {"type": "str"},
            "Has Suboptimal?": {"type": "str"},
            "Language Annotations": {"type": "str"},
            "Registered Dataset Name": {"type": "str"},
            "Robot": {"type": "str"},
            "Robot Morphology": {"type": "str"},
            "Scene Type": {"type": "str"},
        },
        "episode": {
            "image_list": {"type": "array", "length": 1, "items": {"type": "str"}},
            "steps": {
                "type": "array",
                "length": 1,
                "items": {
                    "action": {
                        "open_gripper": {"type": "bool"},
                        "rotation_delta": {"type": "array", "length": 3, "items": {"type": "float"}},
                        "terminate_episode": {"type": "float"},
                        "world_vector": {"type": "array", "length": 3, "items": {"type": "float"}},
                    },
                    "is_first": {"type": "bool"},
                    "is_last": {"type": "bool"},
                    "is_terminal": {"type": "bool"},
                    "observation": {
                        "image": {"bytes": {"type": "NoneType"}, "path": {"type": "NoneType"}},
                        "natural_language_embedding": {"type": "array", "length": 512, "items": {"type": "float"}},
                        "natural_language_instruction": {"type": "str"},
                        "state": {"type": "array", "length": 7, "items": {"type": "float"}},
                    },
                    "reward": {"type": "float"},
                },
            },
        },
    }


def test_describe_compact(data):
    schema = describe(data, compact=True)
    assert schema == {
        "__key__": {"type": "str"},
        "metadata": {
            "# Depth Cams": {"type": "int"},
            "# Episodes": {"type": "str"},
            "# RGB Cams": {"type": "int"},
            "# Wrist Cams": {"type": "int"},
            "Action Space": {"type": "str"},
            "Control Frequency": {"type": "str"},
            "Data Collect Method": {"type": "str"},
            "Dataset": {"type": "str"},
            "Description": {"type": "str"},
            "File Size (GB)": {"type": "float"},
            "Gripper": {"type": "str"},
            "Has Camera Calibration?": {"type": "str"},
            "Has Proprioception?": {"type": "str"},
            "Has Suboptimal?": {"type": "str"},
            "Language Annotations": {"type": "str"},
            "Registered Dataset Name": {"type": "str"},
            "Robot": {"type": "str"},
            "Robot Morphology": {"type": "str"},
            "Scene Type": {"type": "str"},
        },
        "episode": {
            "image_list": {"type": "array", "length": 1, "items": {"type": "str"}},
            "steps": {
                "type": "array",
                "length": 1,
                "items": {
                    "action": {
                        "open_gripper": {"type": "bool"},
                        "rotation_delta": {"type": "array", "length": 3, "items": {"type": "float"}},
                        "terminate_episode": {"type": "float"},
                        "world_vector": {"type": "array", "length": 3, "items": {"type": "float"}},
                    },
                    "is_first": {"type": "bool"},
                    "is_last": {"type": "bool"},
                    "is_terminal": {"type": "bool"},
                    "observation": {
                        "image": {"bytes": {"type": "NoneType"}, "path": {"type": "NoneType"}},
                        "natural_language_embedding": {"type": "array", "length": 512, "items": {"type": "float"}},
                        "natural_language_instruction": {"type": "str"},
                        "state": {"type": "array", "length": 7, "items": {"type": "float"}},
                    },
                    "reward": {"type": "float"},
                },
            },
        },
    }


from embdata.describe import describe_keys
from embdata.describe import full_paths

def test_describe_keys_single_level():
    data = {"a": 1, "b": 2, "c": 3}
    keys = describe_keys(data)
    assert keys == {"a": "a", "b": "b", "c": "c"}

def test_describe_keys_nested():
    data = {"a": 1, "b": {"c": 2, "d": 3}}
    keys = describe_keys(data)
    assert keys == {"a": "a", "c": "b.c", "d": "b.d", "b": "b",'b.c': 'b.c', 'b.d': 'b.d'}

def test_describe_keys_nested_list():
    data = {"a": 1, "b": [{"c": 2, "d": 3}, {"c": 4, "d": 5}]}
    keys = describe_keys(data)
    assert keys == {"a": "a", "b": "b", "b.c": "b.*.c", "b.d": "b.*.d", "c": "b.*.c", "d": "b.*.d"}

def test_describe_keys_include():
    data = {"a": 1, "b": {"c": 2, "d": 3}}
    keys = full_paths(data, keys=["a", "c"])
    assert keys == {"a": "a", "c": "b.c"}

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
