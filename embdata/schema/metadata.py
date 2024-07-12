from pydantic import Field

from embdata.sample import Sample


class Metadata(Sample):
    num_depth_cams: int | str = Field(0, alias="# Depth Cams")
    num_episodes: int | str = Field(0, alias="# Episodes")
    num_rgb_cams: int | str = Field(0, alias="# RGB Cams")
    num_wrist_cams: int | str = Field(0, alias="# Wrist Cams")
    action_space: str = Field("", alias="Action Space")
    control_frequency: str = Field("", alias="Control Frequency")
    data_collect_method: str = Field("", alias="Data Collect Method")
    dataset_name: str = Field("", alias="Dataset")
    description: str = Field("", alias="Description")
    file_size_gb: float | str = Field(0.0, alias="File Size (GB)")
    gripper: str = Field("", alias="Gripper")
    has_camera_calibration: str = Field("", alias="Has Camera Calibration?")
    has_proprioception: str = Field("", alias="Has Proprioception?")
    has_suboptimal: str = Field("", alias="Has Suboptimal?")
    language_annotations: float | str = Field(0.0, alias="Language Annotations")
    registered_dataset_name: str = Field("", alias="Registered Dataset Name")
    robot: str = Field("", alias="Robot")
    robot_morphology: str = Field("", alias="Robot Morphology")
    scene_type: str = Field("", alias="Scene Type")

