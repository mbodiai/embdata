from typing import List

from pydantic import Field

from embdata.geometry import Pose
from embdata.ndarray import NumpyArray
from embdata.sample import Sample
from embdata.sense.camera import CameraParams
from embdata.sense.image import Image


class SceneObject(Sample):
    """Model for Scene Object Poses."""

    name: str = ""
    pose: Pose = Field(default_factory=Pose)
    volume: float | None = None
    bbox_3d: NumpyArray[6, float] | None = None
    pixel_coords: NumpyArray[2, float] | None = None
    bbox_pixels: NumpyArray[4, float] | None = None
    neighbors: List[str] | None = Field(default=None, description="List of neighboring objects")


class Scene(Sample):
    """Model for Scene Data."""

    objects: List[SceneObject] = Field(default_factory=list, description="List of scene objects")
    image: Image | None = None
    annotated: Image | None = None
    depth: Image | None = None
    camera_params: CameraParams = Field(default_factory=CameraParams, description="Camera parameters of the scene")
