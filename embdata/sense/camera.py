from typing import List
import numpy as np
from pydantic import Field, BaseModel
from embdata.ndarray import NumpyArray
from mbodied.types.sample import Sample

class Intrinsics(Sample):
    """Model for Camera Intrinsic Parameters."""
    focal_length_x: float = Field(default=0.0, description="Focal length in x-direction")
    focal_length_y: float = Field(default=0.0, description="Focal length in y-direction")
    optical_center_x: float = Field(default=0.0, description="Optical center in x-direction")
    optical_center_y: float = Field(default=0.0, description="Optical center in y-direction")

    def matrix(self) -> np.ndarray:
        """Convert the intrinsic parameters to a 3x3 matrix."""
        return np.array([
            [self.focal_length_x, 0.0, self.optical_center_x],
            [0.0, self.focal_length_y, self.optical_center_y],
            [0.0, 0.0, 1.0],
        ])


class Extrinsics(Sample):
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], description="Rotation matrix of the camera")
    translation: List[float] = Field(default=[0.0, 0.0, 0.0], description="Translation vector of the camera")

class Distortion(Sample):
    """Model for Camera Distortion Parameters."""
    k1: float = Field(default=0.0, description="Radial distortion coefficient k1")
    k2: float = Field(default=0.0, description="Radial distortion coefficient k2")
    p1: float = Field(default=0.0, description="Tangential distortion coefficient p1")
    p2: float = Field(default=0.0, description="Tangential distortion coefficient p2")
    k3: float = Field(default=0.0, description="Radial distortion coefficient k3")

class CameraParams(Sample):
    """Model for Camera Parameters."""
    intrinsic: Intrinsics = Field(default_factory=Intrinsics, description="Intrinsic parameters of the camera")
    distortion: Distortion = Field(default_factory=Distortion, description="Distortion parameters of the camera")
    extrinsic: Extrinsics = Field(default_factory=Extrinsics, description="Extrinsic parameters of the camera")
    depth_scale: float = Field(default=0.001, description="Depth scale of the camera")

if __name__ == "__main__":
    # Create Intrinsics instance
    intrinsic_params = Intrinsics(
        focal_length_x=911.0, 
        focal_length_y=911.0, 
        optical_center_x=653.0, 
        optical_center_y=371.0
    )

    # Create CameraParams instance with Intrinsics instance
    camera_params = CameraParams(
        intrinsic=intrinsic_params,
        # You can access matrix if needed
        # matrix = intrinsic_params.matrix()
    )

    