# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrap any common image representation in an Image class to convert to any other common format.

The following image representations are supported:
- NumPy array
- PIL Image
- Base64 encoded string
- File path
- URL
- Bytes object

The image can be resized to and from any size, compressed, and converted to and from any supported format:

```python
image = Image("path/to/image.png", size=new_size_tuple).save("path/to/new/image.jpg")
image.save("path/to/new/image.jpg", quality=5)

TODO: Implement Lazy attribute loading for the image data.
"""

import tempfile
from functools import cached_property, wraps
from typing import Any, List, SupportsBytes, Tuple, Union

import cv2
import numpy as np
from PIL import Image as PILModule
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    FilePath,
    InstanceOf,
    computed_field,
    model_serializer,
    model_validator,
)
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import Literal

from embdata.ndarray import NumpyArray
from embdata.sense.image import Image

from typing import Any, ClassVar, Dict, List, SupportsBytes, Tuple

SupportsImage = Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]  # noqa: UP007

DepthArrayLike = NumpyArray[1,Any, Any, np.uint16] | NumpyArray[Any, Any, np.uint16]
class Depth(Image):
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] = "I"
    def __init__( # noqa
        self,
        arg: SupportsImage | None = None, # type: ignore
        path: str | None = None,
        array: np.ndarray | None = None,
        base64: Base64Str | None = None,
        encoding: str = "jpeg",
        size: Tuple[int, ...] | None = None,
        bytes: SupportsBytes | None = None, # noqa
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = "I",
        **kwargs,
    ):
        """Initializes a Depth representation. Unlike the Image class, an empty array is used as the default image.

        Args:
            arg (SupportsImage, optional): The primary image source.
            url (Optional[str], optional): The URL of the image.
            path (Optional[str], optional): The file path of the image.
            base64 (Optional[str], optional): The base64 encoded string of the image.
            array (Optional[np.ndarray], optional): The numpy array of the image.
            pil (Optional[PILImage], optional): The PIL image object.
            encoding (Optional[str], optional): The encoding format of the image. Defaults to 'jpeg'.
            size (Optional[Tuple[int, int]], optional): The size of the image as a (width, height) tuple.
            bytes (Optional[bytes], optional): The bytes object of the image.
            mode (Optional[str], optional): The mode to use for the image. Defaults to 'RGB'.
            **kwargs: Additional keyword arguments.
        """
        kwargs["encoding"] = encoding or "jpeg"
        kwargs["path"] = path
        kwargs["size"] = size[:2] if isinstance(size, Tuple) else size
        kwargs["mode"] = mode
        kwargs["array"] = array
        kwargs["base64"] = base64
        kwargs["bytes"] = bytes
        if isinstance(arg, Image):
            kwargs.update(arg.model_dump())
            arg = None
        if arg is None:
            for k, v in kwargs.items():
                if k in self.SOURCE_TYPES and v is not None:
                    arg = kwargs.pop(k)
                    break
            if arg is None and kwargs.get("size") is not None:
                arg = np.zeros(kwargs["size"] + (3,), dtype=np.uint8)
        kwargs = Image.dispatch_arg(arg, **kwargs)
        super().__init__(**kwargs)
        self.array = np.array(self.pil, dtype=np.uint16)
        self._rgb = rgb

    @computed_field(return_type=DepthArrayLike)
    @cached_property
    def array(self) -> DepthArrayLike:
        """The image represented as a NumPy array."""
        return np.array(self.pil)
    def cluster_points(self, n_clusters: int = 3) -> List[int]:
        """Cluster the points using KMeans.

        Args:
            n_clusters (int): The number of clusters to form.

        Returns:
            List[int]: The cluster labels for each point.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(self.points.T)

    def segment_plane(self, min_samples=3, threshold: float = 0.01, max_trials: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the largest plane using RANSAC."""
        ransac = RANSACRegressor(min_samples=min_samples,residual_threshold=threshold, max_trials=max_trials)
        ransac.fit(self.points[:2].T, self.points[2])
        inlier_mask = ransac.inlier_mask_
        plane_coefficients = np.append(ransac.estimator_.coef_, ransac.estimator_.intercept_)
        return inlier_mask, plane_coefficients



    def colormap(self, depth_scale=1.0, path=None, **kwargs) -> Image:
        """Postprocess the predicted depth tensor."""
        depth_normalized = cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX)
        
        depth_8bit = depth_normalized.astype('uint8')

        import platform
        import matplotlib
        if platform.system() == "Darwin":
            matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        colormap_image = plt.cm.inferno(depth_8bit / 255.0)
        colormap_image_rgb = (colormap_image[..., :3] * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.imsave(f.name, colormap_image_rgb)
            saved_image_path = f.name

            return Image(saved_image_path, mode="RGB")

    def show(self) -> None:
        Image(self.colormap()).show()

    @wraps(Image.save, assigned=( "__doc__"))
    def save(self,*args, **kwargs) -> None:
        """Save the image to a file."""
        self.colormap().save(*args, **kwargs)

    def segment_cylinder(self, min_samples=3, threshold: float = 0.01, max_trials: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the largest cylinder using RANSAC.

        Args:
            min_samples (int): The minimum number of data points to fit a model.
            threshold (float): The maximum distance for a point to be considered as an inlier.
            max_trials (int): The maximum number of iterations for RANSAC.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The inlier points and their indices.
        """
        poly = PolynomialFeatures(degree=2)
        ransac = make_pipeline(poly, RANSACRegressor(min_samples=min_samples, residual_threshold=threshold, max_trials=max_trials))

        X = self.points[:, :2]  # Assuming self.points is of shape (n_points, 3)
        y = self.points[:, 2]

        ransac.fit(X, y)

        inlier_mask = ransac.named_steps["ransacregressor"].inlier_mask_
        inlier_points = self.points[inlier_mask]
        inlier_indices = np.where(inlier_mask)[0]

        return inlier_points, inlier_indices


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = Image(
        "https://github.com/mbodiai/embodied-agents/blob/main/resources/depth_image.png?raw=true",
        encoding="png",
        mode="RGB",
    )
    # print(img)
    # img.save("rgb.png")
    # print(img.numpy()[20, 0])
    depth = Depth( "https://github.com/mbodiai/embodied-agents/blob/main/resources/depth_image.png?raw=true")
    print(depth)
    img  = depth.colormap(path="colormap.png")
    img.save("depth.png")
    # depth.show()
    # depth.colormap(path="depth.png")