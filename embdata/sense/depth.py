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
    PrivateAttr,
    computed_field,
)
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import Literal

from embdata.ndarray import NumpyArray
from embdata.sense.image import Image

SupportsImage = Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]  # noqa: UP007

DepthArrayLike = NumpyArray[1, Any, Any, np.uint16] | NumpyArray[Any, Any, np.uint16]

class Depth(Image):
    """A class for representing depth images and points."""
    DEFAULT_MODE = "I"
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] = DEFAULT_MODE
    points: NumpyArray[Any, 3,  np.float32] | None = None
    encoding: Literal["png"] = "png"
    _rgb: NumpyArray[Any, Any, 3, np.uint8] | None = PrivateAttr(default=None)

    @computed_field(return_type=DepthArrayLike)
    @cached_property
    def array(self) -> DepthArrayLike:
        """The image represented as a NumPy array."""
        return np.array(self.pil)

    @computed_field(return_type=NumpyArray[Any,Any,3,np.uint8])
    @cached_property
    def rgb(self) -> NumpyArray[Any,Any,3,np.uint8]:
        """The rgb image represented as a NumPy array."""
        return self._rgb

    def __init__(  # noqa
        self,
        arg: SupportsImage | None = None,  # type: ignore
        path: str | None = None,
        array: np.ndarray | None = None,
        base64: Base64Str | None = None,
        encoding: str = "png",
        size: Tuple[int, ...] | None = None,
        bytes: SupportsBytes | None = None,  # noqa
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
        kwargs["encoding"] = encoding or "png"
        kwargs["path"] = path
        kwargs["size"] = size[:2] if isinstance(size, Tuple) else size if size is not None else (224,224)
        kwargs["mode"] = mode
        kwargs["array"] = array
        kwargs["base64"] = base64
        kwargs["bytes"] = bytes
        if isinstance(arg, Image):
            kwargs.update(arg.model_dump())
            rgb = arg.array
            kwargs["rgb"] = rgb
            kwargs["pil"] = arg.pil.convert("I")
        elif isinstance(arg, np.ndarray) and arg.ndim == 3 and arg.shape[2] == 3:
            rgb = arg.astype(np.uint8)
        else:
            rgb = None
        if array is not None:
            rgb = array if array.ndim == 3 else array[..., :3]
        if arg is None:
            for k, v in kwargs.items():
                if k in self.SOURCE_TYPES and v is not None:
                    arg = kwargs.pop(k)
                    break
            if arg is None and kwargs.get("size") is not None:
                arg = np.zeros(kwargs["size"] + (3,), dtype=np.uint16)
        if arg is None:
            arg = np.zeros((224, 224, 3), dtype=np.uint16)
        super().__init__(**kwargs)
        self.array = np.array(self.pil, dtype=np.uint16)
        self._rgb = rgb

    def __repr__(self):
        """Return a string representation of the image."""
        if self.base64 is None:
            return f"Image(encoding={self.encoding}, size={self.size})"
        return f"Image(base64={self.base64[:10]}..., encoding={self.encoding}, size={self.size})"

    def __str__(self):
        """Return a string representation of the image."""
        return f"Image(base64={self.base64[:10]}..., encoding={self.encoding}, size={self.size})"

    @staticmethod
    def from_base64(base64_str: str, encoding: str, size=None) -> "Image":
        """Decodes a base64 string to create an Image instance.

        This method takes a base64 encoded string representation of an image,
        decodes it, and creates an Image instance from it. It's useful when
        you have image data in base64 format and want to work with it as an Image object.

        Args:
            base64_str (str): The base64 string to decode.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.

        Example:
            >>> base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
            >>> image = Image.from_base64(base64_str, encoding="png", size=(1, 1))
            >>> print(image.size)
            (1, 1)

            # Example with complex nested structure
            >>> nested_data = {
            ...     "image": Image.from_base64(base64_str, encoding="png"),
            ...     "metadata": {"text": "A small red square", "tags": ["red", "square", "small"]},
            ... }
            >>> print(nested_data["image"].size)
            (1, 1)
            >>> print(nested_data["metadata"]["text"])
            A small red square
        """
        image_data = base64lib.b64decode(base64_str)
        image = PILModule.open(io.BytesIO(image_data))
        return Image(image, encoding, size)

    @staticmethod
    def open(path: str, encoding: str = "jpeg", size=None) -> "Image":
        """Opens an image from a file path and creates an Image instance.

        This method reads an image file from the specified path, 
        and creates an Image instance from it. It's a convenient way to load images from
        your local file system.

        Args:
            path (str): The path to the image file.
            encoding (str): The format used for encoding the image when converting to base64.
                            Defaults to "jpeg".
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
                                              If provided, the image will be resized.

        Returns:
            Image: An instance of the Image class with populated fields.

        Example:
            >>> image = Image.open("/path/to/image.jpg", encoding="jpeg", size=(224, 224))
            >>> print(image.size)
            (224, 224)
        """
        image = PILModule.open(path)
        return Image(image, encoding, size)

    @staticmethod
    def pil_to_data(image: PILImage, encoding: str, size=None) -> dict:
        """Creates an Image instance from a PIL image.

        Args:
            image (PIL.Image.Image): The source PIL image from which to create the Image instance.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        if encoding.lower() == "jpg":
            encoding = "jpeg"
        buffer = io.BytesIO()
        image.save(buffer, format=encoding.upper())
        base64_encoded = base64lib.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/{encoding};base64,{base64_encoded}"
        if size is not None:
            image = image.resize(size)
        else:
            size = image.size
        return {
            "array": np.array(image),
            "base64": base64_encoded,
            "pil": image,
            "size": size,
            "url": data_url,
            "encoding": encoding.lower(),
        }

    @staticmethod
    def load_url(url: str, download=False) -> PILImage | None:
        """Downloads an image from a URL or decodes it from a base64 data URI.

        This method can handle both regular image URLs and base64 data URIs.
        For regular URLs, it downloads the image data. For base64 data URIs,
        it decodes the data directly. It's useful for fetching images from
        the web or working with inline image data.

        Args:
            url (str): The URL of the image to download, or a base64 data URI.
            download (bool): If True, prompts the user before downloading. Defaults to False.

        Returns:
            PIL.Image.Image | None: The downloaded and decoded image as a PIL Image object,
                                    or None if the download fails or is cancelled.

        Example:
            >>> image = Image.load_url("https://example.com/image.jpg")
            >>> if image:
            ...     print(f"Image size: {image.size}")
            ... else:
            ...     print("Failed to load image")

            >>> data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
            >>> image = Image.load_url(data_uri)
            >>> if image:
            ...     print(f"Image size: {image.size}")
            ... else:
            ...     print("Failed to load image")
        """
        if url.startswith("data:image"):
            # Extract the base64 part of the data URI
            base64_str = url.split(";base64", 1)[1]
            image_data = base64lib.b64decode(base64_str)
            return PILModule.open(io.BytesIO(image_data))

        try:
            # Open the URL and read the image data
            import urllib.request

            user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
            headers = {
                "User-Agent": user_agent,
            }
            if download:
                accept = input("Do you want to download the image? (y/n): ")
                if "y" not in accept.lower():
                    return None
            if not url.startswith("http"):
                raise ValueError("URL must start with 'http' or 'https'.")
            request = urllib.request.Request(url, None, headers)  # noqa
            response = urllib.request.urlopen(request)  # noqa
            data = response.read()  # The data u need
            return PILModule.open(io.BytesIO(data))
        except Exception as e:
            logging.warning(f"Failed to load image from URL: {url}. {e}")
            logging.warning("Not validating the Image data")
            return None

    @classmethod
    def from_bytes(cls, bytes_data: bytes, encoding: str = "jpeg", size=None) -> "Image":
        """Creates an Image instance from a bytes object.

        Args:
            bytes_data (bytes): The bytes object to convert to an image.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        image = PILModule.open(io.BytesIO(bytes_data))
        return cls(image, encoding, size)

    @staticmethod
    def bytes_to_data(bytes_data: bytes, encoding: str = "jpeg", size=None) -> dict:
        """Creates an Image instance from a bytes object.

        Args:
            bytes_data (bytes): The bytes object to convert to an image.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        image = PILModule.open(io.BytesIO(bytes_data))
        return Image.pil_to_data(image, encoding, size)

    @model_validator(mode="before")
    @classmethod
    def validate_kwargs(cls, values) -> dict:  # noqa: PLR0912
        # Ensure that exactly one image source is provided
        provided_fields = [
            k for k in values if values[k] is not None and k in ["array", "base64", "path", "pil", "url"]
        ]
        if len(provided_fields) > 1:
            raise ValueError(f"Multiple image sources provided; only one is allowed but got: {provided_fields}")

        # Initialize all fields to None or their default values and add points logic
        validated_values = {
            "array": None,
            "base64": None,
            "encoding": values.get("encoding", "jpeg").lower(),
            "path": None,
            "pil": None,
            "url": None,
            "size": values.get("size", None),
        }
        # Basic point cloud logic
        if "points" in values and values["points"] is not None:
            validated_values["points"] = values["points"]
        else:
            validated_values["points"] = np.zeros((3, 0), dtype=float)

        # Validate the encoding first
        if validated_values["encoding"] not in ["png", "jpeg", "jpg", "bmp", "gif"]:
            msg = "The 'encoding' must be a valid image format (png, jpeg, jpg, bmp, gif)."
            raise ValueError(msg)

        if "bytes" in values and values["bytes"] is not None:
            validated_values.update(
                cls.bytes_to_data(values["bytes"], values["encoding"], values["size"])
            )
            return validated_values

        if "pil" in values and values["pil"] is not None:
            validated_values.update(
                cls.pil_to_data(values["pil"], values["encoding"], values["size"]),
            )
            return validated_values
        # Process the provided image source
        if "path" in provided_fields:
            image = PILModule.open(values["path"])
            validated_values["path"] = values["path"]
            validated_values.update(
                cls.pil_to_data(image, validated_values["encoding"], validated_values["size"])
            )

        elif "array" in provided_fields:
            image = PILModule.fromarray(values["array"])
            validated_values.update(
                cls.pil_to_data(image, validated_values["encoding"], validated_values["size"])
            )

        elif "pil" in provided_fields:
            validated_values.update(
                cls.pil_to_data(values["pil"], validated_values["encoding"], validated_values["size"]),
            )

        elif "base64" in provided_fields:
            validated_values.update(
                cls.from_base64(
                    values["base64"], validated_values["encoding"], validated_values["size"]
                ),
            )

        elif "url" in provided_fields:
            url_path = urlparse(values["url"]).path
            file_extension = (
                Path(url_path).suffix[1:].lower() if Path(url_path).suffix else validated_values["encoding"]
            )
            validated_values["encoding"] = file_extension
            image = cls.load_url(values["url"])
            if image is None:
                validated_values["array"] = np.zeros((224, 224, 3), dtype=np.uint8)
                validated_values["size"] = (224, 224)
                return validated_values

            validated_values.update(cls.pil_to_data(image, file_extension, validated_values["size"]))
            validated_values["url"] = values["url"]

        elif "size" in values and values["size"] is not None:
            array = np.zeros((values["size"][0], values["size"][1], 3), dtype=np.uint8)
            image = PILModule.fromarray(array)
            validated_values.update(
                cls.pil_to_data(image, validated_values["encoding"], validated_values["size"])
            )
        if any(validated_values[k] is None for k in ["array", "base64", "pil", "url"]):
            logging.warning(
                f"Failed to validate image data. Could only fetch {[k for k in validated_values if validated_values[k] is not None]}",
            )
        return validated_values

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