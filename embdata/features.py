# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from typing import Any, Dict

import numpy as np
from datasets import Features, Value
from datasets import Image as HFImage
from PIL.Image import Image as PILImage

IDEFICS_FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}],
        "images": [HFImage()],
    },
)

PHI_FEATURES = Features({"messages": [{"role": Value("string"), "content": Value("string")}], "images": [HFImage()]})


def to_features_dict(indict: Any, exclude_keys: set | None = None) -> Dict[str, Any]:
    """Convert a dictionary to a Datasets Features object.

    This function recursively converts a nested dictionary into a format compatible with
    Hugging Face Datasets' Features. It handles various data types including strings,
    integers, floats, lists, and PIL Images.

    Args:
        indict: The input to convert. Can be a dictionary, string, int, float, list, tuple, numpy array, or PIL Image.
        exclude_keys: A set of keys to exclude from the conversion. Defaults to None.

    Returns:
        A dictionary representation of the Features object for Hugging Face Datasets.

    Raises:
        ValueError: If an empty list is provided or if the input type is not supported.

    Examples:
        Simple dictionary conversion:
        >>> to_features_dict({"name": "Alice", "age": 30})
        {'name': Value(dtype='string', id=None), 'age': Value(dtype='int64', id=None)}

        List conversion:
        >>> to_features_dict({"scores": [85, 90, 95]})
        {'scores': [Value(dtype='int64', id=None)]}

        Numpy array conversion:
        >>> import numpy as np
        >>> to_features_dict({"data": np.array([1, 2, 3])})
        {'data': [Value(dtype='int64', id=None)]}

        PIL Image conversion:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (60, 30), color="red")
        >>> to_features_dict({"image": img})
        {'image': Image(decode=True, id=None)}

        Nested structure with image and text:
        >>> complex_data = {
        ...     "user_info": {"name": "John Doe", "age": 28},
        ...     "posts": [
        ...         {"text": "Hello, world!", "image": Image.new("RGB", (100, 100), color="blue"), "likes": 42},
        ...         {"text": "Another post", "image": Image.new("RGB", (200, 150), color="green"), "likes": 17},
        ...     ],
        ... }
        >>> features = to_features_dict(complex_data)
        >>> features
        {
            'user_info': {
                'name': Value(dtype='string', id=None),
                'age': Value(dtype='int64', id=None)
            },
            'posts': [
                {
                    'text': Value(dtype='string', id=None),
                    'image': Image(decode=True, id=None),
                    'likes': Value(dtype='int64', id=None)
                }
            ]
        }
    """
    if exclude_keys is None:
        exclude_keys = set()
    if isinstance(indict, str):
        return Value("string")
    if isinstance(indict, int | np.integer | np.uint8):
        return Value("int64")
    if isinstance(indict, float):
        return Value("double")

    if isinstance(indict, list | tuple | np.ndarray):
        if len(indict) == 0:
            msg = "Cannot infer schema from empty list"
            raise ValueError(msg)
        return [to_features_dict(indict[0])]

    if isinstance(indict, dict):
        out_dict = {}
        for key, value in indict.items():
            if key in exclude_keys:
                continue
            out_dict[key] = to_features_dict(value, exclude_keys)
        return out_dict
    if hasattr(indict, "pil"):
        return HFImage()
    if isinstance(indict, PILImage):
        return HFImage()

    msg = f"Cannot infer schema from {type(indict)}"
    raise ValueError(msg)
