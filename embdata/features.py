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


def to_features_dict(indict, exclude_keys=None) -> Dict[str, Any]:
    """Convert a dictionary to a Datasets Features object.

    This function recursively converts a nested dictionary into a format compatible with
    Hugging Face Datasets' Features. It handles various data types including strings,
    integers, floats, lists, and PIL Images.

    Args:
        indict (Union[dict, str, int, float, list, tuple, np.ndarray, PILImage]): The input to convert.
        exclude_keys (set, optional): A set of keys to exclude from the conversion. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary representation of the Features object for Hugging Face Datasets.

    Raises:
        ValueError: If an empty list is provided or if the input type is not supported.

    Examples:
        >>> to_features_dict({"name": "Alice", "age": 30})
        {'name': Value(dtype='string', id=None), 'age': Value(dtype='int64', id=None)}

        >>> to_features_dict({"scores": [85, 90, 95]})
        {'scores': [Value(dtype='int64', id=None)]}

        >>> import numpy as np
        >>> to_features_dict({"data": np.array([1, 2, 3])})
        {'data': [Value(dtype='int64', id=None)]}

        >>> from PIL import Image
        >>> img = Image.new("RGB", (60, 30), color="red")
        >>> to_features_dict({"image": img})
        {'image': Image(decode=True, id=None)}

        >>> to_features_dict({"nested": {"a": 1, "b": "text"}})
        {'nested': {'a': Value(dtype='int64', id=None), 'b': Value(dtype='string', id=None)}}

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
            raise ValueError("Cannot infer schema from empty list")
        return [to_features_dict(indict[0])]

    if isinstance(indict, dict):
        out_dict = {}
        for key, value in indict.items():
            if key in exclude_keys:
                continue
            out_dict[key] = to_features_dict(value)
        return out_dict
    if isinstance(indict, PILImage):
        return HFImage()
    raise ValueError(f"Cannot infer schema from {type(indict)}")
