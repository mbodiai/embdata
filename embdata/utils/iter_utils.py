import functools
import operator
import re
from collections import defaultdict as DefaultDict
from functools import reduce
from itertools import zip_longest
from typing import Any, Dict, List, Literal, Union
from embdata.describe import describe_keys
import numpy as np
import torch
from datasets import Dataset
exists_iter = lambda key, c: c is not None and len(c) > 0 and (hasattr(c[0], key) or key in c[0])
get_iter = lambda key, c: None if not exists_iter(key, c) else c[0][key] if key in c[0] else getattr(c[0], key)
get_iter_class = lambda key, c: get_iter(key, c).__class__ if get_iter(key, c) is not None else None
from pydantic import BaseModel, ConfigDict

def flatten(
    obj: Any,
    output_type: Literal["list", "dict", "numpy", "torch"] = "list",
    non_numerical: Literal["ignore", "forbid", "allow"] = "allow",
    ignore: set[str] | None = None,
    sep: str = ".",
    to: Union[str, set[str], List[str], None] = None,
) -> Union[List[Any], Dict[str, Any], np.ndarray, torch.Tensor]:
    """
    Flatten a nested structure (dict, list, tuple) into a flat structure.
    
    Args:
        obj: The object to flatten.
        output_type: The type of output to return.
        non_numerical: How to handle non-numerical values.
        ignore: Set of keys to ignore during flattening.
        sep: Separator for nested keys in the flattened dictionary.
        to: Keys to extract from the flattened structure.

    Returns:
        Flattened structure as specified by output_type.
    """
    accumulator = DefaultDict(list)

    def add_to_accumulator(key, value):
        if ignore and key in ignore:
            return
        if non_numerical == "forbid" and not isinstance(value, (int, float, np.number, torch.Tensor)):
            return
        if non_numerical == "ignore" and not isinstance(value, (int, float, np.number, torch.Tensor)):
            return
        accumulator[key].append(value)

    def recurse(current_obj, current_path=""):
        if isinstance(current_obj, dict):
            for k, v in current_obj.items():
                new_path = f"{current_path}{sep}{k}" if current_path else k
                recurse(v, new_path)
        elif isinstance(current_obj, (list, tuple)):
            for i, v in enumerate(current_obj):
                new_path = f"{current_path}[{i}]"
                recurse(v, new_path)
        else:
            add_to_accumulator(current_path, current_obj)

    recurse(obj)

    if to:
        to_keys = set(to) if isinstance(to, (list, set)) else {to}
        accumulator = {k: v for k, v in accumulator.items() if any(re.match(pattern, k) for pattern in to_keys)}

    if output_type == "dict":
        return dict(accumulator)
    elif output_type == "numpy":
        return np.array(list(accumulator.values()))
    elif output_type == "torch":
        return torch.tensor(list(accumulator.values()))
    else:  # list
        if to:
            return list(accumulator.values())
        else:
            return [item for sublist in accumulator.values() for item in (sublist if isinstance(sublist, list) else [sublist])]

