# Copyright 2024 Mbodi AI
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
"""A base model class for serializing, recording, and manipulating arbitray data.

It was designed to be extensible, flexible, yet strongly typed. In addition to
supporting any json API out of the box, it can be used to represent
arbitrary action and observation spaces in robotics and integrates seemlessly with H5, Gym, Arrow,
PyTorch, numpy, and HuggingFace Datasets.

Methods:
    schema: Get a simplified json schema of your data.
    to: Convert the Sample instance to a different container type:
        -
    default_value: Get the default value for the Sample instance.
    unflatten: Unflatten a one-dimensional array or dictionary into a Sample instance.
    flatten: Flatten the Sample instance into a one-dimensional array or dictionary.
    space_for: Default Gym space generation for a given value.
    init_from: Initialize a Sample instance from a given value.
    from_space: Generate a Sample instance from a Gym space.
    pack_from: Pack a list of samples into a single sample with lists for attributes.
    unpack: Unpack the packed Sample object into a list of Sample objects or dictionaries.
    dict: Return the Sample object as a dictionary with None values excluded.
    field_info: Get the FieldInfo for a given attribute key.
    space: Return the corresponding Gym space for the Sample instance based on its instance attributes.
    random_sample: Generate a random Sample instance based on its instance attributes.

Examples:
    >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    >>> sample.flatten()
    [1, 2, 3, 4, 5]
    >>> sample.schema()
    {'type': 'object',
        'properties': {
            'x': {'type': 'number'},
            'y': {'type': 'number'},
            'z': {'type': 'object'},
        'properties':
        {
        'a':{'type': 'number'},
        'b': {'type': 'number'}
        }
    },
    'extra_field': {
        'type': 'number'
    }
    >>> Sample.unflatten(flat_list, schema)
    Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
"""

import copy
import functools
import logging
import operator
import re
from enum import Enum
from functools import cached_property, reduce
from importlib import import_module
from itertools import zip_longest
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Literal, Union, get_origin

import numpy as np
import torch
from datasets import Dataset, Features, IterableDataset
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from embdata.describe import describe, full_paths
from embdata.features import to_features_dict

OneDimensional = Annotated[Literal["dict", "np", "pt", "list", "sample"], "Numpy, PyTorch, list, sample, or dict"]


class CallableItems:
    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        if isinstance(self.obj, dict):
            yield from dict(self.obj).items()
        elif isinstance(self.obj, list | tuple | np.ndarray | torch.Tensor | Dataset | IterableDataset):
            yield from enumerate(self.obj)
        else:
            yield from self.obj

    def __iter__(self):
        return iter(self.obj)

    def __len__(self):
        return len(self.obj)

    def __getitem__(self, key):
        return self.obj[key]


class Sample(BaseModel):
    """A base model class for serializing, recording, and manipulating arbitray data."""

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=False,
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        from_attributes=True,
    )

    def __init__(self, wrapped=None, **data):
        """A base model class for serializing, recording, and manipulating arbitray data.

        It accepts any keyword arguments and endows them with the following methods:

        Methods:
            schema: Get a simplified json schema of your data.
            to: Convert the Sample instance to a different container type:
                -
            default_value: Get the default value for the Sample instance.
            unflatten: Unflatten a one-dimensional array or dictionary into a Sample instance.
            flatten: Flatten the Sample instance into a one-dimensional array or dictionary.
            space_for: Default Gym space generation for a given value.
            init_from: Initialize a Sample instance from a given value.
            from_space: Generate a Sample instance from a Gym space.
            pack_from: Pack a list of samples into a single sample with lists for attributes.
            unpack: Unpack the packed Sample object into a list of Sample objects or dictionaries.
            dict: Return the Sample object as a dictionary with None values excluded.
            field_info: Get the FieldInfo for a given attribute key.
            space: Return the corresponding Gym space for the Sample instance based on its instance attributes.
            random_sample: Generate a random Sample instance based on its instance attributes.

        Examples:
            >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
            >>> sample.flatten()
            [1, 2, 3, 4, 5]
            >>> sample.schema()
            {'type': 'object',
                'properties': {
                    'x': {'type': 'number'},
                    'y': {'type': 'number'},
                    'z': {'type': 'object'},
                'properties':
                {
                'a':{'type': 'number'},
                'b': {'type': 'number'}
                }
            },
            'extra_field': {
                'type': 'number'
            }
            >>> Sample.unflatten(flat_list, schema)
            Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
        """
        if isinstance(wrapped, Sample):
            # Only wrap if no other data is provided.
            if not data:
                data = {k: v for k, v in wrapped.model_dump() if not k.startswith("_")}
        elif isinstance(wrapped, dict):
            if not data:
                data = {
                    k: Sample(**v) if isinstance(v, dict) else v for k, v in wrapped.items() if not k.startswith("_")
                }
        elif self.__class__ == Sample:
            # Only the Sample class can wrap an arbitrary type.
            if isinstance(wrapped, list | tuple | np.ndarray | torch.Tensor | Dataset):
                # There is no schema to unflatten from, just have it as an attribute.
                data["_items"] = wrapped
                data["wrapped"] = wrapped
            elif wrapped is not None:
                data["wrapped"] = wrapped
        elif isinstance(wrapped, list | tuple | np.ndarray | torch.Tensor | Dataset):
            # Derived classes have a schema to unflatten from.
            d = self.__class__.unflatten(wrapped).model_dump()
            data.update(d)
        elif isinstance(wrapped, spaces.Space):
            data.update(self.from_space(wrapped).model_dump())

        super().__init__(**data)
        if "_items" in data:
            self.items = CallableItems(data["_items"])
        self.__post_init__()

    def __getitem__(self, key: str | int) -> Any:
        """Return the value of the attribute with the specified key.

        If the key is an integer and the Sample object wraps a list, the value is returned at the specified index.
        If the key is a string and contains a separator ('.' or '/'), the value is returned at the specified nested key.
        Otherwise, the value is returned as an attribute of the Sample object.
        """
        if isinstance(key, int) and any(isinstance(self[k], list | tuple | Dataset | IterableDataset | np.ndarray) for k in self.keys()):
            items = [self[k] for k in self.keys()]
            return items[key]

        if self.__class__ == Sample:
            if isinstance(key, int):
                if hasattr(self, "_items"):
                    return self._items[key]
                if hasattr(self, "wrapped") and isinstance(self.wrapped, List | Dataset):
                    return self.wrapped[key]

                items = getattr(self, "items", None)
                items = [] if items is None else self._items if hasattr(self, "_items") else self.values()
                if isinstance(items, Generator):
                    items = functools.reduce(lambda x, y: x + y, items, [])
                if callable(items):
                    items = list(items())
                if len(items) < key or key < 0:
                    msg = f"Index out of range: {key} (expected 0-{len(items) - 1})"
                    raise IndexError(msg)
                try:
                    return items[key]
                except Exception as e:
                    msg = f"Indexing not supported for {type(items)}: {items}"
                    raise TypeError(msg) from e
            if isinstance(key, int):
                msg = f"Sample object does not wrap a list but index was requested: {key}. Did you mean to call items? "
                raise TypeError(msg)

        if isinstance(key, str) and any(c in key for c in "./*"):
            sep = "." if "." in key else "/"
            keys = key.replace("*", "").replace(f"{sep}{sep}", sep).split(sep)
            obj = self
            for k in keys[:-1]:
                if k:
                    k = "_items" if k == "items" else k
                    obj = obj[k]
            k = keys[-1] if keys[-1] != "items" else "_items"
            return obj[k] if k is not None else obj

        try:
            return getattr(self, key)
        except (AttributeError, KeyError, TypeError) as e:
            if hasattr(self, "_extra"):
                try:
                    if isinstance(key, str):
                        sep = "." if "." in key else "/"
                        keys = key.replace("*", "").replace(f"{sep}{sep}", sep).split(sep)
                        key = "__nest__".join(keys)
                        return getattr(self._extra, key)
                except AttributeError:
                    pass
            msg = f"Key: `{key}` not found in Sample {self}. Try using sample[key] instead if key is an integer or contains special characters."
            raise KeyError(msg) from e

    def __setattr__(self, key: str, value: Any) -> None:
        """Set the value of the attribute with the specified key."""
        if self.__class__ == Sample and key == "items":
            super().__setattr__("_items", value)
        else:
            super().__setattr__(key, value)

    def __setitem__(self, key: str | int, value: Any) -> None:
        """Set the value of the attribute with the specified key.

        If the key is an integer and the Sample object wraps a list, the value is set at the specified index.
        If the key is a string and contains a separator ('.' or '/'), the value is set at the specified nested key.
        Otherwise, the value is set as an attribute of the Sample object.
        """
        if self.__class__ == Sample:
            if isinstance(key, int) and hasattr(self, "wrapped") and isinstance(self.wrapped, List | Dataset):
                self.wrapped[key] = value
            elif isinstance(key, int) and len(self) > key:
                msg = f"Index out of range: {key} (expected 0-{len(self) - 1})"
                raise IndexError(msg)
            if isinstance(key, int):
                msg = f"Sample object does not wrap a list but index was requested: {key}"
                raise TypeError(msg)
        if key in self:
            setattr(self, key, value)
            return
        if any(c in key for c in "./*"):
            sep = "." if "." in key else "/"
            keys = key.replace("*", "").replace(f"{sep}{sep}", sep).split(sep)
            obj = self
            for k in keys[:-1]:
                if k:
                    k = "_items" if k == "items" else k
                    if not hasattr(obj, k):
                        setattr(obj, k, Sample())
                    obj = obj[k]
            # print(f"keys: {keys}, value: {value}, obj: {obj}, type: {type(obj)}")
            key = keys[-1] if keys[-1] != "items" else "_items"
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)
        else:
            key = "_items" if key == "items" else key
            setattr(self, key, value)

    def __post_init__(self) -> None:
        if self.__class__ == Sample:
            self._extra: BaseModel = create_model(
                "Sample",
                __doc__=self.__class__.__doc__,
                __config__=self.model_config,
                **{
                    k.replace(".", "__nest__").replace("*", "all"): Annotated[
                        list[type(v[0])] if isinstance(v, list) and len(v) > 0 else type(v),
                        Field(default_factory=lambda: v),
                    ]
                    for k, v in self.dump().items()
                    if not k.startswith("_")
                },
            )()
            self._extra.__getitem__ = self.__class__.__getitem__
            self._extra.__setitem__ = self.__class__.__setitem__
            for k, v in self.dump().items():
                if not k.startswith("_"):
                    self.setdefault(k, v, nest=True)

    def __hash__(self) -> int:
        """Return a hash of the Sample instance."""

        def hash_helper(obj):
            if isinstance(obj, (list, tuple)):
                return hash(tuple(hash_helper(item) for item in obj))
            elif isinstance(obj, dict):
                return hash(tuple((k, hash_helper(v)) for k, v in sorted(obj.items())))
            elif isinstance(obj, Sample):
                return hash(tuple(hash_helper(v) for v in obj.dump().values()))
            else:
                return hash(obj)

        return hash_helper(self.dump())

    def _str(self, obj, prefix="", ignore=None):
        ignore = ignore or set("_items")
        if isinstance(obj, Path):
            obj = str(obj)
        if not hasattr(obj, "_str"):
            return obj
        prefix += " "
        sep = ",\n" + prefix
        out = f"{self.__class__.__name__}(\n{prefix}{sep.join([f'{k}={round(v, 3) if isinstance(v, float) else self._str(v,prefix)}' for k, v in obj if v is not None and k not in ignore])}"
        if hasattr(self, "_items"):
            out = out.replace(")", f"{sep}items={self._items}")
        return out.replace(")", "\n)")

    def __str__(self) -> str:
        """Return a string representation of the Sample instance."""
        unnested = {"_items"}
        for k, _ in self:
            if "." in k:
                unnested.add(k.split(".")[0])
            elif "/" in k:
                unnested.add(k.split("/")[0])
        return self._str(self, prefix="", ignore=set(unnested))

    def __repr__(self) -> str:
        """Return a string representation of the Sample instance."""
        return str(self)

    def __len__(self) -> int:
        """Return the number of attributes in the Sample instance."""
        return len(list(self.items()))

    def __contains__(self, key: str) -> bool:
        """Check if the Sample instance contains the specified attribute."""
        return key in list(self.keys())
    def get(self, key: str, default: Any = None) -> Any:
        """Return the value of the attribute with the specified key or the default value if it does not exist."""
        try:
            return self[key]
        except KeyError:
            return default

    def _dump(
        self, exclude: set[str] | str | None = "None", as_field: str | None = None, recurse=True
    ) -> Dict[str, Any] | Any:
        out = {}
        exclude = set() if exclude is None else exclude if isinstance(exclude, set) else {exclude}
        for k, v in self:
            if as_field is not None and k == as_field:
                return v
            if exclude and "None" in exclude and v is None:
                continue
            if exclude and k in exclude:
                continue
            if recurse and isinstance(v, Sample):
                out[k] = v.dump(exclude=exclude, as_field=as_field, recurse=recurse) if recurse else v
            elif recurse and isinstance(v, list | tuple | Dataset) and len(v) > 0 and isinstance(v[0], Sample):
                out[k] = [item.dump(exclude, as_field, recurse) for item in v]
            else:
                out[k] = v
        return {k: v for k, v in out.items() if v is not None or "None" not in exclude}

    def dump(
        self, exclude: set[str] | str | None = "None", as_field: str | None = None, recurse=True
    ) -> Dict[str, Any] | Any:
        """Dump the Sample instance to a dictionary or value at a specific field if present.

        Args:
            exclude (set[str], optional): Attributes to exclude. Defaults to "None". Indicating to exclude None values.
            as_field (str, optional): The attribute to return as a field. Defaults to None.
            recurse (bool, optional): Whether to convert nested Sample instances to dictionaries. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary representation of the Sample object.
        """
        return self._dump(exclude=exclude, as_field=as_field, recurse=recurse)

    def values(self) -> Generator:
        ignore = set()
        for k, _ in self:
            if "." in k:
                ignore.add(k.split(".")[0])
            elif "/" in k:
                ignore.add(k.split("/")[0])
        for k, v in self:
            if k not in ignore:
                yield v

    def keys(self) -> Generator:
        ignore = set()
        for k, _ in self:
            if "." in k:
                ignore.add(k.split(".")[0])
            elif "/" in k:
                ignore.add(k.split("/")[0])
        for k, _ in self:
            if k not in ignore:
                yield k

    def items(self) -> Generator:
        ignore = set()
        for k, _ in self:
            if "." in k:
                ignore.add(k.split(".")[0])
            elif "/" in k:
                ignore.add(k.split("/")[0])
        for k, v in self:
            if k not in ignore:
                yield k, v

    def dict(self, exclude: set[str] | None | str = "None", recurse=True) -> Dict[str, Any]:  # noqa
        """Return a dictionary representation of the Sample instance.

        Args:
            exclude_none (bool, optional): Whether to exclude None values. Defaults to True.
            exclude (set[str], optional): Set of attribute names to exclude. Defaults to None.
            recurse (bool, optional): Whether to convert nested Sample instances to dictionaries. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary representation of the Sample object.
        """
        exclude = exclude or set()
        if not recurse:
            return {
                k: v
                for k, v in self
                if k not in exclude and not k.startswith("_") and (v is not None or "None" not in exclude)
            }
        return self.dump(exclude=exclude, recurse=True)

    @classmethod
    def unflatten(cls, one_d_array_or_dict, schema=None) -> "Sample":
        """Unflatten a one-dimensional array or dictionary into a Sample instance.

        Args:
            one_d_array_or_dict: A one-dimensional array, dictionary, or tensor to unflatten.
            schema: A dictionary representing the JSON schema. Defaults to using the class's schema.

        Returns:
            Sample: The unflattened Sample instance.

        Examples:
            >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
            >>> flat_dict = sample.flatten(output_type="dict")
            >>> print(flat_dict)
            {'x': 1, 'y': 2, 'z': {'a': 3, 'b': 4}, 'extra_field': 5}
            >>> Sample.unflatten(flat_dict, sample.schema())
            Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
        """
        schema = schema or cls().schema()

        if isinstance(one_d_array_or_dict, (list, np.ndarray, torch.Tensor)):
            flat_data = (
                one_d_array_or_dict.tolist()
                if isinstance(one_d_array_or_dict, (np.ndarray, torch.Tensor))
                else one_d_array_or_dict
            )
            one_d_array_or_dict = {}
            index = 0
            for prop, prop_schema in schema["properties"].items():
                if prop_schema["type"] == "object":
                    one_d_array_or_dict[prop] = {}
                    for sub_prop in prop_schema["properties"]:
                        one_d_array_or_dict[prop][sub_prop] = flat_data[index]
                        index += 1
                else:
                    one_d_array_or_dict[prop] = flat_data[index]
                    index += 1

        def unflatten_recursive(schema_part, data):
            if schema_part["type"] == "object":
                result = {}
                for prop, prop_schema in schema_part["properties"].items():
                    if not prop.startswith("_") and prop in data:  # Skip properties starting with underscore
                        result[prop] = unflatten_recursive(prop_schema, data[prop])
                if schema_part.get("title", "").lower() == cls.__name__.lower():
                    result = cls(**result)
                elif schema_part.get("title", "").lower() == "sample":
                    result = Sample(**result)
                return result
            if schema_part["type"] == "array":
                return [unflatten_recursive(schema_part["items"], item) for item in data]
            return data

        unflattened_dict = unflatten_recursive(schema, one_d_array_or_dict)
        return cls(**unflattened_dict) if not isinstance(unflattened_dict, cls) else unflattened_dict

    # def rearrange(self, pattern: str, **kwargs) -> Any:
    #     """Pack, unpack, flatten, select indices according to an einops-style pattern.

    #     rearrange('(b s) [action state] -> b s [actions state]', s=32) will select the action and state keys
    #      and pack them into batches of size 32.
    #     """
    #     # Parse the input and output patterns
    #     input_pattern, output_pattern = pattern.split('->')
    #     input_pattern = input_pattern.strip()
    #     output_pattern = output_pattern.strip()

    #     # Extract keys from square brackets
    #     input_keys = re.findall(r'\[([^\]]+)\]', input_pattern)
    #     output_keys = re.findall(r'\[([^\]]+)\]', output_pattern)

    #     # Flatten the sample and select only the required keys
    #     flattened = self.flatten(output_type="dict")
    #     selected_data = {key: flattened[key] for key in input_keys[0].split() if key in flattened}

    #     # Convert selected data to numpy arrays
    #     np_data = {k: np.array(v) for k, v in selected_data.items()}

    #     # Apply einops rearrange
    #     rearranged_data = einops_rearrange(np_data, pattern, **kwargs)

    #     if isinstance(rearranged_data, dict):
    #         # If the output is a dictionary, directly assign it to the output Sample
    #         for k, v in rearranged_data.items():
    #             setattr(output_sample, k, v.tolist() if isinstance(v, np.ndarray) else v)
    #     else:
    #         # If the output is not a dictionary, we need to reconstruct it based on the output pattern
    #         output_keys = output_keys[0].split() if output_keys else input_keys[0].split()
    #         for i, key in enumerate(output_keys):
    #             setattr(output_sample, key, rearranged_data[..., i].tolist())

    #     return output_sample
    @staticmethod
    def _flatten_recursive(obj, ignore: None | set = None, non_numerical="allow", sep="."):
        def _flatten(obj, prefix=""):
            out = []
            keys = []
            if isinstance(obj, Sample | dict):
                for k, v in obj.items():
                    if k in ignore:
                        continue
                    new_key = f"{prefix}{k}" if prefix else k
                    subout, subkeys = _flatten(v, f"{new_key}{sep}")
                    out.extend(subout)
                    keys.extend(subkeys)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    subout, subkeys = _flatten(v, f"{prefix}{i}{sep}")
                    out.extend(subout)
                    keys.extend(subkeys)
            else:
                if non_numerical == "forbid" and not isinstance(obj, int | float | np.number):
                    msg = f"Non-numerical value encountered: {obj}"
                    raise ValueError(msg)
                if non_numerical == "ignore" and not isinstance(obj, int | float | np.number):
                    return []
                out.append(obj)
                keys.append(prefix.rstrip(sep))
            return out, keys

        return _flatten(obj)

    @staticmethod
    def flatten_recursive(obj, ignore: None | set = None, non_numerical="allow", sep="."):
        return Sample._flatten_recursive(obj, ignore=ignore, non_numerical=non_numerical, sep=sep)

    def flatten(
        self,
        output_type: OneDimensional = "list",
        non_numerical: Literal["ignore", "forbid", "allow"] = "allow",
        ignore: tuple[str] | None = None,
        sep: str = ".",
        to: str | List[str] | None = None,
    ) -> Dict[str, Any] | np.ndarray | torch.Tensor | List | Any:
        """Flatten the Sample instance into a strictly one-dimensional or two-dimensional structure."""
        from embdata.describe import full_paths
        to = [to] if isinstance(to, str) else to
        full_to = full_paths(self, to).values() if to is not None else None
        ignore = ignore or ()

        def replace_ints_with_wildcard(s, sep="."):
            # Pattern to match integers that are either at the start/end of the string or surrounded by the separator
            pattern = rf"(?<=^{sep})\d+|(?<={sep})\d+(?={sep})|\d+(?={sep}|$)"
            return re.sub(pattern, "*", s).rstrip(f"{sep}*").lstrip(f"{sep}*")

        flattened, keys = self.flatten_recursive(self, ignore=ignore, non_numerical=non_numerical, sep=sep)
        keys = [replace_ints_with_wildcard(k, sep=sep) for k in keys]
        if to is not None:
            result = []
            current_group = {k: [] for k in to}
            num_tos = {k: 0 for k in to}
            for k, v in zip(keys, flattened, strict=False):
                for i, ft in enumerate(full_to):
                    if ft in k:
                        current_group[to[i]].append(v)
                        num_tos[to[i]] += 1

                if all(num_to == num_tos[to[0]] for num_to in num_tos.values()):
                    if all(len(current_group[to[i]]) > 0 for i in range(len(to))):
                        ignore = set({k.split(sep)[0] for k in keys if sep in k})
                        ignore = {k for k in ignore if k not in to}
                        current_group = {k: v[0] if len(v) == 1 else v for k, v in current_group.items() if k not in ignore}
                        if output_type not in ["dict", "sample"]:
                            try:
                                current_group = reduce(operator.iadd, current_group.values(), [])
                            except TypeError:
                                current_group = list(current_group.values())
                        result.append(Sample(**current_group) if output_type == "sample" else current_group)
                    current_group = Sample({k: [] for k in to})
                    num_tos = Sample({k: 0 for k in to})
            flattened = result

        if output_type == "np":
            return np.array(flattened, dtype=object)
        if output_type == "pt":
            return torch.tensor(flattened, dtype=torch.float32)
        return flattened

    def squeeze(self):
        for k, v in self:
            if isinstance(v, Sample):
                setattr(self, k, v.squeeze())
            if isinstance(v, list) and len(v) == 1:
                setattr(self, k, v[0])
            if isinstance(v, dict) and len(v) == 1:
                setattr(self, k, next(iter(v.values())))
        return self

    def setdefault(self, key: str, default: Any, nest=True) -> Any:
        """Set the default value for the attribute with the specified key."""
        if not nest:
            if key in self:
                return self[key]
            self[key] = default
            return default
        keys = key.split(".")
        obj = self
        for k in keys[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(k, {})
            elif isinstance(obj, list):
                if k == "*":
                    return obj
                try:
                    index = int(k)
                    if index >= len(obj):
                        obj.extend([None] * (index - len(obj) + 1))
                    if obj[index] is None:
                        obj[index] = {}
                    obj = obj[index]
                except ValueError:
                    raise AttributeError(f"Invalid list index: {k}")
            elif not isinstance(obj, Sample) and not hasattr(obj, k):
                new_obj = Sample()
                obj[k] = new_obj
                obj = new_obj
            elif hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                obj[k] = Sample()
                obj = getattr(obj, k)
        if isinstance(obj, dict):
            if keys[-1] == "*":
                return obj.setdefault("*", default if isinstance(default, list) else [default])
            return obj.setdefault(keys[-1], default)
        if isinstance(obj, list):
            if keys[-1] == "*":
                return obj
            try:
                index = int(keys[-1])
                if index >= len(obj):
                    obj.extend([None] * (index - len(obj) + 1))
                if obj[index] is None:
                    obj[index] = default
                return obj[index]
            except ValueError:
                raise AttributeError(f"Invalid list index: {keys[-1]}")
        if not hasattr(obj, keys[-1]):
            if keys[-1] == "*":
                setattr(obj, keys[-1], default if isinstance(default, list) else [default])
            else:
                setattr(obj, keys[-1], default)
        return getattr(obj, keys[-1])

    def schema(self, include: Literal["all", "descriptions", "info", "simple"] = "info") -> Dict:
        """Returns a simplified json schema.

        Args:
            include ("all", "descriptions", "info", "simple", optional): The level of detail to include in the schema.
                Defaults to "info".
                for "all", send the full pydantic schema.
                for "descriptions", send the simplified schema with descriptions.
                for "info", send the simplified schema with model info only.
                for "simple", send the simplified schema which is the full schema with:
                    - references resolved
                    - descriptions removed
                    - additionalProperties removed
                    - items removed
                    - allOf removed
                    - anyOf resolved

        Returns:
            dict: A simplified JSON schema.
        """
        schema = self._extra.model_json_schema() if hasattr(self, "_extra") else self.model_json_schema()
        if include == "all":
            return schema

        def resolve_refs(schema: dict) -> dict:
            def _resolve(obj, defs=None):
                if isinstance(obj, dict):
                    if obj and "$ref" in obj and defs is not None:
                        ref_key = obj["$ref"].split("/")[-1]
                        resolved = defs[ref_key]
                        resolved.update({k: _resolve(v) for k, v in obj.items() if k != "$ref" and v is not None})
                        return _resolve(resolved, defs)
                    if "items" in obj:
                        obj["items"] = _resolve(obj["items"], defs)
                    if "properties" in obj:
                        obj["properties"] = {
                            k: _resolve(v, defs) for k, v in obj["properties"].items() if v is not None
                        }
                    if "allOf" in obj:
                        all_of_resolved = {}
                        for item in obj["allOf"]:
                            resolved_item = _resolve(item, defs)
                            all_of_resolved.update(resolved_item)
                        obj.pop("allOf")
                        obj.update(all_of_resolved)
                    if "anyOf" in obj:
                        first_non_null = None
                        for item in obj["anyOf"]:
                            if "type" in item and item["type"] == "null":
                                break
                            first_non_null = item
                        if first_non_null is not None:
                            obj.pop("anyOf")
                            obj.update(_resolve(first_non_null, defs))
                            return obj
                    return {k: _resolve(v, defs) for k, v in obj.items() if v is not None}
                return obj

            schema_copy = copy.deepcopy(schema)
            defs = schema_copy.get("$defs", {})
            schema = _resolve(schema_copy, defs)
            schema.pop("$defs", None)
            return schema

        schema = resolve_refs(schema)

        def simplify(schema, obj):
            title = schema.get("title", "")
            if isinstance(obj, dict):
                obj = Sample(**obj)
                _include = "simple"
            elif hasattr(obj, "_extra"):
                title = obj.__class__.__name__
                _include = include
            if "description" in schema and include != "descriptions":
                del schema["description"]
            if "additionalProperties" in schema:
                del schema["additionalProperties"]
            if "items" in schema:
                schema["items"] = simplify(schema["items"], obj[0])
                schema["maxItems"] = len(obj)
                if schema["items"].get("title"):  # Use the object title instead.
                    del schema["items"]["title"]
            if "type" in schema and "ndarray" in schema["type"]:
                # Handle numpy arrays
                schema["type"] = "array"
                schema["items"] = {"type": "number"}
                del schema["properties"]
                del schema["required"]
            if "type" in schema and schema["type"] == "object":
                if "properties" not in schema:
                    schema = obj.schema(include=_include)
                for k, value in schema["properties"].items():
                    if include == "simple" and k.startswith("_"):
                        continue
                    if hasattr(obj, "model_dump"):
                        obj = obj.model_dump()
                    if k in obj:
                        schema["properties"][k] = simplify(value, obj[k])
                if not schema["properties"]:
                    schema = obj.schema(include=_include)
            if "allOf" in schema or "anyOf" in schema:
                msg = f"Schema contains allOf or anyOf which is unsupported: {schema}"
                raise ValueError(msg)
            if title:
                schema["title"] = title
            return schema

        return simplify(schema, self)

    def infer_features_dict(self) -> Dict[str, Any]:
        """Infers features from the data recusively."""
        feat_dict = {}
        for k, v in self:
            if v is None:
                logging.info("Skipping %s as it is None", k)
                continue
            if isinstance(v, Sample):
                feat_dict[k] = v.infer_features_dict()
            elif isinstance(v, list | tuple | np.ndarray):
                if len(v) > 0 and isinstance(v[0], Sample):
                    feat_dict[k] = [v[0].infer_features_dict()]
                elif len(v) > 0:
                    feat_dict[k] = [to_features_dict(v[0])]
            else:
                feat_dict[k] = to_features_dict(v)
        return feat_dict

    def to(self, container: Any, **kwargs) -> Any:
        """Convert the Sample instance to a different container type.

        Args:
            container (Any): The container type, class, or callable to convert to.

            If a string, convert to one of:
            -'dict', 'list', 'np', 'pt' (pytorch), 'space' (gym.space),
            -'schema', 'json', 'hf' (datasets.Dataset)

            If a class, of type Sample, convert to that class.
            If a callable, convert to the output of the callable.

        Returns:
            Any: The converted container.

        Examples:
        >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
        >>> sample.to("features")
        {'x': Value(dtype='float32', id=None), 'y': Value(dtype='float32', id=None), 'z': {'a': Value(dtype='float32', id=None), 'b': Value(dtype='float32', id=None)}, 'extra_field': Value(dtype='float32', id=None)}
        """
        if isinstance(container, type) and issubclass(container, Sample):
            return container.unflatten(self.flatten())

        if container == "dict":
            return self.dump()
        if container == "list":
            return self.tolist()
        if container in ["np", "numpy"]:
            return self.numpy()
        if container in ["pt", "torch", "pytorch"]:
            return self.torch()
        if container == "space":
            return self.space()
        if container == "schema":
            return self.schema()
        if container == "json":
            return self.model_dump_json()
        if container in ["hf", "huggingface", "dataset", "datasets"]:
            return self.dataset()
        if container == "features":
            return Features(self.infer_features_dict())
        if container == "sample":
            return self.flatten(output_type="sample", **kwargs)
        try:
            logging.warning(f"No matching container found for {container}. Attempting nested conversion.")
            for k, v in self:
                if isinstance(v, Sample):
                    self[k] = v.to(container, **kwargs)
        except Exception as e:  # noqa
            try:
                return container(self)
            except Exception as e1:  # noqa
                try:
                    return container(self.dump(), **kwargs)
                except Exception as e2:  # noqa
                    msg = f"Unsupported container type: {container}"
                    raise ValueError(msg) from e
        return self

    @classmethod
    def default_value(cls) -> "Sample":
        """Get the default value for the Sample instance.

        Returns:
            Sample: The default value for the Sample instance.
        """
        return cls()

    @classmethod
    def space_for(
        cls,
        value: Any,
        max_text_length: int = 1000,
        info: Dict[str, Any] | None = None,
    ) -> spaces.Space:
        """Default Gym space generation for a given value.

        Only used for subclasses that do not override the space method.
        """
        if isinstance(value, Enum) or get_origin(value) == Literal:
            return spaces.Discrete(len(value.__args__))
        if isinstance(value, bool):
            return spaces.Discrete(2)
        if isinstance(value, dict | Sample):
            if isinstance(value, Sample):
                value = value.dump()
            return spaces.Dict(
                {k: Sample.space_for(v, max_text_length, info) for k, v in value.items()},
            )
        if isinstance(value, str):
            return spaces.Text(max_length=max_text_length)

        if isinstance(value, int | float | list | tuple | np.ndarray | np.number):
            bounds = None
            dtype = None
            shape = None
            if info is not None:
                shape = info.get("shape")
                bounds = info.get("bounds")
                dtype = info.get("dtype")
            logging.debug(
                "Generating space for value: %s, shape: %s, bounds: %s, dtype: %s",
                value,
                shape,
                bounds[0] if bounds else None,
                bounds[1] if bounds else None,
                dtype,
            )
            try:
                if not hasattr(value, "shape") and not hasattr(value, "__len__"):
                    shape = ()
                    dtype = type(value)
                    low, high = bounds or (-np.inf, np.inf)
                    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
                value = np.asarray(value, dtype=float)
                shape = shape or value.shape
                dtype = dtype or value.dtype
                if bounds is None:
                    low = np.full(shape, -np.inf, dtype=dtype)
                    high = np.full(shape, np.inf, dtype=dtype)
                else:
                    low, high = bounds
                return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
            except Exception as e:
                logging.info(f"Could not convert value {value} to numpy array: {e}")
                if len(value) > 0 and isinstance(value[0], dict | Sample):
                    return spaces.Tuple(
                        [spaces.Dict(cls.space_for(v, max_text_length, info)) for v in value],
                    )
                return spaces.Tuple(
                    [spaces.Dict(cls.space_for(value[0], max_text_length, info)) for value in value[:1]],
                )
        msg = f"Unsupported object {value} of type: {type(value)} for space generation"
        raise ValueError(msg)

    @classmethod
    def from_space(cls, space: spaces.Space) -> "Sample":
        """Generate a Sample instance from a Gym space."""
        sampled = space.sample()
        if isinstance(sampled, dict):
            return cls(**sampled)
        if isinstance(sampled, np.ndarray | torch.Tensor | list | tuple):
            sampled = np.asarray(sampled)
            if len(sampled.shape) > 0 and isinstance(sampled[0], dict | Sample):
                return cls.unpack_from(sampled)
        return cls(sampled)

    @classmethod
    def unpack_from(
        cls,
        samples: List[Union["Sample", Dict]],
        padding: Literal["truncate", "longest"] = "longest",
        pad_value: Any = None,
    ) -> "Sample":
        """Pack a list of samples or dicts into a single sample with lists of samples for attributes.

        [Sample(a=1, b=4),  ->  Sample(a=[1, 2, 3],
         Sample(a=2, b=5),             b=[4, 5, 6])
         Sample(a=3, b=6)]

        This is equivalent to a zip operation on a list of dicts or transpose on the first two dimensions.

        Args:
            samples: List of Sample objects or dictionaries to pack.
            padding: Strategy for handling samples of different lengths.
                     "truncate" will truncate to the shortest sample,
                     "longest" will pad shorter samples to match the longest.
            pad_value: Value to use for padding when padding="longest". If None, uses cls.default_value().

        Returns:
            A new Sample object with packed attributes.

        Raises:
            ValueError: If the input list is empty.
            TypeError: If the input list contains items that are neither Sample nor dict.
        """
        if not samples:
            msg = "Cannot pack an empty list of samples"
            raise ValueError(msg)

        if isinstance(samples[0], dict):
            attributes = list(samples[0].keys())
        elif isinstance(samples[0], Sample):
            attributes = list(samples[0].dump().keys())
        elif samples[0] is None:
            msg = "Cannot pack a list containing None"
            raise ValueError(msg)
        else:
            msg = f"Cannot determine attributes from the first sample: {samples[0]}"
            raise TypeError(msg)

        pad_value = pad_value if pad_value is not None else cls.default_value()
        if padding == "truncate":
            attributes = [attr for attr in attributes if all(attr in sample for sample in samples)]

        return cls(
            **{
                attr: [
                    sample.get(attr, pad_value) if isinstance(sample, dict) else getattr(sample, attr, pad_value)
                    for sample in samples
                ]
                for attr in attributes
            }
        )

    def pack(
        self,
        to: Literal["dicts", "samples"] = "samples",
        padding: Literal["truncate", "longest"] = "truncate",
        pad_value: Any = None,
    ) -> list["Sample"] | List[Dict[str, Any]]:
        """Unpack the packed Sample object into a list of Sample objects or dictionaries.

        Sample(a=[1, 3, 5],   ->    [Sample(a=1, b=2),
               b=[2, 4, 6]           Sample(a=3, b=4),
                                     Sample(a=5, b=6)]
        )

        This is the inverse of the unpack method (analagous to a permute operation on the first two dimensions).

        Args:
            to: Specifies the output type, either "samples" or "dicts".
            padding: Strategy for handling attributes of different lengths.
                     "truncate" will truncate to the shortest attribute,
                     "longest" will pad shorter attributes to match the longest.
            pad_value: Value to use for padding when padding="longest". If None, uses self.default_value().

        Returns:
            A list of Sample objects or dictionaries, depending on the 'to' parameter.

        Example:
        >>> Sample(a=[1, 3, 5], b=[2, 4, 6]).pack()
        [Sample(a=1, b=2), Sample(a=3, b=4), Sample(a=5, b=6)]
        """
        if not any(isinstance(v, list) for k, v in self):
            return []

        pad_value = pad_value if pad_value is not None else self.default_value()
        attributes = list(self.keys())

        if padding == "truncate":
            min_length = min(len(v) for v in self.values() if isinstance(v, list))
            data = {k: v[:min_length] if isinstance(v, list) else v for k, v in self.items()}
        else:
            data = self
        max_length = max(len(v) if isinstance(v, list) else 1 for v in data.values())

        unzipped = zip_longest(
            *[data[attr] if isinstance(data[attr], list) else [data[attr]] * max_length for attr in attributes],
            fillvalue=pad_value,
        )
        mapper = (lambda items: self.__class__(**dict(items))) if to == "samples" else dict
        return [mapper(zip(attributes, values, strict=False)) for values in unzipped]

    def unpack(self, to: Literal["dicts", "samples", "lists"] = "samples") -> List[Union["Sample", Dict]]:
        return [[x.dump() if to == "dicts" else x for x in samples] for _, samples in self]

    @classmethod
    def pack_from(
        cls,
        *args,
        packed_field: str = "steps",
        padding: Literal["truncate", "longest"] = "truncate",
        pad_value: Any | None = None,
    ) -> "Sample":
        """Pack an iterable of Sample objects or dictionaries into a single Sample object with a single list attribute of Samples.

        [Sample(a=1)], Sample(a=2)], -> Sample(steps=[Sample(a=1,b=1),
        [Sample(b=1), Sample(b=2)]                    Sample(a=2,b=2)])

        This is equivalent to a zip operation on the list of samples (or transpose on Row, Col dimensions where
        Args:
            args: Iterable of Sample objects or dictionaries to pack.
            packed_field: The attribute name to pack the samples into.
            padding: Strategy for handling samples of different lengths.
                     "truncate" will truncate to the shortest sample,
                     "longest" will pad shorter samples to match the longest.
            pad_value: Value to use for padding when padding="longest". If None, uses cls.default_value().

        Returns:
            A new Sample object with a single list attribute containing the packed samples.


        """
        if not args:
            msg = "Cannot pack an empty list of samples"
            raise ValueError(msg)
        if len(args) == 1 and isinstance(args[0], list | tuple):
            args = args[0]
        if not all(isinstance(arg, Sample | dict) for arg in args):
            msg = "All arguments must be Sample objects or dictionaries"
            raise TypeError(msg)

        if padding == "longest":
            return cls(**{packed_field: list(zip_longest(*args, fillvalue=pad_value))})
        return cls(**{packed_field: list(zip(*args, strict=False))})

    def __unpack__(self):
        return self.unpack("dicts")

    @classmethod
    def default_space(cls) -> spaces.Dict:
        """Return the Gym space for the Sample class based on its class attributes."""
        return cls().space()

    @classmethod
    def default_sample(cls) -> Union["Sample", Dict[str, Any]]:
        """Generate a default Sample instance from its class attributes. Useful for padding.

        This is the "no-op" instance and should be overriden as needed.
        """
        return cls()

    def model_info(self) -> Dict[str, Any]:
        """Get the model information.

        This includes various metadata such as shape, bounds, and other information.
        """
        out = {}
        for key, value in dict(self).items():
            info = self.field_info(key) if not isinstance(value, Sample) else value.model_info()
            if info:
                out[key] = info
        return out

    def field_info(self, key: str) -> Dict[str, Any]:
        """Get the extra json values set from a FieldInfo for a given attribute key.

        This can include bounds, shape, and other information.
        """
        info = {}
        if self.model_extra and key in self.model_extra:
            info = FieldInfo(annotation=self.model_extra[key]).json_schema_extra or {}
        if key in self.model_fields:
            info = self.model_fields[key].json_schema_extra or {}
        return info.get("_info", {})

    def add_field_info(self, field_key, info_key, value) -> None:
        if self.model_extra and field_key in self.model_extra:
            info = FieldInfo(annotation=self.model_extra[field_key]).json_schema_extra or {}
            info.update({info_key: value})
            self.model_extra[field_key] = info
        elif field_key in self.model_fields:
            info = self.model_fields[field_key].json_schema_extra or {}
            info.update({info_key: value})

    def space(self) -> spaces.Dict:
        """Return the corresponding Gym space for the Sample instance based on its instance attributes.

        Omits None values.

        Override this method in subclasses to customize the space generation.
        """
        space_dict = {}
        for key, value in self.model_dump(exclude_none=True).items():
            logging.debug("Generating space for key: '%s', value: %s", key, value)
            info = self.field_info(key)
            space_dict[key] = self.space_for(value, info=info)
        return spaces.Dict(space_dict)

    def random_sample(self) -> "Sample":
        """Generate a random Sample instance based on its instance attributes. Omits None values.

        Override this method in subclasses to customize the sample generation.
        """
        return self.__class__.model_validate(self.space().sample())

    @cached_property
    def _numpy(self) -> np.ndarray:
        """Convert the Sample instance to a numpy array."""
        return self.flatten("np")

    @cached_property
    def _tolist(self) -> list:
        """Convert the Sample instance to a list."""
        return self.flatten("list")

    @cached_property
    def _torch(self) -> torch.Tensor:
        import_module("torch")
        """Convert the Sample instance to a PyTorch tensor."""
        return self.flatten("pt")

    @cached_property
    def _json(self) -> str:
        """Convert the Sample instance to a JSON string."""
        return self.model_dump_json()

    def numpy(self) -> np.ndarray:
        """Return the numpy array representation of the Sample instance."""
        return self._numpy
    
    def tolist(self) -> list:
        """Return the list representation of the Sample instance."""
        return self._tolist
    
    def torch(self) -> torch.Tensor:
        """Return the PyTorch tensor representation of the Sample instance."""
        return self._torch

    def json(self) -> str:
        """Return the JSON string representation of the Sample instance."""
        return self._json

    @cached_property
    def _features(self) -> Features:
        """Convert the Sample instance to a HuggingFace Features object."""
        return Features(self.infer_features_dict())

    def features(self) -> Features:
        """Return the HuggingFace Features object for the Sample instance."""
        return self._features

    def dataset(self) -> Dataset:
        """Convert the Sample instance to a HuggingFace Dataset object."""
        data = self
        # HuggingFace datasets require pillow images to be converted to bytes.
        data = self.wrapped if hasattr(self, "wrapped") and self.wrappeed is not None else data.dump(as_field="pil")
        if isinstance(data, list):
            return Dataset.from_list(data, features=self.features())
        if isinstance(data, dict):
            return Dataset.from_dict(data, features=self.features())
        if isinstance(data, Generator):
            return Dataset.from_generator(data, features=self.features())

        msg = f"Unsupported data type {type(data)} for conversion to Dataset."
        raise ValueError(msg)

    def describe(self) -> str:
        """Return a string description of the Sample instance."""
        return describe(self, compact=True, name=self.__class__.__name__)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

    logging.basicConfig(level=logging.DEBUG, force=True)
    s = Sample(x=1, y=2, z={"a": 3, "b": 4, "c": np.array([1, 2, 3])}, extra_field=5)
