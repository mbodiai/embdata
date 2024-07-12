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
    model_field_info: Get the FieldInfo for a given attribute key.
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
import json
import logging
import re
from collections import OrderedDict
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Literal, Sequence, Union, get_origin

import numpy as np
import torch
from datasets import Dataset, Features
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import from_json

from embdata.describe import describe_keys
from embdata.features import to_features_dict

OneDimensional = Annotated[Literal["dict", "np", "pt", "list"], "Numpy, PyTorch, list, or dict"]


class Sample(BaseModel):
    """A base model class for serializing, recording, and manipulating arbitray data."""
    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=False,
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    def __init__(self, item=None, **data):
        """Initialize a Sample instance.

        This method allows for flexible initialization of a Sample object, either from
        an existing item or from keyword arguments. It handles various input types and
        structures, including other Sample instances, dictionaries, lists, and more.

        Args:
            item (Any, optional): An existing item to initialize the Sample from. This can
                be another Sample instance, a dictionary, a list, or other supported types.
            **data: Arbitrary keyword arguments to initialize the Sample's attributes.

        Examples:
            >>> Sample(x=1, y=2, z={"a": 3, "b": 4})
            Sample(x=1, y=2, z={'a': 3, 'b': 4})

            >>> existing_sample = Sample(a=1, b=2)
            >>> Sample(existing_sample)
            Sample(a=1, b=2)

            >>> Sample([1, 2, 3])
            Sample(items=[1, 2, 3])

            >>> Sample({"key": "value"})
            Sample(key='value')

            # Complex nested structure
            >>> nested_sample = Sample(
            ...     image=Sample(
            ...         data=np.random.rand(32, 32, 3),
            ...         metadata={"format": "RGB", "size": (32, 32)}
            ...     ),
            ...     text=Sample(
            ...         content="Hello, world!",
            ...         tokens=["Hello", ",", "world", "!"],
            ...         embeddings=np.random.rand(4, 128)
            ...     )
            ... )
            >>> nested_sample
            Sample(image=Sample(data=array(...), metadata={'format': 'RGB', 'size': (32, 32)}), text=Sample(content='Hello, world!', tokens=['Hello', ',', 'world', '!'], embeddings=array(...)))
        """
        if item is not None and self.__class__.__name__ != "Image":  # Hacky but works for now.
            if isinstance(item, Sample):
                if not data:
                    data = dict(item)
            elif isinstance(item, dict):
                if not data:
                    data = {k: v for k, v in item.items() if not k.startswith("_")}
            elif self.__class__ == Sample:
                # Only the Sample class can wrap an arbitrary type.
                if isinstance(item, list | tuple | np.ndarray | torch.Tensor | Dataset):
                    # There is no schema to unflatten from, just have it as an attribute.
                    data["items"] = item
                else:
                    data["item"] = item
            elif isinstance(item, list | tuple | np.ndarray | torch.Tensor | Dataset):
                # Derived classes have a schema to unflatten from.
                d = self.__class__.unflatten(item).model_dump()
                data.update(d)
            elif isinstance(item, spaces.Space):
                data.update(self.from_space(item).model_dump())

        super().__init__(**data)
        self.__post_init__()

    def __getitem__(self, key: str | int) -> Any:
        """Return the value of the attribute with the specified key."""
        if self.__class__ == Sample:
            if isinstance(key, int):
                if hasattr(self, "items"):
                    return self.items[key]
                raise TypeError("Sample does not wrap a list or Dataset")
            return getattr(self._extra, key)
        if isinstance(key, int):
            raise TypeError(f"Sample does not support access index {key} in a {self.__class__} instance")
        return getattr(self, key)

    def __post_init__(self):  # noqa
        if self.__class__ == Sample:
            self._extra: BaseModel = create_model(
                "Sample",
                __doc__=self.__class__.__doc__,
                __config__=self.model_config,
                **{
                    k: Annotated[
                        list[type(v[0])] if isinstance(v, list) and len(v) > 0 else type(v),
                        Field(default_factory=lambda: v),
                    ]
                    for k, v in self.dump().items()
                    if not k.startswith("_")
                },  # noqa
            )()

    def __hash__(self) -> int:
        """Return a hash of the Sample instance."""
        return hash(tuple(self.dump().values()))

    def __str__(self) -> str:
        """Return a string representation of the Sample instance."""
        sep = ",\n"
        return f"{self.__class__.__name__}({sep.join([f'{k}={round(v, 3) if isinstance(v, float) else v}' for k, v in self.dump().items() if v is not None])})"

    def dump(self, exclude_none=True, exclude: set[str] = None, as_field: str | None = None) -> Dict[str, Any] | Any:
        """Dump the Sample instance to a dictionary or value at a specific field if present.

        If the 'as_field' parameter is provided, the method returns the value at the specified field for
        if it exists for the top-level object and nested objects.

        Args:
            exclude_none (bool, optional): Whether to exclude None values. Defaults to True.
            exclude (set[str], optional): Set of attribute names to exclude. Defaults to None.
            as_field (str, optional): The attribute to return as a field. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary representation of the Sample object.
        """
        out = {}
        for k, v in dict(self).items():
            if as_field is not None and k == as_field:
                return v
            if exclude_none and v is None:
                continue
            if exclude and k in exclude:
                continue
            if isinstance(v, Sample):
                out[k] = v.dump(exclude_none=exclude_none, exclude=exclude, as_field=as_field)
            elif isinstance(v, list | tuple | Dataset):
                if len(v) > 0 and isinstance(v[0], Sample):
                    out[k] = [item.dump(exclude_none=exclude_none, exclude=exclude, as_field=as_field) for item in v]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def dict(self, exclude_none=True, exclude: set[str] = None, recurse=True) -> Dict[str, Any]:
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
            return {k: v for k, v in dict(self).items() if k not in exclude and (v is not None or not exclude_none)}
        return self.dump(exclude_none=exclude_none, exclude=exclude)

    @classmethod
    def unflatten(cls, one_d_array_or_dict, schema=None) -> "Sample":
        """Unflatten a one-dimensional array or dictionary into a Sample instance.

        If a dictionary is provided, its keys are ignored.

        Args:
            one_d_array_or_dict: A one-dimensional array or dictionary to unflatten.
            schema: A dictionary representing the JSON schema. Defaults to using the class's schema.

        Returns:
            Sample: The unflattened Sample instance.

        Examples:
            >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
            >>> flat_list = sample.flatten()
            >>> print(flat_list)
            [1, 2, 3, 4, 5]
            >>> Sample.unflatten(flat_list, sample.schema())
            Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
        """
        schema = schema or cls().schema()
        if isinstance(one_d_array_or_dict, dict):
            flat_data = list(one_d_array_or_dict.values())
        else:
            flat_data = list(one_d_array_or_dict)

        def unflatten_recursive(schema_part, index=0):
            if schema_part["type"] == "object":
                result = {}
                for prop, prop_schema in schema_part["properties"].items():
                    if not prop.startswith("_"):  # Skip properties starting with underscore
                        value, index = unflatten_recursive(prop_schema, index)
                        result[prop] = value
                return result, index
            if schema_part["type"] == "array":
                items = []
                for _ in range(schema_part.get("maxItems", len(flat_data) - index)):
                    value, index = unflatten_recursive(schema_part["items"], index)
                    items.append(value)
                return items, index
            return flat_data[index], index + 1

        unflattened_dict, _ = unflatten_recursive(schema)
        return cls(**unflattened_dict)

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

    def flatten(
        self,
        output_type: OneDimensional = "list",
        non_numerical: Literal["ignore", "forbid", "allow"] = "allow",
        ignore: set[str] | None = None,
        sep: str = ".",
        to: str | set[str] | None = None,
    ) -> Dict[str, Any] | np.ndarray | torch.Tensor | List:
        """Flatten the Sample instance into a one-dimensional structure.

        This method traverses the nested structure of the Sample instance and its attributes,
        creating a flattened representation based on the specified parameters.

        Note: If 'pt' or 'np' is specified as the 'output_type', non-numerical values are excluded.
        
        When 'to' is provided, the method always returns a list or Dataset, regardless of 'output_type'.
        If multiple `to` keys are passed, the output list is concatenated in the order of occurrence.
        The order of elements in the flattened output is guaranteed to be consistent across calls.

        Parameters:
        -----------
        output_type : str, optional (default="list")
            Specifies the type of the output. Options are:
            - "list": Returns a flat list of values.
            - "dict": Returns a dictionary with flattened keys and their corresponding values.
            - "np": Returns a numpy array.
            - "pt": Returns a PyTorch tensor.

        non_numerical : str, optional (default="ignore")
            Determines how non-numerical values are handled. Options are:
            - "ignore": Non-numerical values are excluded from the output.
            - "forbid": Raises a ValueError if non-nu    @classmethod

        Examples:
        ---------
            >>> sample = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5))
            >>> sample.flatten()
            [1, 2, 3, 4, 5]

            >>> sample.flatten(output_type="dict")
            {'a': 1, 'b.c': 2, 'b.d.0': 3, 'b.d.1': 4, 'e.f': 5}

            >>> sample.flatten(ignore={"b"})
            [1, 5]

        Notes:
        ------
            - When 'to' is provided, the method always returns a list or Dataset, regardless of 'output_type'.
            - If multiple `to` keys are passed, the output list is concatenated in the order of occurrence.
            - The order of elements in the flattened output is guaranteed to be consistent across calls.
        """
        if to is not None:
            if isinstance(to, str):
                to = {to}
            elif isinstance(to, list):
                to = set(to)
            elif not isinstance(to, set):
                raise ValueError(f"Unsupported type for 'to': {type(to)}")
        keys = set()
        keys_map = describe_keys(self)
        for k in to or []:
            if not isinstance(k, str):
                raise ValueError(f"Invalid key in 'to': {k} (expected str)")
            if k in keys_map:
                keys.add(keys_map[k])
            else:
                keys.add(k)
        to = keys

        if output_type in ["np", "pt"]:
            non_numerical = "ignore"

        accumulator = {} if output_type == "dict" or to else []

        def ignore_key(k, current_path):
            # Ignore keys starting with underscore by default.
            return (k.startswith("_") and ignore is None) or (ignore is not None and current_path in ignore)

        def replace_digits_with_wildcard(text):
            return re.sub(r"\b\d+\b", "*", text)

        def matches_to_keys(key, to_keys):
            return to_keys and any(to_key.startswith(replace_digits_with_wildcard(key)) for to_key in to_keys)

        def exact_match(key, to_keys):
            return to_keys and any(replace_digits_with_wildcard(key) == to_key for to_key in to_keys)

        def add_to_accumulator(key, value):
            if non_numerical != "ignore" or isinstance(value, int | float | bool | Sample | dict):
                if output_type == "dict" or to:
                    accumulator[key.rstrip(sep)] = value
                else:
                    accumulator.append(value)

        def flatten_recursive(obj, path=""):
            if exact_match(path.rstrip(sep), to):
                add_to_accumulator(path.rstrip(sep), obj)
            elif isinstance(obj, Sample | dict):
                items = obj.dump().items() if isinstance(obj, Sample) else obj.items()
                for k, v in items:
                    key_path = f"{path}{k}"
                    if to and not matches_to_keys(key_path, to):
                        continue
                    if not ignore_key(k, key_path):
                        flatten_recursive(v, key_path + sep)
            elif isinstance(obj, list | tuple | np.ndarray | torch.Tensor | Dataset):
                if isinstance(obj, np.ndarray | torch.Tensor):
                    obj = obj.tolist()
                for i, item in enumerate(obj):
                    flatten_recursive(item, f"{path}{i}{sep}")
            else:
                add_to_accumulator(path, obj)

        flatten_recursive(self)

        if non_numerical == "forbid" and any(
            not isinstance(v, int | float | bool)
            for v in (accumulator.values() if isinstance(accumulator, dict) else accumulator)
        ):  # noqa: E501
            raise ValueError("Non-numerical values found in flattened data.")

        if output_type == "np":
            return np.array(list(accumulator.values()) if isinstance(accumulator, dict) else accumulator)
        if output_type == "pt":
            return torch.tensor(list(accumulator.values()) if isinstance(accumulator, dict) else accumulator)

        if to:
            if output_type == "dict":
                return [{key: accumulator[key] for key in sorted(accumulator.keys())}]
            elif output_type == "list":
                return [list(accumulator.values())]
            elif output_type == "np":
                return [np.array(list(accumulator.values()))]
            elif output_type == "pt":
                return [torch.tensor(list(accumulator.values()))]
            else:
                raise ValueError(f"Unsupported output_type: {output_type}")
        else:
            return accumulator

    def schema(self, include_descriptions=False) -> Dict:
        """Returns a simplified json schema.

        Please note that for union types, only the first non-null type is considered.
        The schema is simplified by:
        -Removing additionalProperties boolean,
        -Merging allOf properties
        Args:
            schema (dict): A dictionary representing the JSON schema.
            include_descriptions (bool): Whether to include descriptions in the schema. Defaults to False.

        Returns:
            dict: A simplified JSON schema.
        """
        schema = self._extra.model_json_schema() if hasattr(self, "_extra") else self.model_json_schema()

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
            if isinstance(obj, dict):
                obj = Sample(**obj)
                _include_descriptions = False
            else:
                _include_descriptions = include_descriptions
            if "description" in schema and not include_descriptions:
                del schema["description"]
            if "additionalProperties" in schema:
                del schema["additionalProperties"]
            if "items" in schema:
                schema["items"] = simplify(schema["items"], obj[0])
            if "type" in schema and "ndarray" in schema["type"]:
                # Handle numpy arrays
                schema["type"] = "array"
                schema["items"] = {"type": "number"}
                del schema["properties"]
                del schema["required"]

            if "type" in schema and schema["type"] == "object":
                if "properties" not in schema:
                    schema = obj.schema(include_descriptions=_include_descriptions)
                for k, value in schema["properties"].items():
                    if isinstance(obj, Sample):
                        obj = obj.dict(recurse=False)
                    if k in obj:
                        schema["properties"][k] = simplify(value, obj[k])
                if not schema["properties"] and isinstance(obj, Sample):
                    schema = obj.schema(include_descriptions=_include_descriptions)
            if "title" in schema:
                del schema["title"]
            if "Title" in schema:
                del schema["Title"]
            if "allOf" in schema or "anyOf" in schema:
                raise ValueError(f"Schema contains allOf or anyOf which is unsupported: {schema}")
            return schema

        return simplify(schema, self)

    def infer_features_dict(self) -> Dict[str, Any]:
        """Infers features from the data recusively."""
        feat_dict = {}
        for k, v in self:
            if v is None:
                logging.info(f"Skipping {k} as it is None")
                continue
            if isinstance(v, Sample):
                feat_dict[k] = v.infer_features_dict()
            elif isinstance(v, list | tuple | np.ndarray):
                if isinstance(v[0], Sample):
                    feat_dict[k] = [v[0].infer_features_dict()]
                else:
                    feat_dict[k] = [to_features_dict(v[0])]
            else:
                feat_dict[k] = to_features_dict(v)
        return feat_dict

    @classmethod
    def read(cls, data: Any) -> "Sample":
        """Read a Sample instance from a JSON string or dictionary or path.

        Args:
            data (Any): The JSON string or dictionary to read.

        Returns:
            Sample: The read Sample instance.
        """
        logging.warning("The `read` method of Sample is not yet tested. Proceed with caution.")
        if isinstance(data, str):
            try:
                data = cls.model_validate(from_json(data))
            except Exception as e:
                logging.info(f"Error reading data: {e}. Attempting to read as JSON.")
                if isinstance(data, str):
                    if Path(data).exists():
                        if hasattr(cls, "open"):
                            data = cls.open(data)
                        else:
                            data = Path(data).read_text()
                            data = json.loads(data)
                else:
                    data = json.load(data)

        if isinstance(data, dict):
            return cls(**data)
        return cls(data)

    def to(self, container: Any) -> Any:
        """Convert the Sample instance to a different container type.

        Args:
            container (Any): The container type to convert to. Supported types are
            'dict', 'list', 'np', 'pt' (pytorch), 'space' (gym.space),
            'schema', 'json', 'hf' (datasets.Dataset) and any subtype of Sample.

        Returns:
            Any: The converted container.

        Examples:
        >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
        >>> sample.to("features")
        {'x': Value(dtype='float32', id=None), 'y': Value(dtype='float32', id=None), 'z': {'a': Value(dtype='float32', id=None), 'b': Value(dtype='float32', id=None)}, 'extra_field': Value(dtype='float32', id=None)}
        """
        if isinstance(container, Sample) and not issubclass(container, Sample):
            return container(**self.dump())
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

        try:
            return self.flatten(to=container)
        except Exception as e:
            raise ValueError(f"Unsupported container type: {container}") from e

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
        info: Annotated = None,
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

        if isinstance(value, int | float | list | tuple | np.ndarray):
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
                value = np.asfarray(value)
                shape = shape or value.shape
                dtype = dtype or value.dtype
                if bounds is None:
                    le = np.full(shape, -np.inf, dtype=dtype)
                    ge = np.full(shape, np.inf, dtype=dtype)
                elif isinstance(bounds, tuple):
                    le, ge = bounds
                return spaces.Box(low=le, high=ge, shape=shape, dtype=dtype)
            except Exception as e:
                logging.info(f"Could not convert value {value} to numpy array: {e}")
                if len(value) > 0 and isinstance(value[0], dict | Sample):
                    return spaces.Tuple(
                        [spaces.Dict(cls.space_for(v, max_text_length, info)) for v in value],
                    )
                return spaces.Tuple(
                    [spaces.Dict(cls.space_for(value[0], max_text_length, info)) for value in value[:1]],
                )
        raise ValueError(f"Unsupported object {value} of type: {type(value)} for space generation")

    @classmethod
    def init_from(cls, d: Any, pack=False) -> "Sample":
        if isinstance(d, spaces.Space):
            return cls.from_space(d)
        if isinstance(d, Union[Sequence, np.ndarray]):  # noqa: UP007
            if pack:
                return cls.pack_from(d)
            return cls.unflatten(d)
        if isinstance(d, dict):
            try:
                return cls.model_validate(d)
            except ValidationError as e:
                logging.info(f" Unable to validate {d} as {cls} {e}. Attempting to unflatten.")

                try:
                    return cls.unflatten(d)
                except Exception as e:
                    logging.info(f" Unable to unflatten {d} as {cls} {e}. Attempting to read.")
                    return cls.read(d)
        return cls(d)

    @classmethod
    def from_space(cls, space: spaces.Space) -> "Sample":
        """Generate a Sample instance from a Gym space."""
        sampled = space.sample()
        if isinstance(sampled, dict | OrderedDict):
            return cls(**sampled)
        if isinstance(sampled, np.ndarray | torch.Tensor | list | tuple):
            sampled = np.asarray(sampled)
            if len(sampled.shape) > 0 and isinstance(sampled[0], dict | Sample):
                return cls.pack_from(sampled)
        return cls(sampled)

    @classmethod
    def pack_from(cls, samples: List[Union["Sample", Dict]], padding: Literal["truncate"] = "truncate") -> "Sample":
        """Pack a list of samples or dicts into a single sample with lists for attributes.

        [Sample(a=1), Sample(a=2)] -> Sample(a=[1, 2])

        This is equivalent to a transpose or zip operation on the attributes or values of the samples.

        Truncates to the length of the shortest sample.
        """
        if padding != "truncate":
            raise ValueError(f"Unsupported padding: {padding}")

        if not samples:
            raise ValueError("No samples provided for packing.")

        if isinstance(samples[0], dict):
            attributes = list(samples[0].keys())
        elif isinstance(samples[0], Sample):
            attributes = list(samples[0].dump().keys())
        else:
            raise ValueError(f"Cannot determine attributes from the first sample: {samples[0]}")

        packed_attributes = {}
        for attr in attributes:
            packed_attributes[attr] = [
                sample[attr] if isinstance(sample, dict) else getattr(sample, attr) for sample in samples
            ]

        return cls(**packed_attributes)

    def unpack(self, to_dicts=False, padding: Literal["truncate"] = "truncate") -> List[Union["Sample", Dict]]:
        """Unpack the packed Sample object into a list of Sample objects or dictionaries.

        Sample(a=[1, 3, 5],  b=[2,4,6])    ->    [Sample(a=1, b=2), Sample(a=3, b=4), Sample(a=5, b=6)]


        This is the inverse of the pack_from method or a permute like ()

        Truncates to the length of the shortest sample.

        Example:
        >>> Sample(a=[1, 3, 5], b=[2, 4, 6]).unpack()
        [Sample(a=1, b=2), Sample(a=3, b=4), Sample(a=5, b=6)]
        """
        if padding != "truncate":
            raise ValueError(f"Unsupported padding: {padding}")

        attributes = [attr for attr in self.dump() if isinstance(getattr(self, attr), list)]

        unzipped = zip(*[getattr(self, attr) for attr in attributes], strict=False)

        if to_dicts:
            return [dict(zip(attributes, values, strict=False)) for values in unzipped]

        return [self.__class__(**dict(zip(attributes, values, strict=False))) for values in unzipped]

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
            info = self.model_field_info(key) if not isinstance(value, Sample) else value.model_info()
            if info:
                out[key] = info
        return out

    def model_field_info(self, key: str) -> Dict[str, Any]:
        """Get the extra json values set from a FieldInfo for a given attribute key.

        This can include bounds, shape, and other information.
        """
        info = {}
        if self.model_extra and key in self.model_extra:
            info = FieldInfo(annotation=self.model_extra[key]).json_schema_extra or {}
        if key in self.model_fields:
            info = self.model_fields[key].json_schema_extra or {}
        return info.get("_info", {})

    def space(self) -> spaces.Dict:
        """Return the corresponding Gym space for the Sample instance based on its instance attributes.

        Omits None values.

        Override this method in subclasses to customize the space generation.
        """
        space_dict = {}
        for key, value in self.model_dump(exclude_none=True).items():
            logging.debug("Generating space for key: '%s', value: %s", key, value)
            info = self.model_field_info(key)
            space_dict[key] = self.space_for(value, info=info)
        return spaces.Dict(space_dict)

    def random_sample(self) -> "Sample":
        """Generate a random Sample instance based on its instance attributes. Omits None values.

        Override this method in subclasses to customize the sample generation.
        """
        return self.__class__.model_validate(self.space().sample())

    def numpy(self) -> "Sample":
        """Convert the Sample instance to a numpy array."""
        return self.flatten("np")

    def tolist(self) -> "Sample":
        """Convert the Sample instance to a list."""
        return self.flatten("list")

    def torch(self) -> "Sample":
        import_module("torch")
        """Convert the Sample instance to a PyTorch tensor."""
        return self.flatten("pt")

    def json(self) -> str:
        """Convert the Sample instance to a JSON string."""
        return self.model_dump_json()

    def features(self) -> Features:
        """Convert the Sample instance to a HuggingFace Features object."""
        return Features(self.infer_features_dict())

    def dataset(self) -> Dataset:
        """Convert the Sample instance to a HuggingFace Dataset object."""
        data = self
        data = self.item if hasattr(self, "item") and self.item is not None else self.dump()
        if isinstance(data, list):
            return Dataset.from_list(data, features=self.features())
        if isinstance(data, dict):
            return Dataset.from_dict(data, features=self.features())
        if isinstance(data, Generator):
            return Dataset.from_generator(data, features=self.features())

        raise ValueError(f"Unsupported data type {type(data)} for conversion to Dataset.")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

    logging.basicConfig(level=logging.DEBUG, force=True)
    s = Sample(x=1, y=2, z={"a": 3, "b": 4, "c": np.array([1, 2, 3])}, extra_field=5)
