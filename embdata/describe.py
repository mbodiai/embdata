# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from typing import Any, Dict

import numpy as np
from datasets import Dataset
from rich import print, print_json, table
from rich.console import Console


def as_table(ds: Any, sep: str = ".", show=True) -> Dict[str, Any]:
    """Render the dictionary as a rich table.

    This function takes a dataset or dictionary and renders it as a table using the rich library.
    It displays the keys and their corresponding values in a visually appealing format.

    Args:
        ds (Any): The dataset or dictionary to be rendered as a table.
        sep (str, optional): The separator used for nested keys. Defaults to ".".
        show (bool, optional): Whether to print the table. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary mapping of keys to their fully qualified names.

    Example:
        >>> data = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> as_table(data)
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ Key   ┃ Value  ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ a     │ 1      │
        │ b.c   │ 2      │
        │ b.d   │ 3      │
        └───────┴────────┘
        {'a': 'a', 'c': 'b.c', 'd': 'b.d'}
    """
    if hasattr(ds, "flatten"):
        ds = ds.flatten("dict")
    rendered = table.Table(title="Keys")
    rendered.add_column("Key", style="cyan")
    rendered.add_column("Value", style="magenta")
    for key, value in describe(ds, show=False).items():
        rendered.add_row(key, str(value))
    if show:
        console = Console()
        console.print(rendered)
    return ds


def full_paths(ds: Any, keys, sep: str = ".", show=False) -> Dict[str, Any]:
    return {k:v for k, v in describe_keys(ds, sep, show).items() if k in keys}

def describe_keys(ds: Any, sep: str = ".", show=False, path="") -> Dict[str, Any]:
    """Describe the keys of a nested dictionary or dataset.

    This function takes a nested dictionary or dataset and returns a flattened representation
    of its keys, where nested keys are joined using the specified separator.

    Args:
        ds (Any): The dataset or dictionary to describe.
        sep (str, optional): The separator used for nested keys. Defaults to ".".
        show (bool, optional): Whether to print the resulting keys. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary mapping of keys to their fully qualified names.

    Example:
        >>> data = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> describe_keys(data)
        {'a': 'a', 'c': 'b.c', 'd': 'b.d'}

    Example:
        >>> data = {"a": 1, "b": [{"c": 2, "d": 3}, {"c": 4, "f": e}]}
        >>> describe_keys(data)
        {'a': 'a', 'c': 'b.*.c', 'd': 'b.*.d', 'f': 'b.*.f'}
    """
    # Map each key to all its fully qualified key
    if isinstance(ds, list) and len(ds) > 0:
        ds = ds[0]
    if hasattr(ds, "dump"):
        ds = ds.dump()
    keys = {}
    if not hasattr(ds, "items"):
        return ""
    for key, value in ds.items():
        keys[key] = f"{path}{sep}{key}" if path and key not in keys  else key if key not in keys else keys[key]
        key = keys[key]
        if isinstance(value, dict):
            sub_keys = describe_keys(value, sep, False)
            if sub_keys:
                for sub_key, sub_value in sub_keys.items():
                    keys[sub_key] = f"{key}{sep}{sub_value}"
                    keys[f"{key}{sep}{sub_key}"] = f"{key}{sep}{sub_value}"
        elif isinstance(value, list | Dataset):
            if len(value) > 0:
                sub_keys = describe_keys(value[0], sep, False)
                if sub_keys:
                    for sub_key, sub_value in sub_keys.items():
                        keys[sub_key] = f"{key}{sep}*{sep}{sub_value}"
                        keys[f"{key}{sep}{sub_key}"] = f"{key}{sep}*{sep}{sub_value}"
    if show:
        print(keys)
    return keys


def describe(ds: Any, name: str = "", compact: bool = True, show=True, check_full=False) -> Dict[str, Any]:  # noqa: FBT001
    """Generate a schema-like description of a dataset or dictionary.

    This function creates a schema-like description of the input data structure,
    which can be either compact or detailed based on the 'compact' parameter.
    Set 'full' to check that every item in the list is of the same type.

    Args:
        ds (Any): The dataset or dictionary to describe.
        name (str, optional): The name of the root object. Defaults to "".
        compact (bool, optional): Whether to generate a compact description. Defaults to True.
        show (bool, optional): Whether to print the resulting schema. Defaults to True.
        check_full (bool, optional): Whether to check that every item in the list is of the same type. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary representing the schema of the input data.

    Example:
        >>> data = {"a": 1, "b": [{"c": 2, "d": 3}]}
        >>> describe(data)
        {'a': {'type': 'int'}, 'b': {'type': 'array', 'items': {'c': {'type': 'int'}, 'd': {'type': 'int'}}}}
    """
    if hasattr(ds, "dict"):
        ds = ds.dict()
    if compact:
        schema = {
            "type": "object"
            if isinstance(ds, dict)
            else "array"
            if isinstance(ds, list | Dataset)
            else str(type(ds).__name__),
        }
    else:
        schema = {
            "name": name,
            "type": "object"
            if isinstance(ds, dict)
            else "array"
            if isinstance(ds, list | Dataset)
            else str(type(ds).__name__),
        }
    if isinstance(ds, list | (dict | Dataset)):
        schema["length"] = len(ds)

    if isinstance(ds, list | Dataset) and ds:
        if check_full:
            schemas = {str(describe(item, compact=compact, show=False, name=name)) for item in ds}
            if len(schemas) != 1:
                msg = f"Items in list are not of the same type: {schemas}. Pass `check_full=False` to ignore."
                raise ValueError(msg)
        schema["items"] = describe(ds[0], compact=compact, show=False, name=name)
    elif isinstance(ds, dict) and not compact:
        schema["properties"] = {key: describe(value, key, compact, show=False, name=name) for key, value in ds.items()}
    elif isinstance(ds, dict) and compact:
        schema = {key: describe(value, key, compact, show=False) for key, value in ds.items()}

    if show:
        print_json(data=schema)
    return schema


def dict_to_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dictionary to a JSON Schema-like structure.

    This function takes a dictionary and converts it into a JSON Schema-like structure,
    inferring types and handling nested structures.

    Args:
        d (Dict[str, Any]): The input dictionary to convert.

    Returns:
        Dict[str, Any]: A JSON Schema-like representation of the input dictionary.

    Example:
        >>> data = {"a": 1, "b": {"c": "hello", "d": [1, 2, 3]}}
        >>> dict_to_schema(data)
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'integer'},
                'b': {
                    'type': 'object',
                    'properties': {
                        'c': {'type': 'string'},
                        'd': {'type': 'array', 'items': {'type': 'integer'}}
                    },
                    'required': ['c', 'd']
                }
            },
            'required': ['a', 'b']
        }
    """
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    for k, v in d.items():
        print(f"Processing key: {k} with value: {v}")
        schema["required"].append(k)
        if isinstance(v, dict):
            schema["properties"][k] = dict_to_schema(v)
        elif isinstance(v, list | Dataset):
            if len(v) > 0:
                schema["properties"][k] = {
                    "type": "array",
                    "items": dict_to_schema(v[0]) if isinstance(v[0], dict) else infer_type(v[0]),
                }
            else:
                schema["properties"][k] = {"type": "array", "items": {}}
        else:
            schema["properties"][k] = infer_type(v)

    return schema


def infer_type(value: Any) -> Dict[str, str]:
    """Infer the type of a given value and return it as a dictionary.

    This function takes a value and returns a dictionary describing its type,
    which is compatible with JSON Schema format.

    Args:
        value (Any): The value to infer the type from.

    Returns:
        Dict[str, str]: A dictionary with a 'type' key describing the inferred type.

    Raises:
        TypeError: If the value's type is not supported.

    Example:
        >>> infer_type(42)
        {'type': 'integer'}
        >>> infer_type("hello")
        {'type': 'string'}
        >>> infer_type([1, 2, 3])
        {'type': 'array', 'items': {'type': 'integer'}}
    """
    if isinstance(value, int | np.integer | np.int32 | np.int64):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, bytes):
        return {"type": "bytes"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, np.ndarray):
        return {"type": "array", "items": infer_type(value[0])}
    raise TypeError(f"Unsupported data type: {type(value)}")
