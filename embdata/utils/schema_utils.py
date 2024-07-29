
import copy
import operator
from functools import reduce

import numpy as np


def unflatten_from_schema(obj, schema, klass) -> dict: # noqa: C901
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    elif hasattr(obj, "values"):
        obj = list(obj.values())
    if schema is None:
        msg = "Schema is required for unflattening a non-dictionary object."
        raise ValueError(msg)

    def unflatten_recursive(schema_part, index=0):
        if schema_part["type"] == "object":
            result = {} if schema_part.get("title") != "Sample" else klass()
            for prop, prop_schema in schema_part["properties"].items():
                value, index = unflatten_recursive(prop_schema, index)
                value = klass(**value) if prop_schema.get("title") == "Sample" else value
                result[prop] = value
            if schema_part.get("title") == "Sample":
                return klass(**result), index
            return result, index
        if schema_part["type"] == "array":
            items = []
            if schema_part.get("shape"):
                array = obj[index : index + sum(schema_part["shape"])]
                if all(isinstance(i, list | tuple | np.ndarray | float | int) for i in array):
                    result = np.array(array).reshape(schema_part["shape"])
                    index += reduce(operator.mul, schema_part["shape"], 1)
                    if schema_part.get("title") == "Sample":
                        return klass(**items), index
                    return result, index
            for _ in range(schema_part.get("maxItems", len(obj) - index)):
                value, index = unflatten_recursive(schema_part["items"], index)
                items.append(value)
            return items, index
        return obj[index], index + 1

    unflattened, _ = unflatten_recursive(schema)
    return unflattened



def resolve_refs(schema: dict, include=None) -> dict:
            def _resolve(obj, defs=None):  # noqa: C901
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


                    fallback = None
                    if "anyOf" in obj:
                        first_non_null = None
                        for item in obj["anyOf"]:
                            if "$ref" in item and "type" in item and item["type"] != "null":
                                first_non_null = item
                                break
                            if "type" in item and item["type"] == "null":
                                if item["type"] == "array" and include != "tensor":
                                    fallback = item
                                    continue
                                else:  # noqa: RET507
                                    break
                            first_non_null = item
                        first_non_null = first_non_null or fallback
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