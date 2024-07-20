from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    List,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    FilePath,
    GetJsonSchemaHandler,
    PositiveInt,
    ValidationError,
    validate_call,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from pydantic_numpy.helper.annotation import (
    MultiArrayNumpyFile,
    pd_np_native_numpy_array_to_data_dict_serializer,
)
from pydantic_numpy.helper.validation import (
    validate_multi_array_numpy_file,
    validate_numpy_array_file,
)
from typing_extensions import TypedDict

SupportedDTypes = type[np.generic]

def create_array_validator(
    shape: Tuple[int, ...] | None, dtype: SupportedDTypes | None) -> Callable[[Any], npt.NDArray]:
    def array_validator(array_data: Any) -> npt.NDArray:
        if isinstance(array_data, dict):
            array = np.array(array_data["data"], dtype=array_data.get("dtype", None))
        elif isinstance(array_data, list | tuple | np.ndarray):
            array = np.array(array_data)
        else:
            msg = f"Unsupported type for array_data: {type(array_data)}"
            raise ValidationError(msg)

        if shape is not None:
            expected_ndim = len(shape)
            actual_ndim = array.ndim
            if actual_ndim != expected_ndim:
                msg = f"Array has {actual_ndim} dimensions, expected {expected_ndim}"
                raise ValidationError(msg)
            for i, (expected, actual) in enumerate(zip(shape, array.shape, strict=False)):
                if expected != -1 and expected is not None and expected != actual:
                    msg = f"Dimension {i} has size {actual}, expected {expected}"
                    raise ValueError(msg)

        if dtype and array.dtype.type != dtype:
            if issubclass(dtype, np.integer) and issubclass(array.dtype.type, np.floating):
                array = np.round(array).astype(dtype, copy=False)
            else:
                array = array.astype(dtype, copy=True)

        return array

    return array_validator


class NumpyDataDict(TypedDict):
    data: List
    data_type: SupportedDTypes


@validate_call
def _deserialize_numpy_array_from_data_dict(data_dict: NumpyDataDict) -> np.ndarray:
    return np.array(data_dict["data"]).astype(data_dict["data_type"])


_common_numpy_array_validator = core_schema.union_schema(
    [
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Path),
                core_schema.no_info_plain_validator_function(validate_numpy_array_file),
            ]
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(MultiArrayNumpyFile),
                core_schema.no_info_plain_validator_function(validate_multi_array_numpy_file),
            ]
        ),
        core_schema.is_instance_schema(np.ndarray),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence),
                core_schema.no_info_plain_validator_function(lambda v: np.asarray(v)),
            ]
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(dict),
                core_schema.no_info_plain_validator_function(_deserialize_numpy_array_from_data_dict),
            ]
        ),
    ]
)


def get_numpy_json_schema(
    _field_core_schema: core_schema.CoreSchema,
    _handler: GetJsonSchemaHandler,
    shape: List[PositiveInt] | None = None,
    data_type: SupportedDTypes | None = None,
) -> JsonSchemaValue:
    """Generates a JSON schema for a NumPy array field within a Pydantic model.

    This function constructs a JSON schema definition compatible with Pydantic models
    that are intended to validate NumPy array inputs. It supports specifying the data type
    and dimensions of the NumPy array, which are used to construct a schema that ensures
    input data matches the expected structure and type.

    Parameters
    ----------
    _field_core_schema : core_schema.CoreSchema
        The core schema component of the Pydantic model, used for building basic schema structures.
    _handler : GetJsonSchemaHandler
        A handler function or object responsible for converting Python types to JSON schema components.
    shape : Optional[List[PositiveInt]], optional
        The expected shape of the NumPy array. If specified, the schema will enforce that the input
    data_type : Optional[SupportedDTypes], optional
        The expected data type of the NumPy array elements. If specified, the schema will enforce
        that the input array's data type is compatible with this. If `None`, any data type is allowed,
        by default None.

    Returns:
    -------
    JsonSchemaValue
        A dictionary representing the JSON schema for a NumPy array field within a Pydantic model.
        This schema includes details about the expected array dimensions and data type.
    """
    array_shape = shape if shape else "Any"
    if data_type:
        array_data_type = data_type.__name__
        item_schema = core_schema.list_schema(
            items_schema=core_schema.any_schema(metadata=f"Must be compatible with numpy.dtype: {array_data_type}"),
        )
    else:
        array_data_type = "Any"
        item_schema = core_schema.list_schema(items_schema=core_schema.any_schema())

    if shape:
        data_schema = core_schema.list_schema(items_schema=item_schema, min_length=shape[0], max_length=shape[0])
    else:
        data_schema = item_schema

    return {
        "title": "Numpy Array",
        "type": f"np.ndarray[{array_shape}, np.dtype[{array_data_type}]]",
        "required": ["data_type", "data"],
        "properties": {
            "data_type": {"title": "dtype", "default": array_data_type, "type": "string"},
            "data": data_schema,
        },
    }


class NumpyArray:
    """A Pydantic field type for Numpy arrays. Shape and data type are validated.

    Lazy loading and caching by default.

    Usage:
    >>> from pydantic import BaseModel
    >>> from embdata.ndarray import NumpyArray
    >>> class MyModel(BaseModel):
    ...     uint8_array: NumpyArray[np.uint8]
    ...     must_have_exact_shape: NumpyArray[1, 2, 3]
    ...     must_be_3d: NumpyArray["*","*","*"] # NumpyArray[Any, Any, Any] also works.
    ...     must_be_1d: NumpyArray["*",] # NumpyArray[Any,] also works.
    """
    shape: ClassVar[Tuple[PositiveInt, ...] | None] = None
    dtype: ClassVar[SupportedDTypes | None] = None
    def __repr__(self) -> str:
        if TYPE_CHECKING:
            class_params = str(*self.shape) if self.shape is not None else "*"
            dtype = f", {self.dtype.__name__}" if self.dtype is not None else ", Any"

            return f"NumpyArray[{class_params}{dtype}]"
        return "NumpyArray"

    def __str__(self) -> str:
        return repr(self)


    @classmethod
    def __class_getitem__(cls, params=None) -> Any:
        _shape = None
        _dtype = None
        if params is None or params == "*" or params == Any or params == (Any,):
            params = ("*",)
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) == 1:
            if isinstance(params[0], type):
                _dtype = params[0]
        else:
            *_shape, _dtype = params
            _shape = tuple(s if s not in ("*", Any) else -1 for s in _shape)

        if _dtype is int:
            _dtype: SupportedDTypes | None = np.int64
        elif _dtype is float:
            _dtype = np.float64
        elif _dtype is not None and _dtype != "*" and _dtype != Any and isinstance(_dtype, type):
            _dtype = np.dtype(_dtype).type
        elif isinstance(_dtype, int):
            _shape += (_dtype,)

        if _shape == ():
            _shape = None

        class ParameterizedNumpyArray(cls):
            shape = _shape
            dtype = _dtype

        print(f"ParameterizedNumpyArray shape: {ParameterizedNumpyArray.shape}")
        print(f"ParameterizedNumpyArray dtype: {ParameterizedNumpyArray.dtype}")

        result = Annotated[np.ndarray | FilePath | MultiArrayNumpyFile, ParameterizedNumpyArray]
        print(f"__class_getitem__ returns: {result}")
        return result

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        np_array_validator = create_array_validator(cls.shape, cls.dtype)
        np_array_schema = core_schema.no_info_plain_validator_function(np_array_validator)

        return core_schema.json_or_python_schema(
            python_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.is_instance_schema(np.ndarray),
                            core_schema.is_instance_schema(list),
                            core_schema.is_instance_schema(tuple),
                            core_schema.is_instance_schema(dict),
                        ]
                    ),
                    np_array_schema,
                ]
            ),
            json_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.list_schema(),
                            core_schema.dict_schema(),
                        ]
                    ),
                    np_array_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                pd_np_native_numpy_array_to_data_dict_serializer,
                is_field_serializer=False,
                when_used="json-unless-none",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, field_core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return get_numpy_json_schema(field_core_schema, handler, cls.shape, cls.dtype)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    class MyModel(BaseModel):
        uint8_array: NumpyArray[np.uint8]
        must_have_exact_shape: NumpyArray[1, 2, 3]
        must_be_3d: NumpyArray["*","*","*"]
        must_be_1d: NumpyArray["*"]

    my_failing_model = MyModel(
        uint8_array=[1, 2, 3, 4],
        must_have_exact_shape=[[[1]], [[2]]],
        must_be_3d=[[[1, 2, 3], [4, 5, 6]]],
        must_be_1d=[[[1, 2, 3]]],
    )
    print(my_failing_model)
