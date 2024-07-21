from typing_extensions import Annotated
import numpy as np
import pytest
from pydantic import Field, ValidationError, ConfigDict
from pydantic.main import BaseModel
from embdata.ndarray import NumpyArray
from typing import Any


class ModelWithArrays(BaseModel):
    any_array: NumpyArray[Any]
    flexible_array: NumpyArray
    int_vector: NumpyArray[3, int]
    float_matrix: NumpyArray[2, 2, np.float64] = Field(default_factory=lambda: np.array([[1.0, 2.0], [3.0, 4.0]]))
    any_3d_array: NumpyArray["*", "*", "*", Any]
    any_float_array: NumpyArray[float] = Field(description="Any float array")
    array: NumpyArray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))


def test_model_with_arrays():
    model = ModelWithArrays(
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
    )

    assert isinstance(model.any_array, np.ndarray)
    assert isinstance(model.flexible_array, np.ndarray)
    assert isinstance(model.int_vector, np.ndarray)
    assert isinstance(model.float_matrix, np.ndarray)
    assert isinstance(model.any_3d_array, np.ndarray)
    assert isinstance(model.any_float_array, np.ndarray)


def test_serialization_deserialization():
    model = ModelWithArrays(
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        array=np.array([1.0, 2.0, 3.0]),
    )

    # Test serialization with model_dump
    serialized = model.model_dump()
    assert isinstance(serialized, dict)

    # Test serialization with model_dump_json
    serialized_json = model.model_dump_json()
    assert isinstance(serialized_json, str)

    # Test deserialization with model_validate
    deserialized = ModelWithArrays.model_validate(serialized)
    assert isinstance(deserialized, ModelWithArrays)

    # Test deserialization with model_validate_json
    deserialized_json = ModelWithArrays.model_validate_json(serialized_json)
    assert isinstance(deserialized_json, ModelWithArrays)

    # Compare original and deserialized models
    assert_models_equal(model, deserialized)
    assert_models_equal(model, deserialized_json)


def assert_models_equal(model1: ModelWithArrays, model2: ModelWithArrays):
    assert np.array_equal(model1.any_array, model2.any_array)
    assert np.array_equal(model1.flexible_array, model2.flexible_array)
    assert np.array_equal(model1.int_vector, model2.int_vector)
    assert np.array_equal(model1.float_matrix, model2.float_matrix)
    assert np.array_equal(model1.any_3d_array, model2.any_3d_array)
    assert np.array_equal(model1.any_float_array, model2.any_float_array)


def test_validation_errors():
    with pytest.raises(ValidationError):
        ModelWithArrays(
            any_array=np.array([1, 2, 3, 4]),  # This is fine
            flexible_array=np.array([1, 2, 3, 4]),  # This is fine
            int_vector=np.array([1, 2]),  # Wrong shape
            float_matrix=np.array([[1, 2], [3, 4]]),  # Wrong dtype
            any_3d_array=np.zeros((2, 3)),  # Wrong number of dimensions
            any_float_array=np.array([1, 2, 3]),  # Wrong dtype
        )


def test_edge_cases():
    # Test with empty arrays
    model = ModelWithArrays(
        any_array=np.array([]),
        flexible_array=np.array([[]]),
        int_vector=np.array([0, 0, 0]),
        float_matrix=np.array([[0.0, 0.0], [0.0, 0.0]]),
        any_3d_array=np.array([[[]]]),
        any_float_array=np.array([]),
        array=np.array([]),
    )
    assert model.any_array.size == 0
    assert model.flexible_array.size == 0
    assert np.all(model.int_vector == 0)
    assert np.all(model.float_matrix == 0.0)
    assert model.any_3d_array.size == 0
    assert model.any_float_array.size == 0

    # Test with extreme values
    model = ModelWithArrays(
        any_array=np.array([np.inf, -np.inf, np.nan], dtype=object),
        flexible_array=np.array([[np.finfo(np.float64).max, np.finfo(np.float64).min]]),
        int_vector=np.array([np.iinfo(np.int64).max, 0, np.iinfo(np.int64).min]),
        float_matrix=np.array([[np.inf, -np.inf], [np.nan, 0.0]]),
        any_3d_array=np.array([[[np.inf, -np.inf, np.nan]]]),
        any_float_array=np.array([np.finfo(np.float64).max, np.finfo(np.float64).min]),
    )
    assert np.any(np.isinf(model.any_array.astype(float)))
    assert np.any(np.isnan(model.any_array.astype(float)))
    assert np.all(model.int_vector == np.array([np.iinfo(np.int64).max, 0, np.iinfo(np.int64).min]))
    assert np.isinf(model.float_matrix).any()
    assert np.isnan(model.float_matrix).any()


def test_type_conversion():
    # Test passing lists instead of numpy arrays
    model = ModelWithArrays(
        any_array=[1, 2, 3, 4],
        flexible_array=[[1, 2], [3, 4]],
        int_vector=[1, 2, 3],
        float_matrix=[[1.0, 2.0], [3.0, 4.0]],
        any_3d_array=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        any_float_array=[1.0, 2.0, 3.0],
    )
    assert isinstance(model.any_array, np.ndarray)
    assert isinstance(model.flexible_array, np.ndarray)
    assert isinstance(model.int_vector, np.ndarray)
    assert isinstance(model.float_matrix, np.ndarray)
    assert isinstance(model.any_3d_array, np.ndarray)
    assert isinstance(model.any_float_array, np.ndarray)


def test_wrong_shape():
    class TestModelWithArrays(BaseModel):
        any_array: NumpyArray[Any]
        flexible_array: NumpyArray[Any]
        int_vector: NumpyArray[3, int]
        float_matrix: NumpyArray[2, 2, np.float64]
        any_3d_array: NumpyArray["*", "*", "*", Any]
        any_float_array: NumpyArray[float]
        array: NumpyArray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))

    with pytest.raises((ValidationError, TypeError)):
        model = TestModelWithArrays(
            any_array=[1, 2, 3, 4],
            flexible_array=[[1, 2], [3, 4]],
            int_vector=[1, 2, 3],
            float_matrix=[[1.0, 2.0], [3.0, 4.0]],
            any_3d_array=[[1, 2], [3, 4]],  # This should raise an error as it's 2D, not 3D
            any_float_array=[1.0, 2.0, 3.0],
        )

    # Test that correct shapes pass validation
    model = TestModelWithArrays(
        any_array=[1, 2, 3, 4],
        flexible_array=[[1, 2], [3, 4]],
        int_vector=[1, 2, 3],
        float_matrix=[[1.0, 2.0], [3.0, 4.0]],
        any_3d_array=[[[1, 2], [3, 4]]],  # This is now 3D
        any_float_array=[1.0, 2.0, 3.0],
    )
    assert isinstance(model.any_3d_array, np.ndarray)
    assert model.any_3d_array.ndim == 3


def test_specific_validation_errors():
    with pytest.raises(ValidationError):
        model = ModelWithArrays(
            any_array=[1, 2, 3, 4],
            flexible_array=[[1, 2], [3, 4]],
            int_vector=[1, 2],  # Wrong shape
            float_matrix=[[1.0, 2.0], [3.0, 4.0]],
            any_3d_array=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            any_float_array=[1.0, 2.0, 3.0],
        )
