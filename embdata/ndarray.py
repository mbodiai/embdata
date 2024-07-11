import pydantic_numpy.typing as pnd

"""
This module provides type aliases for NumPy arrays to be used with Pydantic models.
These types ensure proper validation and serialization of NumPy arrays in Pydantic models.
"""

NumpyArray = pnd.NpNDArray
"""
Type alias for a generic NumPy ndarray.

This type can be used to annotate fields in Pydantic models that should contain
NumPy arrays of any dtype and shape.

Example:
    class MyModel(pydantic.BaseModel):
        data: NumpyArray

    model = MyModel(data=np.array([1, 2, 3]))
"""

NumpyArrayFp32 = pnd.NpNDArrayFp32
"""
Type alias for a NumPy ndarray with dtype float32.

This type can be used to annotate fields in Pydantic models that should contain
NumPy arrays specifically of dtype float32.

Example:
    class MyModel(pydantic.BaseModel):
        data: NumpyArrayFp32

    model = MyModel(data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
"""
