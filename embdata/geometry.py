# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""Classes for representing geometric data in cartesian and polar coordinates.

A 3D pose represents the planar x, y, and theta, while a 6D pose represents the volumetric x, y, z, roll, pitch, and yaw.

Example:
    >>> import math
    >>> pose_3d = Pose3D(x=1, y=2, theta=math.pi / 2)
    >>> pose_3d.to("cm")
    Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)
    >>> pose_3d.to("deg")
    Pose3D(x=1.0, y=2.0, theta=90.0)
    >>> class BoundedPose6D(Pose6D):
    ...     x: float = CoordinateField(bounds=(0, 5))
    >>> pose_6d = BoundedPose6D(x=10, y=2, z=3, roll=0, pitch=0, yaw=0)
    Traceback (most recent call last):
    ...
    ValueError: x value 10 is not within bounds (0, 5)
"""

from typing import Any, Literal, TypeAlias

import numpy as np
from pydantic import ConfigDict, Field, create_model, model_validator
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation

from embdata.sample import Sample
from embdata.units import AngularUnit, LinearUnit, TemporalUnit

InfoUndefined = Literal["undefined"]


def CoordinateField(  # noqa
    default=0.0,
    default_factory=None,
    reference_frame="undefined",
    unit: LinearUnit | AngularUnit | TemporalUnit = "m",
    bounds: tuple | InfoUndefined = "undefined",
    description: str | None = None,
    **kwargs,
):
    """Create a Pydantic Field with extra metadata for coordinates.

    This function extends Pydantic's Field with additional metadata specific to coordinate systems,
    including reference frame, unit, and bounds information.

    Args:
        default: Default value for the field.
        default_factory: Factory for creating the default value.
        reference_frame: Reference frame for the coordinates.
        unit: Unit of the coordinate (LinearUnit, AngularUnit, or TemporalUnit).
        bounds: Tuple representing the allowed range for the coordinate value.
        description: Description of the field.
        **kwargs: Additional keyword arguments for field configuration.

    Returns:
        Field: Pydantic Field with extra metadata.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Robot(BaseModel):
        ...     x: float = CoordinateField(unit="m", bounds=(0, 10))
        ...     angle: float = CoordinateField(unit="rad", bounds=(0, 6.28))
        >>> robot = Robot(x=5, angle=3.14)
        >>> robot.dict()
        {'x': 5.0, 'angle': 3.14}
        >>> Robot(x=15, angle=3.14)
        Traceback (most recent call last):
        ...
        ValueError: x value 15.0 is not within bounds (0, 10)
    """
    json_schema_extra = {
        "_info": {
            "reference_frame": reference_frame,
            "unit": unit,
            "bounds": bounds,
            **kwargs,
        },
    }
    return Field(
        default=default,
        json_schema_extra=json_schema_extra,
        description=description,
        default_factory=default_factory,
    )


class Coordinate(Sample):
    """A list of numbers representing a coordinate in the world frame for an arbitrary space."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, populate_by_name=True)

    @staticmethod
    def convert_linear_unit(value: float, from_unit: str, to_unit: str) -> float:
        """Convert a value from one linear unit to another.

        This method supports conversion between meters (m), centimeters (cm),
        millimeters (mm), inches (in), and feet (ft).

        Args:
            value (float): The value to convert.
            from_unit (str): The unit to convert from.
            to_unit (str): The unit to convert to.

        Returns:
            float: The converted value.

        Examples:
            >>> Coordinate.convert_linear_unit(1.0, "m", "cm")
            100.0
            >>> Coordinate.convert_linear_unit(100.0, "cm", "m")
            1.0
            >>> Coordinate.convert_linear_unit(1.0, "m", "ft")
            3.280839895013123
            >>> Coordinate.convert_linear_unit(12.0, "in", "cm")
            30.48
        """
        conversion_from_factors = {
            "m": 1.0,
            "cm": 0.01,
            "mm": 0.001,
            "in": 0.0254,
            "ft": 0.3048,
        }
        conversion_to_factors = {
            "m": 1.0,
            "cm": 100.0,
            "mm": 1000.0,
            "in": 1.0 / 0.0254,
            "ft": 1.0 / 0.3048,
        }
        from_unit_factor = conversion_from_factors[from_unit]
        to_unit_factor = conversion_to_factors[to_unit]
        if from_unit == to_unit:
            return value
        return value * from_unit_factor * to_unit_factor

    @staticmethod
    def convert_angular_unit(value: float, from_unit: str, to_unit: str) -> float:
        """Convert a value from one angular unit to another.

        This method supports conversion between radians (rad) and degrees (deg).

        Args:
            value (float): The angular value to convert.
            from_unit (str): The unit to convert from ('rad' or 'deg').
            to_unit (str): The unit to convert to ('rad' or 'deg').

        Returns:
            float: The converted angular value.

        Examples:
            >>> Coordinate.convert_angular_unit(1.0, "rad", "deg")
            57.29577951308232
            >>> Coordinate.convert_angular_unit(180.0, "deg", "rad")
            3.141592653589793
            >>> Coordinate.convert_angular_unit(90.0, "deg", "deg")
            90.0
            >>> round(Coordinate.convert_angular_unit(np.pi / 2, "rad", "deg"), 2)
            90.0
        """
        convert_to_rad_from = {
            "rad": 1.0,
            "deg": np.pi / 180.0,
        }
        from_rad_convert_to = {
            "rad": 1.0,
            "deg": 180.0 / np.pi,
        }
        return value * convert_to_rad_from[from_unit] * from_rad_convert_to[to_unit]
    
    def __array__(self):
         """Return a numpy array representation of the pose."""
         return np.array([item for _, item in self])

    @model_validator(mode="after")
    def validate_bounds(self) -> Any:
        """Validate the bounds of the coordinate."""
        for key, value in self:
            bounds = self.field_info(key).get("bounds")
            if bounds and bounds != "undefined":
                if len(bounds) != 2 or not all(isinstance(b, int | float) for b in bounds):
                    msg = f"{key} bounds must consist of two numbers"
                    raise ValueError(msg)

                if hasattr(value, "shape") or isinstance(value, list | tuple):
                    for i, v in enumerate(value):
                        if not bounds[0] <= v <= bounds[1]:
                            msg = f"{key} item {i} ({v}) is out of bounds {bounds}"
                            raise ValueError(msg)
                elif not bounds[0] <= value <= bounds[1]:
                    msg = f"{key} value {value} is not within bounds {bounds}"
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_shape(self) -> "Coordinate":
        for key, value in self:
            shape = self.field_info(key).get("_shape", "undefined")
            if shape != "undefined":
                shape_processed = []
                value_processed = value
                while len(shape_processed) < len(shape):
                    shape_processed.append(len(value_processed))
                    if shape_processed[-1] != len(value_processed):
                        msg = f"{key} value {value} of length {len(value_processed)} at dimension {len(shape_processed)-1}does not have the correct shape {shape}"
                        raise ValueError(msg)
                    value_processed = value_processed[0]
        return self


class Pose3D(Coordinate):
    """Absolute coordinates for a 3D space representing x, y, and theta.

    This class represents a pose in 3D space with x and y coordinates for position
    and theta for orientation.

    Attributes:
        x (float): X-coordinate in meters.
        y (float): Y-coordinate in meters.
        theta (float): Orientation angle in radians.

    Examples:
        >>> import math
        >>> pose = Pose3D(x=1, y=2, theta=math.pi / 2)
        >>> pose
        Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)
        >>> pose.to("cm")
        Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)
    """

    x: float = CoordinateField(unit="m", default=0.0)
    y: float = CoordinateField(unit="m", default=0.0)
    theta: float = CoordinateField(unit="rad", default=0.0)

    def to(self, container_or_unit=None, unit="m", angular_unit="rad", **kwargs) -> Any:
        """Convert the pose to a different unit or container.

        This method allows for flexible conversion of the Pose3D object to different units
        or to a different container type.

        Args:
            container_or_unit (str, optional): The target container type or unit.
            unit (str, optional): The target linear unit. Defaults to "m".
            angular_unit (str, optional): The target angular unit. Defaults to "rad".
            **kwargs: Additional keyword arguments for field configuration.

        Returns:
            Any: The converted pose, either as a new Pose3D object with different units
                 or as a different container type.

        Examples:
            >>> import math
            >>> pose = Pose3D(x=1, y=2, theta=math.pi / 2)
            >>> pose.to("cm")
            Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)
            >>> pose.to("deg")
            Pose3D(x=1.0, y=2.0, theta=90.0)
            >>> pose.to("list")
            [1.0, 2.0, 1.5707963267948966]
            >>> pose.to("dict")
            {'x': 1.0, 'y': 2.0, 'theta': 1.5707963267948966}
        """
        if container_or_unit is not None and container_or_unit not in str(LinearUnit) + str(AngularUnit):
            return super().to(container_or_unit)

        if container_or_unit and container_or_unit in str(LinearUnit):
            unit = container_or_unit
        if container_or_unit and container_or_unit in str(AngularUnit):
            angular_unit = container_or_unit

        converted_fields = {}
        for key, value in self:
            if key in ["x", "y"]:
                converted_field = self.convert_linear_unit(value, self.field_info(key)["unit"], unit)
                converted_fields[key] = (converted_field, CoordinateField(converted_field, unit=unit, **kwargs))
            elif key == "theta":
                converted_field = self.convert_angular_unit(value, self.field_info(key)["unit"], angular_unit)
                converted_fields[key] = (converted_field, CoordinateField(converted_field, unit=angular_unit, **kwargs))
            else:
                converted_fields[key] = self.field_info(key)

        # Create new dynamic model with the same fields as the current model
        return create_model(
            "Pose3D",
            __base__=Coordinate,
            **{k: (float, v[1]) for k, v in converted_fields.items()},
        )(**{k: v[0] for k, v in converted_fields.items()})


PlanarPose: TypeAlias = Pose3D


class Pose6D(Coordinate):
    """Absolute coordinates for a 6D space representing x, y, z, roll, pitch, and yaw.
    
    Examples:
        >>> pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=np.pi / 2)
        >>> pose.to("cm")
        Pose6D(x=100.0, y=200.0, z=300.0, roll=0.0, pitch=0.0, yaw=1.5707963267948966)
        >>> pose.to("deg")
        Pose6D(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=90.0)
        >>> np.round(pose.to("quaternion"), 3)
        array([0.   , 0.   , 0.707, 0.707])
        >>> pose.to("rotation_matrix")
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])

    """

    x: float = CoordinateField(unit="m", default=0.0)
    y: float = CoordinateField(unit="m", default=0.0)
    z: float = CoordinateField(unit="m", default=0.0)
    roll: float = CoordinateField(unit="rad", default=0.0)
    pitch: float = CoordinateField(unit="rad", default=0.0)
    yaw: float = CoordinateField(unit="rad", default=0.0)

    @classmethod
    def from_position_orientation(cls, position, orientation) -> "Pose6D":
        if len(position) != 3 or len(orientation) != 4:
            msg = "Invalid position or orientation format"
            raise ValueError(msg)
        x, y, z = position
        qw, qx, qy, qz = orientation
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rotation.as_euler('xyz')
        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

    def __init__(self, **data):
        if "position" in data and "orientation" in data:
            pose = self.from_position_orientation(data["position"], data["orientation"])
            data = pose.model_dump()
        super().__init__(**data)

    @classmethod
    def from_position_orientation(cls, position, orientation):  # noqa: ANN206
        if len(position) != 3 or len(orientation) != 4:
            msg = "Invalid position or orientation format"
            raise ValueError(msg)
        x, y, z = position
        qw, qx, qy, qz = orientation
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rotation.as_euler("xyz")
        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

    def to(self, container_or_unit=None, sequence="zyx", unit="m", angular_unit="rad", **kwargs) -> Any:
        """Convert the pose to a different unit, container, or representation.

        This method provides a versatile way to transform the Pose6D object into various
        forms, including different units, rotation representations, or container types.

        Args:
            container_or_unit (str, optional): Target container, unit, or representation.
                Special values: "quaternion"/"quat"/"q", "rotation_matrix"/"rotation"/"R".
            sequence (str, optional): Sequence for Euler angles. Defaults to "zyx".
            unit (str, optional): Target linear unit. Defaults to "m".
            angular_unit (str, optional): Target angular unit. Defaults to "rad".
            **kwargs: Additional keyword arguments for field configuration.

        Returns:
            Any: The converted pose, which could be:
                - A new Pose6D object with different units
                - A quaternion (as numpy array)
                - A rotation matrix (as numpy array)
                - A different container type (e.g., list, dict)

        Examples:
            >>> pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=np.pi / 2)
            >>> pose.to("cm")
            Pose6D(x=100.0, y=200.0, z=300.0, roll=0.0, pitch=0.0, yaw=1.5707963267948966)
            >>> pose.to("deg")
            Pose6D(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=90.0)
            >>> np.round(pose.to("quaternion"), 3)
            array([0.   , 0.   , 0.707, 0.707])
            >>> pose.to("rotation_matrix")
            array([[ 0., -1.,  0.],
                   [ 1.,  0.,  0.],
                   [ 0.,  0.,  1.]])
            >>> pose.to("list")
            [1.0, 2.0, 3.0, 0.0, 0.0, 1.5707963267948966]
        """
        if container_or_unit in ("quaternion", "quat", "q"):
            return self.quaternion(sequence=sequence)
        if container_or_unit in ("rotation_matrix", "rotation", "R"):
            return self.rotation_matrix(sequence=sequence)
        if container_or_unit is not None and container_or_unit not in str(LinearUnit) + str(AngularUnit):
            return super().to(container_or_unit)

        if container_or_unit and container_or_unit in str(LinearUnit):
            unit = container_or_unit
        if container_or_unit and container_or_unit in str(AngularUnit):
            angular_unit = container_or_unit

        converted_fields = {}
        for key, value in self.dict().items():
            if key in ["x", "y", "z"]:
                converted_field = self.convert_linear_unit(value, self.field_info(key)["unit"], unit)
                converted_fields[key] = (converted_field, CoordinateField(converted_field, unit=unit, **kwargs))
            elif key in ["roll", "pitch", "yaw"]:
                converted_field = self.convert_angular_unit(value, self.field_info(key)["unit"], angular_unit)
                converted_fields[key] = (converted_field, CoordinateField(converted_field, unit=angular_unit, **kwargs))
            else:
                converted_fields[key] = self.field_info(key)

        # Create new dynamic model with the same fields as the current model
        return create_model(
            "Pose6D",
            __base__=Coordinate,
            **{k: (float, v[1]) for k, v in converted_fields.items()},
        )(**{k: v[0] for k, v in converted_fields.items()})

    def quaternion(self, sequence="zyx") -> np.ndarray:
        """Convert roll, pitch, yaw to a quaternion based on the given sequence.

        This method uses scipy's Rotation class to perform the conversion.

        Args:
            sequence (str, optional): The sequence of rotations. Defaults to "zyx".

        Returns:
            np.ndarray: A quaternion representation of the pose's orientation.

        Example:
            >>> pose = Pose6D(x=0, y=0, z=0, roll=np.pi / 4, pitch=0, yaw=np.pi / 2)
            >>> np.round(pose.quaternion(), 3)
            array([0.653, 0.271, 0.653, 0.271])
        """
        rotation = Rotation.from_euler(sequence, [self.roll, self.pitch, self.yaw])
        return rotation.as_quat()

    def to(self, container_or_unit=None, sequence="zyx", unit="m", angular_unit="rad", **kwargs) -> Any:
        if container_or_unit in ("quaternion", "quat", "q"):
            return self.quaternion(sequence=sequence)
        if container_or_unit in ("rotation_matrix", "rotation", "R"):
            return self.rotation_matrix(sequence=sequence)
        if container_or_unit == "euler":
            return np.array([self.roll, self.pitch, self.yaw])
        
        converted_fields = {}
        for key, value in self.model_dump().items():
            if key in ["x", "y", "z"]:
                converted_field = self.convert_linear_unit(value, self.field_info(key)["unit"], unit)
                converted_fields[key] = (converted_field, CoordinateField(converted_field, unit=unit, **kwargs))
            elif key in ["roll", "pitch", "yaw"]:
                converted_field = self.convert_angular_unit(value, self.field_info(key)["unit"], angular_unit)
                converted_fields[key] = (converted_field, CoordinateField(converted_field, unit=angular_unit, **kwargs))
            else:
                converted_fields[key] = (value, self.field_info(key))

        # Create new dynamic model with the same fields as the current model
        return create_model(
            self.__class__.__name__,
            __base__=Coordinate,
            **{k: (float, v[1]) for k, v in converted_fields.items()},
        )(**{k: v[0] for k, v in converted_fields.items()})

    def rotation_matrix(self, sequence="zyx") -> np.ndarray:
        """Convert roll, pitch, yaw to a rotation matrix based on the given sequence.

        This method uses scipy's Rotation class to perform the conversion.

        Args:
            sequence (str, optional): The sequence of rotations. Defaults to "zyx".

        Returns:
            np.ndarray: A 3x3 rotation matrix representing the pose's orientation.

        Example:
            >>> pose = Pose6D(x=0, y=0, z=0, roll=0, pitch=np.pi / 2, yaw=0)
            >>> np.round(pose.rotation_matrix(), 3)
            array([[ 0., -0.,  1.],
                   [ 0.,  1.,  0.],
                   [-1., -0.,  0.]])
        """
        rotation = Rotation.from_euler(sequence, [self.roll, self.pitch, self.yaw])
        return rotation.as_matrix()


Pose: TypeAlias = Pose6D

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
