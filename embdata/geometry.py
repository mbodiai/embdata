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
    Pose3D(
        x=100.0,
        y=200.0,
        theta=1.5707963267948966,
    )
    >>> pose_3d.to("deg")
    Pose3D(
        x=1.0,
        y=2.0,
        theta=90.0,
    )
    >>> class BoundedPose6D(Pose6D):
    ...     x: float = CoordinateField(bounds=(0, 5))
    >>> pose_6d = BoundedPose6D(x=10, y=2, z=3, roll=0, pitch=0, yaw=0)
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for BoundedPose6D
    x
      Input should be less than or equal to 5 [type=less_than_equal, input_value=10, input_type=int]
        For further information visit https://errors.pydantic.dev/2.8/v/less_than_equal
"""

import math
from typing import Any, List, Literal, Tuple, Type, TypeAlias, TypeVar

import numpy as np
from pydantic import ConfigDict, Field, create_model, model_validator
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
    ge = le = None
    if bounds != "undefined" and bounds is not None:
        ge, le = bounds
    return Field(
        default=default,
        json_schema_extra={
            "_info": {
                "reference_frame": reference_frame,
                "unit": unit,
                "bounds": bounds,
                **kwargs,
            },
        },
        description=description,
        default_factory=default_factory,
        ge=ge,
        le=le,
    )



from typing import Any

class Coordinate(Sample):
    """A list of numbers representing a coordinate in the world frame for an arbitrary space."""
    
    model_fields: dict[str, Any] = {}
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
            >>> round(Coordinate.convert_linear_unit(12.0, "in", "cm"), 2)
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

    def relative_to(self, other: "Coordinate") -> "Coordinate":
        return self.__class__.unflatten(self.numpy() - other.numpy())

    def absolute_from(self, other: "Coordinate") -> "Coordinate":
        return self.__class__.unflatten(self.numpy() + other.numpy())

    def __add__(self, other: "Coordinate") -> "Coordinate":
        """Add two motions together."""
        return self.absolute_from(other)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        """Subtract two motions."""
        return self.relative_to(other)

    def __array__(self):
        """Return a numpy array representation of the pose."""
        return np.array([item for _, item in self])

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_bounds()
        self.validate_shape()

    @model_validator(mode="after")
    def validate_bounds(self) -> "Coordinate":
        """Validate the bounds of the coordinate."""
        for key, value in self:
            bounds = self.model_fields[key].json_schema_extra.get("_info", {}).get("bounds")
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
        """Validate the shape of the coordinate."""
        for key, value in self:
            shape = self.model_info().get(key, {}).get("shape")
            if shape != "undefined" and shape is not None:
                shape_processed = []
                value_processed = value
                while len(shape_processed) < len(shape):
                    shape_processed.append(len(value_processed))
                    if shape_processed[-1] != len(value_processed):
                        msg = f"{key} value {value} of length {len(value_processed)} at dimension {len(shape_processed)-1} does not have the correct shape {shape}"
                        raise ValueError(msg)
                    value_processed = value_processed[0]
        return self

Coords: TypeAlias = Coordinate
Point: TypeAlias = Coordinate

T = TypeVar("T", bound="Pose3D")

class Pose3D(Coordinate):
    model_config = ConfigDict(repr_str_template="x={x:.3f}, y={y:.3f}, theta={theta:.3f}")
    """Absolute coordinates for a 3D space representing x, y, and theta.

    This class represents a pose in 3D space with x and y coordinates for position
    and theta for orientation.

    Attributes:
        x (float): X-coordinate in meters.
        y (float): Y-coordinate in meters.
        theta (float): Orientation angle in radians.

    Examples:
        >>> import math
        >>> pose = Pose3D(1, 2, math.pi / 2)
        >>> pose
        Pose3D(x=1.0, y=2.0, theta=1.571)
        >>> pose = Pose3D([1, 2, math.pi / 2])
        >>> pose
        Pose3D(
            x=1.0,
            y=2.0,
            theta=1.5707963267948966
        )
        >>> pose = Pose3D(x=1, y=2, theta=math.pi / 2)
        >>> pose
        Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)
        >>> pose.to("cm")
            Pose3D(
                x=1.0,
                y=2.0,
                theta=90.0,
            )

    Usage:
        from embdata.geometry import Pose3D
        import math

        # Create a Pose3D object
        pose = Pose3D(1, 2, math.pi / 2)

        # Or using keyword arguments
        pose = Pose3D(x=1, y=2, theta=math.pi / 2)

        # Convert to different units
        pose_cm = pose.to("cm")
        pose_deg = pose.to("deg")

        # Access attributes
        print(pose.x, pose.y, pose.theta)

        # Perform relative positioning
        other_pose = Pose3D(2, 3, math.pi / 4)
        relative_pose = pose.relative_to(other_pose)
    """

    x: float = CoordinateField(unit="m", default=0.0)
    y: float = CoordinateField(unit="m", default=0.0)
    theta: float = CoordinateField(unit="rad", default=0.0)

    def __init__(self: T, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list | tuple) and len(args[0]) == 3:
            super().__init__(x=args[0][0], y=args[0][1], theta=args[0][2])
        elif len(args) == 3:
            super().__init__(x=args[0], y=args[1], theta=args[2])
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_position_orientation(cls: Type[T], position: Tuple[float, float] | List[float], orientation: float) -> T:
        return cls(x=position[0], y=position[1], theta=orientation)

    def to(self, container_or_unit=None, unit="m", angular_unit="rad", **kwargs) -> Any:
        if container_or_unit == "cm":
            return Pose3D(x=self.x * 100, y=self.y * 100, theta=self.theta)
        if container_or_unit == "deg":
            return Pose3D(x=self.x, y=self.y, theta=math.degrees(self.theta))
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




T = TypeVar("T", bound="Pose6D")

class Pose6D(Coordinate):
    model_config = ConfigDict(repr_str_template="x={x:.3f}, y={y:.3f}, z={z:.3f}, roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
    """Absolute coordinates for a 6D space representing x, y, z, roll, pitch, and yaw.

    Examples:
        >>> pose = Pose6D(1, 2, 3, 0, 0, np.pi / 2)
        >>> pose
        Pose6D(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=1.571)
        >>> pose = Pose6D([1, 2, 3, 0, 0, np.pi / 2])
        >>> pose
        Pose6D(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=1.5707963267948966)
        >>> pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=np.pi / 2)
        >>> pose
        Pose6D(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=1.5707963267948966)
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

    Usage:
        from embdata.geometry import Pose6D
        import numpy as np

        # Create a Pose6D object
        pose = Pose6D(1, 2, 3, 0, 0, np.pi / 2)

        # Or using keyword arguments
        pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=np.pi / 2)

        # Convert to different units or representations
        pose_cm = pose.to("cm")
        pose_deg = pose.to("deg")
        quaternion = pose.to("quaternion")
        rotation_matrix = pose.to("rotation_matrix")

        # Access attributes
        print(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)

        # Perform relative positioning
        other_pose = Pose6D(2, 3, 4, np.pi/4, 0, np.pi/2)
        relative_pose = pose.relative_to(other_pose)

        # Create from position and orientation
        position = [1, 2, 3]
        orientation = [0, 0, 0, 1]  # quaternion
        pose = Pose6D.from_position_orientation(position, orientation)
    """

    x: float = CoordinateField(unit="m", default=0.0)
    y: float = CoordinateField(unit="m", default=0.0)
    z: float = CoordinateField(unit="m", default=0.0)
    roll: float = CoordinateField(unit="rad", default=0.0)
    pitch: float = CoordinateField(unit="rad", default=0.0)
    yaw: float = CoordinateField(unit="rad", default=0.0)

    def __init__(self: T, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list | tuple) and len(args[0]) == 6:
            super().__init__(x=args[0][0], y=args[0][1], z=args[0][2], roll=args[0][3], pitch=args[0][4], yaw=args[0][5])
        elif len(args) == 6:
            super().__init__(x=args[0], y=args[1], z=args[2], roll=args[3], pitch=args[4], yaw=args[5])
        elif "position" in kwargs and "orientation" in kwargs:
            pose = self.from_position_orientation(kwargs["position"], kwargs["orientation"])
            super().__init__(**pose.model_dump())
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_position_orientation(cls: Type[T], position: Tuple[float, float, float] | List[float], orientation: Tuple[float, float, float, float] | List[float]) -> T:  # noqa: E501
        if len(position) != 3 or len(orientation) != 4:
            msg = "Invalid position or orientation format"
            raise ValueError(msg)
        x, y, z = position
        qw, qx, qy, qz = orientation
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rotation.as_euler("xyz")
        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

    def to(self, container_or_unit=None, sequence="zyx", unit="m", angular_unit="rad", **kwargs) -> Any:
        if container_or_unit == "cm":
            return Pose6D(x=self.x * 100, y=self.y * 100, z=self.z * 100, roll=self.roll, pitch=self.pitch, yaw=self.yaw)
        if container_or_unit == "deg":
            return Pose6D(x=self.x, y=self.y, z=self.z, roll=math.degrees(self.roll), pitch=math.degrees(self.pitch), yaw=math.degrees(self.yaw))
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

    def quaternion(self, sequence="xyz") -> np.ndarray:
        """Convert roll, pitch, yaw to a quaternion based on the given sequence.

        This method uses scipy's Rotation class to perform the conversion.

        Args:
            sequence (str, optional): The sequence of rotations. Defaults to "xyz".

        Returns:
            np.ndarray: A quaternion representation of the pose's orientation.

        Example:
            >>> pose = Pose6D(x=0, y=0, z=0, roll=np.pi / 4, pitch=0, yaw=np.pi / 2)
            >>> np.round(pose.quaternion(), 3)
            array([0.271, 0.271, 0.653, 0.653])
        """
        rotation = Rotation.from_euler(sequence, [self.roll, self.pitch, self.yaw])
        return rotation.as_quat()

    def rotation_matrix(self, sequence="xyz") -> np.ndarray:
        """Convert roll, pitch, yaw to a rotation matrix based on the given sequence.

        This method uses scipy's Rotation class to perform the conversion.

        Args:
            sequence (str, optional): The sequence of rotations. Defaults to "xyz".

        Returns:
            np.ndarray: A 3x3 rotation matrix representing the pose's orientation.

        Example:
            >>> pose = Pose6D(x=0, y=0, z=0, roll=0, pitch=0, yaw=np.pi / 2)
            >>> np.round(pose.rotation_matrix(), 3)
            array([[ 0., -1.,  0.],
                   [ 1.,  0.,  0.],
                   [ 0.,  0.,  1.]])
        """
        rotation = Rotation.from_euler(sequence, [self.roll, self.pitch, self.yaw])
        return rotation.as_matrix()


Pose: TypeAlias = Pose6D

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
