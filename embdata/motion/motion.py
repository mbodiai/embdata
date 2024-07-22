# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""This module contains the base class for a motion.

There are four basic motion types that are supported:
- Absolute motion: The desired absolute coordinates of a limb or joint in the chosen reference frame.
- Relative motion: The displacement from the current position of a limb or joint (frame-independent).
- Velocity motion: The desired absolute velocity of a limb or joint (frame-independent).
- Torque motion: The desired torque of a limb or joint (frame-independent).

The bounds is a list of two floats representing the lower and upper bounds of the motion.
The shape is a tuple of integers representing the shape of the motion.
The reference_frame is a string representing the reference frame for the coordinates (only applies to absolute motions).

To create a new Pydantic model for a motion, inherit from the Motion class and define pydantic fields with the MotionField,
function as you would with any other Pydantic field.

Example:
    from mbodied_agents.base.motion import Motion, AbsoluteMotionField, MotionField, MotionType, VelocityMotionField
    from mbodied_agents.base.sample import Sample

    class Twist(Motion):
        x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        roll: float = VelocityMotionField(default=0.0, bounds=['-pi', 'pi'])
        pitch: float = VelocityMotionField(default=0.0, bounds=['-pi', 'pi'])
        yaw: float = VelocityMotionField(default=0.0, bounds=['-pi', 'pi'])


This automatically generates a Pydantic model with the specified fields and the additional properties of a motion.
It is vectorizable, serializable, and validated according to its type. Furthermore, convience methods from
the class allow for direct conversion to numpy, pytorch, and gym spaces.
See the Sample class documentation for more information: https://mbodi-ai-mbodied-agents.readthedocs-hosted.com/en/latest/
See the Pydantic documentation for more information on how to define Pydantic models: https://pydantic-docs.helpmanual.io/
"""

from typing import Any

import numpy as np
from pydantic import ConfigDict, model_validator
from pydantic_core import PydanticUndefined
from typing_extensions import Literal

from embdata.geometry import Coordinate, CoordinateField
from embdata.ndarray import NumpyArray
from embdata.units import AngularUnit, LinearUnit

MotionType = Literal[
    "unspecified",
    "absolute",
    "relative",  # No different than an absolute motion but with a moving reference frame.
    "velocity",
    "torque",
    "other",
]


def MotionField(  # noqa
    default: Any = PydanticUndefined,  # noqa: N805
    bounds: list[float] | None = None,  # noqa: N802, D417
    shape: tuple[int] | None = None,
    description: str | None = None,
    reference_frame: str | None = None,
    unit: LinearUnit | AngularUnit = "m",
    motion_type: MotionType = "UNSPECIFIED",
    **kwargs,
) -> Any:
    """Create a field for a motion with specified properties.

    This function creates a CoordinateField with additional motion-specific properties.
    It's used to define fields in Motion subclasses.

    Args:
        default (Any): Default value for the field.
        bounds (list[float] | None): Lower and upper bounds of the motion.
        shape (tuple[int] | None): Shape of the motion data.
        description (str | None): Description of the motion.
        reference_frame (str | None): Reference frame for the coordinates.
        unit (LinearUnit | AngularUnit): Unit of measurement for the motion.
        motion_type (MotionType): Type of the motion (e.g., 'absolute', 'relative', 'velocity', 'torque', 'other').
        **kwargs: Additional keyword arguments to pass to CoordinateField.

    Returns:
        Any: A CoordinateField with motion-specific properties.

    Example:
        >>> class RobotArm(Motion):
        ...     shoulder: float = MotionField(default=0.0, bounds=[-90, 90], unit="deg", motion_type="absolute")
        ...     elbow: float = MotionField(default=0.0, bounds=[-120, 120], unit="deg", motion_type="absolute")
        ...     wrist: float = MotionField(default=0.0, bounds=[-180, 180], unit="deg", motion_type="absolute")
        >>> arm = RobotArm(shoulder=45, elbow=30, wrist=-15)
        >>> print(arm)
        RobotArm(shoulder=45.0, elbow=30.0, wrist=-15.0)
    """
    if description is None:
        description = f"{motion_type.lower()} motion"

    return CoordinateField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        reference_frame=reference_frame,
        unit=unit,
        **kwargs
    )


def AbsoluteMotionField(  # noqa
    default: Any = PydanticUndefined,  # noqa: N805
    bounds: list[float] | None = None,  # noqa: N802, D417
    shape: tuple[int] | None = None,
    description: str | None = None,
    reference_frame: str | None = None,
    unit: LinearUnit | AngularUnit = "m",
    **kwargs,
) -> Any:
    """Field for an absolute motion.

    This field is used to define the shape and bounds of an absolute motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        reference_frame=reference_frame,
        unit=unit,
        motion_type="absolute",
        **kwargs,
    )


def RelativeMotionField(  # noqa
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    reference_frame: str | None = None,
    unit: LinearUnit | AngularUnit = "m",
    **kwargs,
) -> Any:
    """Field for a relative motion."""
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        reference_frame=reference_frame,
        unit=unit,
        motion_type="relative",
        **kwargs,
    )


def VelocityMotionField(  # noqa
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    reference_frame: str | None = None,
    unit: LinearUnit | AngularUnit = "m",
    **kwargs,
) -> Any:
    """Field for a velocity motion."""
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        reference_frame=reference_frame,
        unit=unit,
        motion_type="velocity",
        **kwargs,
    )


def TorqueMotionField(  # noqa
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    reference_frame: str | None = None,
    unit: LinearUnit | AngularUnit = "m",
    **kwargs,
) -> Any:
    """Field for a torque motion."""
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        reference_frame=reference_frame,
        unit=unit,
        motion_type="torque",
        **kwargs,
    )


def AnyMotionField(  # noqa
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    reference_frame: str | None = None,
    unit: LinearUnit | AngularUnit = "m",
    **kwargs,
) -> Any:
    """Field for an other motion."""
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        reference_frame=reference_frame,
        unit=unit,
        motion_type="other",
        **kwargs,
    )


class Motion(Coordinate):
    """Base class for defining motion-related data structures.

    This class extends the Coordinate class and provides a foundation for creating
    motion-specific data models. It does not allow extra fields and enforces
    validation of motion type, shape, and bounds.

    Subclasses of Motion should define their fields using MotionField or its variants
    (e.g., AbsoluteMotionField, VelocityMotionField) to ensure proper validation and
    type checking.

    Attributes:
        Inherited from Coordinate

    Example:
        >>> class Twist(Motion):
        ...     x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        ...     y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        ...     z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        ...     roll: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
        ...     pitch: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
        ...     yaw: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
        >>> twist = Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)
        >>> print(twist)
        Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

    Note:
        The Motion class is designed to work with complex nested structures.
        It can handle various types of motion data, including images and text,
        as long as they are properly defined using the appropriate MotionFields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)

    def make_relative_to(self, other: "Motion") -> "Motion":
        """Make the motion relative to another motion.

        Args:
            other (Motion): The other motion to make this motion relative to.

        Returns:
            Motion: The relative motion.
        """
        return self.__class__((self.numpy() - other.numpy()).tolist())

    def make_absolute(self, reference: "Motion") -> "Motion":
        """Make the motion absolute with respect to another motion.

        Args:
            reference (Motion): The reference motion to make this motion absolute with respect to.

        Returns:
            Motion: The absolute motion.
        """
        return self.__class__((self.numpy() + reference.numpy()).tolist())

    def __add__(self, other: "Motion") -> "Motion":
        """Add two motions together."""
        return self.make_absolute(other)

    def __sub__(self, other: "Motion") -> "Motion":
        """Subtract two motions."""
        return self.make_relative_to(other)

    @model_validator(mode="after")
    def validate_bounds(self) -> "Motion":
        """Validate the bounds of the motion for each field and child of the motion."""
        for k, v in self:
            if isinstance(v, Motion):
                for kk, _ in v:
                    bounds = v.field_info(kk).get("bounds")
                    print(f"{kk} bounds: {bounds}")

                    # Set the child's bounds to the parent's bounds if the child's bounds are None.
                    if v.field_info(kk).get("bounds") is None:
                        v.add_field_info(kk, "bounds", self.field_info(k).get("bounds"))

            elif isinstance(v, int | float | np.ndarray | list) and self.field_info(k).get("bounds") is not None:
                bounds = self.field_info(k).get("bounds")
                if not isinstance(v, np.ndarray | list):
                    v = [v] # noqa
                for i, vv in enumerate(v):
                    if isinstance(vv, int | float) and not bounds[0] <= vv <= bounds[1]:
                        msg = f"{k} item {v} is out of bounds {self.fields[k].bounds}"
                        raise ValueError(msg)
        return self

