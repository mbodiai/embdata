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

from pydantic import ConfigDict, model_validator
from pydantic_core import PydanticUndefined
from typing_extensions import Literal

from embdata.geometry import Coordinate, CoordinateField
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
    """Field for a motion.

    Args:
        default: Default value for the field.
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
        motion_type: Type of the motion. Can be ['absolute', 'relative', 'velocity', 'torque', 'other'].
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
        **kwargs,
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
    """Base class for a motion. Does not allow extra fields.

    Subclass to validate the motion type, shape, and bounds of the motion.

    Example:
    >>> class Twist(Motion):
    ...     x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    ...     y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    ...     z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    ...     roll: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    ...     pitch: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    ...     yaw: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)


    @model_validator(mode="after")
    def validate_shape(self) -> "Motion":
        for key, value in self:
            shape = self.model_field_info(key).get("_shape", "undefined")
            if shape != "undefined":
                shape_processed = []
                value_processed = value
                while len(shape_processed) < len(shape):
                    shape_processed.append(len(value_processed))
                    if shape_processed[-1] != len(value_processed):
                        raise ValueError(
                            f"{key} value {value} of length {len(value_processed)} at dimension {len(shape_processed)-1}does not have the correct shape {shape}",
                        )
                    value_processed = value_processed[0]
        return self


class AnyMotionControl(Motion):
    """Motion Control with arbitrary fields but minimal validation. Should not be subclassed. Subclass Motion instead for validation.

    Pass in names, joints, and any other fields to create a motion control.

    Example:
        >>> class ArmControl(MotionControl):
        ...     names: list[str] = MotionField(default_factory=list, description="Names of the joints.")
        ...     joints: list[float] = MotionField(
        ...         default_factory=lambda: np.zeros(3), bounds=[-1.0, 1.0], shape=(3,), description="Values of the joints."
        ...     )
        >>> arm_control = ArmControl(names=["shoulder", "elbow", "wrist"], joints=[0.1, 0.2])
        Traceback (most recent call last):
            ...
        ValueError: Number of joints 2 does not match number of names 3
        >>> arm_control = ArmControl(names=["shoulder", "elbow", "wrist"], joints=[3.0, 2.0, 1.0])
        Traceback (most recent call last):
            ...
        ValueError: joints item 0 (3.0) is out of bounds [-1.0, 1.0]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", populate_by_name=True)

    names: list[str] | None = None
    joints: list[float] | None = None

    @model_validator(mode="after")
    def validate_joints(self) -> "AnyMotionControl":
        if self.joints is not None and self.names is not None and len(self.joints) != len(self.names):
            raise ValueError(f"Number of joints {len(self.joints)} does not match number of names {len(self.names)}")
        if self.joints is not None:
            for joint in self.joints:
                if not isinstance(joint, int | float):
                    raise ValueError(f"Joint {joint} is not a number")
        return self
