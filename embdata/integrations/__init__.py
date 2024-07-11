# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pydantic import Field

from embdata.motion_controls import MobileSingleHandControl


class LocobotControl(MobileSingleHandControl):
    answer: str | None = Field(
        default="",
        description="Short, one sentence answer to any question a user might have asked. 20 words max.",
    )
    sleep: bool | None = Field(
        default=False,
        description="Whether the robot should go to sleep after executing the motion.",
    )
    home: bool | None = Field(
        default=False,
        description="Whether the robot should go to home after executing the motion.",
    )
