from pydantic import Field

from embdata.motion.control import MobileSingleHandControl


class LocobotAnswer(MobileSingleHandControl):
    answer: str = Field(
        default="",
        description="Short, one sentence answer to any question a user might have asked. 20 words max.",
    )
    sleep: bool = Field(
        default=False,
        description="Whether the robot should go to sleep after executing the motion.",
    )
    home: bool = Field(
        default=False,
        description="Whether the robot should go to home after executing the motion.",
    )
