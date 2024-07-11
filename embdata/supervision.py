# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Literal

from pydantic import Field

SupervisionType = Literal["reward", "language"]


def SupervisionField(  # noqa
    supervision_type: SupervisionType,
    description: str,
    **kwargs,
):
    return Field(
        description=description,
        json_schema_extra={"supervision_type": supervision_type},
        **kwargs,
    )


def RewardField(  # noqa
    description: str,
    **kwargs,
):
    return SupervisionField(
        supervision_type="reward",
        description=description,
        **kwargs,
    )
