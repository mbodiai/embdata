# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import logging
import os
from typing import Any, Dict, Generator

import numpy as np
from datasets import Dataset, DatasetInfo, Features
from datasets import Image as HFImage
from rich import print

from embdata.describe import describe
from embdata.ds.oxe.load import get_hf_dataset
from embdata.ds.oxe.oxe import Metadata
from embdata.ds.oxe.schemas.bridge_ds import (
    Observation,
    State,
    Step,
)
from embdata.episode import VisionMotorEpisode
from embdata.geometry import Pose
from embdata.image import Image
from embdata.motion.control import AbsoluteHandControl, RelativePoseHandControl
from embdata.sample import Sample


def get_absolute_action(step, action_is_relative=False):
    if action_is_relative:
        state = np.array(step["observation"]["state"][:-1] + [0])  # Gripper is not relative.
        action = np.array(
            step["action"]["world_vector"] + step["action"]["rotation_delta"] + [int(step["action"]["open_gripper"])],
        )
        return AbsoluteHandControl(state + action)
    return AbsoluteHandControl(
        step["action"]["world_vector"] + step["action"]["rotation_delta"] + [int(step["action"]["open_gripper"])],
    )


global metadata
metadata = None
info = None
feat = None


def loop() -> Generator[Dict, Any, None]:
    for i in range(6):
        ds = Dataset.load_from_disk(f"bridge/shard_{i}")
        global metadata
        metadata = Metadata(**ds[0]["metadata"])
        global feat
        ds: Dataset = ds.map(to_vision_motor_episodes, batched=True, batch_size=None).to_list()
        feat = Features(feat)
        ds = Dataset.from_list(
            ds,
            info=DatasetInfo(
                citation='Padalkar, Abhishek, et al. "Open x-embodiment: Robotic learning datasets and rt-x models." arXiv preprint arXiv:2310.08864 (2023).',
                supervised_keys={
                    "input": "observation",
                    "output": "action",
                },
                features=feat,
                license="MIT",
                description="Open-X-Embodiment: Bridge V2",
                homepage="https://robotics-transformer-x.github.io/",
            ),
            features=feat,
        )
        ds.push_to_hub(
            "mbodiai/oxe_bridge_v2",
            config_name="default",
            private=False,
            split=f"shard_{i}",
            token=os.getenv("HF_WRITE"),
        )
        ds.save_to_disk(f"bridge_v2/shard_{i}")
        print("[green bold]Saved[/]", i)


loop()
# for i in range(7):
#   hf = get_hf_dataset(dataset_name="bridge",streaming=True).skip(i*5000).take(5000)
#   ds = Dataset.from_list(list(hf),features=hf.features)
#   ds.save_to_disk(f"bridge/shard_{i}")

#   ds.push_to_hub("mbodiai/oxe_bridge", token=os.getenv("HF_WRITE"))

# for i in range(20):
#   ds = Dataset.load_from_disk(f"bridge/shard_{i}")
#   print("Saved", i)

# print([a["action"]["open_gripper"] for a in ds[0]["episode"]["steps"]])


# import pandas as pd

# feat = ds.episodes[0].infer_features_dict()
# feat["episode_idx"] = Value("int32")
# feat["step_idx"] = Value("int32")
# feat = Features(feat)
# print(feat)
# print(ds.episodes[0].steps)
# print(ds.dump()["episodes"][0]["steps"])
# eps = ds.dump(exclude_none=False)["episodes"]

# print(ds.episodes[0].steps[10].dump().keys())
# print(type(ds.episodes[0].steps[0].observation.image.dump()))
# print(ds.dump(as_field="pil")["episodes"][0]["steps"][0]["observation"]["image"])
# print(ds.episodes[0].steps[0].model_info())
# print(ds.episodes[0].steps[0].dump()["action"])
# print(ds.episodes[0].steps[0].dump()["relative_action"])
