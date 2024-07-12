# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import logging
import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
from datasets import (
    Dataset,
    DatasetInfo,
    IterableDataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from datasets.features import Features
from datasets.features import Image as HFImage
from importlib_resources import files
from pydantic import Field

from embdata.describe import describe
from embdata.episode import VisionMotorEpisode, VisionMotorStep
from embdata.geometry import Pose
from embdata.image import Image
from embdata.motion import Motion
from embdata.motion.control import RelativePoseHandControl
from embdata.sample import Sample
from embdata.schema.metadata import Metadata


def get_oxe_metadata(dataset_name: str = "utokyo_xarm_pick_and_place_converted_externally_to_rlds") -> dict:
    path = files("embdata") / Path("schema/oxe/oxe_metadata.csv")
    oxe_dataset_metadata = pd.read_csv(path)

    if dataset_name not in oxe_dataset_metadata["Registered Dataset Name"].values:  # noqa: PD011
        raise ValueError(f"Dataset {dataset_name} not found in OXE Overview metadata.") # noqa

    return oxe_dataset_metadata[oxe_dataset_metadata["Registered Dataset Name"] == dataset_name].to_dict(
        orient="records",
    )[0]


def get_hf_dataset(
    dataset_path: str = "jxu124/OpenX-Embodiment",
    dataset_name: str = "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    split: str = "train",
    streaming: bool = False, # noqa
    download_mode: str | None = None,
) -> Dataset:
    logging.info(f"Fetching dataset {dataset_path}/{dataset_name}") # noqa
    ds = load_dataset(dataset_path, dataset_name, streaming=streaming, split=split, download_mode=download_mode)
    return ds.map(
        lambda example: {
            "metadata": get_oxe_metadata(dataset_name),
            "episode": example["data.pickle"],
        },
        remove_columns=["__url__", "data.pickle"],
    )


def get_state(step: Dict) -> Motion:
    return {
        "end_effector_pose": Pose(step["observation"]["state"]),
        "is_first": step["is_first"],
        "is_last": step["is_last"],
        "is_terminal": step["is_terminal"],
    }

def get_relative_action(step: Dict) -> Motion:
    return RelativePoseHandControl(
        step["action"]["world_frame"] + step["action"]["rotation_delta"] + step["action"]["gripper_open"],
    )

def get_absolute_action(step: Dict) -> Motion:

    return RelativePoseHandControl(
       get_relative_action(step).numpy() + np.array(get_state(step)["end_effector_pose"] + [0]),
    )

def push_to_hub(
    repo_id: str, 
    action_keys: set[str] | None = None, 
    image_keys: set[str] | None = None, 
    observation_keys: set[str] | None = None, 
    state_keys: set[str] | None = None,
    task_keys: set[str] | None = None,
    get_relative_action: Callable[[Dict], Motion] | None = None,
    get_absolute_action: Callable[[Dict], Motion] | None = None,
    get_action: Callable[[Dict], Motion] | None = None,
      **kwargs):
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        repo_id (str): The repository id to push to.
        action_keys (set[str], optional): The action keys. Defaults to None.
        image_keys (set[str], optional): The image keys. Defaults to None.
        observation_keys (set[str], optional): The observation keys. Defaults to None.
        state_keys (set[str], optional): The state keys. Defaults to None.
        metadata_keys (set[str], optional): The metadata keys. Defaults to None.
        **kwargs: Additional keyword arguments for `load_dataset`.
    """
    ds = load_dataset(repo_id, **kwargs)
    describe(ds, compact=True)
    config_names = get_dataset_config_names(repo_id)
    split_names = get_dataset_split_names(repo_id)
    for cn in config_names:
        for sn in split_names:
            ds = load_dataset(repo_id, config_name=cn, split=sn, **kwargs)
            describe(ds, compact=True)
            _push_to_hub(
                ds,
                action_keys=action_keys,
                image_keys=image_keys,
                observation_keys=observation_keys,
                state_keys=state_keys,
                task_keys=task_keys,
                get_relative_action=get_relative_action,
                get_absolute_action=get_absolute_action,
                get_action=get_action,
            )

def _push_to_hub(
    dataset: Dataset | IterableDataset,
    action_keys: set[str] | None = None,
    image_keys: set[str] | None = None,
    observation_keys: set[str] | None = None,
    state_keys: set[str] | None = None,
    task_keys: set[str] | None = None,
    get_relative_action: Callable[[Dict], Motion] | None = None,
    get_absolute_action: Callable[[Dict], Motion] | None = None,
    get_state: Callable[[Dict], Motion] | None = None,
    **kwargs,
) -> None:
    if action_keys is None:
        action_keys = {"action"}
    if image_keys is None:
        image_keys = {"image"}
    if observation_keys is None:
        observation_keys = {"observation"}
    if state_keys is None:
        state_keys = {"state"}


    if isinstance(dataset, Dataset):
        first_step = Sample(dataset[0])
    elif isinstance(dataset, IterableDataset):
        first_step = Sample(next(iter(dataset)))

    action = zip(*[first_step.flatten(to=a) for a in action_keys])
    observation = zip(*[first_step.flatten(to=o) for o in observation_keys])
    state = zip(*[first_step.flatten(to=s) for s in state_keys])
    task = zip(*[first_step.flatten(to=t) for t in task_keys])
    image = zip(*[first_step.flatten(to=i) for i in image_keys])


    metadata = Metadata(dataset[0]["metadata"])
    step = dataset[0]["episode"]["steps"][0]
    feat = VisionMotorStep(
        image=Image(step["observation"]["image"]["bytes"]),
        task=step["observation"]["natural_language_instruction"],
        action=get_relative_action(step),
        observation={
            "image": Image(step["observation"]["image"]["bytes"]),
            "task": step["observation"]["natural_language_instruction"],
        },
        episode_idx=0,
        step_idx=0,
        absolute_action=get_absolute_action(step, action_is_relative=True),
        supervision=step["reward"],
        state=get_state(step)).infer_features_dict()
    feat = {
        "image": HFImage(),
        **{k: v for k, v in feat.items() if k != "image"},
        "metadata": metadata.infer_features_dict(),
    }
    feat = Features(feat)
    ds = list(get_hf_dataset(dataset_name="bridge", streaming=True).skip(10).take(10))
    ds: Dataset = (
        Dataset.from_list(ds)
        .map(
            to_vision_motor_episodes,
            num_proc=4,
            batched=True,
            batch_size=500,
            features=feat,
            remove_columns=["episode", "__key__"],
        )
        .to_list()
    )
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
        split=f"shard_{0}",
        token=os.getenv("HF_WRITE"),
    )

    describe(feat, compact=True, show=True)
    for i in range(30):
        ds = list(get_hf_dataset(dataset_name="bridge", streaming=True).skip(i * 1000).take(1000))
        ds: Dataset = Dataset.from_list(ds).map(
            to_vision_motor_episodes,
            num_proc=4,
            batched=True,
            batch_size=1000,
            features=feat,
            remove_columns=["episode", "__key__"],
        )
        ds.push_to_hub(
            "mbodiai/oxe_bridge_v2",
            config_name="default",
            private=False,
            split=f"shard_{i}",
            token=os.getenv("HF_WRITE"),
        )
        ds.save_to_disk(f"bridge_v2/shard_{i}")
        print("[green bold]Saved[/]", i) # noqa

def to_vision_motor_episodes(item: Dict, get_relative_action, get_absolute_action, metadata: Metadata) -> Dict:
        episodes = [
            VisionMotorEpisode(
                steps=[
                    VisionMotorStep(
                        image=Image(step["observation"]["image"]["bytes"]),
                        task=step["observation"]["natural_language_instruction"],
                        action=get_relative_action(step),
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        timestamp=step_idx / float(metadata.control_frequency),
                        observation={
                            "image": Image(step["observation"]["image"]["bytes"]),
                            "task": step["observation"]["natural_language_instruction"],
                        },
                        absolute_action=get_absolute_action(step, action_is_relative=True),
                        supervision=step["reward"],
                        state=get_state(step),
                    )
                    for step_idx, step in enumerate(episode["steps"])
                ],
            )
            for episode_idx, episode in enumerate(item["episode"])
        ]
        return Sample.pack_from([step for episode in episodes for step in episode.steps]).dump(as_field="pil")


    # ds = list(get_hf_dataset(dataset_name="bridge", streaming=True).take(2))
