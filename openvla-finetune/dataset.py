import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Type
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image
from datasets import load_dataset
import wandb


from prompter import PromptBuilder, PurePromptBuilder
from tokenizer import ActionTokenizer

IGNORE_INDEX = -100


class HuggingFaceDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: Any,
        image_transform: Callable[[Image.Image], torch.Tensor],
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.dataset = load_dataset(
            dataset_name, split=split, streaming=True, trust_remote_code=True
        )

    def __iter__(self):
        for data in self.dataset:
            image = data["observation"]["image"]
            instruction = data["observation"]["task"]
            action = np.array(
                [
                    data["relative_action"]["pose"][k]
                    for k in ["x", "y", "z", "roll", "pitch", "yaw"]
                ]
                + [data["relative_action"]["grasp"]]
            )

            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {
                    "from": "human",
                    "value": f"What action should the robot take to {instruction}?",
                },
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(
                prompt_builder.get_prompt(), add_special_tokens=True
            ).input_ids
            labels = list(input_ids)

            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            pixel_values = self.image_transform(image).to(torch.bfloat16)

            labels[: -(len(action) + 1)] = IGNORE_INDEX

            yield dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
