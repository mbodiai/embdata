import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Type
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb


from prompter import PromptBuilder, PurePromptBuilder
from collator import PaddedCollatorForActionPrediction
from tokenizer import ActionTokenizer
from dataset import HuggingFaceDataset
from xarm_dataset import XarmDataset

IGNORE_INDEX = -100


@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    dataset_name: str = "mbodiai/xarm_overfit"
    split: str = "train"
    run_root_dir: Path = Path("runs")

    batch_size: int = 16
    max_steps: int = 50_000
    save_steps: int = 5000
    learning_rate: float = 5e-6
    grad_accumulation_steps: int = 1
    shuffle_buffer_size: int = 100_000

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    wandb_project: str = "openvla"
    wandb_entity: str = None
    image_augmentation: bool = False
    lr_scheduler_type: str = "constant"


class OpenVLATrainer(Trainer):
    def __init__(self, action_tokenizer: ActionTokenizer, **kwargs):
        super().__init__(**kwargs)
        self.action_tokenizer = action_tokenizer
        self.custom_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs.get("input_ids"),
            labels=labels,
        )
        loss = outputs.loss

        # Compute Accuracy and L1 Loss for Logging
        action_logits = outputs.logits[
            :, model.vision_backbone.featurizer.patch_embed.num_patches : -1
        ]
        action_preds = action_logits.argmax(dim=2)
        action_gt = labels[:, 1:].to(action_preds.device)
        mask = action_gt > self.action_tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # Compute L1 Loss on Predicted (Continuous) Actions
        continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(
                action_preds[mask].cpu().numpy()
            )
        )
        continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(
                action_gt[mask].cpu().numpy()
            )
        )
        action_l1_loss = torch.nn.functional.l1_loss(
            continuous_actions_pred, continuous_actions_gt
        )

        # Store metrics for logging
        self.custom_metrics = {
            "action_accuracy": action_accuracy.item(),
            "l1_loss": action_l1_loss.item(),
        }

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        logs.update(self.custom_metrics)  # Add custom metrics to the logs
        super().log(logs)


def main(cfg: FinetuneConfig):
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = XarmDataset(
        cfg.dataset_name,
        cfg.split,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        PurePromptBuilder,
        cfg.image_augmentation,
    )

    training_args = TrainingArguments(
        output_dir=cfg.run_root_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_dir=cfg.run_root_dir / "logs",
        logging_steps=10,
        report_to="wandb",
        run_name=f"ft+{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}",
        hub_always_push=True,
        hub_strategy="every_save",
        push_to_hub=True,
        hub_private_repo=True,
        hub_model_id="mbodiai/openvla_xarm_overfit",
    )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side=processor.tokenizer.padding_side,
    )

    trainer = OpenVLATrainer(
        model=vla,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        action_tokenizer=action_tokenizer,
    )

    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity)
    trainer.train()


if __name__ == "__main__":
    cfg = FinetuneConfig()
    main(cfg)
