from datasets import load_dataset
from embdata import Episode, Sample, Trajectory, Image
from transformers import GPT2LMHeadModel, CLIPModel, CLIPProcessor, AutoTokenizer
import torch
import torch.nn as nn

# Function to flatten and process a single example
def process_example(example):
    flattened = Sample(example).flatten(
        to={
            "image": "data.pickle.steps.observation.image.bytes",
            "instruction": "data.pickle.steps.observation.natural_language_instruction",
            "action": "data.pickle.steps.action",
            "reward": "data.pickle.steps.reward",
            "is_terminal": "data.pickle.steps.is_terminal"
        }
    )
    return flattened

class GPT2CLIP(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fusion = nn.Linear(self.gpt2.config.hidden_size + self.clip.config.projection_dim, 512)
        self.action_head = nn.Linear(512, num_actions)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        text_features = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, -1, :]
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        fused_features = torch.cat([text_features, image_features], dim=1)
        fused_features = self.fusion(fused_features)
        action_logits = self.action_head(fused_features)
        return action_logits

def prepare_batch(examples):
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    gpt2_inputs = gpt2_tokenizer(examples["instruction"], padding=True, truncation=True, return_tensors="pt")
    clip_inputs = clip_processor(images=[Image(base64=img).pil for img in examples["image"]], return_tensors="pt")
    actions = torch.tensor(examples["action"])
    return {
        "input_ids": gpt2_inputs.input_ids,
        "attention_mask": gpt2_inputs.attention_mask,
        "pixel_values": clip_inputs.pixel_values,
        "labels": actions
    }
