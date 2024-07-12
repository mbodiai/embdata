import pytest
from unittest.mock import Mock, patch
import torch
from torchvision import transforms
from datasets import Dataset
from xarm_dataset import XarmDataset
from embdata.episode import Episode
from embdata.motion.control import HandControl

@pytest.fixture
def mock_dataset():
    return [
        {
            "observation": {
                "image": torch.rand(3, 224, 224),
                "instruction": "Pick up the red cube"
            },
            "action": {
                "pose": [0.1, 0.2, 0.3, 0, 0, 0],
                "grasp": 0.5
            }
        },
        {
            "observation": {
                "image": torch.rand(3, 224, 224),
                "instruction": "Move the cube to the left"
            },
            "action": {
                "pose": [0.2, 0.3, 0.4, 0, 0, 0],
                "grasp": 0.8
            }
        }
    ]

@pytest.fixture
def mock_action_tokenizer():
    return Mock(return_value="Tokenized Action")

@pytest.fixture
def mock_base_tokenizer():
    return Mock(return_value=Mock(input_ids=[1, 2, 3]))

@pytest.fixture
def mock_image_transform():
    return Mock(return_value=torch.rand(3, 224, 224))

@pytest.fixture
def mock_prompt_builder():
    class MockPromptBuilder:
        def __init__(self, *args, **kwargs):
            pass
        def add_turn(self, *args, **kwargs):
            pass
        def get_prompt(self):
            return "Mock Prompt"
    return MockPromptBuilder

def test_xarm_dataset_initialization(mock_dataset, mock_action_tokenizer, mock_base_tokenizer, mock_image_transform, mock_prompt_builder):
    with patch('xarm_dataset.load_dataset', return_value=mock_dataset):
        dataset = XarmDataset(
            "mock_dataset",
            "train",
            mock_action_tokenizer,
            mock_base_tokenizer,
            mock_image_transform,
            mock_prompt_builder
        )
        
        assert isinstance(dataset, XarmDataset)
        assert dataset.action_tokenizer == mock_action_tokenizer
        assert dataset.base_tokenizer == mock_base_tokenizer
        assert dataset.image_transform == mock_image_transform
        assert dataset.prompt_builder_fn == mock_prompt_builder
        assert isinstance(dataset.dataset, list)
        assert len(dataset.dataset) == 2

def test_xarm_dataset_iteration(mock_dataset, mock_action_tokenizer, mock_base_tokenizer, mock_image_transform, mock_prompt_builder):
    with patch('xarm_dataset.load_dataset', return_value=mock_dataset):
        dataset = XarmDataset(
            "mock_dataset",
            "train",
            mock_action_tokenizer,
            mock_base_tokenizer,
            mock_image_transform,
            mock_prompt_builder
        )
        
        for item in dataset:
            assert isinstance(item, dict)
            assert "pixel_values" in item
            assert "input_ids" in item
            assert "labels" in item
            assert isinstance(item["pixel_values"], torch.Tensor)
            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["labels"], torch.Tensor)

def test_xarm_dataset_normalization(mock_dataset, mock_action_tokenizer, mock_base_tokenizer, mock_image_transform, mock_prompt_builder):
    with patch('xarm_dataset.load_dataset', return_value=mock_dataset):
        dataset = XarmDataset(
            "mock_dataset",
            "train",
            mock_action_tokenizer,
            mock_base_tokenizer,
            mock_image_transform,
            mock_prompt_builder
        )
        
        episode = Episode(dataset.dataset)
        normalized_trajectory = episode.trajectory().transform("minmax", min=-1, max=1)
        
        for data, normalized_action in zip(dataset.dataset, normalized_trajectory, strict=True):
            assert all(-1 <= value <= 1 for value in normalized_action)

def test_xarm_dataset_image_augmentation(mock_dataset, mock_action_tokenizer, mock_base_tokenizer, mock_image_transform, mock_prompt_builder):
    with patch('xarm_dataset.load_dataset', return_value=mock_dataset):
        dataset = XarmDataset(
            "mock_dataset",
            "train",
            mock_action_tokenizer,
            mock_base_tokenizer,
            mock_image_transform,
            mock_prompt_builder,
            image_augmentation=True
        )
        
        assert dataset.image_augmentation == True
        assert isinstance(dataset.transform, transforms.Compose)

if __name__ == "__main__":
    pytest.main()
