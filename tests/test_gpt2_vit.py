# import pytest
# from unittest.mock import patch, MagicMock
# from datasets import Dataset
# import torch
# from embdata import Episode, Sample, Trajectory, Image

# # Mock the dataset loading
# @pytest.fixture
# def mock_dataset():
#     with patch('datasets.load_dataset') as mock_load:
#         mock_data = MagicMock()
#         mock_data.__getitem__.return_value = {"train": Dataset.from_dict({"dummy": [1, 2, 3]})}
#         mock_load.return_value = mock_data
#         yield mock_load

# # Test dataset loading
# def test_dataset_loading(mock_dataset):
#     from docs.example import load_dataset
#     dataset = load_dataset("mbodiai/oxe_taco_play")
#     assert dataset is not None
#     mock_dataset.assert_called_once_with("mbodiai/oxe_taco_play")

# # Test process_example function
# def test_process_example():
#     from docs.example import process_example
#     example = {
#         "data": {
#             "pickle": {
#                 "steps": {
#                     "observation": {
#                         "image": {"bytes": b"dummy_image"},
#                         "natural_language_instruction": "dummy_instruction"
#                     },
#                     "action": [0.1, 0.2, 0.3],
#                     "reward": 1.0,
#                     "is_terminal": False
#                 }
#             }
#         }
#     }
#     result = process_example(example)
#     assert isinstance(result, dict)
#     assert all(key in result for key in ["image", "instruction", "action", "reward", "is_terminal"])

# # Test episode creation and trajectory cleaning
# @pytest.fixture
# def mock_episode_trajectory():
#     with patch('embdata.Episode') as mock_Episode, \
#          patch('embdata.Trajectory') as mock_Trajectory:
#         mock_episode = MagicMock()
#         mock_Episode.return_value = mock_episode
#         mock_trajectory = MagicMock()
#         mock_Trajectory.return_value = mock_trajectory
#         yield mock_episode, mock_trajectory

# def test_episode_creation_and_trajectory(mock_episode_trajectory):
#     from docs.example import create_episode, clean_trajectory
#     mock_episode, mock_trajectory = mock_episode_trajectory
#     example = {
#         "image": [b"image1", b"image2"],
#         "instruction": ["instruction1", "instruction2"],
#         "action": [[0.1, 0.2], [0.3, 0.4]],
#         "reward": [1.0, 2.0],
#         "is_terminal": [False, True]
#     }
#     episode = create_episode(example)
#     cleaned_trajectory = clean_trajectory(episode)
#     assert episode == mock_episode
#     mock_episode.append.assert_called()
#     assert cleaned_trajectory == mock_trajectory.low_pass_filter.return_value

# # Test GPT2ViT model
# def test_gpt2vit_model():
#     from docs.example import GPT2ViT
#     model = GPT2ViT(num_actions=3)
#     assert hasattr(model, 'gpt2')
#     assert hasattr(model, 'vit')
#     assert hasattr(model, 'fusion')
#     assert hasattr(model, 'action_head')

# # Test prepare_batch function
# @pytest.fixture
# def mock_tokenizer():
#     with patch('transformers.AutoTokenizer') as mock_AutoTokenizer:
#         mock_tok = MagicMock()
#         mock_AutoTokenizer.from_pretrained.return_value = mock_tok
#         yield mock_tok

# def test_prepare_batch(mock_tokenizer):
#     from docs.example import prepare_batch
#     examples = {
#         "instruction": ["instruction1", "instruction2"],
#         "image": [b"image1", b"image2"],
#         "action": [[0.1, 0.2], [0.3, 0.4]]
#     }
#     result = prepare_batch(examples)
#     assert all(key in result for key in ["input_ids", "attention_mask", "pixel_values", "labels"])

# # Test training loop
# @pytest.mark.parametrize("num_epochs", [1, 2])
# def test_training_loop(num_epochs):
#     from docs.example import train_model
#     model = MagicMock()
#     optimizer = MagicMock()
#     train_dataset = [{"input_ids": torch.rand(1, 10), 
#                       "attention_mask": torch.ones(1, 10), 
#                       "pixel_values": torch.rand(1, 3, 224, 224), 
#                       "labels": torch.rand(1, 3)} for _ in range(5)]
    
#     train_model(model, optimizer, train_dataset, num_epochs=num_epochs, batch_size=2)
#     assert model.train.call_count == num_epochs
#     assert optimizer.zero_grad.call_count == num_epochs * 3  # 5 samples, batch size 2, so 3 batches per epoch
