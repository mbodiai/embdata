# import pytest
# from unittest.mock import patch, MagicMock
# from datasets import Dataset
# import torch
# from embdata import Episode, Sample, Image, Trajectory
# from embdata.trajectory import Trajectory

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
# def test_episode_creation_and_trajectory():
#     from docs.example import Episode, Trajectory, Sample, Image
#     example = {
#         "image": [b"image1", b"image2"],
#         "instruction": ["instruction1", "instruction2"],
#         "action": [[0.1, 0.2], [0.3, 0.4]],
#         "reward": [1.0, 2.0],
#         "is_terminal": [False, True]
#     }
#     episode = Episode()
#     for i in range(len(example["image"])):
#         step = Sample(
#             image=Image(base64=example["image"][i]),
#             instruction=example["instruction"][i],
#             action=example["action"][i],
#             reward=example["reward"][i],
#             is_terminal=example["is_terminal"][i]
#         )
#         episode.append(step)

#     action_trajectory = episode.trajectory(field="action")
#     cleaned_trajectory = action_trajectory.low_pass_filter(cutoff_freq=2)

#     assert isinstance(episode, Episode)
#     assert isinstance(action_trajectory, Trajectory)
#     assert isinstance(cleaned_trajectory, Trajectory)

# # Test GPT2CLIP model
# def test_gpt2clip_model():
#     from docs.example import GPT2CLIP
#     model = GPT2CLIP(num_actions=3)
#     assert hasattr(model, 'gpt2')
#     assert hasattr(model, 'clip')
#     assert hasattr(model, 'fusion')
#     assert hasattr(model, 'action_head')

# # Test prepare_batch function
# @pytest.fixture
# def mock_tokenizer_and_processor():
#     with patch('transformers.AutoTokenizer') as mock_AutoTokenizer, \
#          patch('transformers.CLIPProcessor') as mock_CLIPProcessor:
#         mock_tok = MagicMock()
#         mock_proc = MagicMock()
#         mock_AutoTokenizer.from_pretrained.return_value = mock_tok
#         mock_CLIPProcessor.from_pretrained.return_value = mock_proc
#         yield mock_tok, mock_proc

# def test_prepare_batch(mock_tokenizer_and_processor):
#     from docs.example import prepare_batch, Image
#     mock_tokenizer, mock_processor = mock_tokenizer_and_processor
#     examples = {
#         "instruction": ["instruction1", "instruction2"],
#         "image": [b"image1", b"image2"],
#         "action": [[0.1, 0.2], [0.3, 0.4]]
#     }
#     result = prepare_batch(examples)
#     assert all(key in result for key in ["input_ids", "attention_mask", "pixel_values", "labels"])

# # Test training loop
# def test_training_loop():
#     from docs.example import GPT2CLIP
#     model = GPT2CLIP(num_actions=3)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#     train_dataset = [{"input_ids": torch.rand(1, 10),
#                       "attention_mask": torch.ones(1, 10),
#                       "pixel_values": torch.rand(1, 3, 224, 224),
#                       "labels": torch.rand(1, 3)} for _ in range(5)]

#     num_epochs = 2
#     batch_size = 2

#     for epoch in range(num_epochs):
#         for i in range(0, len(train_dataset), batch_size):
#             batch = train_dataset[i:i+batch_size]
#             optimizer.zero_grad()
#             outputs = model(
#                 torch.cat([b["input_ids"] for b in batch]),
#                 torch.cat([b["attention_mask"] for b in batch]),
#                 torch.cat([b["pixel_values"] for b in batch])
#             )
#             labels = torch.cat([b["labels"] for b in batch])
#             loss = torch.nn.functional.mse_loss(outputs, labels)
#             loss.backward()
#             optimizer.step()

#     assert True  # If we reach this point without errors, the test passes
