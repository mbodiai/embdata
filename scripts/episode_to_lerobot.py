# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""A sample script that demonstrates how to convert an episode to the LeRobot Dataset and push it to the Hub."""

from pathlib import Path

from datasets import load_dataset
from huggingface_hub import create_branch
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, save_meta_data

from embdata.describe import describe
from embdata.episode import Episode
from embdata.sample import Sample


def main() -> None:
    ds = load_dataset("mbodiai/oxe_bridge_v2")
    s = Sample(ds["shard_0"])

    actions = s.flatten(to="action")
    observations = s.flatten(to="observation")
    states = s.flatten(to="state")

    e = Episode(zip(observations, actions, states, strict=False), freq_hz=5)
    describe(e)

    # Convert it to LeRobot format
    lerobot_dataset = e.lerobot()
    stats = compute_stats(lerobot_dataset.hf_dataset, batch_size=32, num_workers=8)

    repo_id = "mbodiai/lerobot_test"
    metadata_dir = Path("output/metadata")
    save_meta_data(lerobot_dataset.info, stats, lerobot_dataset.episode_data_index, metadata_dir)
    lerobot_dataset.hf_dataset.push_to_hub(repo_id, revision="main")
    push_meta_data_to_hub(repo_id, metadata_dir, revision="main")
    create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    # # Check the lerobot dataset from hub.
    # dataset = LeRobotDataset("mbodiai/lerobot_test", version="main")
    # print(dataset)
    # # Load back into Episode
    # e = Episode.from_lerobot(dataset)
    # describe(e)

    # # Apply delta timestamps
    # delta_timestamps = {
    #     # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
    #     "observation.image": [-1, -0.5, -0.20, 0],
    #     # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 20 ms, 10 ms, and current frame
    #     "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, -0.02, -0.01, 0],
    #     # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
    #     "action": [t / dataset.fps for t in range(64)],
    # }
    # dataset = LeRobotDataset("mbodiai/lerobot_test", version="main", delta_timestamps=delta_timestamps)
    # print(f"\n{dataset[0]['observation.image'].shape=}")
    # print(f"{dataset[0]['observation.state'].shape=}")
    # print(f"{dataset[0]['action'].shape=}\n")


if __name__ == "__main__":
    main()
