from datasets import load_dataset

from embdata.episode import Episode
from embdata.sample import Sample

if __name__ == "__main__":
    dataset = load_dataset("mbodiai/xarm_07232024", split="train")
    episode = Episode.from_dataset(dataset)
    # episode = Episode.from_list(dataset.to_list(), observation_key="observation", action_key="action", state_key="state")

    episode.show(mode="remote")
