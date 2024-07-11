from datasets import load_dataset

from embdata.describe import describe
from embdata.sample import Sample
from embdata.trajectory import Trajectory

ds = load_dataset("mbodiai/oxe_bridge_v2")

describe(ds)

s = Sample(ds["shard_0"])
describe(s)


actions = s.flatten(to="action")
observations = s.flatten(to="observation")
describe(actions)
describe(observations)
states = s.flatten(to="state")

from embdata.episode import Episode

e = Episode(zip(observations, actions, states, strict=False), metadata={"freq_hz": 5})
describe(e.unpack())
describe(e)


e.trajectory(freq_hz=1).plot().save("trajectory.png")

e.trajectory().resample(target_hz=50).plot().save("resampled_trajectory.png")

# e.show()
t: Trajectory = e.trajectory().transform("minmax", min=0, max=255).plot().save("minmaxed_trajectory.png")
t.transform("unminmax", orig_min=e.trajectory().min(), orig_max=e.trajectory().max()).plot()


e.trajectory().frequencies().plot().save("frequencies.png")
