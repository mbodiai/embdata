import pytest
from embdata.episode import Episode, TimeStep, VisionMotorStep, ImageTask
from embdata.sample import Sample
from embdata import Image
from embdata.motion.control import RelativePoseHandControl

@pytest.fixture
def time_step():
    return TimeStep(observation=Sample("observation"), action=Sample("action"), supervision=Sample("supervision"))


def test_episode_initialization(time_step):
    episode = Episode(steps=[time_step])
    assert len(episode) == 1
    assert episode[0] == time_step


def test_episode_length(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    assert len(episode) == 3


def test_episode_get_item(time_step):
    episode = Episode(steps=[time_step])
    assert episode[0] == time_step


def test_episode_set_item(time_step):
    time_step2 = TimeStep(observation=Sample("observation"), action=Sample("action"), supervision=Sample("supervision"))
    episode = Episode(steps=[time_step])
    episode[0] = time_step2
    assert episode[0] == time_step2


def test_episode_iteration(time_step):
    episode = Episode(steps=[time_step, time_step])
    for i, step in enumerate(episode.iter()):
        assert step == episode[i]


def test_episode_addition(time_step):
    episode1 = Episode(steps=[time_step])
    episode2 = Episode(steps=[time_step, time_step])
    combined_episode = episode1 + episode2
    assert len(combined_episode) == 3


def test_episode_append(time_step):
    episode = Episode(steps=[])
    episode.append(time_step)
    assert len(episode) == 1
    assert episode[0] == time_step


def test_episode_split(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    episodes = episode.split(lambda step: False)
    assert len(episodes) == 2
    assert len(episodes[0]) == 0
    assert len(episodes[1]) == 3


def test_unpacked_episode(time_step):
    steps = [time_step, time_step, time_step]
    episode = Sample.unpack_from(steps)
    observations, actions, supervisions = episode.observation, episode.action, episode.supervision
    assert len(observations) == 3
    assert len(actions) == 3
    assert len(supervisions) == 3
    assert all(isinstance(observation, Sample) for observation in observations)
    assert all(isinstance(action, Sample) for action in actions)
    assert all(isinstance(supervision, Sample) for supervision in supervisions)


def test_episode_concatenate(time_step):
    episode1 = Episode(steps=[time_step, time_step])
    episode2 = Episode(steps=[time_step, time_step])
    episode3 = Episode(steps=[time_step, time_step])
    concatenated_episode = Episode.concat([episode1, episode2, episode3])
    assert len(concatenated_episode) == 6


def test_episode_from_observations_actions(time_step):
    observations = [Sample("observation1"), Sample("observation2")]
    actions = [Sample("action1"), Sample("action2")]
    episode = Episode.from_observations_actions(observations, actions)
    assert len(episode) == 2
    assert episode[0].observation == observations[0]
    assert episode[0].action == actions[0]
    assert episode[1].observation == observations[1]
    assert episode[1].action == actions[1]


def test_episode_from_list(time_step):
    steps = [
        {"observation": Sample("observation1"), "action": Sample("action1"), "supervision": Sample("supervision1")},
        {"observation": Sample("observation2"), "action": Sample("action2"), "supervision": Sample("supervision2")},
    ]
    episode = Episode(steps, "observation", "action", "supervision")
    assert len(episode) == 2
    assert episode[0].observation == steps[0]["observation"]
    assert episode[0].action == steps[0]["action"]
    assert episode[0].supervision == steps[0]["supervision"]
    assert episode[1].observation == steps[1]["observation"]
    assert episode[1].action == steps[1]["action"]
    assert episode[1].supervision == steps[1]["supervision"]


def test_episode_trajectory(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    trajectory = episode.trajectory("action", freq_hz=1)
    assert len(trajectory) == 3


def test_episode_append(time_step):
    episode = Episode(steps=[])
    episode.append(time_step)
    assert len(episode) == 1
    assert episode[0] == time_step


def test_episode_split(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    episodes = episode.split(lambda step: False)
    assert len(episodes) == 2
    assert len(episodes[0]) == 0
    assert len(episodes[1]) == 3


def test_episode_iteration(time_step):
    episode = Episode(steps=[time_step, time_step])
    for i, step in enumerate(episode.iter()):
        assert step == episode[i]


def test_episode_addition(time_step):
    episode1 = Episode(steps=[time_step])
    episode2 = Episode(steps=[time_step, time_step])
    combined_episode = episode1 + episode2
    assert len(combined_episode) == 3


def test_episode_get_item(time_step):
    episode = Episode(steps=[time_step])
    assert episode[0] == time_step


def test_episode_set_item(time_step):
    time_step2 = TimeStep(observation=Sample("observation"), action=Sample("action"), supervision=Sample("supervision"))
    episode = Episode(steps=[time_step])
    episode[0] = time_step2
    assert episode[0] == time_step2


def test_episode_push_to_hub(time_step):
    episode = Episode(steps=[time_step, time_step, time_step])
    episode.dataset().push_to_hub("mbodiai/episode_test", private=True)


def test_episode_vision_motor_step_dataset():
    episode = Episode([])
    episode.append(
        VisionMotorStep(
            episode_idx=0,
            step_idx=0,
            observation=ImageTask(image=Image(size=(224,224)), task="task"),
            action=RelativePoseHandControl(),
            state=Sample(),
        )
    )
    episode.dataset()

def test_dataset_to_episode():
    episode = Episode(steps=[time_step, time_step, time_step])
    dataset = episode.dataset()
    episode = Episode(steps=dataset.to_list())

if __name__ == "__main__":
    pytest.main(["-vv", __file__])
