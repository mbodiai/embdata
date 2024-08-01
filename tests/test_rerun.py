from embdata.sense.image import Image
from embdata.sample import Sample
from embdata.episode import VisionMotorStep, Episode, ImageTask
from datasets import load_dataset

import os
from embdata.motion.control import HandControl



if __name__ == "__main__":
    dataset = list(load_dataset(path="mbodiai/xarm_horizon_good_3", split="train", token=os.getenv("HUB_TOKEN"), streaming=True).take(20))

    step = []

    for i in range(len(dataset)):

        step.append(VisionMotorStep(episode_idx=dataset[i]["episode_idx"], 
                                    step_idx=dataset[i]["step_idx"], 
                                    observation=ImageTask(image=Image(arg=dataset[i]["observation"]["image"]), task=dataset[i]["observation"]["task"]),
                                    action=HandControl(pose=(dataset[i]["action"]["pose"]), grasp=dataset[i]["action"]["grasp"]),
                                    state=Sample(dataset[i]["state"]),
                                    absolute_pose=HandControl(pose=dataset[i]["state"]["pose"]["pose"], grasp=dataset[i]["state"]["pose"]["grasp"])
                                    ))
        
    episode = Episode(steps=step)

    episode.show(mode="remote", port=5000, ws_port=8888)

