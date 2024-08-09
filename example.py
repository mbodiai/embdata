from datasets import load_dataset

from embdata.episode import Episode
from datasets import get_dataset_config_names

# e.trajectory().plot().show()

repo = "jxu124/OpenX-Embodiment"
for name in get_dataset_config_names(repo):
  print("Loading ", name)
  ds = load_dataset(repo, name).take(100)
  e = Episode(ds)
  print(e)
  e.trajectory().plot().show()

  input("Press Enter to continue...")
