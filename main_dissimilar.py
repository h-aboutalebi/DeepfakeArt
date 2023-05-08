import torch
import os
import random
from tqdm import tqdm
import json

import engine
from HQ_paths import HQ_paths


all_paths = []
for val in HQ_paths.values():
    all_paths += val
print(f"{len(all_paths)=}")
num_examples = 10000  # 10k
pbar = tqdm(total=num_examples, leave=False)
idx = 0
image_dir = "/home/dani/repos/content_replication/datasets/dissimilar"
annotations = {}
random.seed(1)
while idx < num_examples:
    image_0_idx, image_1_idx = random.sample(range(len(all_paths)), k=2)
    save_path_0 = os.path.join(image_dir, f"{idx:03d}_image_0.png")
    save_path_1 = os.path.join(image_dir, f"{idx:03d}_image_1.png")
    os.system(f"cp \"{all_paths[image_0_idx]}\" \"{save_path_0}\"")
    os.system(f"cp \"{all_paths[image_1_idx]}\" \"{save_path_1}\"")
    example_dict = {}
    example_dict['image_0'] = save_path_0
    example_dict['image_1'] = save_path_1
    annotations[idx] = example_dict
    idx += 1
    pbar.update(1)
json_path = os.path.join(image_dir, "dissimilar.json")
with open(json_path, "w") as f:
    json.dump(annotations, f, indent=4)
