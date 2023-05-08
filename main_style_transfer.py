"""
I should have first generated config files for easier reproduction.
"""

import numpy
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

from engine.datasets.style_transfer.ControlNet import Model
from engine.datasets.source import WikiArtDataset

from HQ_set import HQ_set


def idx2prompt(idx, num_per_prompt):
    if idx < num_per_prompt:
        prompt = "a high-quality, detailed, realistic image"
    elif idx < 2 * num_per_prompt:
        prompt = "a high-quality, detailed, cartoon style drawing"
    elif idx < 3 * num_per_prompt:
        prompt = "a high-quality, detailed, oil painting"
    elif idx < 4 * num_per_prompt:
        prompt = "a high-quality, detailed, pencil drawing"
    else:
        prompt = None
    return prompt


model = Model(task_name='Canny')
split_options = [
        "Abstract_Expressionism",
        "Action_painting",
        "Analytical_Cubism",
        "Art_Nouveau_Modern",
        "Baroque",
        "Color_Field_Painting",
        "Contemporary_Realism",
        "Cubism",
        "Early_Renaissance",
        "Expressionism",
        "Fauvism",
        "High_Renaissance",
        "Impressionism",
        "Mannerism_Late_Renaissance",
        "Minimalism",
        "Naive_Art_Primitivism",
        "New_Realism",
        "Northern_Renaissance",
        "Pointillism",
        "Pop_Art",
        "Post_Impressionism",  # OSError: broken data stream when reading image file
        "Realism",
        "Rococo",
        "Romanticism",
        "Symbolism",
        "Synthetic_Cubism",
        "Ukiyo_e",
]
tot_num = 0
annotations = {}
seen = set()
json_path = "/home/dani/repos/content_replication/datasets/style_transfer_ControlNet/style_transfer_ControlNet.json"
for split in split_options:
    image_dir = os.path.join("datasets", "style_transfer_ControlNet", split)
    dataset = WikiArtDataset(split=split)
    torch.manual_seed(0)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_per_prompt = 50
    if 4 * num_per_prompt > len(dataloader):
        num_per_prompt = len(dataloader) // 4
        print(f"Using {num_per_prompt=}")
    print(f"{split=}, selecting 4*{num_per_prompt} from {len(dataloader)} images.")
    tot_num += 4 * num_per_prompt
    split_dict = {}
    for idx, (image, path) in tqdm(enumerate(dataloader)):
        prompt = idx2prompt(idx, num_per_prompt)
        if prompt is None:
            break
        assert type(path) == tuple
        assert len(path) == 1
        path = path[0]
        if os.path.basename(path) in seen:
            continue
        else:
            seen.add(os.path.basename(path))
        if idx not in HQ_set[split]:
            continue
        # """
        assert 0 <= torch.min(image) <= torch.max(image) <= 1
        assert image.shape[0] == 1
        assert image.dtype == torch.float32
        image = image * 255
        image = image[0]
        image = torch.permute(image, dims=[1, 2, 0])
        image = image.numpy()
        image = image.astype(numpy.uint8)
        results = model.process_canny(
            image=image,
            prompt=prompt,
            additional_prompt=a_prompt,
            negative_prompt=n_prompt,
            num_images=1,
            image_resolution=512,
            num_steps=20,
            guidance_scale=9,
            seed=0,
            low_threshold=100,
            high_threshold=200,
        )
        assert len(results) == 2, f"{len(results)=}"
        results[0] = results[0].resize((image.shape[1], image.shape[0]))
        results[1] = results[1].resize((image.shape[1], image.shape[0]))
        results[0] = numpy.array(results[0])
        results[1] = numpy.array(results[1])
        assert type(image) == type(results[0]) == type(results[1]) == numpy.ndarray
        assert image.shape[:2] == results[0].shape[:2] == results[1].shape[:2], \
            f"{image.shape=}, {results[0].shape=}, {results[1].shape=}"
        # """
        filepath_original = os.path.join(image_dir, f"{idx:03d}_original.png")
        filepath_edges = os.path.join(image_dir, f"{idx:03d}_edges.png")
        filepath_generated = os.path.join(image_dir, f"{idx:03d}_generated.png")
        Image.fromarray(image).save(filepath_original)
        Image.fromarray(results[0]).save(filepath_edges)
        Image.fromarray(results[1]).save(filepath_generated)
        example_dict = {}
        example_dict['original'] = filepath_original
        example_dict['edges'] = filepath_edges
        example_dict['generated'] = filepath_generated
        example_dict['prompt'] = prompt
        split_dict[idx] = example_dict
    annotations[split] = split_dict
    with open(json_path, "a") as f:
        json.dump({split: split_dict}, f, indent=4)

with open(json_path, "w") as f:
    json.dump(annotations, f, indent=4)
