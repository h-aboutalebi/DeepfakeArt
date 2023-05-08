import torch
import os
from tqdm import tqdm

from engine.datasets.source import WikiArtDataset

from HQ_set import HQ_set


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
        "Post_Impressionism",
        "Realism",
        "Rococo",
        "Romanticism",
        "Symbolism",
        "Synthetic_Cubism",
        "Ukiyo_e",
]
seen = set()
for split in split_options:
    dataset = WikiArtDataset(split=split)
    torch.manual_seed(0)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
    paths_in_split = []
    for idx, (image, path) in enumerate(dataloader):
        if idx >= 200:
            continue
        assert type(path) == tuple
        assert len(path) == 1
        path = path[0]
        if os.path.basename(path) in seen:
            continue
        else:
            seen.add(os.path.basename(path))
        if idx not in HQ_set[split]:
            continue
        else:
            paths_in_split.append(path)
    text = "\",\n        \"".join(paths_in_split)
    text = "        \"" + text + "\","
    print(text)
