import os
import random
from PIL import Image
import numpy as np
import torch

def restore_image(image, width, height):
    img_array = image.numpy()
    img_array = np.transpose(img_array, (1, 2, 0))
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    return img_resized

def get_shuffled_file_paths(directory):
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
    # Shuffle the list of file paths
    random.shuffle(file_paths)

    return file_paths

def create_data_loader(root_path):
    return

def create_data(model,img_path,tmp_path="/home/hossein/github/content_replication/result_adv"):
    return