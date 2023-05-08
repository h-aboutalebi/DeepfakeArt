#path dataset: /home/dani/datasets/wikiart
import torch
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tools import mask_top_half, compose_side_by_side,get_shuffled_file_paths,mask_random
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
directory="/home/dani/datasets/wikiart"
list_files=get_shuffled_file_paths(directory)
count=20000
for image_address in list_files:
    if(image_address[-3:]!="jpg"):
        continue
    image= Image.open(image_address).convert("RGB")
    width, height = image.size
    name="/home/hossein/github/content_replication/result_tmp2/"+str(count)
    os.makedirs(name)
    masked_image=mask_random(image_address, output_path=name+"/mask.jpg").convert("RGB").resize((512, 512))
    prompt = "generate a painting compatible with the rest of the image"
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image_inpainting = pipe(prompt=prompt, image=image.resize((512, 512)), mask_image=masked_image).images[0]
    compose_side_by_side(image_inpainting.resize((width, height)),image.resize((512, 512)).resize((width, height)),name+"/group.jpg")
    image_inpainting.resize((width, height)).save(name+"/inpainting.jpg")
    (image.resize((512, 512))).resize((width, height)).save(name+"/original.jpg")
    count+=1
    if(count>count+1500):
        break