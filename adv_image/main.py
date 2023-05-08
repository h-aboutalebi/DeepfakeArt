import torchattacks
from robustbench import load_model
from robustbench.data import  load_imagenet
#path dataset: /home/dani/datasets/wikiart
import torch
from PIL import Image
import numpy as np
import os
import shutil
from tools import get_shuffled_file_paths,restore_image
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
output_path="/home/hossein/github/content_replication/result_adv/result_tmp"
model = load_model(model_name="Standard_R50", dataset='imagenet', threat_model='Linf',model_dir=".")
model = model.to("cuda")
count=1 
test_loader = load_imagenet(data_dir='/home/hossein/github/content_replication/result_adv', n_examples=128)
for x_test, y_test, paths in iter(test_loader):
    x_test=x_test.to("cuda")
    attack_fgsm = torchattacks.FGSM(model, eps=8/255)
    attack_APGD = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    attack_PGD = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
    # import ipdb; ipdb.set_trace()
    adv_images_fgsm = attack_fgsm(x_test, torch.zeros(len(y_test),dtype=torch.int64).to("cuda"))
    adv_images_APGD = attack_APGD(x_test, torch.zeros(len(y_test),dtype=torch.int64).to("cuda"))
    adv_images_PGD = attack_PGD(x_test, torch.zeros(len(y_test),dtype=torch.int64).to("cuda"))
    for counter,path in enumerate(paths):
        output_save=os.path.join(output_path,str(count))
        if not os.path.exists(output_save):
            os.makedirs(output_save)
        abs_path=os.path.join("/home/hossein/github/content_replication/result_adv/val/n01440764",path)
        image=Image.open(abs_path)
        image.save(os.path.join(output_save,"original.jpg"))
        adv_images_reconstructed=restore_image(adv_images_fgsm[counter].to("cpu"),image.size[0],image.size[1])
        adv_images_reconstructed.save(os.path.join(output_save,"adv_fgsm.jpg"))
        adv_images_reconstructed=restore_image(adv_images_APGD[counter].to("cpu"),image.size[0],image.size[1])
        adv_images_reconstructed.save(os.path.join(output_save,"adv_APGD.jpg"))
        adv_images_reconstructed=restore_image(adv_images_PGD[counter].to("cpu"),image.size[0],image.size[1])
        adv_images_reconstructed.save(os.path.join(output_save,"adv_PGD.jpg"))
        count+=1
    print("yes")
    count+=1
    
    
