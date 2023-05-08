

<img src="https://github.com/h-aboutalebi/DeepfakeArt/blob/main/images/logo.jpg" alt="logo" width="600" height="200">
Part of <img src="https://github.com/h-aboutalebi/DeepfakeArt/blob/main/images/genai4good.png" alt="genai4good" width="20%" height="20%">

<figure class="image">
<img src="https://github.com/h-aboutalebi/DeepfakeArt/blob/main/images/all.jpg" alt="inpainting">
<figcaption>Generated forgery images from real images using inpainting and style transfer.</figcaption>
</figure>

# DeepfakeArt Benchmark Dataset for Generative AI Art Forgery and Data Poisoning Detection
The tremendous recent advances in generative artificial intelligence techniques have led to significant successes and promise in a wide range of different applications ranging from conversational agents and textual content generation to voice and visual synthesis.  Amid the rise in generative AI and its increasing widespread adoption, there has been significant growing concern over the use of generative AI for malicious purposes.  In the realm of visual content synthesis using generative AI, key areas of significant concern has been image forgery (e.g., generation of images containing or derived from copyright content), and data poisoning (i.e., generation of adversarially contaminated images).  

Motivated to address these key concerns to encourage responsible generative AI, we introduce DeepfakeArt, a large-scale benchmark dataset designed specifically to aid in the building of machine learning algorithms for generative AI art forgery and data poisoning detection. Comprising of over 30,000 records across a variety of generative forgery and data poisoning techniques, each entry consists of a pair of images that are either forgeries / adversarially contaminated or not. Each of the generated images in the DeepfakeArt benchmark dataset has been quality checked in a comprehensive manner by our team.  DeepfakeArt is a core part of GenAI4Good, a global open source initiative for accelerating machine learning for promoting responsible creation and deployment of generative AI for good. 

The generative forgery and data poisoning methods leveraged in the DeepfakeArt benchmark dataset include:
- Inpainting
- Style Transfer
- Adversarial data poisoning
- Segmix
- Cutmix


**Team Members:**
- Hossein Aboutalebi
- Dayou Mao
- Carol Xu
- Alexander Wong


**The DeepfakeArt benchmark dataset is available [here](https://www.kaggle.com/datasets/danielmao2019/deepfakeart)**

## Inpainting Category
<figure class="image">
<img src="https://github.com/h-aboutalebi/DeepfakeArt/blob/main/images/inpainting.jpg" alt="inpainting">
<figcaption>Generated forgery images via inpainting.</figcaption>
</figure>

The source dataset for the inpainting category is WikiArt [(ref)](https://paperswithcode.com/paper/large-scale-classification-of-fine-art). Each image is sampled randomly from the dataset as the source image to generate forgery images. 
Each record in this category consists of three images: 

- source image: The source image used to create a forgery image from
- inpainting image: The inpainting image generated by Stable Diffusion 2 model [(ref)](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- masking image: black-white image which white parts depicts which parts of original image is inpainted by Stable Diffusion 2 to generate inpainting image

The prompt used for the generation of the inpainting image is: "generate a painting compatible with the rest of the image"

This category consists of more than 5000 records. The original images are masked between 40%-60%. We applied one of the followed macking schema randomly:

- side masking: where the top side, bottom side, right side or left side of the source image is maked
- diagonal masking: where the upper right, upper left, lower right, or lower left diagonal side of thw source image is masked
- random masking: where randomly selected parts of the source image are masked

The code for the data generation in this category can be found [here](https://github.com/h-aboutalebi/DeepfakeArt/blob/main/image_inpainting/main.py)

## Style Transfer Category
<figure class="image">
<img src="https://github.com/h-aboutalebi/DeepfakeArt/blob/main/images/style.jpg">
<figcaption>Generated forgery images via style transfer.</figcaption>
</figure>

The source dataset for the style transfer category is WikiArt [(ref)](https://paperswithcode.com/paper/large-scale-classification-of-fine-art). Each record in this category consists of two images: 

- source image: The source image used to create a forgery image from
- style transferred image: The style transferred image generated by ControlNet [(ref)](https://huggingface.co/lllyasviel/ControlNet)
- edge image: This edge image is created using Canny edge detection
- prompt: one of four prompts used for style transfer

Guided by Canny edge detections and prompts, we selected 200 images from each sub-directory of the WikiArt dataset, except for Action_painting and Analytical_Cubism, which contain only 98 and 110 images, respectively. We utilized four distinct prompts for different styles for the generations:

- "a high-quality, detailed, realistic image", 
- "a high-quality, detailed, cartoon style drawing", 
- "a high-quality, detailed, oil painting"
- "a high-quality, detailed, pencil drawing". 

Each prompt was used for a quarter of the images from each sub-directory. 

This category consists of more than 3,213 records. 

The code for the data generation of this category can be found [here](https://github.com/h-aboutalebi/DeepfakeArt/blob/main/main_style_transfer.py)


## Adversarial Data Poisoning Category 

Images in this category are adversarially data poisoned images that were generated using RobustBench [(ref)](https://robustbench.github.io/) and torchattack [(ref)](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html) libraries on WikiArt dataset. Here we used a ResNet50 model trained on ImageNet against:

- FGSM [(ref)](https://arxiv.org/abs/1412.6572) attack 
- PGD [(ref)](https://arxiv.org/pdf/1706.06083.pdf) attack
- APGD [(ref)](https://arxiv.org/pdf/2003.01690.pdf) attack

The dataset used here is WikiArt. For each source image, we have reported 3 attacks results. The images are also center cropped to make it harder for detection. 


 Each record in this category consists of two images: 

- source image: The source image used to create an adversarially data poisoned image from
- adv_image: The adversarially data poisoned image

This category consists of more than 2,730 records. 

The code for this category can be found [(here)](https://github.com/h-aboutalebi/DeepfakeArt/blob/main/adv_image/main.py).

## Cutmix Category

This method is initially proposed in the paper of "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models" [(ref)](https://arxiv.org/abs/2212.03860).

In this section, images were generated using the "Cutmix" technique, which involves selecting a square patch of pixels from a source image and overlaying it onto a target image. Both the source and target images were randomly chosen from the WikiArt dataset. The patch size, source image extraction location, and target image overlay location were all determined randomly. This section contains 3,000 sets of triplets, each consisting of a source image, target image, and the resulting mixed image.


The code for this section can be found [(here)](https://github.com/h-aboutalebi/DeepfakeArt/blob/main/main_Cutmix.py).

## Segmix Category

This method is initially proposed in the paper of "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models" [(ref)](https://arxiv.org/abs/2212.03860)

In this section, images were generated using the "Segmix" technique, which involves selecting an instance of an object from a source image and overlaying it onto a target image. Both the source and target images were randomly chosen from the PASCAL VOC dataset. Instance segmentation masks provided by the PASCAL VOC dataset label the object masks in the source image. The object instance from the source image, resizing of the selected object, and overlaying location onto the target image were all determined randomly. This section comprises 3,000 sets of 4-tuples, each containing a source image, target image, object mask, and the resulting mixed image.

The code for this section can be found [(here)](https://github.com/h-aboutalebi/DeepfakeArt/blob/main/main_Segmix.py)
