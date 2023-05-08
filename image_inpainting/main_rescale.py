from PIL import Image
from tools import compose_side_by_side
orig_adr="/home/hossein/github/content_replication/result_tmp2/2008/original.jpg"
inpainting_adr="/home/hossein/github/content_replication/result_tmp2/2008/inpainting.jpg"
save_dir="/home/hossein/github/content_replication/result_tmp2/2008"


image_orig= Image.open(orig_adr).convert("RGB")
image_in= Image.open(inpainting_adr).convert("RGB")
width, height = image_orig.size
new_image_orig=image_orig.resize((512, 512)).resize((width, height)).resize((int(width//1.2), int(height//1.2)))
new_image_in=image_in.resize((512, 512)).resize((width, height)).resize((int(width/1.2), int(height//1.2)))
compose_side_by_side(new_image_in,new_image_orig,save_dir+"/group_new.jpg")
