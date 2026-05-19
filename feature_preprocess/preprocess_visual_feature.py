import os
import sys
import torch
sys.path.append(".")
torch.set_num_threads(4)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="Sport")
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core

import json
from ILLMRec.model.multimodal_encoder.siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)
from PIL import Image
from tqdm import tqdm
print("Extracting Visual Features")

data_dir = f"./dataset/Amazon_18/{data_name}"
data_type_float = False

data_type = torch.bfloat16

meta_data = json.load(open(f"{data_dir}/meta_info.json", 'r'))
image_folder = f"{data_dir}/image"
train_user = json.load(open(f"{data_dir}/train_users.json", 'r'))
os.makedirs(f"{data_dir}/img_features", exist_ok=True)

image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
vision_tower = SiglipVisionModel.from_pretrained(
"google/siglip-so400m-patch14-384", torch_dtype=data_type, state_dict=None, device_map='cuda'
)
vision_tower.eval()
for v in vision_tower.parameters():
    v.requires_grad = False
crop_size = image_processor.size

print("Processing image")
image_list = []
for i in tqdm(range(len(meta_data))):
    image = Image.open(os.path.join(image_folder, f'{i}.jpg')).convert("RGB")
    image = image.resize((crop_size["height"], crop_size["width"]))
    image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_list.append(image)

print("Encoding image feature")
vis_feature_list = []
batch_size = 16
for i in tqdm(range(0, len(meta_data), batch_size)):
    if i == int(len(meta_data) / batch_size) * batch_size:
        n_remain = len(meta_data) - int(len(meta_data) / batch_size) * batch_size
        stack_image = torch.stack(image_list[i:i+n_remain]).to(data_type).cuda()
        with torch.no_grad():
            vis_feature = vision_tower(stack_image)[1]       
    else:
        stack_image = torch.stack(image_list[i:i+batch_size]).to(data_type).cuda()
        with torch.no_grad():
            vis_feature = vision_tower(stack_image)[1]
    vis_feature_list.append(vis_feature)
    
print("Saving")
vis_feature_list = torch.cat(vis_feature_list)
for i, vis in enumerate(vis_feature_list):
    vis = vis.cpu()
    torch.save(vis, f"{data_dir}/img_features/{i}.pth")