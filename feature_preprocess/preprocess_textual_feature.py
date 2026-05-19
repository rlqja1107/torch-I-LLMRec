import os
import sys
import json
import torch
import argparse
sys.path.append(".")
from tqdm import tqdm
torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='Art')
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core
print("Extracting Textual Features")


data_dir = f"./dataset/Amazon_18/{data_name}"
meta_data = json.load(open(f"{data_dir}/meta_info.json", 'r'))
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda', model_kwargs={"torch_dtype": torch.bfloat16})
model.eval()

text_list = []
text_feature = []
idx = 0
batch_size = 64

os.makedirs(f"{data_dir}/txt_features", exist_ok=True)
    

for i in tqdm(range(len(meta_data))):
    v = meta_data[str(i)]
    item_str = ""
    if idx != batch_size:
        if v['title'] != '':
            item_str += f"Title: {v['title']}. "
        if v['brand'] != '':
            item_str += f"Brand: {v['brand']}. "
        if v['category'] != '':
            item_str += f"Category: {v['category']}. "
        if len(v['feature']) > 0:
            for f in v['feature']:
                if "<span" in f or "<br>" in f:
                    pass
                else:
                    item_str += f"Feature: {f}. "
                    break
        if 'description' in v and v['description'] != '':
            if "<span" in v['description'] or "<br>" in v['description']:
                pass
            else:
                item_str += f"Detailed description: {v['description']}."
        text_list.append(item_str.strip().strip(".")+".")
        idx += 1
    if idx == batch_size or i == len(meta_data) - 1:
        with torch.no_grad():
            text_output = model.encode(text_list, convert_to_numpy=False)
        text_features = torch.stack(text_output).cpu()
        idx = 0
        text_list.clear()
        last_offset = len(meta_data) - i  if i == len(meta_data)-1 else 63
        for txt, j in zip(text_features, range(i+1-text_features.shape[0], i+1)):
            txt = txt.to(torch.bfloat16)
            torch.save(txt, f"{data_dir}/txt_features/{j}.pth")
            
    